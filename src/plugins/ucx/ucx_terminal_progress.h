/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_SRC_PLUGINS_UCX_UCX_TERMINAL_PROGRESS_H
#define NIXL_SRC_PLUGINS_UCX_UCX_TERMINAL_PROGRESS_H

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>

extern "C" {
#include <ucp/api/ucp.h>
}

#include <nixl_types.h>

namespace nixl::ucx {

enum class ucx_callback_kind_t;

enum class terminal_submission_phase_t {
    POSTING,
    DATA_FLUSH,
    NOTIFICATION,
    TERMINAL,
};

struct terminal_submission_result_t {
    std::uint64_t ownerCookie = 0;
    std::uint64_t handleIdentity = 0;
    std::uint64_t generation = 0;
    nixl_status_t status = NIXL_ERR_BACKEND;
    std::uint64_t nativeTimestampNs = 0;
    nixl_xfer_terminal_progress_t diagnostics;
};

class terminal_submission_sink_t {
public:
    virtual ~terminal_submission_sink_t() = default;

    virtual nixl_status_t
    publishTerminal(const terminal_submission_result_t &result) noexcept = 0;
};

/**
 * Bounded MPSC continuation queue drained exclusively by one UCX progress owner.
 */
class ucx_worker_continuation_queue_t final {
public:
    using continuation_t = std::function<nixl_status_t()>;
    using wake_t = std::function<nixl_status_t()>;

    ucx_worker_continuation_queue_t(std::size_t capacity, wake_t wake);

    ucx_worker_continuation_queue_t(const ucx_worker_continuation_queue_t &) = delete;
    ucx_worker_continuation_queue_t &
    operator=(const ucx_worker_continuation_queue_t &) = delete;

    [[nodiscard]] nixl_status_t
    enqueue(continuation_t continuation);
    [[nodiscard]] nixl_status_t
    fail(nixl_status_t status);
    [[nodiscard]] nixl_status_t
    registerProducer();
    [[nodiscard]] nixl_status_t
    retireProducer();
    [[nodiscard]] std::size_t
    producerCount() const;
    [[nodiscard]] std::size_t
    drain();
    [[nodiscard]] nixl_status_t
    close();
    [[nodiscard]] nixl_status_t
    fatalStatus() const;
    [[nodiscard]] std::size_t
    size() const;

private:
    [[nodiscard]] nixl_status_t
    failLocked(nixl_status_t status);

    const std::size_t capacity_;
    const wake_t wake_;
    mutable std::mutex mutex_;
    std::deque<continuation_t> queue_;
    std::deque<continuation_t> fatalDrainQueue_;
    std::size_t registeredProducers_ = 0;
    nixl_status_t fatalStatus_ = NIXL_SUCCESS;
    bool closed_ = false;
};

/**
 * Shared native terminal state for one UCX transfer generation.
 *
 * Endpoint flushes may finish while submission is still being posted. Success
 * is emitted only after posting is sealed, every chunk and flush is complete,
 * and the optional attached notification has completed.
 */
class terminal_submission_state_t final {
public:
    using notification_post_t = std::function<nixl_status_t()>;

    terminal_submission_state_t(std::uint64_t owner_cookie,
                                std::uint64_t handle_identity,
                                std::uint64_t generation,
                                std::size_t minimum_chunks,
                                std::size_t minimum_flushes,
                                bool has_notification,
                                std::shared_ptr<terminal_submission_sink_t> sink);

    terminal_submission_state_t(const terminal_submission_state_t &) = delete;
    terminal_submission_state_t &
    operator=(const terminal_submission_state_t &) = delete;

    [[nodiscard]] nixl_status_t
    registerChunk();
    [[nodiscard]] nixl_status_t
    registerFlush();
    [[nodiscard]] nixl_status_t
    completeFlush(nixl_status_t status, std::uint64_t timestamp_ns);
    [[nodiscard]] nixl_status_t
    completeChunk(nixl_status_t status, std::uint64_t timestamp_ns);
    [[nodiscard]] nixl_status_t
    sealPosting(notification_post_t notification_post,
                std::uint64_t timestamp_ns);
    [[nodiscard]] nixl_status_t
    dispatchNotification(std::uint64_t timestamp_ns);
    [[nodiscard]] nixl_status_t
    completeNotification(nixl_status_t status, std::uint64_t timestamp_ns);
    [[nodiscard]] nixl_status_t
    cancel(nixl_status_t status, std::uint64_t timestamp_ns);

    [[nodiscard]] terminal_submission_phase_t
    phase() const;
    [[nodiscard]] bool
    isTerminal() const;

    void
    recordPostObservation(ucx_callback_kind_t kind,
                          nixl_status_t status,
                          bool callback_before_poster);
    void
    recordCallbackObservation(ucx_callback_kind_t kind,
                              bool before_poster,
                              std::uint64_t timestamp_ns);
    void
    recordPeakContinuationDepth(std::size_t depth);
    void
    recordTerminalInventory(std::size_t active_callback_slots,
                            std::size_t continuation_depth);

private:
    struct transition_t {
        nixl_status_t status = NIXL_SUCCESS;
        std::optional<terminal_submission_result_t> result;
    };

    [[nodiscard]] transition_t
    failLocked(nixl_status_t status, std::uint64_t timestamp_ns);
    [[nodiscard]] transition_t
    advanceLocked(std::uint64_t timestamp_ns);
    [[nodiscard]] transition_t
    makeTerminalLocked(nixl_status_t status, std::uint64_t timestamp_ns);
    [[nodiscard]] nixl_status_t
    finishTransition(transition_t transition) const;
    [[nodiscard]] nixl_status_t
    finishDataTransition(transition_t transition,
                         std::uint64_t timestamp_ns);

    const std::uint64_t ownerCookie_;
    const std::uint64_t handleIdentity_;
    const std::uint64_t generation_;
    const std::size_t minimumChunks_;
    const std::size_t minimumFlushes_;
    const bool hasNotification_;
    const std::shared_ptr<terminal_submission_sink_t> sink_;

    mutable std::mutex mutex_;
    terminal_submission_phase_t phase_ = terminal_submission_phase_t::POSTING;
    std::size_t registeredChunks_ = 0;
    std::size_t registeredFlushes_ = 0;
    std::size_t completedFlushes_ = 0;
    std::size_t completedChunks_ = 0;
    bool postingSealed_ = false;
    bool notificationStarted_ = false;
    bool notificationCompleted_ = false;
    bool published_ = false;
    nixl_xfer_terminal_progress_t diagnostics_;
    std::optional<notification_post_t> notificationPost_;
};

enum class ucx_callback_kind_t {
    DATA_CHUNK,
    ENDPOINT_FLUSH,
    NOTIFICATION,
};

/**
 * Stable callback slot implementing the callback/poster two-party handshake.
 *
 * The UCX callback only records status and schedules owner work. It never
 * queries or releases the request. The progress-owner continuation performs
 * request release after the poster has captured request evidence.
 */
class ucx_callback_slot_t final :
    public std::enable_shared_from_this<ucx_callback_slot_t> {
public:
    using request_release_t = std::function<void(void *)>;
    using owner_before_completion_t = std::function<nixl_status_t(nixl_status_t)>;
    using owner_after_completion_t = std::function<void()>;

    [[nodiscard]] static std::shared_ptr<ucx_callback_slot_t>
    create(std::shared_ptr<terminal_submission_state_t> state,
           ucx_callback_kind_t kind,
           std::shared_ptr<ucx_worker_continuation_queue_t> continuations,
           request_release_t request_release,
           owner_before_completion_t owner_before_completion = {},
           owner_after_completion_t owner_after_completion = {});

    ucx_callback_slot_t(const ucx_callback_slot_t &) = delete;
    ucx_callback_slot_t &
    operator=(const ucx_callback_slot_t &) = delete;

    static void
    ucpCompletion(void *request, ucs_status_t status, void *user_data) noexcept;

    void
    recordCallback(void *request,
                   nixl_status_t status,
                   std::uint64_t timestamp_ns) noexcept;
    [[nodiscard]] nixl_status_t
    armPoster(void *request,
              nixl_status_t post_status,
              std::uint64_t timestamp_ns) noexcept;
    [[nodiscard]] bool
    isDelivered() const noexcept;
    [[nodiscard]] bool
    isScheduled() const noexcept;
    [[nodiscard]] void *
    requestForCancellation() const noexcept;

private:
    ucx_callback_slot_t(std::shared_ptr<terminal_submission_state_t> state,
                        ucx_callback_kind_t kind,
                        std::shared_ptr<ucx_worker_continuation_queue_t> continuations,
                        request_release_t request_release,
                        owner_before_completion_t owner_before_completion,
                        owner_after_completion_t owner_after_completion);

    [[nodiscard]] nixl_status_t
    scheduleLocked() noexcept;
    [[nodiscard]] nixl_status_t
    deliverOnOwner() noexcept;

    const std::shared_ptr<terminal_submission_state_t> state_;
    const ucx_callback_kind_t kind_;
    const std::shared_ptr<ucx_worker_continuation_queue_t> continuations_;
    const request_release_t requestRelease_;
    const owner_before_completion_t ownerBeforeCompletion_;
    const owner_after_completion_t ownerAfterCompletion_;
    mutable std::mutex mutex_;
    bool callbackArrived_ = false;
    bool posterArmed_ = false;
    bool scheduled_ = false;
    bool delivered_ = false;
    void *callbackRequest_ = nullptr;
    void *posterRequest_ = nullptr;
    nixl_status_t callbackStatus_ = NIXL_ERR_BACKEND;
    nixl_status_t postStatus_ = NIXL_ERR_BACKEND;
    std::uint64_t callbackTimestampNs_ = 0;
    std::uint64_t posterTimestampNs_ = 0;
};

[[nodiscard]] std::uint64_t
terminalProgressTimestampNs() noexcept;

} // namespace nixl::ucx

#endif
