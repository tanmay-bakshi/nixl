/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ucx_terminal_progress.h"

#include <algorithm>
#include <ctime>
#include <utility>

#include "ucx_enums.h"

namespace nixl::ucx {

std::uint64_t
terminalProgressTimestampNs() noexcept {
    timespec now = {};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &now) != 0) {
        return 0;
    }
    return static_cast<std::uint64_t>(now.tv_sec) * 1000000000ULL +
        static_cast<std::uint64_t>(now.tv_nsec);
}

ucx_worker_continuation_queue_t::ucx_worker_continuation_queue_t(
    std::size_t capacity,
    wake_t wake)
    : capacity_(capacity), wake_(std::move(wake)) {}

nixl_status_t
ucx_worker_continuation_queue_t::enqueue(continuation_t continuation) {
    if (!continuation) {
        return NIXL_ERR_INVALID_PARAM;
    }
    {
        const std::lock_guard lock(mutex_);
        if (closed_ || fatalStatus_ != NIXL_SUCCESS) {
            return fatalStatus_ != NIXL_SUCCESS ? fatalStatus_ : NIXL_ERR_NOT_ALLOWED;
        }
        if (queue_.size() == capacity_) {
            (void)failLocked(NIXL_ERR_BACKEND);
        } else {
            queue_.push_back(std::move(continuation));
        }
    }
    if (fatalStatus() != NIXL_SUCCESS) {
        return fail(NIXL_ERR_BACKEND);
    }
    if (!wake_) {
        const std::lock_guard lock(mutex_);
        return failLocked(NIXL_ERR_NOT_SUPPORTED);
    }
    const nixl_status_t wake_status = wake_();
    if (wake_status != NIXL_SUCCESS) {
        const std::lock_guard lock(mutex_);
        return failLocked(wake_status);
    }
    return NIXL_SUCCESS;
}

nixl_status_t
ucx_worker_continuation_queue_t::fail(nixl_status_t status) {
    {
        const std::lock_guard lock(mutex_);
        (void)failLocked(status);
    }
    if (wake_) {
        const nixl_status_t wake_status = wake_();
        if (wake_status != NIXL_SUCCESS) {
            const std::lock_guard lock(mutex_);
            (void)failLocked(wake_status);
        }
    }
    return fatalStatus();
}

std::size_t
ucx_worker_continuation_queue_t::drain() {
    std::deque<continuation_t> ready;
    {
        const std::lock_guard lock(mutex_);
        ready.swap(queue_);
    }
    for (continuation_t &continuation : ready) {
        const nixl_status_t status = continuation();
        if (status != NIXL_SUCCESS) {
            (void)fail(status);
        }
    }
    return ready.size();
}

nixl_status_t
ucx_worker_continuation_queue_t::close() {
    const std::lock_guard lock(mutex_);
    closed_ = true;
    if (!queue_.empty()) {
        return failLocked(NIXL_ERR_CANCELED);
    }
    return fatalStatus_;
}

nixl_status_t
ucx_worker_continuation_queue_t::fatalStatus() const {
    const std::lock_guard lock(mutex_);
    return fatalStatus_;
}

std::size_t
ucx_worker_continuation_queue_t::size() const {
    const std::lock_guard lock(mutex_);
    return queue_.size();
}

nixl_status_t
ucx_worker_continuation_queue_t::failLocked(nixl_status_t status) {
    if (fatalStatus_ == NIXL_SUCCESS) {
        fatalStatus_ = status < NIXL_SUCCESS ? status : NIXL_ERR_BACKEND;
    }
    return fatalStatus_;
}

terminal_submission_state_t::terminal_submission_state_t(
    std::uint64_t owner_cookie,
    std::uint64_t handle_identity,
    std::uint64_t generation,
    std::size_t expected_chunks,
    std::size_t expected_flushes,
    bool has_notification,
    std::shared_ptr<terminal_submission_sink_t> sink)
    : ownerCookie_(owner_cookie),
      handleIdentity_(handle_identity),
      generation_(generation),
      expectedChunks_(expected_chunks),
      expectedFlushes_(expected_flushes),
      hasNotification_(has_notification),
      sink_(std::move(sink)) {}

nixl_status_t
terminal_submission_state_t::registerFlush() {
    const std::lock_guard lock(mutex_);
    if (phase_ != terminal_submission_phase_t::POSTING || postingSealed_ ||
        registeredFlushes_ == expectedFlushes_) {
        return NIXL_ERR_NOT_ALLOWED;
    }
    ++registeredFlushes_;
    return NIXL_SUCCESS;
}

nixl_status_t
terminal_submission_state_t::completeFlush(nixl_status_t status,
                                            std::uint64_t timestamp_ns) {
    transition_t transition;
    {
        const std::lock_guard lock(mutex_);
        if (phase_ == terminal_submission_phase_t::TERMINAL) {
            return NIXL_ERR_NOT_ALLOWED;
        }
        if (status != NIXL_SUCCESS) {
            transition = failLocked(status, timestamp_ns);
        } else if (completedFlushes_ == expectedFlushes_) {
            transition = failLocked(NIXL_ERR_BACKEND, timestamp_ns);
        } else {
            ++completedFlushes_;
            transition = advanceLocked(timestamp_ns);
        }
    }
    return finishDataTransition(std::move(transition), timestamp_ns);
}

nixl_status_t
terminal_submission_state_t::completeChunk(nixl_status_t status,
                                            std::uint64_t timestamp_ns) {
    transition_t transition;
    {
        const std::lock_guard lock(mutex_);
        if (phase_ == terminal_submission_phase_t::TERMINAL) {
            return NIXL_ERR_NOT_ALLOWED;
        }
        if (status != NIXL_SUCCESS) {
            transition = failLocked(status, timestamp_ns);
        } else if (completedChunks_ == expectedChunks_) {
            transition = failLocked(NIXL_ERR_BACKEND, timestamp_ns);
        } else {
            ++completedChunks_;
            transition = advanceLocked(timestamp_ns);
        }
    }
    return finishDataTransition(std::move(transition), timestamp_ns);
}

nixl_status_t
terminal_submission_state_t::sealPosting(notification_post_t notification_post,
                                         std::uint64_t timestamp_ns) {
    transition_t transition;
    {
        const std::lock_guard lock(mutex_);
        if (phase_ != terminal_submission_phase_t::POSTING || postingSealed_ ||
            registeredFlushes_ != expectedFlushes_ ||
            (hasNotification_ && !notification_post)) {
            transition = failLocked(NIXL_ERR_BACKEND, timestamp_ns);
        } else {
            postingSealed_ = true;
            phase_ = terminal_submission_phase_t::DATA_FLUSH;
            notificationPost_ = std::move(notification_post);
            transition = advanceLocked(timestamp_ns);
        }
    }
    return finishDataTransition(std::move(transition), timestamp_ns);
}

nixl_status_t
terminal_submission_state_t::dispatchNotification(std::uint64_t timestamp_ns) {
    notification_post_t post;
    {
        const std::lock_guard lock(mutex_);
        if (phase_ != terminal_submission_phase_t::NOTIFICATION ||
            notificationStarted_ || !notificationPost_.has_value()) {
            return NIXL_ERR_NOT_READY;
        }
        notificationStarted_ = true;
        post = std::move(notificationPost_).value();
        notificationPost_.reset();
    }

    const nixl_status_t post_status = post();
    if (post_status == NIXL_IN_PROG) {
        return NIXL_SUCCESS;
    }
    return completeNotification(post_status, timestamp_ns);
}

nixl_status_t
terminal_submission_state_t::completeNotification(nixl_status_t status,
                                                   std::uint64_t timestamp_ns) {
    transition_t transition;
    {
        const std::lock_guard lock(mutex_);
        if (phase_ == terminal_submission_phase_t::TERMINAL) {
            return NIXL_ERR_NOT_ALLOWED;
        }
        if (status != NIXL_SUCCESS) {
            transition = failLocked(status, timestamp_ns);
        } else if (phase_ != terminal_submission_phase_t::NOTIFICATION ||
                   !notificationStarted_ || notificationCompleted_) {
            transition = failLocked(NIXL_ERR_BACKEND, timestamp_ns);
        } else {
            notificationCompleted_ = true;
            transition = advanceLocked(timestamp_ns);
        }
    }
    return finishTransition(std::move(transition));
}

nixl_status_t
terminal_submission_state_t::cancel(nixl_status_t status,
                                    std::uint64_t timestamp_ns) {
    transition_t transition;
    {
        const std::lock_guard lock(mutex_);
        if (status >= NIXL_SUCCESS) {
            status = NIXL_ERR_CANCELED;
        }
        transition = failLocked(status, timestamp_ns);
    }
    return finishTransition(std::move(transition));
}

terminal_submission_phase_t
terminal_submission_state_t::phase() const {
    const std::lock_guard lock(mutex_);
    return phase_;
}

bool
terminal_submission_state_t::isTerminal() const {
    const std::lock_guard lock(mutex_);
    return phase_ == terminal_submission_phase_t::TERMINAL;
}

terminal_submission_state_t::transition_t
terminal_submission_state_t::failLocked(nixl_status_t status,
                                         std::uint64_t timestamp_ns) {
    if (phase_ == terminal_submission_phase_t::TERMINAL) {
        return {.status = NIXL_ERR_NOT_ALLOWED};
    }
    if (status >= NIXL_SUCCESS) {
        status = NIXL_ERR_BACKEND;
    }
    return makeTerminalLocked(status, timestamp_ns);
}

terminal_submission_state_t::transition_t
terminal_submission_state_t::advanceLocked(std::uint64_t timestamp_ns) {
    if (!postingSealed_ || completedChunks_ != expectedChunks_ ||
        completedFlushes_ != expectedFlushes_) {
        return {};
    }
    if (hasNotification_) {
        phase_ = terminal_submission_phase_t::NOTIFICATION;
        if (!notificationCompleted_) {
            return {};
        }
    }
    return makeTerminalLocked(NIXL_SUCCESS, timestamp_ns);
}

terminal_submission_state_t::transition_t
terminal_submission_state_t::makeTerminalLocked(nixl_status_t status,
                                                  std::uint64_t timestamp_ns) {
    if (published_ || sink_ == nullptr) {
        phase_ = terminal_submission_phase_t::TERMINAL;
        return {.status = NIXL_ERR_BACKEND};
    }
    phase_ = terminal_submission_phase_t::TERMINAL;
    published_ = true;
    return {
        .result = terminal_submission_result_t{
            .ownerCookie = ownerCookie_,
            .handleIdentity = handleIdentity_,
            .generation = generation_,
            .status = status,
            .nativeTimestampNs = timestamp_ns,
        },
    };
}

nixl_status_t
terminal_submission_state_t::finishTransition(transition_t transition) const {
    if (transition.status != NIXL_SUCCESS || !transition.result.has_value()) {
        return transition.status;
    }
    return sink_->publishTerminal(transition.result.value());
}

nixl_status_t
terminal_submission_state_t::finishDataTransition(
    transition_t transition,
    std::uint64_t timestamp_ns) {
    const nixl_status_t transition_status = finishTransition(std::move(transition));
    if (transition_status != NIXL_SUCCESS ||
        phase() != terminal_submission_phase_t::NOTIFICATION) {
        return transition_status;
    }
    const nixl_status_t dispatch_status = dispatchNotification(timestamp_ns);
    if (dispatch_status == NIXL_ERR_NOT_READY) {
        return NIXL_SUCCESS;
    }
    return dispatch_status;
}

std::shared_ptr<ucx_callback_slot_t>
ucx_callback_slot_t::create(
    std::shared_ptr<terminal_submission_state_t> state,
    ucx_callback_kind_t kind,
    std::shared_ptr<ucx_worker_continuation_queue_t> continuations,
    request_release_t request_release) {
    return std::shared_ptr<ucx_callback_slot_t>(
        new ucx_callback_slot_t(std::move(state),
                                kind,
                                std::move(continuations),
                                std::move(request_release)));
}

ucx_callback_slot_t::ucx_callback_slot_t(
    std::shared_ptr<terminal_submission_state_t> state,
    ucx_callback_kind_t kind,
    std::shared_ptr<ucx_worker_continuation_queue_t> continuations,
    request_release_t request_release)
    : state_(std::move(state)),
      kind_(kind),
      continuations_(std::move(continuations)),
      requestRelease_(std::move(request_release)) {}

void
ucx_callback_slot_t::ucpCompletion(void *request,
                                   ucs_status_t status,
                                   void *user_data) noexcept {
    auto *slot = static_cast<ucx_callback_slot_t *>(user_data);
    if (slot == nullptr) {
        return;
    }
    slot->recordCallback(
        request, ucsToNixlStatus(status), terminalProgressTimestampNs());
}

void
ucx_callback_slot_t::recordCallback(void *request,
                                    nixl_status_t status,
                                    std::uint64_t timestamp_ns) noexcept {
    const std::lock_guard lock(mutex_);
    if (callbackArrived_ || scheduled_ || delivered_) {
        if (continuations_ != nullptr) {
            (void)continuations_->fail(NIXL_ERR_BACKEND);
        }
        return;
    }
    callbackArrived_ = true;
    callbackRequest_ = request;
    callbackStatus_ = status;
    callbackTimestampNs_ = timestamp_ns;
    (void)scheduleLocked();
}

nixl_status_t
ucx_callback_slot_t::armPoster(void *request,
                               nixl_status_t post_status,
                               std::uint64_t timestamp_ns) noexcept {
    const std::lock_guard lock(mutex_);
    if (posterArmed_ || scheduled_ || delivered_ || state_ == nullptr ||
        continuations_ == nullptr) {
        return NIXL_ERR_NOT_ALLOWED;
    }
    posterArmed_ = true;
    posterRequest_ = request;
    postStatus_ = post_status;
    posterTimestampNs_ = timestamp_ns;

    if (post_status != NIXL_IN_PROG) {
        callbackArrived_ = true;
        callbackRequest_ = request;
        callbackStatus_ = post_status;
        callbackTimestampNs_ = timestamp_ns;
    } else if (request == nullptr) {
        callbackArrived_ = true;
        callbackStatus_ = NIXL_ERR_BACKEND;
        callbackTimestampNs_ = timestamp_ns;
    }
    return scheduleLocked();
}

bool
ucx_callback_slot_t::isDelivered() const noexcept {
    const std::lock_guard lock(mutex_);
    return delivered_;
}

nixl_status_t
ucx_callback_slot_t::scheduleLocked() noexcept {
    if (!callbackArrived_ || !posterArmed_) {
        return NIXL_SUCCESS;
    }
    if (scheduled_) {
        return NIXL_ERR_NOT_ALLOWED;
    }
    scheduled_ = true;
    const std::shared_ptr<ucx_callback_slot_t> self = shared_from_this();
    const nixl_status_t enqueue_status = continuations_->enqueue(
        [self]() noexcept { return self->deliverOnOwner(); });
    return enqueue_status;
}

nixl_status_t
ucx_callback_slot_t::deliverOnOwner() noexcept {
    void *request = nullptr;
    nixl_status_t status = NIXL_ERR_BACKEND;
    std::uint64_t timestamp_ns = 0;
    {
        const std::lock_guard lock(mutex_);
        if (!scheduled_ || delivered_ || !callbackArrived_ || !posterArmed_) {
            return NIXL_ERR_BACKEND;
        }
        delivered_ = true;
        request = posterRequest_;
        status = postStatus_ == NIXL_IN_PROG ? callbackStatus_ : postStatus_;
        timestamp_ns = std::max(callbackTimestampNs_, posterTimestampNs_);
        if (postStatus_ == NIXL_IN_PROG && callbackRequest_ != posterRequest_) {
            status = NIXL_ERR_BACKEND;
        }
    }

    if (request != nullptr && requestRelease_) {
        requestRelease_(request);
    }

    nixl_status_t completion_status = NIXL_ERR_BACKEND;
    switch (kind_) {
    case ucx_callback_kind_t::DATA_CHUNK:
        completion_status = state_->completeChunk(status, timestamp_ns);
        break;
    case ucx_callback_kind_t::ENDPOINT_FLUSH:
        completion_status = state_->completeFlush(status, timestamp_ns);
        break;
    case ucx_callback_kind_t::NOTIFICATION:
        completion_status = state_->completeNotification(status, timestamp_ns);
        break;
    }
    if (completion_status != NIXL_SUCCESS) {
        return completion_status;
    }
    return NIXL_SUCCESS;
}

} // namespace nixl::ucx
