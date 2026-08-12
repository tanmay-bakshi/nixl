/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_SRC_PLUGINS_UCX_UCX_TERMINAL_DEADLINE_H
#define NIXL_SRC_PLUGINS_UCX_UCX_TERMINAL_DEADLINE_H

#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

#include <nixl_types.h>

namespace nixl::ucx {

inline constexpr std::uint64_t nativeTransferTimeoutNs = 60'000'000'000ULL;

struct terminal_deadline_key_t {
    std::uint64_t handleIdentity = 0;
    std::uint64_t generation = 0;

    bool
    operator==(const terminal_deadline_key_t &) const = default;
};

struct terminal_deadline_inventory_t {
    std::size_t active = 0;
    std::uint64_t expired = 0;
    std::uint64_t retired = 0;
    nixl_status_t fatalStatus = NIXL_SUCCESS;
    bool accepting = false;
    bool ownerAlive = false;
};

struct terminal_deadline_snapshot_t {
    terminal_deadline_inventory_t inventory;
    std::vector<terminal_deadline_key_t> activeKeys;
};

enum class terminal_deadline_status_t {
    SUCCESS,
    EXPIRY_WON,
    RETIREMENT_WON,
    INVALID_ARGUMENT,
    DUPLICATE_DEADLINE,
    UNKNOWN_DEADLINE,
    CAPACITY_EXCEEDED,
    OWNER_CLOSED,
    OWNER_FAILED,
};

/**
 * Engine-lifetime monotonic deadline reactor for native transfer generations.
 *
 * The reactor owns only deadline selection. Expiry callbacks must enqueue the
 * terminal transition on the transfer's existing UCX progress owner; the
 * reactor never calls UCX or mutates transfer state directly.
 *
 * Exact keys are never reused during an owner lifetime. The bounded resolution
 * history diagnoses completion races; it is not an identity-recycling scheme.
 */
class terminal_deadline_owner_t final {
public:
    using expiry_t = std::function<nixl_status_t(const terminal_deadline_key_t &)>;
    using fatal_t = std::function<void(nixl_status_t)>;

    explicit terminal_deadline_owner_t(std::size_t capacity, fatal_t fatal = {});
    ~terminal_deadline_owner_t();

    terminal_deadline_owner_t(const terminal_deadline_owner_t &) = delete;
    terminal_deadline_owner_t &
    operator=(const terminal_deadline_owner_t &) = delete;

    [[nodiscard]] terminal_deadline_status_t
    arm(const terminal_deadline_key_t &key,
        std::uint64_t anchor_ns,
        std::uint64_t timeout_ns,
        expiry_t expiry);
    [[nodiscard]] terminal_deadline_status_t
    retire(const terminal_deadline_key_t &key);
    [[nodiscard]] terminal_deadline_status_t
    acknowledgeExpiry(const terminal_deadline_key_t &key);
    [[nodiscard]] nixl_status_t
    beginShutdown() noexcept;
    [[nodiscard]] nixl_status_t
    close() noexcept;
    [[nodiscard]] terminal_deadline_inventory_t
    inventory() const noexcept;
    [[nodiscard]] terminal_deadline_snapshot_t
    snapshot() const;
    [[nodiscard]] nixl_status_t
    fatalStatus() const noexcept;

    [[nodiscard]] static std::uint64_t
    monotonicTimestampNs() noexcept;

private:
    struct key_hash_t {
        [[nodiscard]] std::size_t
        operator()(const terminal_deadline_key_t &key) const noexcept;
    };

    struct record_t {
        enum class phase_t {
            ARMED,
            EXPIRY_CLAIMED,
        };

        std::uint64_t deadlineNs = 0;
        expiry_t expiry;
        phase_t phase = phase_t::ARMED;
    };

    enum class resolution_t {
        EXPIRY,
        RETIREMENT,
    };

    struct resolution_record_t {
        terminal_deadline_key_t key;
        resolution_t resolution = resolution_t::RETIREMENT;
        bool occupied = false;
    };

    struct heap_entry_t {
        terminal_deadline_key_t key;
        std::uint64_t deadlineNs = 0;
    };

    struct later_deadline_t {
        [[nodiscard]] bool
        operator()(const heap_entry_t &left, const heap_entry_t &right) const noexcept;
    };

    [[nodiscard]] terminal_deadline_status_t
    failLocked(nixl_status_t status) noexcept;
    [[nodiscard]] bool
    findResolutionLocked(const terminal_deadline_key_t &key,
                         resolution_t &resolution) const noexcept;
    void
    rememberResolutionLocked(const terminal_deadline_key_t &key, resolution_t resolution) noexcept;
    [[nodiscard]] bool
    signalControl() noexcept;
    void
    drainControl() noexcept;
    [[nodiscard]] bool
    programTimerLocked() noexcept;
    void
    run() noexcept;

    const std::size_t capacity_;
    const fatal_t fatal_;
    const int controlFd_;
    const int timerFd_;
    mutable std::mutex mutex_;
    std::condition_variable stateChanged_;
    std::unordered_map<terminal_deadline_key_t, record_t, key_hash_t> active_;
    std::vector<resolution_record_t> resolutionHistory_;
    std::size_t nextResolution_ = 0;
    std::priority_queue<heap_entry_t, std::vector<heap_entry_t>, later_deadline_t> deadlines_;
    std::thread owner_;
    std::uint64_t expired_ = 0;
    std::uint64_t retired_ = 0;
    std::size_t expiryCallbacksInFlight_ = 0;
    nixl_status_t fatalStatus_ = NIXL_SUCCESS;
    bool fatalNotified_ = false;
    bool accepting_ = true;
    bool closing_ = false;
    bool ownerAlive_ = false;
};

} // namespace nixl::ucx

#endif
