/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "ucx_terminal_deadline.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <utility>

#include <poll.h>
#include <sys/eventfd.h>
#include <sys/timerfd.h>
#include <time.h>
#include <unistd.h>

namespace nixl::ucx {
namespace {

    constexpr clockid_t terminalDeadlineClock = CLOCK_MONOTONIC;
    constexpr std::uint64_t nanosecondsPerSecond = 1000000000ULL;

    [[nodiscard]] int
    makeEventFd() {
        const int descriptor = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
        if (descriptor < 0) {
            throw std::system_error(
                errno, std::generic_category(), "create terminal deadline eventfd");
        }
        return descriptor;
    }

    [[nodiscard]] int
    makeTimerFd() {
        const int descriptor = timerfd_create(terminalDeadlineClock, TFD_CLOEXEC | TFD_NONBLOCK);
        if (descriptor < 0) {
            throw std::system_error(
                errno, std::generic_category(), "create terminal deadline timerfd");
        }
        return descriptor;
    }

} // namespace

terminal_deadline_owner_t::terminal_deadline_owner_t(std::size_t capacity, fatal_t fatal)
    : capacity_(capacity),
      fatal_(std::move(fatal)),
      controlFd_(makeEventFd()),
      timerFd_([this]() {
          try {
              return makeTimerFd();
          }
          catch (...) {
              static_cast<void>(::close(controlFd_));
              throw;
          }
      }()) {
    if (capacity_ == 0) {
        static_cast<void>(::close(timerFd_));
        static_cast<void>(::close(controlFd_));
        throw std::invalid_argument("terminal deadline owner requires positive capacity");
    }
    try {
        active_.reserve(capacity_);
        resolutionHistory_.resize(capacity_);
        owner_ = std::thread([this]() { run(); });
    }
    catch (...) {
        static_cast<void>(::close(timerFd_));
        static_cast<void>(::close(controlFd_));
        throw;
    }
}

terminal_deadline_owner_t::~terminal_deadline_owner_t() {
    static_cast<void>(close());
    static_cast<void>(::close(timerFd_));
    static_cast<void>(::close(controlFd_));
}

std::size_t
terminal_deadline_owner_t::key_hash_t::operator()(
    const terminal_deadline_key_t &key) const noexcept {
    const std::size_t first = std::hash<std::uint64_t>{}(key.handleIdentity);
    const std::size_t second = std::hash<std::uint64_t>{}(key.generation);
    return first ^ (second + 0x9e3779b97f4a7c15ULL + (first << 6U) + (first >> 2U));
}

bool
terminal_deadline_owner_t::later_deadline_t::operator()(const heap_entry_t &left,
                                                        const heap_entry_t &right) const noexcept {
    if (left.deadlineNs != right.deadlineNs) {
        return left.deadlineNs > right.deadlineNs;
    }
    if (left.key.handleIdentity != right.key.handleIdentity) {
        return left.key.handleIdentity > right.key.handleIdentity;
    }
    return left.key.generation > right.key.generation;
}

terminal_deadline_status_t
terminal_deadline_owner_t::arm(const terminal_deadline_key_t &key,
                               std::uint64_t anchor_ns,
                               std::uint64_t timeout_ns,
                               expiry_t expiry) {
    if (key.handleIdentity == 0 || key.generation == 0 || anchor_ns == 0 || timeout_ns == 0 ||
        timeout_ns > std::numeric_limits<std::uint64_t>::max() - anchor_ns || !expiry) {
        return terminal_deadline_status_t::INVALID_ARGUMENT;
    }
    const std::uint64_t deadline_ns = anchor_ns + timeout_ns;
    const std::lock_guard lock(mutex_);
    if (fatalStatus_ != NIXL_SUCCESS) {
        return terminal_deadline_status_t::OWNER_FAILED;
    }
    if (!accepting_ || closing_) {
        return terminal_deadline_status_t::OWNER_CLOSED;
    }
    if (active_.contains(key)) {
        return terminal_deadline_status_t::DUPLICATE_DEADLINE;
    }
    if (active_.size() >= capacity_) {
        static_cast<void>(failLocked(NIXL_ERR_BACKEND));
        return terminal_deadline_status_t::CAPACITY_EXCEEDED;
    }
    try {
        active_.emplace(key, record_t{.deadlineNs = deadline_ns, .expiry = std::move(expiry)});
        try {
            deadlines_.push({.key = key, .deadlineNs = deadline_ns});
        }
        catch (...) {
            active_.erase(key);
            throw;
        }
    }
    catch (const std::bad_alloc &) {
        static_cast<void>(failLocked(NIXL_ERR_BACKEND));
        return terminal_deadline_status_t::OWNER_FAILED;
    }
    if (!signalControl()) {
        active_.erase(key);
        static_cast<void>(failLocked(NIXL_ERR_BACKEND));
        return terminal_deadline_status_t::OWNER_FAILED;
    }
    return terminal_deadline_status_t::SUCCESS;
}

terminal_deadline_status_t
terminal_deadline_owner_t::retire(const terminal_deadline_key_t &key) {
    if (key.handleIdentity == 0 || key.generation == 0) {
        return terminal_deadline_status_t::INVALID_ARGUMENT;
    }
    const std::lock_guard lock(mutex_);
    const auto known = active_.find(key);
    if (known == active_.end()) {
        resolution_t resolution = resolution_t::RETIREMENT;
        if (findResolutionLocked(key, resolution)) {
            return resolution == resolution_t::EXPIRY ? terminal_deadline_status_t::EXPIRY_WON :
                                                        terminal_deadline_status_t::RETIREMENT_WON;
        }
        if (fatalStatus_ != NIXL_SUCCESS) {
            return terminal_deadline_status_t::OWNER_FAILED;
        }
        if (closing_) {
            return terminal_deadline_status_t::OWNER_CLOSED;
        }
        return terminal_deadline_status_t::UNKNOWN_DEADLINE;
    }
    if (fatalStatus_ != NIXL_SUCCESS) {
        return terminal_deadline_status_t::OWNER_FAILED;
    }
    if (closing_) {
        return terminal_deadline_status_t::OWNER_CLOSED;
    }
    if (known->second.phase == record_t::phase_t::EXPIRY_CLAIMED) {
        return terminal_deadline_status_t::EXPIRY_WON;
    }
    rememberResolutionLocked(key, resolution_t::RETIREMENT);
    active_.erase(known);
    ++retired_;
    stateChanged_.notify_all();
    if (!signalControl()) {
        static_cast<void>(failLocked(NIXL_ERR_BACKEND));
        return terminal_deadline_status_t::OWNER_FAILED;
    }
    return terminal_deadline_status_t::SUCCESS;
}

terminal_deadline_status_t
terminal_deadline_owner_t::acknowledgeExpiry(const terminal_deadline_key_t &key) {
    if (key.handleIdentity == 0 || key.generation == 0) {
        return terminal_deadline_status_t::INVALID_ARGUMENT;
    }
    const std::lock_guard lock(mutex_);
    const auto known = active_.find(key);
    if (known == active_.end()) {
        resolution_t resolution = resolution_t::RETIREMENT;
        if (findResolutionLocked(key, resolution)) {
            return resolution == resolution_t::EXPIRY ? terminal_deadline_status_t::EXPIRY_WON :
                                                        terminal_deadline_status_t::RETIREMENT_WON;
        }
        if (fatalStatus_ != NIXL_SUCCESS) {
            return terminal_deadline_status_t::OWNER_FAILED;
        }
        if (closing_) {
            return terminal_deadline_status_t::OWNER_CLOSED;
        }
        return terminal_deadline_status_t::UNKNOWN_DEADLINE;
    }
    if (known->second.phase != record_t::phase_t::EXPIRY_CLAIMED) {
        return terminal_deadline_status_t::RETIREMENT_WON;
    }
    active_.erase(known);
    stateChanged_.notify_all();
    return terminal_deadline_status_t::SUCCESS;
}

nixl_status_t
terminal_deadline_owner_t::beginShutdown() noexcept {
    std::unique_lock lock(mutex_);
    accepting_ = false;
    if (fatalStatus_ != NIXL_SUCCESS) {
        return fatalStatus_;
    }
    const std::uint64_t now_ns = monotonicTimestampNs();
    if (now_ns == 0) {
        static_cast<void>(failLocked(NIXL_ERR_BACKEND));
        return fatalStatus_;
    }
    try {
        for (auto &[key, record] : active_) {
            if (record.phase != record_t::phase_t::ARMED) {
                continue;
            }
            record.deadlineNs = now_ns;
            deadlines_.push({.key = key, .deadlineNs = now_ns});
        }
    }
    catch (const std::bad_alloc &) {
        static_cast<void>(failLocked(NIXL_ERR_BACKEND));
        return fatalStatus_;
    }
    if (!signalControl()) {
        static_cast<void>(failLocked(NIXL_ERR_BACKEND));
        return fatalStatus_;
    }
    stateChanged_.wait(lock, [this]() {
        if (fatalStatus_ != NIXL_SUCCESS) {
            return true;
        }
        return active_.empty() && expiryCallbacksInFlight_ == 0;
    });
    return fatalStatus_;
}

nixl_status_t
terminal_deadline_owner_t::close() noexcept {
    bool join = false;
    {
        const std::lock_guard lock(mutex_);
        accepting_ = false;
        if (!closing_) {
            closing_ = true;
            if (!active_.empty() && fatalStatus_ == NIXL_SUCCESS) {
                static_cast<void>(failLocked(NIXL_ERR_CANCELED));
            }
        }
        join = owner_.joinable() && owner_.get_id() != std::this_thread::get_id();
    }
    static_cast<void>(signalControl());
    if (join) {
        owner_.join();
    }
    const std::lock_guard lock(mutex_);
    return fatalStatus_;
}

terminal_deadline_inventory_t
terminal_deadline_owner_t::inventory() const noexcept {
    const std::lock_guard lock(mutex_);
    return {
        .active = active_.size(),
        .expired = expired_,
        .retired = retired_,
        .fatalStatus = fatalStatus_,
        .accepting = accepting_,
        .ownerAlive = ownerAlive_,
    };
}

terminal_deadline_snapshot_t
terminal_deadline_owner_t::snapshot() const {
    terminal_deadline_snapshot_t result;
    {
        const std::lock_guard lock(mutex_);
        result.inventory = {
            .active = active_.size(),
            .expired = expired_,
            .retired = retired_,
            .fatalStatus = fatalStatus_,
            .accepting = accepting_,
            .ownerAlive = ownerAlive_,
        };
        result.activeKeys.reserve(active_.size());
        for (const auto &[key, record] : active_) {
            static_cast<void>(record);
            result.activeKeys.push_back(key);
        }
    }
    std::sort(result.activeKeys.begin(),
              result.activeKeys.end(),
              [](const terminal_deadline_key_t &left, const terminal_deadline_key_t &right) {
                  if (left.handleIdentity != right.handleIdentity) {
                      return left.handleIdentity < right.handleIdentity;
                  }
                  return left.generation < right.generation;
              });
    return result;
}

nixl_status_t
terminal_deadline_owner_t::fatalStatus() const noexcept {
    const std::lock_guard lock(mutex_);
    return fatalStatus_;
}

std::uint64_t
terminal_deadline_owner_t::monotonicTimestampNs() noexcept {
    timespec now = {};
    if (clock_gettime(terminalDeadlineClock, &now) != 0) {
        return 0;
    }
    return static_cast<std::uint64_t>(now.tv_sec) * nanosecondsPerSecond +
        static_cast<std::uint64_t>(now.tv_nsec);
}

terminal_deadline_status_t
terminal_deadline_owner_t::failLocked(nixl_status_t status) noexcept {
    if (fatalStatus_ == NIXL_SUCCESS) {
        fatalStatus_ = status < NIXL_SUCCESS ? status : NIXL_ERR_BACKEND;
    }
    accepting_ = false;
    if (!fatalNotified_ && fatal_) {
        fatalNotified_ = true;
        fatal_(fatalStatus_);
    }
    stateChanged_.notify_all();
    return terminal_deadline_status_t::OWNER_FAILED;
}

bool
terminal_deadline_owner_t::findResolutionLocked(const terminal_deadline_key_t &key,
                                                resolution_t &resolution) const noexcept {
    for (const resolution_record_t &record : resolutionHistory_) {
        if (record.occupied && record.key == key) {
            resolution = record.resolution;
            return true;
        }
    }
    return false;
}

void
terminal_deadline_owner_t::rememberResolutionLocked(const terminal_deadline_key_t &key,
                                                    resolution_t resolution) noexcept {
    resolutionHistory_[nextResolution_] = {
        .key = key,
        .resolution = resolution,
        .occupied = true,
    };
    ++nextResolution_;
    if (nextResolution_ == resolutionHistory_.size()) {
        nextResolution_ = 0;
    }
}

bool
terminal_deadline_owner_t::signalControl() noexcept {
    const std::uint64_t signal = 1;
    while (true) {
        const ssize_t written = write(controlFd_, &signal, sizeof(signal));
        if (written == static_cast<ssize_t>(sizeof(signal))) {
            return true;
        }
        if (written < 0 && errno == EINTR) {
            continue;
        }
        return written < 0 && errno == EAGAIN;
    }
}

void
terminal_deadline_owner_t::drainControl() noexcept {
    std::uint64_t value = 0;
    while (true) {
        const ssize_t consumed = read(controlFd_, &value, sizeof(value));
        if (consumed == static_cast<ssize_t>(sizeof(value))) {
            continue;
        }
        if (consumed < 0 && errno == EINTR) {
            continue;
        }
        return;
    }
}

bool
terminal_deadline_owner_t::programTimerLocked() noexcept {
    while (!deadlines_.empty()) {
        const heap_entry_t &candidate = deadlines_.top();
        const auto active = active_.find(candidate.key);
        if (active != active_.end() && active->second.deadlineNs == candidate.deadlineNs &&
            active->second.phase == record_t::phase_t::ARMED) {
            break;
        }
        deadlines_.pop();
    }

    itimerspec timer = {};
    if (!deadlines_.empty()) {
        const std::uint64_t deadline_ns = deadlines_.top().deadlineNs;
        timer.it_value.tv_sec = static_cast<time_t>(deadline_ns / nanosecondsPerSecond);
        timer.it_value.tv_nsec = static_cast<long>(deadline_ns % nanosecondsPerSecond);
    }
    return timerfd_settime(timerFd_, TFD_TIMER_ABSTIME, &timer, nullptr) == 0;
}

void
terminal_deadline_owner_t::run() noexcept {
    struct expired_record_t {
        terminal_deadline_key_t key;
        expiry_t expiry;
    };

    std::vector<expired_record_t> expired;
    try {
        expired.reserve(capacity_);
    }
    catch (const std::bad_alloc &) {
        const std::lock_guard lock(mutex_);
        ownerAlive_ = false;
        static_cast<void>(failLocked(NIXL_ERR_BACKEND));
        return;
    }
    {
        const std::lock_guard lock(mutex_);
        ownerAlive_ = true;
        if (!programTimerLocked()) {
            static_cast<void>(failLocked(NIXL_ERR_BACKEND));
        }
    }

    pollfd descriptors[2] = {
        {.fd = controlFd_, .events = POLLIN, .revents = 0},
        {.fd = timerFd_, .events = POLLIN, .revents = 0},
    };
    while (true) {
        const int status = poll(descriptors, 2, -1);
        if (status < 0 && errno == EINTR) {
            continue;
        }
        if (status < 0) {
            const std::lock_guard lock(mutex_);
            static_cast<void>(failLocked(NIXL_ERR_BACKEND));
            break;
        }
        if ((descriptors[0].revents & POLLIN) != 0) {
            drainControl();
        }
        expired.clear();
        if ((descriptors[1].revents & POLLIN) != 0) {
            std::uint64_t expirations = 0;
            ssize_t consumed = 0;
            do {
                consumed = read(timerFd_, &expirations, sizeof(expirations));
            } while (consumed < 0 && errno == EINTR);
            const std::uint64_t now_ns = monotonicTimestampNs();
            const std::lock_guard lock(mutex_);
            if (consumed != static_cast<ssize_t>(sizeof(expirations)) || now_ns == 0) {
                static_cast<void>(failLocked(NIXL_ERR_BACKEND));
                continue;
            }
            while (!deadlines_.empty() && deadlines_.top().deadlineNs <= now_ns) {
                const heap_entry_t candidate = deadlines_.top();
                deadlines_.pop();
                const auto active = active_.find(candidate.key);
                if (active == active_.end() || active->second.deadlineNs != candidate.deadlineNs) {
                    continue;
                }
                rememberResolutionLocked(candidate.key, resolution_t::EXPIRY);
                expired.push_back(
                    {.key = candidate.key, .expiry = std::move(active->second.expiry)});
                active->second.phase = record_t::phase_t::EXPIRY_CLAIMED;
                ++expiryCallbacksInFlight_;
                ++expired_;
            }
            if (!programTimerLocked()) {
                static_cast<void>(failLocked(NIXL_ERR_BACKEND));
            }
        } else {
            const std::lock_guard lock(mutex_);
            if (!programTimerLocked()) {
                static_cast<void>(failLocked(NIXL_ERR_BACKEND));
            }
        }

        for (expired_record_t &record : expired) {
            nixl_status_t expiry_status = NIXL_ERR_BACKEND;
            try {
                expiry_status = record.expiry(record.key);
            }
            catch (...) {
                expiry_status = NIXL_ERR_BACKEND;
            }
            {
                const std::lock_guard lock(mutex_);
                if (expiryCallbacksInFlight_ == 0) {
                    static_cast<void>(failLocked(NIXL_ERR_BACKEND));
                    continue;
                }
                --expiryCallbacksInFlight_;
                if (expiry_status != NIXL_SUCCESS) {
                    static_cast<void>(failLocked(expiry_status));
                }
                stateChanged_.notify_all();
            }
        }

        const std::lock_guard lock(mutex_);
        if (closing_) {
            break;
        }
    }
    const std::lock_guard lock(mutex_);
    ownerAlive_ = false;
}

} // namespace nixl::ucx
