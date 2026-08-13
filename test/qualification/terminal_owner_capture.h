/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_TEST_QUALIFICATION_TERMINAL_OWNER_CAPTURE_H
#define NIXL_TEST_QUALIFICATION_TERMINAL_OWNER_CAPTURE_H

#include "native_producer_api.h"

#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace nixl::qualification {

class terminal_owner_capture_t final {
public:
    terminal_owner_capture_t() {
        api_ = {
            .abi_version = SGLANG_TERMINAL_OWNER_PRODUCER_ABI_VERSION,
            .struct_size = sizeof(sglang_terminal_owner_producer_api_v1),
            .event_struct_size = sizeof(sglang_terminal_owner_producer_event_v1),
            .flags = SGLANG_TERMINAL_OWNER_PRODUCER_REQUIRED_FLAGS,
            .submit = submit,
            .retire = retire,
            .join = join,
        };
    }

    [[nodiscard]] const sglang_terminal_owner_producer_api_v1 *
    api() const noexcept {
        return &api_;
    }

    [[nodiscard]] sglang_terminal_owner_producer_event_v1
    wait(std::chrono::milliseconds timeout) {
        std::unique_lock lock(mutex_);
        const bool ready =
            condition_.wait_for(lock, timeout, [this]() { return events_.size() == 1; });
        if (!ready) {
            throw std::runtime_error("direct owner did not receive one terminal event");
        }
        return events_.front();
    }

private:
    static int
    submit(void *context, const sglang_terminal_owner_producer_event_v1 *event) noexcept {
        if (context == nullptr || event == nullptr) {
            return EINVAL;
        }
        auto &owner = *static_cast<terminal_owner_capture_t *>(context);
        {
            const std::lock_guard lock(owner.mutex_);
            if (!owner.events_.empty()) {
                return EALREADY;
            }
            owner.events_.push_back(*event);
        }
        owner.condition_.notify_all();
        return 0;
    }

    static int
    retire(void *context) noexcept {
        if (context == nullptr) {
            return EINVAL;
        }
        auto &owner = *static_cast<terminal_owner_capture_t *>(context);
        const std::lock_guard lock(owner.mutex_);
        if (owner.retired_) {
            return EALREADY;
        }
        owner.retired_ = true;
        return 0;
    }

    static int
    join(void *context, std::uint64_t timeout_ns) noexcept {
        if (context == nullptr || timeout_ns == 0) {
            return EINVAL;
        }
        auto &owner = *static_cast<terminal_owner_capture_t *>(context);
        const std::lock_guard lock(owner.mutex_);
        return owner.retired_ ? 0 : EPROTO;
    }

    sglang_terminal_owner_producer_api_v1 api_{};
    std::mutex mutex_;
    std::condition_variable condition_;
    std::vector<sglang_terminal_owner_producer_event_v1> events_;
    bool retired_ = false;
};

} // namespace nixl::qualification

#endif // NIXL_TEST_QUALIFICATION_TERMINAL_OWNER_CAPTURE_H
