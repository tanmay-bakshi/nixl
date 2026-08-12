/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_SRC_CORE_TERMINAL_EVENT_CHANNEL_H
#define NIXL_SRC_CORE_TERMINAL_EVENT_CHANNEL_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <variant>
#include <vector>

#include "nixl_types.h"

namespace nixl {

class terminalEventChannelState;

enum class terminal_event_kind_t {
    TRANSFER,
    CAPABILITY,
};

enum class terminal_capability_state_t {
    READY,
    FAILED,
    RETIRED,
};

enum class terminal_event_publish_result_t {
    PUBLISHED,
    ALREADY_PUBLISHED,
    INVALID_EVENT,
    CHANNEL_CLOSED,
    CHANNEL_FATAL,
};

enum class terminal_channel_fatal_t : std::uint32_t {
    NONE = 0,
    QUEUE_OVERFLOW = 1U << 0U,
    EVENTFD_FAILURE = 1U << 1U,
    ACTIVE_SUBSCRIPTIONS_ON_CLOSE = 1U << 2U,
    INVALID_PUBLICATION = 1U << 3U,
};

enum class terminal_channel_close_result_t {
    CLOSED,
    ALREADY_CLOSED,
    ACTIVE_SUBSCRIPTIONS,
};

struct terminal_event_binding_t {
    terminal_event_kind_t kind = terminal_event_kind_t::TRANSFER;
    std::uint64_t ownerCookie = 0;
    std::uint64_t identity = 0;
    std::uint64_t generation = 0;
};

struct terminal_event_t {
    terminal_event_kind_t kind = terminal_event_kind_t::TRANSFER;
    std::uint64_t ownerCookie = 0;
    std::uint64_t identity = 0;
    std::uint64_t generation = 0;
    std::variant<nixl_status_t, terminal_capability_state_t> result = NIXL_ERR_NOT_READY;
    std::uint64_t epoch = 0;
    std::uint64_t nativeTimestampNs = 0;
};

struct terminal_channel_health_t {
    terminal_channel_fatal_t fatal = terminal_channel_fatal_t::NONE;
    int eventfdError = 0;
};

struct terminal_channel_inventory_t {
    std::size_t capacity = 0;
    std::size_t queuedEvents = 0;
    std::size_t activeSubscriptions = 0;
    bool acceptingSubscriptions = false;
    bool closed = false;
    terminal_channel_health_t health;
};

struct terminal_event_batch_t {
    std::vector<terminal_event_t> events;
    std::uint64_t wakeCount = 0;
    terminal_channel_inventory_t inventory;
};

class terminalEventChannel {
public:
    class subscription {
    public:
        subscription(const subscription &) = delete;
        subscription &
        operator=(const subscription &) = delete;
        ~subscription();

        [[nodiscard]] terminal_event_publish_result_t
        publishTransfer(nixl_status_t status, std::uint64_t native_timestamp_ns = 0) noexcept;

        [[nodiscard]] terminal_event_publish_result_t
        publishCapability(terminal_capability_state_t capability_state,
                          std::uint64_t epoch,
                          std::uint64_t native_timestamp_ns = 0) noexcept;

        void
        release() noexcept;

        void
        failInvalidPublication() noexcept;

    private:
        subscription(std::shared_ptr<terminalEventChannelState> state,
                     terminal_event_binding_t binding);

        std::shared_ptr<terminalEventChannelState> state_;
        terminal_event_binding_t binding_;
        std::mutex mutex_;
        bool transferPublished_ = false;
        bool released_ = false;

        friend class terminalEventChannel;
    };

    explicit terminalEventChannel(std::size_t capacity);
    terminalEventChannel(const terminalEventChannel &) = delete;
    terminalEventChannel &
    operator=(const terminalEventChannel &) = delete;
    ~terminalEventChannel();

    [[nodiscard]] std::shared_ptr<subscription>
    subscribe(const terminal_event_binding_t &binding);

    /** The returned descriptor is borrowed and must only be polled, never read or closed. */
    [[nodiscard]] int
    fileno() const noexcept;

    [[nodiscard]] std::optional<terminal_event_t>
    take() noexcept;

    [[nodiscard]] terminal_event_batch_t
    drain();

    [[nodiscard]] terminal_channel_inventory_t
    inventory() const noexcept;

    /** Stops admission. A clean close does not signal the data fd; lifecycle owns shutdown wakeup.
     */
    [[nodiscard]] terminal_channel_close_result_t
    close() noexcept;

private:
    std::shared_ptr<terminalEventChannelState> state_;
};

[[nodiscard]] constexpr terminal_channel_fatal_t
operator|(terminal_channel_fatal_t lhs, terminal_channel_fatal_t rhs) noexcept {
    return static_cast<terminal_channel_fatal_t>(static_cast<std::uint32_t>(lhs) |
                                                 static_cast<std::uint32_t>(rhs));
}

[[nodiscard]] constexpr bool
hasTerminalChannelFatal(terminal_channel_fatal_t value, terminal_channel_fatal_t fatal) noexcept {
    return (static_cast<std::uint32_t>(value) & static_cast<std::uint32_t>(fatal)) != 0;
}

} // namespace nixl

#endif // NIXL_SRC_CORE_TERMINAL_EVENT_CHANNEL_H
