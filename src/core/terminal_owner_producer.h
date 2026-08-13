/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_SRC_CORE_TERMINAL_OWNER_PRODUCER_H
#define NIXL_SRC_CORE_TERMINAL_OWNER_PRODUCER_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>

#include "native_producer_api.h"
#include "terminal_event_api.h"

constexpr std::size_t NIXL_TERMINAL_OWNER_BINDING_DIGEST_BYTES = 32;
constexpr std::int32_t NIXL_TERMINAL_OWNER_REASON_TRANSFER_FAILED = 1;
constexpr std::int32_t NIXL_TERMINAL_OWNER_REASON_INVALID_TRANSITION = 2;

using nixl_terminal_owner_binding_digest_t =
    std::array<std::uint8_t, NIXL_TERMINAL_OWNER_BINDING_DIGEST_BYTES>;

enum class nixl_terminal_owner_producer_fatal_t : std::uint32_t {
    NONE = 0,
    DUPLICATE_BINDING = 1,
    UNKNOWN_BINDING = 2,
    INVALID_BINDING_STATE = 3,
    INVALID_TRANSITION = 4,
    OWNER_SUBMISSION_FAILURE = 5,
    SUBMISSION_AFTER_STOP = 6,
    RETIREMENT_FAILURE = 7,
    JOIN_FAILURE = 8,
    CLOSE_WITH_ACTIVE_CALLBACKS = 9,
    CLOSE_WITH_RETAINED_BINDINGS = 10,
    CLOSE_BEFORE_PRODUCER_JOIN = 11,
};

struct nixl_terminal_owner_producer_inventory_t {
    std::size_t registeringBindings = 0;
    std::size_t submittedBindings = 0;
    std::size_t activeCallbacks = 0;
    std::size_t activeRegistrations = 0;
    std::uint64_t totalSubscriptions = 0;
    std::uint64_t totalDelivered = 0;
    std::uint64_t successfulTerminalEvents = 0;
    std::uint64_t failureTerminalEvents = 0;
    std::uint64_t ownerSubmissionFailures = 0;
    bool admissionOpen = true;
    bool retirementRequested = false;
    bool joined = false;
    bool closed = false;
    nixl_terminal_owner_producer_fatal_t fatal = nixl_terminal_owner_producer_fatal_t::NONE;
    int fatalStatus = 0;
    nixl_terminal_owner_binding_digest_t fatalBinding{};
    bool fatalHasBinding = false;
};

class nixlTerminalOwnerProducerState;
class nixlTerminalOwnerProducerTestPeer;

class nixlTerminalOwnerProducerH final {
public:
    nixlTerminalOwnerProducerH(const sglang_terminal_owner_producer_api_v1 *api, void *context);

    nixlTerminalOwnerProducerH(const nixlTerminalOwnerProducerH &) = delete;
    nixlTerminalOwnerProducerH &
    operator=(const nixlTerminalOwnerProducerH &) = delete;

    void
    stopAdmission() noexcept;

    [[nodiscard]] nixl_status_t
    join(std::uint64_t timeout_ns) noexcept;

    [[nodiscard]] nixl_status_t
    close() noexcept;

    [[nodiscard]] nixl_terminal_owner_producer_inventory_t
    inventory() const noexcept;

private:
    [[nodiscard]] nixl_status_t
    bindAgent(std::uint64_t agent_identity) noexcept;

    [[nodiscard]] nixl_status_t
    beginSubscription(const nixl_terminal_owner_binding_digest_t &binding) noexcept;

    void
    subscriptionRegistrationSucceeded(const nixl_terminal_owner_binding_digest_t &binding) noexcept;

    std::shared_ptr<nixlTerminalOwnerProducerState> state_;
    std::mutex agentMutex_;
    std::uint64_t agentIdentity_ = 0;

    friend class nixlAgent;
    friend class nixlTerminalOwnerTransferAdapter;
    friend class nixlTerminalOwnerProducerTestPeer;
};

class nixlTerminalOwnerTransferAdapter final : public nixlTerminalTransferAdapterBase {
public:
    nixlTerminalOwnerTransferAdapter(nixlBackendTransferEventBinding transfer_binding,
                                     nixl_terminal_owner_binding_digest_t owner_binding,
                                     std::shared_ptr<nixlTerminalOwnerProducerState> producer);

    void
    bindTerminalCallback(terminal_callback_t terminal_callback) override;

    [[nodiscard]] bool
    isTerminal() const noexcept override;

    void
    publish(const nixlBackendTransferTransition &transition) noexcept override;

    void
    release() noexcept override;

    void
    abortRegistration(nixl_status_t status) noexcept;

private:
    void
    publishBound(const nixlBackendTransferTransition &transition) noexcept;

    const nixlBackendTransferEventBinding transferBinding_;
    const nixl_terminal_owner_binding_digest_t ownerBinding_;
    mutable std::mutex mutex_;
    std::shared_ptr<nixlTerminalOwnerProducerState> producer_;
    terminal_callback_t terminalCallback_;
    std::optional<nixlBackendTransferTransition> pendingTransition_;
    bool terminal_ = false;
};

[[nodiscard]] const char *
nixlTerminalOwnerProducerFatalName(nixl_terminal_owner_producer_fatal_t fatal) noexcept;

#endif // NIXL_SRC_CORE_TERMINAL_OWNER_PRODUCER_H
