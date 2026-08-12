/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_SRC_CORE_TERMINAL_EVENT_API_H
#define NIXL_SRC_CORE_TERMINAL_EVENT_API_H

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "backend/backend_aux.h"
#include "nixl_types.h"
#include "terminal_event_channel.h"

class nixlBackendEngine;
class nixlTerminalEventSubscriptionTestPeer;

class nixlTerminalTransferAdapter final : public nixlBackendTransferTransitionSink {
public:
    using terminal_callback_t = std::function<void()>;

    nixlTerminalTransferAdapter(
        nixlBackendTransferEventBinding binding,
        std::shared_ptr<nixl::terminalEventChannel::subscription> channel_subscription);

    void
    bindTerminalCallback(std::function<void()> terminal_callback);

    [[nodiscard]] bool
    isTerminal() const noexcept;

    void
    publish(const nixlBackendTransferTransition &transition) noexcept override;

    void
    release() noexcept;

private:
    void
    publishBound(const nixlBackendTransferTransition &transition) noexcept;

    const nixlBackendTransferEventBinding binding_;
    mutable std::mutex mutex_;
    std::shared_ptr<nixl::terminalEventChannel::subscription> channelSubscription_;
    terminal_callback_t terminalCallback_;
    std::optional<nixlBackendTransferTransition> pendingTransition_;
    bool terminal_ = false;
    bool terminalCallbackDelivered_ = false;
};

class nixlTerminalCapabilityAdapter final : public nixlBackendCapabilityTransitionSink {
public:
    using terminal_callback_t = std::function<void()>;

    nixlTerminalCapabilityAdapter(
        uint64_t remote_handle_identity,
        uint64_t remote_handle_generation,
        std::shared_ptr<nixl::terminalEventChannel::subscription> channel_subscription);

    void
    bindTerminalCallback(std::function<void()> terminal_callback);

    [[nodiscard]] bool
    isTerminal() const noexcept;

    void
    publish(const nixlBackendCapabilityTransition &transition) noexcept override;

    void
    release() noexcept;

private:
    void
    publishBound(const nixlBackendCapabilityTransition &transition) noexcept;

    const uint64_t remoteHandleIdentity_;
    const uint64_t remoteHandleGeneration_;
    mutable std::mutex mutex_;
    std::shared_ptr<nixl::terminalEventChannel::subscription> channelSubscription_;
    terminal_callback_t terminalCallback_;
    std::vector<nixlBackendCapabilityTransition> pendingTransitions_;
    bool terminal_ = false;
    bool terminalCallbackDelivered_ = false;
    uint64_t lastCapabilityEpoch_ = 0;
    bool hasCapabilityState_ = false;
};

class nixlTerminalEventChannelH {
private:
    nixlTerminalEventChannelH(uint64_t owner_identity, size_t capacity);

    const uint64_t ownerIdentity_;
    nixl::terminalEventChannel channel_;

    friend class nixlAgent;
};

class nixlTerminalEventSubscriptionH {
private:
    nixlTerminalEventSubscriptionH(
        uint64_t owner_identity,
        uint64_t subscription_identity,
        nixl_terminal_subscription_info_t info,
        nixlBackendEngine *backend,
        const nixlXferReqH *request,
        std::unique_ptr<nixlBackendEventSubscription> backend_subscription,
        std::shared_ptr<nixlTerminalTransferAdapter> transfer_adapter);

    nixlTerminalEventSubscriptionH(
        uint64_t owner_identity,
        uint64_t subscription_identity,
        nixl_terminal_subscription_info_t info,
        nixlBackendEngine *backend,
        std::unique_ptr<nixlBackendEventSubscription> backend_subscription,
        std::shared_ptr<nixlTerminalCapabilityAdapter> capability_adapter);

    void
    markTerminal() noexcept;

    [[nodiscard]] nixl_status_t
    requestCancellation() noexcept;

    [[nodiscard]] nixlBackendEventSubscriptionInventory
    backendInventory() const noexcept;

    [[nodiscard]] nixl_status_t
    drainCancellation() noexcept;

    [[nodiscard]] bool
    claimPublicRelease() noexcept;

    void
    restorePublicRelease() noexcept;

    [[nodiscard]] nixl_terminal_subscription_info_t
    snapshot() const noexcept;

private:
    const uint64_t ownerIdentity_;
    const uint64_t subscriptionIdentity_;
    nixl_terminal_subscription_info_t info_;
    mutable std::mutex mutex_;
    nixlBackendEngine *const backend_;
    const nixlXferReqH *const request_;
    std::unique_ptr<nixlBackendEventSubscription> backendSubscription_;
    std::shared_ptr<nixlTerminalTransferAdapter> transferAdapter_;
    std::shared_ptr<nixlTerminalCapabilityAdapter> capabilityAdapter_;
    bool cancellationRequested_ = false;
    bool publicReleaseClaimed_ = false;

    friend class nixlAgent;
    friend class nixlTerminalEventSubscriptionTestPeer;
};

#endif // NIXL_SRC_CORE_TERMINAL_EVENT_API_H
