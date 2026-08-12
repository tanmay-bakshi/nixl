/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "terminal_event_api.h"

#include <utility>

namespace {

[[nodiscard]] nixl::terminal_capability_state_t
mapCapabilityState(nixl_backend_capability_state_t state) noexcept {
    switch (state) {
    case nixl_backend_capability_state_t::READY:
        return nixl::terminal_capability_state_t::READY;
    case nixl_backend_capability_state_t::FAILED:
        return nixl::terminal_capability_state_t::FAILED;
    case nixl_backend_capability_state_t::RETIRED:
        return nixl::terminal_capability_state_t::RETIRED;
    }
    return nixl::terminal_capability_state_t::FAILED;
}

} // namespace

nixlTerminalTransferAdapter::nixlTerminalTransferAdapter(
    nixlBackendTransferEventBinding binding,
    std::shared_ptr<nixl::terminalEventChannel::subscription> channel_subscription)
    : binding_(binding),
      channelSubscription_(std::move(channel_subscription)) {}

void
nixlTerminalTransferAdapter::bindTerminalCallback(
    std::function<void()> terminal_callback) {
    terminal_callback_t callback;
    {
        const std::lock_guard lock(mutex_);
        terminalCallback_ = std::move(terminal_callback);
        if (terminal_) {
            callback = terminalCallback_;
        }
    }
    if (callback != nullptr) {
        callback();
    }
}

bool
nixlTerminalTransferAdapter::isTerminal() const noexcept {
    const std::lock_guard lock(mutex_);
    return terminal_;
}

void
nixlTerminalTransferAdapter::publish(
    const nixlBackendTransferTransition &transition) noexcept {
    std::shared_ptr<nixl::terminalEventChannel::subscription> subscription;
    {
        const std::lock_guard lock(mutex_);
        subscription = channelSubscription_;
    }
    if (subscription == nullptr) {
        return;
    }
    if (transition.binding.handleIdentity != binding_.handleIdentity ||
        transition.binding.generation != binding_.generation) {
        subscription->failInvalidPublication();
        subscription->release();
        return;
    }

    const nixl::terminal_event_publish_result_t result =
        subscription->publishTransfer(transition.status, transition.nativeTimestampNs);
    subscription->release();
    std::function<void()> callback;
    {
        const std::lock_guard lock(mutex_);
        terminal_ = true;
        callback = terminalCallback_;
    }
    if (result == nixl::terminal_event_publish_result_t::PUBLISHED && callback != nullptr) {
        callback();
    }
}

void
nixlTerminalTransferAdapter::release() noexcept {
    std::shared_ptr<nixl::terminalEventChannel::subscription> subscription;
    {
        const std::lock_guard lock(mutex_);
        subscription = std::move(channelSubscription_);
    }
    if (subscription != nullptr) {
        subscription->release();
    }
}

nixlTerminalCapabilityAdapter::nixlTerminalCapabilityAdapter(
    uint64_t remote_handle_identity,
    uint64_t remote_handle_generation,
    std::shared_ptr<nixl::terminalEventChannel::subscription> channel_subscription)
    : remoteHandleIdentity_(remote_handle_identity),
      remoteHandleGeneration_(remote_handle_generation),
      channelSubscription_(std::move(channel_subscription)) {}

void
nixlTerminalCapabilityAdapter::bindTerminalCallback(
    std::function<void()> terminal_callback) {
    terminal_callback_t callback;
    {
        const std::lock_guard lock(mutex_);
        terminalCallback_ = std::move(terminal_callback);
        if (terminal_) {
            callback = terminalCallback_;
        }
    }
    if (callback != nullptr) {
        callback();
    }
}

bool
nixlTerminalCapabilityAdapter::isTerminal() const noexcept {
    const std::lock_guard lock(mutex_);
    return terminal_;
}

void
nixlTerminalCapabilityAdapter::publish(
    const nixlBackendCapabilityTransition &transition) noexcept {
    std::shared_ptr<nixl::terminalEventChannel::subscription> subscription;
    {
        const std::lock_guard lock(mutex_);
        subscription = channelSubscription_;
    }
    if (subscription == nullptr) {
        return;
    }
    if (transition.remoteHandleIdentity != remoteHandleIdentity_ ||
        transition.remoteHandleGeneration != remoteHandleGeneration_) {
        subscription->failInvalidPublication();
        subscription->release();
        return;
    }

    static_cast<void>(subscription->publishCapability(
        mapCapabilityState(transition.state),
        transition.capabilityEpoch,
        transition.nativeTimestampNs));
    if (transition.state == nixl_backend_capability_state_t::RETIRED) {
        subscription->release();
        std::function<void()> callback;
        {
            const std::lock_guard lock(mutex_);
            terminal_ = true;
            callback = terminalCallback_;
        }
        if (callback != nullptr) {
            callback();
        }
    }
}

void
nixlTerminalCapabilityAdapter::release() noexcept {
    std::shared_ptr<nixl::terminalEventChannel::subscription> subscription;
    {
        const std::lock_guard lock(mutex_);
        subscription = std::move(channelSubscription_);
    }
    if (subscription != nullptr) {
        subscription->release();
    }
}

nixlTerminalEventChannelH::nixlTerminalEventChannelH(
    uint64_t owner_identity,
    size_t capacity)
    : ownerIdentity_(owner_identity),
      channel_(capacity) {}

nixlTerminalEventSubscriptionH::nixlTerminalEventSubscriptionH(
    uint64_t owner_identity,
    uint64_t subscription_identity,
    nixl_terminal_subscription_info_t info,
    nixlBackendEngine *backend,
    const nixlXferReqH *request,
    std::unique_ptr<nixlBackendEventSubscription> backend_subscription,
    std::shared_ptr<nixlTerminalTransferAdapter> transfer_adapter)
    : ownerIdentity_(owner_identity),
      subscriptionIdentity_(subscription_identity),
      info_(info),
      backend_(backend),
      request_(request),
      backendSubscription_(std::move(backend_subscription)),
      transferAdapter_(std::move(transfer_adapter)) {}

nixlTerminalEventSubscriptionH::nixlTerminalEventSubscriptionH(
    uint64_t owner_identity,
    uint64_t subscription_identity,
    nixl_terminal_subscription_info_t info,
    nixlBackendEngine *backend,
    std::unique_ptr<nixlBackendEventSubscription> backend_subscription,
    std::shared_ptr<nixlTerminalCapabilityAdapter> capability_adapter)
    : ownerIdentity_(owner_identity),
      subscriptionIdentity_(subscription_identity),
      info_(info),
      backend_(backend),
      request_(nullptr),
      backendSubscription_(std::move(backend_subscription)),
      capabilityAdapter_(std::move(capability_adapter)) {}

std::unique_ptr<nixlBackendEventSubscription>
nixlTerminalEventSubscriptionH::takeBackendSubscription() noexcept {
    const std::lock_guard lock(mutex_);
    return std::move(backendSubscription_);
}

void
nixlTerminalEventSubscriptionH::restoreBackendSubscription(
    std::unique_ptr<nixlBackendEventSubscription> backend_subscription) noexcept {
    const std::lock_guard lock(mutex_);
    backendSubscription_ = std::move(backend_subscription);
}

void
nixlTerminalEventSubscriptionH::finishRelease() noexcept {
    const std::lock_guard lock(mutex_);
    info_.active = false;
    if (transferAdapter_ != nullptr) {
        transferAdapter_->release();
        transferAdapter_.reset();
    }
    if (capabilityAdapter_ != nullptr) {
        capabilityAdapter_->release();
        capabilityAdapter_.reset();
    }
}

nixl_terminal_subscription_info_t
nixlTerminalEventSubscriptionH::snapshot() const noexcept {
    const std::lock_guard lock(mutex_);
    return info_;
}
