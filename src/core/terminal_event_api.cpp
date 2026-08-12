/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "terminal_event_api.h"

#include <optional>
#include <utility>

namespace {

[[nodiscard]] std::optional<nixl::terminal_capability_state_t>
mapCapabilityState(nixl_backend_capability_state_t state) noexcept {
    switch (state) {
    case nixl_backend_capability_state_t::READY:
        return nixl::terminal_capability_state_t::READY;
    case nixl_backend_capability_state_t::FAILED:
        return nixl::terminal_capability_state_t::FAILED;
    case nixl_backend_capability_state_t::RETIRED:
        return nixl::terminal_capability_state_t::RETIRED;
    }
    return std::nullopt;
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
        if (terminal_ && !terminalCallbackDelivered_) {
            terminalCallbackDelivered_ = true;
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
        if (terminal_) {
            return;
        }
        subscription = channelSubscription_;
    }
    if (subscription == nullptr) {
        return;
    }
    bool invalid_publication = false;
    nixl::terminal_event_publish_result_t result =
        nixl::terminal_event_publish_result_t::INVALID_EVENT;
    if (transition.binding.handleIdentity != binding_.handleIdentity ||
        transition.binding.generation != binding_.generation) {
        invalid_publication = true;
    } else {
        result = subscription->publishTransfer(
            transition.status, transition.nativeTimestampNs);
        invalid_publication =
            result == nixl::terminal_event_publish_result_t::INVALID_EVENT ||
            result == nixl::terminal_event_publish_result_t::ALREADY_PUBLISHED;
    }

    if (invalid_publication) {
        subscription->failInvalidPublication();
    }
    subscription->release();
    terminal_callback_t callback;
    {
        const std::lock_guard lock(mutex_);
        terminal_ = true;
        if (terminalCallback_ != nullptr && !terminalCallbackDelivered_) {
            terminalCallbackDelivered_ = true;
            callback = terminalCallback_;
        }
    }
    if (callback != nullptr) {
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
        if (terminal_ && !terminalCallbackDelivered_) {
            terminalCallbackDelivered_ = true;
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
    uint64_t last_capability_epoch = 0;
    bool has_capability_state = false;
    {
        const std::lock_guard lock(mutex_);
        if (terminal_) {
            return;
        }
        subscription = channelSubscription_;
        last_capability_epoch = lastCapabilityEpoch_;
        has_capability_state = hasCapabilityState_;
    }
    if (subscription == nullptr) {
        return;
    }

    const std::optional<nixl::terminal_capability_state_t> capability_state =
        mapCapabilityState(transition.state);
    bool invalid_publication = false;
    nixl::terminal_event_publish_result_t result =
        nixl::terminal_event_publish_result_t::INVALID_EVENT;
    const bool terminal_state =
        transition.state == nixl_backend_capability_state_t::FAILED ||
        transition.state == nixl_backend_capability_state_t::RETIRED;
    const bool valid_epoch = !has_capability_state ||
        transition.capabilityEpoch > last_capability_epoch ||
        (terminal_state && transition.capabilityEpoch == last_capability_epoch);
    if (transition.remoteHandleIdentity != remoteHandleIdentity_ ||
        transition.remoteHandleGeneration != remoteHandleGeneration_ ||
        capability_state == std::nullopt || transition.capabilityEpoch == 0 ||
        !valid_epoch) {
        invalid_publication = true;
    } else {
        result = subscription->publishCapability(
            *capability_state,
            transition.capabilityEpoch,
            transition.nativeTimestampNs);
        invalid_publication =
            result == nixl::terminal_event_publish_result_t::INVALID_EVENT ||
            result == nixl::terminal_event_publish_result_t::ALREADY_PUBLISHED;
    }

    const bool publication_failed =
        result != nixl::terminal_event_publish_result_t::PUBLISHED;
    if (!invalid_publication && !publication_failed) {
        const std::lock_guard lock(mutex_);
        lastCapabilityEpoch_ = transition.capabilityEpoch;
        hasCapabilityState_ = true;
    }
    if (invalid_publication) {
        subscription->failInvalidPublication();
    }
    if (terminal_state || publication_failed) {
        subscription->release();
        terminal_callback_t callback;
        {
            const std::lock_guard lock(mutex_);
            terminal_ = true;
            if (terminalCallback_ != nullptr && !terminalCallbackDelivered_) {
                terminalCallbackDelivered_ = true;
                callback = terminalCallback_;
            }
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

void
nixlTerminalEventSubscriptionH::markTerminal() noexcept {
    std::shared_ptr<nixlTerminalTransferAdapter> transfer_adapter;
    std::shared_ptr<nixlTerminalCapabilityAdapter> capability_adapter;
    {
        const std::lock_guard lock(mutex_);
        if (!info_.active) {
            return;
        }
        info_.active = false;
        transfer_adapter = transferAdapter_;
        capability_adapter = capabilityAdapter_;
    }
    if (transfer_adapter != nullptr) {
        transfer_adapter->release();
    }
    if (capability_adapter != nullptr) {
        capability_adapter->release();
    }
}

nixl_status_t
nixlTerminalEventSubscriptionH::requestCancellation() noexcept {
    nixlBackendEventSubscription *backend_subscription = nullptr;
    nixl_terminal_event_kind_t kind = nixl_terminal_event_kind_t::TRANSFER;
    {
        const std::lock_guard lock(mutex_);
        if (!info_.active) {
            return NIXL_SUCCESS;
        }
        if (cancellationRequested_) {
            return NIXL_IN_PROG;
        }
        if (backendSubscription_ == nullptr) {
            return NIXL_ERR_BACKEND;
        }
        cancellationRequested_ = true;
        backend_subscription = backendSubscription_.get();
        kind = info_.kind;
    }

    const nixl_status_t status = backend_subscription->cancel();
    if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
        const std::lock_guard lock(mutex_);
        if (info_.active) {
            cancellationRequested_ = false;
        }
        return status;
    }

    if (kind == nixl_terminal_event_kind_t::CAPABILITY && status == NIXL_SUCCESS) {
        markTerminal();
    }
    {
        const std::lock_guard lock(mutex_);
        return info_.active ? NIXL_IN_PROG : NIXL_SUCCESS;
    }
}

nixlBackendEventSubscriptionInventory
nixlTerminalEventSubscriptionH::backendInventory() const noexcept {
    const std::lock_guard lock(mutex_);
    nixlBackendEventSubscriptionInventory inventory;
    if (backendSubscription_ != nullptr) {
        backendSubscription_->queryInventory(inventory);
    }
    return inventory;
}

nixl_status_t
nixlTerminalEventSubscriptionH::drainCancellation() noexcept {
    nixlBackendEventSubscription *backend_subscription = nullptr;
    {
        const std::lock_guard lock(mutex_);
        if (!info_.active) {
            return NIXL_SUCCESS;
        }
        if (backendSubscription_ == nullptr) {
            return NIXL_ERR_BACKEND;
        }
        backend_subscription = backendSubscription_.get();
    }

    const nixl_status_t status = backend_subscription->drainCancellation();
    if (status != NIXL_SUCCESS) {
        return status;
    }
    const std::lock_guard lock(mutex_);
    return info_.active ? NIXL_ERR_BACKEND : NIXL_SUCCESS;
}

bool
nixlTerminalEventSubscriptionH::claimPublicRelease() noexcept {
    const std::lock_guard lock(mutex_);
    if (publicReleaseClaimed_) {
        return false;
    }
    publicReleaseClaimed_ = true;
    return true;
}

void
nixlTerminalEventSubscriptionH::restorePublicRelease() noexcept {
    const std::lock_guard lock(mutex_);
    publicReleaseClaimed_ = false;
}

nixl_terminal_subscription_info_t
nixlTerminalEventSubscriptionH::snapshot() const noexcept {
    const std::lock_guard lock(mutex_);
    return info_;
}
