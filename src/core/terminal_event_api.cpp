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
    std::optional<nixlBackendTransferTransition> pending;
    {
        const std::lock_guard lock(mutex_);
        if (terminalCallback_ != nullptr) {
            return;
        }
        terminalCallback_ = std::move(terminal_callback);
        pending = std::move(pendingTransition_);
        pendingTransition_.reset();
    }
    if (pending.has_value()) {
        publishBound(*pending);
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
    {
        const std::lock_guard lock(mutex_);
        if (terminal_) {
            return;
        }
        if (terminalCallback_ == nullptr) {
            if (pendingTransition_.has_value()) {
                channelSubscription_->failInvalidPublication();
                terminal_ = true;
                return;
            }
            pendingTransition_ = transition;
            return;
        }
    }
    publishBound(transition);
}

void
nixlTerminalTransferAdapter::publishBound(
    const nixlBackendTransferTransition &transition) noexcept {
    const std::lock_guard lock(mutex_);
    if (terminal_ || terminalCallback_ == nullptr || channelSubscription_ == nullptr) {
        return;
    }

    terminal_ = true;
    terminalCallbackDelivered_ = true;
    const terminal_callback_t authority_commit = terminalCallback_;
    const bool invalid_publication =
        transition.binding.handleIdentity != binding_.handleIdentity ||
        transition.binding.generation != binding_.generation;
    if (invalid_publication) {
        channelSubscription_->failInvalidPublicationAuthoritative(authority_commit);
        channelSubscription_.reset();
        return;
    }

    const nixl::terminal_event_publish_result_t result =
        channelSubscription_->publishTransferAuthoritative(
            transition.status, transition.nativeTimestampNs, authority_commit);
    if (result == nixl::terminal_event_publish_result_t::INVALID_EVENT ||
        result == nixl::terminal_event_publish_result_t::ALREADY_PUBLISHED) {
        channelSubscription_->failInvalidPublicationAuthoritative(authority_commit);
    }
    channelSubscription_.reset();
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
    std::vector<nixlBackendCapabilityTransition> pending;
    {
        const std::lock_guard lock(mutex_);
        if (terminalCallback_ != nullptr) {
            return;
        }
        terminalCallback_ = std::move(terminal_callback);
        pending.swap(pendingTransitions_);
    }
    for (const nixlBackendCapabilityTransition &transition : pending) {
        publishBound(transition);
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
    {
        const std::lock_guard lock(mutex_);
        if (terminal_) {
            return;
        }
        if (terminalCallback_ == nullptr) {
            pendingTransitions_.push_back(transition);
            return;
        }
    }
    publishBound(transition);
}

void
nixlTerminalCapabilityAdapter::publishBound(
    const nixlBackendCapabilityTransition &transition) noexcept {
    const std::lock_guard lock(mutex_);
    if (terminal_ || terminalCallback_ == nullptr || channelSubscription_ == nullptr) {
        return;
    }

    const std::optional<nixl::terminal_capability_state_t> capability_state =
        mapCapabilityState(transition.state);
    const bool terminal_state =
        transition.state == nixl_backend_capability_state_t::FAILED ||
        transition.state == nixl_backend_capability_state_t::RETIRED;
    const bool valid_epoch = !hasCapabilityState_ ||
        transition.capabilityEpoch > lastCapabilityEpoch_ ||
        (terminal_state && transition.capabilityEpoch == lastCapabilityEpoch_);
    const bool invalid_publication =
        transition.remoteHandleIdentity != remoteHandleIdentity_ ||
        transition.remoteHandleGeneration != remoteHandleGeneration_ ||
        capability_state == std::nullopt || transition.capabilityEpoch == 0 ||
        !valid_epoch;
    const terminal_callback_t authority_commit = terminalCallback_;
    if (invalid_publication) {
        terminal_ = true;
        terminalCallbackDelivered_ = true;
        channelSubscription_->failInvalidPublicationAuthoritative(authority_commit);
        channelSubscription_.reset();
        return;
    }

    const nixl::terminal_event_publish_result_t result =
        channelSubscription_->publishCapabilityAuthoritative(
            *capability_state,
            transition.capabilityEpoch,
            transition.nativeTimestampNs,
            terminal_state,
            authority_commit);
    const bool publication_failed =
        result != nixl::terminal_event_publish_result_t::PUBLISHED;
    if (!publication_failed) {
        lastCapabilityEpoch_ = transition.capabilityEpoch;
        hasCapabilityState_ = true;
    }
    if (terminal_state || publication_failed) {
        terminal_ = true;
        terminalCallbackDelivered_ = true;
        channelSubscription_.reset();
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
    const std::lock_guard lock(mutex_);
    if (!info_.active) {
        return;
    }
    info_.active = false;
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
        capabilityAdapter_->release();
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
