/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ucx_backend.h"
#include "common/nixl_log.h"
#include "serdes/serdes.h"

#include <algorithm>
#include <optional>
#include <atomic>
#include <limits>
#include <future>
#include <set>
#include <string.h>
#include <unistd.h>
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include <asio.hpp>

/****************************************
 * Backend request management
*****************************************/

namespace {

std::atomic<uint64_t> next_handle_identity{1};
std::atomic<uint64_t> next_connection_identity{1};

[[nodiscard]] uint64_t
allocateIdentity(std::atomic<uint64_t> &next_identity, const char *kind) {
    uint64_t identity = next_identity.load(std::memory_order_relaxed);
    do {
        if (identity == std::numeric_limits<uint64_t>::max()) {
            throw std::overflow_error(std::string("UCX ") + kind + " identity space exhausted");
        }
    } while (!next_identity.compare_exchange_weak(
        identity, identity + 1, std::memory_order_relaxed, std::memory_order_relaxed));
    return identity;
}

[[nodiscard]] int
hexNibble(char value) noexcept {
    if (value >= '0' && value <= '9') {
        return value - '0';
    }
    if (value >= 'a' && value <= 'f') {
        return value - 'a' + 10;
    }
    if (value >= 'A' && value <= 'F') {
        return value - 'A' + 10;
    }
    return -1;
}

[[nodiscard]] bool
parseWireUuid(std::string_view encoded,
              nixl::ucx::notif_wire_uuid_t &uuid) noexcept {
    nixl::ucx::notif_wire_uuid_t parsed;
    size_t nibble_index = 0;
    for (const char value : encoded) {
        if (value == '-') {
            continue;
        }
        const int nibble = hexNibble(value);
        if (nibble < 0 || nibble_index >= parsed.bytes.size() * 2) {
            return false;
        }
        const size_t byte_index = nibble_index / 2;
        if ((nibble_index & 1U) == 0) {
            parsed.bytes[byte_index] = static_cast<std::uint8_t>(nibble << 4);
        } else {
            parsed.bytes[byte_index] |= static_cast<std::uint8_t>(nibble);
        }
        ++nibble_index;
    }
    if (nibble_index != parsed.bytes.size() * 2 ||
        !nixl::ucx::isCanonicalNotifWireUuid(parsed)) {
        return false;
    }
    uuid = parsed;
    return true;
}

[[nodiscard]] nixl_status_t
notifStateToNixl(nixl::ucx::notif_state_status_t status) noexcept {
    using enum nixl::ucx::notif_state_status_t;
    switch (status) {
    case SUCCESS:
        return NIXL_SUCCESS;
    case NOT_READY:
        return NIXL_ERR_NOT_READY;
    case UNKNOWN_ROUTE:
    case UNKNOWN_CAPABILITY:
        return NIXL_ERR_NOT_FOUND;
    case INVALID_ARGUMENT:
        return NIXL_ERR_INVALID_PARAM;
    case ROUTE_RETIRED:
    case ROUTE_CONFLICT:
    case ROUTE_FAILED:
    case STALE_EPOCH:
    case EPOCH_CONFLICT:
    case EPOCH_EXHAUSTED:
        return NIXL_ERR_NOT_ALLOWED;
    case RANDOM_FAILURE:
        return NIXL_ERR_BACKEND;
    }
    return NIXL_ERR_UNKNOWN;
}

[[nodiscard]] nixl_status_t
notifSubscriptionToNixl(
    nixl::ucx::notif_route_subscription_status_t status) noexcept {
    using enum nixl::ucx::notif_route_subscription_status_t;
    switch (status) {
    case SUCCESS:
        return NIXL_SUCCESS;
    case INVALID_ARGUMENT:
        return NIXL_ERR_INVALID_PARAM;
    case UNKNOWN_ROUTE:
    case UNKNOWN_SUBSCRIPTION:
        return NIXL_ERR_NOT_FOUND;
    case DUPLICATE_SUBSCRIPTION:
    case GENERATION_EXHAUSTED:
        return NIXL_ERR_NOT_ALLOWED;
    }
    return NIXL_ERR_UNKNOWN;
}

} // namespace

nixl_status_t
nixlUcxNotificationQueue::push(nixlAuthenticatedNotification &&notification) {
    if (failed_) {
        return NIXL_ERR_NOT_ALLOWED;
    }

    const bool item_overflow = notification.remoteAgent.size() > maxBytes_ ||
        notification.payload.size() > maxBytes_ - notification.remoteAgent.size();
    size_t notification_size = 0;
    if (!item_overflow) {
        notification_size = notification.remoteAgent.size() + notification.payload.size();
    }
    const bool count_overflow = notifications_.size() >= maxNotifications_;
    const bool byte_overflow = item_overflow || queuedBytes_ > maxBytes_ ||
        notification_size > maxBytes_ - queuedBytes_;
    if (count_overflow || byte_overflow) {
        failed_ = true;
        notifications_.clear();
        queuedBytes_ = 0;
        NIXL_ERROR << "UCX authenticated notification queue exceeded its process bounds";
        return NIXL_ERR_NOT_ALLOWED;
    }

    queuedBytes_ += notification_size;
    notifications_.push_back(std::move(notification));
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxNotificationQueue::drainLegacy(notif_list_t &notifications) {
    if (!notifications.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (failed_) {
        notifications_.clear();
        queuedBytes_ = 0;
        return NIXL_ERR_NOT_ALLOWED;
    }

    notifications.reserve(notifications_.size());
    for (auto &notification : notifications_) {
        notifications.emplace_back(
            std::move(notification.remoteAgent), std::move(notification.payload));
    }
    notifications_.clear();
    queuedBytes_ = 0;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxNotificationQueue::drainAuthenticated(
    authenticated_notif_list_t &notifications) {
    if (!notifications.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (failed_) {
        notifications_.clear();
        queuedBytes_ = 0;
        return NIXL_ERR_NOT_ALLOWED;
    }

    notifications_.swap(notifications);
    queuedBytes_ = 0;
    return NIXL_SUCCESS;
}

void
nixlUcxNotificationQueue::poison() {
    failed_ = true;
    notifications_.clear();
    queuedBytes_ = 0;
}

class nixlUcxTerminalArm final :
    public nixl::ucx::terminal_submission_sink_t,
    public std::enable_shared_from_this<nixlUcxTerminalArm> {
    struct slot_record_t {
        nixlUcxWorker *worker = nullptr;
        std::shared_ptr<nixl::ucx::ucx_callback_slot_t> slot;
    };

public:
    nixlUcxTerminalArm(
        nixlBackendTransferEventBinding binding,
        std::shared_ptr<nixlBackendTransferTransitionSink> sink,
        std::shared_ptr<nixlUcxAttestationState> attestation,
        std::vector<nixlUcxWorker *> workers)
        : binding_(binding),
          sink_(std::move(sink)),
          attestation_(std::move(attestation)),
          workers_(std::move(workers)) {}

    [[nodiscard]] nixl_status_t
    begin(bool has_notification) {
        const std::lock_guard lock(mutex_);
        if (terminal_ || publishing_ || state_ != nullptr ||
            attestation_->getGeneration() != binding_.generation) {
            return NIXL_ERR_NOT_ALLOWED;
        }
        state_ = std::make_shared<nixl::ucx::terminal_submission_state_t>(
            binding_.handleIdentity,
            binding_.handleIdentity,
            binding_.generation,
            1,
            1,
            has_notification,
            shared_from_this());
        return NIXL_SUCCESS;
    }

    [[nodiscard]] nixl_status_t
    makeSlot(
        nixlUcxWorker *worker,
        nixl::ucx::ucx_callback_kind_t kind,
        nixl::ucx::ucx_callback_slot_t::owner_before_completion_t before,
        std::shared_ptr<nixl::ucx::ucx_callback_slot_t> &slot) {
        std::shared_ptr<nixl::ucx::terminal_submission_state_t> state;
        {
            const std::lock_guard lock(mutex_);
            if (terminal_ || publishing_ || state_ == nullptr || worker == nullptr ||
                !worker->hasProgressOwner()) {
                return NIXL_ERR_NOT_ALLOWED;
            }
            state = state_;
        }
        const nixl_status_t register_status =
            kind == nixl::ucx::ucx_callback_kind_t::DATA_CHUNK ?
            state->registerChunk() :
            (kind == nixl::ucx::ucx_callback_kind_t::ENDPOINT_FLUSH ?
                 state->registerFlush() : NIXL_SUCCESS);
        if (register_status != NIXL_SUCCESS) {
            return register_status;
        }

        const std::weak_ptr<nixlUcxTerminalArm> weak_self = shared_from_this();
        const nixl_status_t slot_status = worker->makeTerminalCallbackSlot(
            state,
            kind,
            slot,
            std::move(before),
            [weak_self]() {
                if (const std::shared_ptr<nixlUcxTerminalArm> self = weak_self.lock();
                    self != nullptr) {
                    self->notifySlotDelivered();
                }
            });
        if (slot_status != NIXL_SUCCESS) {
            const nixl_status_t unregister_status =
                kind == nixl::ucx::ucx_callback_kind_t::DATA_CHUNK ?
                state->unregisterChunk() :
                (kind == nixl::ucx::ucx_callback_kind_t::ENDPOINT_FLUSH ?
                     state->unregisterFlush() : NIXL_SUCCESS);
            if (unregister_status != NIXL_SUCCESS) {
                return NIXL_ERR_BACKEND;
            }
            return slot_status;
        }
        if (slot == nullptr) {
            const nixl_status_t unregister_status =
                kind == nixl::ucx::ucx_callback_kind_t::DATA_CHUNK ?
                state->unregisterChunk() :
                (kind == nixl::ucx::ucx_callback_kind_t::ENDPOINT_FLUSH ?
                     state->unregisterFlush() : NIXL_SUCCESS);
            return unregister_status == NIXL_SUCCESS ? NIXL_ERR_BACKEND : unregister_status;
        }
        {
            const std::lock_guard lock(mutex_);
            slots_.push_back({worker, slot});
            ++activeSlots_;
        }
        return NIXL_SUCCESS;
    }

    [[nodiscard]] nixl_status_t
    seal(nixl::ucx::terminal_submission_state_t::notification_post_t notification_post) {
        std::shared_ptr<nixl::ucx::terminal_submission_state_t> state;
        {
            const std::lock_guard lock(mutex_);
            if (terminal_ || state_ == nullptr) {
                return NIXL_ERR_NOT_ALLOWED;
            }
            state = state_;
        }
        return state->sealPosting(
            std::move(notification_post), nixl::ucx::terminalProgressTimestampNs());
    }

    [[nodiscard]] nixl_status_t
    cancel() noexcept {
        const nixl_status_t status = fail(NIXL_ERR_CANCELED);
        if (status == NIXL_IN_PROG) {
            std::unique_lock lock(mutex_);
            delivered_.wait(lock, [this]() { return terminal_; });
            return terminalStatus_ == NIXL_ERR_CANCELED ? NIXL_SUCCESS : terminalStatus_;
        }
        if (status != NIXL_ERR_CANCELED && status != NIXL_SUCCESS) {
            return status;
        }
        std::unique_lock lock(mutex_);
        delivered_.wait(lock, [this]() { return terminal_; });
        return terminalStatus_ == NIXL_ERR_CANCELED ? NIXL_SUCCESS : terminalStatus_;
    }

    [[nodiscard]] nixl_status_t
    fail(nixl_status_t failure_status) noexcept {
        if (failure_status >= NIXL_SUCCESS) {
            return NIXL_ERR_INVALID_PARAM;
        }
        std::shared_ptr<nixl::ucx::terminal_submission_state_t> state;
        std::vector<slot_record_t> slots;
        {
            const std::lock_guard lock(mutex_);
            if (terminal_) {
                return NIXL_SUCCESS;
            }
            if (publishing_) {
                return NIXL_IN_PROG;
            }
            state = state_;
            slots = slots_;
        }
        if (state == nullptr) {
            nixl::ucx::terminal_submission_result_t result{
                .ownerCookie = binding_.handleIdentity,
                .handleIdentity = binding_.handleIdentity,
                .generation = binding_.generation,
                .status = failure_status,
                .nativeTimestampNs = nixl::ucx::terminalProgressTimestampNs(),
            };
            result.diagnostics.autonomous = true;
            result.diagnostics.terminalStatus = failure_status;
            return publishTerminal(result);
        }

        std::unordered_map<
            nixlUcxWorker *,
            std::vector<std::shared_ptr<nixl::ucx::ucx_callback_slot_t>>>
            cancellations;
        for (const slot_record_t &record : slots) {
            if (record.slot->requestForCancellation() != nullptr) {
                cancellations[record.worker].push_back(record.slot);
            }
        }
        for (auto &[worker, pending_slots] : cancellations) {
            const nixl_status_t status = worker->enqueueContinuation(
                [worker, pending_slots = std::move(pending_slots)]() {
                    for (const auto &slot : pending_slots) {
                        if (void *request = slot->requestForCancellation();
                            request != nullptr) {
                            worker->reqCancel(request);
                        }
                    }
                    return NIXL_SUCCESS;
                });
            if (status != NIXL_SUCCESS) {
                return status;
            }
        }
        const nixl_status_t cancel_status = state->cancel(
            failure_status, nixl::ucx::terminalProgressTimestampNs());
        if (cancel_status != NIXL_SUCCESS && cancel_status != failure_status) {
            return cancel_status;
        }
        return failure_status;
    }

    [[nodiscard]] bool
    isTerminal() const noexcept {
        const std::lock_guard lock(mutex_);
        return terminal_;
    }

    [[nodiscard]] nixl_status_t
    status() const noexcept {
        const std::lock_guard lock(mutex_);
        return terminal_ ? terminalStatus_ : NIXL_IN_PROG;
    }

    [[nodiscard]] const nixlBackendTransferEventBinding &
    binding() const noexcept {
        return binding_;
    }

    void
    inventory(nixlBackendEventSubscriptionInventory &inventory) const noexcept {
        const std::lock_guard lock(mutex_);
        std::size_t queued = 0;
        for (const slot_record_t &record : slots_) {
            if (record.slot->isScheduled()) {
                ++queued;
            }
        }
        inventory = {
            .backendProducers = terminal_ ? 0U : 1U,
            .activeCallbackSlots = activeSlots_,
            .queuedOwnerContinuations = queued,
        };
    }

    nixl_status_t
    publishTerminal(
        const nixl::ucx::terminal_submission_result_t &result) noexcept override {
        std::optional<nixl::ucx::terminal_submission_result_t> ready;
        {
            const std::lock_guard lock(mutex_);
            if (terminal_ || publishing_ || pendingTerminal_.has_value()) {
                return NIXL_ERR_NOT_ALLOWED;
            }
            pendingTerminal_ = result;
            if (activeSlots_ == 0) {
                publishing_ = true;
                ready = std::move(pendingTerminal_);
                pendingTerminal_.reset();
            }
        }
        if (!ready.has_value()) {
            return NIXL_SUCCESS;
        }
        return finalizeTerminal(std::move(ready).value());
    }

private:
    [[nodiscard]] nixl_status_t
    finalizeTerminal(nixl::ucx::terminal_submission_result_t completed) noexcept {
        completed.diagnostics.activeCallbackSlotsAtTerminal = 0;
        completed.diagnostics.continuationDepthAtTerminal = 0;
        completed.diagnostics.terminalPublishTimestampNs =
            nixl::ucx::terminalProgressTimestampNs();
        nixl_status_t status = completed.status;
        const nixl_xfer_attestation_t snapshot = attestation_->snapshot();
        if (snapshot.generation == binding_.generation) {
            if (status != NIXL_SUCCESS) {
                attestation_->fail(binding_.generation,
                                   status,
                                   "autonomous terminal failure");
            }
            const nixl_status_t evidence_status = attestation_->recordTerminalProgress(
                binding_.generation, completed.diagnostics);
            if (evidence_status != NIXL_SUCCESS && status == NIXL_SUCCESS) {
                status = evidence_status;
                attestation_->fail(binding_.generation,
                                   status,
                                   "terminal progress evidence failed");
            }
        }
        sink_->publish({
            .binding = binding_,
            .status = status,
            .nativeTimestampNs = completed.nativeTimestampNs,
        });
        {
            const std::lock_guard lock(mutex_);
            terminal_ = true;
            terminalStatus_ = status;
            publishing_ = false;
            state_.reset();
            slots_.clear();
        }
        delivered_.notify_all();
        return status == NIXL_SUCCESS ? NIXL_SUCCESS : status;
    }

    void
    notifySlotDelivered() noexcept {
        std::optional<nixl::ucx::terminal_submission_result_t> ready;
        {
            const std::lock_guard lock(mutex_);
            if (activeSlots_ == 0) {
                return;
            }
            --activeSlots_;
            if (activeSlots_ == 0 && pendingTerminal_.has_value() && !publishing_) {
                publishing_ = true;
                ready = std::move(pendingTerminal_);
                pendingTerminal_.reset();
            }
        }
        if (ready.has_value()) {
            static_cast<void>(finalizeTerminal(std::move(ready).value()));
        }
    }

    const nixlBackendTransferEventBinding binding_;
    const std::shared_ptr<nixlBackendTransferTransitionSink> sink_;
    const std::shared_ptr<nixlUcxAttestationState> attestation_;
    const std::vector<nixlUcxWorker *> workers_;
    mutable std::mutex mutex_;
    std::condition_variable delivered_;
    std::shared_ptr<nixl::ucx::terminal_submission_state_t> state_;
    std::vector<slot_record_t> slots_;
    std::optional<nixl::ucx::terminal_submission_result_t> pendingTerminal_;
    std::size_t activeSlots_ = 0;
    nixl_status_t terminalStatus_ = NIXL_IN_PROG;
    bool publishing_ = false;
    bool terminal_ = false;
};

class nixlUcxTerminalSubscription final : public nixlBackendEventSubscription {
public:
    explicit nixlUcxTerminalSubscription(std::shared_ptr<nixlUcxTerminalArm> arm)
        : arm_(std::move(arm)) {}

    nixl_status_t
    cancel() noexcept override {
        return arm_->cancel();
    }

    void
    queryInventory(nixlBackendEventSubscriptionInventory &inventory) const noexcept override {
        arm_->inventory(inventory);
    }

    nixl_status_t
    drainCancellation() noexcept override {
        return arm_->cancel();
    }

private:
    const std::shared_ptr<nixlUcxTerminalArm> arm_;
};

class nixlUcxCapabilitySink final :
    public nixl::ucx::notif_route_transition_sink_t {
public:
    explicit nixlUcxCapabilitySink(
        std::shared_ptr<nixlBackendCapabilityTransitionSink> sink)
        : sink_(std::move(sink)) {}

    void
    publish(const nixl::ucx::notif_route_transition_t &transition) noexcept override {
        nixl_backend_capability_state_t state =
            nixl_backend_capability_state_t::FAILED;
        switch (transition.state) {
        case nixl::ucx::notif_route_transition_state_t::READY:
            state = nixl_backend_capability_state_t::READY;
            break;
        case nixl::ucx::notif_route_transition_state_t::FAILED:
            state = nixl_backend_capability_state_t::FAILED;
            break;
        case nixl::ucx::notif_route_transition_state_t::RETIRED:
            state = nixl_backend_capability_state_t::RETIRED;
            break;
        }
        sink_->publish({
            .remoteHandleIdentity = transition.route.handleIdentity,
            .remoteHandleGeneration = transition.route.handleGeneration,
            .state = state,
            .capabilityEpoch = transition.capabilityEpoch,
            .nativeTimestampNs = transition.nativeTimestampNs,
        });
    }

private:
    const std::shared_ptr<nixlBackendCapabilityTransitionSink> sink_;
};

class nixlUcxCapabilitySubscription final : public nixlBackendEventSubscription {
public:
    nixlUcxCapabilitySubscription(
        nixl::ucx::notif_capability_state_t &state,
        nixl::ucx::notif_route_subscription_t subscription,
        std::shared_ptr<nixlUcxCapabilitySink> sink)
        : state_(state),
          subscription_(std::move(subscription)),
          sink_(std::move(sink)) {}

    ~nixlUcxCapabilitySubscription() override {
        {
            const std::lock_guard lock(mutex_);
            if (canceled_) {
                return;
            }
        }
        const nixl_status_t status = cancel();
        if (status != NIXL_SUCCESS) {
            NIXL_FATAL << "Exact-route capability subscription destruction failed to drain: "
                       << status;
            std::terminate();
        }
    }

    nixl_status_t
    cancel() noexcept override {
        const std::lock_guard lock(mutex_);
        if (canceled_) {
            return NIXL_SUCCESS;
        }
        const nixl_status_t status = notifSubscriptionToNixl(
            state_.unsubscribeRemoteNotificationState(subscription_));
        if (status == NIXL_SUCCESS) {
            canceled_ = true;
        }
        return status;
    }

    void
    queryInventory(nixlBackendEventSubscriptionInventory &inventory) const noexcept override {
        const nixl::ucx::notif_route_subscription_inventory_t native_inventory =
            state_.querySubscriptionInventory(subscription_);
        inventory = {
            .backendProducers = native_inventory.retainedSubscriptions,
            .activeCallbackSlots = native_inventory.inFlightDeliveries,
        };
    }

private:
    nixl::ucx::notif_capability_state_t &state_;
    const nixl::ucx::notif_route_subscription_t subscription_;
    const std::shared_ptr<nixlUcxCapabilitySink> sink_;
    mutable std::mutex mutex_;
    bool canceled_ = false;
};

class nixlUcxBackendReqH : public nixlBackendReqH {
private:
    enum class pending_kind_t {
        DATA,
        ENDPOINT_FLUSH,
        NOTIFICATION,
    };

    struct pending_request_t {
        nixlUcxReq request;
        pending_kind_t kind;
        uint64_t endpointIdentity;
        uint64_t generation;
    };

    std::set<ucx_connection_ptr_t> connections_;
    std::vector<pending_request_t> requests_;
    nixlUcxWorker *worker_;
    size_t workerId_;
    std::shared_ptr<nixlUcxAttestationState> attestation_;
    mutable std::mutex terminalMutex_;
    std::shared_ptr<nixlUcxTerminalArm> armedTerminal_;
    std::shared_ptr<nixlUcxTerminalArm> activeTerminal_;

    [[nodiscard]] nixl_status_t
    checkConnection(const nixl_status_t status = NIXL_SUCCESS) const {
        NIXL_ASSERT(!connections_.empty());
        for (const auto &conn : connections_) {
            const nixl_status_t conn_status = conn->getEp(workerId_)->checkTxState();
            if (conn_status != NIXL_SUCCESS) {
                return conn_status;
            }
        }
        return status;
    }

protected:
    void
    setWorker(nixlUcxWorker *worker, size_t worker_id) {
        NIXL_ASSERT(worker_ == nullptr || worker == nullptr);
        worker_ = worker;
        workerId_ = worker_id;
    }

    void
    setAttestationState(const std::shared_ptr<nixlUcxAttestationState> &attestation) {
        attestation_ = attestation;
    }

public:
    // Notification to be sent after completion of all requests
    struct Notif {
        std::vector<std::uint8_t> frame;
        const ucx_connection_ptr_t connection;

        Notif(std::vector<std::uint8_t> wire_frame,
              ucx_connection_ptr_t remote_connection)
            : frame(std::move(wire_frame)),
              connection(std::move(remote_connection)) {}
    };

    std::optional<Notif> notif;

    nixlUcxBackendReqH(
        nixlUcxWorker *worker,
        size_t worker_id,
        std::shared_ptr<nixlUcxAttestationState> attestation = nullptr)
        : worker_(worker),
          workerId_(worker_id),
          attestation_(attestation != nullptr ?
                           std::move(attestation) :
                           std::make_shared<nixlUcxAttestationState>(
                               allocateIdentity(next_handle_identity, "handle"))) {}

    [[nodiscard]] nixl_status_t
    prepareAttestation(nixl_xfer_op_t operation,
                       const nixl_meta_dlist_t &local,
                       const nixl_meta_dlist_t &remote,
                       const std::string &local_agent,
                       const std::string &remote_agent,
                       const nixl_remote_agent_authority_t *remote_agent_authority) {
        return attestation_->prepare(
            operation, local, remote, local_agent, remote_agent, remote_agent_authority);
    }

    [[nodiscard]] nixl_status_t
    beginSubmission(bool has_notification) {
        if (!requests_.empty() || notif.has_value()) {
            return NIXL_ERR_REPOST_ACTIVE;
        }
        std::shared_ptr<nixlUcxTerminalArm> armed;
        {
            const std::lock_guard lock(terminalMutex_);
            if (activeTerminal_ != nullptr && !activeTerminal_->isTerminal()) {
                return NIXL_ERR_REPOST_ACTIVE;
            }
            activeTerminal_.reset();
            if (armedTerminal_ != nullptr &&
                armedTerminal_->binding().generation != attestation_->getGeneration() + 1) {
                return NIXL_ERR_NOT_ALLOWED;
            }
            armed = armedTerminal_;
        }

        const nixl_status_t begin_status = attestation_->beginSubmission();
        if (begin_status != NIXL_SUCCESS) {
            return begin_status;
        }
        if (armed == nullptr) {
            return NIXL_SUCCESS;
        }
        const nixl_status_t arm_status = armed->begin(has_notification);
        if (arm_status != NIXL_SUCCESS) {
            attestation_->fail(attestation_->getGeneration(),
                               arm_status,
                               "terminal subscription activation failed");
            return arm_status;
        }
        {
            const std::lock_guard lock(terminalMutex_);
            if (armedTerminal_ != armed) {
                attestation_->fail(attestation_->getGeneration(),
                                   NIXL_ERR_BACKEND,
                                   "terminal subscription ownership changed during activation");
                return NIXL_ERR_BACKEND;
            }
            armedTerminal_.reset();
            activeTerminal_ = std::move(armed);
        }
        return NIXL_SUCCESS;
    }

    [[nodiscard]] nixl_status_t
    armTerminal(const nixlBackendTransferEventBinding &binding,
                const std::shared_ptr<nixlBackendTransferTransitionSink> &sink,
                const std::vector<nixlUcxWorker *> &workers,
                std::unique_ptr<nixlBackendEventSubscription> &subscription) {
        subscription.reset();
        const nixl_xfer_attestation_t attestation = attestation_->snapshot();
        if (sink == nullptr || binding.handleIdentity != attestation.handleIdentity ||
            binding.generation != attestation.generation + 1 || workers.empty()) {
            return NIXL_ERR_INVALID_PARAM;
        }
        for (nixlUcxWorker *worker : workers) {
            if (worker == nullptr || !worker->hasProgressOwner()) {
                return NIXL_ERR_NOT_SUPPORTED;
            }
        }
        const auto arm = std::make_shared<nixlUcxTerminalArm>(
            binding, sink, attestation_, workers);
        {
            const std::lock_guard lock(terminalMutex_);
            if (armedTerminal_ != nullptr ||
                (activeTerminal_ != nullptr && !activeTerminal_->isTerminal())) {
                return NIXL_ERR_NOT_ALLOWED;
            }
            activeTerminal_ = nullptr;
            armedTerminal_ = arm;
        }
        subscription = std::make_unique<nixlUcxTerminalSubscription>(arm);
        return NIXL_SUCCESS;
    }

    [[nodiscard]] std::shared_ptr<nixlUcxTerminalArm>
    getActiveTerminal() const noexcept {
        const std::lock_guard lock(terminalMutex_);
        return activeTerminal_;
    }

    void
    setActiveTerminal(const std::shared_ptr<nixlUcxTerminalArm> &terminal) noexcept {
        const std::lock_guard lock(terminalMutex_);
        activeTerminal_ = terminal;
    }

    [[nodiscard]] bool
    hasAutonomousTerminal() const noexcept {
        return getActiveTerminal() != nullptr;
    }

    [[nodiscard]] bool
    terminalLifecycleDrained() const noexcept {
        const std::lock_guard lock(terminalMutex_);
        return armedTerminal_ == nullptr &&
            (activeTerminal_ == nullptr || activeTerminal_->isTerminal());
    }

    [[nodiscard]] nixl_status_t
    recordSegment(size_t index,
                  nixlUcxEp &ep,
                  const std::vector<nixl_xfer_attestation_transport_t> &endpoint_transports,
                  const std::vector<nixl_xfer_attestation_transport_t> &selected_transports,
                  const std::string &request_info) {
        return attestation_->recordSegment(index,
                                           workerId_,
                                           ep.getWorkerIdentity(),
                                           ep.getIdentity(),
                                           endpoint_transports,
                                           selected_transports,
                                           request_info);
    }

    [[nodiscard]] nixl_status_t
    recordFlush(nixlUcxEp &ep, nixl_status_t status) {
        return attestation_->recordFlush(
            workerId_, ep.getWorkerIdentity(), ep.getIdentity(), status);
    }

    [[nodiscard]] nixl_status_t
    finishSubmission() {
        return attestation_->finishSubmission();
    }

    [[nodiscard]] nixl_xfer_attestation_t
    queryAttestation() const {
        return attestation_->snapshot();
    }

    [[nodiscard]] nixl_status_t
    takeCompletionAttestation(nixl_xfer_attestation_t &attestation) {
        return attestation_->takeCompletion(attestation);
    }

    [[nodiscard]] const std::shared_ptr<nixlUcxAttestationState> &
    getAttestationState() const noexcept {
        return attestation_;
    }

    void
    reserve(size_t size) {
        requests_.reserve(size);
        NIXL_ASSERT(connections_.empty());
    }

    [[nodiscard]] nixl_status_t
    append(nixl_status_t status,
           nixlUcxReq req,
           const ucx_connection_ptr_t &conn,
           pending_kind_t kind = pending_kind_t::NOTIFICATION,
           uint64_t endpoint_identity = 0) {
        switch (status) {
        case NIXL_IN_PROG:
            if (req == nullptr) {
                if (kind != pending_kind_t::NOTIFICATION) {
                    attestation_->fail(attestation_->getGeneration(),
                                       NIXL_ERR_BACKEND,
                                       "UCX returned an empty in-progress request");
                }
                return NIXL_ERR_BACKEND;
            }
            requests_.push_back(
                {req, kind, endpoint_identity, attestation_->getGeneration()});
            connections_.insert(conn);
            break;
        case NIXL_SUCCESS:
            connections_.insert(conn);
            break;
        default:
            if (kind != pending_kind_t::NOTIFICATION) {
                attestation_->fail(
                    attestation_->getGeneration(), status, "UCX request submission failed");
            }
            release();
            return status;
        }
        return NIXL_SUCCESS;
    }

    [[nodiscard]] const std::set<ucx_connection_ptr_t> &
    getConnections() const noexcept {
        return connections_;
    }

    void
    trackConnection(const ucx_connection_ptr_t &connection) {
        connections_.insert(connection);
    }

    [[nodiscard]] virtual bool
    isComposite() const noexcept {
        return false;
    }

    virtual void
    release() {
        const bool has_transport_request =
            std::any_of(requests_.begin(), requests_.end(), [](const auto &pending) {
                return pending.kind != pending_kind_t::NOTIFICATION;
            });
        if (has_transport_request) {
            attestation_->fail(
                attestation_->getGeneration(), NIXL_ERR_CANCELED, "transfer request was released");
        }

        for (const auto &pending : requests_) {
            const nixl_status_t ret =
                nixl::ucx::ucsToNixlStatus(ucp_request_check_status(pending.request));
            if (ret == NIXL_IN_PROG) {
                worker_->reqCancel(pending.request);
            }
            worker_->reqRelease(pending.request);
        }
        requests_.clear();
        connections_.clear();
    }

    [[nodiscard]] virtual nixl_status_t
    status() {
        const std::shared_ptr<nixlUcxTerminalArm> terminal = getActiveTerminal();
        if (terminal != nullptr) {
            return terminal->status();
        }
        if (requests_.empty()) {
            connections_.clear();
            const nixl_xfer_attestation_t attestation = attestation_->snapshot();
            if (attestation.state == nixl_xfer_attestation_state_t::FAILED) {
                return attestation.status;
            }
            return NIXL_SUCCESS;
        }

        worker_->progressLoop();

        /* If last request is incomplete, return NIXL_IN_PROG early without
         * checking other requests */
        const pending_request_t &last = requests_.back();
        const nixl_status_t ret =
            nixl::ucx::ucsToNixlStatus(ucp_request_check_status(last.request));
        if (ret == NIXL_IN_PROG) {
            return NIXL_IN_PROG;
        }

        size_t incomplete_reqs = 0;
        nixl_status_t out_ret = NIXL_SUCCESS;
        for (const auto &pending : requests_) {
            const nixl_status_t ret =
                nixl::ucx::ucsToNixlStatus(ucp_request_check_status(pending.request));
            if (ret == NIXL_SUCCESS) [[likely]] {
                if (pending.kind == pending_kind_t::ENDPOINT_FLUSH) {
                    const nixl_status_t flush_status = attestation_->completeFlush(
                        pending.generation, pending.endpointIdentity);
                    if (flush_status != NIXL_SUCCESS && out_ret == NIXL_SUCCESS) {
                        out_ret = flush_status;
                    }
                }
                worker_->reqRelease(pending.request);
            } else if (ret == NIXL_IN_PROG) {
                if (out_ret == NIXL_SUCCESS) {
                    out_ret = NIXL_IN_PROG;
                }
                requests_[incomplete_reqs++] = pending;
            } else {
                if (pending.kind != pending_kind_t::NOTIFICATION) {
                    attestation_->fail(
                        pending.generation, ret, "UCX transfer request failed");
                }
                if (out_ret >= NIXL_SUCCESS) {
                    out_ret = checkConnection(ret);
                }
                worker_->reqRelease(pending.request);
            }
        }

        requests_.resize(incomplete_reqs);
        if (requests_.empty()) {
            connections_.clear();
        }
        const nixl_xfer_attestation_t attestation = attestation_->snapshot();
        if (attestation.state == nixl_xfer_attestation_state_t::FAILED) {
            return attestation.status;
        }
        return out_ret;
    }

    [[nodiscard]] nixlUcxWorker *
    getWorker() const noexcept {
        return worker_;
    }

    [[nodiscard]] size_t
    getWorkerId() const noexcept {
        return workerId_;
    }

    friend class nixlUcxEngine;
    friend class nixlUcxThreadPoolEngine;
    friend class nixlUcxChunkBackendReqH;
};

/****************************************
 * Progress thread management
*****************************************/

/*
 * This class encapsulates a thread that polls one or multiple UCX workers
 */
class nixlUcxThread {
public:
    nixlUcxThread(const nixlUcxEngine *engine, size_t num_workers) : engine_(engine) {
        workers_.reserve(num_workers);
    }

    virtual ~nixlUcxThread() {
        if (threadActive_) {
            join();
        }
    }

    void
    start() {
        NIXL_ASSERT(!threadActive_);
        threadActive_ = std::make_unique<std::promise<void>>();
        auto active = threadActive_->get_future();
        thread_ = std::make_unique<std::thread>(std::ref(*this));
        active.wait();
    }

    virtual void
    join() {
        NIXL_ASSERT(threadActive_);
        threadActive_.reset();
        thread_->join();
    }

    virtual void
    addWorker(nixlUcxWorker *worker, size_t worker_id) {
        NIXL_ASSERT(workers_.size() < workers_.capacity());
        workers_.push_back(worker);
        workerIds_.push_back(worker_id);
    }

    const std::vector<nixlUcxWorker *> &
    getWorkers() const {
        return workers_;
    }

    size_t
    getWorkerId(size_t idx = 0) const {
        return workerIds_[idx];
    }

    void
    operator()() {
        tlsThread() = this;
        for (nixlUcxWorker *worker : workers_) {
            NIXL_ASSERT(worker->claimProgressOwner() == NIXL_SUCCESS);
        }
        threadActive_->set_value();
        run();
    }

    static nixlUcxThread *&
    tlsThread() {
        static thread_local nixlUcxThread *tls = nullptr;
        return tls;
    }

    static bool
    isProgressThread(const nixlUcxEngine *engine) noexcept {
        nixlUcxThread *thread = tlsThread();
        return thread && thread->engine_ == engine;
    }

    [[nodiscard]] bool
    terminalLifecycleDrained() const noexcept {
        return std::all_of(
            workers_.begin(), workers_.end(), [](const nixlUcxWorker *worker) {
                return worker->terminalLifecycleDrained();
            });
    }

    friend std::ostream &
    operator<<(std::ostream &os, const nixlUcxThread &thread) {
        return os << "thread " << &thread << "{engine: " << thread.engine_ << ", worker_ids: ["
                  << absl::StrJoin(thread.workerIds_, ",") << "]}";
    }

protected:
    virtual void
    run() = 0;

private:
    const nixlUcxEngine *engine_;
    std::vector<nixlUcxWorker *> workers_;
    std::vector<size_t> workerIds_;
    std::unique_ptr<std::thread> thread_;
    std::unique_ptr<std::promise<void>> threadActive_;
};

class nixlUcxSharedThread : public nixlUcxThread {
public:
    nixlUcxSharedThread(const nixlUcxEngine *engine, size_t num_workers, nixlTime::us_t delay)
        : nixlUcxThread(engine, num_workers) {
        if (pipe(controlPipe_) < 0) {
            throw std::runtime_error("Couldn't create progress thread control pipe");
        }
        // TODO: We need delay to manual periodic wakeup/polling as a temporary
        // workaround for UCX bug (poll wouldn't wake up some fds in particular
        // circumstances)

        // This will ensure that the resulting delay is at least 1ms and fits into int in order for
        // it to be compatible with poll()
        int delay_us = std::min((int)delay, std::numeric_limits<int>::max());
        delay_ = std::chrono::ceil<std::chrono::milliseconds>(std::chrono::microseconds(delay_us));

        pollFds_.resize(num_workers + 1);
        pollFds_.back() = {controlPipe_[0], POLLIN, 0};
    }

    ~nixlUcxSharedThread() {
        close(controlPipe_[0]);
        close(controlPipe_[1]);
    }

    void
    join() override {
        const char signal = 'X';
        int ret = write(controlPipe_[1], &signal, sizeof(signal));
        if (ret < 0) NIXL_PERROR << "write to progress thread control pipe failed";
        nixlUcxThread::join();
    }

    void
    addWorker(nixlUcxWorker *worker, size_t worker_id) override {
        pollFds_[getWorkers().size()] = {worker->getEfd(), POLLIN, 0};
        nixlUcxThread::addWorker(worker, worker_id);
    }

protected:
    void
    run() override {
        NIXL_DEBUG << "shared " << *this << " running";
        // Set timeout event so that the main loop would progress all workers on first iteration
        bool timeout = true;
        bool pthr_stop = false;
        while (!pthr_stop) {
            for (size_t i = 0; i < pollFds_.size() - 1; i++) {
                if (!(pollFds_[i].revents & POLLIN) && !timeout) continue;
                pollFds_[i].revents = 0;
                nixlUcxWorker *worker = getWorkers()[i];
                do {
                    worker->progressLoop();
                    NIXL_ASSERT(worker->drainContinuationsOnOwner() == NIXL_SUCCESS);
                } while (worker->arm() == NIXL_IN_PROG);
            }
            timeout = false;

            int ret;
            while ((ret = poll(pollFds_.data(), pollFds_.size(), delay_.count())) < 0)
                NIXL_PTRACE << "Call to poll() was interrupted, retrying";

            if (!ret) {
                timeout = true;
            } else if (pollFds_.back().revents & POLLIN) {
                pollFds_.back().revents = 0;

                char signal;
                int ret = read(pollFds_.back().fd, &signal, sizeof(signal));
                if (ret < 0) NIXL_PERROR << "read() on control pipe failed";

                pthr_stop = true;
            }
        }

        NIXL_DEBUG << "shared " << *this << " exiting";
    }

private:
    std::chrono::milliseconds delay_;
    int controlPipe_[2];
    std::vector<pollfd> pollFds_;
};

nixlUcxThreadEngine::nixlUcxThreadEngine(const nixlBackendInitParams &init_params)
    : nixlUcxEngine(init_params) {
    if (!nixlUcxMtLevelIsSupported(nixl::ucx::mt_mode_t::WORKER)) {
        throw std::invalid_argument("UCX library does not support multi-threading");
    }

    size_t num_workers = getWorkers().size();
    thread_ = std::make_unique<nixlUcxSharedThread>(this, num_workers, init_params.pthrDelay);
    for (size_t i = 0; i < num_workers; i++) {
        thread_->addWorker(getWorkers()[i].get(), i);
    }
    thread_->start();
}

nixlUcxThreadEngine::~nixlUcxThreadEngine() {
    if (!thread_->terminalLifecycleDrained()) {
        NIXL_FATAL << "UCX shared progress owner stopped with live terminal callback state";
        std::terminate();
    }
    thread_->join();
}

void
nixlUcxThreadEngine::appendNotif(nixlAuthenticatedNotification &&notification) const {
    const std::lock_guard lock(notifMutex_);
    (void)notifQueue_.push(std::move(notification));
}

void
nixlUcxThreadEngine::poisonNotifs() const {
    const std::lock_guard lock(notifMutex_);
    notifQueue_.poison();
}

nixl_status_t
nixlUcxThreadEngine::getNotifs(notif_list_t &notif_list) {
    const std::lock_guard lock(notifMutex_);
    return notifQueue_.drainLegacy(notif_list);
}

nixl_status_t
nixlUcxThreadEngine::getAuthenticatedNotifs(authenticated_notif_list_t &notif_list) {
    const std::lock_guard lock(notifMutex_);
    return notifQueue_.drainAuthenticated(notif_list);
}

/****************************************
 * Threadpool engine
 ****************************************/

struct nixlUcxBackendSharedState;

/*
 * This class represents a chunk of a composite request.
 * It is used to encapsulate a batch of requests (subset of the larger batch)
 * performed by a dedicated worker thread of threadpool. It holds a shared state
 * with the main request to track its completion status and control the lifetime.
 */
class nixlUcxChunkBackendReqH : public nixlUcxBackendReqH {
public:
    nixlUcxChunkBackendReqH() : nixlUcxBackendReqH(nullptr, UINT64_MAX) {}

    void
    startXfer(const std::shared_ptr<nixlUcxBackendSharedState> &shared_state,
              nixlUcxWorker *worker,
              size_t worker_id,
              const std::shared_ptr<nixlUcxAttestationState> &attestation,
              const std::shared_ptr<nixlUcxTerminalArm> &terminal) {
        NIXL_ASSERT(sharedState_.get() == nullptr);
        sharedState_ = shared_state;
        setWorker(worker, worker_id);
        setAttestationState(attestation);
        setActiveTerminal(terminal);
    }

    void
    complete(nixl_status_t status);

    [[nodiscard]] nixl_status_t
    status() override;

    friend std::ostream &
    operator<<(std::ostream &os, const nixlUcxChunkBackendReqH &chunk) {
        return os << "chunk " << &chunk << "{worker_id: " << chunk.getWorkerId()
                  << ", state: " << chunk.sharedState_.get() << "}";
    }

private:
    std::shared_ptr<nixlUcxBackendSharedState> sharedState_;
};

/*
 * This class represents a shared state between a main request and all of its
 * chunks. It is used to track the completion status of the request and the
 * number of pending requests, and to control the lifetime of the chunks.
 */
struct nixlUcxBackendSharedState {
    std::atomic<nixl_status_t> status;
    std::atomic<size_t> pendingReqs;
    std::vector<std::unique_ptr<nixlUcxChunkBackendReqH>> chunks;

    nixlUcxBackendSharedState() : status(NIXL_SUCCESS), pendingReqs(0) {}

    friend std::ostream &
    operator<<(std::ostream &os, const nixlUcxBackendSharedState &state) {
        return os << "state " << &state << "{status: " << state.status.load()
                  << ", pending=" << state.pendingReqs.load() << "}";
    }
};

void
nixlUcxChunkBackendReqH::complete(const nixl_status_t status) {
    NIXL_ASSERT(sharedState_.get() != nullptr);
    if (status != NIXL_SUCCESS) {
        nixlUcxBackendReqH::release();
        sharedState_->status.store(status);
    }
    sharedState_->pendingReqs.fetch_sub(1);
    NIXL_TRACE << *this << " completed with status: " << status << ", " << *sharedState_;
    setWorker(nullptr, UINT64_MAX);
    sharedState_.reset();
}

nixl_status_t
nixlUcxChunkBackendReqH::status() {
    // First check if entire request was cancelled or failed
    const nixl_status_t status = sharedState_->status.load();
    if (status != NIXL_SUCCESS) {
        return status;
    }
    return nixlUcxBackendReqH::status();
}

/*
 * This class represents a composite request handle for a UCX backend.
 * It is used to encapsulate multiple parallel requests performed by dedicated
 * worker threads of threadpool, with a single request handle, that it returned
 * to the user.
 */
class nixlUcxCompositeBackendReqH : public nixlUcxBackendReqH {
public:
    nixlUcxCompositeBackendReqH(nixlUcxWorker *worker,
                                size_t worker_id,
                                size_t chunk_size,
                                size_t num_chunks)
        : nixlUcxBackendReqH(worker, worker_id),
          sharedState_(std::make_shared<nixlUcxBackendSharedState>()),
          chunkSize_(chunk_size) {
        sharedState_->chunks.reserve(num_chunks);
        for (size_t index = 0; index < num_chunks; ++index) {
            sharedState_->chunks.push_back(
                std::make_unique<nixlUcxChunkBackendReqH>());
        }
    }

    [[nodiscard]] size_t
    getChunkSize() const noexcept {
        return chunkSize_;
    }

    [[nodiscard]] size_t
    getNumChunks() const noexcept {
        return sharedState_ ? sharedState_->chunks.size() : 0;
    }

    void
    startXfer() {
        NIXL_ASSERT(sharedState_->pendingReqs.load() == 0);
        sharedState_->status.store(NIXL_SUCCESS);
        sharedState_->pendingReqs.store(getNumChunks());
    }

    [[nodiscard]] nixlUcxChunkBackendReqH *
    startChunk(size_t idx, nixlUcxWorker *worker, size_t worker_id) {
        nixlUcxChunkBackendReqH *chunk = sharedState_->chunks[idx].get();
        chunk->startXfer(
            sharedState_, worker, worker_id, getAttestationState(), getActiveTerminal());
        return chunk;
    }

    [[nodiscard]] bool
    isComposite() const noexcept override {
        return true;
    }

    void
    release() override {
        NIXL_TRACE << *this << " releasing";
        nixlUcxBackendReqH::release();
        if (sharedState_) {
            // Set failed status to stop progress chunks
            sharedState_->status.store(NIXL_ERR_NOT_FOUND);
            // Reset shared state - it will be effectively released when the last chunk
            // resets the shared state pointer
            sharedState_.reset();
        }
    }

    [[nodiscard]] nixl_status_t
    status() override {
        getWorker()->progressLoop();

        if (sharedState_->pendingReqs.load()) {
            return NIXL_IN_PROG;
        }

        const nixl_status_t status = nixlUcxBackendReqH::status();
        if (status != NIXL_SUCCESS) {
            return status;
        }

        return sharedState_->status.load();
    }

    friend std::ostream &
    operator<<(std::ostream &os, const nixlUcxCompositeBackendReqH &handle) {
        os << "composite handle " << &handle << "{chunks: " << handle.getNumChunks();
        if (handle.sharedState_) {
            os << ", " << *handle.sharedState_;
        } else {
            os << ", state: nullptr";
        }
        return os << "}}";
    }

private:
    std::shared_ptr<nixlUcxBackendSharedState> sharedState_;
    size_t chunkSize_;
};

class nixlUcxDedicatedThread : public nixlUcxThread {
public:
    nixlUcxDedicatedThread(nixlUcxEngine *engine, asio::io_context &io)
        : nixlUcxThread(engine, 1),
          io_(io) {}

    static nixlUcxDedicatedThread *
    getDedicatedThread() {
        return (nixlUcxDedicatedThread *)tlsThread();
    }

    void
    addRequest(nixlUcxChunkBackendReqH *handle) {
        requests_.push_back(handle);
    }

protected:
    void
    run() override {
        const auto guard = asio::make_work_guard(io_);
        NIXL_DEBUG << "dedicated " << *this << " running";

        while (!io_.stopped()) {
            nixlUcxWorker *const worker = getWorkers()[0];
            if (!requests_.empty() || worker->activeTerminalCallbackCount() != 0) {
                io_.poll_one();
            } else {
                NIXL_TRACE << "dedicated " << *this << " waiting for requests";
                io_.run_one();
            }

            worker->progressLoop();
            NIXL_ASSERT(worker->drainContinuationsOnOwner() == NIXL_SUCCESS);
            if (requests_.empty()) {
                continue;
            }

            for (auto it = requests_.begin(); it != requests_.end();) {
                nixl_status_t status = (*it)->status();
                if (status != NIXL_IN_PROG) {
                    NIXL_TRACE << "dedicated " << *this << " completing " << *(*it)
                               << " with status: " << status;
                    (*it)->complete(status);
                    it = requests_.erase(it);
                } else {
                    ++it;
                }
            }
        }

        if (!requests_.empty()) {
            NIXL_WARN << "dedicated " << *this << " dropping " << requests_.size()
                      << " requests on exit";
            for (auto it = requests_.begin(); it != requests_.end();) {
                NIXL_INFO << "dropping " << *(*it);
                (*it)->complete(NIXL_ERR_BACKEND);
            }
            requests_.clear();
        }

        NIXL_DEBUG << "dedicated " << *this << " exiting";
    }

private:
    asio::io_context &io_;
    std::vector<nixlUcxChunkBackendReqH *> requests_;
};

nixlUcxThreadPoolEngine::nixlUcxThreadPoolEngine(const nixlBackendInitParams &init_params)
    : nixlUcxEngine(init_params) {
    size_t num_threads = nixl_b_params_get(init_params.customParams, "num_threads", 0);
    numSharedWorkers_ = getWorkers().size() - num_threads;
    NIXL_ASSERT(numSharedWorkers_ > 0);

    splitBatchSize_ = nixl_b_params_get(init_params.customParams, "split_batch_size", 1024);

    if (init_params.enableProgTh) {
        sharedThread_ =
            std::make_unique<nixlUcxSharedThread>(this, numSharedWorkers_, init_params.pthrDelay);
        for (size_t i = 0; i < numSharedWorkers_; i++) {
            sharedThread_->addWorker(getWorkers()[i].get(), i);
        }
        sharedThread_->start();
    }

    if (num_threads > 0) {
        io_.reset(new asio::io_context());
        dedicatedThreads_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            size_t worker_id = numSharedWorkers_ + i;
            dedicatedThreads_.emplace_back(std::make_unique<nixlUcxDedicatedThread>(this, *io_));
            dedicatedThreads_.back()->addWorker(getWorker(worker_id).get(), worker_id);
            dedicatedThreads_.back()->start();
        }
    }
}

nixlUcxThreadPoolEngine::~nixlUcxThreadPoolEngine() {
    if (sharedThread_ && !sharedThread_->terminalLifecycleDrained()) {
        NIXL_FATAL << "UCX shared progress owner stopped with live terminal callback state";
        std::terminate();
    }
    for (const auto &thread : dedicatedThreads_) {
        if (!thread->terminalLifecycleDrained()) {
            NIXL_FATAL << "UCX dedicated progress owner stopped with live terminal callback state";
            std::terminate();
        }
    }
    if (sharedThread_) {
        sharedThread_->join();
    }

    if (io_) {
        io_->stop();
        for (auto &thread : dedicatedThreads_) {
            thread->join();
        }
    }
}

nixl_status_t
nixlUcxThreadPoolEngine::prepXfer(const nixl_xfer_op_t &operation,
                                  const nixl_meta_dlist_t &local,
                                  const nixl_meta_dlist_t &remote,
                                  const std::string &remote_agent,
                                  nixlBackendReqH *&handle,
                                  const nixl_opt_b_args_t *opt_args) const {
    size_t batch_size = local.descCount();
    if (batch_size < splitBatchSize_) {
        return nixlUcxEngine::prepXfer(operation, local, remote, remote_agent, handle, opt_args);
    }

    size_t chunk_size = std::max(batch_size / dedicatedThreads_.size(), splitBatchSize_);
    size_t num_chunks = (batch_size + chunk_size - 1) / chunk_size;

    size_t worker_id = getWorkerId();
    const auto comp_handle = new nixlUcxCompositeBackendReqH(
        getWorker(worker_id).get(), worker_id, chunk_size, num_chunks);
    const nixl_status_t status =
        prepareHandleAttestation(
            comp_handle, operation, local, remote, remote_agent, opt_args);
    if (status != NIXL_SUCCESS) {
        delete comp_handle;
        return status;
    }
    NIXL_TRACE << "created " << *comp_handle;
    handle = comp_handle;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxThreadPoolEngine::sendXferRange(const nixl_xfer_op_t &operation,
                                       const nixl_meta_dlist_t &local,
                                       const nixl_meta_dlist_t &remote,
                                       const std::string &remote_agent,
                                       nixlBackendReqH *handle,
                                       size_t start_idx,
                                       size_t end_idx) const {
    const auto int_handle = static_cast<nixlUcxBackendReqH *>(handle);
    if (!int_handle->isComposite()) {
        return nixlUcxEngine::sendXferRange(
            operation, local, remote, remote_agent, handle, start_idx, end_idx);
    }

    const auto comp_handle = static_cast<nixlUcxCompositeBackendReqH *>(int_handle);
    comp_handle->startXfer();
    size_t chunk_size = comp_handle->getChunkSize();
    NIXL_TRACE << "sending " << *comp_handle;

    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    std::atomic<size_t> remaining{comp_handle->getNumChunks()};
    std::atomic<nixl_status_t> status{NIXL_SUCCESS};

    for (size_t i = 0; i < comp_handle->getNumChunks(); i++) {
        io_->post([&, i]() {
            nixlUcxDedicatedThread *thread = nixlUcxDedicatedThread::getDedicatedThread();
            NIXL_ASSERT(thread != nullptr);

            nixlUcxChunkBackendReqH *chunk_handle =
                comp_handle->startChunk(i, thread->getWorkers()[0], thread->getWorkerId());
            NIXL_TRACE << "dedicated " << *thread << " starting " << *chunk_handle;

            size_t start_idx = i * chunk_size;
            size_t end_idx = std::min(start_idx + chunk_size, (size_t)local.descCount());
            nixl_status_t ret = nixlUcxEngine::sendXferRange(
                operation, local, remote, remote_agent, chunk_handle, start_idx, end_idx);
            if (ret != NIXL_SUCCESS) {
                status.store(ret);
                chunk_handle->complete(ret);
            } else if (chunk_handle->hasAutonomousTerminal()) {
                chunk_handle->complete(NIXL_SUCCESS);
            } else {
                NIXL_TRACE << "dedicated " << *thread << " sent " << *chunk_handle;
                thread->addRequest(chunk_handle);
            }

            if (remaining.fetch_sub(1) == 1) {
                promise.set_value();
            }
        });
    }

    future.wait();
    NIXL_TRACE << "sent " << *comp_handle << " with status: " << status.load();
    return status.load();
}

void
nixlUcxThreadPoolEngine::appendNotif(nixlAuthenticatedNotification &&notification) const {
    const std::lock_guard lock(notifMutex_);
    (void)notifQueue_.push(std::move(notification));
}

void
nixlUcxThreadPoolEngine::poisonNotifs() const {
    const std::lock_guard lock(notifMutex_);
    notifQueue_.poison();
}

nixl_status_t
nixlUcxThreadPoolEngine::getNotifs(notif_list_t &notif_list) {
    if (!sharedThread_) {
        progressLoop();
    }

    const std::lock_guard lock(notifMutex_);
    return notifQueue_.drainLegacy(notif_list);
}

nixl_status_t
nixlUcxThreadPoolEngine::getAuthenticatedNotifs(
    authenticated_notif_list_t &notif_list) {
    if (!sharedThread_) {
        progressLoop();
    }

    const std::lock_guard lock(notifMutex_);
    return notifQueue_.drainAuthenticated(notif_list);
}

/****************************************
 * Constructor/Destructor
 *****************************************/

std::unique_ptr<nixlUcxEngine>
nixlUcxEngine::create(const nixlBackendInitParams &init_params) {
    nixlUcxEngine *engine;
    size_t num_threads = nixl_b_params_get(init_params.customParams, "num_threads", 0);
    if (num_threads > 0) {
        engine = new nixlUcxThreadPoolEngine(init_params);
    } else if (init_params.enableProgTh) {
        engine = new nixlUcxThreadEngine(init_params);
    } else {
        engine = new nixlUcxEngine(init_params);
    }
    return std::unique_ptr<nixlUcxEngine>(engine);
}

nixlUcxEngine::nixlUcxEngine(const nixlBackendInitParams &init_params)
    : nixlBackendEngine(&init_params),
      sharedWorkerIndex_(1) {
    std::vector<std::string> devs; /* Empty vector */
    nixl_b_params_t *custom_params = init_params.customParams;

    if (custom_params->count("device_list")!=0)
        devs = absl::StrSplit((*custom_params)["device_list"], ", ");

    size_t num_workers = nixl_b_params_get(custom_params, "num_workers", 1);
    size_t num_threads = nixl_b_params_get(custom_params, "num_threads", 0);
    size_t num_device_channels = nixl_b_params_get(custom_params, "ucx_num_device_channels", 4);

    if (num_workers <= num_threads) {
        /* There must be at least one shared worker */
        num_workers = num_threads + 1;
    }

    ucp_err_handling_mode_t err_handling_mode;
    const auto err_handling_mode_it =
        custom_params->find(std::string(nixl_ucx_err_handling_param_name));
    if (err_handling_mode_it == custom_params->end()) {
        err_handling_mode = UCP_ERR_HANDLING_MODE_PEER;
    } else {
        err_handling_mode = ucx_err_mode_from_string(err_handling_mode_it->second);
    }

    const auto engine_config_it = custom_params->find("engine_config");
    const auto engine_config =
        (engine_config_it != custom_params->end()) ? engine_config_it->second : "";

    uc = std::make_unique<nixlUcxContext>(devs,
                                          init_params.enableProgTh,
                                          num_workers,
                                          init_params.syncMode,
                                          num_device_channels,
                                          engine_config);

    uc->warnAboutHardwareSupportMismatch();

    for (size_t i = 0; i < num_workers; i++) {
        uws.emplace_back(std::make_unique<nixlUcxWorker>(*uc, err_handling_mode));
    }

    localConnectionMetadata_.backendIncarnation =
        nixl::ucx::generateConnectionMetadataUuid();
    localConnectionMetadata_.workers.reserve(uws.size());
    for (const auto &worker : uws) {
        localConnectionMetadata_.workers.push_back({
            .incarnation = nixl::ucx::generateConnectionMetadataUuid(),
            .endpointAddress = worker->epAddr(),
        });
    }

    if (!parseWireUuid(localAgentIncarnation, localAgentIncarnationUuid_)) {
        throw std::invalid_argument("invalid local agent incarnation");
    }
    std::vector<nixl::ucx::notif_wire_uuid_t> local_worker_incarnations;
    local_worker_incarnations.reserve(localConnectionMetadata_.workers.size());
    for (const auto &worker : localConnectionMetadata_.workers) {
        local_worker_incarnations.push_back(worker.incarnation);
    }
    notifState_ = std::make_unique<nixl::ucx::notif_capability_state_t>(
        localAgentIncarnationUuid_,
        localConnectionMetadata_.backendIncarnation,
        std::move(local_worker_incarnations));

    notifCallbackContexts_.reserve(uws.size());
    for (size_t worker_id = 0; worker_id < uws.size(); ++worker_id) {
        notifCallbackContexts_.push_back({.engine = this, .workerId = worker_id});
        uws[worker_id]->regAmCallback(nixl::ucx::am_cb_op_t::NOTIF_STR,
                                     notifAmCb,
                                     &notifCallbackContexts_.back());
    }
}

nixl_mem_list_t nixlUcxEngine::getSupportedMems () const {
    nixl_mem_list_t mems;
    mems.push_back(DRAM_SEG);
    mems.push_back(VRAM_SEG);
    return mems;
}

static std::unordered_map<const nixlUcxEngine *, size_t> &
tlsSharedWorkerMap() {
    static thread_local std::unordered_map<const nixlUcxEngine *, size_t> map;
    return map;
}

// Through parent destructor the unregister will be called.
nixlUcxEngine::~nixlUcxEngine() {
    tlsSharedWorkerMap().erase(this);
    const std::lock_guard lock(connectionMutex_);
    remoteConnMap.clear();
}

/****************************************
 * Connection management
*****************************************/

nixl_status_t nixlUcxEngine::checkConn(const std::string &remote_agent) {
    return getConnection(remote_agent) != nullptr ? NIXL_SUCCESS : NIXL_ERR_NOT_FOUND;
}

nixl_status_t nixlUcxEngine::getConnInfo(std::string &str) const {
    return nixl::ucx::encodeConnectionMetadata(localConnectionMetadata_, str) ==
            nixl::ucx::connection_metadata_status_t::SUCCESS ?
        NIXL_SUCCESS : NIXL_ERR_BACKEND;
}

nixl_status_t nixlUcxEngine::connect(const std::string &remote_agent) {
    if(remote_agent == localAgent) {
        std::string local_conn_info;
        const nixl_status_t status = getConnInfo(local_conn_info);
        if (status != NIXL_SUCCESS) {
            return status;
        }
        return loadRemoteConnInfo(remote_agent, local_conn_info);
    }

    return getConnection(remote_agent) == nullptr ? NIXL_ERR_NOT_FOUND : NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::disconnect(const std::string &remote_agent) {
    const std::lock_guard lock(connectionMutex_);
    const auto it = remoteConnMap.find(remote_agent);

    if (it == remoteConnMap.end()) {
        return NIXL_ERR_NOT_FOUND;
    }

    remoteConnMap.erase(it);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxEngine::queryRemoteAgentAuthority(
    const std::string &remote_agent,
    nixl_remote_agent_authority_t &authority) const {
    const ucx_connection_ptr_t connection = getConnection(remote_agent);
    if (connection == nullptr) {
        return NIXL_ERR_NOT_FOUND;
    }

    authority = {};
    authority.connectionIdentity = connection->identity_;
    authority.endpointIdentities.reserve(connection->eps.size());
    for (const auto &ep : connection->eps) {
        authority.endpointIdentities.push_back(ep->getIdentity());
    }
    std::sort(authority.endpointIdentities.begin(), authority.endpointIdentities.end());
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxEngine::makeRoute(const nixlRemoteAgentBinding &binding,
                         nixl::ucx::notif_route_key_t &route,
                         ucx_connection_ptr_t &connection) const {
    if (binding.remoteAgent.empty() || binding.authority.handleIdentity == 0 ||
        binding.authority.generation == 0 ||
        binding.authority.connectionIdentity == 0 ||
        binding.authority.endpointIdentities.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    connection = getConnection(binding.remoteAgent);
    if (connection == nullptr ||
        connection->getIdentity() != binding.authority.connectionIdentity) {
        return NIXL_ERR_NOT_FOUND;
    }

    std::vector<uint64_t> endpoint_identities;
    endpoint_identities.reserve(connection->eps.size());
    for (const auto &endpoint : connection->eps) {
        endpoint_identities.push_back(endpoint->getIdentity());
    }
    std::sort(endpoint_identities.begin(), endpoint_identities.end());
    if (endpoint_identities != binding.authority.endpointIdentities) {
        return NIXL_ERR_NOT_ALLOWED;
    }

    nixl::ucx::notif_wire_uuid_t remote_agent_incarnation;
    if (!parseWireUuid(binding.agentIncarnation, remote_agent_incarnation)) {
        return NIXL_ERR_INVALID_PARAM;
    }
    route = {
        .handleIdentity = binding.authority.handleIdentity,
        .handleGeneration = binding.authority.generation,
        .remoteAgentIncarnation = remote_agent_incarnation,
        .remoteBackendIncarnation = connection->metadata_.backendIncarnation,
    };
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxEngine::bindRemoteAgent(const nixlRemoteAgentBinding &binding) {
    nixl::ucx::notif_route_key_t route;
    ucx_connection_ptr_t connection;
    const nixl_status_t route_status = makeRoute(binding, route, connection);
    if (route_status != NIXL_SUCCESS) {
        return route_status;
    }

    bool inserted_route = false;
    {
        const std::lock_guard lock(exactRouteMutex_);
        const auto known = exactRoutes_.find(route.handleIdentity);
        if (known != exactRoutes_.end()) {
            if (known->second.route != route ||
                known->second.remoteAgent != binding.remoteAgent ||
                known->second.connectionIdentity != connection->getIdentity()) {
                return NIXL_ERR_NOT_ALLOWED;
            }
        } else {
            exactRoutes_.emplace(route.handleIdentity,
                                 exactRouteRecord{
                                     .route = route,
                                     .remoteAgent = binding.remoteAgent,
                                     .connectionIdentity = connection->getIdentity(),
                                 });
            inserted_route = true;
        }
    }

    nixl::ucx::notif_remote_binding_t remote_binding = {
        .route = route,
        .connectionIdentity = connection->getIdentity(),
    };
    remote_binding.workers.reserve(connection->metadata_.workers.size());
    for (size_t worker_index = 0;
         worker_index < connection->metadata_.workers.size();
         ++worker_index) {
        remote_binding.workers.push_back({
            .incarnation = connection->metadata_.workers[worker_index].incarnation,
            .endpointIdentity =
                connection->eps[worker_index % connection->eps.size()]->getIdentity(),
        });
    }

    nixl::ucx::notif_route_snapshot_t snapshot;
    const nixl::ucx::notif_state_status_t bind_status =
        notifState_->bindRemoteAgent(remote_binding, snapshot);
    if (bind_status != nixl::ucx::notif_state_status_t::SUCCESS) {
        if (inserted_route) {
            const std::lock_guard lock(exactRouteMutex_);
            exactRoutes_.erase(route.handleIdentity);
        }
        return notifStateToNixl(bind_status);
    }

    nixl::ucx::notif_wire_envelope_t offer;
    const auto &worker = localConnectionMetadata_.workers.front().incarnation;
    if (notifState_->makeOffer(route, worker, offer) ==
        nixl::ucx::notif_state_status_t::SUCCESS) {
        const nixl_status_t send_status =
            sendControlFrame(offer, connection->getIdentity(), 0);
        if (send_status != NIXL_SUCCESS) {
            NIXL_WARN << "UCX notification OFFER will be retried for handle "
                      << route.handleIdentity << ": " << send_status;
        }
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxEngine::retireRemoteAgent(const nixlRemoteAgentBinding &binding) {
    nixl::ucx::notif_route_key_t route;
    ucx_connection_ptr_t connection;
    const nixl_status_t route_status = makeRoute(binding, route, connection);
    if (route_status != NIXL_SUCCESS) {
        return route_status;
    }
    const std::optional<exactRouteRecord> record =
        getExactRoute(route.handleIdentity, route.handleGeneration);
    if (!record.has_value() || record->route != route) {
        return NIXL_ERR_NOT_FOUND;
    }
    return notifStateToNixl(notifState_->retireRemoteAgent(route));
}

nixl_status_t
nixlUcxEngine::queryRemoteNotificationState(
    const nixlRemoteAgentBinding &binding) const {
    nixl::ucx::notif_route_key_t route;
    ucx_connection_ptr_t connection;
    const nixl_status_t route_status = makeRoute(binding, route, connection);
    if (route_status != NIXL_SUCCESS) {
        return route_status;
    }

    nixl::ucx::notif_route_snapshot_t snapshot;
    const nixl::ucx::notif_state_status_t query_status =
        notifState_->queryRemoteNotificationState(route, snapshot);
    if (query_status != nixl::ucx::notif_state_status_t::SUCCESS) {
        return notifStateToNixl(query_status);
    }
    switch (snapshot.state) {
    case nixl::ucx::notif_route_state_t::READY:
        return NIXL_SUCCESS;
    case nixl::ucx::notif_route_state_t::NOT_READY: {
        nixl::ucx::notif_wire_envelope_t offer;
        const auto &worker = localConnectionMetadata_.workers.front().incarnation;
        if (notifState_->makeOffer(route, worker, offer) ==
            nixl::ucx::notif_state_status_t::SUCCESS) {
            (void)sendControlFrame(offer, connection->getIdentity(), 0);
        }
        return NIXL_ERR_NOT_READY;
    }
    case nixl::ucx::notif_route_state_t::RETIRED:
    case nixl::ucx::notif_route_state_t::FAILED:
        return NIXL_ERR_NOT_ALLOWED;
    }
    return NIXL_ERR_UNKNOWN;
}

nixl_status_t
nixlUcxEngine::subscribeRemoteNotificationState(
    const nixlRemoteAgentBinding &binding,
    const std::shared_ptr<nixlBackendCapabilityTransitionSink> &sink,
    std::unique_ptr<nixlBackendEventSubscription> &subscription) {
    subscription.reset();
    if (sink == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    for (const std::unique_ptr<nixlUcxWorker> &worker : getWorkers()) {
        if (!worker->hasProgressOwner()) {
            return NIXL_ERR_NOT_SUPPORTED;
        }
    }
    nixl::ucx::notif_route_key_t route;
    ucx_connection_ptr_t connection;
    const nixl_status_t route_status = makeRoute(binding, route, connection);
    if (route_status != NIXL_SUCCESS) {
        return route_status;
    }
    const auto adapter = std::make_shared<nixlUcxCapabilitySink>(sink);
    nixl::ucx::notif_route_subscription_t native_subscription;
    const nixl_status_t subscribe_status = notifSubscriptionToNixl(
        notifState_->subscribeRemoteNotificationState(
            route, adapter, native_subscription));
    if (subscribe_status != NIXL_SUCCESS) {
        return subscribe_status;
    }
    subscription = std::make_unique<nixlUcxCapabilitySubscription>(
        *notifState_, native_subscription, adapter);
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::loadRemoteConnInfo (const std::string &remote_agent,
                                                 const std::string &remote_conn_info)
{
    {
        const std::lock_guard lock(connectionMutex_);
        if (remoteConnMap.count(remote_agent) != 0) {
            return NIXL_ERR_INVALID_PARAM;
        }
    }

    nixl::ucx::connection_metadata_t remote_metadata;
    if (nixl::ucx::decodeConnectionMetadata(remote_conn_info, remote_metadata) !=
        nixl::ucx::connection_metadata_status_t::SUCCESS) {
        return NIXL_ERR_MISMATCH;
    }

    std::shared_ptr<nixlUcxConnection> conn = std::make_shared<nixlUcxConnection>(
        allocateIdentity(next_connection_identity, "connection"),
        std::move(remote_metadata));
    for (size_t worker_id = 0; worker_id < uws.size(); ++worker_id) {
        const auto &remote_worker =
            conn->metadata_.workers[worker_id % conn->metadata_.workers.size()];
        nixlUcxWorker *const worker = uws[worker_id].get();
        const std::weak_ptr<nixlUcxConnection> weak_connection = conn;
        std::unique_ptr<nixlUcxEp> result = uws[worker_id]->connect(
            remote_worker.endpointAddress.data(),
            remote_worker.endpointAddress.size(),
            [this, worker, weak_connection]() {
                const std::shared_ptr<nixlUcxConnection> connection =
                    weak_connection.lock();
                if (connection == nullptr || !connection->claimFailurePublication()) {
                    return;
                }
                const uint64_t connection_identity = connection->getIdentity();
                static_cast<void>(worker->enqueueContinuation(
                    [this, connection_identity]() {
                        failExactRoutesForConnection(connection_identity);
                        return NIXL_SUCCESS;
                    }));
            });
        if (!result) {
            return NIXL_ERR_BACKEND;
        }
        conn->eps.push_back(std::move(result));
    }
    {
        const std::lock_guard lock(connectionMutex_);
        if (!remoteConnMap.emplace(remote_agent, conn).second) {
            return NIXL_ERR_INVALID_PARAM;
        }
    }

    return NIXL_SUCCESS;
}

/****************************************
 * Memory management
*****************************************/
nixl_status_t nixlUcxEngine::registerMem (const nixlBlobDesc &mem,
                                          const nixl_mem_t &nixl_mem,
                                          nixlBackendMD* &out)
{
    auto priv = std::make_unique<nixlUcxPrivateMetadata>();

    // TODO: Add nixl_mem check?
    const int ret = uc->memReg((void*) mem.addr, mem.len, priv->mem, nixl_mem);
    if (ret) {
        return NIXL_ERR_BACKEND;
    }
    priv->rkeyStr = uc->packRkey(priv->mem);

    if (priv->rkeyStr.empty()) {
        return NIXL_ERR_BACKEND;
    }
    out = priv.release();
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::deregisterMem (nixlBackendMD* meta)
{
    nixlUcxPrivateMetadata *priv = (nixlUcxPrivateMetadata*) meta;
    uc->memDereg(priv->mem);
    delete priv;
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::getPublicData (const nixlBackendMD* meta,
                                            std::string &str) const {
    const nixlUcxPrivateMetadata *priv = (nixlUcxPrivateMetadata*) meta;
    str = priv->get();
    return NIXL_SUCCESS;
}

namespace {

[[nodiscard]] std::vector<nixl::ucx::rkey>
makePublicMetadataRkeys(const ucx_connection_ptr_t &conn, const size_t count, const void *buffer) {
    std::vector<nixl::ucx::rkey> result;
    result.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        result.emplace_back(*conn->getEp(i), buffer);
    }
    return result;
}

} // namespace

nixlUcxPublicMetadata::nixlUcxPublicMetadata(const ucx_connection_ptr_t &conn,
                                             std::vector<nixl::ucx::rkey> &&rkeys)
    : nixlBackendMD(false),
      conn(conn),
      rkeys_(std::move(rkeys)) {}

nixl_status_t
nixlUcxEngine::internalMDHelper (const nixl_blob_t &blob,
                                 const std::string &agent,
                                 nixlBackendMD* &output) {
    output = nullptr;
    try {
        const ucx_connection_ptr_t connection = getConnection(agent);
        if (connection == nullptr) {
            // TODO: err: remote connection not found
            return NIXL_ERR_NOT_FOUND;
        }
        // nixlSerDes::_stringToBytes() was used to "unpack" blob here.
        auto metadata = std::make_unique<nixlUcxPublicMetadata>(
            connection, makePublicMetadataRkeys(connection, uws.size(), blob.data()));
        output = metadata.release();
        return NIXL_SUCCESS;
    }
    catch (const std::runtime_error &e) {
        NIXL_ERROR << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlUcxEngine::loadLocalMD (nixlBackendMD* input,
                            nixlBackendMD* &output)
{
    nixlUcxPrivateMetadata* input_md = (nixlUcxPrivateMetadata*) input;
    return internalMDHelper(input_md->rkeyStr, localAgent, output);
}

// To be cleaned up
nixl_status_t nixlUcxEngine::loadRemoteMD (const nixlBlobDesc &input,
                                           const nixl_mem_t &nixl_mem,
                                           const std::string &remote_agent,
                                           nixlBackendMD* &output)
{
    return internalMDHelper(input.metaInfo, remote_agent, output);
}

nixl_status_t nixlUcxEngine::unloadMD (nixlBackendMD* input) {

    nixlUcxPublicMetadata *md = (nixlUcxPublicMetadata*) input; //typecast?
    delete md;

    return NIXL_SUCCESS;
}

/****************************************
 * Data movement
*****************************************/

size_t
nixlUcxEngine::getWorkerId(const nixl_opt_b_args_t *opt_args) const noexcept {
    if (opt_args) {
        const std::optional<size_t> worker_id = getWorkerIdFromOptArgs(*opt_args);
        if (worker_id) {
            return *worker_id;
        }
    }

    auto it = tlsSharedWorkerMap().find(this);
    if (it == tlsSharedWorkerMap().end()) {
        const size_t index = sharedWorkerIndex_.fetch_add(1) % getSharedWorkersSize();
        it = tlsSharedWorkerMap().emplace(this, index).first;
        NIXL_DEBUG << "engine " << this << " bound shared worker " << index << " to thread "
                   << std::this_thread::get_id();
    }
    return it->second;
}

std::optional<size_t>
nixlUcxEngine::getWorkerIdFromOptArgs(const nixl_opt_b_args_t &opt_args) const noexcept {
    constexpr std::string_view worker_id_key = "worker_id=";
    size_t pos = opt_args.customParam.find(worker_id_key);
    if (pos == std::string::npos) {
        return std::nullopt;
    }

    try {
        size_t worker_id = std::stoull(opt_args.customParam.substr(pos + worker_id_key.length()));

        if (worker_id >= getSharedWorkersSize()) {
            NIXL_WARN << "Invalid worker_id " << worker_id << " (must be < "
                      << getSharedWorkersSize() << ")";
            return std::nullopt;
        }

        return worker_id;
    }
    catch (const std::exception &e) {
        NIXL_WARN << "Failed to parse worker_id from customParam: " << e.what();
        return std::nullopt;
    }
}

nixl_status_t nixlUcxEngine::prepXfer (const nixl_xfer_op_t &operation,
                                       const nixl_meta_dlist_t &local,
                                       const nixl_meta_dlist_t &remote,
                                       const std::string &remote_agent,
                                       nixlBackendReqH* &handle,
                                       const nixl_opt_b_args_t* opt_args) const
{
    if (local.descCount() == 0 || remote.descCount() == 0) {
        NIXL_ERROR << "Local or remote descriptor list is empty";
        return NIXL_ERR_INVALID_PARAM;
    }

    const size_t worker_id = getWorkerId(opt_args);
    /* TODO: try to get from a pool first */
    const auto int_handle = new nixlUcxBackendReqH(getWorker(worker_id).get(), worker_id);
    const nixl_status_t status =
        prepareHandleAttestation(
            int_handle, operation, local, remote, remote_agent, opt_args);
    if (status != NIXL_SUCCESS) {
        delete int_handle;
        return status;
    }
    handle = int_handle;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxEngine::prepareHandleAttestation(nixlBackendReqH *handle,
                                        const nixl_xfer_op_t &operation,
                                        const nixl_meta_dlist_t &local,
                                        const nixl_meta_dlist_t &remote,
                                        const std::string &remote_agent,
                                        const nixl_opt_b_args_t *opt_args) const {
    const auto int_handle = static_cast<nixlUcxBackendReqH *>(handle);
    return int_handle->prepareAttestation(
        operation,
        local,
        remote,
        localAgent,
        remote_agent,
        opt_args == nullptr ? nullptr : opt_args->remoteAgentAuthority);
}

nixl_status_t nixlUcxEngine::estimateXferCost (const nixl_xfer_op_t &operation,
                                               const nixl_meta_dlist_t &local,
                                               const nixl_meta_dlist_t &remote,
                                               const std::string &remote_agent,
                                               nixlBackendReqH* const &handle,
                                               std::chrono::microseconds &duration,
                                               std::chrono::microseconds &err_margin,
                                               nixl_cost_t &method,
                                               const nixl_opt_args_t* opt_args) const
{
    const auto int_handle = static_cast<nixlUcxBackendReqH *>(handle);
    const size_t worker_id = int_handle->getWorkerId();

    if (local.descCount() != remote.descCount()) {
        NIXL_ERROR << "Local (" << local.descCount() << ") and remote (" << remote.descCount()
                   << ") descriptor lists differ in size for cost estimation";
        return NIXL_ERR_MISMATCH;
    }

    duration = std::chrono::microseconds(0);
    err_margin = std::chrono::microseconds(0);

    if (local.descCount() == 0) {
        // Nothing to do, use a default value
        method = nixl_cost_t::ANALYTICAL_BACKEND;
        return NIXL_SUCCESS;
    }

    for (int i = 0; i < local.descCount(); i++) {
        const size_t lsize = local[i].len;
        const size_t rsize = remote[i].len;

        const auto lmd = static_cast<nixlUcxPrivateMetadata *>(local[i].metadataP);
        const auto rmd = static_cast<nixlUcxPublicMetadata *>(remote[i].metadataP);

        NIXL_ASSERT(lmd && rmd) << "No metadata found in descriptor lists at index " << i << " during cost estimation";
        NIXL_ASSERT(lsize == rsize) << "Local size (" << lsize << ") != Remote size (" << rsize
                                    << ") at index " << i << " during cost estimation";

        std::chrono::microseconds msg_duration;
        std::chrono::microseconds msg_err_margin;
        nixl_cost_t msg_method;
        const nixl_status_t ret = rmd->conn->getEp(worker_id)->estimateCost(
            lsize, msg_duration, msg_err_margin, msg_method);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Worker failed to estimate cost for segment " << i << " status: " << ret;
            return ret;
        }

        duration += msg_duration;
        err_margin += msg_err_margin;
        method = msg_method;
    }

    return NIXL_SUCCESS;
}

nixlUcxEngine::batchResult
nixlUcxEngine::sendXferRangeBatch(nixlUcxEp &ep,
                                  nixl_xfer_op_t operation,
                                  const nixl_meta_dlist_t &local,
                                  const nixl_meta_dlist_t &remote,
                                  nixlBackendReqH *handle,
                                  size_t worker_id,
                                  size_t start_idx,
                                  size_t end_idx) {
    batchResult result = {NIXL_SUCCESS, 0, nullptr};
    const auto int_handle = static_cast<nixlUcxBackendReqH *>(handle);
    std::vector<nixl_xfer_attestation_transport_t> endpoint_transports;
    const std::shared_ptr<nixlUcxTerminalArm> terminal =
        int_handle->getActiveTerminal();

    for (size_t i = start_idx; i < end_idx; ++i) {
        void *laddr = (void *)local[i].addr;
        size_t lsize = local[i].len;
        uint64_t raddr = static_cast<uint64_t>(remote[i].addr);
        NIXL_ASSERT(lsize == remote[i].len);

        const auto lmd = static_cast<nixlUcxPrivateMetadata *>(local[i].metadataP);
        const auto rmd = static_cast<nixlUcxPublicMetadata *>(remote[i].metadataP);
        auto &rmd_ep = rmd->conn->getEp(worker_id);
        if (rmd_ep.get() != &ep) [[unlikely]] {
            break;
        }

        ++result.size;
        nixlUcxReq req = nullptr;
        std::string request_info;
        std::vector<nixl_xfer_attestation_transport_t> selected_transports;
        std::shared_ptr<nixl::ucx::ucx_callback_slot_t> terminal_slot;
        if (terminal != nullptr) {
            const nixl_status_t slot_status = terminal->makeSlot(
                int_handle->getWorker(),
                nixl::ucx::ucx_callback_kind_t::DATA_CHUNK,
                {},
                terminal_slot);
            if (slot_status != NIXL_SUCCESS) {
                result.status = slot_status;
                break;
            }
        }
        const nixl_status_t ret = operation == NIXL_READ ?
            ep.read(raddr,
                    rmd->getRkey(worker_id),
                    laddr,
                    lmd->mem,
                    lsize,
                    req,
                    request_info,
                    selected_transports,
                    terminal_slot.get()) :
            ep.write(laddr,
                     lmd->mem,
                     raddr,
                     rmd->getRkey(worker_id),
                     lsize,
                     req,
                     request_info,
                     selected_transports,
                     terminal_slot.get());

        if (ret == NIXL_IN_PROG) {
            if (endpoint_transports.empty()) {
                result.status = ep.queryTransports(endpoint_transports);
                if (result.status != NIXL_SUCCESS) {
                    if (terminal_slot == nullptr) {
                        ucp_request_free(req);
                    } else {
                        static_cast<void>(terminal_slot->armPoster(
                            req,
                            ret,
                            nixl::ucx::terminalProgressTimestampNs()));
                    }
                    if (result.req != nullptr) {
                        ucp_request_free(result.req);
                    }
                    result.req = nullptr;
                    break;
                }
            }
            const nixl_status_t evidence_status = int_handle->recordSegment(
                i, ep, endpoint_transports, selected_transports, request_info);
            if (evidence_status != NIXL_SUCCESS) {
                if (terminal_slot == nullptr) {
                    ucp_request_free(req);
                } else {
                    static_cast<void>(terminal_slot->armPoster(
                        req,
                        ret,
                        nixl::ucx::terminalProgressTimestampNs()));
                }
                if (result.req != nullptr) {
                    ucp_request_free(result.req);
                }
                result.status = evidence_status;
                result.req = nullptr;
                break;
            }
            if (terminal_slot != nullptr) {
                const nixl_status_t arm_status = terminal_slot->armPoster(
                    req, ret, nixl::ucx::terminalProgressTimestampNs());
                if (arm_status != NIXL_SUCCESS) {
                    result.status = arm_status;
                    break;
                }
                int_handle->trackConnection(rmd->conn);
            } else if (result.req != nullptr) [[likely]] {
                ucp_request_free(result.req);
            }
            if (terminal_slot == nullptr) {
                result.req = req;
            }
        } else if (ret == NIXL_SUCCESS && terminal_slot != nullptr) {
            if (endpoint_transports.empty()) {
                result.status = ep.queryTransports(endpoint_transports);
            }
            if (result.status == NIXL_SUCCESS) {
                selected_transports = endpoint_transports;
                request_info = "immediate UCX completion";
                result.status = int_handle->recordSegment(
                    i, ep, endpoint_transports, selected_transports, request_info);
            }
            const nixl_status_t arm_status = terminal_slot->armPoster(
                nullptr,
                result.status == NIXL_SUCCESS ? NIXL_SUCCESS : result.status,
                nixl::ucx::terminalProgressTimestampNs());
            if (result.status == NIXL_SUCCESS && arm_status != NIXL_SUCCESS) {
                result.status = arm_status;
            }
            if (result.status != NIXL_SUCCESS) {
                break;
            }
            int_handle->trackConnection(rmd->conn);
        } else if (ret != NIXL_SUCCESS) {
            if (terminal_slot != nullptr) {
                static_cast<void>(terminal_slot->armPoster(
                    nullptr, ret, nixl::ucx::terminalProgressTimestampNs()));
            }
            result.status = ret;
            if (result.req != nullptr) {
                ucp_request_free(result.req);
                result.req = nullptr;
            }
            break;
        }
    }

    if (terminal != nullptr) {
        result.req = nullptr;
    } else if (result.status == NIXL_SUCCESS && result.req) {
        result.status = NIXL_IN_PROG;
    }
    return result;
}

nixl_status_t
nixlUcxEngine::sendXferRange(const nixl_xfer_op_t &operation,
                             const nixl_meta_dlist_t &local,
                             const nixl_meta_dlist_t &remote,
                             const std::string &remote_agent,
                             nixlBackendReqH *handle,
                             size_t start_idx,
                             size_t end_idx) const {
    const auto int_handle = static_cast<nixlUcxBackendReqH *>(handle);
    const size_t worker_id = int_handle->getWorkerId();
    const std::shared_ptr<nixlUcxTerminalArm> terminal =
        int_handle->getActiveTerminal();

    if (operation != NIXL_WRITE && operation != NIXL_READ) {
        return NIXL_ERR_INVALID_PARAM;
    }

    /* Assuming we have a single EP, we need 3 requests: one pending request,
     * one flush request, and one notification request */
    int_handle->reserve(3);

    for (size_t i = start_idx; i < end_idx;) {
        /* Send requests to a single EP */
        const auto rmd = static_cast<nixlUcxPublicMetadata *>(remote[i].metadataP);
        auto &ep = rmd->conn->getEp(worker_id);
        const batchResult result =
            sendXferRangeBatch(
                *ep, operation, local, remote, handle, worker_id, i, end_idx);

        if (result.status == NIXL_SUCCESS && result.size == 0) {
            const nixl_status_t no_progress_status = NIXL_ERR_BACKEND;
            int_handle->getAttestationState()->fail(
                int_handle->getAttestationState()->getGeneration(),
                no_progress_status,
                "UCX transfer submission made no progress");
            if (terminal != nullptr) {
                static_cast<void>(terminal->fail(no_progress_status));
            }
            return no_progress_status;
        }

        if (terminal != nullptr) {
            if (result.status != NIXL_SUCCESS) {
                return result.status;
            }
            i += result.size;
            continue;
        }

        /* Append a single pending request for the entire EP batch */
        const nixl_status_t ret = int_handle->append(
            result.status,
            result.req,
            rmd->conn,
            nixlUcxBackendReqH::pending_kind_t::DATA,
            ep->getIdentity());
        if (ret != NIXL_SUCCESS) {
            int_handle->getAttestationState()->fail(
                int_handle->getAttestationState()->getGeneration(),
                ret,
                "UCX data submission failed");
            return ret;
        }

        i += result.size;
    }

    /*
     * Flush keeps int_handle non-empty until the operation is actually
     * completed, which can happen after local requests completion.
     * We need to flush all distinct connections to ensure that the operation
     * is actually completed.
     */
    for (auto &conn : int_handle->getConnections()) {
        nixlUcxReq req = nullptr;
        auto &ep = conn->getEp(worker_id);
        std::shared_ptr<nixl::ucx::ucx_callback_slot_t> terminal_slot;
        if (terminal != nullptr) {
            const uint64_t generation = int_handle->getAttestationState()->getGeneration();
            const uint64_t endpoint_identity = ep->getIdentity();
            const nixl_status_t slot_status = terminal->makeSlot(
                int_handle->getWorker(),
                nixl::ucx::ucx_callback_kind_t::ENDPOINT_FLUSH,
                [attestation = int_handle->getAttestationState(),
                 generation,
                 endpoint_identity](nixl_status_t callback_status) {
                    if (callback_status != NIXL_SUCCESS) {
                        return callback_status;
                    }
                    return attestation->completeFlush(generation, endpoint_identity);
                },
                terminal_slot);
            if (slot_status != NIXL_SUCCESS) {
                return slot_status;
            }
        }
        const nixl_status_t ret = ep->flushEp(req, terminal_slot.get());
        const nixl_status_t evidence_status = int_handle->recordFlush(
            *ep,
            terminal_slot != nullptr && ret == NIXL_SUCCESS ? NIXL_IN_PROG : ret);
        if (evidence_status != NIXL_SUCCESS) {
            if (terminal_slot != nullptr) {
                static_cast<void>(terminal_slot->armPoster(
                    req, ret, nixl::ucx::terminalProgressTimestampNs()));
            }
            return evidence_status;
        }
        if (terminal_slot != nullptr) {
            const nixl_status_t arm_status = terminal_slot->armPoster(
                req, ret, nixl::ucx::terminalProgressTimestampNs());
            if (arm_status != NIXL_SUCCESS) {
                return arm_status;
            }
            continue;
        }
        const nixl_status_t append_status = int_handle->append(
            ret,
            req,
            conn,
            nixlUcxBackendReqH::pending_kind_t::ENDPOINT_FLUSH,
            ep->getIdentity());
        if (append_status != NIXL_SUCCESS) {
            int_handle->getAttestationState()->fail(
                int_handle->getAttestationState()->getGeneration(),
                append_status,
                "UCX endpoint flush submission failed");
            return append_status;
        }
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxEngine::postXfer(const nixl_xfer_op_t &operation,
                        const nixl_meta_dlist_t &local,
                        const nixl_meta_dlist_t &remote,
                        const std::string &remote_agent,
                        nixlBackendReqH *&handle,
                        const nixl_opt_b_args_t *opt_args) const {
    const size_t lcnt = local.descCount();
    const size_t rcnt = remote.descCount();
    const auto int_handle = static_cast<nixlUcxBackendReqH *>(handle);
    nixl_status_t ret;
    std::vector<std::uint8_t> notification_frame;
    ucx_connection_ptr_t notification_connection;

    if (lcnt != rcnt) {
        NIXL_ERROR << "Local (" << lcnt << ") and remote (" << rcnt
                   << ") descriptor lists differ in size";
        return NIXL_ERR_INVALID_PARAM;
    }

    if (opt_args != nullptr && opt_args->hasNotif) {
        if (opt_args->remoteAgentAuthority == nullptr) {
            return NIXL_ERR_INVALID_PARAM;
        }
        ret = prepareDataFrame(*opt_args->remoteAgentAuthority,
                               int_handle->getWorkerId(),
                               opt_args->notifMsg,
                               notification_frame,
                               notification_connection);
        if (ret != NIXL_SUCCESS) {
            return ret;
        }
        const auto remote_metadata =
            static_cast<nixlUcxPublicMetadata *>(remote[0].metadataP);
        if (remote_metadata == nullptr ||
            remote_metadata->conn != notification_connection) {
            return NIXL_ERR_NOT_ALLOWED;
        }
    }

    const bool has_notification = opt_args != nullptr && opt_args->hasNotif;
    ret = int_handle->beginSubmission(has_notification);
    if (ret != NIXL_SUCCESS) {
        return ret;
    }

    ret = sendXferRange(operation, local, remote, remote_agent, handle, 0, lcnt);
    if (ret != NIXL_SUCCESS) {
        int_handle->getAttestationState()->fail(
            int_handle->getAttestationState()->getGeneration(),
            ret,
            "UCX transfer submission failed");
        if (const std::shared_ptr<nixlUcxTerminalArm> terminal =
                int_handle->getActiveTerminal();
            terminal != nullptr) {
            static_cast<void>(terminal->fail(ret));
        }
        return ret;
    }

    ret = int_handle->finishSubmission();
    if (ret != NIXL_SUCCESS) {
        if (const std::shared_ptr<nixlUcxTerminalArm> terminal =
                int_handle->getActiveTerminal();
            terminal != nullptr) {
            static_cast<void>(terminal->fail(ret));
        }
        return ret;
    }

    const std::shared_ptr<nixlUcxTerminalArm> terminal =
        int_handle->getActiveTerminal();
    if (terminal != nullptr) {
        nixl::ucx::terminal_submission_state_t::notification_post_t notification_post;
        if (has_notification) {
            nixlUcxWorker *const notification_worker = int_handle->getWorker();
            const size_t notification_worker_id = int_handle->getWorkerId();
            notification_post =
                [this,
                 terminal,
                 notification_worker,
                 notification_worker_id,
                 notification_connection,
                 frame = std::move(notification_frame)]() mutable {
                    const nixl_status_t enqueue_status =
                        notification_worker->enqueueContinuation(
                        [this,
                         terminal,
                         notification_worker,
                         notification_worker_id,
                         notification_connection,
                         frame = std::move(frame)]() mutable {
                            std::shared_ptr<nixl::ucx::ucx_callback_slot_t> slot;
                            nixl_status_t status = terminal->makeSlot(
                                notification_worker,
                                nixl::ucx::ucx_callback_kind_t::NOTIFICATION,
                                {},
                                slot);
                            if (status != NIXL_SUCCESS) {
                                return status;
                            }
                            nixlUcxReq request = nullptr;
                            status = notifSendPriv(
                                std::move(frame),
                                notification_connection->getEp(notification_worker_id),
                                &request,
                                slot.get());
                            const nixl_status_t arm_status = slot->armPoster(
                                request,
                                status,
                                nixl::ucx::terminalProgressTimestampNs());
                            if (arm_status != NIXL_SUCCESS) {
                                return arm_status;
                            }
                            return NIXL_SUCCESS;
                        });
                    return enqueue_status == NIXL_SUCCESS ? NIXL_IN_PROG : enqueue_status;
                };
        }
        ret = terminal->seal(std::move(notification_post));
        if (ret != NIXL_SUCCESS) {
            static_cast<void>(terminal->fail(ret));
            return ret;
        }
        return terminal->status();
    }

    ret = int_handle->status();
    if (opt_args != nullptr && opt_args->hasNotif) {
        if (ret == NIXL_SUCCESS) {
            nixlUcxReq req;
            ret = notifSendPriv(std::move(notification_frame),
                                notification_connection->getEp(int_handle->getWorkerId()),
                                &req);
            const nixl_status_t append_status =
                int_handle->append(ret, req, notification_connection);
            if (append_status != NIXL_SUCCESS) {
                return append_status;
            }

            ret = int_handle->status();
        } else if (ret == NIXL_IN_PROG) {
            int_handle->notif.emplace(
                std::move(notification_frame), notification_connection);
        }
    }

    return ret;
}

nixl_status_t nixlUcxEngine::checkXfer (nixlBackendReqH* handle) const
{
    const auto int_handle = static_cast<nixlUcxBackendReqH *>(handle);
    const nixl_status_t handle_status = int_handle->status();

    if ((handle_status == NIXL_IN_PROG) || !int_handle->notif) {
        return handle_status;
    }

    nixlUcxBackendReqH::Notif notif(std::move(int_handle->notif).value());
    int_handle->notif.reset();

    if (handle_status != NIXL_SUCCESS) [[unlikely]] {
        return handle_status;
    }

    const ucx_connection_ptr_t &conn = notif.connection;
    if (conn == nullptr) [[unlikely]] {
        return NIXL_ERR_NOT_FOUND;
    }

    nixlUcxReq req;
    const auto &ep = conn->getEp(int_handle->getWorkerId());
    const nixl_status_t status =
        notifSendPriv(std::move(notif.frame), ep, &req);

    const nixl_status_t append_status = int_handle->append(status, req, conn);
    if (append_status != NIXL_SUCCESS) {
        return append_status;
    }

    return int_handle->status();
}

nixl_status_t
nixlUcxEngine::queryXferAttestation(const nixlBackendReqH *handle,
                                    nixl_xfer_attestation_t &attestation) const {
    if (handle == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const auto int_handle = static_cast<const nixlUcxBackendReqH *>(handle);
    attestation = int_handle->queryAttestation();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxEngine::takeXferCompletionAttestation(
    nixlBackendReqH *handle,
    nixl_xfer_attestation_t &attestation) const {
    if (handle == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const auto int_handle = static_cast<nixlUcxBackendReqH *>(handle);
    return int_handle->takeCompletionAttestation(attestation);
}

nixl_status_t
nixlUcxEngine::subscribeXferTerminal(
    nixlBackendReqH *handle,
    const nixlBackendTransferEventBinding &binding,
    const std::shared_ptr<nixlBackendTransferTransitionSink> &sink,
    std::unique_ptr<nixlBackendEventSubscription> &subscription) {
    subscription.reset();
    if (handle == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    std::vector<nixlUcxWorker *> workers;
    workers.reserve(getWorkers().size());
    for (const std::unique_ptr<nixlUcxWorker> &worker : getWorkers()) {
        workers.push_back(worker.get());
    }
    return static_cast<nixlUcxBackendReqH *>(handle)->armTerminal(
        binding, sink, workers, subscription);
}

nixl_status_t nixlUcxEngine::releaseReqH(nixlBackendReqH* handle) const
{
    const auto int_handle = static_cast<nixlUcxBackendReqH *>(handle);
    if (!int_handle->terminalLifecycleDrained()) {
        return NIXL_ERR_REPOST_ACTIVE;
    }
    int_handle->release();

    /* TODO: return to a pool instead. */
    delete int_handle;

    return NIXL_SUCCESS;
}

unsigned
nixlUcxEngine::progress() {
    // TODO: add listen for connection handling if necessary
    unsigned ret = 0;
    for (auto &uw : uws) {
        ret += uw->progress();
    }
    return ret;
}

void
nixlUcxEngine::progressLoop() {
    while (progress() != 0)
        ;
}

/****************************************
 * Notifications
*****************************************/

nixl_status_t
nixlUcxEngine::notifSendPriv(std::vector<std::uint8_t> &&frame,
                             const std::unique_ptr<nixlUcxEp> &ep,
                             nixlUcxReq *req,
                             nixl::ucx::ucx_callback_slot_t *terminal_slot) const {
    if (ep == nullptr || frame.empty() ||
        frame.size() > nixl::ucx::notif_wire_max_frame_size) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const nixl_status_t endpoint_status = ep->checkTxState();
    if (endpoint_status != NIXL_SUCCESS) {
        return endpoint_status;
    }

    auto *buffer = new std::vector<std::uint8_t>(std::move(frame));
    auto deleter = [buffer, req, terminal_slot](void *completed_request, void *) {
        delete buffer;
        if (req == nullptr && terminal_slot == nullptr && completed_request != nullptr) {
            /* Caller is not interested in the request, free it */
            ucp_request_free(completed_request);
        }
    };

    return ep->sendAm(nixl::ucx::am_cb_op_t::NOTIF_STR,
                      nullptr,
                      0,
                      buffer->data(),
                      buffer->size(),
                      UCP_AM_SEND_FLAG_EAGER,
                      req,
                      deleter,
                      terminal_slot);
}

nixl_status_t
nixlUcxEngine::sendControlFrame(
    const nixl::ucx::notif_wire_envelope_t &envelope,
    uint64_t connection_identity,
    size_t worker_id) const {
    const ucx_connection_ptr_t connection =
        getConnection(connection_identity);
    if (connection == nullptr || worker_id >= connection->eps.size()) {
        return NIXL_ERR_NOT_FOUND;
    }

    std::vector<std::uint8_t> frame;
    const nixl::ucx::notif_wire_status_t encode_status =
        nixl::ucx::encodeNotifWireFrame(envelope, {}, frame);
    if (encode_status != nixl::ucx::notif_wire_status_t::SUCCESS) {
        return NIXL_ERR_BACKEND;
    }
    const nixl_status_t send_status =
        notifSendPriv(std::move(frame), connection->getEp(worker_id));
    return send_status == NIXL_IN_PROG ? NIXL_SUCCESS : send_status;
}

ucx_connection_ptr_t
nixlUcxEngine::getConnection(const std::string &remote_agent) const {
    const std::lock_guard lock(connectionMutex_);
    const auto it = remoteConnMap.find(remote_agent);
    return (it != remoteConnMap.end()) ? it->second : nullptr;
}

ucx_connection_ptr_t
nixlUcxEngine::getConnection(uint64_t connection_identity) const {
    const std::lock_guard lock(connectionMutex_);
    for (const auto &entry : remoteConnMap) {
        const ucx_connection_ptr_t &connection = entry.second;
        if (connection->getIdentity() == connection_identity) {
            return connection;
        }
    }
    return nullptr;
}

std::optional<nixlUcxEngine::exactRouteRecord>
nixlUcxEngine::getExactRoute(uint64_t handle_identity,
                             uint64_t generation) const {
    const std::lock_guard lock(exactRouteMutex_);
    const auto route = exactRoutes_.find(handle_identity);
    if (route == exactRoutes_.end() ||
        route->second.route.handleGeneration != generation) {
        return std::nullopt;
    }
    return route->second;
}

void
nixlUcxEngine::failExactRoutesForConnection(uint64_t connection_identity) noexcept {
    std::vector<nixl::ucx::notif_route_key_t> routes;
    {
        const std::lock_guard lock(exactRouteMutex_);
        for (const auto &[identity, record] : exactRoutes_) {
            static_cast<void>(identity);
            if (record.connectionIdentity == connection_identity) {
                routes.push_back(record.route);
            }
        }
    }
    for (const nixl::ucx::notif_route_key_t &route : routes) {
        const nixl::ucx::notif_state_status_t status =
            notifState_->failRemoteAgent(route);
        if (status != nixl::ucx::notif_state_status_t::SUCCESS &&
            status != nixl::ucx::notif_state_status_t::ROUTE_FAILED &&
            status != nixl::ucx::notif_state_status_t::ROUTE_RETIRED) {
            NIXL_WARN << "Failed to publish exact-route endpoint failure for handle "
                      << route.handleIdentity << ": " << static_cast<int>(status);
        }
    }
}

nixl_status_t
nixlUcxEngine::prepareDataFrame(
    const nixl_remote_agent_authority_t &authority,
    size_t worker_id,
    const std::string &msg,
    std::vector<std::uint8_t> &frame,
    ucx_connection_ptr_t &connection) const {
    if (worker_id >= localConnectionMetadata_.workers.size()) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const std::optional<exactRouteRecord> route =
        getExactRoute(authority.handleIdentity, authority.generation);
    if (!route.has_value() ||
        route->connectionIdentity != authority.connectionIdentity) {
        return NIXL_ERR_NOT_FOUND;
    }

    ucx_connection_ptr_t exact_connection =
        getConnection(route->connectionIdentity);
    if (exact_connection == nullptr) {
        return NIXL_ERR_NOT_FOUND;
    }
    std::vector<uint64_t> endpoint_identities;
    endpoint_identities.reserve(exact_connection->eps.size());
    for (const auto &endpoint : exact_connection->eps) {
        endpoint_identities.push_back(endpoint->getIdentity());
    }
    std::sort(endpoint_identities.begin(), endpoint_identities.end());
    if (endpoint_identities != authority.endpointIdentities) {
        return NIXL_ERR_NOT_ALLOWED;
    }

    nixl::ucx::notif_wire_envelope_t envelope;
    const nixl::ucx::notif_wire_uuid_t &worker =
        localConnectionMetadata_.workers[worker_id].incarnation;
    const nixl::ucx::notif_state_status_t state_status =
        notifState_->prepareData(route->route, worker, envelope);
    if (state_status == nixl::ucx::notif_state_status_t::NOT_READY) {
        nixl::ucx::notif_wire_envelope_t offer;
        if (notifState_->makeOffer(route->route, worker, offer) ==
            nixl::ucx::notif_state_status_t::SUCCESS) {
            (void)sendControlFrame(
                offer, exact_connection->getIdentity(), worker_id);
        }
        return NIXL_ERR_NOT_READY;
    }
    if (state_status != nixl::ucx::notif_state_status_t::SUCCESS) {
        return notifStateToNixl(state_status);
    }

    std::vector<std::uint8_t> encoded;
    const auto payload = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t *>(msg.data()), msg.size());
    const nixl::ucx::notif_wire_status_t encode_status =
        nixl::ucx::encodeNotifWireFrame(envelope, payload, encoded);
    if (encode_status != nixl::ucx::notif_wire_status_t::SUCCESS) {
        return encode_status == nixl::ucx::notif_wire_status_t::FRAME_TOO_LARGE ?
            NIXL_ERR_INVALID_PARAM : NIXL_ERR_BACKEND;
    }

    frame.swap(encoded);
    connection = std::move(exact_connection);
    return NIXL_SUCCESS;
}

void
nixlUcxEngine::appendNotif(nixlAuthenticatedNotification &&notification) const {
    // In the "no progress thread" case the lock in nixlAgent is sufficient.
    (void)notifQueue_.push(std::move(notification));
}

void
nixlUcxEngine::poisonNotifs() const {
    // In the "no progress thread" case the lock in nixlAgent is sufficient.
    notifQueue_.poison();
}

ucs_status_t
nixlUcxEngine::notifAmCb(void *arg, const void *header,
                         size_t header_length, void *data,
                         size_t length,
                         const ucp_am_recv_param_t *param)
{
    if (arg == nullptr || param == nullptr ||
        (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV) != 0 ||
        data == nullptr ||
        header_length != 0 ||
        length > nixlUcxNotificationQueue::maxWireBytes) {
        NIXL_ERROR << "Rejected malformed or oversized UCX notification frame";
        return UCS_OK;
    }

    const auto *context = static_cast<notifCallbackContext *>(arg);
    nixlUcxEngine *engine = context->engine;
    if (engine == nullptr ||
        context->workerId >= engine->localConnectionMetadata_.workers.size()) {
        return UCS_OK;
    }

    const auto wire = std::span<const std::uint8_t>(
        static_cast<const std::uint8_t *>(data), length);
    nixl::ucx::notif_wire_frame_view_t frame;
    if (nixl::ucx::decodeNotifWireFrame(wire, frame) !=
        nixl::ucx::notif_wire_status_t::SUCCESS) {
        NIXL_ERROR << "Rejected malformed UCX notification frame";
        engine->poisonNotifs();
        return UCS_OK;
    }

    const nixl::ucx::notif_wire_uuid_t &local_worker =
        engine->localConnectionMetadata_.workers[context->workerId].incarnation;
    if (frame.envelope.type == nixl::ucx::notif_wire_type_t::OFFER) {
        nixl::ucx::notif_offer_acceptance_t acceptance;
        const nixl::ucx::notif_state_status_t status =
            engine->notifState_->acceptOffer(
                frame.envelope, local_worker, acceptance);
        if (status != nixl::ucx::notif_state_status_t::SUCCESS) {
            NIXL_DEBUG << "Rejected UCX notification OFFER with state status "
                       << static_cast<int>(status);
            return UCS_OK;
        }

        const std::optional<exactRouteRecord> route = engine->getExactRoute(
            acceptance.route.handleIdentity,
            acceptance.route.handleGeneration);
        if (!route.has_value() || route->route != acceptance.route) {
            NIXL_ERROR << "Rejected UCX notification OFFER for an unknown exact route";
            return UCS_OK;
        }
        const nixl_status_t ack_status = engine->sendControlFrame(
            acceptance.acknowledgement,
            route->connectionIdentity,
            context->workerId);
        if (ack_status != NIXL_SUCCESS) {
            NIXL_WARN << "Failed to send UCX notification ACK: " << ack_status;
        }
        if (acceptance.reemitLocalOffer) {
            const nixl_status_t offer_status = engine->sendControlFrame(
                acceptance.localOffer,
                route->connectionIdentity,
                context->workerId);
            if (offer_status != NIXL_SUCCESS) {
                NIXL_WARN << "Failed to re-emit UCX notification OFFER: "
                          << offer_status;
            }
        }
        return UCS_OK;
    }

    if (frame.envelope.type == nixl::ucx::notif_wire_type_t::ACK) {
        nixl::ucx::notif_route_key_t route;
        const nixl::ucx::notif_state_status_t status =
            engine->notifState_->acceptAcknowledgement(frame.envelope, route);
        if (status != nixl::ucx::notif_state_status_t::SUCCESS) {
            NIXL_DEBUG << "Rejected UCX notification ACK with state status "
                       << static_cast<int>(status);
        }
        return UCS_OK;
    }

    const nixl::ucx::notif_data_resolution_t resolution =
        engine->notifState_->resolveData(frame.envelope);
    if (resolution.disposition ==
        nixl::ucx::notif_data_disposition_t::POISON_GLOBAL) {
        NIXL_ERROR << "Unknown UCX notification capability poisoned authenticated draining";
        engine->poisonNotifs();
        return UCS_OK;
    }
    if (resolution.disposition ==
        nixl::ucx::notif_data_disposition_t::DROP_ROUTE) {
        NIXL_ERROR << "Dropped invalid UCX notification for an exact route";
        return UCS_OK;
    }

    const std::optional<exactRouteRecord> route = engine->getExactRoute(
        resolution.authority.route.handleIdentity,
        resolution.authority.route.handleGeneration);
    if (!route.has_value() || route->route != resolution.authority.route) {
        NIXL_ERROR << "Dropped UCX notification with missing exact route ownership";
        return UCS_OK;
    }

    engine->appendNotif({
        .remoteAgent = route->remoteAgent,
        .payload = std::string(
            reinterpret_cast<const char *>(frame.payload.data()),
            frame.payload.size()),
        .handleIdentity = resolution.authority.route.handleIdentity,
        .generation = resolution.authority.route.handleGeneration,
        .connectionIdentity = resolution.authority.connectionIdentity,
        .endpointIdentity = resolution.authority.endpointIdentity,
    });
    return UCS_OK;
}

nixl_status_t
nixlUcxEngine::getNotifs(notif_list_t &notif_list) {
    progressLoop();

    // In the "no progress thread" case the lock in nixlAgent is sufficient.
    return notifQueue_.drainLegacy(notif_list);
}

nixl_status_t
nixlUcxEngine::getAuthenticatedNotifs(authenticated_notif_list_t &notif_list) {
    progressLoop();

    // In the "no progress thread" case the lock in nixlAgent is sufficient.
    return notifQueue_.drainAuthenticated(notif_list);
}

nixl_status_t
nixlUcxEngine::genNotif(const std::string &remote_agent, const std::string &msg) const {
    if (remote_agent != localAgent) {
        return NIXL_ERR_INVALID_PARAM;
    }
    appendNotif({.remoteAgent = localAgent, .payload = msg});
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxEngine::genNotif(const nixlRemoteAgentBinding &binding,
                        const std::string &msg) const {
    const size_t worker_id = getWorkerId();
    std::vector<std::uint8_t> frame;
    ucx_connection_ptr_t connection;
    const nixl_status_t prepare_status = prepareDataFrame(
        binding.authority, worker_id, msg, frame, connection);
    if (prepare_status != NIXL_SUCCESS) {
        return prepare_status;
    }

    const nixl_status_t ret = notifSendPriv(
        std::move(frame), connection->getEp(worker_id));
    if (ret == NIXL_IN_PROG) {
        return NIXL_SUCCESS;
    }
    return ret;
}

nixl_status_t
nixlUcxEngine::prepMemView(const nixl_remote_meta_dlist_t &dlist,
                           nixlMemViewH &mvh,
                           const nixl_opt_b_args_t *opt_args) const {
    const size_t worker_id = getWorkerId(opt_args);
    try {
        mvh = nixl::ucx::createMemList(dlist, worker_id, *getWorker(worker_id));
        return NIXL_SUCCESS;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to prepare remote memory view: " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlUcxEngine::prepMemView(const nixl_meta_dlist_t &dlist,
                           nixlMemViewH &mvh,
                           const nixl_opt_b_args_t *opt_args) const {
    const size_t worker_id = getWorkerId(opt_args);
    try {
        mvh = nixl::ucx::createMemList(dlist, *getWorker(worker_id));
        return NIXL_SUCCESS;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to prepare local memory view: " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

void
nixlUcxEngine::releaseMemView(nixlMemViewH mem_view) const {
    nixl::ucx::releaseMemList(mem_view);
}
