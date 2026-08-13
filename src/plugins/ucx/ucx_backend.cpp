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
#include <array>
#include <atomic>
#include <cerrno>
#include <deque>
#include <fcntl.h>
#include <optional>
#include <limits>
#include <future>
#include <set>
#include <new>
#include <string.h>
#include <unistd.h>
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"

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
parseWireUuid(std::string_view encoded, nixl::ucx::notif_wire_uuid_t &uuid) noexcept {
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
    if (nibble_index != parsed.bytes.size() * 2 || !nixl::ucx::isCanonicalNotifWireUuid(parsed)) {
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
notifSubscriptionToNixl(nixl::ucx::notif_route_subscription_status_t status) noexcept {
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

[[nodiscard]] std::string
formatWireUuid(const nixl::ucx::notif_wire_uuid_t &uuid) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(36);
    for (std::size_t index = 0; index < uuid.bytes.size(); ++index) {
        if (index == 4 || index == 6 || index == 8 || index == 10) {
            result.push_back('-');
        }
        result.push_back(digits[uuid.bytes[index] >> 4U]);
        result.push_back(digits[uuid.bytes[index] & 0x0fU]);
    }
    return result;
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
    const bool byte_overflow =
        item_overflow || queuedBytes_ > maxBytes_ || notification_size > maxBytes_ - queuedBytes_;
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
        notifications.emplace_back(std::move(notification.remoteAgent),
                                   std::move(notification.payload));
    }
    notifications_.clear();
    queuedBytes_ = 0;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxNotificationQueue::drainAuthenticated(authenticated_notif_list_t &notifications) {
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

class nixlUcxTerminalArm;

class nixlUcxNoopTerminalSink final : public nixl::ucx::terminal_submission_sink_t {
public:
    nixl_status_t
    publishTerminal(const nixl::ucx::terminal_submission_result_t &) noexcept override {
        return NIXL_SUCCESS;
    }
};

class nixlUcxNoopTransferSink final : public nixlBackendTransferTransitionSink {
public:
    void
    publish(const nixlBackendTransferTransition &) noexcept override {}
};

class nixlUcxTerminalOwnership final {
public:
    void
    detach(const nixlUcxTerminalArm *arm) noexcept;

private:
    friend class nixlUcxBackendReqH;

    mutable std::mutex mutex_;
    std::shared_ptr<nixlUcxTerminalArm> armed_;
    std::shared_ptr<nixlUcxTerminalArm> active_;
};

class nixlUcxTerminalArm final : public nixl::ucx::terminal_submission_sink_t,
                                 public std::enable_shared_from_this<nixlUcxTerminalArm> {
    struct slot_record_t {
        nixlUcxWorker *worker = nullptr;
        std::shared_ptr<nixl::ucx::ucx_callback_slot_t> slot;
    };

    enum class notification_delivery_outcome_source_t {
        REGISTRY,
        SUBMISSION,
    };

public:
    nixlUcxTerminalArm(nixlBackendTransferEventBinding binding,
                       std::shared_ptr<nixlBackendTransferTransitionSink> sink,
                       std::shared_ptr<nixlUcxAttestationState> attestation,
                       std::vector<nixlUcxWorker *> workers,
                       nixlUcxWorker *primary_worker,
                       nixl::ucx::terminal_deadline_owner_t *deadline_owner,
                       std::weak_ptr<nixlUcxTerminalOwnership> ownership)
        : binding_(binding),
          sink_(std::move(sink)),
          attestation_(std::move(attestation)),
          workers_(std::move(workers)),
          primaryWorker_(primary_worker),
          deadlineOwner_(deadline_owner),
          ownership_(std::move(ownership)) {}

    [[nodiscard]] nixl_status_t
    begin(bool has_notification) {
        const std::lock_guard lock(mutex_);
        if (terminal_ || publishing_ || state_ != nullptr ||
            attestation_->getGeneration() != binding_.generation) {
            return NIXL_ERR_NOT_ALLOWED;
        }
        state_ = std::make_shared<nixl::ucx::terminal_submission_state_t>(binding_.handleIdentity,
                                                                          binding_.handleIdentity,
                                                                          binding_.generation,
                                                                          1,
                                                                          1,
                                                                          has_notification,
                                                                          shared_from_this());
        return NIXL_SUCCESS;
    }

    [[nodiscard]] nixl_status_t
    makeSlot(nixlUcxWorker *worker,
             nixl::ucx::ucx_callback_kind_t kind,
             nixl::ucx::ucx_callback_slot_t::owner_before_completion_t before,
             std::shared_ptr<nixl::ucx::ucx_callback_slot_t> &slot) {
        const std::lock_guard lock(mutex_);
        if (terminal_ || publishing_ || state_ == nullptr || worker == nullptr ||
            !worker->hasProgressOwner()) {
            return NIXL_ERR_NOT_ALLOWED;
        }
        try {
            slots_.reserve(slots_.size() + 1);
        }
        catch (const std::bad_alloc &) {
            return NIXL_ERR_BACKEND;
        }
        const std::shared_ptr<nixl::ucx::terminal_submission_state_t> state = state_;
        const nixl_status_t register_status = kind == nixl::ucx::ucx_callback_kind_t::DATA_CHUNK ?
            state->registerChunk() :
            (kind == nixl::ucx::ucx_callback_kind_t::ENDPOINT_FLUSH ? state->registerFlush() :
                                                                      NIXL_SUCCESS);
        if (register_status != NIXL_SUCCESS) {
            return register_status;
        }

        const std::weak_ptr<nixlUcxTerminalArm> weak_self = shared_from_this();
        const nixl_status_t slot_status =
            worker->makeTerminalCallbackSlot(state, kind, slot, std::move(before), [weak_self]() {
                if (const std::shared_ptr<nixlUcxTerminalArm> self = weak_self.lock();
                    self != nullptr) {
                    self->notifySlotDelivered();
                }
            });
        if (slot_status != NIXL_SUCCESS) {
            const nixl_status_t unregister_status =
                kind == nixl::ucx::ucx_callback_kind_t::DATA_CHUNK ?
                state->unregisterChunk() :
                (kind == nixl::ucx::ucx_callback_kind_t::ENDPOINT_FLUSH ? state->unregisterFlush() :
                                                                          NIXL_SUCCESS);
            if (unregister_status != NIXL_SUCCESS) {
                return NIXL_ERR_BACKEND;
            }
            return slot_status;
        }
        if (slot == nullptr) {
            const nixl_status_t unregister_status =
                kind == nixl::ucx::ucx_callback_kind_t::DATA_CHUNK ?
                state->unregisterChunk() :
                (kind == nixl::ucx::ucx_callback_kind_t::ENDPOINT_FLUSH ? state->unregisterFlush() :
                                                                          NIXL_SUCCESS);
            return unregister_status == NIXL_SUCCESS ? NIXL_ERR_BACKEND : unregister_status;
        }
        slots_.push_back({worker, slot});
        ++activeSlots_;
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
        return state->sealPosting(std::move(notification_post),
                                  nixl::ucx::terminalProgressTimestampNs());
    }

    [[nodiscard]] nixl_status_t
    armDeadline(std::uint64_t anchor_ns) {
        if (anchor_ns == 0 || primaryWorker_ == nullptr || deadlineOwner_ == nullptr) {
            return NIXL_ERR_BACKEND;
        }
        const std::shared_ptr<nixl::ucx::ucx_worker_continuation_queue_t> owner =
            primaryWorker_->getContinuationQueue();
        if (owner == nullptr) {
            return NIXL_ERR_BACKEND;
        }
        const std::lock_guard lock(mutex_);
        if (terminal_ || publishing_ || state_ == nullptr || deadlineArmed_) {
            return NIXL_ERR_NOT_ALLOWED;
        }

        const nixl::ucx::terminal_deadline_key_t key{
            .handleIdentity = binding_.handleIdentity,
            .generation = binding_.generation,
        };
        const std::shared_ptr<nixlUcxTerminalArm> self = shared_from_this();
        const nixl::ucx::terminal_deadline_status_t deadline_status = deadlineOwner_->arm(
            key,
            anchor_ns,
            nixl::ucx::nativeTransferTimeoutNs,
            [self, owner](const nixl::ucx::terminal_deadline_key_t &expired) noexcept {
                static_cast<void>(expired);
                return owner->enqueueProducer([self]() noexcept {
                    const nixl_status_t disposition = self->fail(NIXL_ERR_CANCELED);
                    if (disposition == NIXL_SUCCESS || disposition == NIXL_IN_PROG ||
                        disposition == NIXL_ERR_CANCELED) {
                        return NIXL_SUCCESS;
                    }
                    return disposition;
                });
            });
        if (deadline_status != nixl::ucx::terminal_deadline_status_t::SUCCESS) {
            return deadline_status == nixl::ucx::terminal_deadline_status_t::OWNER_CLOSED ?
                NIXL_ERR_NOT_ALLOWED :
                NIXL_ERR_BACKEND;
        }
        deadlineArmed_ = true;
        return NIXL_SUCCESS;
    }

    [[nodiscard]] nixl_status_t
    installNotificationDelivery(
        const std::shared_ptr<nixl::ucx::notif_delivery_registry_t> &registry,
        const nixl::ucx::notif_delivery_key_t &delivery,
        const std::shared_ptr<nixl::ucx::ucx_worker_continuation_queue_t> &owner,
        const nixl::ucx::notif_delivery_registry_t::authority_validator_t &validate_authority) {
        const std::lock_guard lock(mutex_);
        if (registry == nullptr || owner == nullptr || delivery.identity == 0 || terminal_ ||
            publishing_ || state_ == nullptr || deliveryIdentity_ != 0 || deliveryOutcomeClaimed_ ||
            !validate_authority) {
            return NIXL_ERR_NOT_ALLOWED;
        }

        const std::shared_ptr<nixlUcxTerminalArm> self = shared_from_this();
        const nixl::ucx::notif_delivery_status_t registration = registry->registerDelivery(
            delivery,
            [self, owner](nixl_status_t status, uint64_t timestamp_ns) noexcept {
                const nixl_status_t enqueue_status =
                    owner->enqueueProducer([self, status, timestamp_ns]() noexcept {
                        const nixl_status_t completion_status =
                            self->completeNotificationDelivery(status, timestamp_ns);
                        return completion_status == NIXL_SUCCESS || completion_status == status ?
                            NIXL_SUCCESS :
                            NIXL_ERR_BACKEND;
                    });
                if (enqueue_status != NIXL_SUCCESS) {
                    static_cast<void>(owner->fail(enqueue_status));
                }
            },
            validate_authority);
        if (registration != nixl::ucx::notif_delivery_status_t::SUCCESS) {
            switch (registration) {
            case nixl::ucx::notif_delivery_status_t::CAPACITY_EXCEEDED:
            case nixl::ucx::notif_delivery_status_t::ROUTE_TERMINAL:
            case nixl::ucx::notif_delivery_status_t::REGISTRY_CLOSED:
                return NIXL_ERR_NOT_ALLOWED;
            default:
                return NIXL_ERR_BACKEND;
            }
        }
        deliveryRegistry_ = registry;
        deliveryIdentity_ = delivery.identity;
        return NIXL_SUCCESS;
    }

    [[nodiscard]] nixl_status_t
    completeNotificationDelivery(nixl_status_t status, uint64_t timestamp_ns) noexcept {
        return claimNotificationDeliveryOutcome(
            status, timestamp_ns, notification_delivery_outcome_source_t::REGISTRY);
    }

    [[nodiscard]] nixl_status_t
    claimNotificationDeliveryOutcome(nixl_status_t status,
                                     uint64_t timestamp_ns,
                                     notification_delivery_outcome_source_t source) noexcept {
        if (status == NIXL_IN_PROG) {
            return NIXL_ERR_INVALID_PARAM;
        }
        if (source == notification_delivery_outcome_source_t::SUBMISSION) {
            if (status >= NIXL_SUCCESS) {
                return NIXL_ERR_INVALID_PARAM;
            }
            std::shared_ptr<nixl::ucx::notif_delivery_registry_t> registry;
            uint64_t delivery_identity = 0;
            bool terminal_publication_started = false;
            {
                const std::lock_guard lock(mutex_);
                if (deliveryOutcomeClaimed_) {
                    return status;
                }
                registry = deliveryRegistry_.lock();
                delivery_identity = deliveryIdentity_;
                terminal_publication_started =
                    publishing_ || terminal_ || pendingTerminal_.has_value();
            }
            if (registry != nullptr && delivery_identity != 0) {
                const auto delivery_status =
                    registry->failDelivery(delivery_identity, status, timestamp_ns);
                if (delivery_status == nixl::ucx::notif_delivery_status_t::SUCCESS ||
                    delivery_status == nixl::ucx::notif_delivery_status_t::DUPLICATE_DELIVERY) {
                    return status;
                }
                NIXL_FATAL << "UCX terminal failure could not claim its notification delivery";
            }
            if (terminal_publication_started) {
                return status;
            }
            return failImpl(status, false);
        }

        std::shared_ptr<nixl::ucx::terminal_submission_state_t> state;
        bool terminal = false;
        {
            const std::lock_guard lock(mutex_);
            if (deliveryOutcomeClaimed_) {
                return NIXL_SUCCESS;
            }
            if (deliveryIdentity_ == 0) {
                return terminal_ ? NIXL_SUCCESS : NIXL_ERR_NOT_ALLOWED;
            }
            deliveryOutcomeClaimed_ = true;
            deliveryIdentity_ = 0;
            deliveryRegistry_.reset();
            state = state_;
            terminal = terminal_;
        }
        if (terminal || state == nullptr || state->isTerminal()) {
            return NIXL_SUCCESS;
        }
        if (status == NIXL_SUCCESS) {
            const nixl_status_t receipt_status = state->recordNotificationReceipt(timestamp_ns);
            return receipt_status == NIXL_ERR_NOT_ALLOWED && state->isTerminal() ? NIXL_SUCCESS :
                                                                                   receipt_status;
        }
        const nixl_status_t notification_status =
            state->recordNotificationFailure(status, timestamp_ns);
        if (notification_status == NIXL_SUCCESS) {
            return status;
        }
        if (notification_status != NIXL_ERR_NOT_ALLOWED || state->isTerminal()) {
            return notification_status;
        }
        return failImpl(status, false);
    }

    [[nodiscard]] nixl_status_t
    requestCancellation() noexcept {
        const nixl_status_t status = fail(NIXL_ERR_CANCELED);
        if (status != NIXL_IN_PROG && status != NIXL_ERR_CANCELED && status != NIXL_SUCCESS) {
            return status;
        }
        const std::lock_guard lock(mutex_);
        return terminal_ ? cancellationStatusLocked() : NIXL_IN_PROG;
    }

    [[nodiscard]] nixl_status_t
    drainCancellation() noexcept {
        std::unique_lock lock(mutex_);
        if (!terminal_ && isProgressOwnerThread()) {
            return NIXL_IN_PROG;
        }
        delivered_.wait(lock, [this]() { return terminal_; });
        return cancellationStatusLocked();
    }

    [[nodiscard]] nixl_status_t
    fail(nixl_status_t failure_status) noexcept {
        return failImpl(failure_status, true);
    }

private:
    [[nodiscard]] nixl_status_t
    failImpl(nixl_status_t failure_status, bool fail_delivery) noexcept {
        if (failure_status >= NIXL_SUCCESS) {
            return NIXL_ERR_INVALID_PARAM;
        }
        if (fail_delivery) {
            return claimNotificationDeliveryOutcome(
                failure_status,
                nixl::ucx::terminalProgressTimestampNs(),
                notification_delivery_outcome_source_t::SUBMISSION);
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

        std::unordered_map<nixlUcxWorker *,
                           std::vector<std::shared_ptr<nixl::ucx::ucx_callback_slot_t>>>
            cancellations;
        for (const slot_record_t &record : slots) {
            if (record.slot->requestForCancellation() != nullptr) {
                cancellations[record.worker].push_back(record.slot);
            }
        }
        for (auto &[worker, pending_slots] : cancellations) {
            const nixl_status_t status =
                worker->enqueueContinuation([worker, pending_slots = std::move(pending_slots)]() {
                    for (const auto &slot : pending_slots) {
                        if (void *request = slot->requestForCancellation(); request != nullptr) {
                            worker->reqCancel(request);
                        }
                    }
                    return NIXL_SUCCESS;
                });
            if (status != NIXL_SUCCESS) {
                return status;
            }
        }
        const nixl_status_t cancel_status =
            state->cancel(failure_status, nixl::ucx::terminalProgressTimestampNs());
        if (cancel_status == NIXL_ERR_NOT_ALLOWED && state->isTerminal()) {
            return NIXL_SUCCESS;
        }
        if (cancel_status != NIXL_SUCCESS && cancel_status != failure_status) {
            return cancel_status;
        }
        return failure_status;
    }

public:
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

    [[nodiscard]] std::shared_ptr<nixl::ucx::terminal_submission_state_t>
    getSubmissionState() const noexcept {
        const std::lock_guard lock(mutex_);
        return state_;
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
    publishTerminal(const nixl::ucx::terminal_submission_result_t &result) noexcept override {
        std::optional<nixl::ucx::terminal_submission_result_t> ready;
        {
            const std::lock_guard lock(mutex_);
            if (terminal_ || publishing_ || pendingTerminal_.has_value()) {
                return NIXL_ERR_NOT_ALLOWED;
            }
            nixl::ucx::terminal_submission_result_t accepted = result;
            bool expiry_won = false;
            if (deadlineArmed_) {
                const nixl::ucx::terminal_deadline_key_t key{
                    .handleIdentity = binding_.handleIdentity,
                    .generation = binding_.generation,
                };
                const nixl::ucx::terminal_deadline_status_t retirement =
                    deadlineOwner_->retire(key);
                if (retirement == nixl::ucx::terminal_deadline_status_t::EXPIRY_WON) {
                    expiry_won = true;
                    accepted.status = NIXL_ERR_CANCELED;
                    accepted.diagnostics.terminalStatus = NIXL_ERR_CANCELED;
                } else if (retirement != nixl::ucx::terminal_deadline_status_t::SUCCESS) {
                    NIXL_FATAL << "UCX terminal publication lost its exact deadline";
                }
                deadlineArmed_ = false;
            }
            pendingTerminal_ = accepted;
            if (expiry_won) {
                const nixl::ucx::terminal_deadline_key_t key{
                    .handleIdentity = binding_.handleIdentity,
                    .generation = binding_.generation,
                };
                const nixl::ucx::terminal_deadline_status_t acknowledgement =
                    deadlineOwner_->acknowledgeExpiry(key);
                if (acknowledgement != nixl::ucx::terminal_deadline_status_t::SUCCESS) {
                    NIXL_FATAL << "UCX terminal expiry disposition lost its exact deadline";
                }
            }
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
        const std::shared_ptr<nixlUcxTerminalArm> self = shared_from_this();
        if (completed.status != NIXL_SUCCESS) {
            const nixl_status_t delivery_status = claimNotificationDeliveryOutcome(
                completed.status,
                nixl::ucx::terminalProgressTimestampNs(),
                notification_delivery_outcome_source_t::SUBMISSION);
            if (delivery_status != completed.status) {
                NIXL_FATAL << "UCX terminal publication leaked notification delivery state";
            }
        }
        completed.diagnostics.activeCallbackSlotsAtTerminal = 0;
        completed.diagnostics.continuationDepthAtTerminal = 0;
        completed.diagnostics.terminalPublishTimestampNs = nixl::ucx::terminalProgressTimestampNs();
        nixl_status_t status = completed.status;
        const nixl_xfer_attestation_t snapshot = attestation_->snapshot();
        if (snapshot.generation == binding_.generation) {
            if (status != NIXL_SUCCESS) {
                attestation_->fail(binding_.generation, status, "autonomous terminal failure");
            }
            const nixl_status_t evidence_status =
                attestation_->recordTerminalProgress(binding_.generation, completed.diagnostics);
            if (evidence_status != NIXL_SUCCESS && status == NIXL_SUCCESS) {
                status = evidence_status;
                attestation_->fail(
                    binding_.generation, status, "terminal progress evidence failed");
            }
        }
        {
            const std::lock_guard lock(mutex_);
            terminal_ = true;
            terminalStatus_ = status;
            publishing_ = false;
            state_.reset();
            slots_.clear();
        }
        // Terminal visibility permits immediate request release, so detach first while `self`
        // keeps the arm alive across a sink that releases both public lifetimes.
        if (const std::shared_ptr<nixlUcxTerminalOwnership> ownership = ownership_.lock();
            ownership != nullptr) {
            ownership->detach(self.get());
        }
        sink_->publish({
            .binding = binding_,
            .status = status,
            .nativeTimestampNs = completed.nativeTimestampNs,
        });
        delivered_.notify_all();
        // A failed transfer is terminal data; publishing it completes the owner continuation.
        return NIXL_SUCCESS;
    }

    [[nodiscard]] nixl_status_t
    cancellationStatusLocked() const noexcept {
        return terminalStatus_ == NIXL_ERR_CANCELED ? NIXL_SUCCESS : terminalStatus_;
    }

    [[nodiscard]] bool
    isProgressOwnerThread() const noexcept {
        return std::any_of(workers_.begin(), workers_.end(), [](const nixlUcxWorker *worker) {
            return worker->isProgressOwnerThread();
        });
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
    nixlUcxWorker *const primaryWorker_;
    nixl::ucx::terminal_deadline_owner_t *const deadlineOwner_;
    const std::weak_ptr<nixlUcxTerminalOwnership> ownership_;
    mutable std::mutex mutex_;
    std::condition_variable delivered_;
    std::shared_ptr<nixl::ucx::terminal_submission_state_t> state_;
    std::vector<slot_record_t> slots_;
    std::optional<nixl::ucx::terminal_submission_result_t> pendingTerminal_;
    std::size_t activeSlots_ = 0;
    nixl_status_t terminalStatus_ = NIXL_IN_PROG;
    bool publishing_ = false;
    bool terminal_ = false;
    bool deadlineArmed_ = false;
    bool deliveryOutcomeClaimed_ = false;
    std::weak_ptr<nixl::ucx::notif_delivery_registry_t> deliveryRegistry_;
    uint64_t deliveryIdentity_ = 0;
};

void
nixlUcxTerminalOwnership::detach(const nixlUcxTerminalArm *arm) noexcept {
    const std::lock_guard lock(mutex_);
    if (armed_.get() == arm) {
        armed_.reset();
    }
    if (active_.get() == arm) {
        active_.reset();
    }
}

class nixlUcxTerminalSubscription final : public nixlBackendEventSubscription {
public:
    explicit nixlUcxTerminalSubscription(std::shared_ptr<nixlUcxTerminalArm> arm)
        : arm_(std::move(arm)) {}

    nixl_status_t
    cancel() noexcept override {
        return arm_->requestCancellation();
    }

    void
    queryInventory(nixlBackendEventSubscriptionInventory &inventory) const noexcept override {
        arm_->inventory(inventory);
    }

    nixl_status_t
    drainCancellation() noexcept override {
        return arm_->drainCancellation();
    }

private:
    const std::shared_ptr<nixlUcxTerminalArm> arm_;
};

class nixlUcxCapabilitySink final : public nixl::ucx::notif_route_transition_sink_t {
public:
    explicit nixlUcxCapabilitySink(std::shared_ptr<nixlBackendCapabilityTransitionSink> sink)
        : sink_(std::move(sink)) {}

    void
    publish(const nixl::ucx::notif_route_transition_t &transition) noexcept override {
        nixl_backend_capability_state_t state = nixl_backend_capability_state_t::FAILED;
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
    nixlUcxCapabilitySubscription(std::shared_ptr<nixl::ucx::notif_capability_state_t> state,
                                  nixl::ucx::notif_route_subscription_t subscription,
                                  std::shared_ptr<nixlUcxCapabilitySink> sink)
        : state_(std::move(state)),
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
        const nixl_status_t status =
            notifSubscriptionToNixl(state_->unsubscribeRemoteNotificationState(subscription_));
        if (status == NIXL_SUCCESS) {
            canceled_ = true;
        }
        return status;
    }

    void
    queryInventory(nixlBackendEventSubscriptionInventory &inventory) const noexcept override {
        const nixl::ucx::notif_route_subscription_inventory_t native_inventory =
            state_->querySubscriptionInventory(subscription_);
        inventory = {
            .backendProducers = native_inventory.retainedSubscriptions,
            .activeCallbackSlots = native_inventory.inFlightDeliveries,
        };
    }

private:
    const std::shared_ptr<nixl::ucx::notif_capability_state_t> state_;
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
    nixl::ucx::terminal_deadline_owner_t *deadlineOwner_;
    const std::shared_ptr<nixlUcxTerminalOwnership> terminalOwnership_;

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

        Notif(std::vector<std::uint8_t> wire_frame, ucx_connection_ptr_t remote_connection)
            : frame(std::move(wire_frame)),
              connection(std::move(remote_connection)) {}
    };

    std::optional<Notif> notif;

    nixlUcxBackendReqH(nixlUcxWorker *worker,
                       size_t worker_id,
                       std::shared_ptr<nixlUcxAttestationState> attestation = nullptr,
                       nixl::ucx::terminal_deadline_owner_t *deadline_owner = nullptr)
        : worker_(worker),
          workerId_(worker_id),
          attestation_(attestation != nullptr ?
                           std::move(attestation) :
                           std::make_shared<nixlUcxAttestationState>(
                               allocateIdentity(next_handle_identity, "handle"))),
          deadlineOwner_(deadline_owner),
          terminalOwnership_(std::make_shared<nixlUcxTerminalOwnership>()) {}

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
    beginSubmission(bool has_notification, const std::vector<nixlUcxWorker *> &workers) {
        if (!requests_.empty() || notif.has_value()) {
            return NIXL_ERR_REPOST_ACTIVE;
        }
        std::shared_ptr<nixlUcxTerminalArm> armed;
        {
            const std::lock_guard lock(terminalOwnership_->mutex_);
            if (terminalOwnership_->active_ != nullptr &&
                !terminalOwnership_->active_->isTerminal()) {
                return NIXL_ERR_REPOST_ACTIVE;
            }
            terminalOwnership_->active_.reset();
            if (terminalOwnership_->armed_ != nullptr &&
                terminalOwnership_->armed_->binding().generation !=
                    attestation_->getGeneration() + 1) {
                return NIXL_ERR_NOT_ALLOWED;
            }
            armed = terminalOwnership_->armed_;
            if (armed == nullptr && has_notification) {
                if (workers.empty() ||
                    std::any_of(workers.begin(), workers.end(), [](nixlUcxWorker *worker) {
                        return worker == nullptr || !worker->hasProgressOwner();
                    })) {
                    return NIXL_ERR_NOT_SUPPORTED;
                }
                const nixl_xfer_attestation_t snapshot = attestation_->snapshot();
                armed = std::make_shared<nixlUcxTerminalArm>(
                    nixlBackendTransferEventBinding{
                        .handleIdentity = snapshot.handleIdentity,
                        .generation = snapshot.generation + 1,
                    },
                    std::make_shared<nixlUcxNoopTransferSink>(),
                    attestation_,
                    workers,
                    worker_,
                    deadlineOwner_,
                    terminalOwnership_);
                terminalOwnership_->armed_ = armed;
            }
        }
        // Terminality and an empty request vector leave only the preceding generation's
        // endpoint set, which must not become the next generation's flush set.
        connections_.clear();

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
            const std::lock_guard lock(terminalOwnership_->mutex_);
            if (terminalOwnership_->armed_ != armed) {
                attestation_->fail(attestation_->getGeneration(),
                                   NIXL_ERR_BACKEND,
                                   "terminal subscription ownership changed during activation");
                return NIXL_ERR_BACKEND;
            }
            terminalOwnership_->armed_.reset();
            terminalOwnership_->active_ = std::move(armed);
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
            binding, sink, attestation_, workers, worker_, deadlineOwner_, terminalOwnership_);
        {
            const std::lock_guard lock(terminalOwnership_->mutex_);
            if (terminalOwnership_->armed_ != nullptr ||
                (terminalOwnership_->active_ != nullptr &&
                 !terminalOwnership_->active_->isTerminal())) {
                return NIXL_ERR_NOT_ALLOWED;
            }
            terminalOwnership_->active_ = nullptr;
            terminalOwnership_->armed_ = arm;
        }
        subscription = std::make_unique<nixlUcxTerminalSubscription>(arm);
        return NIXL_SUCCESS;
    }

    [[nodiscard]] std::shared_ptr<nixlUcxTerminalArm>
    getActiveTerminal() const noexcept {
        const std::lock_guard lock(terminalOwnership_->mutex_);
        return terminalOwnership_->active_;
    }

    void
    setActiveTerminal(const std::shared_ptr<nixlUcxTerminalArm> &terminal) noexcept {
        const std::lock_guard lock(terminalOwnership_->mutex_);
        terminalOwnership_->active_ = terminal;
    }

    [[nodiscard]] bool
    hasAutonomousTerminal() const noexcept {
        return getActiveTerminal() != nullptr;
    }

    [[nodiscard]] bool
    terminalLifecycleDrained() const noexcept {
        const std::lock_guard lock(terminalOwnership_->mutex_);
        return terminalOwnership_->armed_ == nullptr &&
            (terminalOwnership_->active_ == nullptr || terminalOwnership_->active_->isTerminal());
    }

    [[nodiscard]] nixlBackendTransferEventBinding
    nextSubmissionBinding() const noexcept {
        const nixl_xfer_attestation_t attestation = attestation_->snapshot();
        return {
            .handleIdentity = attestation.handleIdentity,
            .generation = attestation.generation + 1,
        };
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
            requests_.push_back({req, kind, endpoint_identity, attestation_->getGeneration()});
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

    [[nodiscard]] nixl_status_t
    clearAutonomousPostingState() {
        if (!requests_.empty() || notif.has_value()) {
            return NIXL_ERR_REPOST_ACTIVE;
        }
        connections_.clear();
        setActiveTerminal(nullptr);
        return NIXL_SUCCESS;
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
                    const nixl_status_t flush_status =
                        attestation_->completeFlush(pending.generation, pending.endpointIdentity);
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
                    attestation_->fail(pending.generation, ret, "UCX transfer request failed");
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
        NIXL_ASSERT_ALWAYS(thread_ == nullptr);
    }

    void
    start() {
        NIXL_ASSERT_ALWAYS(thread_ == nullptr);
        const auto started = std::make_shared<std::promise<void>>();
        auto active = started->get_future();
        std::unique_ptr<std::thread> thread = std::make_unique<std::thread>([this, started]() {
            tlsThread() = this;
            for (nixlUcxWorker *worker : workers_) {
                const nixl_status_t owner_status = worker->claimProgressOwner();
                if (owner_status != NIXL_SUCCESS) {
                    NIXL_FATAL << "UCX progress thread failed to claim worker ownership: "
                               << owner_status;
                }
            }
            started->set_value();
            run();
        });
        thread_ = std::move(thread);
        active.wait();
    }

    virtual void
    join() {
        NIXL_ASSERT_ALWAYS(thread_ != nullptr);
        thread_->join();
        thread_.reset();
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

    [[nodiscard]] virtual bool
    terminalLifecycleDrained() const noexcept {
        return engine_->terminalDeadlinesDrained() &&
            std::all_of(workers_.begin(), workers_.end(), [](const nixlUcxWorker *worker) {
                   return worker->terminalLifecycleDrained();
               });
    }

    friend std::ostream &
    operator<<(std::ostream &os, const nixlUcxThread &thread) {
        return os << "thread " << &thread << "{engine: " << thread.engine_ << ", worker_ids: ["
                  << absl::StrJoin(thread.workerIds_, ",") << "]}";
    }

protected:
    [[nodiscard]] bool
    isActive() const noexcept {
        return thread_ != nullptr;
    }

    virtual void
    run() = 0;

private:
    const nixlUcxEngine *engine_;
    std::vector<nixlUcxWorker *> workers_;
    std::vector<size_t> workerIds_;
    std::unique_ptr<std::thread> thread_;
};

class nixlUcxSharedThread : public nixlUcxThread {
public:
    nixlUcxSharedThread(const nixlUcxEngine *engine, size_t num_workers, nixlTime::us_t delay)
        : nixlUcxThread(engine, num_workers) {
        // TODO: We need delay to manual periodic wakeup/polling as a temporary
        // workaround for UCX bug (poll wouldn't wake up some fds in particular
        // circumstances)

        // This will ensure that the resulting delay is at least 1ms and fits into int in order for
        // it to be compatible with poll()
        int delay_us = std::min((int)delay, std::numeric_limits<int>::max());
        delay_ = std::chrono::ceil<std::chrono::milliseconds>(std::chrono::microseconds(delay_us));

        pollFds_.resize(num_workers + 1);
        if (pipe(controlPipe_) < 0) {
            throw std::runtime_error("Couldn't create progress thread control pipe");
        }
        pollFds_.back() = {controlPipe_[0], POLLIN, 0};
    }

    ~nixlUcxSharedThread() override {
        if (isActive()) {
            join();
        }
        close(controlPipe_[0]);
        close(controlPipe_[1]);
    }

    void
    join() override {
        const char signal = 'X';
        ssize_t write_status = 0;
        do {
            write_status = write(controlPipe_[1], &signal, sizeof(signal));
        } while (write_status < 0 && errno == EINTR);
        if (write_status != sizeof(signal)) {
            NIXL_PFATAL << "Failed to stop UCX shared progress owner";
        }
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
        bool stop_requested = false;
        while (true) {
            for (size_t i = 0; i < pollFds_.size() - 1; i++) {
                if (!(pollFds_[i].revents & POLLIN) && !timeout && !stop_requested) {
                    continue;
                }
                pollFds_[i].revents = 0;
                nixlUcxWorker *worker = getWorkers()[i];
                while (true) {
                    worker->progressLoop();
                    const nixl_status_t drain_status = worker->drainContinuationsOnOwner();
                    if (drain_status != NIXL_SUCCESS) {
                        NIXL_FATAL << "UCX shared progress owner failed to drain continuations: "
                                   << drain_status;
                    }
                    const nixl_status_t arm_status = worker->arm();
                    if (arm_status == NIXL_IN_PROG) {
                        continue;
                    }
                    if (arm_status != NIXL_SUCCESS) {
                        NIXL_FATAL << "UCX shared progress owner failed to arm its worker: "
                                   << arm_status;
                    }
                    break;
                }
            }
            timeout = false;

            if (stop_requested && terminalLifecycleDrained()) {
                break;
            }

            int ret;
            while ((ret = poll(pollFds_.data(), pollFds_.size(), delay_.count())) < 0 &&
                   errno == EINTR) {
                NIXL_PTRACE << "Call to poll() was interrupted, retrying";
            }
            if (ret < 0) {
                NIXL_PFATAL << "UCX shared progress poll failed";
            }

            if (!ret) {
                timeout = true;
            } else if (std::any_of(pollFds_.begin(), pollFds_.end(), [](const pollfd &descriptor) {
                           return (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0;
                       })) {
                NIXL_FATAL << "UCX shared progress poll reported a failed descriptor";
            } else if (pollFds_.back().revents & POLLIN) {
                pollFds_.back().revents = 0;

                char signal;
                const ssize_t read_status = read(pollFds_.back().fd, &signal, sizeof(signal));
                if (read_status != sizeof(signal)) {
                    NIXL_PFATAL << "UCX shared progress control read failed";
                }
                stop_requested = true;
                timeout = true;
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
    closeNotificationIngress();
    drainNotificationDeliveries();
    drainTerminalDeadlines();
    thread_->join();
    closeTerminalDeadlines();
    if (!notificationDeliveriesDrained()) {
        NIXL_FATAL << "UCX notification delivery registry leaked at shutdown";
    }
    if (!notificationIngressDrained()) {
        NIXL_FATAL << "UCX notification ingress registry leaked at shutdown";
    }
}

void
nixlUcxThreadEngine::appendNotif(nixlAuthenticatedNotification &&notification) const {
    const std::lock_guard lock(notifMutex_);
    (void)notifQueue_.push(std::move(notification));
}

nixl_status_t
nixlUcxThreadEngine::admitNotif(nixlAuthenticatedNotification &&notification) const {
    const std::lock_guard lock(notifMutex_);
    return notifQueue_.push(std::move(notification));
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
    nixl_status_t completion_status = status;
    if (completion_status == NIXL_SUCCESS && hasAutonomousTerminal()) {
        completion_status = clearAutonomousPostingState();
    }
    if (completion_status != NIXL_SUCCESS) {
        nixlUcxBackendReqH::release();
        sharedState_->status.store(completion_status);
    }
    sharedState_->pendingReqs.fetch_sub(1);
    NIXL_TRACE << *this << " completed with status: " << completion_status << ", " << *sharedState_;
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
                                size_t num_chunks,
                                nixl::ucx::terminal_deadline_owner_t *deadline_owner)
        : nixlUcxBackendReqH(worker, worker_id, nullptr, deadline_owner),
          sharedState_(std::make_shared<nixlUcxBackendSharedState>()),
          chunkSize_(chunk_size) {
        sharedState_->chunks.reserve(num_chunks);
        for (size_t index = 0; index < num_chunks; ++index) {
            sharedState_->chunks.push_back(std::make_unique<nixlUcxChunkBackendReqH>());
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

    void
    failUnstartedChunk(size_t idx, nixl_status_t status) {
        NIXL_ASSERT_ALWAYS(status != NIXL_SUCCESS && status != NIXL_IN_PROG);
        nixlUcxChunkBackendReqH *const chunk = sharedState_->chunks[idx].get();
        chunk->startXfer(sharedState_, nullptr, UINT64_MAX, getAttestationState(), {});
        chunk->complete(status);
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
    // One task owns one composite chunk until its exact worker has posted it. The 4096-task bound
    // permits thousands of descriptors on one owner while keeping overload process-fatal instead
    // of allowing a native-control backlog whose request and terminal lifetimes cannot be dropped.
    static constexpr size_t taskCapacity = 4096;

    explicit nixlUcxDedicatedThread(nixlUcxEngine *engine) : nixlUcxThread(engine, 1) {
        if (pipe2(controlPipe_, O_CLOEXEC | O_NONBLOCK) < 0) {
            throw std::runtime_error("Couldn't create dedicated progress thread control pipe");
        }
    }

    ~nixlUcxDedicatedThread() override {
        if (isActive()) {
            join();
        }
        close(controlPipe_[0]);
        close(controlPipe_[1]);
    }

    static nixlUcxDedicatedThread *
    getDedicatedThread() {
        return (nixlUcxDedicatedThread *)tlsThread();
    }

    void
    addRequest(nixlUcxChunkBackendReqH *handle) {
        requests_.push_back(handle);
        liveRequests_.fetch_add(1, std::memory_order_release);
    }

    [[nodiscard]] nixl_status_t
    post(nixl::ucx::ucx_worker_continuation_queue_t::continuation_t task) {
        try {
            const std::lock_guard lock(tasksMutex_);
            if (!acceptingTasks_) {
                return NIXL_ERR_NOT_ALLOWED;
            }
            if (admittedTasks_ >= taskCapacity) {
                NIXL_FATAL << "UCX dedicated progress owner task queue exceeded its process bound";
            }
            tasks_.push_back(std::move(task));
            ++admittedTasks_;
            wakeOwner();
        }
        catch (const std::bad_alloc &) {
            return NIXL_ERR_BACKEND;
        }
        return NIXL_SUCCESS;
    }

    void
    closeAdmission() {
        {
            const std::lock_guard lock(tasksMutex_);
            acceptingTasks_ = false;
            stopRequested_ = true;
            wakeOwner();
        }
    }

    void
    join() override {
        closeAdmission();
        nixlUcxThread::join();
        NIXL_ASSERT_ALWAYS(terminalLifecycleDrained());
    }

    [[nodiscard]] bool
    terminalLifecycleDrained() const noexcept override {
        const std::lock_guard lock(tasksMutex_);
        return admittedTasks_ == 0 && tasks_.empty() &&
            liveRequests_.load(std::memory_order_acquire) == 0 &&
            nixlUcxThread::terminalLifecycleDrained();
    }

protected:
    void
    run() override {
        NIXL_DEBUG << "dedicated " << *this << " running";
        nixlUcxWorker *const worker = getWorkers()[0];
        std::array<pollfd, 2> poll_fds = {
            pollfd{worker->getEfd(), POLLIN, 0},
            pollfd{controlPipe_[0], POLLIN, 0},
        };
        while (true) {
            std::deque<nixl::ucx::ucx_worker_continuation_queue_t::continuation_t> ready;
            {
                const std::lock_guard lock(tasksMutex_);
                ready.swap(tasks_);
            }
            for (auto &task : ready) {
                const nixl_status_t task_status = task();
                if (task_status != NIXL_SUCCESS) {
                    NIXL_FATAL << "UCX dedicated progress owner task failed after admission: "
                               << task_status;
                }
                const std::lock_guard lock(tasksMutex_);
                NIXL_ASSERT_ALWAYS(admittedTasks_ > 0);
                --admittedTasks_;
            }

            worker->progressLoop();
            const nixl_status_t drain_status = worker->drainContinuationsOnOwner();
            if (drain_status != NIXL_SUCCESS) {
                NIXL_FATAL << "UCX dedicated progress owner failed to drain continuations: "
                           << drain_status;
            }
            for (auto it = requests_.begin(); it != requests_.end();) {
                nixl_status_t status = (*it)->status();
                if (status != NIXL_IN_PROG) {
                    NIXL_TRACE << "dedicated " << *this << " completing " << *(*it)
                               << " with status: " << status;
                    (*it)->complete(status);
                    it = requests_.erase(it);
                    const std::size_t previous =
                        liveRequests_.fetch_sub(1, std::memory_order_acq_rel);
                    NIXL_ASSERT_ALWAYS(previous > 0);
                } else {
                    ++it;
                }
            }

            bool stop_requested = false;
            {
                const std::lock_guard lock(tasksMutex_);
                stop_requested = stopRequested_ && tasks_.empty() && admittedTasks_ == 0;
            }
            const bool worker_lifecycle_drained = worker->terminalLifecycleDrained();
            if (stop_requested && requests_.empty() && worker_lifecycle_drained) {
                break;
            }

            // Outstanding transport and terminal-callback lifetimes need active UCX progress.
            // Eventfd readiness is an idle wakeup mechanism; TCP transmit progression does not
            // promise another edge after every incomplete progress call.
            if (!requests_.empty() || !worker_lifecycle_drained) {
                continue;
            }

            const nixl_status_t arm_status = worker->arm();
            if (arm_status == NIXL_IN_PROG) {
                continue;
            }
            if (arm_status != NIXL_SUCCESS) {
                NIXL_FATAL << "UCX dedicated progress owner failed to arm its worker: "
                           << arm_status;
            }

            int poll_status;
            while ((poll_status = poll(poll_fds.data(), poll_fds.size(), -1)) < 0 &&
                   errno == EINTR) {
                NIXL_PTRACE << "Dedicated progress poll was interrupted, retrying";
            }
            if (poll_status <= 0) {
                NIXL_PFATAL << "UCX dedicated progress poll failed";
            }
            if (std::any_of(poll_fds.begin(), poll_fds.end(), [](const pollfd &descriptor) {
                    return (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0;
                })) {
                NIXL_FATAL << "UCX dedicated progress poll reported a failed descriptor";
            }
            if ((poll_fds[1].revents & POLLIN) != 0) {
                poll_fds[1].revents = 0;
                drainWakeups();
            }
            poll_fds[0].revents = 0;
        }

        NIXL_DEBUG << "dedicated " << *this << " exiting";
    }

private:
    void
    wakeOwner() const noexcept {
        const char signal = 'W';
        ssize_t written = 0;
        do {
            written = write(controlPipe_[1], &signal, sizeof(signal));
        } while (written < 0 && errno == EINTR);
        if (written == sizeof(signal) ||
            (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))) {
            return;
        }
        NIXL_PFATAL << "Failed to wake UCX dedicated progress owner";
    }

    void
    drainWakeups() const noexcept {
        std::array<char, 64> signals;
        while (true) {
            const ssize_t read_status = read(controlPipe_[0], signals.data(), signals.size());
            if (read_status > 0) {
                continue;
            }
            if (read_status < 0 && errno == EINTR) {
                continue;
            }
            if (read_status < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                return;
            }
            NIXL_PFATAL << "Failed to drain UCX dedicated progress wakeups";
        }
    }

    int controlPipe_[2];
    mutable std::mutex tasksMutex_;
    std::deque<nixl::ucx::ucx_worker_continuation_queue_t::continuation_t> tasks_;
    bool acceptingTasks_ = true;
    bool stopRequested_ = false;
    size_t admittedTasks_ = 0;
    std::atomic<size_t> liveRequests_ = 0;
    std::vector<nixlUcxChunkBackendReqH *> requests_;
};

nixlUcxThreadPoolEngine::nixlUcxThreadPoolEngine(const nixlBackendInitParams &init_params)
    : nixlUcxThreadPoolEngine(init_params, std::nullopt) {}

nixlUcxThreadPoolEngine::nixlUcxThreadPoolEngine(
    const nixlBackendInitParams &init_params,
    std::optional<qualification_constructor_failure_point_t> failure_point)
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
        if (failure_point ==
            qualification_constructor_failure_point_t::AFTER_SHARED_PROGRESS_OWNER_STARTED) {
            throw std::runtime_error(
                "injected failure after UCX shared progress owner construction");
        }
    }

    if (num_threads > 0) {
        dedicatedThreads_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            size_t worker_id = numSharedWorkers_ + i;
            dedicatedThreads_.emplace_back(std::make_unique<nixlUcxDedicatedThread>(this));
            dedicatedThreads_.back()->addWorker(getWorker(worker_id).get(), worker_id);
            dedicatedThreads_.back()->start();
            if (i == 0 &&
                failure_point ==
                    qualification_constructor_failure_point_t::
                        AFTER_FIRST_DEDICATED_PROGRESS_OWNER_STARTED) {
                throw std::runtime_error(
                    "injected failure after first UCX dedicated progress owner construction");
            }
        }
    }
}

nixlUcxThreadPoolEngine::~nixlUcxThreadPoolEngine() {
    closeNotificationIngress();
    drainNotificationDeliveries();
    drainTerminalDeadlines();
    for (auto &thread : dedicatedThreads_) {
        static_cast<nixlUcxDedicatedThread *>(thread.get())->closeAdmission();
    }
    for (auto &thread : dedicatedThreads_) {
        thread->join();
    }
    if (sharedThread_) {
        sharedThread_->join();
    }
    closeTerminalDeadlines();
    if (!notificationDeliveriesDrained()) {
        NIXL_FATAL << "UCX notification delivery registry leaked at shutdown";
    }
    if (!notificationIngressDrained()) {
        NIXL_FATAL << "UCX notification ingress registry leaked at shutdown";
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
        getWorker(worker_id).get(), worker_id, chunk_size, num_chunks, getTerminalDeadlineOwner());
    const nixl_status_t status =
        prepareHandleAttestation(comp_handle, operation, local, remote, remote_agent, opt_args);
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
        auto *const target_thread = static_cast<nixlUcxDedicatedThread *>(
            dedicatedThreads_[i % dedicatedThreads_.size()].get());
        const nixl_status_t post_status = target_thread->post([&, i, target_thread]() {
            nixlUcxDedicatedThread *thread = nixlUcxDedicatedThread::getDedicatedThread();
            NIXL_ASSERT_ALWAYS(thread == target_thread);

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
            return NIXL_SUCCESS;
        });
        if (post_status != NIXL_SUCCESS) {
            status.store(post_status);
            comp_handle->failUnstartedChunk(i, post_status);
            if (remaining.fetch_sub(1) == 1) {
                promise.set_value();
            }
        }
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

nixl_status_t
nixlUcxThreadPoolEngine::admitNotif(nixlAuthenticatedNotification &&notification) const {
    const std::lock_guard lock(notifMutex_);
    return notifQueue_.push(std::move(notification));
}

void
nixlUcxThreadPoolEngine::poisonNotifs() const {
    const std::lock_guard lock(notifMutex_);
    notifQueue_.poison();
}

nixl_status_t
nixlUcxThreadPoolEngine::getNotifs(notif_list_t &notif_list) {
    if (!sharedThread_) {
        for (size_t worker_id = 0; worker_id < numSharedWorkers_; ++worker_id) {
            getWorker(worker_id)->progressLoop();
        }
    }

    const std::lock_guard lock(notifMutex_);
    return notifQueue_.drainLegacy(notif_list);
}

nixl_status_t
nixlUcxThreadPoolEngine::getAuthenticatedNotifs(authenticated_notif_list_t &notif_list) {
    if (!sharedThread_) {
        for (size_t worker_id = 0; worker_id < numSharedWorkers_; ++worker_id) {
            getWorker(worker_id)->progressLoop();
        }
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

    if (custom_params->count("device_list") != 0) {
        devs = absl::StrSplit((*custom_params)["device_list"], ", ");
    }

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

    const bool has_internal_progress = init_params.enableProgTh || num_threads > 0;
    uc = std::make_unique<nixlUcxContext>(devs,
                                          has_internal_progress,
                                          num_workers,
                                          init_params.syncMode,
                                          num_device_channels,
                                          engine_config);

    uc->warnAboutHardwareSupportMismatch();

    for (size_t i = 0; i < num_workers; i++) {
        uws.emplace_back(std::make_unique<nixlUcxWorker>(*uc, err_handling_mode));
    }
    terminalDeadlineOwner_ = std::make_unique<nixl::ucx::terminal_deadline_owner_t>(
        nixlUcxNotificationQueue::maxPendingNotifications,
        [this](nixl_status_t status) noexcept { failTerminalLifecycle(status); });

    localConnectionMetadata_.backendIncarnation = nixl::ucx::generateConnectionMetadataUuid();
    localConnectionMetadata_.workers.reserve(uws.size());
    const size_t shared_worker_count = num_workers - num_threads;
    for (size_t worker_id = 0; worker_id < uws.size(); ++worker_id) {
        const auto &worker = uws[worker_id];
        localConnectionMetadata_.workers.push_back({
            .incarnation = nixl::ucx::generateConnectionMetadataUuid(),
            .endpointAddress = worker->epAddr(),
            .supportsAttachedReceipt = init_params.enableProgTh || worker_id >= shared_worker_count,
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
    notifState_ = std::make_shared<nixl::ucx::notif_capability_state_t>(
        localAgentIncarnationUuid_,
        localConnectionMetadata_.backendIncarnation,
        std::move(local_worker_incarnations));
    endpointFailureState_ =
        std::make_shared<nixl::ucx::notif_endpoint_failure_state_t>(notifState_);
    deliveryRegistry_ = std::make_shared<nixl::ucx::notif_delivery_registry_t>(
        nixlUcxNotificationQueue::maxPendingNotifications);
    ingressRegistry_ = std::make_shared<nixl::ucx::notif_ingress_registry_t>(
        nixlUcxNotificationQueue::maxPendingNotifications);

    notifCallbackContexts_.reserve(uws.size());
    for (size_t worker_id = 0; worker_id < uws.size(); ++worker_id) {
        notifCallbackContexts_.push_back({.engine = this, .workerId = worker_id});
        const int callback_status = uws[worker_id]->regAmCallback(
            nixl::ucx::am_cb_op_t::NOTIF_STR, notifAmCb, &notifCallbackContexts_.back());
        if (callback_status != UCS_OK) {
            throw std::runtime_error("UCX notification callback registration failed");
        }
    }
}

nixl_mem_list_t
nixlUcxEngine::getSupportedMems() const {
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
    closeTerminalDeadlines();
    if (deliveryRegistry_->inventory().outstanding != 0) {
        NIXL_FATAL << "UCX base engine destruction observed live notification delivery state";
    }
    const nixl::ucx::notif_ingress_inventory_t ingress = ingressRegistry_->inventory();
    if (ingress.pending != 0 || ingress.admitting != 0 || ingress.committed != 0 ||
        ingress.replaying != 0 || ingress.quarantined != 0) {
        NIXL_FATAL << "UCX base engine destruction observed live notification ingress state";
    }
    tlsSharedWorkerMap().erase(this);
    remote_connection_map_t retired_connections = detachRemoteConnections();
    // Endpoint teardown enters UCX. The connection registry lock must never be held while the
    // retired map releases its final endpoint owners because notification callbacks acquire the
    // same registry lock from inside UCX progress.
    retired_connections.clear();
}

void
nixlUcxEngine::failTerminalLifecycle(nixl_status_t status) noexcept {
    if (status >= NIXL_SUCCESS) {
        status = NIXL_ERR_BACKEND;
    }
    nixl_status_t expected = NIXL_SUCCESS;
    if (!terminalLifecycleFatal_.compare_exchange_strong(
            expected, status, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return;
    }
    for (const std::unique_ptr<nixlUcxWorker> &worker : uws) {
        static_cast<void>(worker->getContinuationQueue()->fail(status));
    }
}

void
nixlUcxEngine::drainTerminalDeadlines() noexcept {
    if (terminalDeadlineOwner_ == nullptr) {
        return;
    }
    const nixl_status_t status = terminalDeadlineOwner_->beginShutdown();
    if (status != NIXL_SUCCESS) {
        NIXL_FATAL << "UCX terminal deadline owner failed to begin shutdown drain";
    }
}

void
nixlUcxEngine::closeTerminalDeadlines() noexcept {
    if (terminalDeadlineOwner_ == nullptr) {
        return;
    }
    const nixl_status_t status = terminalDeadlineOwner_->close();
    const nixl::ucx::terminal_deadline_snapshot_t snapshot = terminalDeadlineOwner_->snapshot();
    if (status != NIXL_SUCCESS || snapshot.inventory.active != 0 || !snapshot.activeKeys.empty()) {
        NIXL_FATAL << "UCX terminal deadline owner failed to drain exact native obligations";
    }
}

/****************************************
 * Connection management
 *****************************************/

nixl_status_t
nixlUcxEngine::checkConn(const std::string &remote_agent) {
    return getConnection(remote_agent) != nullptr ? NIXL_SUCCESS : NIXL_ERR_NOT_FOUND;
}

nixl_status_t
nixlUcxEngine::getConnInfo(std::string &str) const {
    return nixl::ucx::encodeConnectionMetadata(localConnectionMetadata_, str) ==
            nixl::ucx::connection_metadata_status_t::SUCCESS ?
        NIXL_SUCCESS :
        NIXL_ERR_BACKEND;
}

nixl_status_t
nixlUcxEngine::connect(const std::string &remote_agent) {
    if (remote_agent == localAgent) {
        std::string local_conn_info;
        const nixl_status_t status = getConnInfo(local_conn_info);
        if (status != NIXL_SUCCESS) {
            return status;
        }
        return loadRemoteConnInfo(remote_agent, local_conn_info);
    }

    return getConnection(remote_agent) == nullptr ? NIXL_ERR_NOT_FOUND : NIXL_SUCCESS;
}

nixl_status_t
nixlUcxEngine::disconnect(const std::string &remote_agent) {
    ucx_connection_ptr_t connection;
    {
        const std::lock_guard lock(connectionMutex_);
        const auto it = remoteConnMap.find(remote_agent);
        if (it == remoteConnMap.end()) {
            return NIXL_ERR_NOT_FOUND;
        }
        connection = it->second;
    }
    const uint64_t connection_identity = connection->getIdentity();
    const nixl_status_t route_status =
        notifStateToNixl(endpointFailureState_->failRemoteConnection(connection_identity));
    const auto delivery_status = deliveryRegistry_->failConnection(
        connection_identity, NIXL_ERR_REMOTE_DISCONNECT, nixl::ucx::terminalProgressTimestampNs());
    const auto ingress_status = ingressRegistry_->failConnection(connection_identity);
    if (delivery_status != nixl::ucx::notif_delivery_status_t::SUCCESS ||
        ingress_status != nixl::ucx::notif_ingress_status_t::SUCCESS) {
        return NIXL_ERR_BACKEND;
    }
    if (route_status != NIXL_SUCCESS && route_status != NIXL_ERR_NOT_FOUND) {
        return route_status;
    }
    const auto drain_status = ingressRegistry_->drainConnection(
        connection_identity, std::chrono::nanoseconds(nixl::ucx::nativeTransferTimeoutNs));
    if (drain_status != nixl::ucx::notif_ingress_status_t::SUCCESS) {
        failTerminalLifecycle(NIXL_ERR_BACKEND);
        NIXL_FATAL << "UCX connection destruction timed out with committed receipt authority";
    }
    const ucx_connection_ptr_t retired_connection =
        detachRemoteConnection(remote_agent, connection);
    if (retired_connection == nullptr) {
        return NIXL_ERR_NOT_FOUND;
    }
    if (ingressRegistry_->retireConnection(connection_identity) !=
        nixl::ucx::notif_ingress_status_t::SUCCESS) {
        NIXL_FATAL << "UCX connection destruction raced a committed receipt obligation";
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxEngine::queryRemoteAgentAuthority(const std::string &remote_agent,
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
        binding.authority.generation == 0 || binding.authority.connectionIdentity == 0 ||
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

    nixl::ucx::notif_remote_binding_t remote_binding = {
        .route = route,
        .connectionIdentity = connection->getIdentity(),
    };
    remote_binding.workers.reserve(connection->metadata_.workers.size());
    for (size_t worker_index = 0; worker_index < connection->metadata_.workers.size();
         ++worker_index) {
        remote_binding.workers.push_back({
            .incarnation = connection->metadata_.workers[worker_index].incarnation,
            .endpointIdentity =
                connection->eps[worker_index % connection->eps.size()]->getIdentity(),
        });
    }

    nixl::ucx::notif_route_snapshot_t snapshot;
    const nixl::ucx::notif_state_status_t bind_status = endpointFailureState_->bindRemoteAgent(
        {
            .route = route,
            .remoteAgent = binding.remoteAgent,
            .connectionIdentity = connection->getIdentity(),
        },
        remote_binding,
        connection->endpointFailureObserved_,
        snapshot);
    if (bind_status != nixl::ucx::notif_state_status_t::SUCCESS) {
        return connection->endpointFailureObserved() ? NIXL_ERR_REMOTE_DISCONNECT :
                                                       notifStateToNixl(bind_status);
    }

    nixl::ucx::notif_wire_envelope_t offer;
    const auto &worker = localConnectionMetadata_.workers.front().incarnation;
    if (notifState_->makeOffer(route, worker, offer) == nixl::ucx::notif_state_status_t::SUCCESS) {
        const nixl_status_t send_status = sendControlFrame(offer, connection->getIdentity(), 0);
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
    const nixl_status_t state_status = notifStateToNixl(notifState_->retireRemoteAgent(route));
    const auto delivery_status = deliveryRegistry_->failRoute(
        route, NIXL_ERR_NOT_ALLOWED, nixl::ucx::terminalProgressTimestampNs());
    if (delivery_status != nixl::ucx::notif_delivery_status_t::SUCCESS) {
        NIXL_FATAL << "UCX route retirement failed to drain notification deliveries";
    }
    const auto ingress_status = ingressRegistry_->failRoute(route);
    if (ingress_status != nixl::ucx::notif_ingress_status_t::SUCCESS) {
        NIXL_FATAL << "UCX route retirement failed to fence notification ingress";
    }
    const auto drain_status = ingressRegistry_->drainRoute(
        route, std::chrono::nanoseconds(nixl::ucx::nativeTransferTimeoutNs));
    if (drain_status != nixl::ucx::notif_ingress_status_t::SUCCESS) {
        failTerminalLifecycle(NIXL_ERR_BACKEND);
        NIXL_FATAL << "UCX route retirement timed out with committed receipt authority";
    }
    if (ingressRegistry_->retireRoute(route) != nixl::ucx::notif_ingress_status_t::SUCCESS) {
        NIXL_FATAL << "UCX route retirement raced a committed receipt obligation";
    }
    return state_status;
}

nixl_status_t
nixlUcxEngine::queryRemoteNotificationState(const nixlRemoteAgentBinding &binding) const {
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
        notifState_->subscribeRemoteNotificationState(route, adapter, native_subscription));
    if (subscribe_status != NIXL_SUCCESS) {
        return subscribe_status;
    }
    subscription =
        std::make_unique<nixlUcxCapabilitySubscription>(notifState_, native_subscription, adapter);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxEngine::loadRemoteConnInfo(const std::string &remote_agent,
                                  const std::string &remote_conn_info) {
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
        allocateIdentity(next_connection_identity, "connection"), std::move(remote_metadata));
    for (size_t worker_id = 0; worker_id < uws.size(); ++worker_id) {
        const auto &remote_worker =
            conn->metadata_.workers[worker_id % conn->metadata_.workers.size()];
        const std::shared_ptr<nixl::ucx::ucx_worker_continuation_queue_t> continuations =
            uws[worker_id]->getContinuationQueue();
        const std::weak_ptr<nixlUcxConnection> weak_connection = conn;
        const std::shared_ptr<nixl::ucx::notif_endpoint_failure_state_t> failure_state =
            endpointFailureState_;
        const std::shared_ptr<nixl::ucx::notif_delivery_registry_t> delivery_registry =
            deliveryRegistry_;
        const std::shared_ptr<nixl::ucx::notif_ingress_registry_t> ingress_registry =
            ingressRegistry_;
        std::unique_ptr<nixlUcxEp> result = uws[worker_id]->connect(
            remote_worker.endpointAddress.data(),
            remote_worker.endpointAddress.size(),
            [continuations,
             failure_state,
             delivery_registry,
             ingress_registry,
             weak_connection]() noexcept {
                std::shared_ptr<nixlUcxConnection> connection = weak_connection.lock();
                if (connection == nullptr || !connection->claimEndpointFailure()) {
                    return;
                }
                const uint64_t connection_identity = connection->getIdentity();
                const nixl_status_t enqueue_status =
                    continuations->enqueueProducer([failure_state,
                                                    delivery_registry,
                                                    ingress_registry,
                                                    connection = std::move(connection),
                                                    connection_identity]() noexcept {
                        // Retaining the connection keeps every endpoint alive until its exact
                        // routes have reached a terminal state.
                        static_cast<void>(connection);
                        const nixl_status_t state_status = notifStateToNixl(
                            failure_state->failRemoteConnection(connection_identity));
                        const auto delivery_status = delivery_registry->failConnection(
                            connection_identity,
                            NIXL_ERR_REMOTE_DISCONNECT,
                            nixl::ucx::terminalProgressTimestampNs());
                        const auto ingress_status =
                            ingress_registry->failConnection(connection_identity);
                        return state_status != NIXL_SUCCESS ||
                                delivery_status != nixl::ucx::notif_delivery_status_t::SUCCESS ||
                                ingress_status != nixl::ucx::notif_ingress_status_t::SUCCESS ?
                            NIXL_ERR_REMOTE_DISCONNECT :
                            NIXL_SUCCESS;
                    });
                if (enqueue_status != NIXL_SUCCESS) {
                    static_cast<void>(continuations->fail(enqueue_status));
                }
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
nixl_status_t
nixlUcxEngine::registerMem(const nixlBlobDesc &mem,
                           const nixl_mem_t &nixl_mem,
                           nixlBackendMD *&out) {
    auto priv = std::make_unique<nixlUcxPrivateMetadata>();

    // TODO: Add nixl_mem check?
    const int ret = uc->memReg((void *)mem.addr, mem.len, priv->mem, nixl_mem);
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

nixl_status_t
nixlUcxEngine::deregisterMem(nixlBackendMD *meta) {
    nixlUcxPrivateMetadata *priv = (nixlUcxPrivateMetadata *)meta;
    uc->memDereg(priv->mem);
    delete priv;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxEngine::getPublicData(const nixlBackendMD *meta, std::string &str) const {
    const nixlUcxPrivateMetadata *priv = (nixlUcxPrivateMetadata *)meta;
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
nixlUcxEngine::internalMDHelper(const nixl_blob_t &blob,
                                const std::string &agent,
                                nixlBackendMD *&output) {
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
nixlUcxEngine::loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) {
    nixlUcxPrivateMetadata *input_md = (nixlUcxPrivateMetadata *)input;
    return internalMDHelper(input_md->rkeyStr, localAgent, output);
}

// To be cleaned up
nixl_status_t
nixlUcxEngine::loadRemoteMD(const nixlBlobDesc &input,
                            const nixl_mem_t &nixl_mem,
                            const std::string &remote_agent,
                            nixlBackendMD *&output) {
    return internalMDHelper(input.metaInfo, remote_agent, output);
}

nixl_status_t
nixlUcxEngine::unloadMD(nixlBackendMD *input) {

    nixlUcxPublicMetadata *md = (nixlUcxPublicMetadata *)input; // typecast?
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

nixl_status_t
nixlUcxEngine::prepXfer(const nixl_xfer_op_t &operation,
                        const nixl_meta_dlist_t &local,
                        const nixl_meta_dlist_t &remote,
                        const std::string &remote_agent,
                        nixlBackendReqH *&handle,
                        const nixl_opt_b_args_t *opt_args) const {
    if (local.descCount() == 0 || remote.descCount() == 0) {
        NIXL_ERROR << "Local or remote descriptor list is empty";
        return NIXL_ERR_INVALID_PARAM;
    }

    const size_t worker_id = getWorkerId(opt_args);
    /* TODO: try to get from a pool first */
    const auto int_handle = new nixlUcxBackendReqH(
        getWorker(worker_id).get(), worker_id, nullptr, getTerminalDeadlineOwner());
    const nixl_status_t status =
        prepareHandleAttestation(int_handle, operation, local, remote, remote_agent, opt_args);
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
    return int_handle->prepareAttestation(operation,
                                          local,
                                          remote,
                                          localAgent,
                                          remote_agent,
                                          opt_args == nullptr ? nullptr :
                                                                opt_args->remoteAgentAuthority);
}

nixl_status_t
nixlUcxEngine::estimateXferCost(const nixl_xfer_op_t &operation,
                                const nixl_meta_dlist_t &local,
                                const nixl_meta_dlist_t &remote,
                                const std::string &remote_agent,
                                nixlBackendReqH *const &handle,
                                std::chrono::microseconds &duration,
                                std::chrono::microseconds &err_margin,
                                nixl_cost_t &method,
                                const nixl_opt_args_t *opt_args) const {
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

        NIXL_ASSERT(lmd && rmd) << "No metadata found in descriptor lists at index " << i
                                << " during cost estimation";
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
    const std::shared_ptr<nixlUcxTerminalArm> terminal = int_handle->getActiveTerminal();

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
            const nixl_status_t slot_status =
                terminal->makeSlot(int_handle->getWorker(),
                                   nixl::ucx::ucx_callback_kind_t::DATA_CHUNK,
                                   {},
                                   terminal_slot);
            if (slot_status != NIXL_SUCCESS) {
                result.status = slot_status;
                break;
            }
        }
        const nixl_status_t ret = operation == NIXL_READ ? ep.read(raddr,
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
                            req, ret, nixl::ucx::terminalProgressTimestampNs()));
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
                        req, ret, nixl::ucx::terminalProgressTimestampNs()));
                }
                if (result.req != nullptr) {
                    ucp_request_free(result.req);
                }
                result.status = evidence_status;
                result.req = nullptr;
                break;
            }
            if (terminal_slot != nullptr) {
                const nixl_status_t arm_status =
                    terminal_slot->armPoster(req, ret, nixl::ucx::terminalProgressTimestampNs());
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
    const std::shared_ptr<nixlUcxTerminalArm> terminal = int_handle->getActiveTerminal();

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
            sendXferRangeBatch(*ep, operation, local, remote, handle, worker_id, i, end_idx);

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
        const nixl_status_t ret = int_handle->append(result.status,
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
                [attestation = int_handle->getAttestationState(), generation, endpoint_identity](
                    nixl_status_t callback_status) {
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
            *ep, terminal_slot != nullptr && ret == NIXL_SUCCESS ? NIXL_IN_PROG : ret);
        if (evidence_status != NIXL_SUCCESS) {
            if (terminal_slot != nullptr) {
                static_cast<void>(
                    terminal_slot->armPoster(req, ret, nixl::ucx::terminalProgressTimestampNs()));
            }
            return evidence_status;
        }
        if (terminal_slot != nullptr) {
            const nixl_status_t arm_status =
                terminal_slot->armPoster(req, ret, nixl::ucx::terminalProgressTimestampNs());
            if (arm_status != NIXL_SUCCESS) {
                return arm_status;
            }
            continue;
        }
        const nixl_status_t append_status = int_handle->append(
            ret, req, conn, nixlUcxBackendReqH::pending_kind_t::ENDPOINT_FLUSH, ep->getIdentity());
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
    nixl::ucx::notif_delivery_key_t notification_delivery;

    const nixl_status_t lifecycle_status = terminalLifecycleFatal_.load(std::memory_order_acquire);
    if (lifecycle_status != NIXL_SUCCESS) {
        return lifecycle_status;
    }

    if (lcnt != rcnt) {
        NIXL_ERROR << "Local (" << lcnt << ") and remote (" << rcnt
                   << ") descriptor lists differ in size";
        return NIXL_ERR_INVALID_PARAM;
    }

    if (opt_args != nullptr && opt_args->hasNotif) {
        if (opt_args->remoteAgentAuthority == nullptr) {
            return NIXL_ERR_INVALID_PARAM;
        }
        const uint64_t delivery_identity =
            allocateIdentity(nextDeliveryIdentity_, "notification delivery");
        const nixlBackendTransferEventBinding source_binding = int_handle->nextSubmissionBinding();
        notification_delivery.sourceHandleIdentity = source_binding.handleIdentity;
        notification_delivery.sourceGeneration = source_binding.generation;
        ret = prepareDataFrame(*opt_args->remoteAgentAuthority,
                               int_handle->getWorkerId(),
                               opt_args->notifMsg,
                               delivery_identity,
                               notification_frame,
                               notification_connection,
                               &notification_delivery);
        if (ret != NIXL_SUCCESS) {
            return ret;
        }
        const auto remote_metadata = static_cast<nixlUcxPublicMetadata *>(remote[0].metadataP);
        if (remote_metadata == nullptr || remote_metadata->conn != notification_connection) {
            return NIXL_ERR_NOT_ALLOWED;
        }
    }

    const bool has_notification = opt_args != nullptr && opt_args->hasNotif;
    std::vector<nixlUcxWorker *> progress_workers;
    progress_workers.reserve(getWorkers().size());
    for (const std::unique_ptr<nixlUcxWorker> &worker : getWorkers()) {
        progress_workers.push_back(worker.get());
    }
    ret = int_handle->beginSubmission(has_notification, progress_workers);
    if (ret != NIXL_SUCCESS) {
        return ret;
    }

    const std::shared_ptr<nixlUcxTerminalArm> terminal = int_handle->getActiveTerminal();
    if (has_notification) {
        if (terminal == nullptr) {
            NIXL_FATAL << "UCX attached notification started without terminal ownership";
        }
        nixlUcxWorker *const notification_worker = int_handle->getWorker();
        const std::shared_ptr<nixl::ucx::ucx_worker_continuation_queue_t>
            notification_continuations = notification_worker->getContinuationQueue();
        const nixl_status_t registration_status = terminal->installNotificationDelivery(
            deliveryRegistry_,
            notification_delivery,
            notification_continuations,
            [notification_state = notifState_, notification_delivery]() {
                return notification_state->validateOutboundDeliveryAuthority(
                           notification_delivery.route,
                           notification_delivery.capability,
                           notification_delivery.capabilityEpoch,
                           notification_delivery.connectionIdentity,
                           notification_delivery.receiptWorkerIncarnation,
                           notification_delivery.endpointIdentity) ==
                    nixl::ucx::notif_state_status_t::SUCCESS;
            });
        if (registration_status != NIXL_SUCCESS) {
            static_cast<void>(terminal->fail(registration_status));
            return registration_status;
        }
    }

    ret = sendXferRange(operation, local, remote, remote_agent, handle, 0, lcnt);
    if (ret != NIXL_SUCCESS) {
        if (has_notification) {
            static_cast<void>(deliveryRegistry_->failDelivery(
                notification_delivery.identity, ret, nixl::ucx::terminalProgressTimestampNs()));
        }
        int_handle->getAttestationState()->fail(int_handle->getAttestationState()->getGeneration(),
                                                ret,
                                                "UCX transfer submission failed");
        if (const std::shared_ptr<nixlUcxTerminalArm> terminal = int_handle->getActiveTerminal();
            terminal != nullptr) {
            static_cast<void>(terminal->fail(ret));
        }
        return ret;
    }

    ret = int_handle->finishSubmission();
    if (ret != NIXL_SUCCESS) {
        if (has_notification) {
            static_cast<void>(deliveryRegistry_->failDelivery(
                notification_delivery.identity, ret, nixl::ucx::terminalProgressTimestampNs()));
        }
        if (const std::shared_ptr<nixlUcxTerminalArm> terminal = int_handle->getActiveTerminal();
            terminal != nullptr) {
            static_cast<void>(terminal->fail(ret));
        }
        return ret;
    }

    if (terminal != nullptr) {
        ret = terminal->armDeadline(nixl::ucx::terminal_deadline_owner_t::monotonicTimestampNs());
        if (ret != NIXL_SUCCESS) {
            static_cast<void>(terminal->fail(ret));
            return ret;
        }
        nixl::ucx::terminal_submission_state_t::notification_post_t notification_post;
        if (has_notification) {
            nixlUcxWorker *const notification_worker = int_handle->getWorker();
            const size_t notification_worker_id = int_handle->getWorkerId();
            notification_post = [this,
                                 terminal,
                                 notification_worker,
                                 notification_worker_id,
                                 notification_connection,
                                 delivery_identity = notification_delivery.identity,
                                 frame = std::move(notification_frame)]() mutable {
                const nixl_status_t enqueue_status =
                    notification_worker->enqueueContinuation([this,
                                                              terminal,
                                                              notification_worker,
                                                              notification_worker_id,
                                                              notification_connection,
                                                              delivery_registry = deliveryRegistry_,
                                                              delivery_identity,
                                                              frame = std::move(frame)]() mutable {
                        std::shared_ptr<nixl::ucx::ucx_callback_slot_t> slot;
                        nixl_status_t status = terminal->makeSlot(
                            notification_worker,
                            nixl::ucx::ucx_callback_kind_t::NOTIFICATION_SEND,
                            [delivery_registry,
                             delivery_identity](nixl_status_t local_status) noexcept {
                                const auto completion = delivery_registry->completeLocal(
                                    delivery_identity,
                                    local_status,
                                    nixl::ucx::terminalProgressTimestampNs());
                                return completion == nixl::ucx::notif_delivery_status_t::SUCCESS ||
                                        completion ==
                                            nixl::ucx::notif_delivery_status_t::DUPLICATE_DELIVERY ?
                                    NIXL_SUCCESS :
                                    NIXL_ERR_BACKEND;
                            },
                            slot);
                        if (status != NIXL_SUCCESS) {
                            return status;
                        }
                        nixlUcxReq request = nullptr;
                        status =
                            notifSendPriv(std::move(frame),
                                          notification_connection->getEp(notification_worker_id),
                                          &request,
                                          slot.get());
                        const nixl_status_t arm_status = slot->armPoster(
                            request, status, nixl::ucx::terminalProgressTimestampNs());
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
            int_handle->notif.emplace(std::move(notification_frame), notification_connection);
        }
    }

    return ret;
}

nixl_status_t
nixlUcxEngine::checkXfer(nixlBackendReqH *handle) const {
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
    const nixl_status_t status = notifSendPriv(std::move(notif.frame), ep, &req);

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
nixlUcxEngine::takeXferCompletionAttestation(nixlBackendReqH *handle,
                                             nixl_xfer_attestation_t &attestation) const {
    if (handle == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const auto int_handle = static_cast<nixlUcxBackendReqH *>(handle);
    return int_handle->takeCompletionAttestation(attestation);
}

nixl_status_t
nixlUcxEngine::subscribeXferTerminal(nixlBackendReqH *handle,
                                     const nixlBackendTransferEventBinding &binding,
                                     const std::shared_ptr<nixlBackendTransferTransitionSink> &sink,
                                     std::unique_ptr<nixlBackendEventSubscription> &subscription) {
    subscription.reset();
    const nixl_status_t lifecycle_status = terminalLifecycleFatal_.load(std::memory_order_acquire);
    if (lifecycle_status != NIXL_SUCCESS) {
        return lifecycle_status;
    }
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

nixl_status_t
nixlUcxEngine::releaseReqH(nixlBackendReqH *handle) const {
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
    if (ep == nullptr || frame.empty() || frame.size() > nixl::ucx::notif_wire_max_frame_size) {
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
nixlUcxEngine::sendControlFrame(const nixl::ucx::notif_wire_envelope_t &envelope,
                                uint64_t connection_identity,
                                size_t worker_id,
                                nixlUcxReq *req,
                                nixl::ucx::ucx_callback_slot_t *terminal_slot) const {
    const ucx_connection_ptr_t connection = getConnection(connection_identity);
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
        notifSendPriv(std::move(frame), connection->getEp(worker_id), req, terminal_slot);
    return req == nullptr && terminal_slot == nullptr && send_status == NIXL_IN_PROG ?
        NIXL_SUCCESS :
        send_status;
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

nixlUcxEngine::remote_connection_map_t
nixlUcxEngine::detachRemoteConnections() noexcept {
    remote_connection_map_t retired_connections;
    {
        const std::lock_guard lock(connectionMutex_);
        retired_connections.swap(remoteConnMap);
    }
    return retired_connections;
}

ucx_connection_ptr_t
nixlUcxEngine::detachRemoteConnection(const std::string &remote_agent,
                                      const ucx_connection_ptr_t &expected_connection) noexcept {
    ucx_connection_ptr_t retired_connection;
    {
        const std::lock_guard lock(connectionMutex_);
        const auto it = remoteConnMap.find(remote_agent);
        if (it == remoteConnMap.end() || it->second != expected_connection) {
            return nullptr;
        }
        retired_connection = std::move(it->second);
        remoteConnMap.erase(it);
    }
    return retired_connection;
}

std::optional<nixlUcxEngine::exactRouteRecord>
nixlUcxEngine::getExactRoute(uint64_t handle_identity, uint64_t generation) const {
    return endpointFailureState_->getExactRoute(handle_identity, generation);
}

nixl_status_t
nixlUcxEngine::prepareDataFrame(const nixl_remote_agent_authority_t &authority,
                                size_t worker_id,
                                const std::string &msg,
                                uint64_t delivery_identity,
                                std::vector<std::uint8_t> &frame,
                                ucx_connection_ptr_t &connection,
                                nixl::ucx::notif_delivery_key_t *delivery_key) const {
    if (worker_id >= localConnectionMetadata_.workers.size()) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const std::optional<exactRouteRecord> route =
        getExactRoute(authority.handleIdentity, authority.generation);
    if (!route.has_value() || route->connectionIdentity != authority.connectionIdentity) {
        return NIXL_ERR_NOT_FOUND;
    }

    ucx_connection_ptr_t exact_connection = getConnection(route->connectionIdentity);
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

    if (delivery_key != nullptr) {
        const size_t remote_worker_id = worker_id % exact_connection->metadata_.workers.size();
        const size_t receipt_worker_id = remote_worker_id % getWorkers().size();
        if (!exact_connection->metadata_.workers[remote_worker_id].supportsAttachedReceipt ||
            !localConnectionMetadata_.workers[receipt_worker_id].supportsAttachedReceipt) {
            return NIXL_ERR_NOT_SUPPORTED;
        }
    }

    nixl::ucx::notif_wire_envelope_t envelope;
    const nixl::ucx::notif_wire_uuid_t &worker =
        localConnectionMetadata_.workers[worker_id].incarnation;
    const nixl::ucx::notif_state_status_t state_status =
        notifState_->prepareData(route->route,
                                 worker,
                                 delivery_identity,
                                 delivery_key == nullptr ? 0 : delivery_key->sourceHandleIdentity,
                                 delivery_key == nullptr ? 0 : delivery_key->sourceGeneration,
                                 envelope);
    if (state_status == nixl::ucx::notif_state_status_t::NOT_READY) {
        nixl::ucx::notif_wire_envelope_t offer;
        if (notifState_->makeOffer(route->route, worker, offer) ==
            nixl::ucx::notif_state_status_t::SUCCESS) {
            (void)sendControlFrame(offer, exact_connection->getIdentity(), worker_id);
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
            NIXL_ERR_INVALID_PARAM :
            NIXL_ERR_BACKEND;
    }

    frame.swap(encoded);
    if (delivery_key != nullptr) {
        if (delivery_identity == 0) {
            return NIXL_ERR_INVALID_PARAM;
        }
        const uint64_t source_handle_identity = delivery_key->sourceHandleIdentity;
        const uint64_t source_generation = delivery_key->sourceGeneration;
        const size_t remote_worker_id = worker_id % exact_connection->metadata_.workers.size();
        const size_t receipt_worker_id = remote_worker_id % getWorkers().size();
        *delivery_key = {
            .identity = delivery_identity,
            .sourceHandleIdentity = source_handle_identity,
            .sourceGeneration = source_generation,
            .route = route->route,
            .capability = envelope.capability,
            .capabilityEpoch = envelope.capabilityEpoch,
            .connectionIdentity = exact_connection->getIdentity(),
            .receiptWorkerIncarnation =
                exact_connection->metadata_.workers[remote_worker_id].incarnation,
            .endpointIdentity = exact_connection->getEp(receipt_worker_id)->getIdentity(),
        };
    }
    connection = std::move(exact_connection);
    return NIXL_SUCCESS;
}

void
nixlUcxEngine::appendNotif(nixlAuthenticatedNotification &&notification) const {
    // In the "no progress thread" case the lock in nixlAgent is sufficient.
    (void)notifQueue_.push(std::move(notification));
}

nixl_status_t
nixlUcxEngine::admitNotif(nixlAuthenticatedNotification &&notification) const {
    return notifQueue_.push(std::move(notification));
}

void
nixlUcxEngine::poisonNotifs() const {
    // In the "no progress thread" case the lock in nixlAgent is sufficient.
    notifQueue_.poison();
}

void
nixlUcxEngine::drainNotificationDeliveries() const {
    const auto status =
        deliveryRegistry_->failAll(NIXL_ERR_CANCELED, nixl::ucx::terminalProgressTimestampNs());
    if (status != nixl::ucx::notif_delivery_status_t::SUCCESS) {
        NIXL_FATAL << "UCX notification delivery shutdown failure";
    }
}

void
nixlUcxEngine::queryTerminalLifecycleInventory(
    nixlBackendTerminalLifecycleInventory &inventory) const noexcept {
    inventory = {};
    try {
        const nixl::ucx::terminal_deadline_snapshot_t deadlines =
            terminalDeadlineOwner_->snapshot();
        inventory.activeNativeDeadlines = deadlines.inventory.active;
        const nixl::ucx::notif_delivery_snapshot_t source = deliveryRegistry_->snapshot();
        inventory.sourceDeliveriesOutstanding = source.inventory.outstanding;
        for (const nixl::ucx::notif_delivery_inventory_record_t &record : source.activeRecords) {
            inventory.sourceLocalPending += record.localPending ? 1U : 0U;
            inventory.sourceReceiptPending += record.receiptPending ? 1U : 0U;
            inventory.sourceDeliveries.push_back({
                .deliveryIdentity = record.key.identity,
                .sourceHandleIdentity = record.key.sourceHandleIdentity,
                .sourceGeneration = record.key.sourceGeneration,
                .localPending = record.localPending,
                .receiptPending = record.receiptPending,
                .deadlineActive = std::any_of(
                    deadlines.activeKeys.begin(),
                    deadlines.activeKeys.end(),
                    [&record](const nixl::ucx::terminal_deadline_key_t &deadline) {
                        return deadline.handleIdentity == record.key.sourceHandleIdentity &&
                            deadline.generation == record.key.sourceGeneration;
                    }),
            });
        }
        for (const nixl::ucx::terminal_deadline_key_t &deadline : deadlines.activeKeys) {
            inventory.nativeDeadlines.push_back({
                .handleIdentity = deadline.handleIdentity,
                .generation = deadline.generation,
            });
        }

        const nixl::ucx::notif_ingress_snapshot_t destination = ingressRegistry_->snapshot();
        inventory.destinationPending = destination.inventory.pending;
        inventory.destinationAdmitting = destination.inventory.admitting;
        inventory.destinationCommitted = destination.inventory.committed;
        inventory.destinationReplaying = destination.inventory.replaying;
        inventory.destinationQuarantined = destination.inventory.quarantined;
        for (const nixl::ucx::notif_ingress_inventory_record_t &record :
             destination.activeRecords) {
            nixl_backend_terminal_destination_phase_t phase =
                nixl_backend_terminal_destination_phase_t::PENDING;
            switch (record.phase) {
            case nixl::ucx::notif_ingress_inventory_phase_t::PENDING:
                break;
            case nixl::ucx::notif_ingress_inventory_phase_t::ADMITTING:
                phase = nixl_backend_terminal_destination_phase_t::ADMITTING;
                break;
            case nixl::ucx::notif_ingress_inventory_phase_t::COMMITTED:
                phase = nixl_backend_terminal_destination_phase_t::COMMITTED;
                break;
            case nixl::ucx::notif_ingress_inventory_phase_t::REPLAYING:
                phase = nixl_backend_terminal_destination_phase_t::REPLAYING;
                break;
            case nixl::ucx::notif_ingress_inventory_phase_t::QUARANTINED:
                phase = nixl_backend_terminal_destination_phase_t::QUARANTINED;
                break;
            }
            inventory.destinationDeliveries.push_back({
                .sourceBackendIncarnation = formatWireUuid(record.key.sourceBackendIncarnation),
                .sourceHandleIdentity = record.key.sourceHandleIdentity,
                .sourceGeneration = record.key.sourceGeneration,
                .deliveryIdentity = record.key.deliveryIdentity,
                .phase = phase,
            });
        }
    }
    catch (const std::bad_alloc &) {
        NIXL_FATAL << "UCX terminal lifecycle inventory allocation failed";
    }
}

bool
nixlUcxEngine::terminalDeadlinesDrained() const noexcept {
    return terminalDeadlineOwner_ == nullptr || terminalDeadlineOwner_->inventory().active == 0;
}

void
nixlUcxEngine::closeNotificationIngress() const {
    const auto status = ingressRegistry_->failAll();
    if (status != nixl::ucx::notif_ingress_status_t::SUCCESS) {
        NIXL_FATAL << "UCX notification ingress shutdown failure";
    }
}

bool
nixlUcxEngine::notificationDeliveriesDrained() const noexcept {
    return deliveryRegistry_->inventory().outstanding == 0;
}

bool
nixlUcxEngine::notificationIngressDrained() const noexcept {
    const nixl::ucx::notif_ingress_inventory_t inventory = ingressRegistry_->inventory();
    return inventory.pending == 0 && inventory.admitting == 0 && inventory.committed == 0 &&
        inventory.replaying == 0 && inventory.quarantined == 0;
}

nixl_status_t
nixlUcxEngine::scheduleAdmissionReceipt(const nixl::ucx::notif_ingress_key_t &key,
                                        const nixl::ucx::notif_wire_envelope_t &receipt,
                                        size_t worker_id,
                                        uint64_t connection_identity) const {
    if (worker_id >= getWorkers().size()) {
        return NIXL_ERR_INVALID_PARAM;
    }
    nixlUcxWorker *const receipt_worker = getWorker(worker_id).get();
    const std::shared_ptr<nixl::ucx::notif_ingress_registry_t> ingress = ingressRegistry_;
    return receipt_worker->getContinuationQueue()->enqueueProducer(
        [this, ingress, key, receipt_worker, receipt, connection_identity, worker_id]() noexcept {
            const auto keep_alive = std::make_shared<nixl::ucx::terminal_submission_state_t>(
                1, 1, 1, 0, 0, false, std::make_shared<nixlUcxNoopTerminalSink>());
            std::shared_ptr<nixl::ucx::ucx_callback_slot_t> slot;
            const nixl_status_t slot_status = receipt_worker->makeTerminalCallbackSlot(
                keep_alive,
                nixl::ucx::ucx_callback_kind_t::NOTIFICATION_SEND,
                slot,
                [ingress, key](nixl_status_t send_status) noexcept {
                    const auto completion = ingress->completeReceipt(key, send_status);
                    if (completion == nixl::ucx::notif_ingress_status_t::SUCCESS) {
                        return NIXL_SUCCESS;
                    }
                    if (completion == nixl::ucx::notif_ingress_status_t::RECEIPT_FAILED) {
                        NIXL_FATAL << "UCX committed admission receipt failed to send";
                    }
                    NIXL_FATAL << "UCX receipt-send owner lost its committed obligation";
                    return NIXL_ERR_BACKEND;
                },
                {});
            if (slot_status != NIXL_SUCCESS) {
                return slot_status;
            }
            const auto retention = ingress->retainReceiptSlot(key, slot);
            if (retention != nixl::ucx::notif_ingress_status_t::SUCCESS) {
                return NIXL_ERR_BACKEND;
            }
            nixlUcxReq request = nullptr;
            const nixl_status_t send_status =
                sendControlFrame(receipt, connection_identity, worker_id, &request, slot.get());
            const nixl_status_t arm_status =
                slot->armPoster(request, send_status, nixl::ucx::terminalProgressTimestampNs());
            return arm_status;
        });
}

nixl_status_t
nixlUcxEngine::scheduleOfferControlReply(
    const nixl::ucx::notif_wire_envelope_t &acknowledgement,
    const std::optional<nixl::ucx::notif_wire_envelope_t> &local_offer,
    size_t worker_id,
    uint64_t connection_identity) const {
    if (worker_id >= getWorkers().size()) {
        return NIXL_ERR_INVALID_PARAM;
    }
    return getWorker(worker_id)->getContinuationQueue()->enqueueProducer(
        [this, acknowledgement, local_offer, worker_id, connection_identity]() noexcept {
            const nixl_status_t acknowledgement_status =
                sendControlFrame(acknowledgement, connection_identity, worker_id);
            if (acknowledgement_status != NIXL_SUCCESS) {
                NIXL_WARN << "Failed to send UCX notification ACK: " << acknowledgement_status;
            }
            if (local_offer.has_value()) {
                const nixl_status_t offer_status =
                    sendControlFrame(*local_offer, connection_identity, worker_id);
                if (offer_status != NIXL_SUCCESS) {
                    NIXL_WARN << "Failed to re-emit UCX notification OFFER: " << offer_status;
                }
            }
            return NIXL_SUCCESS;
        });
}

nixl_status_t
nixlUcxEngine::installAdmissionReceiptBarrier(
    nixlBackendAdmissionReceiptBarrier *barrier) noexcept {
    nixlBackendAdmissionReceiptBarrier *expected = nullptr;
    if (barrier != nullptr) {
        return admissionReceiptBarrier_.compare_exchange_strong(
                   expected, barrier, std::memory_order_acq_rel, std::memory_order_acquire) ?
            NIXL_SUCCESS :
            NIXL_ERR_NOT_ALLOWED;
    }
    admissionReceiptBarrier_.store(nullptr, std::memory_order_release);
    return NIXL_SUCCESS;
}

ucs_status_t
nixlUcxEngine::notifAmCb(void *arg,
                         const void *header,
                         size_t header_length,
                         void *data,
                         size_t length,
                         const ucp_am_recv_param_t *param) {
    try {
        if (arg == nullptr || param == nullptr ||
            (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV) != 0 || data == nullptr ||
            header_length != 0 || length > nixlUcxNotificationQueue::maxWireBytes) {
            NIXL_ERROR << "Rejected malformed or oversized UCX notification frame";
            return UCS_OK;
        }

        const auto *context = static_cast<notifCallbackContext *>(arg);
        nixlUcxEngine *engine = context->engine;
        if (engine == nullptr ||
            context->workerId >= engine->localConnectionMetadata_.workers.size()) {
            return UCS_OK;
        }

        const auto wire =
            std::span<const std::uint8_t>(static_cast<const std::uint8_t *>(data), length);
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
                engine->notifState_->acceptOffer(frame.envelope, local_worker, acceptance);
            if (status != nixl::ucx::notif_state_status_t::SUCCESS) {
                if (status == nixl::ucx::notif_state_status_t::EPOCH_CONFLICT &&
                    acceptance.route.handleIdentity != 0) {
                    const auto delivery_status = engine->deliveryRegistry_->failRoute(
                        acceptance.route,
                        NIXL_ERR_NOT_ALLOWED,
                        nixl::ucx::terminalProgressTimestampNs());
                    const auto ingress_status =
                        engine->ingressRegistry_->failRoute(acceptance.route);
                    if (delivery_status != nixl::ucx::notif_delivery_status_t::SUCCESS ||
                        ingress_status != nixl::ucx::notif_ingress_status_t::SUCCESS) {
                        NIXL_FATAL << "UCX capability conflict failed to drain exact-route state";
                    }
                }
                NIXL_DEBUG << "Rejected UCX notification OFFER with state status "
                           << static_cast<int>(status);
                return UCS_OK;
            }
            if (acceptance.remoteCapabilityChanged) {
                const auto delivery_status = engine->deliveryRegistry_->failSupersededAuthority(
                    acceptance.route,
                    frame.envelope.capability,
                    frame.envelope.capabilityEpoch,
                    NIXL_ERR_NOT_ALLOWED,
                    nixl::ucx::terminalProgressTimestampNs());
                if (delivery_status != nixl::ucx::notif_delivery_status_t::SUCCESS) {
                    NIXL_FATAL << "UCX capability supersession failed to drain stale deliveries";
                }
            }

            const std::optional<exactRouteRecord> route = engine->getExactRoute(
                acceptance.route.handleIdentity, acceptance.route.handleGeneration);
            if (!route.has_value() || route->route != acceptance.route) {
                NIXL_ERROR << "Rejected UCX notification OFFER for an unknown exact route";
                return UCS_OK;
            }
            const std::optional<nixl::ucx::notif_wire_envelope_t> local_offer =
                acceptance.reemitLocalOffer ?
                std::optional<nixl::ucx::notif_wire_envelope_t>(acceptance.localOffer) :
                std::nullopt;
            const nixl_status_t schedule_status =
                engine->scheduleOfferControlReply(acceptance.acknowledgement,
                                                  local_offer,
                                                  context->workerId,
                                                  route->connectionIdentity);
            if (schedule_status != NIXL_SUCCESS) {
                NIXL_FATAL << "UCX could not schedule its notification OFFER reply: "
                           << schedule_status;
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


        if (frame.envelope.type == nixl::ucx::notif_wire_type_t::DATA_RECEIPT) {
            const nixl::ucx::notif_data_resolution_t resolution =
                engine->notifState_->resolveDataReceipt(frame.envelope);
            if (resolution.disposition != nixl::ucx::notif_data_disposition_t::DELIVER) {
                NIXL_FATAL << "Unknown or unauthenticated UCX DATA_RECEIPT failed closed";
            }
            const nixl::ucx::notif_delivery_receipt_t receipt_key = {
                .identity = resolution.authority.deliveryIdentity,
                .sourceHandleIdentity = resolution.authority.sourceHandleIdentity,
                .sourceGeneration = resolution.authority.sourceGeneration,
                .route = resolution.authority.route,
                .capability = resolution.authority.capability,
                .capabilityEpoch = resolution.authority.capabilityEpoch,
                .connectionIdentity = resolution.authority.connectionIdentity,
                .receiptWorkerIncarnation = resolution.authority.senderWorkerIncarnation,
                .endpointIdentity = resolution.authority.endpointIdentity,
            };
            const nixl::ucx::notif_delivery_status_t receipt_status =
                engine->deliveryRegistry_->acceptReceipt(receipt_key,
                                                         nixl::ucx::terminalProgressTimestampNs());
            if (receipt_status != nixl::ucx::notif_delivery_status_t::SUCCESS &&
                receipt_status != nixl::ucx::notif_delivery_status_t::DUPLICATE_DELIVERY) {
                NIXL_FATAL << "UCX DATA_RECEIPT conflicted with live delivery authority";
            }
            return UCS_OK;
        }

        const nixl::ucx::notif_data_resolution_t resolution =
            engine->notifState_->resolveData(frame.envelope);
        if (resolution.disposition == nixl::ucx::notif_data_disposition_t::POISON_GLOBAL) {
            NIXL_ERROR << "Unknown UCX notification capability poisoned authenticated draining";
            engine->poisonNotifs();
            return UCS_OK;
        }
        if (resolution.disposition == nixl::ucx::notif_data_disposition_t::DROP_ROUTE) {
            NIXL_ERROR << "Dropped invalid UCX notification for an exact route";
            return UCS_OK;
        }

        const std::optional<exactRouteRecord> route = engine->getExactRoute(
            resolution.authority.route.handleIdentity, resolution.authority.route.handleGeneration);
        if (!route.has_value() || route->route != resolution.authority.route) {
            NIXL_ERROR << "Dropped UCX notification with missing exact route ownership";
            return UCS_OK;
        }

        if (frame.envelope.type == nixl::ucx::notif_wire_type_t::ATTACHED_DATA) {
            nixl::ucx::notif_wire_envelope_t frozen_receipt;
            const nixl::ucx::notif_state_status_t receipt_state =
                engine->notifState_->makeDataReceipt(resolution.authority,
                                                     local_worker,
                                                     frame.envelope.capability,
                                                     frame.envelope.capabilityEpoch,
                                                     frame.envelope.deliveryIdentity,
                                                     frame.envelope.sourceHandleIdentity,
                                                     frame.envelope.sourceGeneration,
                                                     frozen_receipt);
            if (receipt_state != nixl::ucx::notif_state_status_t::SUCCESS) {
                NIXL_ERROR << "UCX attached ingress lost its exact route before reservation";
                return UCS_OK;
            }
            const nixl::ucx::notif_ingress_key_t ingress_key = {
                .sourceBackendIncarnation = frame.envelope.senderBackendIncarnation,
                .sourceHandleIdentity = frame.envelope.sourceHandleIdentity,
                .sourceGeneration = frame.envelope.sourceGeneration,
                .deliveryIdentity = frame.envelope.deliveryIdentity,
                .destinationAgentIncarnation = frame.envelope.recipientAgentIncarnation,
                .destinationBackendIncarnation = frame.envelope.recipientBackendIncarnation,
                .route = resolution.authority.route,
                .capability = frame.envelope.capability,
                .capabilityEpoch = frame.envelope.capabilityEpoch,
                .connectionIdentity = resolution.authority.connectionIdentity,
                .sourceWorkerIncarnation = frame.envelope.senderWorkerIncarnation,
                .receiptWorkerIncarnation = local_worker,
                .endpointIdentity = resolution.authority.endpointIdentity,
            };
            const auto reservation =
                engine->ingressRegistry_->reserve(ingress_key, frame.payload, frozen_receipt);
            if (reservation == nixl::ucx::notif_ingress_status_t::IDENTITY_CONFLICT ||
                reservation == nixl::ucx::notif_ingress_status_t::CAPACITY_EXCEEDED ||
                reservation == nixl::ucx::notif_ingress_status_t::INVALID_ARGUMENT) {
                NIXL_FATAL << "UCX attached ingress identity failed closed";
            }
            nixlAuthenticatedNotification admitted = {
                .remoteAgent = route->remoteAgent,
                .payload = std::string(reinterpret_cast<const char *>(frame.payload.data()),
                                       frame.payload.size()),
                .handleIdentity = resolution.authority.route.handleIdentity,
                .generation = resolution.authority.route.handleGeneration,
                .connectionIdentity = resolution.authority.connectionIdentity,
                .endpointIdentity = resolution.authority.endpointIdentity,
            };
            nixl::ucx::notif_wire_envelope_t receipt;
            const auto admission = engine->ingressRegistry_->admit(
                ingress_key,
                frame.payload,
                std::move(admitted),
                [engine](nixlAuthenticatedNotification &&notification) {
                    return engine->admitNotif(std::move(notification));
                },
                receipt);
            if (admission == nixl::ucx::notif_ingress_status_t::DUPLICATE_PENDING) {
                return UCS_OK;
            }
            if (admission != nixl::ucx::notif_ingress_status_t::SUCCESS &&
                admission != nixl::ucx::notif_ingress_status_t::DUPLICATE_COMMITTED) {
                NIXL_FATAL << "UCX attached ingress could not commit queue admission";
            }
            nixlBackendAdmissionReceiptBarrier *const barrier =
                engine->admissionReceiptBarrier_.exchange(nullptr, std::memory_order_acq_rel);
            if (barrier != nullptr) {
                const nixlBackendAdmissionReceiptAuthority authority = {
                    .sourceHandleIdentity = ingress_key.sourceHandleIdentity,
                    .sourceGeneration = ingress_key.sourceGeneration,
                    .deliveryIdentity = ingress_key.deliveryIdentity,
                };
                const nixl_status_t defer_status = barrier->deferAfterAdmission(
                    authority,
                    [engine,
                     ingress_key,
                     receipt,
                     worker_id = context->workerId,
                     connection_identity = resolution.authority.connectionIdentity]() noexcept {
                        return engine->scheduleAdmissionReceipt(
                            ingress_key, receipt, worker_id, connection_identity);
                    });
                if (defer_status != NIXL_SUCCESS) {
                    NIXL_FATAL << "UCX qualification barrier could not retain receipt authority: "
                               << defer_status;
                }
                return UCS_OK;
            }
            const nixl_status_t schedule_status = engine->scheduleAdmissionReceipt(
                ingress_key, receipt, context->workerId, resolution.authority.connectionIdentity);
            if (schedule_status != NIXL_SUCCESS) {
                NIXL_FATAL << "UCX could not schedule authenticated DATA_RECEIPT: "
                           << schedule_status;
            }
            return UCS_OK;
        }

        nixlAuthenticatedNotification admitted = {
            .remoteAgent = route->remoteAgent,
            .payload = std::string(reinterpret_cast<const char *>(frame.payload.data()),
                                   frame.payload.size()),
            .handleIdentity = resolution.authority.route.handleIdentity,
            .generation = resolution.authority.route.handleGeneration,
            .connectionIdentity = resolution.authority.connectionIdentity,
            .endpointIdentity = resolution.authority.endpointIdentity,
        };
        const nixl_status_t admission_status = engine->admitNotif(std::move(admitted));
        if (admission_status != NIXL_SUCCESS) {
            NIXL_FATAL << "UCX authenticated notification admission failed: " << admission_status;
        }
        return UCS_OK;
    }
    catch (const std::exception &error) {
        NIXL_FATAL << "UCX notification callback threw across its C boundary: " << error.what();
    }
    catch (...) {
        NIXL_FATAL << "UCX notification callback threw a non-standard exception across its C "
                      "boundary";
    }
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
nixlUcxEngine::genNotif(const nixlRemoteAgentBinding &binding, const std::string &msg) const {
    const size_t worker_id = getWorkerId();
    std::vector<std::uint8_t> frame;
    ucx_connection_ptr_t connection;
    const nixl_status_t prepare_status =
        prepareDataFrame(binding.authority, worker_id, msg, 0, frame, connection);
    if (prepare_status != NIXL_SUCCESS) {
        return prepare_status;
    }

    const nixl_status_t ret = notifSendPriv(std::move(frame), connection->getEp(worker_id));
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
