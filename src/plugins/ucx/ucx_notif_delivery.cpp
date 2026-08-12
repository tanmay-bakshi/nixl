/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "ucx_notif_delivery.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace nixl::ucx {

notif_delivery_registry_t::notif_delivery_registry_t(std::size_t capacity,
                                                     std::size_t completed_capacity)
    : capacity_(capacity),
      completedCapacity_(completed_capacity == 0 ? capacity : completed_capacity) {
    if (capacity_ == 0 || completedCapacity_ == 0) {
        throw std::invalid_argument("notification delivery registry requires positive bounds");
    }
}

notif_delivery_status_t
notif_delivery_registry_t::registerDelivery(const notif_delivery_key_t &key,
                                            terminal_t terminal,
                                            const authority_validator_t &validate_authority) {
    if (key.identity == 0 || key.sourceHandleIdentity == 0 || key.sourceGeneration == 0 ||
        key.route.handleIdentity == 0 || key.route.handleGeneration == 0 ||
        !isCanonicalNotifWireUuid(key.capability) || key.capabilityEpoch == 0 ||
        key.connectionIdentity == 0 || !isCanonicalNotifWireUuid(key.receiptWorkerIncarnation) ||
        key.endpointIdentity == 0 || !terminal || !validate_authority) {
        return notif_delivery_status_t::INVALID_ARGUMENT;
    }
    {
        const std::lock_guard lock(mutex_);
        if (!accepting_) {
            return notif_delivery_status_t::REGISTRY_CLOSED;
        }
        const auto outstanding = outstanding_.find(key.identity);
        if (outstanding != outstanding_.end()) {
            return outstanding->second.key == key ? notif_delivery_status_t::DUPLICATE_DELIVERY :
                                                    notif_delivery_status_t::IDENTITY_CONFLICT;
        }
        const auto completed = completed_.find(key.identity);
        if (completed != completed_.end()) {
            return completed->second.key == key ? notif_delivery_status_t::DUPLICATE_DELIVERY :
                                                  notif_delivery_status_t::IDENTITY_CONFLICT;
        }
        if (outstanding_.size() >= capacity_) {
            return notif_delivery_status_t::CAPACITY_EXCEEDED;
        }

        // Route retirement and delivery admission serialize through this lock. The validator
        // takes the route-state lock only while this admission lock is held; terminal route
        // transitions update route state before entering this registry, so either validation
        // rejects the stale authority or the subsequent terminal transition claims the record.
        if (!validate_authority()) {
            return notif_delivery_status_t::ROUTE_TERMINAL;
        }
        try {
            outstanding_.emplace(key.identity,
                                 record_t{
                                     .key = key,
                                     .terminal = std::move(terminal),
                                 });
        }
        catch (const std::bad_alloc &) {
            return notif_delivery_status_t::CAPACITY_EXCEEDED;
        }
    }
    return notif_delivery_status_t::SUCCESS;
}

notif_delivery_status_t
notif_delivery_registry_t::completeLocal(std::uint64_t identity,
                                         nixl_status_t status,
                                         std::uint64_t timestamp_ns) {
    std::vector<completion_t> completions;
    notif_delivery_status_t result = notif_delivery_status_t::SUCCESS;
    {
        const std::lock_guard lock(mutex_);
        const auto known = outstanding_.find(identity);
        if (known == outstanding_.end()) {
            return completed_.contains(identity) ? notif_delivery_status_t::DUPLICATE_DELIVERY :
                                                   notif_delivery_status_t::UNKNOWN_DELIVERY;
        }
        if (status != NIXL_SUCCESS) {
            completions.push_back({std::move(known->second.terminal), status, timestamp_ns});
            rememberCompletedLocked(known->second.key, known->second.receiptComplete);
            outstanding_.erase(known);
        } else if (known->second.localComplete) {
            result = notif_delivery_status_t::DUPLICATE_DELIVERY;
        } else {
            known->second.localComplete = true;
            if (known->second.receiptComplete) {
                completions.push_back({std::move(known->second.terminal),
                                       NIXL_SUCCESS,
                                       known->second.receiptTimestampNs});
                rememberCompletedLocked(known->second.key, true);
                outstanding_.erase(known);
            }
        }
    }
    dispatch(std::move(completions));
    return result;
}

notif_delivery_status_t
notif_delivery_registry_t::acceptReceipt(const notif_delivery_receipt_t &receipt,
                                         std::uint64_t timestamp_ns) {
    std::vector<completion_t> completions;
    notif_delivery_status_t result = notif_delivery_status_t::SUCCESS;
    {
        const std::lock_guard lock(mutex_);
        const auto known = outstanding_.find(receipt.identity);
        if (known == outstanding_.end()) {
            auto completed = completed_.find(receipt.identity);
            if (completed == completed_.end()) {
                return notif_delivery_status_t::UNKNOWN_DELIVERY;
            }
            const notif_delivery_key_t &key = completed->second.key;
            const bool identical = key.identity == receipt.identity &&
                key.sourceHandleIdentity == receipt.sourceHandleIdentity &&
                key.sourceGeneration == receipt.sourceGeneration && key.route == receipt.route &&
                key.capability == receipt.capability &&
                key.capabilityEpoch == receipt.capabilityEpoch &&
                key.connectionIdentity == receipt.connectionIdentity &&
                key.receiptWorkerIncarnation == receipt.receiptWorkerIncarnation &&
                key.endpointIdentity == receipt.endpointIdentity;
            if (!identical) {
                return notif_delivery_status_t::IDENTITY_CONFLICT;
            }
            // Cancellation, timeout, and route failure tombstone the complete authority before
            // releasing native resources. An exact late receipt is therefore stale, not a
            // conflict, and can never resurrect or republish the delivery.
            completed->second.receiptAccepted = true;
            return notif_delivery_status_t::DUPLICATE_DELIVERY;
        }
        const notif_delivery_key_t &key = known->second.key;
        if (key.identity != receipt.identity ||
            key.sourceHandleIdentity != receipt.sourceHandleIdentity ||
            key.sourceGeneration != receipt.sourceGeneration || key.route != receipt.route ||
            key.capability != receipt.capability ||
            key.capabilityEpoch != receipt.capabilityEpoch ||
            key.connectionIdentity != receipt.connectionIdentity ||
            key.receiptWorkerIncarnation != receipt.receiptWorkerIncarnation ||
            key.endpointIdentity != receipt.endpointIdentity) {
            return notif_delivery_status_t::IDENTITY_CONFLICT;
        }
        if (known->second.receiptComplete) {
            result = notif_delivery_status_t::DUPLICATE_DELIVERY;
        } else {
            known->second.receiptComplete = true;
            known->second.receiptTimestampNs = timestamp_ns;
            if (known->second.localComplete) {
                completions.push_back(
                    {std::move(known->second.terminal), NIXL_SUCCESS, timestamp_ns});
                rememberCompletedLocked(known->second.key, true);
                outstanding_.erase(known);
            }
        }
    }
    dispatch(std::move(completions));
    return result;
}

notif_delivery_status_t
notif_delivery_registry_t::failMatchingLocked(const std::function<bool(const record_t &)> &matches,
                                              nixl_status_t status,
                                              std::uint64_t timestamp_ns,
                                              std::vector<completion_t> &completions) {
    if (status >= NIXL_SUCCESS) {
        return notif_delivery_status_t::INVALID_ARGUMENT;
    }
    for (auto current = outstanding_.begin(); current != outstanding_.end();) {
        if (!matches(current->second)) {
            ++current;
            continue;
        }
        completions.push_back({std::move(current->second.terminal), status, timestamp_ns});
        rememberCompletedLocked(current->second.key, current->second.receiptComplete);
        current = outstanding_.erase(current);
    }
    return notif_delivery_status_t::SUCCESS;
}

notif_delivery_status_t
notif_delivery_registry_t::failConnection(std::uint64_t connection_identity,
                                          nixl_status_t status,
                                          std::uint64_t timestamp_ns) {
    std::vector<completion_t> completions;
    notif_delivery_status_t result;
    {
        const std::lock_guard lock(mutex_);
        result = failMatchingLocked(
            [connection_identity](const record_t &record) {
                return record.key.connectionIdentity == connection_identity;
            },
            status,
            timestamp_ns,
            completions);
    }
    dispatch(std::move(completions));
    return result;
}

notif_delivery_status_t
notif_delivery_registry_t::failRoute(const notif_route_key_t &route,
                                     nixl_status_t status,
                                     std::uint64_t timestamp_ns) {
    std::vector<completion_t> completions;
    notif_delivery_status_t result;
    {
        const std::lock_guard lock(mutex_);
        result = failMatchingLocked(
            [&route](const record_t &record) { return record.key.route == route; },
            status,
            timestamp_ns,
            completions);
    }
    dispatch(std::move(completions));
    return result;
}

notif_delivery_status_t
notif_delivery_registry_t::failSupersededAuthority(const notif_route_key_t &route,
                                                   const notif_wire_uuid_t &capability,
                                                   std::uint64_t capability_epoch,
                                                   nixl_status_t status,
                                                   std::uint64_t timestamp_ns) {
    if (!isCanonicalNotifWireUuid(capability) || capability_epoch == 0) {
        return notif_delivery_status_t::INVALID_ARGUMENT;
    }
    std::vector<completion_t> completions;
    notif_delivery_status_t result;
    {
        const std::lock_guard lock(mutex_);
        result = failMatchingLocked(
            [&route, &capability, capability_epoch](const record_t &record) {
                return record.key.route == route &&
                    (record.key.capability != capability ||
                     record.key.capabilityEpoch != capability_epoch);
            },
            status,
            timestamp_ns,
            completions);
    }
    dispatch(std::move(completions));
    return result;
}

notif_delivery_status_t
notif_delivery_registry_t::failDelivery(std::uint64_t identity,
                                        nixl_status_t status,
                                        std::uint64_t timestamp_ns) {
    std::vector<completion_t> completions;
    notif_delivery_status_t result;
    {
        const std::lock_guard lock(mutex_);
        if (outstanding_.find(identity) == outstanding_.end()) {
            return completed_.contains(identity) ? notif_delivery_status_t::DUPLICATE_DELIVERY :
                                                   notif_delivery_status_t::UNKNOWN_DELIVERY;
        }
        result = failMatchingLocked(
            [identity](const record_t &record) { return record.key.identity == identity; },
            status,
            timestamp_ns,
            completions);
    }
    dispatch(std::move(completions));
    return result;
}

notif_delivery_status_t
notif_delivery_registry_t::failAll(nixl_status_t status, std::uint64_t timestamp_ns) {
    std::vector<completion_t> completions;
    notif_delivery_status_t result;
    {
        const std::lock_guard lock(mutex_);
        accepting_ = false;
        result = failMatchingLocked(
            [](const record_t &) { return true; }, status, timestamp_ns, completions);
    }
    dispatch(std::move(completions));
    return result;
}

notif_delivery_inventory_t
notif_delivery_registry_t::inventory() const noexcept {
    const std::lock_guard lock(mutex_);
    return {
        .outstanding = outstanding_.size(),
        .completed = completed_.size(),
        .accepting = accepting_,
    };
}

notif_delivery_snapshot_t
notif_delivery_registry_t::snapshot() const {
    notif_delivery_snapshot_t result;
    {
        const std::lock_guard lock(mutex_);
        result.inventory = {
            .outstanding = outstanding_.size(),
            .completed = completed_.size(),
            .accepting = accepting_,
        };
        result.activeRecords.reserve(outstanding_.size());
        for (const auto &[identity, record] : outstanding_) {
            static_cast<void>(identity);
            result.activeRecords.push_back({
                .key = record.key,
                .localPending = !record.localComplete,
                .receiptPending = !record.receiptComplete,
            });
        }
    }
    std::sort(result.activeRecords.begin(),
              result.activeRecords.end(),
              [](const notif_delivery_inventory_record_t &left,
                 const notif_delivery_inventory_record_t &right) {
                  return left.key.identity < right.key.identity;
              });
    return result;
}

std::vector<notif_delivery_inventory_record_t>
notif_delivery_registry_t::activeRecords() const {
    return snapshot().activeRecords;
}

void
notif_delivery_registry_t::rememberCompletedLocked(const notif_delivery_key_t &key,
                                                   bool receipt_accepted) {
    if (completedCapacity_ == 0) {
        return;
    }
    while (completedOrder_.size() >= completedCapacity_) {
        completed_.erase(completedOrder_.front());
        completedOrder_.pop_front();
    }
    completed_[key.identity] = {.key = key, .receiptAccepted = receipt_accepted};
    completedOrder_.push_back(key.identity);
}

void
notif_delivery_registry_t::dispatch(std::vector<completion_t> completions) noexcept {
    for (completion_t &completion : completions) {
        completion.terminal(completion.status, completion.timestampNs);
    }
}

} // namespace nixl::ucx
