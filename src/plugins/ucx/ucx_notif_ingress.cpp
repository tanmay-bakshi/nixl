/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "ucx_notif_ingress.h"

#include <openssl/evp.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <utility>

namespace nixl::ucx {
namespace {

    std::size_t
    combineHash(std::size_t seed, std::size_t value) noexcept {
        return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U));
    }

    std::size_t
    uuidHash(const notif_wire_uuid_t &uuid) noexcept {
        std::size_t result = 0;
        for (std::uint8_t byte : uuid.bytes) {
            result = combineHash(result, byte);
        }
        return result;
    }

} // namespace

notif_ingress_registry_t::notif_ingress_registry_t(std::size_t capacity,
                                                   std::size_t completed_capacity)
    : capacity_(capacity),
      completedCapacity_(completed_capacity == 0 ? capacity : completed_capacity) {
    if (capacity_ == 0 || completedCapacity_ == 0) {
        throw std::invalid_argument("notification ingress registry requires positive bounds");
    }
}

std::size_t
notif_ingress_registry_t::identity_hash_t::operator()(const identity_t &identity) const noexcept {
    std::size_t result = uuidHash(identity.sourceBackendIncarnation);
    result = combineHash(result, identity.sourceHandleIdentity);
    result = combineHash(result, identity.sourceGeneration);
    return combineHash(result, identity.deliveryIdentity);
}

std::size_t
notif_ingress_registry_t::route_hash_t::operator()(const notif_route_key_t &route) const noexcept {
    std::size_t result = combineHash(route.handleIdentity, route.handleGeneration);
    result = combineHash(result, uuidHash(route.remoteAgentIncarnation));
    return combineHash(result, uuidHash(route.remoteBackendIncarnation));
}

std::size_t
notif_ingress_registry_t::source_epoch_hash_t::operator()(
    const source_epoch_t &source) const noexcept {
    std::size_t result = uuidHash(source.sourceBackendIncarnation);
    result = combineHash(result, source.sourceHandleIdentity);
    return combineHash(result, source.sourceGeneration);
}

notif_ingress_registry_t::identity_t
notif_ingress_registry_t::identity(const notif_ingress_key_t &key) noexcept {
    return {
        .sourceBackendIncarnation = key.sourceBackendIncarnation,
        .sourceHandleIdentity = key.sourceHandleIdentity,
        .sourceGeneration = key.sourceGeneration,
        .deliveryIdentity = key.deliveryIdentity,
    };
}

notif_ingress_registry_t::source_epoch_t
notif_ingress_registry_t::sourceEpoch(const notif_ingress_key_t &key) noexcept {
    return {
        .sourceBackendIncarnation = key.sourceBackendIncarnation,
        .sourceHandleIdentity = key.sourceHandleIdentity,
        .sourceGeneration = key.sourceGeneration,
    };
}

bool
notif_ingress_registry_t::digest(std::span<const std::uint8_t> payload,
                                 std::array<std::uint8_t, 32> &result) noexcept {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(),
                                                                    &EVP_MD_CTX_free);
    if (context == nullptr || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(context.get(), payload.data(), payload.size()) != 1) {
        return false;
    }
    unsigned int size = 0;
    return EVP_DigestFinal_ex(context.get(), result.data(), &size) == 1 && size == result.size();
}

bool
notif_ingress_registry_t::valid(const notif_ingress_key_t &key,
                                const notif_wire_envelope_t &receipt) noexcept {
    return isCanonicalNotifWireUuid(key.sourceBackendIncarnation) &&
        key.sourceBackendIncarnation == key.route.remoteBackendIncarnation &&
        key.sourceHandleIdentity != 0 && key.sourceGeneration != 0 && key.deliveryIdentity != 0 &&
        isCanonicalNotifWireUuid(key.destinationAgentIncarnation) &&
        isCanonicalNotifWireUuid(key.destinationBackendIncarnation) &&
        key.route.handleIdentity != 0 && key.route.handleGeneration != 0 &&
        isCanonicalNotifWireUuid(key.route.remoteAgentIncarnation) &&
        isCanonicalNotifWireUuid(key.route.remoteBackendIncarnation) &&
        isCanonicalNotifWireUuid(key.capability) && key.capabilityEpoch != 0 &&
        key.connectionIdentity != 0 && isCanonicalNotifWireUuid(key.sourceWorkerIncarnation) &&
        isCanonicalNotifWireUuid(key.receiptWorkerIncarnation) && key.endpointIdentity != 0 &&
        receipt.type == notif_wire_type_t::DATA_RECEIPT &&
        receipt.senderAgentIncarnation == key.destinationAgentIncarnation &&
        receipt.recipientAgentIncarnation == key.route.remoteAgentIncarnation &&
        receipt.senderBackendIncarnation == key.destinationBackendIncarnation &&
        receipt.recipientBackendIncarnation == key.sourceBackendIncarnation &&
        receipt.senderWorkerIncarnation == key.receiptWorkerIncarnation &&
        receipt.capability == key.capability && receipt.capabilityEpoch == key.capabilityEpoch &&
        receipt.deliveryIdentity == key.deliveryIdentity &&
        receipt.sourceHandleIdentity == key.sourceHandleIdentity &&
        receipt.sourceGeneration == key.sourceGeneration;
}

bool
notif_ingress_registry_t::matches(const replay_evidence_t &evidence,
                                  const notif_ingress_key_t &key,
                                  const std::array<std::uint8_t, 32> &payload_digest,
                                  const notif_wire_envelope_t *receipt) noexcept {
    return evidence.key == key && evidence.payloadDigest == payload_digest &&
        (receipt == nullptr || evidence.receipt == *receipt);
}

notif_ingress_status_t
notif_ingress_registry_t::reserve(const notif_ingress_key_t &key,
                                  std::span<const std::uint8_t> payload,
                                  const notif_wire_envelope_t &receipt) {
    if (!valid(key, receipt)) {
        return notif_ingress_status_t::INVALID_ARGUMENT;
    }
    std::array<std::uint8_t, 32> payload_digest{};
    if (!digest(payload, payload_digest)) {
        return notif_ingress_status_t::INTERNAL_ERROR;
    }

    const identity_t primary = identity(key);
    const std::lock_guard lock(mutex_);
    const auto active = active_.find(primary);
    if (active != active_.end()) {
        if (!matches(active->second, key, payload_digest, &receipt)) {
            return notif_ingress_status_t::IDENTITY_CONFLICT;
        }
        return active->second.phase == phase_t::COMMITTED ?
            notif_ingress_status_t::DUPLICATE_COMMITTED :
            notif_ingress_status_t::DUPLICATE_PENDING;
    }
    const auto completed = completed_.find(primary);
    if (completed != completed_.end()) {
        return matches(completed->second, key, payload_digest, &receipt) ?
            notif_ingress_status_t::DUPLICATE_COMMITTED :
            notif_ingress_status_t::IDENTITY_CONFLICT;
    }
    const auto quarantined = quarantined_.find(primary);
    if (quarantined != quarantined_.end()) {
        return matches(quarantined->second, key, payload_digest, &receipt) ?
            notif_ingress_status_t::RECEIPT_FAILED :
            notif_ingress_status_t::IDENTITY_CONFLICT;
    }
    if (closed_ || terminalRoutes_.contains(key.route) ||
        terminalConnections_.contains(key.connectionIdentity)) {
        return notif_ingress_status_t::ROUTE_TERMINAL;
    }
    const source_epoch_t source = sourceEpoch(key);
    const auto high_water = highWatermarks_.find(source);
    if (high_water != highWatermarks_.end() && key.deliveryIdentity <= high_water->second) {
        return notif_ingress_status_t::ROUTE_TERMINAL;
    }
    if (active_.size() + replaySends_.size() >= capacity_) {
        return notif_ingress_status_t::CAPACITY_EXCEEDED;
    }
    active_.emplace(primary,
                    record_t{
                        {
                            .key = key,
                            .payloadDigest = payload_digest,
                            .receipt = receipt,
                        },
                    });
    highWatermarks_[source] = key.deliveryIdentity;
    return notif_ingress_status_t::SUCCESS;
}

notif_ingress_status_t
notif_ingress_registry_t::admit(
    const notif_ingress_key_t &key,
    std::span<const std::uint8_t> payload,
    nixlAuthenticatedNotification notification,
    const std::function<nixl_status_t(nixlAuthenticatedNotification &&)> &push,
    notif_wire_envelope_t &receipt) {
    if (!push) {
        return notif_ingress_status_t::INVALID_ARGUMENT;
    }
    std::array<std::uint8_t, 32> payload_digest{};
    if (!digest(payload, payload_digest)) {
        return notif_ingress_status_t::INTERNAL_ERROR;
    }
    const identity_t primary = identity(key);
    {
        const std::lock_guard lock(mutex_);
        const auto known = active_.find(primary);
        if (known == active_.end()) {
            const auto completed = completed_.find(primary);
            if (completed == completed_.end()) {
                return notif_ingress_status_t::UNKNOWN_DELIVERY;
            }
            if (!matches(completed->second, key, payload_digest)) {
                return notif_ingress_status_t::IDENTITY_CONFLICT;
            }
            receipt = completed->second.receipt;
            if (replaySends_.contains(primary)) {
                return notif_ingress_status_t::DUPLICATE_PENDING;
            }
            if (active_.size() + replaySends_.size() >= capacity_) {
                return notif_ingress_status_t::CAPACITY_EXCEEDED;
            }
            replaySends_.emplace(primary, replay_send_t{.key = key});
            return notif_ingress_status_t::DUPLICATE_COMMITTED;
        }
        if (!matches(known->second, key, payload_digest)) {
            return notif_ingress_status_t::IDENTITY_CONFLICT;
        }
        receipt = known->second.receipt;
        if (known->second.phase != phase_t::PENDING) {
            return notif_ingress_status_t::DUPLICATE_PENDING;
        }
        known->second.phase = phase_t::ADMITTING;
    }

    nixl_status_t push_status = NIXL_ERR_BACKEND;
    bool push_threw = false;
    try {
        push_status = push(std::move(notification));
    }
    catch (...) {
        push_threw = true;
    }

    const std::lock_guard lock(mutex_);
    const auto known = active_.find(primary);
    if (known == active_.end() || !matches(known->second, key, payload_digest) ||
        known->second.phase != phase_t::ADMITTING) {
        return notif_ingress_status_t::INTERNAL_ERROR;
    }
    if (push_threw) {
        quarantined_[primary] = known->second;
        active_.erase(known);
        drained_.notify_all();
        return notif_ingress_status_t::INTERNAL_ERROR;
    }
    if (push_status != NIXL_SUCCESS) {
        if (closed_ || terminalRoutes_.contains(key.route) ||
            terminalConnections_.contains(key.connectionIdentity)) {
            active_.erase(known);
            drained_.notify_all();
            return notif_ingress_status_t::ROUTE_TERMINAL;
        }
        known->second.phase = phase_t::PENDING;
        return notif_ingress_status_t::CAPACITY_EXCEEDED;
    }
    known->second.phase = phase_t::COMMITTED;
    known->second.receiptClaimed = true;
    return notif_ingress_status_t::SUCCESS;
}

notif_ingress_status_t
notif_ingress_registry_t::completeReceipt(const notif_ingress_key_t &key, nixl_status_t status) {
    const identity_t primary = identity(key);
    const std::lock_guard lock(mutex_);
    const auto known = active_.find(primary);
    if (known != active_.end()) {
        if (known->second.key != key) {
            return notif_ingress_status_t::IDENTITY_CONFLICT;
        }
        if (known->second.phase != phase_t::COMMITTED || !known->second.receiptClaimed) {
            return notif_ingress_status_t::INVALID_TRANSITION;
        }
        if (status != NIXL_SUCCESS) {
            quarantined_[primary] = known->second;
            active_.erase(known);
            drained_.notify_all();
            return notif_ingress_status_t::RECEIPT_FAILED;
        }
        if (!rememberCompletedLocked(known->second)) {
            quarantined_[primary] = known->second;
            active_.erase(known);
            drained_.notify_all();
            return notif_ingress_status_t::CAPACITY_EXCEEDED;
        }
        active_.erase(known);
        drained_.notify_all();
        return notif_ingress_status_t::SUCCESS;
    }

    const auto replay = replaySends_.find(primary);
    if (replay != replaySends_.end()) {
        if (replay->second.key != key) {
            return notif_ingress_status_t::IDENTITY_CONFLICT;
        }
        replaySends_.erase(replay);
        drained_.notify_all();
        if (status != NIXL_SUCCESS) {
            const auto completed = completed_.find(primary);
            if (completed != completed_.end()) {
                quarantined_[primary] = completed->second;
                completed_.erase(completed);
                completedOrder_.erase(
                    std::remove(completedOrder_.begin(), completedOrder_.end(), primary),
                    completedOrder_.end());
            }
            return notif_ingress_status_t::RECEIPT_FAILED;
        }
        return notif_ingress_status_t::SUCCESS;
    }
    return completed_.contains(primary) ? notif_ingress_status_t::DUPLICATE_COMMITTED :
                                          notif_ingress_status_t::UNKNOWN_DELIVERY;
}

notif_ingress_status_t
notif_ingress_registry_t::retainReceiptSlot(const notif_ingress_key_t &key,
                                            std::shared_ptr<ucx_callback_slot_t> slot) {
    if (slot == nullptr) {
        return notif_ingress_status_t::INVALID_ARGUMENT;
    }
    const identity_t primary = identity(key);
    const std::lock_guard lock(mutex_);
    const auto known = active_.find(primary);
    if (known != active_.end()) {
        if (known->second.key != key) {
            return notif_ingress_status_t::IDENTITY_CONFLICT;
        }
        if (known->second.phase != phase_t::COMMITTED || !known->second.receiptClaimed ||
            known->second.receiptSlot != nullptr) {
            return notif_ingress_status_t::INVALID_TRANSITION;
        }
        known->second.receiptSlot = std::move(slot);
        return notif_ingress_status_t::SUCCESS;
    }
    const auto replay = replaySends_.find(primary);
    if (replay == replaySends_.end()) {
        return notif_ingress_status_t::UNKNOWN_DELIVERY;
    }
    if (replay->second.key != key) {
        return notif_ingress_status_t::IDENTITY_CONFLICT;
    }
    if (replay->second.receiptSlot != nullptr) {
        return notif_ingress_status_t::INVALID_TRANSITION;
    }
    replay->second.receiptSlot = std::move(slot);
    return notif_ingress_status_t::SUCCESS;
}

notif_ingress_status_t
notif_ingress_registry_t::failRoute(const notif_route_key_t &route) {
    const std::lock_guard lock(mutex_);
    terminalRoutes_.insert(route);
    for (auto current = active_.begin(); current != active_.end();) {
        if (current->second.key.route != route || current->second.phase != phase_t::PENDING) {
            ++current;
            continue;
        }
        current = active_.erase(current);
    }
    drained_.notify_all();
    return notif_ingress_status_t::SUCCESS;
}

notif_ingress_status_t
notif_ingress_registry_t::failConnection(std::uint64_t connection_identity) {
    if (connection_identity == 0) {
        return notif_ingress_status_t::INVALID_ARGUMENT;
    }
    const std::lock_guard lock(mutex_);
    terminalConnections_.insert(connection_identity);
    for (auto current = active_.begin(); current != active_.end();) {
        if (current->second.key.connectionIdentity != connection_identity ||
            current->second.phase != phase_t::PENDING) {
            ++current;
            continue;
        }
        current = active_.erase(current);
    }
    drained_.notify_all();
    return notif_ingress_status_t::SUCCESS;
}

notif_ingress_status_t
notif_ingress_registry_t::failAll() {
    const std::lock_guard lock(mutex_);
    closed_ = true;
    for (auto current = active_.begin(); current != active_.end();) {
        if (current->second.phase != phase_t::PENDING) {
            ++current;
            continue;
        }
        current = active_.erase(current);
    }
    drained_.notify_all();
    return notif_ingress_status_t::SUCCESS;
}

notif_ingress_status_t
notif_ingress_registry_t::drainRoute(const notif_route_key_t &route,
                                     std::chrono::nanoseconds timeout) const {
    if (timeout <= std::chrono::nanoseconds::zero()) {
        return notif_ingress_status_t::INVALID_ARGUMENT;
    }
    std::unique_lock lock(mutex_);
    return drained_.wait_for(lock, timeout, [this, &route] { return routeDrainedLocked(route); }) ?
        notif_ingress_status_t::SUCCESS :
        notif_ingress_status_t::DRAIN_TIMEOUT;
}

notif_ingress_status_t
notif_ingress_registry_t::drainConnection(std::uint64_t connection_identity,
                                          std::chrono::nanoseconds timeout) const {
    if (connection_identity == 0 || timeout <= std::chrono::nanoseconds::zero()) {
        return notif_ingress_status_t::INVALID_ARGUMENT;
    }
    std::unique_lock lock(mutex_);
    return drained_.wait_for(lock, timeout, [this, connection_identity] {
        return connectionDrainedLocked(connection_identity);
    }) ? notif_ingress_status_t::SUCCESS : notif_ingress_status_t::DRAIN_TIMEOUT;
}

notif_ingress_status_t
notif_ingress_registry_t::retireRoute(const notif_route_key_t &route) {
    const std::lock_guard lock(mutex_);
    if (std::any_of(active_.begin(),
                    active_.end(),
                    [&route](const auto &entry) { return entry.second.key.route == route; }) ||
        std::any_of(replaySends_.begin(), replaySends_.end(), [&route](const auto &entry) {
            return entry.second.key.route == route;
        })) {
        return notif_ingress_status_t::INVALID_TRANSITION;
    }
    terminalRoutes_.erase(route);
    for (auto current = completed_.begin(); current != completed_.end();) {
        if (current->second.key.route != route) {
            ++current;
            continue;
        }
        completedOrder_.erase(
            std::remove(completedOrder_.begin(), completedOrder_.end(), current->first),
            completedOrder_.end());
        current = completed_.erase(current);
    }
    std::unordered_set<source_epoch_t, source_epoch_hash_t> retained_sources;
    for (const auto &[primary, evidence] : completed_) {
        static_cast<void>(primary);
        retained_sources.insert(sourceEpoch(evidence.key));
    }
    for (const auto &[primary, evidence] : quarantined_) {
        static_cast<void>(primary);
        retained_sources.insert(sourceEpoch(evidence.key));
    }
    for (auto current = highWatermarks_.begin(); current != highWatermarks_.end();) {
        const bool belongs_to_route =
            current->first.sourceBackendIncarnation == route.remoteBackendIncarnation &&
            current->first.sourceHandleIdentity == route.handleIdentity &&
            current->first.sourceGeneration == route.handleGeneration;
        current = belongs_to_route && !retained_sources.contains(current->first) ?
            highWatermarks_.erase(current) :
            std::next(current);
    }
    return notif_ingress_status_t::SUCCESS;
}

notif_ingress_status_t
notif_ingress_registry_t::retireConnection(std::uint64_t connection_identity) {
    if (connection_identity == 0) {
        return notif_ingress_status_t::INVALID_ARGUMENT;
    }
    const std::lock_guard lock(mutex_);
    if (std::any_of(active_.begin(),
                    active_.end(),
                    [connection_identity](const auto &entry) {
                        return entry.second.key.connectionIdentity == connection_identity;
                    }) ||
        std::any_of(
            replaySends_.begin(), replaySends_.end(), [connection_identity](const auto &entry) {
                return entry.second.key.connectionIdentity == connection_identity;
            })) {
        return notif_ingress_status_t::INVALID_TRANSITION;
    }
    terminalConnections_.erase(connection_identity);
    for (auto current = completed_.begin(); current != completed_.end();) {
        if (current->second.key.connectionIdentity != connection_identity) {
            ++current;
            continue;
        }
        completedOrder_.erase(
            std::remove(completedOrder_.begin(), completedOrder_.end(), current->first),
            completedOrder_.end());
        current = completed_.erase(current);
    }
    return notif_ingress_status_t::SUCCESS;
}

notif_ingress_inventory_t
notif_ingress_registry_t::inventory() const noexcept {
    const std::lock_guard lock(mutex_);
    notif_ingress_inventory_t result = {
        .replaying = replaySends_.size(),
        .completed = completed_.size(),
        .quarantined = quarantined_.size(),
    };
    for (const auto &[primary, record] : active_) {
        static_cast<void>(primary);
        switch (record.phase) {
        case phase_t::PENDING:
            ++result.pending;
            break;
        case phase_t::ADMITTING:
            ++result.admitting;
            break;
        case phase_t::COMMITTED:
            ++result.committed;
            break;
        }
    }
    return result;
}

notif_ingress_snapshot_t
notif_ingress_registry_t::snapshot() const {
    notif_ingress_snapshot_t result;
    {
        const std::lock_guard lock(mutex_);
        result.inventory = {
            .replaying = replaySends_.size(),
            .completed = completed_.size(),
            .quarantined = quarantined_.size(),
        };
        result.activeRecords.reserve(active_.size() + replaySends_.size() + quarantined_.size());
        for (const auto &[primary, record] : active_) {
            static_cast<void>(primary);
            notif_ingress_inventory_phase_t phase = notif_ingress_inventory_phase_t::PENDING;
            if (record.phase == phase_t::ADMITTING) {
                phase = notif_ingress_inventory_phase_t::ADMITTING;
                ++result.inventory.admitting;
            } else if (record.phase == phase_t::COMMITTED) {
                phase = notif_ingress_inventory_phase_t::COMMITTED;
                ++result.inventory.committed;
            } else {
                ++result.inventory.pending;
            }
            result.activeRecords.push_back({.key = record.key, .phase = phase});
        }
        for (const auto &[primary, record] : replaySends_) {
            static_cast<void>(primary);
            result.activeRecords.push_back({
                .key = record.key,
                .phase = notif_ingress_inventory_phase_t::REPLAYING,
            });
        }
        for (const auto &[primary, evidence] : quarantined_) {
            static_cast<void>(primary);
            result.activeRecords.push_back({
                .key = evidence.key,
                .phase = notif_ingress_inventory_phase_t::QUARANTINED,
            });
        }
    }
    std::sort(
        result.activeRecords.begin(),
        result.activeRecords.end(),
        [](const notif_ingress_inventory_record_t &left,
           const notif_ingress_inventory_record_t &right) {
            const auto &left_uuid = left.key.sourceBackendIncarnation.bytes;
            const auto &right_uuid = right.key.sourceBackendIncarnation.bytes;
            if (left_uuid != right_uuid) {
                return left_uuid < right_uuid;
            }
            if (left.key.sourceHandleIdentity != right.key.sourceHandleIdentity) {
                return left.key.sourceHandleIdentity < right.key.sourceHandleIdentity;
            }
            if (left.key.sourceGeneration != right.key.sourceGeneration) {
                return left.key.sourceGeneration < right.key.sourceGeneration;
            }
            return left.key.deliveryIdentity < right.key.deliveryIdentity;
        });
    return result;
}

std::vector<notif_ingress_inventory_record_t>
notif_ingress_registry_t::activeRecords() const {
    return snapshot().activeRecords;
}

bool
notif_ingress_registry_t::routeDrainedLocked(const notif_route_key_t &route) const noexcept {
    return std::none_of(active_.begin(),
                        active_.end(),
                        [&route](const auto &entry) { return entry.second.key.route == route; }) &&
        std::none_of(replaySends_.begin(), replaySends_.end(), [&route](const auto &entry) {
               return entry.second.key.route == route;
           });
}

bool
notif_ingress_registry_t::connectionDrainedLocked(
    std::uint64_t connection_identity) const noexcept {
    return std::none_of(active_.begin(),
                        active_.end(),
                        [connection_identity](const auto &entry) {
                            return entry.second.key.connectionIdentity == connection_identity;
                        }) &&
        std::none_of(
               replaySends_.begin(), replaySends_.end(), [connection_identity](const auto &entry) {
                   return entry.second.key.connectionIdentity == connection_identity;
               });
}

bool
notif_ingress_registry_t::rememberCompletedLocked(const replay_evidence_t &evidence) {
    while (completedOrder_.size() >= completedCapacity_) {
        const auto evictable = std::find_if(
            completedOrder_.begin(), completedOrder_.end(), [this](const identity_t &candidate) {
                return !replaySends_.contains(candidate);
            });
        if (evictable == completedOrder_.end()) {
            return false;
        }
        completed_.erase(*evictable);
        completedOrder_.erase(evictable);
    }
    const identity_t primary = identity(evidence.key);
    completed_[primary] = evidence;
    completedOrder_.push_back(primary);
    return true;
}

} // namespace nixl::ucx
