/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ucx_notif_state.h"

#include <algorithm>
#include <condition_variable>
#include <ctime>
#include <deque>
#include <limits>
#include <stdexcept>
#include <utility>

#include "common/uuid_v4.h"

namespace nixl::ucx {
namespace {

    constexpr std::size_t capability_generation_attempts = 16;

    [[nodiscard]] std::size_t
    combineHash(std::size_t seed, std::size_t value) noexcept {
        return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U));
    }

    [[nodiscard]] bool
    uuidLess(const notif_wire_uuid_t &left, const notif_wire_uuid_t &right) noexcept {
        return left.bytes < right.bytes;
    }

    [[nodiscard]] bool
    isValidRoute(const notif_route_key_t &route) noexcept {
        return route.handleIdentity != 0 && route.handleGeneration != 0 &&
            isCanonicalNotifWireUuid(route.remoteAgentIncarnation) &&
            isCanonicalNotifWireUuid(route.remoteBackendIncarnation);
    }

    [[nodiscard]] notif_wire_uuid_t
    generateCapability() {
        const UUIDv4 generated;
        return {.bytes = generated.get_data()};
    }

    [[nodiscard]] std::uint64_t
    monotonicRawTimestampNs() noexcept {
        timespec timestamp = {};
        if (::clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp) != 0) {
            return 0;
        }
        return static_cast<std::uint64_t>(timestamp.tv_sec) * 1'000'000'000ULL +
            static_cast<std::uint64_t>(timestamp.tv_nsec);
    }

} // namespace

class notif_capability_state_t::route_delivery_state_t {
public:
    explicit route_delivery_state_t(std::shared_ptr<notif_route_transition_sink_t> sink)
        : sink_(std::move(sink)) {}

    [[nodiscard]] bool
    enqueue(notif_route_transition_t transition) {
        const std::lock_guard lock(mutex_);
        if (cancelled_) {
            return false;
        }

        // Route transitions enqueue while the route mutex is held. One dispatcher then invokes
        // the sink outside that mutex, preserving order even when a callback re-enters the state.
        pending_.push_back(std::move(transition));
        if (dispatching_) {
            return false;
        }
        dispatching_ = true;
        return true;
    }

    void
    dispatch() noexcept {
        while (true) {
            notif_route_transition_t transition;
            std::shared_ptr<notif_route_transition_sink_t> sink;
            {
                const std::lock_guard lock(mutex_);
                if (cancelled_ || pending_.empty()) {
                    dispatching_ = false;
                    return;
                }
                transition = std::move(pending_.front());
                pending_.pop_front();
                sink = sink_;
                ++callbacksInFlight_;
            }
            sink->publish(transition);
            {
                const std::lock_guard lock(mutex_);
                --callbacksInFlight_;
                if (callbacksInFlight_ == 0) {
                    callbacksDrained_.notify_all();
                }
            }
        }
    }

    void
    cancel() noexcept {
        std::unique_lock lock(mutex_);
        cancelled_ = true;
        pending_.clear();
        sink_.reset();
        callbacksDrained_.wait(lock, [this] { return callbacksInFlight_ == 0; });
    }

    [[nodiscard]] std::size_t
    inFlightCount() const noexcept {
        const std::lock_guard lock(mutex_);
        return callbacksInFlight_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable callbacksDrained_;
    std::shared_ptr<notif_route_transition_sink_t> sink_;
    std::deque<notif_route_transition_t> pending_;
    std::size_t callbacksInFlight_ = 0;
    bool dispatching_ = false;
    bool cancelled_ = false;
};

std::size_t
notif_capability_state_t::uuid_hash_t::operator()(const notif_wire_uuid_t &uuid) const noexcept {
    std::size_t hash = 1469598103934665603ULL;
    for (const std::uint8_t byte : uuid.bytes) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::size_t
notif_capability_state_t::route_hash_t::operator()(const notif_route_key_t &route) const noexcept {
    std::size_t hash = std::hash<std::uint64_t>{}(route.handleIdentity);
    hash = combineHash(hash, std::hash<std::uint64_t>{}(route.handleGeneration));
    hash = combineHash(hash, uuid_hash_t{}(route.remoteAgentIncarnation));
    return combineHash(hash, uuid_hash_t{}(route.remoteBackendIncarnation));
}

std::size_t
notif_capability_state_t::peer_hash_t::operator()(const peer_key_t &peer) const noexcept {
    return combineHash(uuid_hash_t{}(peer.agentIncarnation),
                       uuid_hash_t{}(peer.backendIncarnation));
}

notif_capability_state_t::notif_capability_state_t(
    const notif_wire_uuid_t &local_agent_incarnation,
    const notif_wire_uuid_t &local_backend_incarnation,
    std::vector<notif_wire_uuid_t> local_worker_incarnations)
    : localAgentIncarnation_(local_agent_incarnation),
      localBackendIncarnation_(local_backend_incarnation) {
    if (!isCanonicalNotifWireUuid(localAgentIncarnation_) ||
        !isCanonicalNotifWireUuid(localBackendIncarnation_) || local_worker_incarnations.empty()) {
        throw std::invalid_argument("invalid local notification identity");
    }

    for (const notif_wire_uuid_t &worker : local_worker_incarnations) {
        if (!isCanonicalNotifWireUuid(worker) || !localWorkers_.insert(worker).second) {
            throw std::invalid_argument("invalid local notification worker manifest");
        }
    }
}

bool
notif_capability_state_t::isLocalWorker(const notif_wire_uuid_t &worker) const {
    return localWorkers_.find(worker) != localWorkers_.end();
}

bool
notif_capability_state_t::hasExactHandleConflictLocked(const notif_route_key_t &route) const {
    for (const auto &[known_route, binding] : bindings_) {
        (void)binding;
        if (known_route.handleIdentity == route.handleIdentity &&
            known_route.handleGeneration == route.handleGeneration && known_route != route) {
            return true;
        }
    }
    return false;
}

notif_state_status_t
notif_capability_state_t::bindRemoteAgent(const notif_remote_binding_t &binding,
                                          notif_route_snapshot_t &snapshot) {
    notif_remote_binding_t normalized = binding;
    if (!isValidRoute(normalized.route) || normalized.connectionIdentity == 0 ||
        normalized.workers.empty()) {
        return notif_state_status_t::INVALID_ARGUMENT;
    }

    std::sort(normalized.workers.begin(),
              normalized.workers.end(),
              [](const notif_remote_worker_t &left, const notif_remote_worker_t &right) {
                  return uuidLess(left.incarnation, right.incarnation);
              });
    for (std::size_t index = 0; index < normalized.workers.size(); ++index) {
        const notif_remote_worker_t &worker = normalized.workers[index];
        if (!isCanonicalNotifWireUuid(worker.incarnation) || worker.endpointIdentity == 0 ||
            (index > 0 && worker.incarnation == normalized.workers[index - 1].incarnation)) {
            return notif_state_status_t::INVALID_ARGUMENT;
        }
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const auto known = bindings_.find(normalized.route);
    if (known != bindings_.end()) {
        if (known->second.binding != normalized) {
            return notif_state_status_t::ROUTE_CONFLICT;
        }
        snapshot = makeSnapshotLocked(known->second);
        return known->second.retired ? notif_state_status_t::ROUTE_RETIRED :
                                       notif_state_status_t::SUCCESS;
    }
    if (hasExactHandleConflictLocked(normalized.route)) {
        return notif_state_status_t::ROUTE_CONFLICT;
    }

    const peer_key_t peer = {
        .agentIncarnation = normalized.route.remoteAgentIncarnation,
        .backendIncarnation = normalized.route.remoteBackendIncarnation,
    };
    if (activePeers_.find(peer) != activePeers_.end()) {
        return notif_state_status_t::ROUTE_CONFLICT;
    }
    if (nextCapabilityEpoch_ == std::numeric_limits<std::uint64_t>::max()) {
        return notif_state_status_t::EPOCH_EXHAUSTED;
    }

    notif_wire_uuid_t capability;
    bool generated = false;
    for (std::size_t attempt = 0; attempt < capability_generation_attempts; ++attempt) {
        capability = generateCapability();
        if (localCapabilities_.find(capability) == localCapabilities_.end()) {
            generated = true;
            break;
        }
    }
    if (!generated) {
        return notif_state_status_t::RANDOM_FAILURE;
    }

    binding_state_t state;
    state.binding = std::move(normalized);
    for (const notif_remote_worker_t &worker : state.binding.workers) {
        state.remoteWorkers.emplace(worker.incarnation, worker.endpointIdentity);
    }
    state.localCapability = capability;
    state.localCapabilityEpoch = nextCapabilityEpoch_++;

    const notif_route_key_t route = state.binding.route;
    const auto [inserted, did_insert] = bindings_.emplace(route, std::move(state));
    if (!did_insert) {
        return notif_state_status_t::ROUTE_CONFLICT;
    }
    activePeers_.emplace(peer, route);
    localCapabilities_.emplace(capability, route);
    snapshot = makeSnapshotLocked(inserted->second);
    return notif_state_status_t::SUCCESS;
}

notif_state_status_t
notif_capability_state_t::retireRemoteAgent(const notif_route_key_t &route) {
    pending_route_delivery_t pending;
    {
        const std::lock_guard lock(mutex_);
        const auto known = bindings_.find(route);
        if (known == bindings_.end()) {
            return notif_state_status_t::UNKNOWN_ROUTE;
        }
        if (known->second.retired) {
            return notif_state_status_t::SUCCESS;
        }

        known->second.retired = true;
        const peer_key_t peer = {
            .agentIncarnation = route.remoteAgentIncarnation,
            .backendIncarnation = route.remoteBackendIncarnation,
        };
        const auto active = activePeers_.find(peer);
        if (active != activePeers_.end() && active->second == route) {
            activePeers_.erase(active);
        }
        const std::uint64_t epoch = known->second.remoteCapabilityEpoch != 0 ?
            known->second.remoteCapabilityEpoch :
            known->second.localCapabilityEpoch;
        pending =
            enqueueRouteTransitionLocked(route, notif_route_transition_state_t::RETIRED, epoch);
    }
    dispatchRouteTransition(std::move(pending));
    return notif_state_status_t::SUCCESS;
}

notif_state_status_t
notif_capability_state_t::failRemoteAgent(const notif_route_key_t &route) {
    pending_route_delivery_t pending;
    {
        const std::lock_guard lock(mutex_);
        const auto known = bindings_.find(route);
        if (known == bindings_.end()) {
            return notif_state_status_t::UNKNOWN_ROUTE;
        }
        if (known->second.retired || known->second.failed) {
            return notif_state_status_t::SUCCESS;
        }

        known->second.failed = true;
        const peer_key_t peer = {
            .agentIncarnation = route.remoteAgentIncarnation,
            .backendIncarnation = route.remoteBackendIncarnation,
        };
        const auto active = activePeers_.find(peer);
        if (active != activePeers_.end() && active->second == route) {
            activePeers_.erase(active);
        }
        const std::uint64_t epoch = known->second.remoteCapabilityEpoch != 0 ?
            known->second.remoteCapabilityEpoch :
            known->second.localCapabilityEpoch;
        pending =
            enqueueRouteTransitionLocked(route, notif_route_transition_state_t::FAILED, epoch);
    }
    dispatchRouteTransition(std::move(pending));
    return notif_state_status_t::SUCCESS;
}

notif_route_snapshot_t
notif_capability_state_t::makeSnapshotLocked(const binding_state_t &binding) const {
    notif_route_state_t route_state = notif_route_state_t::NOT_READY;
    if (binding.failed) {
        route_state = notif_route_state_t::FAILED;
    } else if (binding.retired) {
        route_state = notif_route_state_t::RETIRED;
    } else if (binding.hasRemoteCapability) {
        route_state = notif_route_state_t::READY;
    }

    return {
        .binding = binding.binding,
        .state = route_state,
        .localCapability = binding.localCapability,
        .localCapabilityEpoch = binding.localCapabilityEpoch,
        .localOfferAcknowledged = binding.localOfferAcknowledged,
        .hasRemoteCapability = binding.hasRemoteCapability,
        .remoteCapability = binding.remoteCapability,
        .remoteCapabilityEpoch = binding.remoteCapabilityEpoch,
    };
}

notif_capability_state_t::pending_route_delivery_t
notif_capability_state_t::enqueueRouteTransitionLocked(const notif_route_key_t &route,
                                                       notif_route_transition_state_t state,
                                                       std::uint64_t capability_epoch) const {
    const auto known = routeSubscriptions_.find(route);
    if (known == routeSubscriptions_.end()) {
        return {};
    }

    pending_route_delivery_t pending = {
        .delivery = known->second.delivery,
    };
    pending.dispatchRequired = pending.delivery->enqueue({
        .route = route,
        .state = state,
        .capabilityEpoch = capability_epoch,
        .nativeTimestampNs = monotonicRawTimestampNs(),
    });
    return pending;
}

void
notif_capability_state_t::dispatchRouteTransition(pending_route_delivery_t pending) noexcept {
    if (pending.delivery == nullptr || !pending.dispatchRequired) {
        return;
    }
    pending.delivery->dispatch();
}

notif_state_status_t
notif_capability_state_t::queryRemoteNotificationState(const notif_route_key_t &route,
                                                       notif_route_snapshot_t &snapshot) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto known = bindings_.find(route);
    if (known == bindings_.end()) {
        return notif_state_status_t::UNKNOWN_ROUTE;
    }
    snapshot = makeSnapshotLocked(known->second);
    return notif_state_status_t::SUCCESS;
}

notif_route_subscription_status_t
notif_capability_state_t::subscribeRemoteNotificationState(
    const notif_route_key_t &route,
    std::shared_ptr<notif_route_transition_sink_t> sink,
    notif_route_subscription_t &subscription) {
    if (sink == nullptr) {
        return notif_route_subscription_status_t::INVALID_ARGUMENT;
    }

    pending_route_delivery_t pending;
    notif_route_subscription_t candidate;
    {
        const std::lock_guard lock(mutex_);
        const auto binding = bindings_.find(route);
        if (binding == bindings_.end()) {
            return notif_route_subscription_status_t::UNKNOWN_ROUTE;
        }
        if (routeSubscriptions_.find(route) != routeSubscriptions_.end()) {
            return notif_route_subscription_status_t::DUPLICATE_SUBSCRIPTION;
        }
        if (nextSubscriptionGeneration_ == std::numeric_limits<std::uint64_t>::max()) {
            return notif_route_subscription_status_t::GENERATION_EXHAUSTED;
        }

        candidate = {
            .route = route,
            .generation = nextSubscriptionGeneration_++,
        };
        const bool did_insert =
            routeSubscriptions_
                .emplace(route,
                         route_subscription_record_t{
                             .generation = candidate.generation,
                             .delivery = std::make_shared<route_delivery_state_t>(std::move(sink)),
                         })
                .second;
        if (!did_insert) {
            return notif_route_subscription_status_t::DUPLICATE_SUBSCRIPTION;
        }

        const binding_state_t &state = binding->second;
        const std::uint64_t epoch = state.remoteCapabilityEpoch != 0 ? state.remoteCapabilityEpoch :
                                                                       state.localCapabilityEpoch;
        if (state.failed) {
            pending =
                enqueueRouteTransitionLocked(route, notif_route_transition_state_t::FAILED, epoch);
        } else if (state.retired) {
            pending =
                enqueueRouteTransitionLocked(route, notif_route_transition_state_t::RETIRED, epoch);
        } else if (state.hasRemoteCapability) {
            pending =
                enqueueRouteTransitionLocked(route, notif_route_transition_state_t::READY, epoch);
        }
    }

    subscription = candidate;
    dispatchRouteTransition(std::move(pending));
    return notif_route_subscription_status_t::SUCCESS;
}

notif_route_subscription_status_t
notif_capability_state_t::unsubscribeRemoteNotificationState(
    const notif_route_subscription_t &subscription) noexcept {
    std::shared_ptr<route_delivery_state_t> delivery;
    {
        const std::lock_guard lock(mutex_);
        const auto known = routeSubscriptions_.find(subscription.route);
        if (known == routeSubscriptions_.end() ||
            known->second.generation != subscription.generation || known->second.closing) {
            return notif_route_subscription_status_t::UNKNOWN_SUBSCRIPTION;
        }
        known->second.closing = true;
        delivery = known->second.delivery;
    }
    delivery->cancel();

    // Keep the exact generation registered until its copied callback has drained. This makes
    // successful unsubscribe the lifecycle boundary backend inventory can safely project.
    const std::lock_guard lock(mutex_);
    const auto known = routeSubscriptions_.find(subscription.route);
    if (known != routeSubscriptions_.end() &&
        known->second.generation == subscription.generation && known->second.closing) {
        routeSubscriptions_.erase(known);
    }
    return notif_route_subscription_status_t::SUCCESS;
}

notif_route_subscription_inventory_t
notif_capability_state_t::querySubscriptionInventory(
    const notif_route_subscription_t &subscription) const noexcept {
    std::shared_ptr<route_delivery_state_t> delivery;
    {
        const std::lock_guard lock(mutex_);
        const auto known = routeSubscriptions_.find(subscription.route);
        if (known == routeSubscriptions_.end() ||
            known->second.generation != subscription.generation) {
            return {};
        }
        delivery = known->second.delivery;
    }
    return {
        .retainedSubscriptions = 1,
        .inFlightDeliveries = delivery->inFlightCount(),
    };
}

notif_wire_envelope_t
notif_capability_state_t::makeEnvelopeLocked(notif_wire_type_t type,
                                             const binding_state_t &binding,
                                             const notif_wire_uuid_t &local_sender_worker,
                                             const notif_wire_uuid_t &capability,
                                             std::uint64_t capability_epoch) const {
    return {
        .type = type,
        .senderAgentIncarnation = localAgentIncarnation_,
        .recipientAgentIncarnation = binding.binding.route.remoteAgentIncarnation,
        .senderBackendIncarnation = localBackendIncarnation_,
        .recipientBackendIncarnation = binding.binding.route.remoteBackendIncarnation,
        .senderWorkerIncarnation = local_sender_worker,
        .capability = capability,
        .capabilityEpoch = capability_epoch,
    };
}

notif_state_status_t
notif_capability_state_t::makeOffer(const notif_route_key_t &route,
                                    const notif_wire_uuid_t &local_sender_worker,
                                    notif_wire_envelope_t &offer) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto known = bindings_.find(route);
    if (known == bindings_.end()) {
        return notif_state_status_t::UNKNOWN_ROUTE;
    }
    const binding_state_t &binding = known->second;
    if (binding.retired) {
        return notif_state_status_t::ROUTE_RETIRED;
    }
    if (binding.failed) {
        return notif_state_status_t::ROUTE_FAILED;
    }
    if (!isLocalWorker(local_sender_worker)) {
        return notif_state_status_t::INVALID_ARGUMENT;
    }

    offer = makeEnvelopeLocked(notif_wire_type_t::OFFER,
                               binding,
                               local_sender_worker,
                               binding.localCapability,
                               binding.localCapabilityEpoch);
    return notif_state_status_t::SUCCESS;
}

notif_state_status_t
notif_capability_state_t::validateInboundLocked(const notif_wire_envelope_t &envelope,
                                                const binding_state_t &binding,
                                                std::uint64_t &endpoint_identity) const {
    if (envelope.senderAgentIncarnation != binding.binding.route.remoteAgentIncarnation ||
        envelope.recipientAgentIncarnation != localAgentIncarnation_ ||
        envelope.senderBackendIncarnation != binding.binding.route.remoteBackendIncarnation ||
        envelope.recipientBackendIncarnation != localBackendIncarnation_) {
        return notif_state_status_t::INVALID_ARGUMENT;
    }

    const auto worker = binding.remoteWorkers.find(envelope.senderWorkerIncarnation);
    if (worker == binding.remoteWorkers.end()) {
        return notif_state_status_t::INVALID_ARGUMENT;
    }
    endpoint_identity = worker->second;
    return notif_state_status_t::SUCCESS;
}

notif_state_status_t
notif_capability_state_t::acceptOffer(const notif_wire_envelope_t &offer,
                                      const notif_wire_uuid_t &local_sender_worker,
                                      notif_offer_acceptance_t &acceptance) {
    if (offer.type != notif_wire_type_t::OFFER || !isCanonicalNotifWireUuid(offer.capability) ||
        offer.capabilityEpoch == 0) {
        return notif_state_status_t::INVALID_ARGUMENT;
    }

    std::unique_lock<std::mutex> lock(mutex_);
    if (!isLocalWorker(local_sender_worker) ||
        offer.recipientAgentIncarnation != localAgentIncarnation_ ||
        offer.recipientBackendIncarnation != localBackendIncarnation_) {
        return notif_state_status_t::INVALID_ARGUMENT;
    }

    const peer_key_t peer = {
        .agentIncarnation = offer.senderAgentIncarnation,
        .backendIncarnation = offer.senderBackendIncarnation,
    };
    const auto active = activePeers_.find(peer);
    if (active == activePeers_.end()) {
        return notif_state_status_t::UNKNOWN_ROUTE;
    }
    const auto known = bindings_.find(active->second);
    if (known == bindings_.end()) {
        return notif_state_status_t::UNKNOWN_ROUTE;
    }

    binding_state_t &binding = known->second;
    std::uint64_t endpoint_identity = 0;
    const notif_state_status_t validation =
        validateInboundLocked(offer, binding, endpoint_identity);
    (void)endpoint_identity;
    if (validation != notif_state_status_t::SUCCESS) {
        return validation;
    }
    if (binding.retired) {
        return notif_state_status_t::ROUTE_RETIRED;
    }
    if (binding.failed) {
        return notif_state_status_t::ROUTE_FAILED;
    }

    pending_route_delivery_t pending;
    bool reemit_local_offer = false;
    bool capability_changed = false;
    if (!binding.hasRemoteCapability) {
        binding.hasRemoteCapability = true;
        binding.remoteCapability = offer.capability;
        binding.remoteCapabilityEpoch = offer.capabilityEpoch;
        reemit_local_offer = true;
        capability_changed = true;
    } else if (offer.capabilityEpoch < binding.remoteCapabilityEpoch) {
        return notif_state_status_t::STALE_EPOCH;
    } else if (offer.capabilityEpoch == binding.remoteCapabilityEpoch) {
        if (offer.capability != binding.remoteCapability) {
            binding.failed = true;
            binding.hasRemoteCapability = false;
            pending = enqueueRouteTransitionLocked(binding.binding.route,
                                                   notif_route_transition_state_t::FAILED,
                                                   offer.capabilityEpoch);
            lock.unlock();
            dispatchRouteTransition(std::move(pending));
            return notif_state_status_t::EPOCH_CONFLICT;
        }
    } else {
        binding.remoteCapability = offer.capability;
        binding.remoteCapabilityEpoch = offer.capabilityEpoch;
        reemit_local_offer = true;
        capability_changed = true;
    }

    acceptance = {
        .route = binding.binding.route,
        .acknowledgement = makeEnvelopeLocked(notif_wire_type_t::ACK,
                                              binding,
                                              local_sender_worker,
                                              offer.capability,
                                              offer.capabilityEpoch),
        .localOffer = makeEnvelopeLocked(notif_wire_type_t::OFFER,
                                         binding,
                                         local_sender_worker,
                                         binding.localCapability,
                                         binding.localCapabilityEpoch),
        .reemitLocalOffer = reemit_local_offer,
    };
    if (capability_changed) {
        pending = enqueueRouteTransitionLocked(binding.binding.route,
                                               notif_route_transition_state_t::READY,
                                               binding.remoteCapabilityEpoch);
    }
    lock.unlock();
    dispatchRouteTransition(std::move(pending));
    return notif_state_status_t::SUCCESS;
}

notif_state_status_t
notif_capability_state_t::acceptAcknowledgement(const notif_wire_envelope_t &acknowledgement,
                                                notif_route_key_t &route) {
    if (acknowledgement.type != notif_wire_type_t::ACK ||
        !isCanonicalNotifWireUuid(acknowledgement.capability) ||
        acknowledgement.capabilityEpoch == 0) {
        return notif_state_status_t::INVALID_ARGUMENT;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const auto capability = localCapabilities_.find(acknowledgement.capability);
    if (capability == localCapabilities_.end()) {
        return notif_state_status_t::UNKNOWN_CAPABILITY;
    }
    const auto known = bindings_.find(capability->second);
    if (known == bindings_.end()) {
        return notif_state_status_t::UNKNOWN_ROUTE;
    }

    binding_state_t &binding = known->second;
    std::uint64_t endpoint_identity = 0;
    const notif_state_status_t validation =
        validateInboundLocked(acknowledgement, binding, endpoint_identity);
    (void)endpoint_identity;
    if (validation != notif_state_status_t::SUCCESS ||
        acknowledgement.capabilityEpoch != binding.localCapabilityEpoch) {
        return notif_state_status_t::INVALID_ARGUMENT;
    }
    if (binding.failed) {
        return notif_state_status_t::ROUTE_FAILED;
    }

    binding.localOfferAcknowledged = true;
    route = binding.binding.route;
    return notif_state_status_t::SUCCESS;
}

notif_state_status_t
notif_capability_state_t::prepareData(const notif_route_key_t &route,
                                      const notif_wire_uuid_t &local_sender_worker,
                                      notif_wire_envelope_t &data) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto known = bindings_.find(route);
    if (known == bindings_.end()) {
        return notif_state_status_t::UNKNOWN_ROUTE;
    }
    const binding_state_t &binding = known->second;
    if (binding.retired) {
        return notif_state_status_t::ROUTE_RETIRED;
    }
    if (binding.failed) {
        return notif_state_status_t::ROUTE_FAILED;
    }
    if (!isLocalWorker(local_sender_worker)) {
        return notif_state_status_t::INVALID_ARGUMENT;
    }
    if (!binding.hasRemoteCapability) {
        return notif_state_status_t::NOT_READY;
    }

    data = makeEnvelopeLocked(notif_wire_type_t::DATA,
                              binding,
                              local_sender_worker,
                              binding.remoteCapability,
                              binding.remoteCapabilityEpoch);
    return notif_state_status_t::SUCCESS;
}

notif_data_resolution_t
notif_capability_state_t::resolveData(const notif_wire_envelope_t &data) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto capability = localCapabilities_.find(data.capability);
    if (capability == localCapabilities_.end()) {
        return {};
    }

    notif_data_resolution_t resolution = {
        .disposition = notif_data_disposition_t::DROP_ROUTE,
        .status = notif_state_status_t::UNKNOWN_ROUTE,
        .authority =
            {
                .route = capability->second,
                .senderWorkerIncarnation = data.senderWorkerIncarnation,
            },
    };
    const auto known = bindings_.find(capability->second);
    if (known == bindings_.end()) {
        return resolution;
    }

    const binding_state_t &binding = known->second;
    resolution.authority.connectionIdentity = binding.binding.connectionIdentity;
    resolution.authority.tombstoned = binding.retired;

    std::uint64_t endpoint_identity = 0;
    const notif_state_status_t validation = validateInboundLocked(data, binding, endpoint_identity);
    if (data.type != notif_wire_type_t::DATA ||
        data.capabilityEpoch != binding.localCapabilityEpoch ||
        validation != notif_state_status_t::SUCCESS) {
        resolution.status = notif_state_status_t::INVALID_ARGUMENT;
        return resolution;
    }
    resolution.authority.endpointIdentity = endpoint_identity;
    if (binding.failed) {
        resolution.status = notif_state_status_t::ROUTE_FAILED;
        return resolution;
    }

    resolution.disposition = notif_data_disposition_t::DELIVER;
    resolution.status = notif_state_status_t::SUCCESS;
    return resolution;
}

} // namespace nixl::ucx
