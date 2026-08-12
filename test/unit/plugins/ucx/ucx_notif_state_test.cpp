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

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ucx_notif_state.h"

namespace {

using namespace nixl::ucx;

void
require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

notif_wire_uuid_t
makeUuid(std::uint8_t seed) {
    notif_wire_uuid_t uuid;
    for (std::size_t index = 0; index < uuid.bytes.size(); ++index) {
        uuid.bytes[index] = static_cast<std::uint8_t>(seed + index * 11);
    }
    uuid.bytes[6] = static_cast<std::uint8_t>((uuid.bytes[6] & 0x0fU) | 0x40U);
    uuid.bytes[8] = static_cast<std::uint8_t>((uuid.bytes[8] & 0x3fU) | 0x80U);
    return uuid;
}

notif_route_key_t
makeRoute(std::uint64_t handle_identity,
          std::uint64_t handle_generation,
          const notif_wire_uuid_t &remote_agent,
          const notif_wire_uuid_t &remote_backend) {
    return {
        .handleIdentity = handle_identity,
        .handleGeneration = handle_generation,
        .remoteAgentIncarnation = remote_agent,
        .remoteBackendIncarnation = remote_backend,
    };
}

notif_remote_binding_t
makeBinding(const notif_route_key_t &route,
            std::uint64_t connection_identity,
            std::vector<notif_remote_worker_t> workers) {
    return {
        .route = route,
        .connectionIdentity = connection_identity,
        .workers = std::move(workers),
    };
}

notif_route_snapshot_t
bind(notif_capability_state_t &state, const notif_remote_binding_t &binding) {
    notif_route_snapshot_t snapshot;
    require(state.bindRemoteAgent(binding, snapshot) == notif_state_status_t::SUCCESS,
            "valid remote notification binding failed");
    require(isCanonicalNotifWireUuid(snapshot.localCapability),
            "binding did not generate a canonical capability");
    require(snapshot.localCapabilityEpoch != 0, "binding generated a zero capability epoch");
    return snapshot;
}

notif_wire_envelope_t
makeInbound(const notif_wire_envelope_t &local_offer,
            notif_wire_type_t type,
            const notif_wire_uuid_t &remote_worker,
            const notif_wire_uuid_t &capability,
            std::uint64_t capability_epoch) {
    return {
        .type = type,
        .senderAgentIncarnation = local_offer.recipientAgentIncarnation,
        .recipientAgentIncarnation = local_offer.senderAgentIncarnation,
        .senderBackendIncarnation = local_offer.recipientBackendIncarnation,
        .recipientBackendIncarnation = local_offer.senderBackendIncarnation,
        .senderWorkerIncarnation = remote_worker,
        .capability = capability,
        .capabilityEpoch = capability_epoch,
    };
}

class querying_transition_sink_t final : public notif_route_transition_sink_t {
public:
    querying_transition_sink_t(notif_capability_state_t &state, notif_route_key_t route)
        : state_(state),
          route_(std::move(route)) {
        transitions_.reserve(8);
    }

    void
    publish(const notif_route_transition_t &transition) noexcept override {
        notif_route_snapshot_t snapshot;
        reentrantQuerySucceeded_ =
            state_.queryRemoteNotificationState(route_, snapshot) == notif_state_status_t::SUCCESS;
        transitions_.push_back(transition);
    }

    [[nodiscard]] const std::vector<notif_route_transition_t> &
    transitions() const noexcept {
        return transitions_;
    }

    [[nodiscard]] bool
    reentrantQuerySucceeded() const noexcept {
        return reentrantQuerySucceeded_;
    }

private:
    notif_capability_state_t &state_;
    notif_route_key_t route_;
    std::vector<notif_route_transition_t> transitions_;
    bool reentrantQuerySucceeded_ = false;
};

void
testConvergenceAndEarlyData() {
    const notif_wire_uuid_t agent_a = makeUuid(1);
    const notif_wire_uuid_t backend_a = makeUuid(2);
    const notif_wire_uuid_t worker_a1 = makeUuid(3);
    const notif_wire_uuid_t worker_a2 = makeUuid(4);
    const notif_wire_uuid_t agent_b = makeUuid(21);
    const notif_wire_uuid_t backend_b = makeUuid(22);
    const notif_wire_uuid_t worker_b1 = makeUuid(23);
    const notif_wire_uuid_t worker_b2 = makeUuid(24);

    notif_capability_state_t state_a(agent_a, backend_a, {worker_a1, worker_a2});
    notif_capability_state_t state_b(agent_b, backend_b, {worker_b1, worker_b2});
    const notif_route_key_t route_ab = makeRoute(101, 1, agent_b, backend_b);
    const notif_route_key_t route_ba = makeRoute(201, 7, agent_a, backend_a);
    bind(state_a, makeBinding(route_ab, 1001, {{worker_b1, 5001}, {worker_b2, 5001}}));
    bind(state_b, makeBinding(route_ba, 2001, {{worker_a1, 6001}, {worker_a2, 6002}}));

    notif_wire_envelope_t data_sentinel;
    data_sentinel.type = notif_wire_type_t::ACK;
    require(state_b.prepareData(route_ba, worker_b1, data_sentinel) ==
                notif_state_status_t::NOT_READY,
            "outbound route became ready before an OFFER");
    require(data_sentinel.type == notif_wire_type_t::ACK,
            "failed DATA preparation mutated its output");

    notif_wire_envelope_t offer_a;
    require(state_a.makeOffer(route_ab, worker_a1, offer_a) == notif_state_status_t::SUCCESS,
            "local OFFER generation failed");
    notif_offer_acceptance_t accepted_by_b;
    require(state_b.acceptOffer(offer_a, worker_b1, accepted_by_b) == notif_state_status_t::SUCCESS,
            "reciprocal OFFER was not accepted");
    require(accepted_by_b.route == route_ba && accepted_by_b.reemitLocalOffer,
            "first OFFER did not identify and converge the exact route");

    notif_route_snapshot_t snapshot_b;
    require(state_b.queryRemoteNotificationState(route_ba, snapshot_b) ==
                    notif_state_status_t::SUCCESS &&
                snapshot_b.state == notif_route_state_t::READY &&
                !snapshot_b.localOfferAcknowledged,
            "OFFER did not make the outbound route ready before ACK");

    notif_wire_envelope_t early_data;
    require(state_b.prepareData(route_ba, worker_b2, early_data) == notif_state_status_t::SUCCESS,
            "DATA could not race ahead of ACK");
    const notif_data_resolution_t early_resolution = state_a.resolveData(early_data);
    require(early_resolution.disposition == notif_data_disposition_t::DELIVER &&
                early_resolution.authority.route == route_ab &&
                early_resolution.authority.connectionIdentity == 1001 &&
                early_resolution.authority.senderWorkerIncarnation == worker_b2 &&
                early_resolution.authority.endpointIdentity == 5001 &&
                !early_resolution.authority.tombstoned,
            "early DATA did not resolve to exact receiver-local authority");

    notif_route_key_t acknowledged_route;
    require(state_a.acceptAcknowledgement(accepted_by_b.acknowledgement, acknowledged_route) ==
                    notif_state_status_t::SUCCESS &&
                acknowledged_route == route_ab,
            "ACK did not suppress retries for the exact local OFFER");
    notif_route_snapshot_t snapshot_a;
    require(state_a.queryRemoteNotificationState(route_ab, snapshot_a) ==
                    notif_state_status_t::SUCCESS &&
                snapshot_a.localOfferAcknowledged &&
                snapshot_a.state == notif_route_state_t::NOT_READY,
            "ACK incorrectly created outbound readiness");

    notif_offer_acceptance_t accepted_by_a;
    require(state_a.acceptOffer(accepted_by_b.localOffer, worker_a2, accepted_by_a) ==
                    notif_state_status_t::SUCCESS &&
                accepted_by_a.reemitLocalOffer,
            "reciprocal convergence OFFER failed");
    require(state_b.acceptAcknowledgement(accepted_by_a.acknowledgement, acknowledged_route) ==
                    notif_state_status_t::SUCCESS &&
                acknowledged_route == route_ba,
            "reciprocal ACK failed");

    notif_offer_acceptance_t duplicate;
    require(state_a.acceptOffer(accepted_by_b.localOffer, worker_a1, duplicate) ==
                    notif_state_status_t::SUCCESS &&
                !duplicate.reemitLocalOffer,
            "duplicate OFFER was not an idempotent no-op");
}

void
testOfferReplayAndEpochConflict() {
    const notif_wire_uuid_t agent_a = makeUuid(41);
    const notif_wire_uuid_t backend_a = makeUuid(42);
    const notif_wire_uuid_t worker_a = makeUuid(43);
    const notif_wire_uuid_t agent_b = makeUuid(51);
    const notif_wire_uuid_t backend_b = makeUuid(52);
    const notif_wire_uuid_t worker_b = makeUuid(53);
    notif_capability_state_t state_a(agent_a, backend_a, {worker_a});
    const notif_route_key_t route_ab = makeRoute(301, 4, agent_b, backend_b);
    bind(state_a, makeBinding(route_ab, 3001, {{worker_b, 7001}}));

    notif_wire_envelope_t local_offer;
    require(state_a.makeOffer(route_ab, worker_a, local_offer) == notif_state_status_t::SUCCESS,
            "local replay-test OFFER generation failed");
    const notif_wire_uuid_t remote_capability = makeUuid(61);
    const notif_wire_envelope_t remote_offer =
        makeInbound(local_offer, notif_wire_type_t::OFFER, worker_b, remote_capability, 10);

    notif_offer_acceptance_t acceptance;
    require(state_a.acceptOffer(remote_offer, worker_a, acceptance) ==
                    notif_state_status_t::SUCCESS &&
                acceptance.reemitLocalOffer,
            "new high-epoch OFFER was not accepted");
    require(state_a.acceptOffer(remote_offer, worker_a, acceptance) ==
                    notif_state_status_t::SUCCESS &&
                !acceptance.reemitLocalOffer,
            "exact OFFER replay was not idempotent");

    notif_wire_envelope_t stale = remote_offer;
    stale.capability = makeUuid(62);
    stale.capabilityEpoch = 9;
    require(state_a.acceptOffer(stale, worker_a, acceptance) == notif_state_status_t::STALE_EPOCH,
            "stale OFFER epoch was not rejected deterministically");
    notif_route_snapshot_t snapshot;
    require(
        state_a.queryRemoteNotificationState(route_ab, snapshot) == notif_state_status_t::SUCCESS &&
            snapshot.remoteCapability == remote_capability && snapshot.remoteCapabilityEpoch == 10,
        "stale OFFER mutated the accepted route");

    notif_wire_envelope_t conflict = remote_offer;
    conflict.capability = makeUuid(63);
    require(state_a.acceptOffer(conflict, worker_a, acceptance) ==
                notif_state_status_t::EPOCH_CONFLICT,
            "equal-epoch capability conflict did not fail closed");
    notif_wire_envelope_t data;
    require(state_a.prepareData(route_ab, worker_a, data) == notif_state_status_t::ROUTE_FAILED,
            "failed route remained eligible for new DATA sends");
}

void
testRetirementPreservesExactAttribution() {
    const notif_wire_uuid_t agent_a = makeUuid(71);
    const notif_wire_uuid_t backend_a = makeUuid(72);
    const notif_wire_uuid_t worker_a = makeUuid(73);
    const notif_wire_uuid_t agent_b = makeUuid(81);
    const notif_wire_uuid_t backend_b = makeUuid(82);
    const notif_wire_uuid_t worker_b = makeUuid(83);
    notif_capability_state_t state_a(agent_a, backend_a, {worker_a});
    const notif_route_key_t generation_one = makeRoute(401, 1, agent_b, backend_b);
    const notif_route_snapshot_t first =
        bind(state_a, makeBinding(generation_one, 4001, {{worker_b, 8001}}));

    notif_wire_envelope_t first_offer;
    require(state_a.makeOffer(generation_one, worker_a, first_offer) ==
                notif_state_status_t::SUCCESS,
            "generation-one OFFER generation failed");
    const notif_wire_envelope_t delayed_data = makeInbound(first_offer,
                                                           notif_wire_type_t::DATA,
                                                           worker_b,
                                                           first.localCapability,
                                                           first.localCapabilityEpoch);
    require(state_a.retireRemoteAgent(generation_one) == notif_state_status_t::SUCCESS,
            "exact route retirement failed");
    notif_wire_envelope_t output;
    require(state_a.makeOffer(generation_one, worker_a, output) ==
                    notif_state_status_t::ROUTE_RETIRED &&
                state_a.prepareData(generation_one, worker_a, output) ==
                    notif_state_status_t::ROUTE_RETIRED,
            "retired route remained eligible for new sends");

    const notif_data_resolution_t delayed = state_a.resolveData(delayed_data);
    require(delayed.disposition == notif_data_disposition_t::DELIVER &&
                delayed.authority.route == generation_one &&
                delayed.authority.connectionIdentity == 4001 &&
                delayed.authority.endpointIdentity == 8001 && delayed.authority.tombstoned,
            "delayed DATA lost tombstoned generation attribution");

    const notif_route_key_t generation_two = makeRoute(401, 2, agent_b, backend_b);
    const notif_route_snapshot_t second =
        bind(state_a, makeBinding(generation_two, 4002, {{worker_b, 8002}}));
    require(second.localCapabilityEpoch > first.localCapabilityEpoch &&
                second.localCapability != first.localCapability,
            "replacement binding did not advance capability identity");
    notif_wire_envelope_t second_offer;
    require(state_a.makeOffer(generation_two, worker_a, second_offer) ==
                notif_state_status_t::SUCCESS,
            "generation-two OFFER generation failed");
    const notif_data_resolution_t current =
        state_a.resolveData(makeInbound(second_offer,
                                        notif_wire_type_t::DATA,
                                        worker_b,
                                        second.localCapability,
                                        second.localCapabilityEpoch));
    require(current.disposition == notif_data_disposition_t::DELIVER &&
                current.authority.route == generation_two && !current.authority.tombstoned,
            "replacement DATA did not resolve to its exact generation");
    require(state_a.resolveData(delayed_data).authority.route == generation_one,
            "replacement binding stole tombstoned capability attribution");
}

void
testSiblingIsolationAndUnknownCapability() {
    const notif_wire_uuid_t agent_a = makeUuid(91);
    const notif_wire_uuid_t backend_a = makeUuid(92);
    const notif_wire_uuid_t worker_a = makeUuid(93);
    const notif_wire_uuid_t agent_b = makeUuid(101);
    const notif_wire_uuid_t backend_b = makeUuid(102);
    const notif_wire_uuid_t worker_b = makeUuid(103);
    const notif_wire_uuid_t agent_c = makeUuid(111);
    const notif_wire_uuid_t backend_c = makeUuid(112);
    const notif_wire_uuid_t worker_c = makeUuid(113);
    notif_capability_state_t state_a(agent_a, backend_a, {worker_a});
    const notif_route_key_t route_ab = makeRoute(501, 1, agent_b, backend_b);
    const notif_route_key_t route_ac = makeRoute(502, 1, agent_c, backend_c);
    const notif_route_snapshot_t binding_b =
        bind(state_a, makeBinding(route_ab, 5001, {{worker_b, 9001}}));
    const notif_route_snapshot_t binding_c =
        bind(state_a, makeBinding(route_ac, 5002, {{worker_c, 9002}}));

    notif_wire_envelope_t offer_b;
    notif_wire_envelope_t offer_c;
    require(state_a.makeOffer(route_ab, worker_a, offer_b) == notif_state_status_t::SUCCESS &&
                state_a.makeOffer(route_ac, worker_a, offer_c) == notif_state_status_t::SUCCESS,
            "sibling OFFER generation failed");
    notif_wire_envelope_t invalid_b = makeInbound(offer_b,
                                                  notif_wire_type_t::DATA,
                                                  makeUuid(120),
                                                  binding_b.localCapability,
                                                  binding_b.localCapabilityEpoch);
    const notif_data_resolution_t invalid = state_a.resolveData(invalid_b);
    require(invalid.disposition == notif_data_disposition_t::DROP_ROUTE &&
                invalid.authority.route == route_ab,
            "known invalid DATA did not isolate the exact route");

    const notif_wire_envelope_t valid_c = makeInbound(offer_c,
                                                      notif_wire_type_t::DATA,
                                                      worker_c,
                                                      binding_c.localCapability,
                                                      binding_c.localCapabilityEpoch);
    require(state_a.resolveData(valid_c).disposition == notif_data_disposition_t::DELIVER,
            "invalid sibling DATA poisoned a valid route");
    const notif_wire_envelope_t valid_b = makeInbound(offer_b,
                                                      notif_wire_type_t::DATA,
                                                      worker_b,
                                                      binding_b.localCapability,
                                                      binding_b.localCapabilityEpoch);
    require(state_a.resolveData(valid_b).disposition == notif_data_disposition_t::DELIVER,
            "route-scoped failure permanently poisoned its known route");

    notif_wire_envelope_t unknown = valid_c;
    unknown.capability = makeUuid(121);
    const notif_data_resolution_t unknown_resolution = state_a.resolveData(unknown);
    require(unknown_resolution.disposition == notif_data_disposition_t::POISON_GLOBAL &&
                unknown_resolution.status == notif_state_status_t::UNKNOWN_CAPABILITY,
            "unknown capability did not poison authenticated draining");
}

void
testCapabilitySubscriptionTransitions() {
    const notif_wire_uuid_t local_agent = makeUuid(131);
    const notif_wire_uuid_t local_backend = makeUuid(132);
    const notif_wire_uuid_t local_worker = makeUuid(133);
    const notif_wire_uuid_t remote_agent = makeUuid(141);
    const notif_wire_uuid_t remote_backend = makeUuid(142);
    const notif_wire_uuid_t remote_worker = makeUuid(143);
    notif_capability_state_t state(local_agent, local_backend, {local_worker});
    const notif_route_key_t route = makeRoute(601, 3, remote_agent, remote_backend);
    bind(state, makeBinding(route, 6001, {{remote_worker, 10001}}));

    const auto sink = std::make_shared<querying_transition_sink_t>(state, route);
    notif_route_subscription_t subscription;
    require(state.subscribeRemoteNotificationState(route, sink, subscription) ==
                notif_route_subscription_status_t::SUCCESS,
            "subscription before capability readiness failed");
    require(subscription.route == route && subscription.generation != 0,
            "subscription did not bind the exact route generation");
    require(sink->transitions().empty(), "NOT_READY subscription published an event");

    notif_route_subscription_t duplicate_output;
    const auto duplicate_sink = std::make_shared<querying_transition_sink_t>(state, route);
    require(state.subscribeRemoteNotificationState(route, duplicate_sink, duplicate_output) ==
                notif_route_subscription_status_t::DUPLICATE_SUBSCRIPTION,
            "duplicate exact-route subscription was accepted");
    require(duplicate_output.generation == 0, "failed duplicate subscription mutated its output");

    notif_wire_envelope_t local_offer;
    require(state.makeOffer(route, local_worker, local_offer) == notif_state_status_t::SUCCESS,
            "subscription-test OFFER generation failed");
    const notif_wire_uuid_t first_capability = makeUuid(151);
    notif_offer_acceptance_t acceptance;
    require(
        state.acceptOffer(
            makeInbound(local_offer, notif_wire_type_t::OFFER, remote_worker, first_capability, 20),
            local_worker,
            acceptance) == notif_state_status_t::SUCCESS,
        "first subscribed capability was rejected");
    require(sink->transitions().size() == 1 && sink->transitions()[0].route == route &&
                sink->transitions()[0].state == notif_route_transition_state_t::READY &&
                sink->transitions()[0].capabilityEpoch == 20 &&
                sink->transitions()[0].nativeTimestampNs != 0,
            "first readiness transition was not published exactly");
    require(sink->reentrantQuerySucceeded(),
            "subscription callback could not re-enter route state");

    require(
        state.acceptOffer(
            makeInbound(local_offer, notif_wire_type_t::OFFER, remote_worker, first_capability, 20),
            local_worker,
            acceptance) == notif_state_status_t::SUCCESS &&
            sink->transitions().size() == 1,
        "idempotent capability replay published a duplicate transition");

    const notif_wire_uuid_t second_capability = makeUuid(152);
    require(state.acceptOffer(
                makeInbound(
                    local_offer, notif_wire_type_t::OFFER, remote_worker, second_capability, 21),
                local_worker,
                acceptance) == notif_state_status_t::SUCCESS,
            "capability epoch advance failed");
    require(sink->transitions().size() == 2 &&
                sink->transitions()[1].state == notif_route_transition_state_t::READY &&
                sink->transitions()[1].capabilityEpoch == 21 &&
                sink->transitions()[1].nativeTimestampNs >=
                    sink->transitions()[0].nativeTimestampNs,
            "capability epoch advance did not publish ordered readiness");

    require(
        state.acceptOffer(
            makeInbound(local_offer, notif_wire_type_t::OFFER, remote_worker, makeUuid(153), 21),
            local_worker,
            acceptance) == notif_state_status_t::EPOCH_CONFLICT,
        "equal-epoch conflict did not fail the subscribed route");
    require(sink->transitions().size() == 3 &&
                sink->transitions()[2].state == notif_route_transition_state_t::FAILED &&
                sink->transitions()[2].capabilityEpoch == 21,
            "route failure did not publish the exact capability epoch");

    require(state.retireRemoteAgent(route) == notif_state_status_t::SUCCESS,
            "subscribed route retirement failed");
    require(sink->transitions().size() == 4 &&
                sink->transitions()[3].state == notif_route_transition_state_t::RETIRED &&
                sink->transitions()[3].capabilityEpoch == 21,
            "route retirement did not publish after failure");
    require(state.unsubscribeRemoteNotificationState(subscription) ==
                    notif_route_subscription_status_t::SUCCESS &&
                state.unsubscribeRemoteNotificationState(subscription) ==
                    notif_route_subscription_status_t::UNKNOWN_SUBSCRIPTION,
            "exact subscription cancellation was not take-once");
}

void
testCapabilitySubscriptionSnapshotAndCancellation() {
    const notif_wire_uuid_t local_agent = makeUuid(161);
    const notif_wire_uuid_t local_backend = makeUuid(162);
    const notif_wire_uuid_t local_worker = makeUuid(163);
    const notif_wire_uuid_t remote_agent = makeUuid(171);
    const notif_wire_uuid_t remote_backend = makeUuid(172);
    const notif_wire_uuid_t remote_worker = makeUuid(173);
    notif_capability_state_t state(local_agent, local_backend, {local_worker});
    const notif_route_key_t route = makeRoute(701, 9, remote_agent, remote_backend);
    bind(state, makeBinding(route, 7001, {{remote_worker, 11001}}));

    notif_wire_envelope_t local_offer;
    require(state.makeOffer(route, local_worker, local_offer) == notif_state_status_t::SUCCESS,
            "snapshot-test OFFER generation failed");
    notif_offer_acceptance_t acceptance;
    require(
        state.acceptOffer(
            makeInbound(local_offer, notif_wire_type_t::OFFER, remote_worker, makeUuid(181), 30),
            local_worker,
            acceptance) == notif_state_status_t::SUCCESS,
        "snapshot-test readiness failed");

    const auto ready_sink = std::make_shared<querying_transition_sink_t>(state, route);
    notif_route_subscription_t first_subscription;
    require(state.subscribeRemoteNotificationState(route, ready_sink, first_subscription) ==
                    notif_route_subscription_status_t::SUCCESS &&
                ready_sink->transitions().size() == 1 &&
                ready_sink->transitions()[0].state == notif_route_transition_state_t::READY &&
                ready_sink->transitions()[0].capabilityEpoch == 30,
            "subscribe-after-ready did not publish the atomic snapshot");
    require(state.unsubscribeRemoteNotificationState(first_subscription) ==
                notif_route_subscription_status_t::SUCCESS,
            "ready subscription cancellation failed");

    const auto cancelled_sink = std::make_shared<querying_transition_sink_t>(state, route);
    notif_route_subscription_t cancelled_subscription;
    require(state.subscribeRemoteNotificationState(route, cancelled_sink, cancelled_subscription) ==
                    notif_route_subscription_status_t::SUCCESS &&
                cancelled_sink->transitions().size() == 1,
            "replacement subscription did not receive current readiness");
    require(state.unsubscribeRemoteNotificationState(cancelled_subscription) ==
                notif_route_subscription_status_t::SUCCESS,
            "replacement subscription cancellation failed");
    require(state.retireRemoteAgent(route) == notif_state_status_t::SUCCESS &&
                cancelled_sink->transitions().size() == 1,
            "cancelled subscription received a retirement transition");

    const auto retired_sink = std::make_shared<querying_transition_sink_t>(state, route);
    notif_route_subscription_t retired_subscription;
    require(state.subscribeRemoteNotificationState(route, retired_sink, retired_subscription) ==
                    notif_route_subscription_status_t::SUCCESS &&
                retired_sink->transitions().size() == 1 &&
                retired_sink->transitions()[0].state == notif_route_transition_state_t::RETIRED &&
                retired_sink->transitions()[0].capabilityEpoch == 30,
            "subscribe-after-retirement did not publish the atomic snapshot");
    require(state.unsubscribeRemoteNotificationState(cancelled_subscription) ==
                notif_route_subscription_status_t::UNKNOWN_SUBSCRIPTION,
            "stale cancellation removed a replacement subscription generation");
    require(state.unsubscribeRemoteNotificationState(retired_subscription) ==
                notif_route_subscription_status_t::SUCCESS,
            "retired subscription cancellation failed");

    const notif_wire_uuid_t pending_remote_agent = makeUuid(182);
    const notif_wire_uuid_t pending_remote_backend = makeUuid(183);
    const notif_wire_uuid_t pending_remote_worker = makeUuid(184);
    const notif_route_key_t pending_route =
        makeRoute(702, 1, pending_remote_agent, pending_remote_backend);
    bind(state, makeBinding(pending_route, 7002, {{pending_remote_worker, 11002}}));
    const auto pending_sink = std::make_shared<querying_transition_sink_t>(state, pending_route);
    notif_route_subscription_t pending_subscription;
    require(
        state.subscribeRemoteNotificationState(pending_route, pending_sink, pending_subscription) ==
                notif_route_subscription_status_t::SUCCESS &&
            pending_sink->transitions().empty(),
        "pending-route subscription published before readiness");
    require(state.unsubscribeRemoteNotificationState(pending_subscription) ==
                notif_route_subscription_status_t::SUCCESS,
            "pending-route subscription cancellation failed");
    notif_wire_envelope_t pending_local_offer;
    require(state.makeOffer(pending_route, local_worker, pending_local_offer) ==
                    notif_state_status_t::SUCCESS &&
                state.acceptOffer(makeInbound(pending_local_offer,
                                              notif_wire_type_t::OFFER,
                                              pending_remote_worker,
                                              makeUuid(185),
                                              1),
                                  local_worker,
                                  acceptance) == notif_state_status_t::SUCCESS &&
                pending_sink->transitions().empty(),
            "cancelled pending subscription received readiness");

    notif_route_subscription_t unknown_output;
    require(state.subscribeRemoteNotificationState(
                makeRoute(999, 1, makeUuid(191), makeUuid(192)), retired_sink, unknown_output) ==
                    notif_route_subscription_status_t::UNKNOWN_ROUTE &&
                unknown_output.generation == 0,
            "unknown-route subscription did not fail without output mutation");
}

} // namespace

int
main() {
    try {
        testConvergenceAndEarlyData();
        testOfferReplayAndEpochConflict();
        testRetirementPreservesExactAttribution();
        testSiblingIsolationAndUnknownCapability();
        testCapabilitySubscriptionTransitions();
        testCapabilitySubscriptionSnapshotAndCancellation();
    }
    catch (const std::exception &error) {
        std::cerr << "ucx_notif_state_test failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "ucx_notif_state_test passed\n";
    return 0;
}
