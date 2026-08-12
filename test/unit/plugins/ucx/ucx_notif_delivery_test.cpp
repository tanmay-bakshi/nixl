/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "ucx_notif_delivery.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

using namespace nixl::ucx;

namespace {

void
require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

notif_wire_uuid_t
uuid(std::uint8_t seed) {
    notif_wire_uuid_t value;
    for (std::size_t index = 0; index < value.bytes.size(); ++index) {
        value.bytes[index] = static_cast<std::uint8_t>(seed + index);
    }
    value.bytes[6] = static_cast<std::uint8_t>((value.bytes[6] & 0x0fU) | 0x40U);
    value.bytes[8] = static_cast<std::uint8_t>((value.bytes[8] & 0x3fU) | 0x80U);
    return value;
}

notif_delivery_key_t
key(std::uint64_t identity, std::uint64_t generation = 7) {
    return {
        .identity = identity,
        .sourceHandleIdentity = 47,
        .sourceGeneration = 53,
        .route =
            {
                .handleIdentity = 11,
                .handleGeneration = generation,
                .remoteAgentIncarnation = uuid(1),
                .remoteBackendIncarnation = uuid(17),
            },
        .capability = uuid(33),
        .capabilityEpoch = 5,
        .connectionIdentity = 41,
        .receiptWorkerIncarnation = uuid(49),
        .endpointIdentity = 43,
    };
}

notif_delivery_receipt_t
receipt(const notif_delivery_key_t &delivery) {
    return {
        .identity = delivery.identity,
        .sourceHandleIdentity = delivery.sourceHandleIdentity,
        .sourceGeneration = delivery.sourceGeneration,
        .route = delivery.route,
        .capability = delivery.capability,
        .capabilityEpoch = delivery.capabilityEpoch,
        .connectionIdentity = delivery.connectionIdentity,
        .receiptWorkerIncarnation = delivery.receiptWorkerIncarnation,
        .endpointIdentity = delivery.endpointIdentity,
    };
}

notif_delivery_status_t
registerDelivery(notif_delivery_registry_t &registry,
                 const notif_delivery_key_t &delivery,
                 notif_delivery_registry_t::terminal_t terminal) {
    return registry.registerDelivery(delivery, std::move(terminal), [] { return true; });
}

struct terminal_observation_t {
    std::uint64_t identity = 0;
    nixl_status_t status = NIXL_ERR_BACKEND;
    std::uint64_t timestampNs = 0;
};

notif_delivery_registry_t::terminal_t
recordTerminal(std::vector<terminal_observation_t> &observations, std::uint64_t identity) {
    return [&observations, identity](nixl_status_t status, std::uint64_t timestamp_ns) {
        observations.push_back({identity, status, timestamp_ns});
    };
}

void
requireTerminal(const std::vector<terminal_observation_t> &observations,
                std::uint64_t identity,
                nixl_status_t status,
                std::uint64_t timestamp_ns,
                const char *message) {
    const auto match =
        std::find_if(observations.begin(), observations.end(), [identity](const auto &observation) {
            return observation.identity == identity;
        });
    require(match != observations.end() && match->status == status &&
                match->timestampNs == timestamp_ns,
            message);
}

void
testJoinBothOrdersAndDuplicateReceipt() {
    notif_delivery_registry_t registry(4);
    std::vector<nixl_status_t> statuses;
    require(registerDelivery(registry,
                             key(1),
                             [&](nixl_status_t status, std::uint64_t) {
                                 statuses.push_back(status);
                             }) == notif_delivery_status_t::SUCCESS,
            "first delivery registration failed");
    require(registry.acceptReceipt(receipt(key(1)), 10) == notif_delivery_status_t::SUCCESS &&
                statuses.empty(),
            "receipt completed before local send lifetime");
    require(registry.completeLocal(1, NIXL_SUCCESS, 11) == notif_delivery_status_t::SUCCESS &&
                statuses == std::vector<nixl_status_t>{NIXL_SUCCESS},
            "receipt-first join did not complete exactly once");
    require(registry.acceptReceipt(receipt(key(1)), 12) ==
                    notif_delivery_status_t::DUPLICATE_DELIVERY &&
                statuses.size() == 1,
            "byte-identical receipt replay was not idempotent");

    require(registerDelivery(registry,
                             key(2),
                             [&](nixl_status_t status, std::uint64_t) {
                                 statuses.push_back(status);
                             }) == notif_delivery_status_t::SUCCESS,
            "second delivery registration failed");
    require(registry.completeLocal(2, NIXL_SUCCESS, 20) == notif_delivery_status_t::SUCCESS &&
                statuses.size() == 1,
            "local send completion escaped the remote receipt join");
    require(registry.acceptReceipt(receipt(key(2)), 21) == notif_delivery_status_t::SUCCESS &&
                statuses.size() == 2 && statuses.back() == NIXL_SUCCESS,
            "local-first join did not complete on receipt");
}

void
testConflictFailureAndBounds() {
    notif_delivery_registry_t registry(1);
    std::vector<nixl_status_t> statuses;
    require(registerDelivery(registry,
                             key(3),
                             [&](nixl_status_t status, std::uint64_t) {
                                 statuses.push_back(status);
                             }) == notif_delivery_status_t::SUCCESS,
            "bounded registry rejected its first delivery");
    require(registerDelivery(registry, key(4), [](nixl_status_t, std::uint64_t) {}) ==
                notif_delivery_status_t::CAPACITY_EXCEEDED,
            "bounded registry silently exceeded capacity");
    notif_delivery_key_t conflict = key(3);
    ++conflict.capabilityEpoch;
    require(registry.acceptReceipt(receipt(conflict), 30) ==
                notif_delivery_status_t::IDENTITY_CONFLICT,
            "same-identity capability conflict was accepted");
    require(registry.failConnection(41, NIXL_ERR_REMOTE_DISCONNECT, 31) ==
                    notif_delivery_status_t::SUCCESS &&
                statuses == std::vector<nixl_status_t>{NIXL_ERR_REMOTE_DISCONNECT},
            "connection failure did not claim exact outstanding delivery");
    require(registry.inventory().outstanding == 0,
            "connection failure leaked live delivery inventory");
}

void
testReceiptAuthorityAndReceiptFirstFailureTombstone() {
    notif_delivery_registry_t registry(4);
    std::vector<nixl_status_t> statuses;
    const notif_delivery_key_t delivery = key(5);
    require(registerDelivery(registry,
                             delivery,
                             [&](nixl_status_t status, std::uint64_t) {
                                 statuses.push_back(status);
                             }) == notif_delivery_status_t::SUCCESS,
            "delivery registration failed");

    notif_delivery_receipt_t wrong_source = receipt(delivery);
    ++wrong_source.sourceGeneration;
    require(registry.acceptReceipt(wrong_source, 40) ==
                    notif_delivery_status_t::IDENTITY_CONFLICT &&
                statuses.empty(),
            "receipt with conflicting source generation was accepted");

    require(registry.acceptReceipt(receipt(delivery), 41) == notif_delivery_status_t::SUCCESS &&
                statuses.empty(),
            "receipt-first delivery completed before local lifetime");
    require(registry.completeLocal(delivery.identity, NIXL_ERR_BACKEND, 42) ==
                    notif_delivery_status_t::SUCCESS &&
                statuses == std::vector<nixl_status_t>{NIXL_ERR_BACKEND},
            "local failure did not terminalize receipt-first delivery");
    require(registry.acceptReceipt(receipt(delivery), 43) ==
                    notif_delivery_status_t::DUPLICATE_DELIVERY &&
                statuses.size() == 1,
            "accepted receipt was lost from a failed delivery tombstone");
}

void
testExactLateReceiptsRemainDuplicateStale() {
    notif_delivery_registry_t registry(8);
    std::vector<terminal_observation_t> observations;
    notif_delivery_key_t cancelled = key(10, 10);
    notif_delivery_key_t retired = key(11, 11);
    notif_delivery_key_t failed = key(12, 12);
    notif_delivery_key_t shutdown = key(13, 13);
    cancelled.connectionIdentity = 110;
    retired.connectionIdentity = 111;
    failed.connectionIdentity = 112;
    shutdown.connectionIdentity = 113;

    for (const notif_delivery_key_t &delivery : {cancelled, retired, failed, shutdown}) {
        require(
            registerDelivery(registry, delivery, recordTerminal(observations, delivery.identity)) ==
                notif_delivery_status_t::SUCCESS,
            "late-receipt fixture registration failed");
    }
    require(registry.failDelivery(cancelled.identity, NIXL_ERR_CANCELED, 100) ==
                notif_delivery_status_t::SUCCESS,
            "delivery cancellation failed");
    require(registry.failRoute(retired.route, NIXL_ERR_NOT_FOUND, 101) ==
                notif_delivery_status_t::SUCCESS,
            "route retirement failed");
    require(registry.failConnection(failed.connectionIdentity, NIXL_ERR_REMOTE_DISCONNECT, 102) ==
                notif_delivery_status_t::SUCCESS,
            "connection failure failed");
    require(registry.failAll(NIXL_ERR_CANCELED, 103) == notif_delivery_status_t::SUCCESS,
            "registry shutdown failed");

    require(observations.size() == 4, "terminal arbitration did not publish exactly once");
    requireTerminal(observations,
                    cancelled.identity,
                    NIXL_ERR_CANCELED,
                    100,
                    "cancellation terminal evidence changed");
    requireTerminal(observations,
                    retired.identity,
                    NIXL_ERR_NOT_FOUND,
                    101,
                    "retirement terminal evidence changed");
    requireTerminal(observations,
                    failed.identity,
                    NIXL_ERR_REMOTE_DISCONNECT,
                    102,
                    "failure terminal evidence changed");
    requireTerminal(observations,
                    shutdown.identity,
                    NIXL_ERR_CANCELED,
                    103,
                    "shutdown terminal evidence changed");

    for (const notif_delivery_key_t &delivery : {cancelled, retired, failed, shutdown}) {
        require(registry.acceptReceipt(receipt(delivery), 200 + delivery.identity) ==
                    notif_delivery_status_t::DUPLICATE_DELIVERY,
                "exact late receipt was not classified duplicate-stale");
    }
    const notif_delivery_inventory_t inventory = registry.inventory();
    require(inventory.outstanding == 0 && inventory.completed == 4 && !inventory.accepting &&
                observations.size() == 4,
            "late receipts resurrected or republished terminal delivery state");
}

void
testReceiptAuthorityConflictsBeforeAndAfterTerminalization() {
    notif_delivery_registry_t registry(4);
    std::vector<terminal_observation_t> observations;
    const notif_delivery_key_t delivery = key(20);
    require(registerDelivery(registry, delivery, recordTerminal(observations, delivery.identity)) ==
                notif_delivery_status_t::SUCCESS,
            "authority-conflict delivery registration failed");

    const notif_delivery_receipt_t exact = receipt(delivery);
    std::vector<std::pair<std::string_view, notif_delivery_receipt_t>> conflicts;
    notif_delivery_receipt_t wrong_route = exact;
    ++wrong_route.route.handleGeneration;
    conflicts.emplace_back("route", wrong_route);
    notif_delivery_receipt_t wrong_epoch = exact;
    ++wrong_epoch.capabilityEpoch;
    conflicts.emplace_back("epoch", wrong_epoch);
    notif_delivery_receipt_t wrong_capability = exact;
    wrong_capability.capability = uuid(34);
    conflicts.emplace_back("capability", wrong_capability);
    notif_delivery_receipt_t wrong_worker = exact;
    wrong_worker.receiptWorkerIncarnation = uuid(50);
    conflicts.emplace_back("worker", wrong_worker);
    notif_delivery_receipt_t wrong_connection = exact;
    ++wrong_connection.connectionIdentity;
    conflicts.emplace_back("connection", wrong_connection);
    notif_delivery_receipt_t wrong_endpoint = exact;
    ++wrong_endpoint.endpointIdentity;
    conflicts.emplace_back("endpoint", wrong_endpoint);
    notif_delivery_receipt_t wrong_source_handle = exact;
    ++wrong_source_handle.sourceHandleIdentity;
    conflicts.emplace_back("source handle", wrong_source_handle);
    notif_delivery_receipt_t wrong_source_generation = exact;
    ++wrong_source_generation.sourceGeneration;
    conflicts.emplace_back("source generation", wrong_source_generation);

    for (const auto &[name, conflict] : conflicts) {
        static_cast<void>(name);
        require(registry.acceptReceipt(conflict, 300) == notif_delivery_status_t::IDENTITY_CONFLICT,
                "live delivery accepted conflicting receipt authority");
    }
    require(observations.empty(), "conflicting receipt terminalized a live delivery");
    require(registry.failDelivery(delivery.identity, NIXL_ERR_CANCELED, 301) ==
                notif_delivery_status_t::SUCCESS,
            "authority-conflict delivery cancellation failed");
    require(registry.acceptReceipt(exact, 302) == notif_delivery_status_t::DUPLICATE_DELIVERY,
            "exact post-cancellation receipt was not duplicate-stale");
    for (const auto &[name, conflict] : conflicts) {
        static_cast<void>(name);
        require(registry.acceptReceipt(conflict, 303) == notif_delivery_status_t::IDENTITY_CONFLICT,
                "terminal tombstone accepted conflicting receipt authority");
    }
    require(observations.size() == 1 && observations.front().status == NIXL_ERR_CANCELED &&
                registry.inventory().outstanding == 0,
            "conflicting late receipt changed terminal arbitration");
}

void
testRegistrationRacingTerminalAuthorityNeverRemainsLive() {
    {
        notif_delivery_registry_t registry(4);
        std::vector<terminal_observation_t> observations;
        const notif_delivery_key_t delivery = key(30);
        std::promise<void> validator_entered_promise;
        std::future<void> validator_entered = validator_entered_promise.get_future();
        std::promise<void> release_validator_promise;
        std::shared_future<void> release_validator = release_validator_promise.get_future().share();
        auto registration = std::async(std::launch::async, [&] {
            return registry.registerDelivery(
                delivery, recordTerminal(observations, delivery.identity), [&] {
                    validator_entered_promise.set_value();
                    release_validator.wait();
                    return true;
                });
        });
        require(validator_entered.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
                "registration validator did not enter");
        std::promise<void> failure_started_promise;
        std::future<void> failure_started = failure_started_promise.get_future();
        auto failure = std::async(std::launch::async, [&] {
            failure_started_promise.set_value();
            return registry.failRoute(delivery.route, NIXL_ERR_REMOTE_DISCONNECT, 400);
        });
        require(failure_started.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
                "racing route failure did not start");
        release_validator_promise.set_value();
        require(registration.get() == notif_delivery_status_t::SUCCESS,
                "registration that won authority validation was rejected");
        require(failure.get() == notif_delivery_status_t::SUCCESS, "racing route failure failed");
        require(registry.inventory().outstanding == 0 && observations.size() == 1 &&
                    observations.front().status == NIXL_ERR_REMOTE_DISCONNECT,
                "route failure race left a validated delivery live");
        require(registry.acceptReceipt(receipt(delivery), 401) ==
                    notif_delivery_status_t::DUPLICATE_DELIVERY,
                "late receipt escaped the racing route-failure tombstone");
    }

    {
        notif_delivery_registry_t registry(4);
        std::vector<terminal_observation_t> observations;
        const notif_delivery_key_t delivery = key(31);
        std::atomic<bool> authority_valid = true;
        std::promise<void> validator_entered_promise;
        std::future<void> validator_entered = validator_entered_promise.get_future();
        std::promise<void> release_validator_promise;
        std::shared_future<void> release_validator = release_validator_promise.get_future().share();
        auto registration = std::async(std::launch::async, [&] {
            return registry.registerDelivery(
                delivery, recordTerminal(observations, delivery.identity), [&] {
                    validator_entered_promise.set_value();
                    release_validator.wait();
                    return authority_valid.load(std::memory_order_acquire);
                });
        });
        require(validator_entered.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
                "failure-race validator did not enter");
        authority_valid.store(false, std::memory_order_release);
        std::promise<void> failure_started_promise;
        std::future<void> failure_started = failure_started_promise.get_future();
        auto failure = std::async(std::launch::async, [&] {
            failure_started_promise.set_value();
            return registry.failConnection(
                delivery.connectionIdentity, NIXL_ERR_REMOTE_DISCONNECT, 410);
        });
        require(failure_started.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
                "racing connection failure did not start");
        release_validator_promise.set_value();
        require(registration.get() == notif_delivery_status_t::ROUTE_TERMINAL,
                "registration survived invalidated external authority");
        require(failure.get() == notif_delivery_status_t::SUCCESS,
                "racing connection failure failed");
        require(registry.inventory().outstanding == 0 && registry.inventory().completed == 0 &&
                    observations.empty(),
                "failed authority race retained an unowned delivery");
        require(registry.acceptReceipt(receipt(delivery), 411) ==
                    notif_delivery_status_t::UNKNOWN_DELIVERY,
                "rejected registration manufactured receipt authority");
    }
}

void
testCapabilityEpochSupersessionFailsOnlyAffectedDeliveries() {
    notif_delivery_registry_t registry(8);
    std::vector<terminal_observation_t> observations;
    notif_delivery_key_t old_capability = key(40, 20);
    old_capability.capability = uuid(70);
    old_capability.capabilityEpoch = 8;
    notif_delivery_key_t old_epoch = key(41, 20);
    old_epoch.capability = uuid(71);
    old_epoch.capabilityEpoch = 8;
    notif_delivery_key_t current = key(42, 20);
    current.capability = uuid(71);
    current.capabilityEpoch = 9;
    notif_delivery_key_t sibling = key(43, 21);
    sibling.capability = uuid(70);
    sibling.capabilityEpoch = 8;

    for (const notif_delivery_key_t &delivery : {old_capability, old_epoch, current, sibling}) {
        require(
            registerDelivery(registry, delivery, recordTerminal(observations, delivery.identity)) ==
                notif_delivery_status_t::SUCCESS,
            "supersession fixture registration failed");
    }
    require(registry.failSupersededAuthority(current.route,
                                             current.capability,
                                             current.capabilityEpoch,
                                             NIXL_ERR_NOT_ALLOWED,
                                             500) == notif_delivery_status_t::SUCCESS,
            "capability supersession arbitration failed");
    require(registry.inventory().outstanding == 2 && observations.size() == 2,
            "capability supersession selected the wrong delivery population");
    requireTerminal(observations,
                    old_capability.identity,
                    NIXL_ERR_NOT_ALLOWED,
                    500,
                    "superseded capability remained live");
    requireTerminal(observations,
                    old_epoch.identity,
                    NIXL_ERR_NOT_ALLOWED,
                    500,
                    "superseded capability epoch remained live");
    require(registry.acceptReceipt(receipt(old_capability), 501) ==
                    notif_delivery_status_t::DUPLICATE_DELIVERY &&
                registry.acceptReceipt(receipt(old_epoch), 502) ==
                    notif_delivery_status_t::DUPLICATE_DELIVERY,
            "superseded delivery lost duplicate-stale receipt authority");

    require(registry.completeLocal(current.identity, NIXL_SUCCESS, 503) ==
                    notif_delivery_status_t::SUCCESS &&
                registry.acceptReceipt(receipt(current), 504) == notif_delivery_status_t::SUCCESS,
            "current capability could not complete after supersession");
    require(registry.failDelivery(sibling.identity, NIXL_ERR_CANCELED, 505) ==
                notif_delivery_status_t::SUCCESS,
            "sibling cleanup failed");
    requireTerminal(observations,
                    current.identity,
                    NIXL_SUCCESS,
                    504,
                    "current capability was failed by supersession");
    requireTerminal(observations,
                    sibling.identity,
                    NIXL_ERR_CANCELED,
                    505,
                    "sibling route was failed by supersession");
    require(registry.inventory().outstanding == 0 && observations.size() == 4,
            "supersession fixture retained delivery inventory");
}

void
testRejectedRegistrationRollsBackAndShutdownClosesAdmission() {
    notif_delivery_registry_t registry(2);
    std::vector<terminal_observation_t> observations;
    const notif_delivery_key_t delivery = key(60, 60);
    require(registry.registerDelivery(
                delivery, recordTerminal(observations, delivery.identity), [] { return false; }) ==
                notif_delivery_status_t::ROUTE_TERMINAL &&
                registry.inventory().outstanding == 0 && observations.empty(),
            "failed authority validation retained partial registration state");
    require(registerDelivery(
                registry, delivery, recordTerminal(observations, delivery.identity)) ==
                notif_delivery_status_t::SUCCESS,
            "rolled-back delivery identity could not register against live authority");
    require(registry.failAll(NIXL_ERR_CANCELED, 600) ==
                    notif_delivery_status_t::SUCCESS &&
                observations.size() == 1,
            "registry close did not terminalize the rolled-back fixture exactly once");
    require(registerDelivery(registry, key(61, 61), [](nixl_status_t, std::uint64_t) {}) ==
                notif_delivery_status_t::REGISTRY_CLOSED,
            "closed delivery registry accepted a new registration");
}

void
testInventorySnapshotIsAtomicExactAndCanonical() {
    notif_delivery_registry_t registry(4);
    std::vector<terminal_observation_t> observations;
    const notif_delivery_key_t receipt_pending = key(9, 70);
    const notif_delivery_key_t local_pending = key(3, 71);
    const notif_delivery_key_t both_pending = key(7, 72);
    for (const notif_delivery_key_t &delivery :
         {receipt_pending, local_pending, both_pending}) {
        require(
            registerDelivery(registry, delivery, recordTerminal(observations, delivery.identity)) ==
                notif_delivery_status_t::SUCCESS,
            "snapshot fixture registration failed");
    }
    require(registry.completeLocal(receipt_pending.identity, NIXL_SUCCESS, 700) ==
                    notif_delivery_status_t::SUCCESS &&
                registry.acceptReceipt(receipt(local_pending), 701) ==
                    notif_delivery_status_t::SUCCESS,
            "snapshot fixture could not establish asymmetric join states");

    const notif_delivery_snapshot_t snapshot = registry.snapshot();
    require(snapshot.inventory.outstanding == snapshot.activeRecords.size() &&
                snapshot.inventory.outstanding == 3 && snapshot.inventory.completed == 0 &&
                snapshot.inventory.accepting,
            "delivery snapshot count and exact records were not one conserved view");
    require(snapshot.activeRecords[0].key == local_pending &&
                snapshot.activeRecords[0].localPending &&
                !snapshot.activeRecords[0].receiptPending &&
                snapshot.activeRecords[1].key == both_pending &&
                snapshot.activeRecords[1].localPending &&
                snapshot.activeRecords[1].receiptPending &&
                snapshot.activeRecords[2].key == receipt_pending &&
                !snapshot.activeRecords[2].localPending &&
                snapshot.activeRecords[2].receiptPending,
            "delivery snapshot was incomplete, unsorted, or lost join state");
}

} // namespace

int
main() {
    try {
        testJoinBothOrdersAndDuplicateReceipt();
        testConflictFailureAndBounds();
        testReceiptAuthorityAndReceiptFirstFailureTombstone();
        testExactLateReceiptsRemainDuplicateStale();
        testReceiptAuthorityConflictsBeforeAndAfterTerminalization();
        testRegistrationRacingTerminalAuthorityNeverRemainsLive();
        testCapabilityEpochSupersessionFailsOnlyAffectedDeliveries();
        testRejectedRegistrationRollsBackAndShutdownClosesAdmission();
        testInventorySnapshotIsAtomicExactAndCanonical();
    }
    catch (const std::exception &error) {
        std::cerr << "ucx_notif_delivery_test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "ucx_notif_delivery_test passed\n";
    return EXIT_SUCCESS;
}
