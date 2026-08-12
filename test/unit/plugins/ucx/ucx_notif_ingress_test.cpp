/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "ucx_notif_ingress.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <stdexcept>

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

notif_ingress_key_t
key(std::uint64_t delivery_identity, std::uint64_t route_generation = 7) {
    const notif_wire_uuid_t source_agent = uuid(1);
    const notif_wire_uuid_t source_backend = uuid(17);
    return {
        .sourceBackendIncarnation = source_backend,
        .sourceHandleIdentity = 47,
        .sourceGeneration = 53,
        .deliveryIdentity = delivery_identity,
        .destinationAgentIncarnation = uuid(33),
        .destinationBackendIncarnation = uuid(49),
        .route =
            {
                .handleIdentity = 11,
                .handleGeneration = route_generation,
                .remoteAgentIncarnation = source_agent,
                .remoteBackendIncarnation = source_backend,
            },
        .capability = uuid(65),
        .capabilityEpoch = 5,
        .connectionIdentity = 41,
        .sourceWorkerIncarnation = uuid(81),
        .receiptWorkerIncarnation = uuid(97),
        .endpointIdentity = 43,
    };
}

notif_wire_envelope_t
receipt(const notif_ingress_key_t &ingress) {
    return {
        .type = notif_wire_type_t::DATA_RECEIPT,
        .senderAgentIncarnation = ingress.destinationAgentIncarnation,
        .recipientAgentIncarnation = ingress.route.remoteAgentIncarnation,
        .senderBackendIncarnation = ingress.destinationBackendIncarnation,
        .recipientBackendIncarnation = ingress.sourceBackendIncarnation,
        .senderWorkerIncarnation = ingress.receiptWorkerIncarnation,
        .capability = ingress.capability,
        .capabilityEpoch = ingress.capabilityEpoch,
        .deliveryIdentity = ingress.deliveryIdentity,
        .sourceHandleIdentity = ingress.sourceHandleIdentity,
        .sourceGeneration = ingress.sourceGeneration,
    };
}

constexpr std::array<std::uint8_t, 5> payload = {1, 2, 3, 4, 5};

nixlAuthenticatedNotification
notification(const notif_ingress_key_t &ingress) {
    return {
        .remoteAgent = "source",
        .payload = "payload",
        .handleIdentity = ingress.route.handleIdentity,
        .generation = ingress.route.handleGeneration,
        .connectionIdentity = ingress.connectionIdentity,
        .endpointIdentity = ingress.endpointIdentity,
    };
}

const auto acceptNotification = [](nixlAuthenticatedNotification &&) { return NIXL_SUCCESS; };

void
testPrimaryIdentityRejectsAuthorityConflict() {
    notif_ingress_registry_t registry(4);
    const notif_ingress_key_t original = key(1);
    require(registry.reserve(original, payload, receipt(original)) ==
                notif_ingress_status_t::SUCCESS,
            "initial ingress reservation failed");

    notif_ingress_key_t conflict = original;
    ++conflict.capabilityEpoch;
    require(registry.reserve(conflict, payload, receipt(conflict)) ==
                notif_ingress_status_t::IDENTITY_CONFLICT,
            "same delivery identity created parallel conflicting authority");
    require(registry.inventory().pending == 1, "conflicting authority changed pending inventory");
}

void
testIrrevocableAdmissionAndBoundedReplay() {
    notif_ingress_registry_t registry(4);
    const notif_ingress_key_t ingress = key(2);
    const notif_wire_envelope_t frozen_receipt = receipt(ingress);
    require(registry.reserve(ingress, payload, frozen_receipt) == notif_ingress_status_t::SUCCESS,
            "ingress reservation failed");

    notif_wire_envelope_t selected_receipt;
    require(registry.admit(
                ingress, payload, notification(ingress), acceptNotification, selected_receipt) ==
                    notif_ingress_status_t::SUCCESS &&
                selected_receipt == frozen_receipt,
            "admission winner did not retain its frozen receipt");
    require(registry.failRoute(ingress.route) == notif_ingress_status_t::SUCCESS,
            "route retirement failed");
    require(registry.completeReceipt(ingress, NIXL_SUCCESS) == notif_ingress_status_t::SUCCESS,
            "committed receipt did not reach a completed tombstone");

    notif_ingress_inventory_t inventory = registry.inventory();
    require(inventory.pending == 0 && inventory.admitting == 0 && inventory.committed == 0 &&
                inventory.replaying == 0 && inventory.completed == 1,
            "initial receipt completion leaked live ingress inventory");

    require(registry.reserve(ingress, payload, frozen_receipt) ==
                notif_ingress_status_t::DUPLICATE_COMMITTED,
            "completed byte-identical replay was not recognized");
    require(registry.admit(
                ingress, payload, notification(ingress), acceptNotification, selected_receipt) ==
                notif_ingress_status_t::DUPLICATE_COMMITTED,
            "completed replay did not claim one receipt resend");
    require(registry.admit(
                ingress, payload, notification(ingress), acceptNotification, selected_receipt) ==
                notif_ingress_status_t::DUPLICATE_PENDING,
            "concurrent replay claimed a second receipt resend");
    require(registry.inventory().replaying == 1,
            "single replay-send claim was not lifecycle-counted");
    require(registry.completeReceipt(ingress, NIXL_SUCCESS) == notif_ingress_status_t::SUCCESS &&
                registry.inventory().replaying == 0,
            "replay receipt completion leaked its obligation");

    const notif_ingress_key_t later = key(3);
    require(registry.reserve(later, payload, receipt(later)) ==
                notif_ingress_status_t::ROUTE_TERMINAL,
            "route retirement did not fence a later reservation");
}

void
testPendingRetirementAndReceiptFailure() {
    notif_ingress_registry_t registry(4);
    const notif_ingress_key_t pending = key(4, 8);
    require(registry.reserve(pending, payload, receipt(pending)) == notif_ingress_status_t::SUCCESS,
            "pending reservation failed");
    require(registry.failConnection(pending.connectionIdentity) ==
                    notif_ingress_status_t::SUCCESS &&
                registry.inventory().pending == 0,
            "connection failure did not revoke pending admission");
    require(registry.reserve(pending, payload, receipt(pending)) ==
                notif_ingress_status_t::ROUTE_TERMINAL,
            "connection failure did not fence reserve-after-failure");

    notif_ingress_key_t committed = key(5, 9);
    committed.connectionIdentity = 42;
    notif_wire_envelope_t selected_receipt;
    require(registry.reserve(committed, payload, receipt(committed)) ==
                    notif_ingress_status_t::SUCCESS &&
                registry.admit(committed,
                               payload,
                               notification(committed),
                               acceptNotification,
                               selected_receipt) == notif_ingress_status_t::SUCCESS,
            "receipt-failure fixture could not commit admission");
    require(registry.completeReceipt(committed, NIXL_ERR_REMOTE_DISCONNECT) ==
                notif_ingress_status_t::RECEIPT_FAILED,
            "failed receipt send was recorded as successful");
    const notif_ingress_inventory_t inventory = registry.inventory();
    require(inventory.committed == 0 && inventory.quarantined == 1,
            "failed committed receipt did not enter quarantine exactly once");
}

void
testShutdownClosesAdmissionWithoutRevokingCommit() {
    notif_ingress_registry_t registry(4);
    const notif_ingress_key_t pending = key(6, 10);
    const notif_ingress_key_t committed = key(7, 11);
    notif_wire_envelope_t selected_receipt;
    require(registry.reserve(pending, payload, receipt(pending)) ==
                    notif_ingress_status_t::SUCCESS &&
                registry.reserve(committed, payload, receipt(committed)) ==
                    notif_ingress_status_t::SUCCESS &&
                registry.admit(committed,
                               payload,
                               notification(committed),
                               acceptNotification,
                               selected_receipt) == notif_ingress_status_t::SUCCESS,
            "shutdown fixture setup failed");
    require(registry.failAll() == notif_ingress_status_t::SUCCESS,
            "shutdown admission close failed");
    require(registry.inventory().pending == 0 && registry.inventory().committed == 1,
            "shutdown revoked a committed admission or retained pending work");
    require(registry.completeReceipt(committed, NIXL_SUCCESS) == notif_ingress_status_t::SUCCESS,
            "shutdown prevented a committed admission from draining");

    const notif_ingress_key_t later = key(8, 12);
    require(registry.reserve(later, payload, receipt(later)) ==
                notif_ingress_status_t::ROUTE_TERMINAL,
            "closed ingress accepted new work");
}

void
testMonotonicAntiReplaySurvivesTombstoneEviction() {
    notif_ingress_registry_t registry(4, 1);
    const notif_ingress_key_t first = key(20, 20);
    notif_ingress_key_t second = first;
    second.deliveryIdentity = 21;

    notif_wire_envelope_t selected_receipt;
    for (const notif_ingress_key_t &ingress : {first, second}) {
        require(registry.reserve(ingress, payload, receipt(ingress)) ==
                        notif_ingress_status_t::SUCCESS &&
                    registry.admit(ingress,
                                   payload,
                                   notification(ingress),
                                   acceptNotification,
                                   selected_receipt) == notif_ingress_status_t::SUCCESS &&
                    registry.completeReceipt(ingress, NIXL_SUCCESS) ==
                        notif_ingress_status_t::SUCCESS,
                "anti-replay fixture could not complete a delivery");
    }
    require(registry.inventory().completed == 1,
            "bounded completed ledger did not evict its oldest tombstone");
    require(registry.reserve(first, payload, receipt(first)) ==
                notif_ingress_status_t::ROUTE_TERMINAL,
            "tombstone eviction re-admitted an old delivery identity");
}

void
testFailedReceiptRetainsExactQuarantineAuthority() {
    notif_ingress_registry_t registry(4);
    const notif_ingress_key_t ingress = key(30, 30);
    notif_wire_envelope_t selected_receipt;
    require(
        registry.reserve(ingress, payload, receipt(ingress)) == notif_ingress_status_t::SUCCESS &&
            registry.admit(
                ingress, payload, notification(ingress), acceptNotification, selected_receipt) ==
                notif_ingress_status_t::SUCCESS &&
            registry.completeReceipt(ingress, NIXL_ERR_REMOTE_DISCONNECT) ==
                notif_ingress_status_t::RECEIPT_FAILED,
        "failed receipt fixture did not quarantine exact authority");
    require(registry.reserve(ingress, payload, receipt(ingress)) ==
                notif_ingress_status_t::RECEIPT_FAILED,
            "quarantined identity was admitted again");

    notif_ingress_key_t conflict = ingress;
    ++conflict.capabilityEpoch;
    require(registry.reserve(conflict, payload, receipt(conflict)) ==
                notif_ingress_status_t::IDENTITY_CONFLICT,
            "quarantine lost conflicting-authority detection");
}

void
testAdmissionIsOneLinearizedQueueTransaction() {
    notif_ingress_registry_t registry(4);
    const notif_ingress_key_t ingress = key(40, 40);
    notif_wire_envelope_t selected_receipt;
    require(registry.reserve(ingress, payload, receipt(ingress)) == notif_ingress_status_t::SUCCESS,
            "atomic-admission reservation failed");

    std::size_t pushes = 0;
    const auto reject = [&pushes](nixlAuthenticatedNotification &&) {
        ++pushes;
        return NIXL_ERR_BACKEND;
    };
    require(registry.admit(ingress, payload, notification(ingress), reject, selected_receipt) ==
                    notif_ingress_status_t::CAPACITY_EXCEEDED &&
                pushes == 1 && registry.inventory().pending == 1,
            "failed queue admission escaped pending state");

    const auto accept = [&pushes](nixlAuthenticatedNotification &&) {
        ++pushes;
        return NIXL_SUCCESS;
    };
    require(registry.admit(ingress, payload, notification(ingress), accept, selected_receipt) ==
                    notif_ingress_status_t::SUCCESS &&
                pushes == 2 && registry.inventory().committed == 1,
            "successful queue push and commit were not one transition");
}

void
testAdmissionPublishesOutsideIngressLock() {
    notif_ingress_registry_t registry(4);
    const notif_ingress_key_t ingress = key(41, 41);
    const notif_wire_envelope_t frozen_receipt = receipt(ingress);
    require(registry.reserve(ingress, payload, frozen_receipt) ==
                notif_ingress_status_t::SUCCESS,
            "unlocked-admission reservation failed");

    bool observed_admitting = false;
    notif_ingress_status_t duplicate_status = notif_ingress_status_t::INTERNAL_ERROR;
    const auto inspect = [&](nixlAuthenticatedNotification &&) {
        const notif_ingress_inventory_t inventory = registry.inventory();
        observed_admitting = inventory.pending == 0 && inventory.admitting == 1 &&
            inventory.committed == 0;
        notif_wire_envelope_t duplicate_receipt;
        duplicate_status = registry.admit(ingress,
                                          payload,
                                          notification(ingress),
                                          acceptNotification,
                                          duplicate_receipt);
        return NIXL_SUCCESS;
    };

    notif_wire_envelope_t selected_receipt;
    require(registry.admit(
                ingress, payload, notification(ingress), inspect, selected_receipt) ==
                notif_ingress_status_t::SUCCESS &&
                observed_admitting &&
                duplicate_status == notif_ingress_status_t::DUPLICATE_PENDING &&
                selected_receipt == frozen_receipt && registry.inventory().admitting == 0 &&
                registry.inventory().committed == 1,
            "queue publication held the ingress lock or lost its transactional phase");
    require(registry.completeReceipt(ingress, NIXL_SUCCESS) ==
                notif_ingress_status_t::SUCCESS,
            "unlocked-admission receipt did not drain");
}

void
testAdmissionFailureAfterRouteTerminationRetiresReservation() {
    notif_ingress_registry_t registry(4);
    const notif_ingress_key_t ingress = key(42, 42);
    require(registry.reserve(ingress, payload, receipt(ingress)) ==
                notif_ingress_status_t::SUCCESS,
            "terminal-admission reservation failed");

    const auto terminate = [&](nixlAuthenticatedNotification &&) {
        require(registry.failConnection(ingress.connectionIdentity) ==
                    notif_ingress_status_t::SUCCESS &&
                    registry.inventory().admitting == 1,
                "connection failure did not preserve the in-flight admission transaction");
        return NIXL_ERR_BACKEND;
    };
    notif_wire_envelope_t selected_receipt;
    require(registry.admit(
                ingress, payload, notification(ingress), terminate, selected_receipt) ==
                notif_ingress_status_t::ROUTE_TERMINAL &&
                registry.inventory().pending == 0 && registry.inventory().admitting == 0 &&
                registry.inventory().committed == 0,
            "failed in-flight admission survived terminal route ownership");
}

void
testThrowingAdmissionQuarantinesAmbiguousAuthority() {
    notif_ingress_registry_t registry(4);
    const notif_ingress_key_t ingress = key(43, 43);
    require(registry.reserve(ingress, payload, receipt(ingress)) ==
                notif_ingress_status_t::SUCCESS,
            "throwing-admission reservation failed");

    const auto throw_from_push = [](nixlAuthenticatedNotification &&) -> nixl_status_t {
        throw std::runtime_error("synthetic queue publication failure");
    };
    notif_wire_envelope_t selected_receipt;
    require(registry.admit(
                ingress, payload, notification(ingress), throw_from_push, selected_receipt) ==
                notif_ingress_status_t::INTERNAL_ERROR &&
                registry.inventory().admitting == 0 && registry.inventory().quarantined == 1,
            "ambiguous throwing admission did not fail closed into quarantine");
}

void
testDisconnectRetainsCommittedReceiptUntilDrain() {
    notif_ingress_registry_t registry(4);
    const notif_ingress_key_t ingress = key(50, 50);
    notif_wire_envelope_t selected_receipt;
    require(registry.reserve(ingress, payload, receipt(ingress)) ==
                    notif_ingress_status_t::SUCCESS &&
                registry.admit(
                    ingress, payload, notification(ingress), acceptNotification, selected_receipt) ==
                    notif_ingress_status_t::SUCCESS &&
                registry.failConnection(ingress.connectionIdentity) ==
                    notif_ingress_status_t::SUCCESS,
            "disconnect-drain fixture could not retain committed receipt authority");

    auto drain = std::async(std::launch::async, [&registry, &ingress] {
        require(registry.drainConnection(ingress.connectionIdentity, std::chrono::seconds(5)) ==
                    notif_ingress_status_t::SUCCESS,
                "disconnect drain failed");
    });
    require(drain.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout,
            "disconnect drain returned before committed receipt terminality");
    require(registry.completeReceipt(ingress, NIXL_SUCCESS) ==
                notif_ingress_status_t::SUCCESS,
            "disconnect-drain receipt could not complete");
    require(drain.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
            "disconnect drain did not wake on committed receipt terminality");
    drain.get();
    require(registry.retireConnection(ingress.connectionIdentity) ==
                notif_ingress_status_t::SUCCESS,
            "drained connection authority could not retire");
}

void
testRetirementDoesNotEraseQuarantine() {
    notif_ingress_registry_t registry(4);
    const notif_ingress_key_t ingress = key(51, 51);
    notif_wire_envelope_t selected_receipt;
    require(registry.reserve(ingress, payload, receipt(ingress)) ==
                    notif_ingress_status_t::SUCCESS &&
                registry.admit(
                    ingress, payload, notification(ingress), acceptNotification, selected_receipt) ==
                    notif_ingress_status_t::SUCCESS &&
                registry.failRoute(ingress.route) == notif_ingress_status_t::SUCCESS &&
                registry.completeReceipt(ingress, NIXL_ERR_REMOTE_DISCONNECT) ==
                    notif_ingress_status_t::RECEIPT_FAILED,
            "retirement-quarantine fixture did not reach terminal receipt failure");
    require(registry.drainRoute(ingress.route, std::chrono::seconds(5)) ==
                notif_ingress_status_t::SUCCESS,
            "route drain failed");
    require(registry.retireRoute(ingress.route) == notif_ingress_status_t::SUCCESS &&
                registry.inventory().quarantined == 1,
            "route retirement erased process-fatal quarantine evidence");
}

void
testDrainTimeoutPreservesLiveAuthority() {
    notif_ingress_registry_t registry(4);
    const notif_ingress_key_t ingress = key(52, 52);
    notif_wire_envelope_t selected_receipt;
    require(registry.reserve(ingress, payload, receipt(ingress)) ==
                    notif_ingress_status_t::SUCCESS &&
                registry.admit(
                    ingress, payload, notification(ingress), acceptNotification, selected_receipt) ==
                    notif_ingress_status_t::SUCCESS &&
                registry.failConnection(ingress.connectionIdentity) ==
                    notif_ingress_status_t::SUCCESS,
            "timeout fixture could not retain committed authority");

    require(registry.drainConnection(ingress.connectionIdentity, std::chrono::milliseconds(5)) ==
                notif_ingress_status_t::DRAIN_TIMEOUT,
            "live committed authority did not enforce its drain deadline");
    const notif_ingress_snapshot_t timed_out = registry.snapshot();
    require(timed_out.inventory.committed == 1 && timed_out.activeRecords.size() == 1 &&
                timed_out.activeRecords.front().key == ingress,
            "drain timeout erased or rewrote live authority");

    require(registry.completeReceipt(ingress, NIXL_ERR_REMOTE_DISCONNECT) ==
                    notif_ingress_status_t::RECEIPT_FAILED &&
                registry.drainConnection(ingress.connectionIdentity, std::chrono::seconds(5)) ==
                    notif_ingress_status_t::SUCCESS,
            "timed-out authority could not resolve into quarantine");
    const notif_ingress_snapshot_t resolved = registry.snapshot();
    require(resolved.inventory.quarantined == 1 && resolved.activeRecords.size() == 1 &&
                resolved.activeRecords.front().phase ==
                    notif_ingress_inventory_phase_t::QUARANTINED,
            "post-timeout resolution lost quarantine authority");
}

void
testInventorySnapshotIsAtomicExactAndCanonical() {
    notif_ingress_registry_t registry(8);
    notif_wire_envelope_t selected_receipt;

    const notif_ingress_key_t replaying = key(20, 60);
    require(registry.reserve(replaying, payload, receipt(replaying)) ==
                    notif_ingress_status_t::SUCCESS &&
                registry.admit(replaying,
                               payload,
                               notification(replaying),
                               acceptNotification,
                               selected_receipt) == notif_ingress_status_t::SUCCESS &&
                registry.completeReceipt(replaying, NIXL_SUCCESS) ==
                    notif_ingress_status_t::SUCCESS &&
                registry.reserve(replaying, payload, receipt(replaying)) ==
                    notif_ingress_status_t::DUPLICATE_COMMITTED &&
                registry.admit(replaying,
                               payload,
                               notification(replaying),
                               acceptNotification,
                               selected_receipt) == notif_ingress_status_t::DUPLICATE_COMMITTED,
            "snapshot fixture could not establish replay state");

    const notif_ingress_key_t committed = key(30, 61);
    require(registry.reserve(committed, payload, receipt(committed)) ==
                    notif_ingress_status_t::SUCCESS &&
                registry.admit(committed,
                               payload,
                               notification(committed),
                               acceptNotification,
                               selected_receipt) == notif_ingress_status_t::SUCCESS,
            "snapshot fixture could not establish committed state");

    const notif_ingress_key_t pending = key(40, 62);
    require(registry.reserve(pending, payload, receipt(pending)) ==
                notif_ingress_status_t::SUCCESS,
            "snapshot fixture could not establish pending state");

    const notif_ingress_key_t quarantined = key(50, 63);
    require(registry.reserve(quarantined, payload, receipt(quarantined)) ==
                    notif_ingress_status_t::SUCCESS &&
                registry.admit(quarantined,
                               payload,
                               notification(quarantined),
                               acceptNotification,
                               selected_receipt) == notif_ingress_status_t::SUCCESS &&
                registry.completeReceipt(quarantined, NIXL_ERR_REMOTE_DISCONNECT) ==
                    notif_ingress_status_t::RECEIPT_FAILED,
            "snapshot fixture could not establish quarantine state");

    const notif_ingress_key_t admitting = key(60, 64);
    require(registry.reserve(admitting, payload, receipt(admitting)) ==
                notif_ingress_status_t::SUCCESS,
            "snapshot fixture could not establish admission state");
    bool observed_snapshot = false;
    const auto inspect = [&](nixlAuthenticatedNotification &&) {
        const notif_ingress_snapshot_t snapshot = registry.snapshot();
        const std::size_t exact_records = snapshot.inventory.pending +
            snapshot.inventory.admitting + snapshot.inventory.committed +
            snapshot.inventory.replaying + snapshot.inventory.quarantined;
        const std::vector<notif_ingress_inventory_phase_t> expected_phases = {
            notif_ingress_inventory_phase_t::REPLAYING,
            notif_ingress_inventory_phase_t::COMMITTED,
            notif_ingress_inventory_phase_t::PENDING,
            notif_ingress_inventory_phase_t::QUARANTINED,
            notif_ingress_inventory_phase_t::ADMITTING,
        };
        observed_snapshot = snapshot.inventory.pending == 1 &&
            snapshot.inventory.admitting == 1 && snapshot.inventory.committed == 1 &&
            snapshot.inventory.replaying == 1 && snapshot.inventory.completed == 1 &&
            snapshot.inventory.quarantined == 1 &&
            exact_records == snapshot.activeRecords.size() &&
            snapshot.activeRecords.size() == expected_phases.size();
        for (std::size_t index = 0;
             observed_snapshot && index < snapshot.activeRecords.size();
             ++index) {
            observed_snapshot = snapshot.activeRecords[index].key.deliveryIdentity ==
                    static_cast<std::uint64_t>((index + 2) * 10) &&
                snapshot.activeRecords[index].phase == expected_phases[index];
        }
        return NIXL_SUCCESS;
    };
    require(registry.admit(admitting,
                           payload,
                           notification(admitting),
                           inspect,
                           selected_receipt) == notif_ingress_status_t::SUCCESS &&
                observed_snapshot,
            "ingress snapshot counts and exact records were torn or noncanonical");
}

} // namespace

int
main() {
    try {
        testPrimaryIdentityRejectsAuthorityConflict();
        testIrrevocableAdmissionAndBoundedReplay();
        testPendingRetirementAndReceiptFailure();
        testShutdownClosesAdmissionWithoutRevokingCommit();
        testMonotonicAntiReplaySurvivesTombstoneEviction();
        testFailedReceiptRetainsExactQuarantineAuthority();
        testAdmissionIsOneLinearizedQueueTransaction();
        testAdmissionPublishesOutsideIngressLock();
        testAdmissionFailureAfterRouteTerminationRetiresReservation();
        testThrowingAdmissionQuarantinesAmbiguousAuthority();
        testDisconnectRetainsCommittedReceiptUntilDrain();
        testRetirementDoesNotEraseQuarantine();
        testDrainTimeoutPreservesLiveAuthority();
        testInventorySnapshotIsAtomicExactAndCanonical();
    }
    catch (const std::exception &error) {
        std::cerr << "ucx_notif_ingress_test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "ucx_notif_ingress_test passed\n";
    return EXIT_SUCCESS;
}
