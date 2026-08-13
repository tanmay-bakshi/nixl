/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "terminal_ucx_peer_fixture.h"

#include "ucx_backend.h"
#include "ucx_notif_delivery.h"
#include "ucx_notif_ingress.h"
#include "ucx_terminal_deadline.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <dirent.h>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

class nixlUcxConstructorFailureQualification {
public:
    static void
    constructAfterSharedProgressOwnerStart(const nixlBackendInitParams &params) {
        nixlUcxThreadPoolEngine engine(
            params,
            nixlUcxThreadPoolEngine::qualification_constructor_failure_point_t::
                AFTER_SHARED_PROGRESS_OWNER_STARTED);
    }

    static void
    constructAfterFirstDedicatedProgressOwnerStart(const nixlBackendInitParams &params) {
        nixlUcxThreadPoolEngine engine(
            params,
            nixlUcxThreadPoolEngine::qualification_constructor_failure_point_t::
                AFTER_FIRST_DEDICATED_PROGRESS_OWNER_STARTED);
    }
};

class nixlUcxConnectionRetirementQualification {
public:
    struct observation_t {
        std::size_t successfulLookups = 0;
        std::size_t detachedConnections = 0;
        bool registryLockReleased = false;
        bool connectionRetainedAfterDetach = false;
        bool connectionReleasedAfterRetirement = false;
    };

    static observation_t
    exercise(nixlUcxEngine &engine, const std::string &remote_agent) {
        std::atomic<bool> stop_lookup = false;
        std::atomic<std::size_t> successful_lookups = 0;
        std::thread lookup([&]() {
            while (!stop_lookup.load(std::memory_order_acquire)) {
                if (engine.getConnection(remote_agent) != nullptr) {
                    successful_lookups.fetch_add(1, std::memory_order_relaxed);
                }
                std::this_thread::yield();
            }
        });

        const auto lookup_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (successful_lookups.load(std::memory_order_relaxed) == 0 &&
               std::chrono::steady_clock::now() < lookup_deadline) {
            std::this_thread::yield();
        }

        nixlUcxEngine::remote_connection_map_t retired_connections =
            engine.detachRemoteConnections();
        stop_lookup.store(true, std::memory_order_release);
        lookup.join();

        std::weak_ptr<nixlUcxConnection> retired_connection;
        const auto retired = retired_connections.find(remote_agent);
        if (retired != retired_connections.end()) {
            retired_connection = retired->second;
        }
        const std::size_t detached_connection_count = retired_connections.size();
        const bool connection_retained = !retired_connection.expired();
        std::unique_lock registry_lock(engine.connectionMutex_, std::try_to_lock);
        const bool registry_lock_released = registry_lock.owns_lock();
        if (registry_lock_released) {
            registry_lock.unlock();
        }

        retired_connections.clear();
        return {
            .successfulLookups = successful_lookups.load(std::memory_order_relaxed),
            .detachedConnections = detached_connection_count,
            .registryLockReleased = registry_lock_released,
            .connectionRetainedAfterDetach = connection_retained,
            .connectionReleasedAfterRetirement = retired_connection.expired(),
        };
    }
};

namespace {

using namespace std::chrono_literals;
using nixl::qualification::tcp_peer_channel_observation_t;
using namespace nixl::ucx;

class qualification_error final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct options_t {
    std::filesystem::path output;
};

struct case_result_t {
    std::string name;
    tcp_peer_channel_observation_t inventory;
};

struct process_resource_snapshot_t {
    std::unordered_set<std::uint64_t> tasks;
    std::unordered_set<std::uint64_t> descriptors;

    [[nodiscard]] bool
    operator==(const process_resource_snapshot_t &) const = default;
};

void
require(bool condition, std::string_view message) {
    if (!condition) {
        throw qualification_error(std::string(message));
    }
}

[[nodiscard]] std::unordered_set<std::uint64_t>
readNumericDirectory(const std::filesystem::path &path) {
    DIR *const directory = ::opendir(path.c_str());
    require(directory != nullptr, std::string("could not open process inventory ") + path.string());
    std::unordered_set<std::uint64_t> entries;
    while (true) {
        const dirent *const entry = ::readdir(directory);
        if (entry == nullptr) {
            break;
        }
        const std::string_view name(entry->d_name);
        if (name == "." || name == "..") {
            continue;
        }
        require(std::all_of(
                    name.begin(), name.end(), [](char byte) { return byte >= '0' && byte <= '9'; }),
                std::string("process inventory contains a nonnumeric entry: ") + std::string(name));
        entries.insert(std::stoull(std::string(name)));
    }
    require(::closedir(directory) == 0,
            std::string("could not close process inventory ") + path.string());
    return entries;
}

[[nodiscard]] process_resource_snapshot_t
processResourceSnapshot() {
    return {
        .tasks = readNumericDirectory("/proc/self/task"),
        .descriptors = readNumericDirectory("/proc/self/fd"),
    };
}

template<typename Constructor>
void
requireConstructorFailureRestoresResources(Constructor &&construct, std::string_view context) {
    const process_resource_snapshot_t before = processResourceSnapshot();
    bool failed = false;
    try {
        construct();
    }
    catch (const std::runtime_error &) {
        failed = true;
    }
    require(failed, std::string(context) + " did not inject its constructor failure");

    constexpr auto restoration_timeout = 2s;
    const auto deadline = std::chrono::steady_clock::now() + restoration_timeout;
    process_resource_snapshot_t after = processResourceSnapshot();
    while (!(after == before) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
        after = processResourceSnapshot();
    }
    require(after == before,
            std::string(context) + " did not restore its exact thread and descriptor inventory");
}

[[nodiscard]] nixlBackendInitParams
constructorFailureParams(nixl_b_params_t &custom_params) {
    custom_params = {
        {"num_workers", "3"},
        {"num_threads", "2"},
        {"split_batch_size", "2"},
        {"ucx_error_handling_mode", "peer"},
    };
    return {
        .localAgent = "terminal-owner-constructor-qualification",
        .localAgentIncarnation = "00000000-0000-4000-8000-000000000123",
        .type = "UCX",
        .customParams = &custom_params,
        .enableProgTh = true,
        .pthrDelay = 1,
        .syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW,
        .enableTelemetry_ = false,
    };
}

void
testPartialConstructionAfterSharedOwnerStart() {
    nixl_b_params_t custom_params;
    const nixlBackendInitParams params = constructorFailureParams(custom_params);
    requireConstructorFailureRestoresResources(
        [&params]() {
            nixlUcxConstructorFailureQualification::constructAfterSharedProgressOwnerStart(params);
        },
        "shared-owner constructor rollback");
}

void
testPartialConstructionAfterFirstDedicatedOwnerStart() {
    nixl_b_params_t custom_params;
    const nixlBackendInitParams params = constructorFailureParams(custom_params);
    requireConstructorFailureRestoresResources(
        [&params]() {
            nixlUcxConstructorFailureQualification::constructAfterFirstDedicatedProgressOwnerStart(
                params);
        },
        "dedicated-owner constructor rollback");
}

void
testConnectionRetirementOutsideRegistryLock() {
    const std::string local_agent = "terminal-connection-retirement-qualification";
    nixl_b_params_t custom_params = {
        {"num_workers", "2"},
        {"num_threads", "0"},
        {"ucx_error_handling_mode", "peer"},
    };
    const nixlBackendInitParams params = {
        .localAgent = local_agent,
        .localAgentIncarnation = "00000000-0000-4000-8000-000000000124",
        .type = "UCX",
        .customParams = &custom_params,
        .enableProgTh = true,
        .pthrDelay = 1,
        .syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW,
        .enableTelemetry_ = false,
    };
    std::unique_ptr<nixlUcxEngine> engine = nixlUcxEngine::create(params);
    require(engine != nullptr, "connection-retirement engine construction failed");
    require(engine->connect(local_agent) == NIXL_SUCCESS,
            "connection-retirement fixture could not establish its UCX loopback");

    const auto observation =
        nixlUcxConnectionRetirementQualification::exercise(*engine, local_agent);
    require(observation.successfulLookups > 0 && observation.detachedConnections == 1 &&
                observation.registryLockReleased && observation.connectionRetainedAfterDetach &&
                observation.connectionReleasedAfterRetirement,
            "connection retirement destroyed a UCX endpoint under the registry lock");
}

[[nodiscard]] options_t
parseOptions(int argc, char **argv) {
    require(argc == 4 && std::string_view(argv[1]) == "--all" &&
                std::string_view(argv[2]) == "--output",
            "usage: terminal_ucx_owner_lifecycle_qualification --all --output PATH");
    const std::filesystem::path output(argv[3]);
    require(!output.empty(), "output path is empty");
    require(!std::filesystem::exists(output), "output path already exists");
    return {.output = output};
}

[[nodiscard]] tcp_peer_channel_observation_t
closedInventory() {
    return {
        .capacity = 1,
        .queuedChannelEvents = 0,
        .activeChannelSubscriptions = 0,
        .retainedPublicSubscriptions = 0,
        .backendProducers = 0,
        .activeCallbackSlots = 0,
        .queuedOwnerContinuations = 0,
        .acceptingSubscriptions = false,
        .closed = true,
        .fatal = 0,
        .eventfdError = 0,
    };
}

void
requireCleanInventory(const tcp_peer_channel_observation_t &inventory, std::string_view context) {
    const bool clean = inventory.capacity > 0 && inventory.queuedChannelEvents == 0 &&
        inventory.activeChannelSubscriptions == 0 && inventory.retainedPublicSubscriptions == 0 &&
        inventory.backendProducers == 0 && inventory.activeCallbackSlots == 0 &&
        inventory.queuedOwnerContinuations == 0 && !inventory.acceptingSubscriptions &&
        inventory.closed && inventory.fatal == 0 && inventory.eventfdError == 0;
    require(clean, std::string(context) + " retained terminal lifecycle inventory");
}

[[nodiscard]] notif_wire_uuid_t
uuid(std::uint8_t seed) {
    notif_wire_uuid_t value;
    for (std::size_t index = 0; index < value.bytes.size(); ++index) {
        value.bytes[index] = static_cast<std::uint8_t>(seed + index);
    }
    value.bytes[6] = static_cast<std::uint8_t>((value.bytes[6] & 0x0fU) | 0x40U);
    value.bytes[8] = static_cast<std::uint8_t>((value.bytes[8] & 0x3fU) | 0x80U);
    return value;
}

[[nodiscard]] notif_delivery_key_t
deliveryKey(std::uint64_t identity) {
    return {
        .identity = identity,
        .sourceHandleIdentity = 47,
        .sourceGeneration = 53,
        .route =
            {
                .handleIdentity = 11,
                .handleGeneration = 7,
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

[[nodiscard]] notif_delivery_receipt_t
deliveryReceipt(const notif_delivery_key_t &delivery) {
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

[[nodiscard]] notif_ingress_key_t
ingressKey(std::uint64_t delivery_identity) {
    const notif_wire_uuid_t source_agent = uuid(65);
    const notif_wire_uuid_t source_backend = uuid(81);
    return {
        .sourceBackendIncarnation = source_backend,
        .sourceHandleIdentity = 71,
        .sourceGeneration = 73,
        .deliveryIdentity = delivery_identity,
        .destinationAgentIncarnation = uuid(97),
        .destinationBackendIncarnation = uuid(113),
        .route =
            {
                .handleIdentity = 79,
                .handleGeneration = 83,
                .remoteAgentIncarnation = source_agent,
                .remoteBackendIncarnation = source_backend,
            },
        .capability = uuid(129),
        .capabilityEpoch = 89,
        .connectionIdentity = 101,
        .sourceWorkerIncarnation = uuid(145),
        .receiptWorkerIncarnation = uuid(161),
        .endpointIdentity = 103,
    };
}

[[nodiscard]] notif_wire_envelope_t
ingressReceipt(const notif_ingress_key_t &ingress) {
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

[[nodiscard]] nixlAuthenticatedNotification
ingressNotification(const notif_ingress_key_t &ingress) {
    return {
        .remoteAgent = "source",
        .payload = "terminal-owner-lifecycle",
        .handleIdentity = ingress.route.handleIdentity,
        .generation = ingress.route.handleGeneration,
        .connectionIdentity = ingress.connectionIdentity,
        .endpointIdentity = ingress.endpointIdentity,
    };
}

void
requireIngressDrained(const notif_ingress_registry_t &registry, std::string_view context) {
    const notif_ingress_snapshot_t snapshot = registry.snapshot();
    const notif_ingress_inventory_t &inventory = snapshot.inventory;
    require(inventory.pending == 0 && inventory.admitting == 0 && inventory.committed == 0 &&
                inventory.replaying == 0 && inventory.quarantined == 0 &&
                snapshot.activeRecords.empty(),
            std::string(context) + " retained a destination admission obligation");
}

void
requireDeliveryDrained(const notif_delivery_registry_t &registry, std::string_view context) {
    const notif_delivery_snapshot_t snapshot = registry.snapshot();
    require(snapshot.inventory.outstanding == 0 && snapshot.activeRecords.empty(),
            std::string(context) + " retained a source receipt obligation");
}

void
testReceiptIdentityBinding() {
    notif_delivery_registry_t registry(1);
    const notif_delivery_key_t delivery = deliveryKey(1);
    std::size_t terminal_count = 0;
    nixl_status_t terminal_status = NIXL_ERR_BACKEND;
    require(registry.registerDelivery(
                delivery,
                [&](nixl_status_t status, std::uint64_t) {
                    terminal_status = status;
                    ++terminal_count;
                },
                []() { return true; }) == notif_delivery_status_t::SUCCESS,
            "identity fixture could not register its delivery");
    require(registry.completeLocal(delivery.identity, NIXL_SUCCESS, 1) ==
                notif_delivery_status_t::SUCCESS,
            "identity fixture local completion failed");

    std::vector<notif_delivery_receipt_t> conflicts;
    notif_delivery_receipt_t route_identity = deliveryReceipt(delivery);
    ++route_identity.route.handleIdentity;
    conflicts.push_back(route_identity);
    notif_delivery_receipt_t route_generation = deliveryReceipt(delivery);
    ++route_generation.route.handleGeneration;
    conflicts.push_back(route_generation);
    notif_delivery_receipt_t capability_epoch = deliveryReceipt(delivery);
    ++capability_epoch.capabilityEpoch;
    conflicts.push_back(capability_epoch);
    notif_delivery_receipt_t worker = deliveryReceipt(delivery);
    worker.receiptWorkerIncarnation = uuid(177);
    conflicts.push_back(worker);
    notif_delivery_receipt_t endpoint = deliveryReceipt(delivery);
    ++endpoint.endpointIdentity;
    conflicts.push_back(endpoint);

    for (const notif_delivery_receipt_t &conflict : conflicts) {
        require(registry.acceptReceipt(conflict, 2) == notif_delivery_status_t::IDENTITY_CONFLICT,
                "conflicting receipt authority was accepted");
    }
    require(terminal_count == 0,
            "conflicting receipt published terminality before exact admission");
    require(registry.acceptReceipt(deliveryReceipt(delivery), 3) ==
                    notif_delivery_status_t::SUCCESS &&
                terminal_count == 1 && terminal_status == NIXL_SUCCESS,
            "exact receipt authority did not complete exactly once");
    require(registry.failAll(NIXL_ERR_CANCELED, 4) == notif_delivery_status_t::SUCCESS,
            "identity fixture close failed");
    requireDeliveryDrained(registry, "identity fixture");
}

void
testSourceAndDestinationObligationsDrain() {
    notif_delivery_registry_t source(1);
    const notif_delivery_key_t delivery = deliveryKey(2);
    std::size_t source_terminal_count = 0;
    nixl_status_t source_terminal_status = NIXL_ERR_BACKEND;
    require(source.registerDelivery(
                delivery,
                [&](nixl_status_t status, std::uint64_t) {
                    source_terminal_status = status;
                    ++source_terminal_count;
                },
                []() { return true; }) == notif_delivery_status_t::SUCCESS,
            "source obligation registration failed");
    require(source.completeLocal(delivery.identity, NIXL_SUCCESS, 10) ==
                    notif_delivery_status_t::SUCCESS &&
                source.acceptReceipt(deliveryReceipt(delivery), 11) ==
                    notif_delivery_status_t::SUCCESS &&
                source_terminal_count == 1 && source_terminal_status == NIXL_SUCCESS,
            "source receipt obligation did not join exactly once");
    require(source.failAll(NIXL_ERR_CANCELED, 12) == notif_delivery_status_t::SUCCESS,
            "source obligation registry close failed");
    requireDeliveryDrained(source, "source obligation fixture");

    notif_ingress_registry_t destination(1);
    const notif_ingress_key_t ingress = ingressKey(2);
    const std::array<std::uint8_t, 4> payload = {1, 2, 3, 4};
    notif_wire_envelope_t selected_receipt;
    std::size_t destination_push_count = 0;
    require(destination.reserve(ingress, payload, ingressReceipt(ingress)) ==
                notif_ingress_status_t::SUCCESS,
            "destination obligation reservation failed");
    require(destination.admit(
                ingress,
                payload,
                ingressNotification(ingress),
                [&](nixlAuthenticatedNotification &&) {
                    ++destination_push_count;
                    return NIXL_SUCCESS;
                },
                selected_receipt) == notif_ingress_status_t::SUCCESS &&
                destination_push_count == 1 && selected_receipt == ingressReceipt(ingress),
            "destination queue admission did not commit exactly once");
    require(destination.completeReceipt(ingress, NIXL_SUCCESS) == notif_ingress_status_t::SUCCESS,
            "destination receipt obligation did not complete");
    require(destination.failAll() == notif_ingress_status_t::SUCCESS,
            "destination obligation registry close failed");
    requireIngressDrained(destination, "destination obligation fixture");
}

void
testAcceptedTaskRace() {
    notif_ingress_registry_t registry(1);
    const notif_ingress_key_t ingress = ingressKey(3);
    const std::array<std::uint8_t, 4> payload = {5, 6, 7, 8};
    require(registry.reserve(ingress, payload, ingressReceipt(ingress)) ==
                notif_ingress_status_t::SUCCESS,
            "admission-close race reservation failed");

    std::promise<void> push_entered_promise;
    std::future<void> push_entered = push_entered_promise.get_future();
    std::promise<void> release_push_promise;
    const std::shared_future<void> release_push = release_push_promise.get_future().share();
    std::atomic<std::size_t> push_count = 0;
    notif_wire_envelope_t selected_receipt;
    std::future<notif_ingress_status_t> admission = std::async(std::launch::async, [&]() {
        return registry.admit(
            ingress,
            payload,
            ingressNotification(ingress),
            [&](nixlAuthenticatedNotification &&) {
                ++push_count;
                push_entered_promise.set_value();
                release_push.wait();
                return NIXL_SUCCESS;
            },
            selected_receipt);
    });

    if (push_entered.wait_for(5s) != std::future_status::ready) {
        release_push_promise.set_value();
        static_cast<void>(admission.get());
        throw qualification_error("admission-close race never reached queue admission");
    }
    const notif_ingress_snapshot_t admitting = registry.snapshot();
    require(admitting.inventory.admitting == 1 && admitting.activeRecords.size() == 1 &&
                admitting.activeRecords[0].phase == notif_ingress_inventory_phase_t::ADMITTING,
            "admission-close race did not expose one irrevocable task");
    require(registry.failAll() == notif_ingress_status_t::SUCCESS,
            "admission-close race could not close admission");
    release_push_promise.set_value();
    require(admission.get() == notif_ingress_status_t::SUCCESS && push_count.load() == 1,
            "accepted task racing close did not execute exactly once");
    require(registry.completeReceipt(ingress, NIXL_SUCCESS) == notif_ingress_status_t::SUCCESS,
            "accepted task racing close did not drain its receipt");
    requireIngressDrained(registry, "admission-close race");
}

void
testCallbackOnlyLifecycle() {
    notif_delivery_registry_t registry(1);
    const notif_delivery_key_t delivery = deliveryKey(4);
    std::size_t terminal_count = 0;
    nixl_status_t terminal_status = NIXL_ERR_BACKEND;
    std::uint64_t terminal_timestamp_ns = 0;
    require(registry.registerDelivery(
                delivery,
                [&](nixl_status_t status, std::uint64_t timestamp_ns) {
                    terminal_status = status;
                    terminal_timestamp_ns = timestamp_ns;
                    ++terminal_count;
                },
                []() { return true; }) == notif_delivery_status_t::SUCCESS,
            "callback-only delivery registration failed");
    require(registry.completeLocal(delivery.identity, NIXL_SUCCESS, 20) ==
                    notif_delivery_status_t::SUCCESS &&
                terminal_count == 0,
            "callback-only delivery escaped its receipt join");
    require(
        registry.acceptReceipt(deliveryReceipt(delivery), 21) == notif_delivery_status_t::SUCCESS &&
            terminal_count == 1 && terminal_status == NIXL_SUCCESS && terminal_timestamp_ns == 21,
        "callback-only delivery did not dispatch exactly once");
    require(registry.failAll(NIXL_ERR_CANCELED, 22) == notif_delivery_status_t::SUCCESS,
            "callback-only registry close failed");
    const notif_delivery_snapshot_t snapshot = registry.snapshot();
    require(!snapshot.inventory.accepting, "callback-only registry remained open");
    requireDeliveryDrained(registry, "callback-only fixture");
}

void
testExactZeroNativeInventory() {
    notif_delivery_registry_t source(1);
    require(source.failAll(NIXL_ERR_CANCELED, 1) == notif_delivery_status_t::SUCCESS,
            "empty source registry close failed");
    const notif_delivery_snapshot_t source_snapshot = source.snapshot();
    require(source_snapshot.inventory.outstanding == 0 &&
                source_snapshot.inventory.completed == 0 && !source_snapshot.inventory.accepting &&
                source_snapshot.activeRecords.empty(),
            "empty source registry did not report exact zero");

    notif_ingress_registry_t destination(1);
    require(destination.failAll() == notif_ingress_status_t::SUCCESS,
            "empty destination registry close failed");
    const notif_ingress_snapshot_t destination_snapshot = destination.snapshot();
    const notif_ingress_inventory_t &inventory = destination_snapshot.inventory;
    require(inventory.pending == 0 && inventory.admitting == 0 && inventory.committed == 0 &&
                inventory.replaying == 0 && inventory.completed == 0 &&
                inventory.quarantined == 0 && destination_snapshot.activeRecords.empty(),
            "empty destination registry did not report exact zero");
}

void
testNativeTransferDeadline() {
    require(nativeTransferTimeoutNs == 60'000'000'000ULL,
            "native transfer timeout differs from the frozen contract");
    terminal_deadline_owner_t owner(2);
    const auto owner_start_deadline = std::chrono::steady_clock::now() + 2s;
    while (!owner.inventory().ownerAlive &&
           std::chrono::steady_clock::now() < owner_start_deadline) {
        std::this_thread::yield();
    }
    require(owner.inventory().ownerAlive, "native deadline owner did not start");

    const terminal_deadline_key_t transfer = {.handleIdentity = 1, .generation = 1};
    const terminal_deadline_key_t activity = {.handleIdentity = 2, .generation = 1};
    constexpr std::uint64_t transfer_timeout_ns = 30'000'000;
    constexpr std::uint64_t activity_timeout_ns = 250'000'000;
    std::promise<std::uint64_t> expired_promise;
    std::future<std::uint64_t> expired = expired_promise.get_future();
    std::atomic<std::size_t> expiry_count = 0;
    std::atomic<bool> expiry_identity_matches = false;
    std::atomic<bool> expiry_acknowledged = false;
    const std::uint64_t transfer_anchor = terminal_deadline_owner_t::monotonicTimestampNs();
    require(owner.arm(transfer,
                      transfer_anchor,
                      transfer_timeout_ns,
                      [&](const terminal_deadline_key_t &expired_key) {
                          expiry_identity_matches.store(expired_key == transfer);
                          ++expiry_count;
                          expired_promise.set_value(
                              terminal_deadline_owner_t::monotonicTimestampNs());
                          expiry_acknowledged.store(owner.acknowledgeExpiry(expired_key) ==
                                                    terminal_deadline_status_t::SUCCESS);
                          return NIXL_SUCCESS;
                      }) == terminal_deadline_status_t::SUCCESS,
            "native transfer deadline registration failed");
    std::this_thread::sleep_for(10ms);
    const std::uint64_t activity_anchor = terminal_deadline_owner_t::monotonicTimestampNs();
    require(owner.arm(activity,
                      activity_anchor,
                      activity_timeout_ns,
                      [](const auto &) { return NIXL_SUCCESS; }) ==
                terminal_deadline_status_t::SUCCESS,
            "intervening deadline activity failed");

    require(expired.wait_for(2s) == std::future_status::ready,
            "post-anchored native transfer deadline did not expire");
    const std::uint64_t expired_ns = expired.get();
    require(expired_ns >= transfer_anchor + transfer_timeout_ns &&
                expired_ns < activity_anchor + activity_timeout_ns && expiry_count.load() == 1 &&
                expiry_identity_matches.load() && expiry_acknowledged.load(),
            "intervening activity reset or duplicated the native transfer deadline");
    require(owner.retire(transfer) == terminal_deadline_status_t::EXPIRY_WON,
            "native transfer deadline lost its exact expiry winner");
    require(owner.retire(activity) == terminal_deadline_status_t::SUCCESS,
            "intervening deadline activity could not retire");
    require(owner.close() == NIXL_SUCCESS, "native deadline owner close failed");
    const terminal_deadline_snapshot_t snapshot = owner.snapshot();
    require(snapshot.inventory.active == 0 && snapshot.inventory.expired == 1 &&
                snapshot.inventory.retired == 1 && snapshot.inventory.fatalStatus == NIXL_SUCCESS &&
                !snapshot.inventory.accepting && !snapshot.inventory.ownerAlive &&
                snapshot.activeKeys.empty(),
            "native deadline owner retained lifecycle inventory");
}

[[nodiscard]] std::vector<case_result_t>
runCases() {
    const auto admission = nixl::qualification::runTcpAdmissionReceiptReleaseFixture("thread_pool");
    require(admission.transfer.terminalStatus == NIXL_SUCCESS &&
                admission.transfer.terminalEventCount == 1 && admission.transfer.ownerWoken &&
                admission.stateWhileHeld == nixl_xfer_attestation_state_t::REMOTE_FLUSHED &&
                admission.heldSourceHandleIdentity != 0 && admission.heldSourceGeneration != 0 &&
                admission.heldDeliveryIdentity != 0 && admission.notificationCount == 1 &&
                admission.subscriptionActiveWhileHeld && admission.remoteFlushedWhileHeld &&
                admission.peerExitedCleanly && !admission.peerExitedBySignal,
            "admission-receipt fixture did not preserve terminal authority");
    requireCleanInventory(admission.channel, "admission-receipt fixture");
    testReceiptIdentityBinding();
    testSourceAndDestinationObligationsDrain();

    const auto notification = nixl::qualification::runTcpNotificationFailureFixture("shared");
    require(notification.transfer.terminalStatus == NIXL_ERR_REMOTE_DISCONNECT &&
                notification.transfer.terminalEventCount == 1 && notification.transfer.ownerWoken &&
                notification.dataRemoteFlushedBeforeFailure &&
                notification.notificationFailureAfterRemoteFlush &&
                notification.faultPeerEngine == "shared" &&
                notification.faultPeerAdmissionReceiptHeld && notification.peerExitedBySignal,
            "notification-failure fixture did not preserve its terminal boundary");
    requireCleanInventory(notification.channel, "notification-failure fixture");

    const auto shutdown = nixl::qualification::runTcpShutdownCancellationFixture("thread_pool");
    require(shutdown.transfer.terminalStatus == NIXL_ERR_CANCELED &&
                shutdown.transfer.terminalEventCount == 1 && shutdown.transfer.ownerWoken &&
                shutdown.cancelStatus == NIXL_SUCCESS && shutdown.postedInFlight &&
                shutdown.drained && shutdown.peerExitedBySignal &&
                (shutdown.inventoryBeforeCancellation.backendProducers > 0 ||
                 shutdown.inventoryBeforeCancellation.activeCallbackSlots > 0 ||
                 shutdown.inventoryBeforeCancellation.queuedOwnerContinuations > 0),
            "shutdown-cancellation fixture did not drain an owner-only live request");
    requireCleanInventory(shutdown.inventoryAfterClose, "shutdown-cancellation fixture");

    testAcceptedTaskRace();
    testCallbackOnlyLifecycle();

    const auto unequal = nixl::qualification::runTcpUnequalWorkerFixture(4, 2);
    require(unequal.sourceWorkerCount == 4 && unequal.destinationWorkerCount == 2 &&
                unequal.completedTransferCount == 4 && unequal.notificationCount == 4 &&
                unequal.exercisedSourceWorkers == std::vector<std::size_t>{0, 1, 2, 3} &&
                unequal.peerExitedCleanly,
            "cross-owner fixture did not exercise its declared worker mapping");
    requireCleanInventory(unequal.channel, "cross-owner fixture");

    testPartialConstructionAfterSharedOwnerStart();
    testPartialConstructionAfterFirstDedicatedOwnerStart();
    testConnectionRetirementOutsideRegistryLock();
    testExactZeroNativeInventory();
    testNativeTransferDeadline();

    const tcp_peer_channel_observation_t local_inventory = closedInventory();
    return {
        {
            .name = "attached_notification_waits_for_authenticated_remote_queue_admission_receipt",
            .inventory = admission.channel,
        },
        {
            .name = "receipt_identity_binds_route_handle_generation_epoch_worker_and_endpoint",
            .inventory = admission.channel,
        },
        {
            .name = "pending_source_receipt_and_destination_obligation_drain_to_zero",
            .inventory = admission.channel,
        },
        {
            .name = "notification_failure_after_final_remote_flush_callback",
            .inventory = notification.channel,
        },
        {
            .name = "owner_only_live_request_drains_during_shutdown",
            .inventory = shutdown.inventoryAfterClose,
        },
        {
            .name = "accepted_task_racing_admission_close_executes_exactly_once",
            .inventory = local_inventory,
        },
        {
            .name = "callback_only_lifecycle_drains_without_legacy_request",
            .inventory = local_inventory,
        },
        {
            .name = "partial_construction_after_shared_owner_start_restores_threads_and_fds",
            .inventory = local_inventory,
        },
        {
            .name =
                "partial_construction_after_first_dedicated_owner_start_restores_threads_and_fds",
            .inventory = local_inventory,
        },
        {
            .name = "connection_retirement_destroys_ucx_endpoints_after_registry_unlock",
            .inventory = local_inventory,
        },
        {
            .name = "cross_owner_notification_completion_keeps_shared_owner_alive_last",
            .inventory = unequal.channel,
        },
        {
            .name = "exact_zero_terminal_lifecycle_inventory",
            .inventory = local_inventory,
        },
        {
            .name = "native_transfer_deadline_is_post_anchored_and_never_reset",
            .inventory = local_inventory,
        },
    };
}

void
writeJsonString(std::ostream &output, std::string_view value) {
    output << '"';
    for (const unsigned char byte : value) {
        switch (byte) {
        case '"':
            output << "\\\"";
            break;
        case '\\':
            output << "\\\\";
            break;
        case '\b':
            output << "\\b";
            break;
        case '\f':
            output << "\\f";
            break;
        case '\n':
            output << "\\n";
            break;
        case '\r':
            output << "\\r";
            break;
        case '\t':
            output << "\\t";
            break;
        default:
            if (byte < 0x20U) {
                constexpr char digits[] = "0123456789abcdef";
                output << "\\u00" << digits[(byte >> 4U) & 0x0fU] << digits[byte & 0x0fU];
            } else {
                output << static_cast<char>(byte);
            }
        }
    }
    output << '"';
}

void
writeInventory(std::ostream &output, const tcp_peer_channel_observation_t &inventory) {
    output << "{\n"
           << "        \"capacity\": " << inventory.capacity << ",\n"
           << "        \"queued_channel_events\": " << inventory.queuedChannelEvents << ",\n"
           << "        \"active_channel_subscriptions\": " << inventory.activeChannelSubscriptions
           << ",\n"
           << "        \"retained_public_subscriptions\": " << inventory.retainedPublicSubscriptions
           << ",\n"
           << "        \"backend_producers\": " << inventory.backendProducers << ",\n"
           << "        \"active_callback_slots\": " << inventory.activeCallbackSlots << ",\n"
           << "        \"queued_owner_continuations\": " << inventory.queuedOwnerContinuations
           << ",\n"
           << "        \"accepting_subscriptions\": "
           << (inventory.acceptingSubscriptions ? "true" : "false") << ",\n"
           << "        \"closed\": " << (inventory.closed ? "true" : "false") << ",\n"
           << "        \"fatal\": \"" << (inventory.fatal == 0 ? "NONE" : "UNKNOWN") << "\",\n"
           << "        \"eventfd_error\": " << inventory.eventfdError << '\n'
           << "      }";
}

void
writeReceipt(const std::filesystem::path &path, const std::vector<case_result_t> &cases) {
    require(!cases.empty(), "owner-lifecycle case inventory is empty");
    for (const case_result_t &result : cases) {
        requireCleanInventory(result.inventory, result.name);
    }
    const tcp_peer_channel_observation_t shutdown = closedInventory();
    requireCleanInventory(shutdown, "aggregate shutdown");

    std::ofstream output(path, std::ios::out | std::ios::binary);
    require(output.is_open(), "could not create owner-lifecycle receipt");
    output << "{\n  \"schema\": \"nixl-terminal-ucx-owner-lifecycle/v1\",\n"
           << "  \"status\": \"pass\",\n  \"cases\": [\n";
    for (std::size_t index = 0; index < cases.size(); ++index) {
        const case_result_t &result = cases[index];
        output << "    {\n      \"name\": ";
        writeJsonString(output, result.name);
        output << ",\n      \"status\": \"pass\",\n      \"inventory\": ";
        writeInventory(output, result.inventory);
        output << "\n    }" << (index + 1 == cases.size() ? "\n" : ",\n");
    }
    output << "  ],\n  \"shutdown\": ";
    writeInventory(output, shutdown);
    output << "\n}\n";
    output.close();
    require(output.good(), "could not finalize owner-lifecycle receipt");
}

int
run(int argc, char **argv) {
    const options_t options = parseOptions(argc, argv);
    const std::vector<case_result_t> cases = runCases();
    writeReceipt(options.output, cases);
    std::cout << "terminal UCX owner lifecycle qualification passed\n";
    return EXIT_SUCCESS;
}

} // namespace

int
main(int argc, char **argv) {
    if (nixl::qualification::isTerminalUcxPeerWorkerInvocation(argc, argv)) {
        return nixl::qualification::runTerminalUcxPeerWorker(argc, argv);
    }
    try {
        return run(argc, argv);
    }
    catch (const std::exception &error) {
        std::cerr << "terminal UCX owner lifecycle qualification failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
