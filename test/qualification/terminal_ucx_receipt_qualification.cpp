/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "terminal_ucx_peer_fixture.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

struct options_t {
    std::string testCase;
    std::string engine;
};

[[nodiscard]] options_t
parseOptions(int argc, char **argv) {
    options_t options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--case" && index + 1 < argc) {
            options.testCase = argv[++index];
            continue;
        }
        if (argument == "--engine" && index + 1 < argc) {
            options.engine = argv[++index];
            continue;
        }
        throw std::runtime_error("unknown or incomplete argument");
    }
    if (options.testCase != "admission-release" && options.testCase != "admission-peer-death" &&
        options.testCase != "4-to-2" && options.testCase != "2-to-4") {
        throw std::runtime_error(
            "case must be admission-release, admission-peer-death, 4-to-2, or 2-to-4");
    }
    if ((options.testCase == "admission-release" || options.testCase == "admission-peer-death") &&
        options.engine != "shared" && options.engine != "thread_pool") {
        throw std::runtime_error("admission-receipt cases require a shared or thread_pool engine");
    }
    return options;
}

void
requireCleanChannel(const nixl::qualification::tcp_peer_channel_observation_t &channel) {
    const bool clean = channel.queuedChannelEvents == 0 &&
        channel.activeChannelSubscriptions == 0 && channel.retainedPublicSubscriptions == 0 &&
        channel.backendProducers == 0 && channel.activeCallbackSlots == 0 &&
        channel.queuedOwnerContinuations == 0 && !channel.acceptingSubscriptions &&
        channel.closed && channel.fatal == 0 && channel.eventfdError == 0;
    if (!clean) {
        throw std::runtime_error("receipt qualification retained terminal-channel state");
    }
}

void
validateUnequalWorkers(const nixl::qualification::tcp_unequal_worker_observation_t &observation,
                       std::size_t source_workers,
                       std::size_t destination_workers) {
    bool exercised_every_source = observation.exercisedSourceWorkers.size() == source_workers;
    for (std::size_t worker_id = 0; exercised_every_source && worker_id < source_workers;
         ++worker_id) {
        exercised_every_source = observation.exercisedSourceWorkers[worker_id] == worker_id;
    }
    if (observation.sourceWorkerCount != source_workers ||
        observation.destinationWorkerCount != destination_workers ||
        observation.completedTransferCount != source_workers ||
        observation.notificationCount != source_workers || !exercised_every_source ||
        !observation.peerExitedCleanly) {
        throw std::runtime_error("unequal-worker routing did not exercise its declared mapping");
    }
    requireCleanChannel(observation.channel);
}

void
validateAdmissionRelease(
    const nixl::qualification::tcp_admission_receipt_observation_t &observation) {
    if (observation.transfer.terminalStatus != NIXL_SUCCESS ||
        observation.transfer.terminalEventCount != 1 || !observation.transfer.ownerWoken ||
        observation.stateWhileHeld != nixl_xfer_attestation_state_t::REMOTE_FLUSHED ||
        observation.heldSourceHandleIdentity == 0 || observation.heldSourceGeneration == 0 ||
        observation.heldDeliveryIdentity == 0 || observation.notificationCount != 1 ||
        !observation.subscriptionActiveWhileHeld || !observation.remoteFlushedWhileHeld ||
        !observation.peerExitedCleanly || observation.peerExitedBySignal) {
        throw std::runtime_error("admission release did not preserve terminal authority");
    }
    requireCleanChannel(observation.channel);
}

void
validateAdmissionPeerDeath(
    const nixl::qualification::tcp_admission_receipt_observation_t &observation) {
    if (observation.transfer.terminalStatus != NIXL_ERR_REMOTE_DISCONNECT ||
        observation.transfer.terminalEventCount != 1 || !observation.transfer.ownerWoken ||
        observation.stateWhileHeld != nixl_xfer_attestation_state_t::REMOTE_FLUSHED ||
        observation.heldSourceHandleIdentity == 0 || observation.heldSourceGeneration == 0 ||
        observation.heldDeliveryIdentity == 0 || !observation.subscriptionActiveWhileHeld ||
        !observation.remoteFlushedWhileHeld || observation.peerExitedCleanly ||
        !observation.peerExitedBySignal) {
        throw std::runtime_error("admission peer death did not fail exact source authority");
    }
    requireCleanChannel(observation.channel);
}

int
run(int argc, char **argv) {
    const options_t options = parseOptions(argc, argv);
    if (options.testCase == "admission-release") {
        validateAdmissionRelease(
            nixl::qualification::runTcpAdmissionReceiptReleaseFixture(options.engine));
    } else if (options.testCase == "admission-peer-death") {
        validateAdmissionPeerDeath(
            nixl::qualification::runTcpAdmissionReceiptPeerDeathFixture(options.engine));
    } else if (options.testCase == "4-to-2") {
        validateUnequalWorkers(nixl::qualification::runTcpUnequalWorkerFixture(4, 2), 4, 2);
    } else {
        validateUnequalWorkers(nixl::qualification::runTcpUnequalWorkerFixture(2, 4), 2, 4);
    }
    std::cout << "terminal UCX receipt qualification passed for " << options.testCase << '\n';
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
        std::cerr << "terminal UCX receipt qualification failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
