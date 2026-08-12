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

void
require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

[[nodiscard]] std::string
parseEngine(int argc, char **argv) {
    require(argc == 3 && std::string_view(argv[1]) == "--engine",
            "usage: terminal_ucx_endpoint_failure_qualification --engine shared|thread_pool");
    const std::string engine(argv[2]);
    require(engine == "shared" || engine == "thread_pool", "invalid progress engine");
    return engine;
}

void
validate(const nixl::qualification::tcp_endpoint_failure_observation_t &observation) {
    const auto &transfer = observation.transfer;
    const auto &capability = observation.capability;
    const auto &channel = observation.channel;
    require(observation.peerExitedBySignal, "independent TCP peer did not exit through SIGKILL");
    require(transfer.terminalStatus == NIXL_ERR_REMOTE_DISCONNECT &&
                transfer.terminalEventCount == 1 && transfer.ownerWoken,
            "endpoint failure did not publish one exact transfer terminal event");
    require(capability.handleIdentity != 0 && capability.handleGeneration != 0 &&
                capability.states.size() == 2 && capability.epochs.size() == 2 &&
                capability.states[0] == nixl_terminal_capability_state_t::READY &&
                capability.states[1] == nixl_terminal_capability_state_t::FAILED &&
                capability.epochs[1] >= capability.epochs[0] &&
                capability.releaseStatus == NIXL_SUCCESS && capability.subscriptionTerminal,
            "endpoint failure did not publish READY followed by one FAILED route transition");
    require(channel.queuedChannelEvents == 0 && channel.activeChannelSubscriptions == 0 &&
                channel.retainedPublicSubscriptions == 0 && channel.backendProducers == 0 &&
                channel.activeCallbackSlots == 0 && channel.queuedOwnerContinuations == 0 &&
                !channel.acceptingSubscriptions && channel.closed && channel.fatal == 0 &&
                channel.eventfdError == 0,
            "endpoint failure teardown retained native or public lifecycle inventory");
}

} // namespace

int
main(int argc, char **argv) {
    if (nixl::qualification::isTerminalUcxPeerWorkerInvocation(argc, argv)) {
        return nixl::qualification::runTerminalUcxPeerWorker(argc, argv);
    }
    try {
        const std::string engine = parseEngine(argc, argv);
        const auto observation = nixl::qualification::runTcpEndpointFailureFixture(engine);
        validate(observation);
        std::cout << "engine=" << engine
                  << " transfer_status=" << observation.transfer.terminalStatus
                  << " capability_transitions=" << observation.capability.states.size()
                  << " backend_producers=" << observation.channel.backendProducers
                  << " active_callback_slots=" << observation.channel.activeCallbackSlots
                  << " queued_owner_continuations=" << observation.channel.queuedOwnerContinuations
                  << '\n';
    }
    catch (const std::exception &error) {
        std::cerr << "terminal UCX endpoint failure qualification failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
