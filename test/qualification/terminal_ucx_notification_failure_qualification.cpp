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

[[nodiscard]] std::string
parseEngine(int argc, char **argv) {
    std::string engine;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--engine" && index + 1 < argc) {
            engine = argv[++index];
            continue;
        }
        throw std::runtime_error("unknown or incomplete argument");
    }
    if (engine != "shared" && engine != "thread_pool") {
        throw std::runtime_error("engine must be shared or thread_pool");
    }
    return engine;
}

void
requireCleanChannel(const nixl::qualification::tcp_peer_channel_observation_t &channel) {
    const bool clean = channel.queuedChannelEvents == 0 &&
        channel.activeChannelSubscriptions == 0 && channel.retainedPublicSubscriptions == 0 &&
        channel.backendProducers == 0 && channel.activeCallbackSlots == 0 &&
        channel.queuedOwnerContinuations == 0 && !channel.acceptingSubscriptions &&
        channel.closed && channel.fatal == 0 && channel.eventfdError == 0;
    if (!clean) {
        throw std::runtime_error("notification fixture retained terminal-channel state");
    }
}

int
run(int argc, char **argv) {
    const std::string engine = parseEngine(argc, argv);
    const nixl::qualification::tcp_notification_failure_observation_t observation =
        nixl::qualification::runTcpNotificationFailureFixture(engine);
    if (observation.transfer.terminalStatus != NIXL_ERR_REMOTE_DISCONNECT ||
        observation.transfer.terminalEventCount != 1 || !observation.transfer.ownerWoken ||
        !observation.dataRemoteFlushedBeforeFailure || !observation.peerExitedBySignal) {
        throw std::runtime_error("notification fixture did not prove its terminal boundary");
    }
    requireCleanChannel(observation.channel);
    std::cout << "terminal UCX notification failure passed for " << engine << '\n';
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
        std::cerr << "terminal UCX notification-failure qualification failed: " << error.what()
                  << '\n';
        return EXIT_FAILURE;
    }
}
