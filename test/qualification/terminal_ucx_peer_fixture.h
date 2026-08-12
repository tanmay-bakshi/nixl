/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_TEST_QUALIFICATION_TERMINAL_UCX_PEER_FIXTURE_H
#define NIXL_TEST_QUALIFICATION_TERMINAL_UCX_PEER_FIXTURE_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "nixl.h"

namespace nixl::qualification {

struct tcp_peer_terminal_observation_t {
    nixl_status_t terminalStatus = NIXL_ERR_BACKEND;
    std::size_t terminalEventCount = 0;
    bool ownerWoken = false;
};

struct tcp_peer_channel_observation_t {
    std::size_t capacity = 0;
    std::size_t queuedChannelEvents = 0;
    std::size_t activeChannelSubscriptions = 0;
    std::size_t retainedPublicSubscriptions = 0;
    std::size_t backendProducers = 0;
    std::size_t activeCallbackSlots = 0;
    std::size_t queuedOwnerContinuations = 0;
    bool acceptingSubscriptions = false;
    bool closed = false;
    std::uint32_t fatal = 0;
    int eventfdError = 0;
};

struct tcp_peer_capability_observation_t {
    std::uint64_t handleIdentity = 0;
    std::uint64_t handleGeneration = 0;
    std::vector<nixl_terminal_capability_state_t> states;
    std::vector<std::uint64_t> epochs;
    nixl_status_t releaseStatus = NIXL_ERR_BACKEND;
    bool subscriptionTerminal = false;
};

struct tcp_endpoint_failure_observation_t {
    tcp_peer_terminal_observation_t transfer;
    tcp_peer_capability_observation_t capability;
    tcp_peer_channel_observation_t channel;
    bool peerExitedBySignal = false;
};

struct tcp_notification_failure_observation_t {
    tcp_peer_terminal_observation_t transfer;
    tcp_peer_channel_observation_t channel;
    bool dataRemoteFlushedBeforeFailure = false;
    bool peerExitedBySignal = false;
};

[[nodiscard]] bool
isTerminalUcxPeerWorkerInvocation(int argc, char **argv) noexcept;

[[nodiscard]] int
runTerminalUcxPeerWorker(int argc, char **argv);

[[nodiscard]] tcp_endpoint_failure_observation_t
runTcpEndpointFailureFixture(const std::string &engine);

[[nodiscard]] tcp_notification_failure_observation_t
runTcpNotificationFailureFixture(const std::string &engine);

} // namespace nixl::qualification

#endif // NIXL_TEST_QUALIFICATION_TERMINAL_UCX_PEER_FIXTURE_H
