/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_TEST_QUALIFICATION_TERMINAL_UCX_PEER_FIXTURE_H
#define NIXL_TEST_QUALIFICATION_TERMINAL_UCX_PEER_FIXTURE_H

#include <array>
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
    bool dataBoundaryRemoteFlushed = false;
    bool peerExitedBySignal = false;
};

struct tcp_direct_owner_failure_observation_t {
    std::array<std::uint8_t, 32> expectedBinding{};
    std::array<std::uint8_t, 32> deliveredBinding{};
    std::uint64_t nativeTimestampNs = 0;
    std::uint16_t eventKind = 0;
    std::int32_t reasonCode = 0;
    std::int64_t backendStatus = 0;
    std::size_t terminalEventCount = 0;
    std::size_t activeCallbacksAfterTerminal = 0;
    std::size_t activeRegistrationsAfterTerminal = 0;
    std::size_t retainedBindingsAfterTerminal = 0;
    std::size_t successfulTerminalEvents = 0;
    std::size_t failureTerminalEvents = 0;
    nixl_status_t subscriptionReleaseStatus = NIXL_ERR_BACKEND;
    nixl_status_t retirementJoinStatus = NIXL_ERR_BACKEND;
    nixl_status_t closeStatus = NIXL_ERR_BACKEND;
    bool subscriptionTerminal = false;
    bool bindingExact = false;
    bool retirementRequested = false;
    bool joined = false;
    bool closed = false;
    bool peerExitedBySignal = false;
};

struct tcp_notification_failure_observation_t {
    tcp_peer_terminal_observation_t transfer;
    tcp_peer_channel_observation_t channel;
    bool dataRemoteFlushedBeforeFailure = false;
    bool notificationFailureAfterRemoteFlush = false;
    std::string faultPeerEngine;
    bool faultPeerAdmissionReceiptHeld = false;
    bool peerExitedBySignal = false;
};

struct tcp_shutdown_cancellation_observation_t {
    tcp_peer_terminal_observation_t transfer;
    tcp_peer_channel_observation_t inventoryBeforeCancellation;
    tcp_peer_channel_observation_t inventoryAfterClose;
    nixl_status_t cancelStatus = NIXL_ERR_BACKEND;
    bool postedInFlight = false;
    bool drained = false;
    bool peerExitedBySignal = false;
};

struct tcp_unequal_worker_observation_t {
    tcp_peer_channel_observation_t channel;
    std::size_t sourceWorkerCount = 0;
    std::size_t destinationWorkerCount = 0;
    std::vector<std::size_t> exercisedSourceWorkers;
    std::size_t completedTransferCount = 0;
    std::size_t notificationCount = 0;
    bool peerExitedCleanly = false;
};

struct tcp_admission_receipt_observation_t {
    tcp_peer_terminal_observation_t transfer;
    tcp_peer_channel_observation_t channel;
    nixl_xfer_attestation_state_t stateWhileHeld = nixl_xfer_attestation_state_t::PREPARED;
    std::uint64_t heldSourceHandleIdentity = 0;
    std::uint64_t heldSourceGeneration = 0;
    std::uint64_t heldDeliveryIdentity = 0;
    std::size_t notificationCount = 0;
    bool subscriptionActiveWhileHeld = false;
    bool remoteFlushedWhileHeld = false;
    bool peerExitedCleanly = false;
    bool peerExitedBySignal = false;
};

[[nodiscard]] bool
isTerminalUcxPeerWorkerInvocation(int argc, char **argv) noexcept;

[[nodiscard]] int
runTerminalUcxPeerWorker(int argc, char **argv);

[[nodiscard]] tcp_endpoint_failure_observation_t
runTcpEndpointFailureFixture(const std::string &engine);

[[nodiscard]] tcp_direct_owner_failure_observation_t
runTcpDirectOwnerEndpointFailureFixture(const std::string &engine);

[[nodiscard]] tcp_notification_failure_observation_t
runTcpNotificationFailureFixture(const std::string &engine);

[[nodiscard]] tcp_shutdown_cancellation_observation_t
runTcpShutdownCancellationFixture(const std::string &engine);

[[nodiscard]] tcp_unequal_worker_observation_t
runTcpUnequalWorkerFixture(std::size_t source_worker_count, std::size_t destination_worker_count);

[[nodiscard]] tcp_admission_receipt_observation_t
runTcpAdmissionReceiptReleaseFixture(const std::string &engine);

[[nodiscard]] tcp_admission_receipt_observation_t
runTcpAdmissionReceiptPeerDeathFixture(const std::string &engine);

} // namespace nixl::qualification

#endif // NIXL_TEST_QUALIFICATION_TERMINAL_UCX_PEER_FIXTURE_H
