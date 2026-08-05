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

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "backend/backend_engine.h"
#include "ucx_attestation.h"
#include "ucx_backend.h"

namespace {

void
require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

nixl_meta_dlist_t
makeDescriptors(uintptr_t base, size_t count, size_t length) {
    nixl_meta_dlist_t descriptors(DRAM_SEG);
    for (size_t index = 0; index < count; ++index) {
        descriptors.addDesc(
            nixlMetaDesc(base + index * length, length, index, nullptr));
    }
    return descriptors;
}

std::unique_ptr<nixlUcxAttestationState>
makeState(size_t count = 2) {
    require(nixlEnumStrings::statusStr(NIXL_SUCCESS) == "NIXL_SUCCESS",
            "libnixl status-string symbol is unavailable");
    auto state = std::make_unique<nixlUcxAttestationState>(41);
    const nixl_meta_dlist_t local = makeDescriptors(0x1000, count, 64);
    const nixl_meta_dlist_t remote = makeDescriptors(0x8000, count, 64);
    const nixl_remote_agent_authority_t authority = {
        .handleIdentity = 51,
        .generation = 7,
        .connectionIdentity = 61,
        .endpointIdentities = {200, 201, 202, 203},
    };
    require(state->prepare(
                NIXL_WRITE, local, remote, "prefill", "decode", &authority) == NIXL_SUCCESS,
            "failed to prepare attestation state");
    return state;
}

void
testTransportModuleRequirements() {
    const std::vector<nixl_xfer_attestation_transport_t> selected_transports = {
        {.transport = "posix", .device = "memory"},
        {.transport = "self", .device = "memory"},
        {.transport = "sysv", .device = "memory"},
        {.transport = "tcp", .device = "eth0"},
        {.transport = "cuda_copy", .device = "cuda"},
        {.transport = "cuda_ipc", .device = "cuda"},
        {.transport = "gdr_copy", .device = "cuda"},
        {.transport = "dc_mlx5", .device = "mlx5_0:1"},
        {.transport = "gga_mlx5", .device = "mlx5_0:1"},
        {.transport = "rc_mlx5", .device = "mlx5_0:1"},
        {.transport = "ud_mlx5", .device = "mlx5_0:1"},
        {.transport = "rc_gda", .device = "mlx5_0:1"},
        {.transport = "rc_verbs", .device = "mlx5_0:1"},
        {.transport = "ud_verbs", .device = "mlx5_0:1"},
        {.transport = "srd", .device = "efa0"},
        {.transport = "cma", .device = "memory"},
        {.transport = "knem", .device = "memory"},
        {.transport = "xpmem", .device = "memory"},
    };
    const std::set<std::pair<std::string, std::string>> expected = {
        {"libuct_cma", "libuct_cma.so"},
        {"libuct_cuda", "libuct_cuda.so"},
        {"libuct_cuda_gdrcopy", "libuct_cuda_gdrcopy.so"},
        {"libuct_ib", "libuct_ib.so"},
        {"libuct_ib_efa", "libuct_ib_efa.so"},
        {"libuct_ib_mlx5", "libuct_ib_mlx5.so"},
        {"libuct_ib_mlx5_gda", "libuct_ib_mlx5_gda.so"},
        {"libuct_knem", "libuct_knem.so"},
        {"libuct_xpmem", "libuct_xpmem.so"},
    };

    std::vector<nixlUcxTransportModuleRequirement> requirements;
    std::string error;
    require(nixlUcxGetTransportModuleRequirements(
                selected_transports, requirements, error) == NIXL_SUCCESS,
            "known transport module requirements were rejected");
    std::set<std::pair<std::string, std::string>> actual;
    for (const auto &requirement : requirements) {
        actual.emplace(requirement.component, requirement.soname);
    }
    require(actual == expected, "transport module requirements changed");
    require(error.empty(), "successful transport module requirements returned an error");

    for (const auto &unsupported :
         std::vector<nixl_xfer_attestation_transport_t>{
             {.transport = "rocm_copy", .device = "rocm"},
             {.transport = "foo_mlx5", .device = "mlx5_0:1"},
         }) {
        require(nixlUcxGetTransportModuleRequirements(
                    {unsupported}, requirements, error) == NIXL_ERR_BACKEND,
                "unknown transport module requirement did not fail closed");
        require(requirements.empty(),
                "failed transport module requirements leaked evidence");
        require(!error.empty(),
                "failed transport module requirements omitted an error");
    }
}

void
testStructuredSelectedTransportsAreCanonicalAndBoundToEndpoint() {
    const std::vector<nixl_xfer_attestation_transport_t> endpoint_transports = {
        {.transport = "tcp", .device = "lo"},
        {.transport = "rc_mlx5", .device = "mlx5_0:1"},
        {.transport = "cuda_ipc", .device = "cuda"},
        {.transport = "cuda_ipc", .device = "cuda"},
    };
    const std::vector<nixl_xfer_attestation_transport_t> expected = {
        {.transport = "cuda_ipc", .device = "cuda"},
        {.transport = "rc_mlx5", .device = "mlx5_0:1"},
    };
    const std::vector<nixl_xfer_attestation_transport_t> selected_transports = {
        {.transport = "rc_mlx5", .device = "mlx5_0:1"},
        {.transport = "cuda_ipc", .device = "cuda"},
        {.transport = "cuda_ipc", .device = "cuda"},
    };

    auto state = makeState(1);
    require(state->beginSubmission() == NIXL_SUCCESS, "submission did not begin");
    const std::string contradictory_request_info =
        "{proto|init} diagnostic claims tcp/lo";
    require(state->recordSegment(0,
                                 0,
                                 101,
                                 201,
                                 endpoint_transports,
                                 selected_transports,
                                 contradictory_request_info) == NIXL_SUCCESS,
            "structured selected resources were rejected");
    const nixl_xfer_attestation_t snapshot = state->snapshot();
    require(snapshot.segments[0].selectedTransports == expected,
            "selected resources are not canonical and unique");
    require(snapshot.segments[0].requestInfo == contradictory_request_info,
            "diagnostic request information was not retained verbatim");
    require(snapshot.endpoints[0].transports ==
                std::vector<nixl_xfer_attestation_transport_t>({
                    {.transport = "cuda_ipc", .device = "cuda"},
                    {.transport = "rc_mlx5", .device = "mlx5_0:1"},
                    {.transport = "tcp", .device = "lo"},
                }),
            "endpoint resources are not canonical and unique");

    auto wrong_device_state = makeState(1);
    require(wrong_device_state->beginSubmission() == NIXL_SUCCESS,
            "wrong-device submission did not begin");
    require(
        wrong_device_state->recordSegment(
            0,
            0,
            101,
            201,
            endpoint_transports,
            {{.transport = "cuda_ipc", .device = "forged"}},
            "diagnostic claims cuda_ipc/cuda") == NIXL_ERR_BACKEND,
        "selected resource with a forged device was accepted");

    const std::vector<nixl_xfer_attestation_transport_t> incomplete_context = {
        {.transport = "cuda_ipc", .device = ""},
    };
    auto incomplete_state = makeState(1);
    require(incomplete_state->beginSubmission() == NIXL_SUCCESS,
            "incomplete-context submission did not begin");
    require(incomplete_state->recordSegment(
                0,
                0,
                101,
                201,
                incomplete_context,
                {{.transport = "cuda_ipc", .device = "cuda"}},
                "") == NIXL_ERR_BACKEND,
            "incomplete endpoint resource context was accepted");

    auto empty_selected_state = makeState(1);
    require(empty_selected_state->beginSubmission() == NIXL_SUCCESS,
            "empty-selection submission did not begin");
    require(empty_selected_state->recordSegment(
                0, 0, 101, 201, endpoint_transports, {}, "") == NIXL_ERR_BACKEND,
            "empty selected-resource evidence was accepted");
}

void
testLoadedArtifactPathsAreLoadTimeAbsolute() {
    std::string canonical_path = "stale";
    std::string error;
    require(nixlUcxCanonicalizeLoadedPath(
                "libuct_cuda", "libuct_cuda.so.0", canonical_path, error) ==
                NIXL_ERR_BACKEND,
            "relative loaded artifact path was accepted");
    require(canonical_path.empty(), "relative loaded artifact path leaked stale output");
    require(!error.empty(), "relative loaded artifact path omitted an error");

    require(nixlUcxCanonicalizeLoadedPath(
                "test-executable", "/proc/self/exe", canonical_path, error) ==
                NIXL_SUCCESS,
            "absolute loaded artifact path was rejected");
    require(std::filesystem::path(canonical_path).is_absolute(),
            "canonical loaded artifact path is not absolute");
    require(error.empty(), "canonical loaded artifact path returned an error");
}

void
testRuntimeArtifactsIdentifyLoadedObjects() {
    const auto state = makeState();
    const nixl_xfer_attestation_t snapshot = state->snapshot();
    require(snapshot.runtimeArtifacts.size() == 3,
            "runtime identity did not cover every evidence interpreter");

    std::set<std::string> components;
    for (const auto &artifact : snapshot.runtimeArtifacts) {
        components.insert(artifact.component);
        require(!artifact.path.empty() && std::filesystem::path(artifact.path).is_absolute(),
                "runtime artifact path is not absolute");

        std::error_code path_error;
        const std::filesystem::path canonical_path =
            std::filesystem::canonical(artifact.path, path_error);
        require(!path_error && canonical_path == artifact.path,
                "runtime artifact path is not canonical");
        require(!artifact.version.empty(), "runtime artifact version is empty");
        require(!artifact.buildId.empty() && artifact.buildId.size() % 2 == 0,
                "runtime artifact build ID is empty or malformed");
        const bool build_id_is_hex =
            std::all_of(artifact.buildId.begin(), artifact.buildId.end(), [](char value) {
                return std::isxdigit(static_cast<unsigned char>(value)) != 0;
            });
        require(build_id_is_hex, "runtime artifact build ID is not hexadecimal");
    }

    require(components == std::set<std::string>({"libnixl", "libucp", "ucx-plugin"}),
            "runtime identity components changed");
}

const std::vector<nixl_xfer_attestation_transport_t> transports = {
    {.transport = "self", .device = "memory"},
    {.transport = "tcp", .device = "eth0"},
};
const std::vector<nixl_xfer_attestation_transport_t> selected_transports = {
    {.transport = "tcp", .device = "eth0"},
};

class UnsupportedBackend final : public nixlBackendEngine {
public:
    explicit UnsupportedBackend(const nixlBackendInitParams &params)
        : nixlBackendEngine(&params) {}

    bool
    supportsRemote() const override {
        return false;
    }

    bool
    supportsLocal() const override {
        return false;
    }

    bool
    supportsNotif() const override {
        return false;
    }

    nixl_mem_list_t
    getSupportedMems() const override {
        return {};
    }

    nixl_status_t
    registerMem(const nixlBlobDesc &, const nixl_mem_t &, nixlBackendMD *&) override {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    nixl_status_t
    deregisterMem(nixlBackendMD *) override {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    nixl_status_t
    connect(const std::string &) override {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    nixl_status_t
    disconnect(const std::string &) override {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    nixl_status_t
    unloadMD(nixlBackendMD *) override {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    nixl_status_t
    prepXfer(const nixl_xfer_op_t &,
             const nixl_meta_dlist_t &,
             const nixl_meta_dlist_t &,
             const std::string &,
             nixlBackendReqH *&,
             const nixl_opt_b_args_t *) const override {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    nixl_status_t
    postXfer(const nixl_xfer_op_t &,
             const nixl_meta_dlist_t &,
             const nixl_meta_dlist_t &,
             const std::string &,
             nixlBackendReqH *&,
             const nixl_opt_b_args_t *) const override {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    nixl_status_t
    checkXfer(nixlBackendReqH *) const override {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    nixl_status_t
    releaseReqH(nixlBackendReqH *) const override {
        return NIXL_ERR_NOT_SUPPORTED;
    }
};

void
testUnsupportedBackendFailsClosed() {
    nixl_b_params_t custom_params;
    nixlBackendInitParams params = {
        .localAgent = "local",
        .type = "unsupported",
        .customParams = &custom_params,
        .enableProgTh = false,
        .pthrDelay = 0,
        .syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE,
        .enableTelemetry_ = false,
    };
    UnsupportedBackend backend(params);
    nixl_xfer_attestation_t attestation;
    require(backend.queryXferAttestation(nullptr, attestation) == NIXL_ERR_NOT_SUPPORTED,
            "unsupported query did not fail closed");
    require(backend.takeXferCompletionAttestation(nullptr, attestation) ==
                NIXL_ERR_NOT_SUPPORTED,
            "unsupported completion take did not fail closed");
}

void
testAuthenticatedNotificationQueueIsSingularAndBounded() {
    nixlUcxNotificationQueue queue(2, 64);
    require(queue.push({.remoteAgent = "prefill", .payload = "legacy"}) == NIXL_SUCCESS,
            "notification queue rejected a valid legacy-drain item");

    notif_list_t legacy;
    require(queue.drainLegacy(legacy) == NIXL_SUCCESS,
            "legacy notification drain failed");
    require(legacy == notif_list_t({{"prefill", "legacy"}}),
            "legacy notification drain changed ownership or payload");

    authenticated_notif_list_t authenticated;
    require(queue.drainAuthenticated(authenticated) == NIXL_SUCCESS,
            "authenticated drain after legacy drain failed");
    require(authenticated.empty(),
            "legacy drain left a duplicate authenticated shadow queue");

    require(queue.push({.remoteAgent = "prefill",
                        .payload = "authenticated",
                        .connectionIdentity = 7,
                        .endpointIdentity = 11}) == NIXL_SUCCESS,
            "notification queue rejected a valid authenticated-drain item");
    require(queue.drainAuthenticated(authenticated) == NIXL_SUCCESS,
            "authenticated notification drain failed");
    require(authenticated.size() == 1 && authenticated.front().remoteAgent == "prefill" &&
                authenticated.front().payload == "authenticated" &&
                authenticated.front().connectionIdentity == 7 &&
                authenticated.front().endpointIdentity == 11,
            "authenticated notification evidence changed during drain");
    legacy.clear();
    require(queue.drainLegacy(legacy) == NIXL_SUCCESS,
            "legacy drain after authenticated drain failed");
    require(legacy.empty(),
            "authenticated drain left a duplicate legacy shadow queue");

    authenticated.clear();
    require(queue.push({.remoteAgent = "prefill", .payload = "x"}) == NIXL_SUCCESS,
            "notification queue failed before its declared count bound");
    require(queue.push({.remoteAgent = "prefill", .payload = "y"}) == NIXL_SUCCESS,
            "notification queue failed at its declared count bound");
    require(queue.push({.remoteAgent = "prefill", .payload = "overflow"}) ==
                NIXL_ERR_NOT_ALLOWED,
            "notification queue accepted an item beyond its declared count bound");
    require(queue.drainAuthenticated(authenticated) == NIXL_ERR_NOT_ALLOWED,
            "overflowed notification queue did not fail closed");
    require(authenticated.empty(),
            "overflowed notification queue exposed partially trusted messages");
    legacy.clear();
    require(queue.drainLegacy(legacy) == NIXL_ERR_NOT_ALLOWED,
            "overflowed notification queue did not poison the legacy drain");
    require(legacy.empty(),
            "overflowed notification queue exposed legacy messages");
    require(queue.push({.remoteAgent = "prefill", .payload = "after-failure"}) ==
                NIXL_ERR_NOT_ALLOWED,
            "overflowed notification queue recovered without re-establishing authority");

    nixlUcxNotificationQueue byte_bounded_queue(8, 8);
    require(byte_bounded_queue.push({.remoteAgent = "name", .payload = "data"}) ==
                NIXL_SUCCESS,
            "notification queue rejected an item at its exact byte bound");
    require(byte_bounded_queue.push({.remoteAgent = "n", .payload = ""}) ==
                NIXL_ERR_NOT_ALLOWED,
            "notification queue ignored remote-agent bytes at its byte bound");
    authenticated.clear();
    require(byte_bounded_queue.drainAuthenticated(authenticated) == NIXL_ERR_NOT_ALLOWED,
            "byte-overflowed notification queue did not fail closed");
    require(authenticated.empty(),
            "byte-overflowed notification queue exposed partially trusted messages");
}

void
testFirstFailureIsPermanent() {
    auto state = makeState();
    require(state->beginSubmission() == NIXL_SUCCESS, "submission did not begin");
    const uint64_t generation = state->getGeneration();
    state->fail(generation, NIXL_ERR_MISMATCH, "first failure");
    state->fail(generation, NIXL_ERR_CANCELED, "later failure");
    state->fail(generation - 1, NIXL_ERR_BACKEND, "stale failure");

    const nixl_xfer_attestation_t snapshot = state->snapshot();
    require(snapshot.state == nixl_xfer_attestation_state_t::FAILED,
            "failed state was not retained");
    require(snapshot.status == NIXL_ERR_MISMATCH, "first failure status was overwritten");
    require(snapshot.error == "first failure", "first failure detail was overwritten");
    require(state->beginSubmission() == NIXL_ERR_MISMATCH,
            "failed handle accepted a repost");
}

void
postImmediateCompletion(nixlUcxAttestationState &state,
                        uint64_t worker_identity,
                        uint64_t endpoint_identity) {
    require(
        state.recordSegment(
            0,
            0,
            worker_identity,
            endpoint_identity,
            transports,
            selected_transports,
            "segment-0 protocol tcp/eth0") ==
            NIXL_SUCCESS,
        "first segment evidence was rejected");
    require(
        state.recordSegment(
            1,
            0,
            worker_identity,
            endpoint_identity,
            transports,
            selected_transports,
            "segment-1 protocol tcp/eth0") ==
            NIXL_SUCCESS,
        "second segment evidence was rejected");
    require(state.recordFlush(0, worker_identity, endpoint_identity, NIXL_SUCCESS) == NIXL_SUCCESS,
            "immediate endpoint flush was rejected");
    require(state.finishSubmission() == NIXL_SUCCESS, "submission did not seal");
}

void
testTakeOnceAndGenerationRollover() {
    auto state = makeState();
    require(state->beginSubmission() == NIXL_SUCCESS, "first submission did not begin");
    postImmediateCompletion(*state, 101, 201);

    const nixl_xfer_attestation_t before_take = state->snapshot();
    require(before_take.state == nixl_xfer_attestation_state_t::REMOTE_FLUSHED,
            "first generation did not remotely flush");
    require(std::is_sorted(
                before_take.runtimeArtifacts.begin(),
                before_take.runtimeArtifacts.end(),
                [](const auto &left, const auto &right) {
                    return left.component < right.component;
                }),
            "sealed runtime artifacts are not canonical");
    require(state->beginSubmission() == NIXL_ERR_NOT_ALLOWED,
            "repost before take was accepted");

    nixl_xfer_attestation_t first;
    require(state->takeCompletion(first) == NIXL_SUCCESS,
            "first completion evidence was not claimable");
    require(first.completionClaimed, "taken completion was not marked claimed");
    require(state->takeCompletion(first) == NIXL_ERR_NOT_ALLOWED,
            "completion evidence was claimable twice");

    require(state->beginSubmission() == NIXL_SUCCESS,
            "claimed handle did not accept a new generation");
    require(state->getGeneration() == first.generation + 1,
            "submission generation did not advance");
    postImmediateCompletion(*state, 102, 202);

    nixl_xfer_attestation_t second;
    require(state->takeCompletion(second) == NIXL_SUCCESS,
            "second completion evidence was not claimable");
    require(second.descriptorDigest == first.descriptorDigest,
            "descriptor binding changed across handle reuse");
    require(second.evidenceDigest != first.evidenceDigest,
            "generation was not bound into transfer evidence");
}

void
testStaleEventsPoisonCurrentGeneration() {
    {
        auto state = makeState();
        require(state->beginSubmission() == NIXL_SUCCESS, "first submission did not begin");
        postImmediateCompletion(*state, 101, 201);
        nixl_xfer_attestation_t completion;
        require(state->takeCompletion(completion) == NIXL_SUCCESS,
                "first completion was not claimable");
        require(state->beginSubmission() == NIXL_SUCCESS, "second submission did not begin");
        state->completeFlush(completion.generation, 201);
        const nixl_xfer_attestation_t snapshot = state->snapshot();
        require(snapshot.state == nixl_xfer_attestation_state_t::FAILED,
                "stale completion did not fail the current generation");
        require(snapshot.generation == completion.generation + 1,
                "stale completion changed the reused generation");
        require(snapshot.endpoints.empty(),
                "stale completion injected endpoint state into the reused generation");
        require(std::none_of(snapshot.segments.begin(),
                             snapshot.segments.end(),
                             [](const auto &segment) { return segment.posted; }),
                "stale completion mutated segment state in the reused generation");
        require(snapshot.evidenceDigest.empty(),
                "stale completion left successful evidence");
    }

    {
        auto state = makeState();
        require(state->beginSubmission() == NIXL_SUCCESS, "first submission did not begin");
        postImmediateCompletion(*state, 101, 201);
        nixl_xfer_attestation_t completion;
        require(state->takeCompletion(completion) == NIXL_SUCCESS,
                "first completion was not claimable");
        require(state->beginSubmission() == NIXL_SUCCESS, "second submission did not begin");
        state->fail(completion.generation, NIXL_ERR_CANCELED, "old failure");
        const nixl_xfer_attestation_t snapshot = state->snapshot();
        require(snapshot.state == nixl_xfer_attestation_state_t::FAILED,
                "stale failure did not fail the current generation");
        require(snapshot.generation == completion.generation + 1,
                "stale failure changed the reused generation");
        require(snapshot.endpoints.empty(),
                "stale failure injected endpoint state into the reused generation");
        require(snapshot.error == "stale submission failure",
                "stale failure was not distinguished");
    }
}

void
testExactSegmentsAndAllFlushesBindCompletion() {
    auto state = makeState(4);
    require(state->beginSubmission() == NIXL_SUCCESS, "submission did not begin");

    require(state->recordSegment(
                0, 0, 101, 201, transports, selected_transports, "segment-0 protocol tcp/eth0") ==
                NIXL_SUCCESS,
            "segment 0 evidence was rejected");
    require(state->recordSegment(
                1, 0, 101, 201, transports, selected_transports, "segment-1 protocol tcp/eth0") ==
                NIXL_SUCCESS,
            "segment 1 evidence was rejected");
    require(state->recordSegment(
                2, 1, 102, 202, transports, selected_transports, "segment-2 protocol tcp/eth0") ==
                NIXL_SUCCESS,
            "segment 2 evidence was rejected");
    require(state->recordSegment(
                3, 1, 102, 202, transports, selected_transports, "segment-3 protocol tcp/eth0") ==
                NIXL_SUCCESS,
            "segment 3 evidence was rejected");
    require(state->recordFlush(0, 101, 201, NIXL_IN_PROG) == NIXL_SUCCESS,
            "first endpoint flush was rejected");
    require(state->recordFlush(1, 102, 202, NIXL_IN_PROG) == NIXL_SUCCESS,
            "second endpoint flush was rejected");
    require(state->finishSubmission() == NIXL_SUCCESS, "submission did not seal");

    nixl_xfer_attestation_t completion;
    require(state->takeCompletion(completion) == NIXL_ERR_NOT_ALLOWED,
            "completion was available before endpoint flushes");
    state->completeFlush(state->getGeneration(), 201);
    require(state->snapshot().state == nixl_xfer_attestation_state_t::IN_PROGRESS,
            "one endpoint flush completed the whole submission");
    require(state->takeCompletion(completion) == NIXL_ERR_NOT_ALLOWED,
            "completion was available with one endpoint outstanding");
    state->completeFlush(state->getGeneration(), 202);
    require(state->takeCompletion(completion) == NIXL_SUCCESS,
            "completion was unavailable after every endpoint flushed");

    require(completion.segments.size() == 4, "completion lost segment evidence");
    require(completion.endpoints.size() == 2, "completion lost endpoint evidence");
    require(completion.endpoints[0].segmentIndices == std::vector<size_t>({0, 1}),
            "first endpoint segment binding changed");
    require(completion.endpoints[1].segmentIndices == std::vector<size_t>({2, 3}),
            "second endpoint segment binding changed");
    for (size_t index = 0; index < completion.segments.size(); ++index) {
        require(completion.segments[index].index == index, "segment order changed");
        require(completion.segments[index].posted, "segment was not marked posted");
        require(completion.segments[index].selectedTransports ==
                    std::vector<nixl_xfer_attestation_transport_t>({
                        {.transport = "tcp", .device = "eth0"},
                    }),
                "segment selected-resource evidence changed");
    }
}

void
testCompositePostingSealsOnlyAfterBarrier() {
    constexpr size_t chunk_count = 4;
    auto state = makeState(chunk_count);
    require(state->beginSubmission() == NIXL_SUCCESS, "submission did not begin");

    std::array<std::thread, chunk_count> threads;
    std::array<nixl_status_t, chunk_count> segment_statuses;
    std::array<nixl_status_t, chunk_count> flush_statuses;
    for (size_t index = 0; index < chunk_count; ++index) {
        threads[index] = std::thread([&state, &segment_statuses, &flush_statuses, index]() {
            const uint64_t worker_identity = 100 + index;
            const uint64_t endpoint_identity = 200 + index;
            segment_statuses[index] =
                state->recordSegment(index,
                                     index,
                                     worker_identity,
                                     endpoint_identity,
                                     transports,
                                     selected_transports,
                                     "chunk protocol " + std::to_string(index) + " tcp/eth0");
            flush_statuses[index] =
                state->recordFlush(index, worker_identity, endpoint_identity, NIXL_SUCCESS);
        });
    }
    for (auto &thread : threads) {
        thread.join();
    }
    for (size_t index = 0; index < chunk_count; ++index) {
        require(segment_statuses[index] == NIXL_SUCCESS,
                "chunk segment evidence was rejected");
        require(flush_statuses[index] == NIXL_SUCCESS,
                "chunk flush evidence was rejected");
    }

    nixl_xfer_attestation_t completion;
    const nixl_xfer_attestation_t before_seal = state->snapshot();
    require(before_seal.state == nixl_xfer_attestation_state_t::POSTING,
            "composite submission completed before the posting barrier");
    require(!before_seal.submissionSealed,
            "composite submission sealed before the posting barrier");
    require(state->takeCompletion(completion) == NIXL_ERR_NOT_ALLOWED,
            "unsealed composite evidence was claimable");

    require(state->finishSubmission() == NIXL_SUCCESS,
            "composite submission did not seal after the barrier");
    require(state->takeCompletion(completion) == NIXL_SUCCESS,
            "sealed composite completion was not claimable");
    require(completion.endpoints.size() == chunk_count,
            "composite completion lost chunk endpoints");
}

} // namespace

int
main() {
    try {
        testUnsupportedBackendFailsClosed();
        testAuthenticatedNotificationQueueIsSingularAndBounded();
        testTransportModuleRequirements();
        testStructuredSelectedTransportsAreCanonicalAndBoundToEndpoint();
        testLoadedArtifactPathsAreLoadTimeAbsolute();
        testRuntimeArtifactsIdentifyLoadedObjects();
        testFirstFailureIsPermanent();
        testTakeOnceAndGenerationRollover();
        testStaleEventsPoisonCurrentGeneration();
        testExactSegmentsAndAllFlushesBindCompletion();
        testCompositePostingSealsOnlyAfterBarrier();
    }
    catch (const std::exception &error) {
        std::cerr << "UCX attestation test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "UCX attestation state-machine tests passed\n";
    return EXIT_SUCCESS;
}
