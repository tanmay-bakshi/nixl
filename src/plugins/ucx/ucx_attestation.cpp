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

#include "ucx_attestation.h"

#include <algorithm>
#include <array>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <tuple>

#include <openssl/evp.h>

namespace {

void
appendUint64(std::string &buffer, uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8) {
        buffer.push_back(static_cast<char>((value >> shift) & 0xff));
    }
}

void
appendString(std::string &buffer, const std::string &value) {
    appendUint64(buffer, value.size());
    buffer.append(value);
}

[[nodiscard]] nixl_status_t
sha256(const std::string &input, std::string &digest) {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
        EVP_MD_CTX_new(), &EVP_MD_CTX_free);
    if (context == nullptr) {
        return NIXL_ERR_BACKEND;
    }

    if (EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(context.get(), input.data(), input.size()) != 1) {
        return NIXL_ERR_BACKEND;
    }

    std::array<unsigned char, EVP_MAX_MD_SIZE> bytes{};
    unsigned int size = 0;
    if (EVP_DigestFinal_ex(context.get(), bytes.data(), &size) != 1) {
        return NIXL_ERR_BACKEND;
    }

    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < size; ++i) {
        output << std::setw(2) << static_cast<unsigned int>(bytes[i]);
    }
    digest = output.str();
    return NIXL_SUCCESS;
}

[[nodiscard]] nixl_status_t
makeDescriptorDigest(const nixl_xfer_attestation_t &attestation, std::string &digest) {
    std::string input("nixl-descriptor-attestation-v1", 30);
    appendUint64(input, attestation.handleIdentity);
    appendString(input, attestation.backend);
    appendString(input, attestation.localAgent);
    appendString(input, attestation.remoteAgent);
    appendUint64(input, static_cast<uint64_t>(attestation.operation));
    appendUint64(input, static_cast<uint64_t>(attestation.localMemoryType));
    appendUint64(input, static_cast<uint64_t>(attestation.remoteMemoryType));
    appendUint64(input, attestation.segments.size());

    for (const auto &segment : attestation.segments) {
        appendUint64(input, segment.index);
        appendUint64(input, segment.localAddress);
        appendUint64(input, segment.remoteAddress);
        appendUint64(input, segment.localDeviceId);
        appendUint64(input, segment.remoteDeviceId);
        appendUint64(input, segment.length);
    }

    return sha256(input, digest);
}

[[nodiscard]] nixl_status_t
makeEvidenceDigest(const nixl_xfer_attestation_t &attestation, std::string &digest) {
    std::string input("nixl-transfer-evidence-v1", 25);
    appendUint64(input, attestation.handleIdentity);
    appendUint64(input, attestation.generation);
    appendString(input, attestation.descriptorDigest);
    appendUint64(input, attestation.segments.size());

    for (const auto &segment : attestation.segments) {
        appendUint64(input, segment.index);
        appendUint64(input, segment.workerId);
        appendUint64(input, segment.workerIdentity);
        appendUint64(input, segment.endpointIdentity);
        appendString(input, segment.requestInfo);
    }

    auto endpoints = attestation.endpoints;
    std::sort(endpoints.begin(), endpoints.end(), [](const auto &left, const auto &right) {
        return std::tie(left.workerIdentity, left.endpointIdentity) <
            std::tie(right.workerIdentity, right.endpointIdentity);
    });
    appendUint64(input, endpoints.size());
    for (auto &endpoint : endpoints) {
        std::sort(endpoint.segmentIndices.begin(), endpoint.segmentIndices.end());
        std::sort(endpoint.transports.begin(),
                  endpoint.transports.end(),
                  [](const auto &left, const auto &right) {
                      return std::tie(left.transport, left.device) <
                          std::tie(right.transport, right.device);
                  });
        appendUint64(input, endpoint.workerId);
        appendUint64(input, endpoint.workerIdentity);
        appendUint64(input, endpoint.endpointIdentity);
        appendUint64(input, endpoint.segmentIndices.size());
        for (size_t index : endpoint.segmentIndices) {
            appendUint64(input, index);
        }
        appendUint64(input, endpoint.transports.size());
        for (const auto &transport : endpoint.transports) {
            appendString(input, transport.transport);
            appendString(input, transport.device);
        }
        appendUint64(input, endpoint.flushPosted ? 1 : 0);
        appendUint64(input, endpoint.remoteFlushed ? 1 : 0);
    }

    auto runtime_artifacts = attestation.runtimeArtifacts;
    std::sort(runtime_artifacts.begin(),
              runtime_artifacts.end(),
              [](const auto &left, const auto &right) {
                  return std::tie(left.component, left.path, left.buildId, left.version) <
                      std::tie(right.component, right.path, right.buildId, right.version);
              });
    appendUint64(input, runtime_artifacts.size());
    for (const auto &artifact : runtime_artifacts) {
        appendString(input, artifact.component);
        appendString(input, artifact.path);
        appendString(input, artifact.buildId);
        appendString(input, artifact.version);
    }

    return sha256(input, digest);
}

} // namespace

nixlUcxAttestationState::nixlUcxAttestationState(uint64_t handle_identity) {
    attestation_.handleIdentity = handle_identity;
    attestation_.backend = "UCX";
}

nixl_status_t
nixlUcxAttestationState::prepare(nixl_xfer_op_t operation,
                                 const nixl_meta_dlist_t &local,
                                 const nixl_meta_dlist_t &remote,
                                 const std::string &local_agent,
                                 const std::string &remote_agent) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (local.descCount() == 0 || local.descCount() != remote.descCount()) {
        return failLocked(NIXL_ERR_INVALID_PARAM, "invalid descriptor count");
    }

    attestation_.localAgent = local_agent;
    attestation_.remoteAgent = remote_agent;
    attestation_.operation = operation;
    attestation_.localMemoryType = local.getType();
    attestation_.remoteMemoryType = remote.getType();
    attestation_.segments.clear();
    attestation_.segments.reserve(local.descCount());

    for (size_t i = 0; i < static_cast<size_t>(local.descCount()); ++i) {
        if (local[i].len != remote[i].len) {
            return failLocked(NIXL_ERR_MISMATCH, "local and remote segment lengths differ");
        }

        attestation_.segments.push_back({
            .index = i,
            .localAddress = local[i].addr,
            .remoteAddress = remote[i].addr,
            .localDeviceId = local[i].devId,
            .remoteDeviceId = remote[i].devId,
            .length = local[i].len,
        });
    }

    nixl_status_t status = makeDescriptorDigest(attestation_, attestation_.descriptorDigest);
    if (status != NIXL_SUCCESS) {
        return failLocked(status, "failed to digest transfer descriptors");
    }

    attestation_.generation = 0;
    attestation_.state = nixl_xfer_attestation_state_t::PREPARED;
    attestation_.status = NIXL_ERR_NOT_POSTED;
    attestation_.submissionSealed = false;
    attestation_.completionClaimed = false;
    attestation_.endpoints.clear();
    attestation_.evidenceDigest.clear();
    attestation_.error.clear();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxAttestationState::beginSubmission() {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (attestation_.state == nixl_xfer_attestation_state_t::FAILED) {
        return attestation_.status;
    }
    if (attestation_.state == nixl_xfer_attestation_state_t::POSTING ||
        attestation_.state == nixl_xfer_attestation_state_t::IN_PROGRESS) {
        return NIXL_ERR_REPOST_ACTIVE;
    }
    if (attestation_.state == nixl_xfer_attestation_state_t::REMOTE_FLUSHED &&
        !attestation_.completionClaimed) {
        return NIXL_ERR_NOT_ALLOWED;
    }
    if (attestation_.state != nixl_xfer_attestation_state_t::PREPARED &&
        attestation_.state != nixl_xfer_attestation_state_t::REMOTE_FLUSHED) {
        return failLocked(NIXL_ERR_BACKEND, "invalid state before submission");
    }
    if (attestation_.generation == std::numeric_limits<uint64_t>::max()) {
        return failLocked(NIXL_ERR_NOT_ALLOWED, "submission generation exhausted");
    }

    ++attestation_.generation;
    attestation_.state = nixl_xfer_attestation_state_t::POSTING;
    attestation_.status = NIXL_IN_PROG;
    attestation_.submissionSealed = false;
    attestation_.completionClaimed = false;
    attestation_.endpoints.clear();
    attestation_.evidenceDigest.clear();
    attestation_.error.clear();
    for (auto &segment : attestation_.segments) {
        segment.workerId = 0;
        segment.workerIdentity = 0;
        segment.endpointIdentity = 0;
        segment.requestInfo.clear();
        segment.posted = false;
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxAttestationState::recordSegment(
    size_t index,
    size_t worker_id,
    uint64_t worker_identity,
    uint64_t endpoint_identity,
    const std::vector<nixl_xfer_attestation_transport_t> &transports,
    const std::string &request_info) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (attestation_.state == nixl_xfer_attestation_state_t::FAILED) {
        return attestation_.status;
    }
    if (attestation_.state != nixl_xfer_attestation_state_t::POSTING ||
        index >= attestation_.segments.size() || request_info.empty() || transports.empty()) {
        return failLocked(NIXL_ERR_BACKEND, "invalid segment evidence");
    }

    auto &segment = attestation_.segments[index];
    if (segment.posted) {
        return failLocked(NIXL_ERR_BACKEND, "segment evidence was recorded more than once");
    }

    auto canonical_transports = transports;
    std::sort(canonical_transports.begin(),
              canonical_transports.end(),
              [](const auto &left, const auto &right) {
                  return std::tie(left.transport, left.device) <
                      std::tie(right.transport, right.device);
              });

    segment.workerId = worker_id;
    segment.workerIdentity = worker_identity;
    segment.endpointIdentity = endpoint_identity;
    segment.requestInfo = request_info;
    segment.posted = true;

    auto *endpoint = findEndpointLocked(endpoint_identity);
    if (endpoint == nullptr) {
        attestation_.endpoints.push_back({
            .workerId = worker_id,
            .workerIdentity = worker_identity,
            .endpointIdentity = endpoint_identity,
            .segmentIndices = {index},
            .transports = std::move(canonical_transports),
        });
        return NIXL_SUCCESS;
    }

    if (endpoint->workerId != worker_id || endpoint->workerIdentity != worker_identity ||
        endpoint->transports != canonical_transports) {
        return failLocked(NIXL_ERR_BACKEND, "endpoint context changed during submission");
    }
    endpoint->segmentIndices.push_back(index);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxAttestationState::recordFlush(size_t worker_id,
                                     uint64_t worker_identity,
                                     uint64_t endpoint_identity,
                                     nixl_status_t status) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (attestation_.state == nixl_xfer_attestation_state_t::FAILED) {
        return attestation_.status;
    }
    if (attestation_.state != nixl_xfer_attestation_state_t::POSTING) {
        return failLocked(NIXL_ERR_BACKEND, "endpoint flush was recorded outside posting");
    }

    auto *endpoint = findEndpointLocked(endpoint_identity);
    if (endpoint == nullptr || endpoint->workerId != worker_id ||
        endpoint->workerIdentity != worker_identity || endpoint->flushPosted) {
        return failLocked(NIXL_ERR_BACKEND, "invalid endpoint flush evidence");
    }

    if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
        return failLocked(status, "endpoint flush post failed");
    }

    endpoint->flushPosted = true;
    endpoint->remoteFlushed = status == NIXL_SUCCESS;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxAttestationState::finishSubmission() {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (attestation_.state == nixl_xfer_attestation_state_t::FAILED) {
        return attestation_.status;
    }
    if (attestation_.state != nixl_xfer_attestation_state_t::POSTING ||
        attestation_.endpoints.empty()) {
        return failLocked(NIXL_ERR_BACKEND, "submission did not produce endpoint evidence");
    }

    const bool all_segments_posted =
        std::all_of(attestation_.segments.begin(), attestation_.segments.end(),
                    [](const auto &segment) { return segment.posted; });
    const bool all_flushes_posted =
        std::all_of(attestation_.endpoints.begin(), attestation_.endpoints.end(),
                    [](const auto &endpoint) { return endpoint.flushPosted; });
    if (!all_segments_posted || !all_flushes_posted) {
        return failLocked(NIXL_ERR_BACKEND, "submission evidence is incomplete");
    }

    attestation_.submissionSealed = true;
    if (allRemoteFlushedLocked()) {
        return sealCompletionLocked();
    }

    attestation_.state = nixl_xfer_attestation_state_t::IN_PROGRESS;
    attestation_.status = NIXL_IN_PROG;
    return NIXL_SUCCESS;
}

void
nixlUcxAttestationState::completeFlush(uint64_t generation, uint64_t endpoint_identity) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (attestation_.state == nixl_xfer_attestation_state_t::FAILED) {
        return;
    }
    if (generation != attestation_.generation) {
        (void)failLocked(NIXL_ERR_BACKEND, "stale endpoint flush completion");
        return;
    }
    if (attestation_.state != nixl_xfer_attestation_state_t::POSTING &&
        attestation_.state != nixl_xfer_attestation_state_t::IN_PROGRESS) {
        (void)failLocked(NIXL_ERR_BACKEND, "endpoint flush completed outside an active submission");
        return;
    }

    auto *endpoint = findEndpointLocked(endpoint_identity);
    if (endpoint == nullptr || !endpoint->flushPosted) {
        (void)failLocked(NIXL_ERR_BACKEND, "unknown endpoint flush completion");
        return;
    }
    if (endpoint->remoteFlushed) {
        (void)failLocked(NIXL_ERR_BACKEND, "endpoint flush completed more than once");
        return;
    }

    endpoint->remoteFlushed = true;
    if (attestation_.submissionSealed && allRemoteFlushedLocked()) {
        (void)sealCompletionLocked();
    }
}

void
nixlUcxAttestationState::fail(uint64_t generation,
                              nixl_status_t status,
                              const std::string &error) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (attestation_.state == nixl_xfer_attestation_state_t::FAILED) {
        return;
    }
    if (generation != attestation_.generation) {
        (void)failLocked(NIXL_ERR_BACKEND, "stale submission failure");
        return;
    }
    (void)failLocked(status, error);
}

uint64_t
nixlUcxAttestationState::getGeneration() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return attestation_.generation;
}

nixl_xfer_attestation_t
nixlUcxAttestationState::snapshot() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    nixl_xfer_attestation_t result = attestation_;
    std::sort(result.endpoints.begin(), result.endpoints.end(), [](const auto &left, const auto &right) {
        return std::tie(left.workerIdentity, left.endpointIdentity) <
            std::tie(right.workerIdentity, right.endpointIdentity);
    });
    for (auto &endpoint : result.endpoints) {
        std::sort(endpoint.segmentIndices.begin(), endpoint.segmentIndices.end());
        std::sort(endpoint.transports.begin(),
                  endpoint.transports.end(),
                  [](const auto &left, const auto &right) {
                      return std::tie(left.transport, left.device) <
                          std::tie(right.transport, right.device);
                  });
    }
    std::sort(result.runtimeArtifacts.begin(),
              result.runtimeArtifacts.end(),
              [](const auto &left, const auto &right) {
                  return std::tie(left.component, left.path, left.buildId, left.version) <
                      std::tie(right.component, right.path, right.buildId, right.version);
              });
    return result;
}

nixl_status_t
nixlUcxAttestationState::takeCompletion(nixl_xfer_attestation_t &attestation) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (attestation_.state != nixl_xfer_attestation_state_t::REMOTE_FLUSHED ||
        attestation_.status != NIXL_SUCCESS || !attestation_.submissionSealed ||
        attestation_.completionClaimed || attestation_.descriptorDigest.empty() ||
        attestation_.evidenceDigest.empty()) {
        return NIXL_ERR_NOT_ALLOWED;
    }

    attestation_.completionClaimed = true;
    attestation = attestation_;
    return NIXL_SUCCESS;
}

nixl_xfer_attestation_endpoint_t *
nixlUcxAttestationState::findEndpointLocked(uint64_t endpoint_identity) {
    auto endpoint = std::find_if(
        attestation_.endpoints.begin(), attestation_.endpoints.end(),
        [endpoint_identity](const auto &candidate) {
            return candidate.endpointIdentity == endpoint_identity;
        });
    return endpoint == attestation_.endpoints.end() ? nullptr : &*endpoint;
}

bool
nixlUcxAttestationState::allRemoteFlushedLocked() const {
    return std::all_of(attestation_.endpoints.begin(), attestation_.endpoints.end(),
                       [](const auto &endpoint) { return endpoint.remoteFlushed; });
}

nixl_status_t
nixlUcxAttestationState::failLocked(nixl_status_t status, const std::string &error) {
    if (attestation_.state == nixl_xfer_attestation_state_t::FAILED) {
        return attestation_.status;
    }
    attestation_.state = nixl_xfer_attestation_state_t::FAILED;
    attestation_.status = status < 0 ? status : NIXL_ERR_BACKEND;
    attestation_.evidenceDigest.clear();
    attestation_.error = error;
    return attestation_.status;
}

nixl_status_t
nixlUcxAttestationState::sealCompletionLocked() {
    if (attestation_.state == nixl_xfer_attestation_state_t::FAILED) {
        return attestation_.status;
    }
    if (!attestation_.submissionSealed || attestation_.endpoints.empty() ||
        !allRemoteFlushedLocked()) {
        return failLocked(NIXL_ERR_BACKEND, "completion evidence is incomplete");
    }
    attestation_.state = nixl_xfer_attestation_state_t::REMOTE_FLUSHED;
    attestation_.status = NIXL_SUCCESS;
    nixl_status_t status = makeEvidenceDigest(attestation_, attestation_.evidenceDigest);
    if (status != NIXL_SUCCESS) {
        return failLocked(status, "failed to digest transfer evidence");
    }
    return NIXL_SUCCESS;
}
