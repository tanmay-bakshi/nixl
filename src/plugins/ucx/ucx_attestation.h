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
#ifndef NIXL_SRC_PLUGINS_UCX_UCX_ATTESTATION_H
#define NIXL_SRC_PLUGINS_UCX_UCX_ATTESTATION_H

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "backend/backend_aux.h"

struct nixlUcxTransportModuleRequirement {
    std::string component;
    std::string soname;
};

[[nodiscard]] nixl_status_t
nixlUcxGetTransportModuleRequirements(
    const std::vector<nixl_xfer_attestation_transport_t> &transports,
    std::vector<nixlUcxTransportModuleRequirement> &requirements,
    std::string &error);

[[nodiscard]] nixl_status_t
nixlUcxCanonicalizeLoadedPath(const std::string &component,
                              const std::string &loaded_path,
                              std::string &canonical_path,
                              std::string &error);

class nixlUcxAttestationState {
public:
    explicit nixlUcxAttestationState(uint64_t handle_identity);

    [[nodiscard]] nixl_status_t
    prepare(nixl_xfer_op_t operation,
            const nixl_meta_dlist_t &local,
            const nixl_meta_dlist_t &remote,
            const std::string &local_agent,
            const std::string &remote_agent);

    [[nodiscard]] nixl_status_t
    beginSubmission();

    [[nodiscard]] nixl_status_t
    recordSegment(size_t index,
                  size_t worker_id,
                  uint64_t worker_identity,
                  uint64_t endpoint_identity,
                  const std::vector<nixl_xfer_attestation_transport_t> &transports,
                  const std::string &request_info);

    [[nodiscard]] nixl_status_t
    recordFlush(size_t worker_id,
                uint64_t worker_identity,
                uint64_t endpoint_identity,
                nixl_status_t status);

    [[nodiscard]] nixl_status_t
    finishSubmission();

    void
    completeFlush(uint64_t generation, uint64_t endpoint_identity);

    void
    fail(uint64_t generation, nixl_status_t status, const std::string &error);

    [[nodiscard]] uint64_t
    getGeneration() const;

    [[nodiscard]] nixl_xfer_attestation_t
    snapshot() const;

    [[nodiscard]] nixl_status_t
    takeCompletion(nixl_xfer_attestation_t &attestation);

private:
    [[nodiscard]] nixl_xfer_attestation_endpoint_t *
    findEndpointLocked(uint64_t endpoint_identity);

    [[nodiscard]] bool
    allRemoteFlushedLocked() const;

    [[nodiscard]] nixl_status_t
    failLocked(nixl_status_t status, const std::string &error);

    [[nodiscard]] nixl_status_t
    sealCompletionLocked();

    mutable std::mutex mutex_;
    std::vector<nixl_runtime_artifact_t> baseRuntimeArtifacts_;
    nixl_xfer_attestation_t attestation_;
};

#endif
