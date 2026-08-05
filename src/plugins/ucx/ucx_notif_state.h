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
#ifndef NIXL_SRC_PLUGINS_UCX_UCX_NOTIF_STATE_H
#define NIXL_SRC_PLUGINS_UCX_UCX_NOTIF_STATE_H

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ucx_notif_wire.h"

namespace nixl::ucx {

struct notif_route_key_t {
    std::uint64_t handleIdentity = 0;
    std::uint64_t handleGeneration = 0;
    notif_wire_uuid_t remoteAgentIncarnation;
    notif_wire_uuid_t remoteBackendIncarnation;

    bool
    operator==(const notif_route_key_t &) const = default;
};

struct notif_remote_worker_t {
    notif_wire_uuid_t incarnation;
    std::uint64_t endpointIdentity = 0;

    bool
    operator==(const notif_remote_worker_t &) const = default;
};

struct notif_remote_binding_t {
    notif_route_key_t route;
    std::uint64_t connectionIdentity = 0;
    std::vector<notif_remote_worker_t> workers;

    bool
    operator==(const notif_remote_binding_t &) const = default;
};

enum class notif_state_status_t {
    SUCCESS,
    INVALID_ARGUMENT,
    UNKNOWN_ROUTE,
    UNKNOWN_CAPABILITY,
    ROUTE_CONFLICT,
    ROUTE_RETIRED,
    ROUTE_FAILED,
    NOT_READY,
    STALE_EPOCH,
    EPOCH_CONFLICT,
    RANDOM_FAILURE,
    EPOCH_EXHAUSTED,
};

enum class notif_route_state_t {
    NOT_READY,
    READY,
    RETIRED,
    FAILED,
};

struct notif_route_snapshot_t {
    notif_remote_binding_t binding;
    notif_route_state_t state = notif_route_state_t::NOT_READY;
    notif_wire_uuid_t localCapability;
    std::uint64_t localCapabilityEpoch = 0;
    bool localOfferAcknowledged = false;
    bool hasRemoteCapability = false;
    notif_wire_uuid_t remoteCapability;
    std::uint64_t remoteCapabilityEpoch = 0;
};

struct notif_offer_acceptance_t {
    notif_route_key_t route;
    notif_wire_envelope_t acknowledgement;
    notif_wire_envelope_t localOffer;
    bool reemitLocalOffer = false;
};

enum class notif_data_disposition_t {
    DELIVER,
    DROP_ROUTE,
    POISON_GLOBAL,
};

struct notif_data_authority_t {
    notif_route_key_t route;
    std::uint64_t connectionIdentity = 0;
    notif_wire_uuid_t senderWorkerIncarnation;
    std::uint64_t endpointIdentity = 0;
    bool tombstoned = false;
};

struct notif_data_resolution_t {
    notif_data_disposition_t disposition = notif_data_disposition_t::POISON_GLOBAL;
    notif_state_status_t status = notif_state_status_t::UNKNOWN_CAPABILITY;
    notif_data_authority_t authority;
};

class notif_capability_state_t {
public:
    notif_capability_state_t(const notif_wire_uuid_t &local_agent_incarnation,
                             const notif_wire_uuid_t &local_backend_incarnation,
                             std::vector<notif_wire_uuid_t> local_worker_incarnations);

    notif_capability_state_t(const notif_capability_state_t &) = delete;
    notif_capability_state_t &
    operator=(const notif_capability_state_t &) = delete;

    [[nodiscard]] notif_state_status_t
    bindRemoteAgent(const notif_remote_binding_t &binding, notif_route_snapshot_t &snapshot);

    [[nodiscard]] notif_state_status_t
    retireRemoteAgent(const notif_route_key_t &route);

    [[nodiscard]] notif_state_status_t
    queryRemoteNotificationState(const notif_route_key_t &route,
                                 notif_route_snapshot_t &snapshot) const;

    [[nodiscard]] notif_state_status_t
    makeOffer(const notif_route_key_t &route,
              const notif_wire_uuid_t &local_sender_worker,
              notif_wire_envelope_t &offer) const;

    [[nodiscard]] notif_state_status_t
    acceptOffer(const notif_wire_envelope_t &offer,
                const notif_wire_uuid_t &local_sender_worker,
                notif_offer_acceptance_t &acceptance);

    [[nodiscard]] notif_state_status_t
    acceptAcknowledgement(const notif_wire_envelope_t &acknowledgement, notif_route_key_t &route);

    [[nodiscard]] notif_state_status_t
    prepareData(const notif_route_key_t &route,
                const notif_wire_uuid_t &local_sender_worker,
                notif_wire_envelope_t &data) const;

    [[nodiscard]] notif_data_resolution_t
    resolveData(const notif_wire_envelope_t &data) const;

private:
    struct uuid_hash_t {
        [[nodiscard]] std::size_t
        operator()(const notif_wire_uuid_t &uuid) const noexcept;
    };

    struct route_hash_t {
        [[nodiscard]] std::size_t
        operator()(const notif_route_key_t &route) const noexcept;
    };

    struct peer_key_t {
        notif_wire_uuid_t agentIncarnation;
        notif_wire_uuid_t backendIncarnation;

        bool
        operator==(const peer_key_t &) const = default;
    };

    struct peer_hash_t {
        [[nodiscard]] std::size_t
        operator()(const peer_key_t &peer) const noexcept;
    };

    struct binding_state_t {
        notif_remote_binding_t binding;
        std::unordered_map<notif_wire_uuid_t, std::uint64_t, uuid_hash_t> remoteWorkers;
        notif_wire_uuid_t localCapability;
        std::uint64_t localCapabilityEpoch = 0;
        bool localOfferAcknowledged = false;
        bool retired = false;
        bool failed = false;
        bool hasRemoteCapability = false;
        notif_wire_uuid_t remoteCapability;
        std::uint64_t remoteCapabilityEpoch = 0;
    };

    [[nodiscard]] bool
    isLocalWorker(const notif_wire_uuid_t &worker) const;

    [[nodiscard]] bool
    hasExactHandleConflictLocked(const notif_route_key_t &route) const;

    [[nodiscard]] notif_state_status_t
    validateInboundLocked(const notif_wire_envelope_t &envelope,
                          const binding_state_t &binding,
                          std::uint64_t &endpoint_identity) const;

    [[nodiscard]] notif_wire_envelope_t
    makeEnvelopeLocked(notif_wire_type_t type,
                       const binding_state_t &binding,
                       const notif_wire_uuid_t &local_sender_worker,
                       const notif_wire_uuid_t &capability,
                       std::uint64_t capability_epoch) const;

    [[nodiscard]] notif_route_snapshot_t
    makeSnapshotLocked(const binding_state_t &binding) const;

    notif_wire_uuid_t localAgentIncarnation_;
    notif_wire_uuid_t localBackendIncarnation_;
    std::unordered_set<notif_wire_uuid_t, uuid_hash_t> localWorkers_;
    mutable std::mutex mutex_;
    std::uint64_t nextCapabilityEpoch_ = 1;
    std::unordered_map<notif_route_key_t, binding_state_t, route_hash_t> bindings_;
    std::unordered_map<peer_key_t, notif_route_key_t, peer_hash_t> activePeers_;
    std::unordered_map<notif_wire_uuid_t, notif_route_key_t, uuid_hash_t> localCapabilities_;
};

} // namespace nixl::ucx

#endif
