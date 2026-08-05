/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NIXL_SRC_CORE_AGENT_DATA_H
#define NIXL_SRC_CORE_AGENT_DATA_H

#include "mem_section.h"
#include "telemetry.h"
#include "stream/metadata_stream.h"
#include "sync.h"

#include <memory>
#include <unordered_set>

#if HAVE_ETCD
#include <etcd/SyncClient.hpp>

namespace etcd {
class SyncClient;
}

#define NIXL_ETCD_NAMESPACE_DEFAULT "/nixl/agents/"
#endif // HAVE_ETCD

class nixlBackendEngine;

class nixlRemoteAgentH {
private:
    nixlRemoteAgentH(uint64_t owner_identity,
                     uint64_t identity,
                     uint64_t generation,
                     std::string name,
                     std::string agent_incarnation)
        : ownerIdentity_(owner_identity),
          identity_(identity),
          generation_(generation),
          name_(std::move(name)),
          agentIncarnation_(std::move(agent_incarnation)) {}

    const uint64_t ownerIdentity_;
    const uint64_t identity_;
    const uint64_t generation_;
    const std::string name_;
    const std::string agentIncarnation_;
    bool active_ = true;
    std::unordered_map<nixlBackendEngine *, nixlRemoteAgentBinding> bindings_;

public:
    [[nodiscard]] const std::string &
    getName() const noexcept {
        return name_;
    }

    [[nodiscard]] uint64_t getIdentity() const noexcept { return identity_; }
    [[nodiscard]] uint64_t getGeneration() const noexcept { return generation_; }

    friend class nixlAgent;
    friend class nixlAgentData;
};

using backend_list_t = std::vector<nixlBackendEngine*>;

//Internal typedef to define metadata communication request types
//To be extended with ETCD operations
enum nixl_comm_t {
    SOCK_SEND,
    SOCK_FETCH,
    SOCK_INVAL,
    SOCK_MAX,
#if HAVE_ETCD
    ETCD_SEND,
    ETCD_FETCH,
    ETCD_INVAL
#endif // HAVE_ETCD
};

//Command to be sent to listener thread from NIXL API
// 1) Command type
// 2) IP Address
// 3) Port
// 4) Metadata to send (for sendLocalMD calls)
using nixl_comm_req_t = std::tuple<nixl_comm_t, std::string, int, nixl_blob_t>;

using nixl_socket_peer_t = std::pair<std::string, int>;

using nixl_socket_map_t = std::map<nixl_socket_peer_t, int>;

class nixlAgentData {
    private:
        const std::string name_;
        const uint64_t identity_;
        const std::string incarnation_;
        const nixlAgentConfig config_;
        const bool useEtcd_;
        const bool needsCommThread_;
        nixlLock        lock;
        bool telemetryEnabled = false;
        bool efaWarningChecked = false;

        // some handle that can be used to instantiate an object from the lib
        std::map<std::string, void*> backendLibs;

        // Bookkeeping from backend type and memory type to backend engine
        backend_list_t                         notifEngines;
        std::array<backend_list_t, FILE_SEG+1> memToBackend;

        // Bookkeeping from memory view handles to backend engines
        std::unordered_map<nixlMemViewH, nixlBackendEngine &> mvhToEngine;

        std::unordered_map<std::string, std::unordered_map<nixl_backend_t, nixl_blob_t>>
            remoteBackends_;
        std::unordered_map<std::string, nixlRemoteAgentH *> activeRemoteHandles_;
        std::unordered_map<std::string, uint64_t> remoteGenerations_;
        std::unordered_set<std::string> quarantinedRemoteNames_;
        std::unordered_set<const nixlRemoteAgentH *> ownedRemoteHandles_;
        std::unordered_map<uint64_t, nixlRemoteAgentH *> remoteHandlesByIdentity_;
        std::vector<std::unique_ptr<nixlRemoteAgentH>> remoteHandles_;


        // State/methods for listener thread
        std::unique_ptr<nixlMDStreamListener> listener;
        nixl_socket_map_t remoteSockets;
        std::thread commThread;
        std::vector<nixl_comm_req_t> commQueue;
        std::mutex commLock;
        std::atomic<bool> commThreadStop;
        std::atomic<bool> agentShutdown;
        std::exception_ptr commThreadException_;

        // The order of the following data members is crucial for destruction.
        // Bookkeeping for local connection metadata and user handles per backend
        std::unordered_map<nixl_backend_t, std::unique_ptr<nixlBackendH>> backendHandles_;
        std::unordered_map<nixl_backend_t, nixl_blob_t> connMd_;
        backend_map_t backendEngines_;
        std::unordered_map<std::string, nixlRemoteSection> remoteSections_;
        std::unique_ptr<nixlTelemetry> telemetry_;
        nixlLocalSection localSection_;

        void
        commWorker(nixlAgent &myAgent) noexcept;
        void
        commWorkerInternal(nixlAgent *myAgent);
        void enqueueCommWork(nixl_comm_req_t request);
        void getCommWork(std::vector<nixl_comm_req_t> &req_list);
        [[nodiscard]] nixl_status_t
        prepareRemoteSections(const std::string &remote_name,
                              nixlSerDes &sd,
                              nixlRemoteSectionUpdate &update) const;
        nixl_status_t
        applyRemoteSections(const std::string &remote_name,
                            nixlRemoteSectionUpdate &&update,
                            bool &rollback_ambiguous);
        nixl_status_t
        invalidateRemoteData(const std::string &remote_name);
        [[nodiscard]] nixl_status_t
        validateRemoteHandle(const nixlRemoteAgentH *handle) const;
        [[nodiscard]] nixlRemoteAgentH *
        findActiveRemoteHandle(const std::string &remote_name) const;
        [[nodiscard]] nixlRemoteAgentH *
        findOwnedRemoteHandle(uint64_t identity, uint64_t generation) const;
        void
        deactivateRemoteHandle(const std::string &remote_name);

        [[nodiscard]] static backend_set_t
        getBackends(const nixl_opt_args_t *opt_args,
                    const nixlMemSection &section,
                    nixl_mem_t mem_type);
        void
        warnAboutEfaHardwareMismatch();

    public:
        nixlAgentData(const std::string &name, const nixlAgentConfig &config);

        void
        addErrorTelemetry(nixl_status_t err_status) {
            if (telemetry_) {
                telemetry_->updateErrorCount(err_status);
            }
        }

    friend class nixlAgent;
};

class nixlBackendEngine;

// This class hides away the nixlBackendEngine from user of the Agent API
class nixlBackendH {
    private:
        nixlBackendEngine* engine;

        explicit nixlBackendH(nixlBackendEngine *engine) noexcept : engine(engine) {}

    public:
        ~nixlBackendH() = default;

        // TODO? engine->getType() returns a const nixl_backend_t&
        nixl_backend_t
        getType() const noexcept {
            return engine->getType();
        }

        bool
        supportsRemote() const {
            return engine->supportsRemote();
        }

        bool
        supportsLocal() const {
            return engine->supportsLocal();
        }

        bool
        supportsNotif() const {
            return engine->supportsNotif();
        }

    friend class nixlAgentData;
    friend class nixlAgent;
};

#endif
