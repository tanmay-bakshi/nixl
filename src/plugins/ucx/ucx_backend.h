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
#ifndef NIXL_SRC_PLUGINS_UCX_UCX_BACKEND_H
#define NIXL_SRC_PLUGINS_UCX_UCX_BACKEND_H

#include <vector>
#include <cstring>
#include <iostream>
#include <thread>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <poll.h>
#include <optional>

#include "nixl.h"

#include "backend/backend_engine.h"
#include "common/nixl_time.h"

#include "mem_list.h"
#include "rkey.h"
#include "ucx_attestation.h"
#include "ucx_connection_metadata.h"
#include "ucx_enums.h"
#include "ucx_notif_state.h"
#include "ucx_notif_wire.h"
#include "ucx_utils.h"

class nixlUcxNotificationQueue {
public:
    static constexpr size_t maxPendingNotifications = 65536;
    static constexpr size_t maxPendingBytes = 64 * 1024 * 1024;
    static constexpr size_t maxWireBytes = nixl::ucx::notif_wire_max_frame_size;

    explicit nixlUcxNotificationQueue(
        size_t max_notifications = maxPendingNotifications,
        size_t max_bytes = maxPendingBytes)
        : maxNotifications_(max_notifications), maxBytes_(max_bytes) {}

    [[nodiscard]] nixl_status_t
    push(nixlAuthenticatedNotification &&notification);
    [[nodiscard]] nixl_status_t
    drainLegacy(notif_list_t &notifications);
    [[nodiscard]] nixl_status_t
    drainAuthenticated(authenticated_notif_list_t &notifications);
    void
    poison();

private:
    const size_t maxNotifications_;
    const size_t maxBytes_;
    authenticated_notif_list_t notifications_;
    size_t queuedBytes_ = 0;
    bool failed_ = false;
};

class nixlUcxConnection : public nixlBackendConnMD {
    private:
        const uint64_t identity_;
        const nixl::ucx::connection_metadata_t metadata_;
        std::vector<std::unique_ptr<nixlUcxEp>> eps;
        std::atomic<bool> failurePublished_{false};

    public:
        nixlUcxConnection(uint64_t identity, nixl::ucx::connection_metadata_t metadata)
            : identity_(identity), metadata_(std::move(metadata)) {}

        [[nodiscard]] const std::unique_ptr<nixlUcxEp>& getEp(size_t ep_id) const noexcept {
            return eps[ep_id];
        }

        [[nodiscard]] uint64_t
        getIdentity() const noexcept {
            return identity_;
        }

        [[nodiscard]] const nixl::ucx::connection_metadata_t &
        getMetadata() const noexcept {
            return metadata_;
        }

        [[nodiscard]] bool
        claimFailurePublication() noexcept {
            bool expected = false;
            return failurePublished_.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel);
        }

    friend class nixlUcxEngine;
};

using ucx_connection_ptr_t = std::shared_ptr<nixlUcxConnection>;

// A private metadata has to implement get, and has all the metadata
class nixlUcxPrivateMetadata : public nixlBackendMD {
    private:
        nixlUcxMem mem;
        nixl_blob_t rkeyStr;

    public:
        nixlUcxPrivateMetadata() : nixlBackendMD(true) {
        }

        [[nodiscard]] const std::string& get() const noexcept {
            return rkeyStr;
        }

        [[nodiscard]] const nixlUcxMem &
        getMem() const noexcept {
            return mem;
        }

    friend class nixlUcxEngine;
};

// A public metadata has to implement put, and only has the remote metadata
class nixlUcxPublicMetadata : public nixlBackendMD {
public:
    nixlUcxPublicMetadata() = delete;
    nixlUcxPublicMetadata(const ucx_connection_ptr_t &conn, std::vector<nixl::ucx::rkey> &&rkeys);

    [[nodiscard]] const nixl::ucx::rkey &
    getRkey(const size_t id) const {
        return rkeys_[id];
    }

    const ucx_connection_ptr_t conn;

private:
    const std::vector<nixl::ucx::rkey> rkeys_;
};

class nixlUcxEngine : public nixlBackendEngine {
public:
    static std::unique_ptr<nixlUcxEngine>
    create(const nixlBackendInitParams &init_params);

    ~nixlUcxEngine();

    bool
    supportsRemote() const override {
        return true;
    }

    bool
    supportsLocal() const override {
        return true;
    }

    bool
    supportsNotif() const override {
        return true;
    }

    bool
    supportsAuthenticatedNotif() const override {
        return true;
    }

    nixl_mem_list_t
    getSupportedMems() const override;

    /* Object management */
    nixl_status_t
    getPublicData(const nixlBackendMD *meta, std::string &str) const override;
    nixl_status_t
    getConnInfo(std::string &str) const override;
    nixl_status_t
    loadRemoteConnInfo(const std::string &remote_agent,
                       const std::string &remote_conn_info) override;

    nixl_status_t
    connect(const std::string &remote_agent) override;
    nixl_status_t
    disconnect(const std::string &remote_agent) override;

    nixl_status_t
    queryRemoteAgentAuthority(const std::string &remote_agent,
                              nixl_remote_agent_authority_t &authority) const override;

    nixl_status_t
    bindRemoteAgent(const nixlRemoteAgentBinding &binding) override;
    nixl_status_t
    retireRemoteAgent(const nixlRemoteAgentBinding &binding) override;
    nixl_status_t
    queryRemoteNotificationState(const nixlRemoteAgentBinding &binding) const override;

    nixl_status_t
    subscribeXferTerminal(
        nixlBackendReqH *handle,
        const nixlBackendTransferEventBinding &binding,
        const std::shared_ptr<nixlBackendTransferTransitionSink> &sink,
        std::unique_ptr<nixlBackendEventSubscription> &subscription) override;

    nixl_status_t
    subscribeRemoteNotificationState(
        const nixlRemoteAgentBinding &binding,
        const std::shared_ptr<nixlBackendCapabilityTransitionSink> &sink,
        std::unique_ptr<nixlBackendEventSubscription> &subscription) override;

    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;
    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

    nixl_status_t
    loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override;

    nixl_status_t
    loadRemoteMD(const nixlBlobDesc &input,
                 const nixl_mem_t &nixl_mem,
                 const std::string &remote_agent,
                 nixlBackendMD *&output) override;
    nixl_status_t
    unloadMD(nixlBackendMD *input) override;

    // Data transfer
    nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    estimateXferCost(const nixl_xfer_op_t &operation,
                     const nixl_meta_dlist_t &local,
                     const nixl_meta_dlist_t &remote,
                     const std::string &remote_agent,
                     nixlBackendReqH *const &handle,
                     std::chrono::microseconds &duration,
                     std::chrono::microseconds &err_margin,
                     nixl_cost_t &method,
                     const nixl_opt_args_t *opt_args = nullptr) const override;

    nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    checkXfer(nixlBackendReqH *handle) const override;
    nixl_status_t
    queryXferAttestation(const nixlBackendReqH *handle,
                         nixl_xfer_attestation_t &attestation) const override;
    nixl_status_t
    takeXferCompletionAttestation(nixlBackendReqH *handle,
                                  nixl_xfer_attestation_t &attestation) const override;
    nixl_status_t
    releaseReqH(nixlBackendReqH *handle) const override;

    unsigned
    progress();

    void
    progressLoop();

    nixl_status_t
    getNotifs(notif_list_t &notif_list) override;
    nixl_status_t
    getAuthenticatedNotifs(authenticated_notif_list_t &notif_list) override;
    nixl_status_t
    genNotif(const std::string &remote_agent, const std::string &msg) const override;
    nixl_status_t
    genNotif(const nixlRemoteAgentBinding &binding,
             const std::string &msg) const override;

    // public function for UCX worker to mark connections as connected
    nixl_status_t
    checkConn(const std::string &remote_agent);

    nixl_status_t
    prepMemView(const nixl_remote_meta_dlist_t &,
                nixlMemViewH &,
                const nixl_opt_b_args_t * = nullptr) const override;

    nixl_status_t
    prepMemView(const nixl_meta_dlist_t &,
                nixlMemViewH &,
                const nixl_opt_b_args_t * = nullptr) const override;

    void releaseMemView(nixlMemViewH) const override;

protected:
    const std::vector<std::unique_ptr<nixlUcxWorker>> &
    getWorkers() const {
        return uws;
    }

    const std::unique_ptr<nixlUcxWorker> &
    getWorker(size_t worker_id) const {
        return uws[worker_id];
    }

    [[nodiscard]] size_t
    getWorkerId(const nixl_opt_b_args_t *opt_args = nullptr) const noexcept;

    virtual size_t
    getSharedWorkersSize() const {
        return uws.size();
    }

    virtual void
    appendNotif(nixlAuthenticatedNotification &&notification) const;
    virtual void
    poisonNotifs() const;

    virtual nixl_status_t
    sendXferRange(const nixl_xfer_op_t &operation,
                  const nixl_meta_dlist_t &local,
                  const nixl_meta_dlist_t &remote,
                  const std::string &remote_agent,
                  nixlBackendReqH *handle,
                  size_t start_idx,
                  size_t end_idx) const;

    [[nodiscard]] nixl_status_t
    prepareHandleAttestation(nixlBackendReqH *handle,
                             const nixl_xfer_op_t &operation,
                             const nixl_meta_dlist_t &local,
                             const nixl_meta_dlist_t &remote,
                             const std::string &remote_agent,
                             const nixl_opt_b_args_t *opt_args) const;

    nixlUcxEngine(const nixlBackendInitParams &init_params);

    mutable nixlUcxNotificationQueue notifQueue_;

private:
    struct notifCallbackContext {
        nixlUcxEngine *engine;
        size_t workerId;
    };

    struct exactRouteRecord {
        nixl::ucx::notif_route_key_t route;
        std::string remoteAgent;
        uint64_t connectionIdentity;
    };

    // Memory management helpers
    nixl_status_t
    internalMDHelper(const nixl_blob_t &blob, const std::string &agent, nixlBackendMD *&output);

    // Notifications
    static ucs_status_t
    notifAmCb(void *arg,
              const void *header,
              size_t header_length,
              void *data,
              size_t length,
              const ucp_am_recv_param_t *param);

    nixl_status_t
    notifSendPriv(std::vector<std::uint8_t> &&frame,
                  const std::unique_ptr<nixlUcxEp> &ep,
                  nixlUcxReq *req = nullptr,
                  nixl::ucx::ucx_callback_slot_t *terminal_slot = nullptr) const;

    nixl_status_t
    sendControlFrame(const nixl::ucx::notif_wire_envelope_t &envelope,
                     uint64_t connection_identity,
                     size_t worker_id) const;

    [[nodiscard]] nixl_status_t
    makeRoute(const nixlRemoteAgentBinding &binding,
              nixl::ucx::notif_route_key_t &route,
              ucx_connection_ptr_t &connection) const;

    [[nodiscard]] nixl_status_t
    prepareDataFrame(const nixl_remote_agent_authority_t &authority,
                     size_t worker_id,
                     const std::string &msg,
                     std::vector<std::uint8_t> &frame,
                     ucx_connection_ptr_t &connection) const;

    [[nodiscard]] std::optional<exactRouteRecord>
    getExactRoute(uint64_t handle_identity, uint64_t generation) const;

    void
    failExactRoutesForConnection(uint64_t connection_identity) noexcept;

    ucx_connection_ptr_t
    getConnection(const std::string &remote_agent) const;

    ucx_connection_ptr_t
    getConnection(uint64_t connection_identity) const;

    struct batchResult {
        nixl_status_t status;
        size_t size;
        nixlUcxReq req;
    };

    static batchResult
    sendXferRangeBatch(nixlUcxEp &ep,
                       nixl_xfer_op_t operation,
                       const nixl_meta_dlist_t &local,
                       const nixl_meta_dlist_t &remote,
                       nixlBackendReqH *handle,
                       size_t worker_id,
                       size_t start_idx,
                       size_t end_idx);

    /**
     * Get the worker ID from the optional arguments.
     * Returns std::nullopt if the 'worker_id' option extraction fails.
     */
    [[nodiscard]] std::optional<size_t>
    getWorkerIdFromOptArgs(const nixl_opt_b_args_t &opt_args) const noexcept;

    /* UCX data */
    std::unique_ptr<nixlUcxContext> uc;
    std::vector<std::unique_ptr<nixlUcxWorker>> uws;
    nixl::ucx::connection_metadata_t localConnectionMetadata_;
    std::vector<notifCallbackContext> notifCallbackContexts_;
    nixl::ucx::notif_wire_uuid_t localAgentIncarnationUuid_;
    std::unique_ptr<nixl::ucx::notif_capability_state_t> notifState_;
    mutable std::atomic<size_t> sharedWorkerIndex_;

    // Map of agent name to saved nixlUcxConnection info
    mutable std::mutex connectionMutex_;
    std::unordered_map<std::string, ucx_connection_ptr_t> remoteConnMap;
    mutable std::mutex exactRouteMutex_;
    std::unordered_map<uint64_t, exactRouteRecord> exactRoutes_;
};

class nixlUcxThread;

/**
 * Represents an engine with a single progress thread for all shared workers
 */
class nixlUcxThreadEngine : public nixlUcxEngine {
public:
    nixlUcxThreadEngine(const nixlBackendInitParams &init_params);
    ~nixlUcxThreadEngine();

    nixl_status_t
    getNotifs(notif_list_t &notif_list) override;
    nixl_status_t
    getAuthenticatedNotifs(authenticated_notif_list_t &notif_list) override;

protected:
    void
    appendNotif(nixlAuthenticatedNotification &&notification) const override;
    void
    poisonNotifs() const override;

private:
    std::unique_ptr<nixlUcxThread> thread_;
    mutable std::mutex notifMutex_;
};

namespace asio {
class io_context;
}

class nixlUcxThreadPoolEngine : public nixlUcxEngine {
public:
    nixlUcxThreadPoolEngine(const nixlBackendInitParams &init_params);
    ~nixlUcxThreadPoolEngine();

    nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    size_t
    getSharedWorkersSize() const override {
        return numSharedWorkers_;
    }

    nixl_status_t
    getNotifs(notif_list_t &notif_list) override;
    nixl_status_t
    getAuthenticatedNotifs(authenticated_notif_list_t &notif_list) override;

protected:
    void
    appendNotif(nixlAuthenticatedNotification &&notification) const override;
    void
    poisonNotifs() const override;

    nixl_status_t
    sendXferRange(const nixl_xfer_op_t &operation,
                  const nixl_meta_dlist_t &local,
                  const nixl_meta_dlist_t &remote,
                  const std::string &remote_agent,
                  nixlBackendReqH *handle,
                  size_t start_idx,
                  size_t end_idx) const override;

private:
    std::unique_ptr<asio::io_context> io_;
    std::unique_ptr<nixlUcxThread> sharedThread_;
    std::vector<std::unique_ptr<nixlUcxThread>> dedicatedThreads_;
    size_t numSharedWorkers_;
    mutable std::mutex notifMutex_;
    size_t splitBatchSize_;
};

#endif
