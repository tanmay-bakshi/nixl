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
#ifndef __BACKEND_AUX_H_
#define __BACKEND_AUX_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "nixl_types.h"
#include "nixl_descriptors.h"
#include "common/nixl_time.h"

// Might be removed to be decided by backend, or changed to high
// level direction or so.
typedef std::vector<std::pair<std::string, std::string>> notif_list_t;

struct nixlAuthenticatedNotification {
    std::string remoteAgent;
    nixl_blob_t payload;
    uint64_t handleIdentity = 0;
    uint64_t generation = 0;
    uint64_t connectionIdentity = 0;
    uint64_t endpointIdentity = 0;
};

using authenticated_notif_list_t = std::vector<nixlAuthenticatedNotification>;

/**
 * @brief Exact agent-owned authority installed in an authenticated backend.
 */
struct nixlRemoteAgentBinding {
    std::string remoteAgent;
    std::string agentIncarnation;
    nixl_remote_agent_authority_t authority;
};

/**
 * @brief Exact identity of the next backend transfer submission.
 */
struct nixlBackendTransferEventBinding {
    uint64_t handleIdentity = 0;
    uint64_t generation = 0;
};

/**
 * @brief Terminal transition emitted autonomously by a backend progress owner.
 */
struct nixlBackendTransferTransition {
    nixlBackendTransferEventBinding binding;
    nixl_status_t status = NIXL_ERR_NOT_READY;
    uint64_t nativeTimestampNs = 0;
};

enum class nixl_backend_capability_state_t {
    READY,
    FAILED,
    RETIRED,
};

/**
 * @brief Exact notification-route transition emitted by a backend.
 */
struct nixlBackendCapabilityTransition {
    uint64_t remoteHandleIdentity = 0;
    uint64_t remoteHandleGeneration = 0;
    nixl_backend_capability_state_t state = nixl_backend_capability_state_t::FAILED;
    uint64_t capabilityEpoch = 0;
    uint64_t nativeTimestampNs = 0;
};

/**
 * @brief Externally owned sink for one exact transfer generation.
 */
class nixlBackendTransferTransitionSink {
public:
    virtual ~nixlBackendTransferTransitionSink() = default;

    virtual void
    publish(const nixlBackendTransferTransition &transition) noexcept = 0;
};

/**
 * @brief Externally owned sink for one exact remote notification route.
 */
class nixlBackendCapabilityTransitionSink {
public:
    virtual ~nixlBackendCapabilityTransitionSink() = default;

    virtual void
    publish(const nixlBackendCapabilityTransition &transition) noexcept = 0;
};

/**
 * @brief Per-subscription native lifecycle inventory.
 */
struct nixlBackendEventSubscriptionInventory {
    size_t backendProducers = 0;
    size_t activeCallbackSlots = 0;
    size_t queuedOwnerContinuations = 0;
};

/**
 * @brief Exact active source-delivery identity owned by one backend engine.
 */
struct nixlBackendTerminalSourceDelivery {
    uint64_t deliveryIdentity = 0;
    uint64_t sourceHandleIdentity = 0;
    uint64_t sourceGeneration = 0;
    bool localPending = false;
    bool receiptPending = false;
    bool deadlineActive = false;
};

/**
 * @brief Exact active native transfer deadline owned by one backend engine.
 */
struct nixlBackendTerminalDeadline {
    uint64_t handleIdentity = 0;
    uint64_t generation = 0;
};

/**
 * @brief Backend destination phase for one active admission obligation.
 */
enum class nixl_backend_terminal_destination_phase_t {
    PENDING,
    ADMITTING,
    COMMITTED,
    REPLAYING,
    QUARANTINED,
};

/**
 * @brief Immutable authority for one committed remote admission.
 */
struct nixlBackendAdmissionReceiptAuthority {
    uint64_t sourceHandleIdentity = 0;
    uint64_t sourceGeneration = 0;
    uint64_t deliveryIdentity = 0;
};

/**
 * @brief One-shot qualification barrier after committed remote admission.
 *
 * Production never installs this hook. The callback must take ownership of the
 * unchanged receipt-scheduling continuation and return without waiting. This
 * lets a real transport test hold the receipt at the exact authority boundary
 * without blocking the UCX callback or progress owner.
 */
class nixlBackendAdmissionReceiptBarrier {
public:
    using schedule_receipt_t = std::function<nixl_status_t()>;

    virtual ~nixlBackendAdmissionReceiptBarrier() = default;

    [[nodiscard]] virtual nixl_status_t
    deferAfterAdmission(const nixlBackendAdmissionReceiptAuthority &authority,
                        schedule_receipt_t schedule_receipt) noexcept = 0;
};

/**
 * @brief Exact active destination-delivery identity owned by one backend engine.
 */
struct nixlBackendTerminalDestinationDelivery {
    std::string sourceBackendIncarnation;
    uint64_t sourceHandleIdentity = 0;
    uint64_t sourceGeneration = 0;
    uint64_t deliveryIdentity = 0;
    nixl_backend_terminal_destination_phase_t phase =
        nixl_backend_terminal_destination_phase_t::PENDING;
};

/**
 * @brief Engine-global terminal-delivery lifecycle inventory.
 *
 * Completed tombstones are bounded replay evidence and are intentionally excluded. The exact
 * identity collections contain only live delivery, replay, or quarantine obligations.
 */
struct nixlBackendTerminalLifecycleInventory {
    size_t sourceDeliveriesOutstanding = 0;
    size_t sourceLocalPending = 0;
    size_t sourceReceiptPending = 0;
    size_t destinationPending = 0;
    size_t destinationAdmitting = 0;
    size_t destinationCommitted = 0;
    size_t destinationReplaying = 0;
    size_t destinationQuarantined = 0;
    size_t activeNativeDeadlines = 0;
    std::vector<nixlBackendTerminalSourceDelivery> sourceDeliveries;
    std::vector<nixlBackendTerminalDestinationDelivery> destinationDeliveries;
    std::vector<nixlBackendTerminalDeadline> nativeDeadlines;
};

/**
 * @brief Backend-owned lifetime for an autonomous event subscription.
 */
class nixlBackendEventSubscription {
public:
    virtual ~nixlBackendEventSubscription() = default;

    virtual nixl_status_t
    cancel() noexcept = 0;

    virtual void
    queryInventory(nixlBackendEventSubscriptionInventory &inventory) const noexcept {
        inventory = {};
    }

    /**
     * Drive backend-owned cancellation progress during fail-closed agent shutdown.
     *
     * Autonomous backends may block on their native completion primitive. This call must not
     * return success until callbacks, producers, and continuations owned by this subscription are
     * fully drained. Backends with asynchronous cancellation must override this method; the
     * default fails closed instead of issuing a second cancellation request.
     */
    virtual nixl_status_t
    drainCancellation() noexcept {
        return NIXL_ERR_NOT_SUPPORTED;
    }
};

struct nixlBackendOptionalArgs {
    // During postXfer, user might ask for a notification if supported
    nixl_blob_t notifMsg;
    bool hasNotif = false;
    nixl_blob_t customParam;
    const nixl_remote_agent_authority_t *remoteAgentAuthority = nullptr;
};

using nixl_opt_b_args_t = nixlBackendOptionalArgs;

// A base class to point to backend initialization data
// User doesn't know about fields such as local_agent but can access it
// after the backend is initialized by agent. If we needed to make it private
// from the user, we should make nixlBackendEngine/nixlAgent friend classes.
class nixlBackendInitParams {
public:
    std::string localAgent;
    std::string localAgentIncarnation;

    nixl_backend_t type;
    nixl_b_params_t *customParams;

    bool enableProgTh;
    nixlTime::us_t pthrDelay;
    nixl_thread_sync_t syncMode;
    bool enableTelemetry_;
};

// Pure virtual class to have a common pointer type
class nixlBackendReqH {
public:
    nixlBackendReqH() {}

    virtual ~nixlBackendReqH() {}
};

// Pure virtual class to have a common pointer type for different backendMD.
class nixlBackendMD {
protected:
    bool isPrivateMD;

public:
    nixlBackendMD(bool isPrivate) {
        isPrivateMD = isPrivate;
    }

    virtual ~nixlBackendMD() {}
};

// Each backend can have different connection requirement
// This class would include the required information to make
// a connection to a remote node. Note that local information
// is passed during the constructor and through BackendInitParams
class nixlBackendConnMD {
public:
    // And some other details
    std::string dstIpAddress;
    uint16_t dstPort;
};

// A pointer required to a metadata object for backends next to each BasicDesc
class nixlMetaDesc : public nixlBasicDesc {
public:
    // To be able to point to any object
    nixlBackendMD *metadataP;

    // Reuse parent constructor without the metadata pointer
    using nixlBasicDesc::nixlBasicDesc;

    nixlMetaDesc() : nixlBasicDesc() {
        metadataP = nullptr;
    }

    nixlMetaDesc(uintptr_t addr, size_t len, uint64_t dev_id, nixlBackendMD *metadata)
        : nixlBasicDesc(addr, len, dev_id),
          metadataP(metadata) {}

    // No serializer or deserializer, using parent not to expose the metadata

    inline friend bool
    operator==(const nixlMetaDesc &lhs, const nixlMetaDesc &rhs) {
        return (((nixlBasicDesc)lhs == (nixlBasicDesc)rhs) && (lhs.metadataP == rhs.metadataP));
    }

    inline void
    print(const std::string &suffix) const {
        nixlBasicDesc::print(", Backend ptr val: " + std::to_string((uintptr_t)metadataP) + suffix);
    }
};

struct nixlRemoteMetaDesc : public nixlMetaDesc {
    std::string remoteAgent;

    using nixlMetaDesc::nixlMetaDesc;

    explicit nixlRemoteMetaDesc(const std::string &remote_agent)
        : nixlMetaDesc(),
          remoteAgent(remote_agent) {}
};

inline bool
operator==(const nixlRemoteMetaDesc &lhs, const nixlRemoteMetaDesc &rhs) {
    return (static_cast<const nixlMetaDesc &>(lhs) == static_cast<const nixlMetaDesc &>(rhs)) &&
        (lhs.remoteAgent == rhs.remoteAgent);
}

typedef nixlDescList<nixlMetaDesc> nixl_meta_dlist_t;
using nixl_remote_meta_dlist_t = nixlDescList<nixlRemoteMetaDesc>;

#endif
