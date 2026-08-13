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
#ifndef _NIXL_TYPES_H
#define _NIXL_TYPES_H
#include <vector>
#include <string>
#include <unordered_map>
#include <optional>
#include <chrono>
#include <cstdint>


/*** Forward declarations ***/
class nixlSerDes;
class nixlDlistH;
class nixlBackendH;
class nixlXferReqH;
class nixlAgentData;
class nixlRemoteAgentH;
class nixlTerminalEventChannelH;
class nixlTerminalEventSubscriptionH;
class nixlTerminalOwnerProducerH;

/*** NIXL memory type, operation and status enums ***/

/**
 * @enum   nixl_mem_t
 * @brief  An enumeration of segment types for NIXL
 *         FILE_SEG must be last
 */
enum nixl_mem_t { DRAM_SEG, VRAM_SEG, BLK_SEG, OBJ_SEG, FILE_SEG };

/**
 * @enum   nixl_xfer_op_t
 * @brief  An enumeration of different transfer types for NIXL
 */
enum nixl_xfer_op_t { NIXL_READ, NIXL_WRITE };

/**
 * @enum   nixl_status_t
 * @brief  An enumeration of status values and error codes for NIXL
 */
enum nixl_status_t {
    NIXL_IN_PROG = 1,
    NIXL_SUCCESS = 0,
    NIXL_ERR_NOT_POSTED = -1,
    NIXL_ERR_INVALID_PARAM = -2,
    NIXL_ERR_BACKEND = -3,
    NIXL_ERR_NOT_FOUND = -4,
    NIXL_ERR_MISMATCH = -5,
    NIXL_ERR_NOT_ALLOWED = -6,
    NIXL_ERR_REPOST_ACTIVE = -7,
    NIXL_ERR_UNKNOWN = -8,
    NIXL_ERR_NOT_SUPPORTED = -9,
    NIXL_ERR_REMOTE_DISCONNECT = -10,
    NIXL_ERR_CANCELED = -11,
    NIXL_ERR_NO_TELEMETRY = -12,
    NIXL_ERR_NOT_READY = -13
};

/**
 * @enum nixl_thread_sync_t
 * @brief An enumeration of supported synchronization modes for NIXL
 */
enum class nixl_thread_sync_t {
    NIXL_THREAD_SYNC_NONE,
    NIXL_THREAD_SYNC_STRICT,
    NIXL_THREAD_SYNC_RW,
    NIXL_THREAD_SYNC_DEFAULT = NIXL_THREAD_SYNC_NONE,
};

/**
 * @namespace nixlEnumStrings
 * @brief     This namespace to get string representation
 *            of different enums
 */
namespace nixlEnumStrings {
std::string
memTypeStr(const nixl_mem_t &mem);
std::string
xferOpStr(const nixl_xfer_op_t &op);
std::string
statusStr(const nixl_status_t &status);
} // namespace nixlEnumStrings

/*** NIXL typedefs and defines used in the API ***/

/**
 * @brief A typedef for a std::string to identify nixl backends
 */
using nixl_backend_t = std::string;

/**
 * @brief A typedef for a std::string to identify nixl telemetry plugins
 */
using nixl_telemetry_plugin_t = std::string;

/**
 * @brief A typedef for a std::string as nixl blob
 *        std::string supports \0 natively, so it can be looked as a void* of data,
 *        with specified length. Giving it a new name to be clear in the API and
 *        preventing users to think it's a string and call c_str().
 */
using nixl_blob_t = std::string;

/**
 * @brief A typedef for a std::vector<nixl_mem_t> to create nixl_mem_list_t objects.
 */
using nixl_mem_list_t = std::vector<nixl_mem_t>;

/**
 * @brief A typedef for a  std::unordered_map<std::string, std::string>
 *        to hold nixl_b_params_t .
 */
using nixl_b_params_t = std::unordered_map<std::string, std::string>;

/**
 * @brief A typedef for a  std::unordered_map<std::string, std::vector<nixl_blob_t>>
 *        to hold nixl_notifs_t (nixl notifications)
 */
using nixl_notifs_t = std::unordered_map<std::string, std::vector<nixl_blob_t>>;
/**
 * @brief Notifications attributed to an agent-owned remote handle.
 */
using nixl_remote_notifs_t = std::unordered_map<const nixlRemoteAgentH *, std::vector<nixl_blob_t>>;


/**
 * @brief A constant to define the default communication port.
 */
inline constexpr uint16_t default_comm_port = 8888;

/**
 * @brief A constant to define the default metadata label for ETCD server key.
 *        Appended to the agent's key prefix to form the full key for metadata.
 */
extern const std::string default_metadata_label;

/**
 * @brief A constant to define the default partial metadata label for ETCD server key.
 *        Appended to the agent's key prefix to form the full key for partial metadata.
 */
extern const std::string default_partial_metadata_label;


/**
 * @enum nixl_cost_t
 * @brief An enumeration of cost types for transfer cost estimation.
 */
enum class nixl_cost_t {
    ANALYTICAL_BACKEND = 0, // Analytical backend cost estimate
};

/**
 * @brief A typedef for std::optional<nixl_b_params_t> for querying memory results
 *        Validity of a nixl_query_resp_t can be checked by has_value() method,
 *        and if true, the dictionary can be accessed by value() method.
 */
using nixl_query_resp_t = std::optional<nixl_b_params_t>;

/**
 * @struct nixlAgentOptionalArgs
 * @brief A structure for optional argument that can be provided to relevant agent methods.
 */
struct nixlAgentOptionalArgs {
    /**
     * @var backends vector to specify a list of backend handles, to limit the list
     *      of backends to be considered. Used in registerMem / deregisterMem
     *      makeConnection / prepXferDlist / makeXferReq / createXferReq / GetNotifs / GenNotif
     */
    std::vector<nixlBackendH *> backends;

    /**
     * @var notif Optional notification message used in createXferReq / makeXferReq / postXferReq.
     *            If set, notification is enabled even for empty-string messages.
     *            This is the preferred field for new API users.
     */
    std::optional<nixl_blob_t> notif;

    /**
     * @var notifMsg Legacy notification payload kept for backward compatibility.
     *               Deprecated in favor of @ref notif.
     */
    nixl_blob_t notifMsg;

    /**
     * @var hasNotif Legacy notification flag kept for backward compatibility.
     *      Deprecated in favor of @ref notif.
     */
    bool hasNotif = false;

    /**
     * @var skipDescMerge Legacy flag to skip merging consecutive descriptors.
     *      Deprecated. Kept for backward compatibility.
     */
    bool skipDescMerge = false;

    /**
     * @var includeConnInfo legacy flag to include connection information in partial metadata.
     *                      Deprecated, but still supported for backward compatibility.
     */
    bool includeConnInfo = false;

    /**
     * @var ipAddr Used to specify the IP address of a remote peer for metadata transfer.
     *                      used in sendLocalMD, fetchRemoteMD, invalidateLocalMD,
     * sendLocalPartialMD.
     */
    std::string ipAddr;

    /**
     * @var port Used to specify the port of a remote peer, ipAddr must also be set
     *                      used in sendLocalMD, fetchRemoteMD, invalidateLocalMD,
     * sendLocalPartialMD.
     */
    int port = default_comm_port;

    /**
     * @var metadataLabel Used to specify the label of the metadata to be sent/fetched
     *                    when working with ETCD metadata server. The label will be appended to the
     *                    agent's key prefix, and the full key will be used to store/fetch
     *                    the metadata key-value pair from the server.
     *                    Used in fetchRemoteMD, sendLocalPartialMD.
     *                    Note that sendLocalMD always uses default_metadata_label and ignores this
     * parameter. Note that invalidateLocalMD invalidates all labels and ignores this parameter.
     */
    std::string metadataLabel;

    /**
     * @var Backend custom parameter
     */
    nixl_blob_t customParam;
};

/**
 * @brief A typedef for a nixlAgentOptionalArgs
 *        for providing extra optional arguments
 */
using nixl_opt_args_t = nixlAgentOptionalArgs;

/**
 * @brief An alias for a nixlMemViewH
 *        Represents a memory view handle
 */
using nixlMemViewH = void *;

/**
 * @brief A typedefs for a point in time
 */
using chrono_point_t = std::chrono::steady_clock::time_point;

/**
 * @brief A typedefs for a period of time in microseconds
 */
using chrono_period_us_t = std::chrono::microseconds;

/**
 * @struct nixlXferTelemetry
 * @brief A structure for telemetry output from agent API
 */
struct nixlXferTelemetry {
    /**
     * @var startTime Time that the transfer was posted
     */
    chrono_point_t startTime;

    /**
     * @var postDuration Time it took to do the post operation
     */
    chrono_period_us_t postDuration;

    /**
     * @var xferDuration Time it took to complete the transfer
     *      if checkXferReq is called late, that might impact this result
     */
    chrono_period_us_t xferDuration;

    /**
     * @var totalBytes Amount of bytes transferred in the request
     */
    size_t totalBytes;

    /**
     * @var descCount Number of descriptors in the transfer request.
     *      If any merging of descriptors were performed, it will be reflected here.
     */
    size_t descCount;
};

/**
 * @brief A typedef for a nixlXferTelemetry
 *        for telemetry output.
 */
using nixl_xfer_telem_t = nixlXferTelemetry;

/**
 * @enum nixl_xfer_attestation_state_t
 * @brief State of the current generation of a handle-bound transfer attestation.
 */
enum class nixl_xfer_attestation_state_t {
    PREPARED,
    POSTING,
    IN_PROGRESS,
    REMOTE_FLUSHED,
    FAILED,
};

/**
 * @struct nixlXferAttestationTransport
 * @brief Endpoint-wide transport and device context reported by the backend.
 */
struct nixlXferAttestationTransport {
    std::string transport;
    std::string device;

    bool
    operator==(const nixlXferAttestationTransport &) const = default;
};

using nixl_xfer_attestation_transport_t = nixlXferAttestationTransport;

/**
 * @struct nixlXferAttestationSegment
 * @brief Immutable descriptor binding and selected transport evidence for one segment.
 */
struct nixlXferAttestationSegment {
    size_t index = 0;
    uintptr_t localAddress = 0;
    uintptr_t remoteAddress = 0;
    uint64_t localDeviceId = 0;
    uint64_t remoteDeviceId = 0;
    size_t length = 0;
    size_t workerId = 0;
    uint64_t workerIdentity = 0;
    uint64_t endpointIdentity = 0;
    std::string requestInfo;
    std::vector<nixl_xfer_attestation_transport_t> selectedTransports;
    bool posted = false;
};

using nixl_xfer_attestation_segment_t = nixlXferAttestationSegment;

/**
 * @struct nixlXferAttestationEndpoint
 * @brief Endpoint identity, context, and remote-flush evidence for one submission.
 */
struct nixlXferAttestationEndpoint {
    size_t workerId = 0;
    uint64_t workerIdentity = 0;
    uint64_t endpointIdentity = 0;
    std::vector<size_t> segmentIndices;
    std::vector<nixl_xfer_attestation_transport_t> transports;
    bool flushPosted = false;
    bool remoteFlushed = false;
};

using nixl_xfer_attestation_endpoint_t = nixlXferAttestationEndpoint;

/**
 * @struct nixlRuntimeArtifact
 * @brief Identity of a loaded runtime artifact used to interpret transport evidence.
 */
struct nixlRuntimeArtifact {
    std::string component;
    std::string path;
    std::string buildId;
    std::string version;
};

using nixl_runtime_artifact_t = nixlRuntimeArtifact;

/**
 * @struct nixlXferTerminalProgress
 * @brief Native callback and owner-queue evidence for one autonomous submission.
 */
struct nixlXferTerminalProgress {
    bool autonomous = false;
    size_t dataCallbacks = 0;
    size_t endpointFlushCallbacks = 0;
    size_t notificationCallbacks = 0;
    size_t asynchronousRequests = 0;
    size_t immediateCompletions = 0;
    size_t callbacksBeforePosterReturn = 0;
    size_t peakContinuationDepth = 0;
    size_t activeCallbackSlotsAtTerminal = 0;
    size_t continuationDepthAtTerminal = 0;
    uint64_t lastDataCallbackTimestampNs = 0;
    uint64_t lastFlushCallbackTimestampNs = 0;
    uint64_t notificationCallbackTimestampNs = 0;
    uint64_t terminalPublishTimestampNs = 0;
    nixl_status_t terminalStatus = NIXL_ERR_NOT_POSTED;
};

using nixl_xfer_terminal_progress_t = nixlXferTerminalProgress;

/**
 * @struct nixlXferAttestation
 * @brief Handle-bound evidence for one transfer submission generation.
 */
struct nixlXferAttestation {
    uint64_t handleIdentity = 0;
    uint64_t generation = 0;
    nixl_xfer_attestation_state_t state = nixl_xfer_attestation_state_t::PREPARED;
    nixl_status_t status = NIXL_ERR_NOT_POSTED;
    bool submissionSealed = false;
    bool completionClaimed = false;
    std::string backend;
    std::string localAgent;
    std::string remoteAgent;
    uint64_t remoteAgentHandleIdentity = 0;
    uint64_t remoteAgentGeneration = 0;
    uint64_t remoteConnectionIdentity = 0;
    std::vector<uint64_t> authorizedEndpointIdentities;
    nixl_xfer_op_t operation = NIXL_WRITE;
    nixl_mem_t localMemoryType = DRAM_SEG;
    nixl_mem_t remoteMemoryType = DRAM_SEG;
    std::vector<nixl_xfer_attestation_segment_t> segments;
    std::vector<nixl_xfer_attestation_endpoint_t> endpoints;
    nixl_xfer_terminal_progress_t terminalProgress;
    std::vector<nixl_runtime_artifact_t> runtimeArtifacts;
    std::string descriptorDigest;
    std::string evidenceDigest;
    std::string error;
};

using nixl_xfer_attestation_t = nixlXferAttestation;

/**
 * @enum nixl_terminal_event_kind_t
 * @brief Kind of autonomous event delivered through an agent terminal channel.
 */
enum class nixl_terminal_event_kind_t {
    TRANSFER,
    CAPABILITY,
};

/**
 * @enum nixl_terminal_capability_state_t
 * @brief State of one exact remote notification route.
 */
enum class nixl_terminal_capability_state_t {
    READY,
    FAILED,
    RETIRED,
};

/**
 * @enum nixl_terminal_channel_fatal_t
 * @brief Sticky process-fatal conditions reported by a terminal channel.
 */
enum class nixl_terminal_channel_fatal_t : uint32_t {
    NONE = 0,
    QUEUE_OVERFLOW = 1U << 0U,
    EVENTFD_FAILURE = 1U << 1U,
    ACTIVE_SUBSCRIPTIONS_ON_CLOSE = 1U << 2U,
    INVALID_PUBLICATION = 1U << 3U,
};

/**
 * @enum nixl_terminal_destination_phase_t
 * @brief Active destination lifecycle phase for an attached notification.
 */
enum class nixl_terminal_destination_phase_t {
    PENDING,
    ADMITTING,
    COMMITTED,
    REPLAYING,
    QUARANTINED,
};

/**
 * @struct nixlTerminalSourceDelivery
 * @brief Exact active source-delivery identity and join state.
 */
struct nixlTerminalSourceDelivery {
    nixl_backend_t backend;
    uint64_t deliveryIdentity = 0;
    uint64_t sourceHandleIdentity = 0;
    uint64_t sourceGeneration = 0;
    bool localPending = false;
    bool receiptPending = false;
    bool deadlineActive = false;
};

using nixl_terminal_source_delivery_t = nixlTerminalSourceDelivery;

/**
 * @struct nixlTerminalDeadline
 * @brief Exact active native transfer deadline identity.
 */
struct nixlTerminalDeadline {
    nixl_backend_t backend;
    uint64_t handleIdentity = 0;
    uint64_t generation = 0;
};

using nixl_terminal_deadline_t = nixlTerminalDeadline;

/**
 * @struct nixlTerminalDestinationDelivery
 * @brief Exact active destination-delivery identity and lifecycle phase.
 */
struct nixlTerminalDestinationDelivery {
    nixl_backend_t backend;
    std::string sourceBackendIncarnation;
    uint64_t sourceHandleIdentity = 0;
    uint64_t sourceGeneration = 0;
    uint64_t deliveryIdentity = 0;
    nixl_terminal_destination_phase_t phase = nixl_terminal_destination_phase_t::PENDING;
};

using nixl_terminal_destination_delivery_t = nixlTerminalDestinationDelivery;

/**
 * @struct nixlTerminalBackendLifecycleInventory
 * @brief Aggregate live terminal-delivery obligations across unique backend engines.
 *
 * Completed tombstones are bounded replay evidence and do not contribute to this inventory.
 */
struct nixlTerminalBackendLifecycleInventory {
    size_t sourceDeliveriesOutstanding = 0;
    size_t sourceLocalPending = 0;
    size_t sourceReceiptPending = 0;
    size_t destinationPending = 0;
    size_t destinationAdmitting = 0;
    size_t destinationCommitted = 0;
    size_t destinationReplaying = 0;
    size_t destinationQuarantined = 0;
    size_t activeNativeDeadlines = 0;
    std::vector<nixl_terminal_source_delivery_t> sourceDeliveries;
    std::vector<nixl_terminal_destination_delivery_t> destinationDeliveries;
    std::vector<nixl_terminal_deadline_t> nativeDeadlines;
};

using nixl_terminal_backend_lifecycle_inventory_t = nixlTerminalBackendLifecycleInventory;

/**
 * @struct nixlTerminalEvent
 * @brief Immutable-by-convention autonomous event drained from an agent channel.
 */
struct nixlTerminalEvent {
    nixl_terminal_event_kind_t kind = nixl_terminal_event_kind_t::TRANSFER;
    uint64_t ownerCookie = 0;
    uint64_t identity = 0;
    uint64_t generation = 0;
    nixl_status_t transferStatus = NIXL_ERR_NOT_READY;
    nixl_terminal_capability_state_t capabilityState = nixl_terminal_capability_state_t::FAILED;
    uint64_t capabilityEpoch = 0;
    uint64_t nativeTimestampNs = 0;
};

using nixl_terminal_event_t = nixlTerminalEvent;

/**
 * @struct nixlTerminalChannelInventory
 * @brief Fail-closed channel state and subscription inventory.
 */
struct nixlTerminalChannelInventory {
    size_t capacity = 0;
    size_t queuedChannelEvents = 0;
    size_t activeChannelSubscriptions = 0;
    size_t retainedPublicSubscriptions = 0;
    size_t backendProducers = 0;
    size_t activeCallbackSlots = 0;
    size_t queuedOwnerContinuations = 0;
    nixl_terminal_backend_lifecycle_inventory_t backendLifecycle;
    bool acceptingSubscriptions = false;
    bool closed = false;
    nixl_terminal_channel_fatal_t fatal = nixl_terminal_channel_fatal_t::NONE;
    int eventfdError = 0;
};

using nixl_terminal_channel_inventory_t = nixlTerminalChannelInventory;

/**
 * @struct nixlTerminalEventBatch
 * @brief One nonblocking drain result from a terminal channel.
 */
struct nixlTerminalEventBatch {
    std::vector<nixl_terminal_event_t> events;
    uint64_t wakeCount = 0;
    nixl_terminal_channel_inventory_t inventory;
};

using nixl_terminal_event_batch_t = nixlTerminalEventBatch;

/**
 * @struct nixlTerminalSubscriptionInfo
 * @brief Exact binding and lifecycle state of one agent-owned subscription.
 */
struct nixlTerminalSubscriptionInfo {
    nixl_terminal_event_kind_t kind = nixl_terminal_event_kind_t::TRANSFER;
    uint64_t ownerCookie = 0;
    uint64_t identity = 0;
    uint64_t generation = 0;
    bool active = false;
};

using nixl_terminal_subscription_info_t = nixlTerminalSubscriptionInfo;

/**
 * @struct nixlRemoteAgentAuthority
 * @brief Immutable authority captured from one active remote-agent generation.
 */
struct nixlRemoteAgentAuthority {
    uint64_t handleIdentity = 0;
    uint64_t generation = 0;
    uint64_t connectionIdentity = 0;
    std::vector<uint64_t> endpointIdentities;
};

using nixl_remote_agent_authority_t = nixlRemoteAgentAuthority;


/**
 * @brief A define for an empty string, that indicates the descriptor list is being
 *        prepared for the local agent as an initiator in prepXferDlist method.
 */
#define NIXL_INIT_AGENT ""

/**
 * @brief A constant for an invalid agent name.
 */
extern const std::string nixl_null_agent;

#endif
