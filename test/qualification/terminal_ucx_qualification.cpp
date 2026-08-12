/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "terminal_ucx_api_adapter.h"
#include "terminal_ucx_peer_fixture.h"

#include <poll.h>
#include <time.h>
#include <unistd.h>

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t channel_capacity = 64;
constexpr std::size_t descriptor_bytes = 8U * 1024U * 1024U;
constexpr std::size_t large_descriptor_count = 8;
constexpr std::size_t arena_bytes = descriptor_bytes * large_descriptor_count;
constexpr int event_timeout_ms = 30000;
constexpr std::uint64_t device_id = 0;
constexpr std::string_view self_na_reason =
    "ucx_self_is_same_worker_only_and_nixl_local_routes_have_no_remote_agent_handle";

class qualification_error final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct options_t {
    std::string transport;
    std::string engine;
    std::filesystem::path output;
};

struct population_spec_t {
    std::string name;
    std::size_t descriptorCount = 0;
    std::size_t bytesPerDescriptor = 0;
    std::uint8_t seed = 0;
};

struct population_result_t {
    population_spec_t spec;
    std::size_t byteCount = 0;
    std::size_t destinationByteCount = 0;
    std::string sourceSha256;
    std::string destinationSha256;
    nixl_status_t terminalStatus = NIXL_ERR_BACKEND;
    nixl_status_t attestationStatus = NIXL_ERR_BACKEND;
    nixl_status_t secondTakeStatus = NIXL_ERR_BACKEND;
    nixl_xfer_attestation_t attestation;
    std::uint64_t eventNativeTimestampNs = 0;
    std::uint64_t drainTimestampNs = 0;
    std::size_t terminalEventCount = 0;
    bool subscriptionBeforePost = false;
};

struct registration_result_t {
    nixl_status_t status = NIXL_ERR_BACKEND;
    std::uintptr_t baseAddress = 0;
    std::size_t byteCapacity = 0;
};

struct terminal_fault_result_t {
    nixl_status_t terminalStatus = NIXL_ERR_BACKEND;
    std::size_t terminalEventCount = 0;
    bool ownerWoken = false;
};

struct queue_overflow_result_t {
    std::uint32_t fatal = 0;
    std::size_t wakeCount = 0;
    std::size_t admittedEvents = 0;
};

struct shutdown_result_t : terminal_fault_result_t {
    nixl_status_t cancelStatus = NIXL_ERR_BACKEND;
    std::size_t backendProducersBeforeCancel = 0;
    std::size_t activeCallbackSlotsBeforeCancel = 0;
    std::size_t queuedOwnerContinuationsBeforeCancel = 0;
    bool postedInFlight = false;
    bool drained = false;
};

struct route_result_t {
    std::string name;
    std::uint64_t identity = 0;
    std::uint64_t generation = 0;
    std::vector<nixl_terminal_capability_state_t> states;
    std::vector<std::uint64_t> epochs;
    nixl_status_t releaseStatus = NIXL_ERR_BACKEND;
    bool subscriptionTerminal = false;
    nixlTerminalEventSubscriptionH *subscription = nullptr;
};

struct capability_result_t {
    bool subscribeBeforeReady = false;
    bool snapshotAfterReady = false;
    std::vector<route_result_t> routes;
};

struct tcp_fixture_result_t {
    nixl::qualification::tcp_endpoint_failure_observation_t endpointFailure;
    nixl::qualification::tcp_notification_failure_observation_t notificationFailure;
    nixl::qualification::tcp_shutdown_cancellation_observation_t shutdownCancellation;
};

struct tcp_fault_results_t {
    terminal_fault_result_t remoteFailure;
    terminal_fault_result_t notificationFailure;
    bool dataRemoteFlushedBeforeNotificationFailure = false;
};

void
require(bool condition, std::string_view message) {
    if (!condition) {
        throw qualification_error(std::string(message));
    }
}

void
requireStatus(nixl_status_t actual, nixl_status_t expected, std::string_view operation) {
    if (actual == expected) {
        return;
    }
    throw qualification_error(std::string(operation) + " failed with " +
                              nixlEnumStrings::statusStr(actual));
}

[[nodiscard]] std::uint64_t
monotonicRawNs() {
    timespec timestamp = {};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp) != 0) {
        throw qualification_error("CLOCK_MONOTONIC_RAW failed");
    }
    return static_cast<std::uint64_t>(timestamp.tv_sec) * 1'000'000'000ULL +
        static_cast<std::uint64_t>(timestamp.tv_nsec);
}

[[nodiscard]] options_t
parseOptions(int argc, char **argv) {
    options_t options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--transport" && index + 1 < argc) {
            options.transport = argv[++index];
            continue;
        }
        if (argument == "--engine" && index + 1 < argc) {
            options.engine = argv[++index];
            continue;
        }
        if (argument == "--output" && index + 1 < argc) {
            options.output = argv[++index];
            continue;
        }
        throw qualification_error("unknown or incomplete argument: " + argument);
    }
    require(options.transport == "self" || options.transport == "tcp",
            "transport must be self or tcp");
    require(options.engine == "shared" || options.engine == "thread_pool",
            "engine must be shared or thread_pool");
    require(!options.output.empty(), "output path is required");
    return options;
}

[[nodiscard]] nixlAgentConfig
agentConfig() {
    nixlAgentConfig config;
    config.useProgThread = true;
    config.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
    return config;
}

[[nodiscard]] nixl_b_params_t
backendParameters(const std::string &transport, const std::string &engine) {
    nixl_b_params_t parameters;
    parameters["ucx_error_handling_mode"] = transport == "self" ? "none" : "peer";
    if (engine == "shared") {
        parameters["num_workers"] = "2";
        parameters["num_threads"] = "0";
        parameters["split_batch_size"] = "2";
        return parameters;
    }
    parameters["num_workers"] = "3";
    parameters["num_threads"] = "2";
    parameters["split_batch_size"] = "2";
    return parameters;
}

class registered_arena_t final {
public:
    registered_arena_t(nixlAgent &agent,
                       nixlBackendH *backend,
                       std::size_t byte_capacity,
                       std::size_t descriptor_capacity)
        : agent_(&agent),
          backend_(backend),
          bytes_(byte_capacity, 0),
          descriptorCapacity_(descriptor_capacity) {
        require(descriptor_capacity > 0 && byte_capacity % descriptor_capacity == 0,
                "registered arena geometry is invalid");
        nixl_opt_args_t arguments;
        arguments.backends = {backend_};
        result_.status = agent_->registerMem(registrationList(), &arguments);
        requireStatus(result_.status, NIXL_SUCCESS, "register DRAM arena");
        result_.baseAddress = reinterpret_cast<std::uintptr_t>(bytes_.data());
        result_.byteCapacity = bytes_.size();
        registered_ = true;
    }

    registered_arena_t(const registered_arena_t &) = delete;
    registered_arena_t &
    operator=(const registered_arena_t &) = delete;

    ~registered_arena_t() {
        if (!registered_ || agent_ == nullptr) {
            return;
        }
        nixl_opt_args_t arguments;
        arguments.backends = {backend_};
        static_cast<void>(agent_->deregisterMem(registrationList(), &arguments));
    }

    void
    abandonAgent() noexcept {
        agent_ = nullptr;
        registered_ = false;
    }

    void
    fill(const population_spec_t &spec) {
        validateSpec(spec);
        for (std::size_t descriptor = 0; descriptor < spec.descriptorCount; ++descriptor) {
            std::uint8_t *const base = bytes_.data() + descriptor * descriptorStride();
            for (std::size_t offset = 0; offset < spec.bytesPerDescriptor; ++offset) {
                base[offset] =
                    static_cast<std::uint8_t>(spec.seed + descriptor * 17U + offset * 29U);
            }
        }
    }

    void
    clear(const population_spec_t &spec) {
        validateSpec(spec);
        for (std::size_t descriptor = 0; descriptor < spec.descriptorCount; ++descriptor) {
            std::fill_n(bytes_.data() + descriptor * descriptorStride(),
                        spec.bytesPerDescriptor,
                        std::uint8_t{0});
        }
    }

    [[nodiscard]] nixl_xfer_dlist_t
    transferList(const population_spec_t &spec) const {
        validateSpec(spec);
        nixl_xfer_dlist_t descriptors(DRAM_SEG);
        for (std::size_t descriptor = 0; descriptor < spec.descriptorCount; ++descriptor) {
            descriptors.addDesc(nixlBasicDesc(
                reinterpret_cast<std::uintptr_t>(bytes_.data() + descriptor * descriptorStride()),
                spec.bytesPerDescriptor,
                device_id));
        }
        return descriptors;
    }

    [[nodiscard]] std::string
    sha256(const population_spec_t &spec) const {
        validateSpec(spec);
        std::string input;
        input.reserve(spec.descriptorCount * spec.bytesPerDescriptor);
        for (std::size_t descriptor = 0; descriptor < spec.descriptorCount; ++descriptor) {
            const char *const base =
                reinterpret_cast<const char *>(bytes_.data() + descriptor * descriptorStride());
            input.append(base, spec.bytesPerDescriptor);
        }
        return sha256Bytes(input);
    }

    [[nodiscard]] const registration_result_t &
    registration() const noexcept {
        return result_;
    }

private:
    [[nodiscard]] std::size_t
    descriptorStride() const noexcept {
        return bytes_.size() / descriptorCapacity_;
    }

    void
    validateSpec(const population_spec_t &spec) const {
        require(spec.descriptorCount > 0 && spec.descriptorCount <= descriptorCapacity_,
                "population descriptor count exceeds registration");
        require(spec.bytesPerDescriptor > 0 && spec.bytesPerDescriptor <= descriptorStride(),
                "population descriptor length exceeds registration");
    }

    [[nodiscard]] nixl_reg_dlist_t
    registrationList() const {
        nixl_reg_dlist_t descriptors(DRAM_SEG);
        for (std::size_t descriptor = 0; descriptor < descriptorCapacity_; ++descriptor) {
            descriptors.addDesc(nixlBlobDesc(
                reinterpret_cast<std::uintptr_t>(bytes_.data() + descriptor * descriptorStride()),
                descriptorStride(),
                device_id));
        }
        return descriptors;
    }

    [[nodiscard]] static std::string
    sha256Bytes(std::string_view input) {
        std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(),
                                                                        &EVP_MD_CTX_free);
        if (context == nullptr || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
            EVP_DigestUpdate(context.get(), input.data(), input.size()) != 1) {
            throw qualification_error("failed to initialize SHA-256");
        }
        std::array<unsigned char, EVP_MAX_MD_SIZE> digest = {};
        unsigned int digest_size = 0;
        if (EVP_DigestFinal_ex(context.get(), digest.data(), &digest_size) != 1 ||
            digest_size != 32U) {
            throw qualification_error("failed to finish SHA-256");
        }
        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (unsigned int index = 0; index < digest_size; ++index) {
            output << std::setw(2) << static_cast<unsigned int>(digest[index]);
        }
        return output.str();
    }

    nixlAgent *agent_;
    nixlBackendH *backend_;
    std::vector<std::uint8_t> bytes_;
    std::size_t descriptorCapacity_;
    registration_result_t result_;
    bool registered_ = false;
};

class terminal_event_inbox_t final {
public:
    explicit terminal_event_inbox_t(nixl::qualification::terminal_ucx_api_adapter_t &adapter)
        : adapter_(adapter) {
        requireStatus(adapter_.fileno(fd_), NIXL_SUCCESS, "get terminal event fd");
        require(fd_ >= 0, "terminal event fd is invalid");
    }

    [[nodiscard]] nixl_terminal_event_t
    take(nixl_terminal_event_kind_t kind, std::uint64_t owner_cookie) {
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(event_timeout_ms);
        while (true) {
            const auto found =
                std::find_if(pending_.begin(),
                             pending_.end(),
                             [kind, owner_cookie](const nixl_terminal_event_t &event) {
                                 return event.kind == kind && event.ownerCookie == owner_cookie;
                             });
            if (found != pending_.end()) {
                nixl_terminal_event_t event = *found;
                pending_.erase(found);
                return event;
            }
            const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - std::chrono::steady_clock::now());
            require(remaining.count() > 0, "terminal event deadline expired");
            pollfd descriptor = {.fd = fd_, .events = POLLIN, .revents = 0};
            const int result =
                poll(&descriptor,
                     1,
                     static_cast<int>(std::min<std::int64_t>(remaining.count(), event_timeout_ms)));
            if (result < 0) {
                throw qualification_error("terminal event poll failed");
            }
            require(result == 1 && (descriptor.revents & POLLIN) != 0,
                    "terminal event poll timed out or returned no data");
            nixl_terminal_event_batch_t batch;
            requireStatus(adapter_.drain(batch), NIXL_SUCCESS, "drain terminal events");
            require(!batch.events.empty() || batch.wakeCount > 0,
                    "terminal wake carried no event or health transition");
            pending_.insert(pending_.end(), batch.events.begin(), batch.events.end());
        }
    }

    [[nodiscard]] nixl_terminal_event_batch_t
    drainAfterWake() {
        pollfd descriptor = {.fd = fd_, .events = POLLIN, .revents = 0};
        const int result = poll(&descriptor, 1, event_timeout_ms);
        if (result < 0) {
            throw qualification_error("terminal health poll failed");
        }
        require(result == 1 && (descriptor.revents & POLLIN) != 0,
                "terminal health transition did not wake the owner");
        nixl_terminal_event_batch_t batch;
        requireStatus(adapter_.drain(batch), NIXL_SUCCESS, "drain terminal health transition");
        return batch;
    }

private:
    nixl::qualification::terminal_ucx_api_adapter_t &adapter_;
    int fd_ = -1;
    std::vector<nixl_terminal_event_t> pending_;
};

struct endpoint_t {
    std::unique_ptr<nixlAgent> agent;
    nixlBackendH *backend = nullptr;
};

[[nodiscard]] endpoint_t
makeEndpoint(const std::string &name, const options_t &options) {
    endpoint_t endpoint;
    endpoint.agent = std::make_unique<nixlAgent>(name, agentConfig());
    requireStatus(
        endpoint.agent->createBackend(
            "UCX", backendParameters(options.transport, options.engine), endpoint.backend),
        NIXL_SUCCESS,
        "create UCX backend");
    require(endpoint.backend != nullptr, "UCX backend handle is null");
    return endpoint;
}

[[nodiscard]] nixlRemoteAgentH *
loadRemote(nixlAgent &local, nixlAgent &remote) {
    nixl_blob_t metadata;
    requireStatus(remote.getLocalMD(metadata), NIXL_SUCCESS, "get remote metadata");
    nixlRemoteAgentH *handle = nullptr;
    requireStatus(local.loadRemoteMD(metadata, handle), NIXL_SUCCESS, "load remote metadata");
    require(handle != nullptr, "remote metadata produced no handle");
    return handle;
}

[[nodiscard]] nixlXferReqH *
createRequest(nixlAgent &agent,
              nixlBackendH *backend,
              const registered_arena_t &source,
              const registered_arena_t &destination,
              const population_spec_t &spec,
              const std::string &remote_name,
              const nixlRemoteAgentH *remote_handle,
              std::string_view notification) {
    nixl_opt_args_t arguments;
    arguments.backends = {backend};
    if (!notification.empty()) {
        arguments.notif = std::string(notification);
    }
    nixlXferReqH *request = nullptr;
    const nixl_status_t status = remote_handle == nullptr ?
        agent.createXferReq(NIXL_WRITE,
                            source.transferList(spec),
                            destination.transferList(spec),
                            remote_name,
                            request,
                            &arguments) :
        agent.createXferReq(NIXL_WRITE,
                            source.transferList(spec),
                            destination.transferList(spec),
                            remote_handle,
                            request,
                            &arguments);
    requireStatus(status, NIXL_SUCCESS, "create transfer request");
    require(request != nullptr, "create transfer returned a null request");
    return request;
}

[[nodiscard]] population_result_t
runPopulation(nixlAgent &source_agent,
              nixlBackendH *source_backend,
              const std::string &remote_name,
              const nixlRemoteAgentH *remote_handle,
              nixl::qualification::terminal_ucx_api_adapter_t &adapter,
              terminal_event_inbox_t &inbox,
              registered_arena_t &source,
              registered_arena_t &destination,
              const population_spec_t &spec,
              std::uint64_t owner_cookie,
              std::string_view notification) {
    source.fill(spec);
    destination.clear(spec);
    nixlXferReqH *request = createRequest(source_agent,
                                          source_backend,
                                          source,
                                          destination,
                                          spec,
                                          remote_name,
                                          remote_handle,
                                          notification);
    nixlTerminalEventSubscriptionH *subscription = nullptr;
    requireStatus(adapter.subscribeTransfer(request, owner_cookie, subscription),
                  NIXL_SUCCESS,
                  "subscribe transfer terminal event");
    require(subscription != nullptr, "transfer subscription is null");
    nixl_terminal_subscription_info_t subscription_info;
    requireStatus(adapter.querySubscription(subscription, subscription_info),
                  NIXL_SUCCESS,
                  "query armed transfer subscription");
    require(subscription_info.active, "transfer subscription was not active before post");

    const nixl_status_t post_status = source_agent.postXferReq(request);
    require(post_status == NIXL_SUCCESS || post_status == NIXL_IN_PROG, "post transfer failed");
    const nixl_terminal_event_t event =
        inbox.take(nixl_terminal_event_kind_t::TRANSFER, owner_cookie);
    const std::uint64_t drain_timestamp = monotonicRawNs();
    require(event.transferStatus == NIXL_SUCCESS, "transfer terminal event failed");
    requireStatus(adapter.querySubscription(subscription, subscription_info),
                  NIXL_SUCCESS,
                  "query terminal transfer subscription");
    require(!subscription_info.active, "terminal transfer subscription remained active");

    population_result_t result;
    result.spec = spec;
    result.byteCount = spec.descriptorCount * spec.bytesPerDescriptor;
    result.destinationByteCount = result.byteCount;
    result.sourceSha256 = source.sha256(spec);
    result.destinationSha256 = destination.sha256(spec);
    result.terminalStatus = event.transferStatus;
    result.eventNativeTimestampNs = event.nativeTimestampNs;
    result.drainTimestampNs = drain_timestamp;
    result.terminalEventCount = 1;
    result.subscriptionBeforePost = true;
    result.attestationStatus =
        source_agent.takeXferCompletionAttestation(request, result.attestation);
    nixl_xfer_attestation_t duplicate;
    result.secondTakeStatus = source_agent.takeXferCompletionAttestation(request, duplicate);

    requireStatus(result.attestationStatus, NIXL_SUCCESS, "take completion attestation");
    requireStatus(result.secondTakeStatus, NIXL_ERR_NOT_ALLOWED, "take completion twice");
    require(result.sourceSha256 == result.destinationSha256,
            "destination bytes differ from source");
    requireStatus(adapter.release(subscription), NIXL_SUCCESS, "release transfer subscription");
    requireStatus(source_agent.releaseXferReq(request), NIXL_SUCCESS, "release transfer request");
    return result;
}

[[nodiscard]] terminal_fault_result_t
runCancellation(const options_t &options, std::uint64_t suffix) {
    endpoint_t endpoint = makeEndpoint("qualification-cancel-" + std::to_string(suffix), options);
    registered_arena_t source(*endpoint.agent, endpoint.backend, descriptor_bytes, 1);
    registered_arena_t destination(*endpoint.agent, endpoint.backend, descriptor_bytes, 1);
    const population_spec_t spec{
        .name = "cancellation",
        .descriptorCount = 1,
        .bytesPerDescriptor = descriptor_bytes,
        .seed = 71,
    };
    source.fill(spec);
    destination.clear(spec);
    nixlXferReqH *request = createRequest(*endpoint.agent,
                                          endpoint.backend,
                                          source,
                                          destination,
                                          spec,
                                          "qualification-cancel-" + std::to_string(suffix),
                                          nullptr,
                                          {});
    nixl::qualification::terminal_ucx_api_adapter_t adapter(*endpoint.agent, 4);
    terminal_event_inbox_t inbox(adapter);
    nixlTerminalEventSubscriptionH *subscription = nullptr;
    constexpr std::uint64_t cookie = 7001;
    requireStatus(adapter.subscribeTransfer(request, cookie, subscription),
                  NIXL_SUCCESS,
                  "subscribe cancellation transfer");
    const nixl_status_t cancel_status = adapter.release(subscription);
    require(cancel_status == NIXL_SUCCESS || cancel_status == NIXL_IN_PROG,
            "cancel armed transfer subscription failed");
    const nixl_terminal_event_t event = inbox.take(nixl_terminal_event_kind_t::TRANSFER, cookie);
    if (cancel_status == NIXL_IN_PROG) {
        requireStatus(adapter.release(subscription),
                      NIXL_SUCCESS,
                      "release terminal cancellation subscription");
    }
    requireStatus(
        endpoint.agent->releaseXferReq(request), NIXL_SUCCESS, "release canceled transfer request");
    requireStatus(adapter.close(), NIXL_SUCCESS, "close cancellation channel");
    return {
        .terminalStatus = event.transferStatus,
        .terminalEventCount = 1,
        .ownerWoken = true,
    };
}

[[nodiscard]] shutdown_result_t
runShutdownCancellation(const options_t &options, std::uint64_t suffix) {
    endpoint_t endpoint = makeEndpoint("qualification-shutdown-" + std::to_string(suffix), options);
    registered_arena_t source(*endpoint.agent, endpoint.backend, descriptor_bytes, 1);
    registered_arena_t destination(*endpoint.agent, endpoint.backend, descriptor_bytes, 1);
    const population_spec_t spec{
        .name = "shutdown",
        .descriptorCount = 1,
        .bytesPerDescriptor = descriptor_bytes,
        .seed = 103,
    };
    nixlXferReqH *request = createRequest(*endpoint.agent,
                                          endpoint.backend,
                                          source,
                                          destination,
                                          spec,
                                          "qualification-shutdown-" + std::to_string(suffix),
                                          nullptr,
                                          {});
    nixl::qualification::terminal_ucx_api_adapter_t adapter(*endpoint.agent, 4);
    terminal_event_inbox_t inbox(adapter);
    nixlTerminalEventSubscriptionH *subscription = nullptr;
    constexpr std::uint64_t cookie = 7101;
    requireStatus(adapter.subscribeTransfer(request, cookie, subscription),
                  NIXL_SUCCESS,
                  "subscribe shutdown transfer");
    nixl_status_t cancel_status = adapter.release(subscription);
    require(cancel_status == NIXL_SUCCESS || cancel_status == NIXL_IN_PROG,
            "shutdown cancellation failed");
    const nixl_terminal_event_t event = inbox.take(nixl_terminal_event_kind_t::TRANSFER, cookie);
    if (cancel_status == NIXL_IN_PROG) {
        cancel_status = adapter.release(subscription);
    }
    requireStatus(cancel_status, NIXL_SUCCESS, "drain shutdown cancellation");
    requireStatus(
        endpoint.agent->releaseXferReq(request), NIXL_SUCCESS, "release shutdown request");
    requireStatus(adapter.close(), NIXL_SUCCESS, "close shutdown channel");
    nixl::qualification::terminal_channel_inventory_t inventory;
    requireStatus(
        adapter.queryInventory(inventory), NIXL_SUCCESS, "query shutdown cancellation inventory");
    const bool drained = inventory.backendProducers == 0 && inventory.activeCallbackSlots == 0 &&
        inventory.queuedOwnerContinuations == 0 && inventory.activeChannelSubscriptions == 0 &&
        inventory.retainedPublicSubscriptions == 0;
    shutdown_result_t result;
    result.terminalStatus = event.transferStatus;
    result.terminalEventCount = 1;
    result.ownerWoken = true;
    result.cancelStatus = cancel_status;
    result.drained = drained;
    return result;
}

[[nodiscard]] queue_overflow_result_t
runQueueOverflow(const options_t &options, std::uint64_t suffix) {
    const std::string name = "qualification-overflow-" + std::to_string(suffix);
    endpoint_t endpoint = makeEndpoint(name, options);
    registered_arena_t source(*endpoint.agent, endpoint.backend, descriptor_bytes, 1);
    registered_arena_t destination(*endpoint.agent, endpoint.backend, descriptor_bytes, 1);
    const population_spec_t spec{
        .name = "overflow",
        .descriptorCount = 1,
        .bytesPerDescriptor = descriptor_bytes,
        .seed = 151,
    };
    nixl::qualification::terminal_ucx_api_adapter_t adapter(*endpoint.agent, 1);
    terminal_event_inbox_t inbox(adapter);
    std::array<nixlXferReqH *, 2> requests = {};
    std::array<nixlTerminalEventSubscriptionH *, 2> subscriptions = {};
    for (std::size_t index = 0; index < requests.size(); ++index) {
        requests[index] = createRequest(
            *endpoint.agent, endpoint.backend, source, destination, spec, name, nullptr, {});
        requireStatus(
            adapter.subscribeTransfer(requests[index], 7201 + index, subscriptions[index]),
            NIXL_SUCCESS,
            "subscribe overflow transfer");
    }
    for (nixlTerminalEventSubscriptionH *subscription : subscriptions) {
        const nixl_status_t status = adapter.release(subscription);
        require(status == NIXL_SUCCESS || status == NIXL_IN_PROG,
                "cancel overflow transfer failed");
    }
    const nixl_terminal_event_batch_t batch = inbox.drainAfterWake();
    nixl::qualification::terminal_channel_inventory_t inventory;
    requireStatus(adapter.queryInventory(inventory), NIXL_SUCCESS, "query overflow inventory");
    for (nixlXferReqH *request : requests) {
        requireStatus(
            endpoint.agent->releaseXferReq(request), NIXL_SUCCESS, "release overflow request");
    }
    requireStatus(adapter.close(), NIXL_SUCCESS, "close overflow channel");
    return {
        .fatal = inventory.fatal,
        .wakeCount = batch.wakeCount,
        .admittedEvents = batch.events.size(),
    };
}

[[nodiscard]] route_result_t
subscribeRoute(nixl::qualification::terminal_ucx_api_adapter_t &adapter,
               const nixlRemoteAgentH *remote_handle,
               nixlBackendH *backend,
               std::string name,
               std::uint64_t owner_cookie,
               bool &active_before_ready) {
    nixlTerminalEventSubscriptionH *subscription = nullptr;
    requireStatus(adapter.subscribeCapability(remote_handle, backend, owner_cookie, subscription),
                  NIXL_SUCCESS,
                  "subscribe remote capability");
    require(subscription != nullptr, "capability subscription is null");
    nixl_terminal_subscription_info_t info;
    requireStatus(adapter.querySubscription(subscription, info),
                  NIXL_SUCCESS,
                  "query capability subscription");
    active_before_ready = info.active;
    return {
        .name = std::move(name),
        .identity = info.identity,
        .generation = info.generation,
        .releaseStatus = NIXL_IN_PROG,
        .subscription = subscription,
    };
}

void
appendCapabilityEvent(route_result_t &route,
                      terminal_event_inbox_t &inbox,
                      std::uint64_t owner_cookie) {
    const nixl_terminal_event_t event =
        inbox.take(nixl_terminal_event_kind_t::CAPABILITY, owner_cookie);
    require(event.identity == route.identity && event.generation == route.generation,
            "capability event exact route changed");
    route.states.push_back(event.capabilityState);
    route.epochs.push_back(event.capabilityEpoch);
}

[[nodiscard]] capability_result_t
runCapabilities(const options_t &options,
                nixlAgent &source,
                nixlBackendH *source_backend,
                nixl::qualification::terminal_ucx_api_adapter_t &adapter,
                terminal_event_inbox_t &inbox,
                std::uint64_t suffix,
                const nixl::qualification::tcp_peer_capability_observation_t &endpoint_failure) {
    capability_result_t result;
    nixl_blob_t source_metadata;
    requireStatus(source.getLocalMD(source_metadata), NIXL_SUCCESS, "get source metadata");

    endpoint_t epoch = makeEndpoint("qualification-epoch-" + std::to_string(suffix), options);
    nixlRemoteAgentH *epoch_handle = loadRemote(source, *epoch.agent);
    constexpr std::uint64_t epoch_cookie = 8001;
    bool epoch_active_before_ready = false;
    route_result_t epoch_route = subscribeRoute(adapter,
                                                epoch_handle,
                                                source_backend,
                                                "epoch_advance",
                                                epoch_cookie,
                                                epoch_active_before_ready);
    result.subscribeBeforeReady = epoch_active_before_ready;
    nixlRemoteAgentH *epoch_reverse = nullptr;
    requireStatus(epoch.agent->loadRemoteMD(source_metadata, epoch_reverse),
                  NIXL_SUCCESS,
                  "load epoch reverse route");
    appendCapabilityEvent(epoch_route, inbox, epoch_cookie);
    require(epoch_route.states.back() == nixl_terminal_capability_state_t::READY,
            "epoch route did not become ready");
    requireStatus(
        epoch.agent->invalidateRemoteMD(epoch_reverse), NIXL_SUCCESS, "retire epoch reverse route");
    epoch_reverse = nullptr;
    requireStatus(epoch.agent->loadRemoteMD(source_metadata, epoch_reverse),
                  NIXL_SUCCESS,
                  "reload epoch reverse route");
    appendCapabilityEvent(epoch_route, inbox, epoch_cookie);
    require(epoch_route.states.back() == nixl_terminal_capability_state_t::READY &&
                epoch_route.epochs.back() > epoch_route.epochs.front(),
            "epoch route did not advance readiness");

    epoch_route.releaseStatus = adapter.release(epoch_route.subscription);
    requireStatus(epoch_route.releaseStatus, NIXL_SUCCESS, "release epoch capability");

    route_result_t failed_route{
        .name = "endpoint_failure",
        .identity = endpoint_failure.handleIdentity,
        .generation = endpoint_failure.handleGeneration,
        .states = endpoint_failure.states,
        .epochs = endpoint_failure.epochs,
        .releaseStatus = endpoint_failure.releaseStatus,
        .subscriptionTerminal = endpoint_failure.subscriptionTerminal,
    };

    endpoint_t retired = makeEndpoint("qualification-retired-" + std::to_string(suffix), options);
    nixlRemoteAgentH *retired_handle = loadRemote(source, *retired.agent);
    constexpr std::uint64_t retirement_transition_cookie = 8099;
    bool retirement_active_before_ready = false;
    route_result_t retirement_transition = subscribeRoute(adapter,
                                                          retired_handle,
                                                          source_backend,
                                                          "retirement-transition",
                                                          retirement_transition_cookie,
                                                          retirement_active_before_ready);
    nixlRemoteAgentH *retired_reverse = nullptr;
    requireStatus(retired.agent->loadRemoteMD(source_metadata, retired_reverse),
                  NIXL_SUCCESS,
                  "load retirement reverse metadata");
    appendCapabilityEvent(retirement_transition, inbox, retirement_transition_cookie);
    require(retirement_transition.states.back() == nixl_terminal_capability_state_t::READY,
            "retirement route did not become ready");
    requireStatus(adapter.release(retirement_transition.subscription),
                  NIXL_SUCCESS,
                  "release retirement transition subscription");

    constexpr std::uint64_t retirement_cookie = 8003;
    bool retirement_snapshot_active = false;
    route_result_t retired_route = subscribeRoute(adapter,
                                                  retired_handle,
                                                  source_backend,
                                                  "retirement",
                                                  retirement_cookie,
                                                  retirement_snapshot_active);
    appendCapabilityEvent(retired_route, inbox, retirement_cookie);
    result.snapshotAfterReady = retirement_snapshot_active &&
        retired_route.states.back() == nixl_terminal_capability_state_t::READY;
    requireStatus(source.invalidateRemoteMD(retired_handle), NIXL_SUCCESS, "retire exact route");
    appendCapabilityEvent(retired_route, inbox, retirement_cookie);
    require(retired_route.states.back() == nixl_terminal_capability_state_t::RETIRED,
            "retirement route did not publish retired");
    nixl_terminal_subscription_info_t retired_info;
    requireStatus(adapter.querySubscription(retired_route.subscription, retired_info),
                  NIXL_SUCCESS,
                  "query retired subscription");
    retired_route.subscriptionTerminal = !retired_info.active;
    retired_route.releaseStatus = adapter.release(retired_route.subscription);
    requireStatus(retired_route.releaseStatus, NIXL_SUCCESS, "release retired capability");

    result.routes = {
        std::move(epoch_route),
        std::move(failed_route),
        std::move(retired_route),
    };
    for (route_result_t &route : result.routes) {
        route.subscription = nullptr;
    }
    return result;
}

struct ready_route_t {
    nixlRemoteAgentH *sourceHandle = nullptr;
    nixlRemoteAgentH *destinationHandle = nullptr;
};

[[nodiscard]] ready_route_t
makeReadyRoute(nixlAgent &source,
               nixlBackendH *source_backend,
               nixlAgent &destination,
               nixl::qualification::terminal_ucx_api_adapter_t &adapter,
               terminal_event_inbox_t &inbox,
               std::uint64_t owner_cookie) {
    ready_route_t route;
    route.sourceHandle = loadRemote(source, destination);
    bool active_before_ready = false;
    route_result_t readiness = subscribeRoute(adapter,
                                              route.sourceHandle,
                                              source_backend,
                                              "readiness",
                                              owner_cookie,
                                              active_before_ready);
    route.destinationHandle = loadRemote(destination, source);
    appendCapabilityEvent(readiness, inbox, owner_cookie);
    require(readiness.states.size() == 1 &&
                readiness.states.front() == nixl_terminal_capability_state_t::READY,
            "TCP route did not become ready");
    requireStatus(
        adapter.release(readiness.subscription), NIXL_SUCCESS, "release readiness subscription");
    return route;
}

[[nodiscard]] std::size_t
takeAttachedNotification(nixlAgent &destination,
                         const nixlRemoteAgentH *source_handle,
                         std::string_view expected) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(event_timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        nixl_remote_notifs_t notifications;
        requireStatus(destination.getRemoteNotifs(notifications),
                      NIXL_SUCCESS,
                      "drain authenticated notifications");
        const auto found = notifications.find(source_handle);
        if (found == notifications.end()) {
            std::this_thread::yield();
            continue;
        }
        require(found->second.size() == 1 && found->second.front() == expected,
                "attached authenticated notification changed identity or payload");
        return found->second.size();
    }
    throw qualification_error("attached authenticated notification did not arrive");
}

[[nodiscard]] std::string
jsonEscape(std::string_view value) {
    std::ostringstream output;
    for (const unsigned char character : value) {
        switch (character) {
        case '"':
            output << "\\\"";
            break;
        case '\\':
            output << "\\\\";
            break;
        case '\b':
            output << "\\b";
            break;
        case '\f':
            output << "\\f";
            break;
        case '\n':
            output << "\\n";
            break;
        case '\r':
            output << "\\r";
            break;
        case '\t':
            output << "\\t";
            break;
        default:
            if (character < 0x20U) {
                output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<unsigned int>(character) << std::dec;
            } else {
                output << character;
            }
        }
    }
    return output.str();
}

void
writeString(std::ostream &output, std::string_view value) {
    output << '"' << jsonEscape(value) << '"';
}

[[nodiscard]] const char *
attestationStateName(nixl_xfer_attestation_state_t state) {
    switch (state) {
    case nixl_xfer_attestation_state_t::PREPARED:
        return "PREPARED";
    case nixl_xfer_attestation_state_t::POSTING:
        return "POSTING";
    case nixl_xfer_attestation_state_t::IN_PROGRESS:
        return "IN_PROGRESS";
    case nixl_xfer_attestation_state_t::REMOTE_FLUSHED:
        return "REMOTE_FLUSHED";
    case nixl_xfer_attestation_state_t::FAILED:
        return "FAILED";
    }
    return "UNKNOWN";
}

[[nodiscard]] const char *
capabilityStateName(nixl_terminal_capability_state_t state) {
    switch (state) {
    case nixl_terminal_capability_state_t::READY:
        return "READY";
    case nixl_terminal_capability_state_t::FAILED:
        return "FAILED";
    case nixl_terminal_capability_state_t::RETIRED:
        return "RETIRED";
    }
    return "UNKNOWN";
}

[[nodiscard]] const char *
fatalName(std::uint32_t fatal) {
    if (fatal == static_cast<std::uint32_t>(nixl_terminal_channel_fatal_t::NONE)) {
        return "NONE";
    }
    if (fatal == static_cast<std::uint32_t>(nixl_terminal_channel_fatal_t::QUEUE_OVERFLOW)) {
        return "QUEUE_OVERFLOW";
    }
    if (fatal == static_cast<std::uint32_t>(nixl_terminal_channel_fatal_t::EVENTFD_FAILURE)) {
        return "EVENTFD_FAILURE";
    }
    if (fatal ==
        static_cast<std::uint32_t>(nixl_terminal_channel_fatal_t::ACTIVE_SUBSCRIPTIONS_ON_CLOSE)) {
        return "ACTIVE_SUBSCRIPTIONS_ON_CLOSE";
    }
    if (fatal == static_cast<std::uint32_t>(nixl_terminal_channel_fatal_t::INVALID_PUBLICATION)) {
        return "INVALID_PUBLICATION";
    }
    return "MULTIPLE";
}

void
writeRuntimeArtifacts(std::ostream &output, const std::vector<nixl_runtime_artifact_t> &artifacts) {
    output << '[';
    for (std::size_t index = 0; index < artifacts.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        const nixl_runtime_artifact_t &artifact = artifacts[index];
        output << "{\"component\":";
        writeString(output, artifact.component);
        output << ",\"path\":";
        writeString(output, artifact.path);
        output << ",\"build_id\":";
        writeString(output, artifact.buildId);
        output << ",\"version\":";
        writeString(output, artifact.version);
        output << '}';
    }
    output << ']';
}

void
writeSelectedTransports(std::ostream &output, const nixl_xfer_attestation_t &attestation) {
    std::set<std::string> transports;
    for (const nixl_xfer_attestation_segment_t &segment : attestation.segments) {
        for (const nixl_xfer_attestation_transport_t &transport : segment.selectedTransports) {
            transports.insert(transport.transport);
        }
    }
    output << '[';
    std::size_t index = 0;
    for (const std::string &transport : transports) {
        if (index++ != 0) {
            output << ',';
        }
        writeString(output, transport);
    }
    output << ']';
}

void
writeTerminalProgress(std::ostream &output, const nixl_xfer_terminal_progress_t &progress) {
    output << "{\"autonomous\":" << (progress.autonomous ? "true" : "false")
           << ",\"data_callbacks\":" << progress.dataCallbacks
           << ",\"endpoint_flush_callbacks\":" << progress.endpointFlushCallbacks
           << ",\"notification_callbacks\":" << progress.notificationCallbacks
           << ",\"asynchronous_requests\":" << progress.asynchronousRequests
           << ",\"immediate_completions\":" << progress.immediateCompletions
           << ",\"callbacks_before_poster_return\":" << progress.callbacksBeforePosterReturn
           << ",\"peak_continuation_depth\":" << progress.peakContinuationDepth
           << ",\"active_callback_slots_at_terminal\":" << progress.activeCallbackSlotsAtTerminal
           << ",\"continuation_depth_at_terminal\":" << progress.continuationDepthAtTerminal
           << ",\"last_data_callback_timestamp_ns\":" << progress.lastDataCallbackTimestampNs
           << ",\"last_flush_callback_timestamp_ns\":" << progress.lastFlushCallbackTimestampNs
           << ",\"notification_callback_timestamp_ns\":" << progress.notificationCallbackTimestampNs
           << ",\"terminal_publish_timestamp_ns\":" << progress.terminalPublishTimestampNs
           << ",\"terminal_status\":";
    writeString(output, nixlEnumStrings::statusStr(progress.terminalStatus));
    output << '}';
}

void
writePopulation(std::ostream &output, const population_result_t &result) {
    output << "{\"population\":";
    writeString(output, result.spec.name);
    output << ",\"descriptor_count\":" << result.spec.descriptorCount
           << ",\"byte_count\":" << result.byteCount
           << ",\"destination_byte_count\":" << result.destinationByteCount
           << ",\"source_sha256\":";
    writeString(output, result.sourceSha256);
    output << ",\"destination_sha256\":";
    writeString(output, result.destinationSha256);
    output << ",\"bytes_verified\":"
           << (result.sourceSha256 == result.destinationSha256 ? "true" : "false")
           << ",\"terminal_status\":";
    writeString(output, nixlEnumStrings::statusStr(result.terminalStatus));
    output << ",\"terminal_event_count\":" << result.terminalEventCount
           << ",\"attestation_state\":";
    writeString(output, attestationStateName(result.attestation.state));
    output << ",\"attestation_status\":";
    writeString(output, nixlEnumStrings::statusStr(result.attestationStatus));
    output << ",\"attestation_sha256\":";
    writeString(output, result.attestation.evidenceDigest);
    output << ",\"completion_claimed\":"
           << (result.attestation.completionClaimed ? "true" : "false")
           << ",\"take_once_second_status\":";
    writeString(output, nixlEnumStrings::statusStr(result.secondTakeStatus));
    output << ",\"selected_transports\":";
    writeSelectedTransports(output, result.attestation);
    output << ",\"subscription_before_post\":" << (result.subscriptionBeforePost ? "true" : "false")
           << ",\"event_native_timestamp_ns\":" << result.eventNativeTimestampNs
           << ",\"drain_timestamp_ns\":" << result.drainTimestampNs << ",\"terminal_progress\":";
    writeTerminalProgress(output, result.attestation.terminalProgress);
    output << '}';
}

void
writeNotApplicable(std::ostream &output) {
    output << "{\"applicability\":\"not_applicable\",\"reason\":";
    writeString(output, self_na_reason);
    output << ",\"evidence_anchors\":["
              "\"ucx/src/uct/sm/self/self.c:144\","
              "\"ucx/src/ucp/core/ucp_ep.c:1098\","
              "\"nixl/src/core/nixl_agent.cpp:2449\"]}";
}

void
writeRegistration(std::ostream &output, const registration_result_t &registration) {
    output << "{\"status\":";
    writeString(output, nixlEnumStrings::statusStr(registration.status));
    output << ",\"memory_type\":\"DRAM\",\"base_address\":" << registration.baseAddress
           << ",\"byte_capacity\":" << registration.byteCapacity << '}';
}

void
writeFault(std::ostream &output, const terminal_fault_result_t &fault) {
    output << "{\"applicability\":\"applicable\",\"terminal_status\":";
    writeString(output, nixlEnumStrings::statusStr(fault.terminalStatus));
    output << ",\"terminal_event_count\":" << fault.terminalEventCount
           << ",\"owner_woken\":" << (fault.ownerWoken ? "true" : "false") << '}';
}

void
writeShutdownInventory(std::ostream &output,
                       const nixl::qualification::terminal_channel_inventory_t &inventory) {
    output << "{\"capacity\":" << inventory.capacity
           << ",\"queued_channel_events\":" << inventory.queuedChannelEvents
           << ",\"active_channel_subscriptions\":" << inventory.activeChannelSubscriptions
           << ",\"retained_public_subscriptions\":" << inventory.retainedPublicSubscriptions
           << ",\"backend_producers\":" << inventory.backendProducers
           << ",\"active_callback_slots\":" << inventory.activeCallbackSlots
           << ",\"queued_owner_continuations\":" << inventory.queuedOwnerContinuations
           << ",\"accepting_subscriptions\":"
           << (inventory.acceptingSubscriptions ? "true" : "false")
           << ",\"closed\":" << (inventory.closed ? "true" : "false") << ",\"fatal\":";
    writeString(output, fatalName(inventory.fatal));
    output << ",\"eventfd_error\":" << inventory.eventfdError << '}';
}

void
writeCapability(std::ostream &output, bool self_transport, const capability_result_t &capability) {
    if (self_transport) {
        writeNotApplicable(output);
        return;
    }
    output << "{\"applicability\":\"applicable\",\"subscribe_before_ready\":"
           << (capability.subscribeBeforeReady ? "true" : "false")
           << ",\"snapshot_after_ready\":" << (capability.snapshotAfterReady ? "true" : "false")
           << ",\"routes\":[";
    for (std::size_t route_index = 0; route_index < capability.routes.size(); ++route_index) {
        if (route_index != 0) {
            output << ',';
        }
        const route_result_t &route = capability.routes[route_index];
        output << "{\"name\":";
        writeString(output, route.name);
        output << ",\"handle_identity\":" << route.identity
               << ",\"handle_generation\":" << route.generation << ",\"states\":[";
        for (std::size_t state_index = 0; state_index < route.states.size(); ++state_index) {
            if (state_index != 0) {
                output << ',';
            }
            writeString(output, capabilityStateName(route.states[state_index]));
        }
        output << "],\"epochs\":[";
        for (std::size_t epoch_index = 0; epoch_index < route.epochs.size(); ++epoch_index) {
            if (epoch_index != 0) {
                output << ',';
            }
            output << route.epochs[epoch_index];
        }
        output << "],\"subscription_terminal\":" << (route.subscriptionTerminal ? "true" : "false")
               << ",\"release_status\":";
        writeString(output, nixlEnumStrings::statusStr(route.releaseStatus));
        output << '}';
    }
    output << "]}";
}

void
writeCoordinate(const options_t &options,
                const registration_result_t &source_registration,
                const registration_result_t &destination_registration,
                const std::vector<population_result_t> &populations,
                const capability_result_t &capability,
                std::size_t notification_success_count,
                const terminal_fault_result_t &cancellation,
                const queue_overflow_result_t &overflow,
                const shutdown_result_t &shutdown_cancellation,
                const tcp_fault_results_t &tcp_faults,
                const nixl::qualification::terminal_channel_inventory_t &inventory) {
    require(!populations.empty(), "coordinate has no completion population");
    std::ofstream output(options.output);
    if (!output) {
        throw qualification_error("failed to open coordinate receipt");
    }
    const bool self_transport = options.transport == "self";
    output << "{\"transport\":";
    writeString(output, options.transport);
    output << ",\"engine\":";
    writeString(output, options.engine);
    output << ",\"agent_shape\":";
    writeString(output, self_transport ? "one_agent_local_route" : "two_distinct_agents");
    output << ",\"agent_count\":" << (self_transport ? 1 : 2)
           << ",\"remote_agent_handle_present\":" << (self_transport ? "false" : "true")
           << ",\"registrations\":{\"source\":";
    writeRegistration(output, source_registration);
    output << ",\"destination\":";
    writeRegistration(output, destination_registration);
    output << "},\"completion_populations\":[";
    for (std::size_t index = 0; index < populations.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        writePopulation(output, populations[index]);
    }
    output << "],\"remote_route_capability\":";
    writeCapability(output, self_transport, capability);
    output << ",\"attached_authenticated_notification\":";
    if (self_transport) {
        writeNotApplicable(output);
    } else {
        output << "{\"applicability\":\"applicable\",\"success_count\":"
               << notification_success_count << '}';
    }
    output << ",\"faults\":{\"transfer_cancellation\":";
    writeFault(output, cancellation);
    output << ",\"queue_overflow\":{\"applicability\":\"applicable\",\"fatal\":";
    writeString(output, fatalName(overflow.fatal));
    output << ",\"owner_woken\":" << (overflow.wakeCount > 0 ? "true" : "false")
           << ",\"admitted_event_preserved\":" << (overflow.admittedEvents == 1 ? "true" : "false")
           << "},\"shutdown_cancellation_drain\":";
    output << "{\"applicability\":\"applicable\",\"terminal_status\":";
    writeString(output, nixlEnumStrings::statusStr(shutdown_cancellation.terminalStatus));
    output << ",\"terminal_event_count\":" << shutdown_cancellation.terminalEventCount
           << ",\"owner_woken\":" << (shutdown_cancellation.ownerWoken ? "true" : "false")
           << ",\"cancel_status\":";
    writeString(output, nixlEnumStrings::statusStr(shutdown_cancellation.cancelStatus));
    output << ",\"posted_in_flight\":" << (shutdown_cancellation.postedInFlight ? "true" : "false")
           << ",\"backend_producers_before_cancel\":"
           << shutdown_cancellation.backendProducersBeforeCancel
           << ",\"active_callback_slots_before_cancel\":"
           << shutdown_cancellation.activeCallbackSlotsBeforeCancel
           << ",\"queued_owner_continuations_before_cancel\":"
           << shutdown_cancellation.queuedOwnerContinuationsBeforeCancel
           << ",\"drained\":" << (shutdown_cancellation.drained ? "true" : "false")
           << "},\"remote_failure\":";
    if (self_transport) {
        writeNotApplicable(output);
    } else {
        writeFault(output, tcp_faults.remoteFailure);
    }
    output << ",\"notification_failure\":";
    if (self_transport) {
        writeNotApplicable(output);
    } else {
        output << "{\"applicability\":\"applicable\",\"terminal_status\":";
        writeString(output,
                    nixlEnumStrings::statusStr(tcp_faults.notificationFailure.terminalStatus));
        output << ",\"terminal_event_count\":" << tcp_faults.notificationFailure.terminalEventCount
               << ",\"owner_woken\":"
               << (tcp_faults.notificationFailure.ownerWoken ? "true" : "false")
               << ",\"data_remote_flushed_before_failure\":"
               << (tcp_faults.dataRemoteFlushedBeforeNotificationFailure ? "true" : "false") << '}';
    }
    output << "},\"runtime_artifacts\":";
    writeRuntimeArtifacts(output, populations.front().attestation.runtimeArtifacts);
    output << ",\"shutdown\":";
    writeShutdownInventory(output, inventory);
    output << "}\n";
}

int
run(int argc, char **argv) {
    const options_t options = parseOptions(argc, argv);
    const std::uint64_t suffix = static_cast<std::uint64_t>(getpid());
    const std::string source_name = "qualification-source-" + std::to_string(suffix);
    endpoint_t source = makeEndpoint(source_name, options);
    endpoint_t destination;
    if (options.transport == "self") {
        destination.backend = source.backend;
    } else {
        destination = makeEndpoint("qualification-destination-" + std::to_string(suffix), options);
    }
    nixlAgent &destination_agent = options.transport == "self" ? *source.agent : *destination.agent;
    nixlBackendH *destination_backend =
        options.transport == "self" ? source.backend : destination.backend;
    registered_arena_t source_arena(
        *source.agent, source.backend, arena_bytes, large_descriptor_count);
    registered_arena_t destination_arena(
        destination_agent, destination_backend, arena_bytes, large_descriptor_count);

    nixlRemoteAgentH *destination_handle = nullptr;
    nixlRemoteAgentH *source_handle = nullptr;
    nixl::qualification::terminal_ucx_api_adapter_t adapter(*source.agent, channel_capacity);
    terminal_event_inbox_t inbox(adapter);
    capability_result_t capability;
    tcp_fault_results_t tcp_faults;
    std::optional<nixl::qualification::tcp_shutdown_cancellation_observation_t>
        tcp_shutdown_cancellation;
    if (options.transport == "tcp") {
        ready_route_t route =
            makeReadyRoute(*source.agent, source.backend, destination_agent, adapter, inbox, 9001);
        destination_handle = route.sourceHandle;
        source_handle = route.destinationHandle;
    }

    const std::vector<population_spec_t> specs = {
        {
            .name = "small",
            .descriptorCount = 1,
            .bytesPerDescriptor = 1024,
            .seed = 17,
        },
        {
            .name = "large",
            .descriptorCount = large_descriptor_count,
            .bytesPerDescriptor = descriptor_bytes,
            .seed = 93,
        },
    };
    std::vector<population_result_t> populations;
    populations.reserve(specs.size());
    std::size_t notification_success_count = 0;
    for (std::size_t index = 0; index < specs.size(); ++index) {
        const std::string notification = options.transport == "tcp" ?
            "terminal-qualification:" + specs[index].name :
            std::string{};
        populations.push_back(runPopulation(*source.agent,
                                            source.backend,
                                            source_name,
                                            destination_handle,
                                            adapter,
                                            inbox,
                                            source_arena,
                                            destination_arena,
                                            specs[index],
                                            100 + index,
                                            notification));
        if (options.transport == "tcp") {
            notification_success_count +=
                takeAttachedNotification(destination_agent, source_handle, notification);
        }
    }

    if (options.transport == "tcp") {
        requireStatus(source.agent->invalidateRemoteMD(destination_handle),
                      NIXL_SUCCESS,
                      "retire population route");
        const tcp_fixture_result_t fixtures{
            .endpointFailure = nixl::qualification::runTcpEndpointFailureFixture(options.engine),
            .notificationFailure =
                nixl::qualification::runTcpNotificationFailureFixture(options.engine),
            .shutdownCancellation =
                nixl::qualification::runTcpShutdownCancellationFixture(options.engine),
        };
        require(fixtures.endpointFailure.peerExitedBySignal,
                "endpoint-failure peer did not die independently");
        require(fixtures.notificationFailure.peerExitedBySignal,
                "notification-failure peer did not die independently");
        require(fixtures.shutdownCancellation.peerExitedBySignal,
                "shutdown-cancellation peer did not die independently");
        capability = runCapabilities(options,
                                     *source.agent,
                                     source.backend,
                                     adapter,
                                     inbox,
                                     suffix,
                                     fixtures.endpointFailure.capability);
        tcp_faults.remoteFailure = {
            .terminalStatus = fixtures.endpointFailure.transfer.terminalStatus,
            .terminalEventCount = fixtures.endpointFailure.transfer.terminalEventCount,
            .ownerWoken = fixtures.endpointFailure.transfer.ownerWoken,
        };
        tcp_faults.notificationFailure = {
            .terminalStatus = fixtures.notificationFailure.transfer.terminalStatus,
            .terminalEventCount = fixtures.notificationFailure.transfer.terminalEventCount,
            .ownerWoken = fixtures.notificationFailure.transfer.ownerWoken,
        };
        tcp_faults.dataRemoteFlushedBeforeNotificationFailure =
            fixtures.notificationFailure.dataRemoteFlushedBeforeFailure;
        tcp_shutdown_cancellation = fixtures.shutdownCancellation;
    }

    const terminal_fault_result_t cancellation = runCancellation(options, suffix);
    requireStatus(cancellation.terminalStatus, NIXL_ERR_CANCELED, "cancellation terminal result");
    const queue_overflow_result_t overflow = runQueueOverflow(options, suffix);
    require(overflow.fatal ==
                    static_cast<std::uint32_t>(nixl_terminal_channel_fatal_t::QUEUE_OVERFLOW) &&
                overflow.admittedEvents == 1,
            "queue overflow did not preserve exactly one admitted event");
    shutdown_result_t shutdown_cancellation = runShutdownCancellation(options, suffix);
    if (tcp_shutdown_cancellation.has_value()) {
        shutdown_result_t observed;
        observed.terminalStatus = tcp_shutdown_cancellation->transfer.terminalStatus;
        observed.terminalEventCount = tcp_shutdown_cancellation->transfer.terminalEventCount;
        observed.ownerWoken = tcp_shutdown_cancellation->transfer.ownerWoken;
        observed.cancelStatus = tcp_shutdown_cancellation->cancelStatus;
        observed.backendProducersBeforeCancel =
            tcp_shutdown_cancellation->inventoryBeforeCancellation.backendProducers;
        observed.activeCallbackSlotsBeforeCancel =
            tcp_shutdown_cancellation->inventoryBeforeCancellation.activeCallbackSlots;
        observed.queuedOwnerContinuationsBeforeCancel =
            tcp_shutdown_cancellation->inventoryBeforeCancellation.queuedOwnerContinuations;
        observed.postedInFlight = tcp_shutdown_cancellation->postedInFlight;
        observed.drained = tcp_shutdown_cancellation->drained;
        shutdown_cancellation = observed;
    }
    requireStatus(shutdown_cancellation.terminalStatus,
                  NIXL_ERR_CANCELED,
                  "shutdown cancellation terminal result");
    require(shutdown_cancellation.drained, "shutdown cancellation retained native lifecycle state");

    requireStatus(adapter.close(), NIXL_SUCCESS, "close primary terminal channel");
    nixl::qualification::terminal_channel_inventory_t inventory;
    requireStatus(
        adapter.queryInventory(inventory), NIXL_SUCCESS, "query primary terminal inventory");
    require(inventory.queuedChannelEvents == 0 && inventory.activeChannelSubscriptions == 0 &&
                inventory.retainedPublicSubscriptions == 0 && inventory.backendProducers == 0 &&
                inventory.activeCallbackSlots == 0 && inventory.queuedOwnerContinuations == 0 &&
                !inventory.acceptingSubscriptions && inventory.closed && inventory.fatal == 0 &&
                inventory.eventfdError == 0,
            "primary terminal channel did not shut down with zero inventory");

    writeCoordinate(options,
                    source_arena.registration(),
                    destination_arena.registration(),
                    populations,
                    capability,
                    notification_success_count,
                    cancellation,
                    overflow,
                    shutdown_cancellation,
                    tcp_faults,
                    inventory);
    return EXIT_SUCCESS;
}

} // namespace

int
main(int argc, char **argv) {
    if (nixl::qualification::isTerminalUcxPeerWorkerInvocation(argc, argv)) {
        return nixl::qualification::runTerminalUcxPeerWorker(argc, argv);
    }
    try {
        return run(argc, argv);
    }
    catch (const std::exception &error) {
        std::cerr << "terminal UCX qualification failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
