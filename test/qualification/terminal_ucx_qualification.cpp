/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "terminal_ucx_api_adapter.h"

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
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t channel_capacity = 64;
constexpr std::uint64_t capability_cookie = 1;
constexpr std::uint64_t first_transfer_cookie = 10;
constexpr int event_timeout_ms = 30000;
constexpr std::uint64_t device_id = 0;

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
    std::string completionMode;
    std::size_t byteCount = 0;
    std::string sourceSha256;
    std::string destinationSha256;
    nixl_status_t terminalStatus = NIXL_ERR_BACKEND;
    nixl_status_t attestationStatus = NIXL_ERR_BACKEND;
    nixl_status_t secondTakeStatus = NIXL_ERR_BACKEND;
    nixl_xfer_attestation_t attestation;
    std::uint64_t terminalNativeTimestampNs = 0;
    std::uint64_t drainTimestampNs = 0;
    bool callbackBeforeReturnObserved = false;
    bool subscriptionBeforePost = false;
    bool terminalAfterNotificationCompletion = false;
};

struct capability_result_t {
    std::vector<nixl_terminal_capability_state_t> states;
    std::vector<std::uint64_t> epochs;
    bool snapshotReady = false;
    bool retired = false;
};

class registered_buffers_t final {
public:
    registered_buffers_t(nixlAgent &agent,
                         nixlBackendH *backend,
                         const population_spec_t &spec,
                         bool source)
        : agent_(agent),
          backend_(backend),
          spec_(spec),
          source_(source) {
        buffers_.reserve(spec_.descriptorCount);
        for (std::size_t index = 0; index < spec_.descriptorCount; ++index) {
            buffers_.emplace_back(spec_.bytesPerDescriptor, 0);
            if (source_) {
                for (std::size_t offset = 0; offset < spec_.bytesPerDescriptor; ++offset) {
                    buffers_.back()[offset] =
                        static_cast<std::uint8_t>(spec_.seed + index * 17U + offset * 29U);
                }
            }
        }

        nixl_opt_args_t arguments;
        arguments.backends = {backend_};
        const nixl_status_t status = agent_.registerMem(registrationList(), &arguments);
        requireStatus(status, NIXL_SUCCESS, "register DRAM buffers");
        registered_ = true;
    }

    registered_buffers_t(const registered_buffers_t &) = delete;
    registered_buffers_t &
    operator=(const registered_buffers_t &) = delete;

    ~registered_buffers_t() {
        if (!registered_) {
            return;
        }
        nixl_opt_args_t arguments;
        arguments.backends = {backend_};
        static_cast<void>(agent_.deregisterMem(registrationList(), &arguments));
    }

    [[nodiscard]] nixl_reg_dlist_t
    registrationList() const {
        nixl_reg_dlist_t descriptors(DRAM_SEG);
        for (const auto &buffer : buffers_) {
            descriptors.addDesc(nixlBlobDesc(
                reinterpret_cast<std::uintptr_t>(buffer.data()), buffer.size(), device_id));
        }
        return descriptors;
    }

    [[nodiscard]] nixl_xfer_dlist_t
    transferList() const {
        nixl_xfer_dlist_t descriptors(DRAM_SEG);
        for (const auto &buffer : buffers_) {
            descriptors.addDesc(nixlBasicDesc(
                reinterpret_cast<std::uintptr_t>(buffer.data()), buffer.size(), device_id));
        }
        return descriptors;
    }

    [[nodiscard]] std::string
    sha256() const {
        std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(),
                                                                        &EVP_MD_CTX_free);
        if (context == nullptr || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
            throw qualification_error("failed to initialize SHA-256");
        }
        for (const auto &buffer : buffers_) {
            if (EVP_DigestUpdate(context.get(), buffer.data(), buffer.size()) != 1) {
                throw qualification_error("failed to update SHA-256");
            }
        }
        std::array<unsigned char, EVP_MAX_MD_SIZE> digest = {};
        unsigned int size = 0;
        if (EVP_DigestFinal_ex(context.get(), digest.data(), &size) != 1 || size != 32U) {
            throw qualification_error("failed to finish SHA-256");
        }
        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (unsigned int index = 0; index < size; ++index) {
            output << std::setw(2) << static_cast<unsigned int>(digest[index]);
        }
        return output.str();
    }

    [[nodiscard]] std::size_t
    byteCount() const noexcept {
        return spec_.descriptorCount * spec_.bytesPerDescriptor;
    }

private:
    static void
    requireStatus(nixl_status_t actual, nixl_status_t expected, std::string_view operation) {
        if (actual == expected) {
            return;
        }
        throw qualification_error(std::string(operation) + " failed with " +
                                  nixlEnumStrings::statusStr(actual));
    }

    nixlAgent &agent_;
    nixlBackendH *backend_;
    population_spec_t spec_;
    bool source_;
    std::vector<std::vector<std::uint8_t>> buffers_;
    bool registered_ = false;
};

[[nodiscard]] std::uint64_t
monotonicRawNs() {
    timespec timestamp = {};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp) != 0) {
        throw qualification_error("CLOCK_MONOTONIC_RAW failed");
    }
    return static_cast<std::uint64_t>(timestamp.tv_sec) * 1000000000ULL +
        static_cast<std::uint64_t>(timestamp.tv_nsec);
}

void
requireStatus(nixl_status_t actual, nixl_status_t expected, std::string_view operation) {
    if (actual == expected) {
        return;
    }
    throw qualification_error(std::string(operation) + " failed with " +
                              nixlEnumStrings::statusStr(actual));
}

void
require(bool condition, std::string_view message) {
    if (!condition) {
        throw qualification_error(std::string(message));
    }
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

[[nodiscard]] nixl_terminal_event_batch_t
waitAndDrain(nixl::qualification::terminal_ucx_api_adapter_t &adapter) {
    int fd = -1;
    requireStatus(adapter.fileno(fd), NIXL_SUCCESS, "get terminal event fd");
    pollfd descriptor = {.fd = fd, .events = POLLIN, .revents = 0};
    const int result = poll(&descriptor, 1, event_timeout_ms);
    if (result < 0) {
        throw qualification_error("terminal event poll failed");
    }
    require(result == 1 && (descriptor.revents & POLLIN) != 0,
            "terminal event poll timed out or returned no data");
    nixl_terminal_event_batch_t batch;
    requireStatus(adapter.drain(batch), NIXL_SUCCESS, "drain terminal events");
    require(!batch.events.empty(), "terminal wake produced an empty batch");
    return batch;
}

void
consumeCapabilityEvents(const nixl_terminal_event_batch_t &batch, capability_result_t &result) {
    for (const auto &event : batch.events) {
        if (event.kind != nixl_terminal_event_kind_t::CAPABILITY) {
            throw qualification_error("unexpected transfer event in capability batch");
        }
        result.states.push_back(event.capabilityState);
        result.epochs.push_back(event.capabilityEpoch);
    }
}

[[nodiscard]] population_result_t
runPopulation(nixlAgent &source_agent,
              nixlBackendH *source_backend,
              const nixlRemoteAgentH *destination_handle,
              nixl::qualification::terminal_ucx_api_adapter_t &adapter,
              const population_spec_t &spec,
              registered_buffers_t &source,
              registered_buffers_t &destination,
              std::uint64_t owner_cookie) {
    nixl_opt_args_t transfer_arguments;
    transfer_arguments.backends = {source_backend};
    transfer_arguments.notif = "terminal-qualification:" + spec.name;

    nixlXferReqH *request = nullptr;
    requireStatus(source_agent.createXferReq(NIXL_WRITE,
                                             source.transferList(),
                                             destination.transferList(),
                                             destination_handle,
                                             request,
                                             &transfer_arguments),
                  NIXL_SUCCESS,
                  "create transfer request");
    require(request != nullptr, "create transfer returned a null request");

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

    const nixl_terminal_event_batch_t batch = waitAndDrain(adapter);
    const std::uint64_t drain_timestamp_ns = monotonicRawNs();
    require(batch.events.size() == 1, "transfer emitted other than one terminal event");
    const nixl_terminal_event_t &event = batch.events.front();
    require(event.kind == nixl_terminal_event_kind_t::TRANSFER,
            "transfer subscription emitted a capability event");
    require(event.ownerCookie == owner_cookie, "transfer event owner cookie mismatch");

    requireStatus(adapter.querySubscription(subscription, subscription_info),
                  NIXL_SUCCESS,
                  "query terminal transfer subscription");
    require(!subscription_info.active, "terminal transfer subscription remained active");

    population_result_t result;
    result.completionMode = spec.name;
    result.byteCount = source.byteCount();
    result.sourceSha256 = source.sha256();
    result.terminalStatus = event.transferStatus;
    result.terminalNativeTimestampNs = event.nativeTimestampNs;
    result.drainTimestampNs = drain_timestamp_ns;
    result.subscriptionBeforePost = true;
    result.terminalAfterNotificationCompletion = true;
    result.callbackBeforeReturnObserved = false;
    result.attestationStatus =
        source_agent.takeXferCompletionAttestation(request, result.attestation);
    nixl_xfer_attestation_t second_attestation;
    result.secondTakeStatus =
        source_agent.takeXferCompletionAttestation(request, second_attestation);

    requireStatus(adapter.release(subscription), NIXL_SUCCESS, "release transfer subscription");
    requireStatus(source_agent.releaseXferReq(request), NIXL_SUCCESS, "release transfer request");

    result.destinationSha256 = destination.sha256();
    return result;
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

void
writeRuntimeArtifacts(std::ostream &output, const std::vector<nixl_runtime_artifact_t> &artifacts) {
    output << '[';
    for (std::size_t index = 0; index < artifacts.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        const auto &artifact = artifacts[index];
        output << '{';
        output << "\"component\":";
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
    for (const auto &segment : attestation.segments) {
        for (const auto &transport : segment.selectedTransports) {
            transports.insert(transport.transport);
        }
    }
    output << '[';
    std::size_t index = 0;
    for (const auto &transport : transports) {
        if (index++ != 0) {
            output << ',';
        }
        writeString(output, transport);
    }
    output << ']';
}

void
writePopulation(std::ostream &output, const population_result_t &result) {
    output << '{';
    output << "\"completion_mode\":";
    writeString(output, result.completionMode);
    output << ",\"memory_type\":\"DRAM\"";
    output << ",\"byte_count\":" << result.byteCount;
    output << ",\"source_sha256\":";
    writeString(output, result.sourceSha256);
    output << ",\"destination_sha256\":";
    writeString(output, result.destinationSha256);
    output << ",\"bytes_verified\":"
           << (result.sourceSha256 == result.destinationSha256 ? "true" : "false");
    output << ",\"terminal_status\":";
    writeString(output, nixlEnumStrings::statusStr(result.terminalStatus));
    output << ",\"attestation_state\":";
    writeString(output, attestationStateName(result.attestation.state));
    output << ",\"attestation_status\":";
    writeString(output, nixlEnumStrings::statusStr(result.attestationStatus));
    output << ",\"completion_claimed\":"
           << (result.attestation.completionClaimed ? "true" : "false");
    output << ",\"take_once_second_status\":";
    writeString(output, nixlEnumStrings::statusStr(result.secondTakeStatus));
    output << ",\"selected_transports\":";
    writeSelectedTransports(output, result.attestation);
    output << ",\"terminal_native_timestamp_ns\":" << result.terminalNativeTimestampNs;
    output << ",\"drain_timestamp_ns\":" << result.drainTimestampNs;
    output << ",\"subscription_before_post\":"
           << (result.subscriptionBeforePost ? "true" : "false");
    output << ",\"terminal_after_notification_completion\":"
           << (result.terminalAfterNotificationCompletion ? "true" : "false");
    output << ",\"callback_before_return_observed\":"
           << (result.callbackBeforeReturnObserved ? "true" : "false");
    output << '}';
}

void
writeCoordinate(const options_t &options,
                const std::vector<population_result_t> &populations,
                const capability_result_t &capabilities,
                const nixl::qualification::terminal_channel_inventory_t &inventory) {
    std::ofstream output(options.output);
    if (!output) {
        throw qualification_error("failed to open output receipt");
    }
    output << '{';
    output << "\"schema\":\"nixl-terminal-ucx-coordinate/v1\"";
    output << ",\"status\":\"pass\"";
    output << ",\"transport\":";
    writeString(output, options.transport);
    output << ",\"engine\":";
    writeString(output, options.engine);
    output << ",\"populations\":[";
    for (std::size_t index = 0; index < populations.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        writePopulation(output, populations[index]);
    }
    output << ']';
    output << ",\"runtime_artifacts\":";
    writeRuntimeArtifacts(output, populations.front().attestation.runtimeArtifacts);
    output << ",\"capability\":{\"states\":[";
    for (std::size_t index = 0; index < capabilities.states.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        writeString(output, capabilityStateName(capabilities.states[index]));
    }
    output << "],\"epochs\":[";
    for (std::size_t index = 0; index < capabilities.epochs.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << capabilities.epochs[index];
    }
    output << "],\"snapshot_ready\":" << (capabilities.snapshotReady ? "true" : "false");
    output << ",\"retired\":" << (capabilities.retired ? "true" : "false") << '}';
    output << ",\"shutdown\":{\"active_subscriptions\":" << inventory.activeSubscriptions;
    output << ",\"active_producers\":" << inventory.activeProducers;
    output << ",\"queued_events\":" << inventory.queuedEvents;
    output << ",\"channel_closed\":" << (inventory.closed ? "true" : "false");
    output << ",\"fatal_code\":" << inventory.fatal << "}}\n";
}

int
run(int argc, char **argv) {
    const options_t options = parseOptions(argc, argv);
    const std::string suffix = std::to_string(static_cast<unsigned long long>(getpid()));
    nixlAgent source_agent("qualification-source-" + suffix, agentConfig());
    nixlAgent destination_agent("qualification-destination-" + suffix, agentConfig());

    nixlBackendH *source_backend = nullptr;
    nixlBackendH *destination_backend = nullptr;
    const nixl_b_params_t parameters = backendParameters(options.transport, options.engine);
    requireStatus(source_agent.createBackend("UCX", parameters, source_backend),
                  NIXL_SUCCESS,
                  "create source UCX backend");
    requireStatus(destination_agent.createBackend("UCX", parameters, destination_backend),
                  NIXL_SUCCESS,
                  "create destination UCX backend");
    require(source_backend != nullptr && destination_backend != nullptr,
            "UCX backend handle is null");

    const std::size_t small_descriptor_count = options.engine == "thread_pool" ? 4 : 1;
    const std::vector<population_spec_t> specs = {
        {.name = "immediate",
         .descriptorCount = small_descriptor_count,
         .bytesPerDescriptor = 1024,
         .seed = 17},
        {.name = "asynchronous",
         .descriptorCount = options.engine == "thread_pool" ? 64U : 1U,
         .bytesPerDescriptor =
             options.engine == "thread_pool" ? 1024U * 1024U : 64U * 1024U * 1024U,
         .seed = 93},
    };

    std::vector<std::unique_ptr<registered_buffers_t>> source_buffers;
    std::vector<std::unique_ptr<registered_buffers_t>> destination_buffers;
    source_buffers.reserve(specs.size());
    destination_buffers.reserve(specs.size());
    for (const population_spec_t &spec : specs) {
        source_buffers.push_back(
            std::make_unique<registered_buffers_t>(source_agent, source_backend, spec, true));
        destination_buffers.push_back(std::make_unique<registered_buffers_t>(
            destination_agent, destination_backend, spec, false));
    }

    nixl_blob_t destination_metadata;
    requireStatus(destination_agent.getLocalMD(destination_metadata),
                  NIXL_SUCCESS,
                  "get destination metadata");
    nixlRemoteAgentH *destination_handle = nullptr;
    requireStatus(source_agent.loadRemoteMD(destination_metadata, destination_handle),
                  NIXL_SUCCESS,
                  "load destination metadata");
    require(destination_handle != nullptr, "destination remote handle is null");

    nixl::qualification::terminal_ucx_api_adapter_t adapter(source_agent, channel_capacity);
    nixlTerminalEventSubscriptionH *capability_subscription = nullptr;
    requireStatus(
        adapter.subscribeCapability(
            destination_handle, source_backend, capability_cookie, capability_subscription),
        NIXL_SUCCESS,
        "subscribe exact route capability");
    require(capability_subscription != nullptr, "capability subscription is null");

    capability_result_t capabilities;
    consumeCapabilityEvents(waitAndDrain(adapter), capabilities);
    capabilities.snapshotReady = capabilities.states.size() == 1 &&
        capabilities.states.front() == nixl_terminal_capability_state_t::READY;
    require(capabilities.snapshotReady, "capability subscription did not snapshot READY");

    std::vector<population_result_t> populations;
    populations.reserve(specs.size());
    for (std::size_t index = 0; index < specs.size(); ++index) {
        populations.push_back(runPopulation(source_agent,
                                            source_backend,
                                            destination_handle,
                                            adapter,
                                            specs[index],
                                            *source_buffers[index],
                                            *destination_buffers[index],
                                            first_transfer_cookie + index));
    }

    requireStatus(source_agent.invalidateRemoteMD(destination_handle),
                  NIXL_SUCCESS,
                  "retire exact remote route");
    consumeCapabilityEvents(waitAndDrain(adapter), capabilities);
    capabilities.retired = !capabilities.states.empty() &&
        capabilities.states.back() == nixl_terminal_capability_state_t::RETIRED;
    require(capabilities.retired, "capability subscription did not publish RETIRED");
    requireStatus(
        adapter.release(capability_subscription), NIXL_SUCCESS, "release capability subscription");

    requireStatus(adapter.close(), NIXL_SUCCESS, "close terminal channel");
    nixl::qualification::terminal_channel_inventory_t inventory;
    requireStatus(adapter.queryInventory(inventory), NIXL_SUCCESS, "query terminal inventory");
    require(inventory.activeSubscriptions == 0 && inventory.queuedEvents == 0 && inventory.closed &&
                inventory.fatal == 0,
            "terminal channel did not shut down cleanly");
    writeCoordinate(options, populations, capabilities, inventory);
    return EXIT_SUCCESS;
}

} // namespace

int
main(int argc, char **argv) {
    try {
        return run(argc, argv);
    }
    catch (const std::exception &error) {
        std::cerr << "terminal UCX qualification failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
