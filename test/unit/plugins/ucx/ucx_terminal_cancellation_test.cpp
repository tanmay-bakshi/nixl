/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "ucx_backend.h"

namespace {

using namespace std::chrono_literals;

void
require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

class recording_sink_t final : public nixlBackendTransferTransitionSink {
public:
    explicit recording_sink_t(bool cancel_on_publish = false)
        : cancelOnPublish_(cancel_on_publish) {}

    void
    bind(nixlBackendEventSubscription *subscription) {
        const std::lock_guard lock(mutex_);
        subscription_ = subscription;
    }

    void
    publish(const nixlBackendTransferTransition &transition) noexcept override {
        nixlBackendEventSubscription *subscription = nullptr;
        {
            const std::lock_guard lock(mutex_);
            subscription = subscription_;
        }

        nixl_status_t cancel_status = NIXL_ERR_NOT_READY;
        if (cancelOnPublish_ && subscription != nullptr) {
            cancel_status = subscription->cancel();
        }

        {
            const std::lock_guard lock(mutex_);
            transitions_.push_back(transition);
            cancelStatuses_.push_back(cancel_status);
        }
        delivered_.notify_all();
    }

    [[nodiscard]] bool
    waitForTransitions(std::size_t count) {
        std::unique_lock lock(mutex_);
        return delivered_.wait_for(
            lock, 10s, [this, count]() { return transitions_.size() >= count; });
    }

    [[nodiscard]] std::vector<nixlBackendTransferTransition>
    transitions() const {
        const std::lock_guard lock(mutex_);
        return transitions_;
    }

    [[nodiscard]] std::vector<nixl_status_t>
    cancelStatuses() const {
        const std::lock_guard lock(mutex_);
        return cancelStatuses_;
    }

private:
    const bool cancelOnPublish_;
    mutable std::mutex mutex_;
    std::condition_variable delivered_;
    nixlBackendEventSubscription *subscription_ = nullptr;
    std::vector<nixlBackendTransferTransition> transitions_;
    std::vector<nixl_status_t> cancelStatuses_;
};

class loopback_fixture_t final {
public:
    loopback_fixture_t()
        : local_(DRAM_SEG),
          remote_(DRAM_SEG) {
        nixlBackendInitParams init;
        init.enableProgTh = true;
        init.pthrDelay = 1;
        init.localAgent = agentName_;
        init.localAgentIncarnation = "00000000-0000-4000-8000-000000000001";
        init.customParams = &customParams_;
        init.type = "UCX";

        engine_ = nixlUcxEngine::create(init);
        require(engine_ != nullptr && !engine_->getInitErr(),
                "real UCX loopback engine initialization failed");

        std::string connection_info;
        require(engine_->getConnInfo(connection_info) == NIXL_SUCCESS,
                "real UCX loopback connection export failed");
        require(engine_->loadRemoteConnInfo(agentName_, connection_info) == NIXL_SUCCESS,
                "real UCX loopback connection import failed");

        nixlBlobDesc source_desc;
        source_desc.addr = reinterpret_cast<uintptr_t>(source_.data());
        source_desc.len = source_.size();
        source_desc.devId = 0;
        require(engine_->registerMem(source_desc, DRAM_SEG, sourceMd_) == NIXL_SUCCESS,
                "real UCX loopback source registration failed");

        nixlBlobDesc destination_desc;
        destination_desc.addr = reinterpret_cast<uintptr_t>(destination_.data());
        destination_desc.len = destination_.size();
        destination_desc.devId = 0;
        require(engine_->registerMem(destination_desc, DRAM_SEG, destinationMd_) ==
                    NIXL_SUCCESS,
                "real UCX loopback destination registration failed");
        require(engine_->loadLocalMD(destinationMd_, remoteDestinationMd_) == NIXL_SUCCESS,
                "real UCX loopback destination metadata import failed");

        local_.addDesc(nixlMetaDesc(reinterpret_cast<uintptr_t>(source_.data()),
                                    source_.size(),
                                    0,
                                    sourceMd_));
        remote_.addDesc(nixlMetaDesc(reinterpret_cast<uintptr_t>(destination_.data()),
                                     destination_.size(),
                                     0,
                                     remoteDestinationMd_));
    }

    ~loopback_fixture_t() {
        if (remoteDestinationMd_ != nullptr) {
            static_cast<void>(engine_->unloadMD(remoteDestinationMd_));
        }
        if (destinationMd_ != nullptr) {
            static_cast<void>(engine_->deregisterMem(destinationMd_));
        }
        if (sourceMd_ != nullptr) {
            static_cast<void>(engine_->deregisterMem(sourceMd_));
        }
        if (engine_ != nullptr) {
            static_cast<void>(engine_->disconnect(agentName_));
        }
    }

    [[nodiscard]] nixlBackendReqH *
    prepare() {
        nixlBackendReqH *handle = nullptr;
        const nixl_status_t status = engine_->prepXfer(
            NIXL_WRITE, local_, remote_, agentName_, handle, nullptr);
        require(status == NIXL_SUCCESS,
                std::string("real UCX loopback transfer preparation failed: ") +
                    nixlEnumStrings::statusStr(status));
        require(handle != nullptr, "real UCX loopback preparation returned no handle");
        return handle;
    }

    [[nodiscard]] nixlBackendTransferEventBinding
    nextBinding(const nixlBackendReqH *handle) const {
        nixl_xfer_attestation_t attestation;
        require(engine_->queryXferAttestation(handle, attestation) == NIXL_SUCCESS,
                "real UCX loopback attestation query failed");
        return {
            .handleIdentity = attestation.handleIdentity,
            .generation = attestation.generation + 1,
        };
    }

    [[nodiscard]] std::unique_ptr<nixlBackendEventSubscription>
    subscribe(nixlBackendReqH *handle,
              const std::shared_ptr<recording_sink_t> &sink) {
        std::unique_ptr<nixlBackendEventSubscription> subscription;
        require(engine_->subscribeXferTerminal(
                    handle, nextBinding(handle), sink, subscription) == NIXL_SUCCESS,
                "real UCX loopback terminal subscription failed");
        require(subscription != nullptr,
                "real UCX loopback terminal subscription returned no lifetime");
        sink->bind(subscription.get());
        return subscription;
    }

    void
    post(nixlBackendReqH *&handle) {
        const nixl_status_t status = engine_->postXfer(
            NIXL_WRITE, local_, remote_, agentName_, handle, nullptr);
        require(status == NIXL_SUCCESS || status == NIXL_IN_PROG,
                "real UCX loopback transfer post failed");
    }

    void
    claimCompletion(nixlBackendReqH *handle) {
        nixl_xfer_attestation_t attestation;
        require(engine_->takeXferCompletionAttestation(handle, attestation) == NIXL_SUCCESS,
                "real UCX loopback completion attestation was not claimable");
        require(attestation.status == NIXL_SUCCESS,
                "real UCX loopback completion attestation was not successful");
    }

    void
    release(nixlBackendReqH *handle) {
        require(engine_->releaseReqH(handle) == NIXL_SUCCESS,
                "real UCX loopback request release failed");
    }

    void
    resetPayload(std::uint8_t value) {
        source_.fill(value);
        destination_.fill(0);
    }

    void
    requirePayloadCopied() const {
        require(source_ == destination_, "real UCX loopback payload did not transfer");
    }

private:
    static constexpr std::size_t payloadSize_ = 4096;
    const std::string agentName_ = "terminal-cancellation-loopback";
    nixl_b_params_t customParams_;
    std::unique_ptr<nixlUcxEngine> engine_;
    std::array<std::uint8_t, payloadSize_> source_{};
    std::array<std::uint8_t, payloadSize_> destination_{};
    nixlBackendMD *sourceMd_ = nullptr;
    nixlBackendMD *destinationMd_ = nullptr;
    nixlBackendMD *remoteDestinationMd_ = nullptr;
    nixl_meta_dlist_t local_;
    nixl_meta_dlist_t remote_;
};

void
requireDrained(const nixlBackendEventSubscription &subscription,
               std::string_view message) {
    nixlBackendEventSubscriptionInventory inventory;
    subscription.queryInventory(inventory);
    require(inventory.backendProducers == 0 && inventory.activeCallbackSlots == 0 &&
                inventory.queuedOwnerContinuations == 0,
            message);
}

void
testCancelBeforePostPermitsRearmAndReuse() {
    loopback_fixture_t fixture;
    nixlBackendReqH *handle = fixture.prepare();

    auto canceled_sink = std::make_shared<recording_sink_t>();
    auto canceled_subscription = fixture.subscribe(handle, canceled_sink);
    require(canceled_subscription->cancel() == NIXL_SUCCESS,
            "pre-post cancellation did not complete synchronously");
    require(canceled_sink->waitForTransitions(1),
            "pre-post cancellation did not publish its terminal event");
    const auto canceled = canceled_sink->transitions();
    require(canceled.size() == 1 && canceled.front().status == NIXL_ERR_CANCELED,
            "pre-post cancellation did not publish one exact canceled event");
    requireDrained(*canceled_subscription,
                   "pre-post cancellation retained backend lifecycle ownership");

    fixture.resetPayload(0x5a);
    auto reused_sink = std::make_shared<recording_sink_t>();
    auto reused_subscription = fixture.subscribe(handle, reused_sink);
    fixture.post(handle);
    require(reused_sink->waitForTransitions(1),
            "rearmed request did not publish terminal success");
    const auto reused = reused_sink->transitions();
    require(reused.size() == 1 && reused.front().status == NIXL_SUCCESS,
            "rearmed request did not terminate successfully");
    requireDrained(*reused_subscription,
                   "rearmed request retained backend lifecycle ownership");
    fixture.requirePayloadCopied();
    fixture.claimCompletion(handle);
    fixture.release(handle);
}

void
testCancelBeforePostPermitsRelease() {
    loopback_fixture_t fixture;
    nixlBackendReqH *handle = fixture.prepare();
    auto sink = std::make_shared<recording_sink_t>();
    auto subscription = fixture.subscribe(handle, sink);

    require(subscription->cancel() == NIXL_SUCCESS,
            "release-path pre-post cancellation failed");
    require(sink->waitForTransitions(1),
            "release-path pre-post cancellation did not publish");
    requireDrained(*subscription,
                   "release-path pre-post cancellation retained backend ownership");
    fixture.release(handle);
}

void
testOwnerCallbackCancellationRaceIsNonBlocking() {
    loopback_fixture_t fixture;
    nixlBackendReqH *handle = fixture.prepare();

    constexpr std::size_t iterations = 128;
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        fixture.resetPayload(static_cast<std::uint8_t>(iteration + 1));
        auto sink = std::make_shared<recording_sink_t>(true);
        auto subscription = fixture.subscribe(handle, sink);
        fixture.post(handle);
        require(sink->waitForTransitions(1),
                "owner callback cancellation race did not terminate");

        const auto transitions = sink->transitions();
        const auto cancellation_statuses = sink->cancelStatuses();
        require(transitions.size() == 1 && transitions.front().status == NIXL_SUCCESS,
                "owner callback race changed successful terminal publication");
        require(cancellation_statuses.size() == 1 &&
                    cancellation_statuses.front() == NIXL_SUCCESS,
                "owner callback cancellation did not observe committed terminality");
        requireDrained(*subscription,
                       "owner callback cancellation race retained backend ownership");
        fixture.requirePayloadCopied();
        fixture.claimCompletion(handle);
    }

    fixture.release(handle);
}

} // namespace

int
main() {
    try {
        testCancelBeforePostPermitsRearmAndReuse();
        testCancelBeforePostPermitsRelease();
        testOwnerCallbackCancellationRaceIsNonBlocking();
    }
    catch (const std::exception &error) {
        std::cerr << "UCX terminal cancellation test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "UCX terminal cancellation tests passed\n";
    return 0;
}
