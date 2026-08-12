/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "ucx_terminal_progress.h"
#include "ucx_utils.h"

namespace {

using namespace nixl::ucx;

void
require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

class recording_sink_t final : public terminal_submission_sink_t {
public:
    nixl_status_t
    publishTerminal(const terminal_submission_result_t &result) noexcept override {
        results.push_back(result);
        return publishStatus;
    }

    std::vector<terminal_submission_result_t> results;
    nixl_status_t publishStatus = NIXL_SUCCESS;
};

std::shared_ptr<ucx_worker_continuation_queue_t>
makeQueue(std::size_t capacity, std::size_t &wake_count) {
    return std::make_shared<ucx_worker_continuation_queue_t>(
        capacity,
        [&wake_count]() {
            ++wake_count;
            return NIXL_SUCCESS;
        });
}

void
testCallbackBeforePosterReturn() {
    std::size_t wake_count = 0;
    std::size_t release_count = 0;
    auto queue = makeQueue(8, wake_count);
    auto sink = std::make_shared<recording_sink_t>();
    auto state = std::make_shared<terminal_submission_state_t>(
        11, 101, 7, 1, 1, false, sink);
    require(state->registerChunk() == NIXL_SUCCESS, "chunk registration failed");
    require(state->registerFlush() == NIXL_SUCCESS, "flush registration failed");
    require(state->completeChunk(NIXL_SUCCESS, 10) == NIXL_SUCCESS,
            "chunk completion failed");
    require(state->sealPosting({}, 11) == NIXL_SUCCESS, "posting seal failed");

    void *request = reinterpret_cast<void *>(0x1000);
    auto slot = ucx_callback_slot_t::create(
        state,
        ucx_callback_kind_t::ENDPOINT_FLUSH,
        queue,
        [&](void *completed) {
            require(completed == request, "released request identity changed");
            ++release_count;
        });

    ucx_callback_slot_t::ucpCompletion(request, UCS_OK, slot.get());
    require(queue->size() == 0 && release_count == 0 && sink->results.empty(),
            "early callback advanced before poster evidence");
    require(slot->armPoster(request, NIXL_IN_PROG, 20) == NIXL_SUCCESS,
            "poster handshake failed");
    require(queue->size() == 1 && release_count == 0 && sink->results.empty(),
            "poster handshake did not defer owner work");
    require(queue->drain() == 1, "owner did not drain callback continuation");
    require(release_count == 1 && slot->isDelivered(),
            "request was not released exactly once by owner");
    require(sink->results.size() == 1 &&
                sink->results[0].status == NIXL_SUCCESS &&
                sink->results[0].ownerCookie == 11 &&
                sink->results[0].handleIdentity == 101 &&
                sink->results[0].generation == 7 &&
                sink->results[0].nativeTimestampNs >= 10 &&
                sink->results[0].nativeTimestampNs != 0,
            "terminal result did not preserve exact identity and timestamp");
    require(wake_count == 1 && queue->fatalStatus() == NIXL_SUCCESS,
            "healthy callback path poisoned its continuation queue");
}

void
testCallbackCanScheduleBeforePosterArms() {
    std::size_t wake_count = 0;
    std::size_t release_count = 0;
    auto queue = makeQueue(8, wake_count);
    auto sink = std::make_shared<recording_sink_t>();
    auto state = std::make_shared<terminal_submission_state_t>(
        20, 110, 16, 1, 1, false, sink);
    require(state->registerChunk() == NIXL_SUCCESS,
            "scheduled-before-poster chunk registration failed");
    require(state->registerFlush() == NIXL_SUCCESS,
            "scheduled-before-poster flush registration failed");
    require(state->completeChunk(NIXL_SUCCESS, 110) == NIXL_SUCCESS,
            "scheduled-before-poster chunk completion failed");
    require(state->sealPosting({}, 111) == NIXL_SUCCESS,
            "scheduled-before-poster posting seal failed");

    void *request = reinterpret_cast<void *>(0xb000);
    auto slot = ucx_callback_slot_t::create(
        state,
        ucx_callback_kind_t::ENDPOINT_FLUSH,
        queue,
        [&](void *completed) {
            require(completed == request,
                    "scheduled-before-poster request identity changed");
            ++release_count;
        });
    slot->recordCallback(request, NIXL_SUCCESS, 112);
    require(slot->armPoster(request, NIXL_IN_PROG, 113) == NIXL_SUCCESS,
            "poster rejected an early callback after scheduling");
    require(queue->drain() == 1 && release_count == 1 && slot->isDelivered(),
            "scheduled-before-poster callback ownership did not drain");
    require(sink->results.size() == 1 &&
                sink->results[0].status == NIXL_SUCCESS,
            "scheduled-before-poster path did not terminate successfully");
}

void
testNotificationAfterFlush() {
    std::size_t wake_count = 0;
    auto queue = makeQueue(8, wake_count);
    auto sink = std::make_shared<recording_sink_t>();
    auto state = std::make_shared<terminal_submission_state_t>(
        12, 102, 8, 1, 1, true, sink);
    std::vector<std::string> order;
    void *flush_request = reinterpret_cast<void *>(0x2000);
    void *notification_request = reinterpret_cast<void *>(0x3000);

    auto notification_slot = ucx_callback_slot_t::create(
        state,
        ucx_callback_kind_t::NOTIFICATION,
        queue,
        [&](void *completed) {
            require(completed == notification_request,
                    "notification request identity changed");
            order.emplace_back("notification-release");
        });
    auto flush_slot = ucx_callback_slot_t::create(
        state,
        ucx_callback_kind_t::ENDPOINT_FLUSH,
        queue,
        [&](void *completed) {
            require(completed == flush_request, "flush request identity changed");
            order.emplace_back("flush-release");
        });

    require(state->registerChunk() == NIXL_SUCCESS, "chunk registration failed");
    require(state->registerFlush() == NIXL_SUCCESS, "flush registration failed");
    require(state->completeChunk(NIXL_SUCCESS, 30) == NIXL_SUCCESS,
            "chunk completion failed");
    require(state->sealPosting(
                [&]() {
                    order.emplace_back("notification-post");
                    notification_slot->recordCallback(
                        notification_request, NIXL_SUCCESS, 42);
                    require(notification_slot->armPoster(
                                notification_request, NIXL_IN_PROG, 43) == NIXL_SUCCESS,
                            "notification poster handshake failed");
                    return NIXL_IN_PROG;
                },
                31) == NIXL_SUCCESS,
            "posting seal failed");

    flush_slot->recordCallback(flush_request, NIXL_SUCCESS, 40);
    require(flush_slot->armPoster(flush_request, NIXL_IN_PROG, 41) == NIXL_SUCCESS,
            "flush poster handshake failed");
    require(queue->drain() == 1, "flush continuation was not isolated");
    require(order == std::vector<std::string>({"flush-release", "notification-post"}),
            "notification was not posted after flush completion on the owner");
    require(sink->results.empty() && queue->size() == 1,
            "notification post incorrectly implied terminality");
    require(queue->drain() == 1, "notification continuation was not delivered");
    require(order == std::vector<std::string>(
                         {"flush-release", "notification-post", "notification-release"}),
            "notification completion ordering changed");
    require(sink->results.size() == 1 && sink->results[0].status == NIXL_SUCCESS,
            "notification completion did not seal terminal success");
    require(wake_count == 2, "each owner continuation did not wake its worker");
}

void
testCallbacksBeforePostingSeal() {
    std::size_t wake_count = 0;
    auto queue = makeQueue(8, wake_count);
    auto sink = std::make_shared<recording_sink_t>();
    auto state = std::make_shared<terminal_submission_state_t>(
        18, 108, 14, 1, 1, true, sink);
    std::vector<std::string> order;
    void *data_request = reinterpret_cast<void *>(0x6000);
    void *flush_request = reinterpret_cast<void *>(0x7000);
    void *notification_request = reinterpret_cast<void *>(0x8000);

    auto data_slot = ucx_callback_slot_t::create(
        state,
        ucx_callback_kind_t::DATA_CHUNK,
        queue,
        [&](void *completed) {
            require(completed == data_request, "data request identity changed");
            order.emplace_back("data-release");
        });
    auto flush_slot = ucx_callback_slot_t::create(
        state,
        ucx_callback_kind_t::ENDPOINT_FLUSH,
        queue,
        [&](void *completed) {
            require(completed == flush_request, "flush request identity changed");
            order.emplace_back("flush-release");
        });
    auto notification_slot = ucx_callback_slot_t::create(
        state,
        ucx_callback_kind_t::NOTIFICATION,
        queue,
        [&](void *completed) {
            require(completed == notification_request,
                    "notification request identity changed");
            order.emplace_back("notification-release");
        });

    require(state->registerChunk() == NIXL_SUCCESS, "chunk registration failed");
    require(state->registerFlush() == NIXL_SUCCESS, "flush registration failed");
    data_slot->recordCallback(data_request, NIXL_SUCCESS, 90);
    require(data_slot->armPoster(data_request, NIXL_IN_PROG, 91) == NIXL_SUCCESS,
            "data poster handshake failed");
    flush_slot->recordCallback(flush_request, NIXL_SUCCESS, 92);
    require(flush_slot->armPoster(flush_request, NIXL_IN_PROG, 93) == NIXL_SUCCESS,
            "flush poster handshake failed");
    require(queue->drain() == 2, "early data callbacks were not delivered");
    require(state->phase() == terminal_submission_phase_t::POSTING &&
                sink->results.empty(),
            "callbacks completed the submission before posting was sealed");

    require(state->sealPosting(
                [&]() {
                    order.emplace_back("notification-post");
                    notification_slot->recordCallback(
                        notification_request, NIXL_SUCCESS, 94);
                    require(notification_slot->armPoster(
                                notification_request, NIXL_IN_PROG, 95) == NIXL_SUCCESS,
                            "notification poster handshake failed");
                    return NIXL_IN_PROG;
                },
                96) == NIXL_SUCCESS,
            "posting seal stranded notification dispatch");
    require(order == std::vector<std::string>(
                         {"data-release", "flush-release", "notification-post"}),
            "posting seal did not dispatch notification after early callbacks");
    require(queue->drain() == 1, "notification continuation was not delivered");
    require(sink->results.size() == 1 &&
                sink->results[0].status == NIXL_SUCCESS &&
                sink->results[0].nativeTimestampNs == 95,
            "early-callback submission did not publish one terminal result");
    require(wake_count == 3, "early callback path emitted unexpected wake count");
}

void
testCancellationAndFailure() {
    std::size_t wake_count = 0;
    auto queue = makeQueue(8, wake_count);
    auto sink = std::make_shared<recording_sink_t>();
    auto state = std::make_shared<terminal_submission_state_t>(
        13, 103, 9, 1, 1, false, sink);
    require(state->registerChunk() == NIXL_SUCCESS, "chunk registration failed");
    require(state->registerFlush() == NIXL_SUCCESS, "flush registration failed");
    require(state->completeChunk(NIXL_SUCCESS, 50) == NIXL_SUCCESS,
            "chunk completion failed");
    require(state->sealPosting({}, 51) == NIXL_SUCCESS, "posting seal failed");
    void *request = reinterpret_cast<void *>(0x4000);
    auto slot = ucx_callback_slot_t::create(
        state, ucx_callback_kind_t::ENDPOINT_FLUSH, queue, [](void *) {});
    slot->recordCallback(request, NIXL_ERR_CANCELED, 52);
    require(slot->armPoster(request, NIXL_IN_PROG, 53) == NIXL_SUCCESS,
            "canceled request handshake failed");
    require(queue->drain() == 1, "canceled request was not delivered");
    require(sink->results.size() == 1 &&
                sink->results[0].status == NIXL_ERR_CANCELED,
            "cancellation was not terminal and exact");

    auto mismatch_sink = std::make_shared<recording_sink_t>();
    auto mismatch_state = std::make_shared<terminal_submission_state_t>(
        14, 104, 10, 1, 1, false, mismatch_sink);
    require(mismatch_state->registerChunk() == NIXL_SUCCESS,
            "mismatch chunk registration failed");
    require(mismatch_state->registerFlush() == NIXL_SUCCESS,
            "mismatch flush registration failed");
    require(mismatch_state->completeChunk(NIXL_SUCCESS, 60) == NIXL_SUCCESS,
            "mismatch chunk completion failed");
    require(mismatch_state->sealPosting({}, 61) == NIXL_SUCCESS,
            "mismatch posting seal failed");
    auto mismatch_slot = ucx_callback_slot_t::create(
        mismatch_state, ucx_callback_kind_t::ENDPOINT_FLUSH, queue, [](void *) {});
    mismatch_slot->recordCallback(reinterpret_cast<void *>(0x5000), NIXL_SUCCESS, 62);
    require(mismatch_slot->armPoster(
                reinterpret_cast<void *>(0x5001), NIXL_IN_PROG, 63) == NIXL_SUCCESS,
            "mismatched request handshake failed");
    require(queue->drain() == 1, "mismatched request was not delivered");
    require(mismatch_sink->results.size() == 1 &&
                mismatch_sink->results[0].status == NIXL_ERR_BACKEND,
            "request identity mismatch did not fail closed");
}

void
testNotificationFailure() {
    auto sink = std::make_shared<recording_sink_t>();
    auto state = std::make_shared<terminal_submission_state_t>(
        15, 105, 11, 1, 1, true, sink);
    require(state->registerChunk() == NIXL_SUCCESS, "chunk registration failed");
    require(state->registerFlush() == NIXL_SUCCESS, "flush registration failed");
    require(state->completeChunk(NIXL_SUCCESS, 70) == NIXL_SUCCESS,
            "chunk completion failed");
    require(state->sealPosting(
                []() { return NIXL_ERR_REMOTE_DISCONNECT; }, 71) == NIXL_SUCCESS,
            "posting seal failed");
    require(state->completeFlush(NIXL_SUCCESS, 72) == NIXL_SUCCESS,
            "notification failure transition failed");
    require(sink->results.size() == 1 &&
                sink->results[0].status == NIXL_ERR_REMOTE_DISCONNECT,
            "notification failure did not become exact terminal failure");
}

void
testCompositeFolding() {
    auto sink = std::make_shared<recording_sink_t>();
    auto state = std::make_shared<terminal_submission_state_t>(
        16, 106, 12, 3, 3, false, sink);
    for (std::size_t index = 0; index < 3; ++index) {
        require(state->registerChunk() == NIXL_SUCCESS,
                "composite chunk registration failed");
        require(state->registerFlush() == NIXL_SUCCESS,
                "composite flush registration failed");
    }
    require(state->completeFlush(NIXL_SUCCESS, 80) == NIXL_SUCCESS,
            "first early flush failed");
    require(state->completeChunk(NIXL_SUCCESS, 81) == NIXL_SUCCESS,
            "first early chunk failed");
    require(state->completeFlush(NIXL_SUCCESS, 82) == NIXL_SUCCESS,
            "second early flush failed");
    require(state->completeChunk(NIXL_SUCCESS, 83) == NIXL_SUCCESS,
            "second early chunk failed");
    require(state->sealPosting({}, 84) == NIXL_SUCCESS,
            "composite posting seal failed");
    require(state->phase() == terminal_submission_phase_t::DATA_FLUSH &&
                sink->results.empty(),
            "partial composite became terminal");
    require(state->completeChunk(NIXL_SUCCESS, 85) == NIXL_SUCCESS &&
                sink->results.empty(),
            "chunk folding ignored a pending endpoint flush");
    require(state->completeFlush(NIXL_SUCCESS, 86) == NIXL_SUCCESS,
            "final composite flush failed");
    require(sink->results.size() == 1 &&
                sink->results[0].status == NIXL_SUCCESS &&
                sink->results[0].nativeTimestampNs == 86,
            "composite did not publish one folded terminal result");
    require(state->completeChunk(NIXL_SUCCESS, 87) == NIXL_ERR_NOT_ALLOWED,
            "terminal composite accepted a duplicate chunk");
}

void
testContinuationOverflowAndShutdown() {
    std::size_t wake_count = 0;
    auto queue = makeQueue(1, wake_count);
    std::size_t run_count = 0;
    require(queue->enqueue([&]() {
                ++run_count;
                return NIXL_SUCCESS;
            }) == NIXL_SUCCESS,
            "first continuation enqueue failed");
    require(queue->enqueue([]() { return NIXL_SUCCESS; }) == NIXL_ERR_BACKEND,
            "bounded continuation queue did not fail on overflow");
    require(queue->fatalStatus() == NIXL_ERR_BACKEND && wake_count == 2,
            "overflow did not store fatal state and wake the owner");
    require(queue->drain() == 1 && run_count == 1,
            "overflow damaged the continuation already admitted");
    require(queue->close() == NIXL_ERR_BACKEND,
            "overflow fatal state disappeared during shutdown");

    std::size_t shutdown_wakes = 0;
    auto shutdown_queue = makeQueue(2, shutdown_wakes);
    require(shutdown_queue->enqueue([]() { return NIXL_SUCCESS; }) == NIXL_SUCCESS,
            "shutdown continuation enqueue failed");
    require(shutdown_queue->close() == NIXL_ERR_CANCELED,
            "shutdown with pending continuation did not fail closed");
    require(shutdown_queue->enqueue([]() { return NIXL_SUCCESS; }) == NIXL_ERR_CANCELED,
            "closed continuation queue accepted new work");
    require(shutdown_queue->drain() == 1,
            "shutdown drain lost an admitted continuation");
}

void
testOwnerEnqueueDrainsToQuiescenceWithoutReentrantWake() {
    std::size_t wake_count = 0;
    auto queue = makeQueue(8, wake_count);
    require(queue->bindOwnerThread(std::this_thread::get_id()) == NIXL_SUCCESS,
            "continuation owner binding failed");

    std::vector<int> order;
    require(queue->enqueue([&]() {
                order.push_back(1);
                return queue->enqueue([&]() {
                    order.push_back(2);
                    return NIXL_SUCCESS;
                });
            }) == NIXL_SUCCESS,
            "owner continuation enqueue failed");
    require(wake_count == 0,
            "owner-thread enqueue re-entered UCX worker signalling");
    require(queue->drain() == 2 && order == std::vector<int>({1, 2}),
            "owner drain stranded a continuation scheduled by its predecessor");
    require(queue->size() == 0 && queue->fatalStatus() == NIXL_SUCCESS,
            "quiescent owner drain left native work or a fatal state");
}

void
testOverflowDrainsCallbackOwnership() {
    std::size_t wake_count = 0;
    auto queue = makeQueue(1, wake_count);
    auto sink = std::make_shared<recording_sink_t>();
    auto state = std::make_shared<terminal_submission_state_t>(
        19, 109, 15, 1, 2, false, sink);
    require(state->registerChunk() == NIXL_SUCCESS,
            "overflow chunk registration failed");
    require(state->registerFlush() == NIXL_SUCCESS &&
                state->registerFlush() == NIXL_SUCCESS,
            "overflow flush registration failed");
    require(state->completeChunk(NIXL_SUCCESS, 100) == NIXL_SUCCESS,
            "overflow chunk completion failed");
    require(state->sealPosting({}, 101) == NIXL_SUCCESS,
            "overflow posting seal failed");

    std::size_t release_count = 0;
    std::size_t retired_count = 0;
    require(queue->registerProducer() == NIXL_SUCCESS &&
                queue->registerProducer() == NIXL_SUCCESS,
            "overflow producer registration failed");
    auto make_slot = [&](void *request) {
        return ucx_callback_slot_t::create(
            state,
            ucx_callback_kind_t::ENDPOINT_FLUSH,
            queue,
            [&, request](void *completed) {
                require(completed == request, "overflow request identity changed");
                ++release_count;
            },
            {},
            [&]() {
                ++retired_count;
                static_cast<void>(queue->retireProducer());
            });
    };
    void *first_request = reinterpret_cast<void *>(0x9000);
    void *second_request = reinterpret_cast<void *>(0xa000);
    auto first = make_slot(first_request);
    auto second = make_slot(second_request);
    first->recordCallback(first_request, NIXL_SUCCESS, 102);
    require(first->armPoster(first_request, NIXL_IN_PROG, 103) == NIXL_SUCCESS,
            "first overflow callback was not queued");
    second->recordCallback(second_request, NIXL_SUCCESS, 104);
    require(second->armPoster(second_request, NIXL_IN_PROG, 105) ==
                NIXL_ERR_BACKEND,
            "overflow callback did not report sticky fatal status");
    require(queue->size() == 2 && queue->fatalStatus() == NIXL_ERR_BACKEND,
            "overflow cleanup authority was not retained");
    require(queue->drain() == 2,
            "overflow owner did not drain both callback continuations");
    require(release_count == 2 && retired_count == 2 && queue->size() == 0 &&
                first->isDelivered() && second->isDelivered(),
            "overflow callback ownership was not conserved");
    require(sink->results.size() == 1 &&
                sink->results[0].status == NIXL_SUCCESS,
            "overflow cleanup changed transfer terminality");
}

void
testNoProgressOwnerIsUnsupported() {
    nixlUcxContext context({},
                           false,
                           1,
                           nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT,
                           0,
                           "TLS=self");
    nixlUcxWorker worker(context);
    auto sink = std::make_shared<recording_sink_t>();
    auto state = std::make_shared<terminal_submission_state_t>(
        17, 107, 13, 1, 1, false, sink);
    std::shared_ptr<ucx_callback_slot_t> slot;
    require(worker.makeTerminalCallbackSlot(
                state, ucx_callback_kind_t::ENDPOINT_FLUSH, slot) ==
                NIXL_ERR_NOT_SUPPORTED &&
                slot == nullptr,
            "autonomous callback support appeared without a progress owner");
    require(worker.claimProgressOwner() == NIXL_SUCCESS,
            "progress owner claim failed");
    require(worker.makeTerminalCallbackSlot(
                state, ucx_callback_kind_t::ENDPOINT_FLUSH, slot) == NIXL_SUCCESS &&
                slot != nullptr,
            "claimed progress owner could not allocate a stable callback slot");
}

void
testCallbackSlotAllocationFailureDoesNotRegisterProducer() {
    nixlUcxContext context({},
                           false,
                           1,
                           nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT,
                           0,
                           "TLS=self");
    nixlUcxWorker worker(context);
    require(worker.claimProgressOwner() == NIXL_SUCCESS,
            "progress owner claim failed for allocation fault test");
    auto sink = std::make_shared<recording_sink_t>();
    auto state = std::make_shared<terminal_submission_state_t>(
        21, 111, 17, 1, 1, false, sink);
    std::shared_ptr<ucx_callback_slot_t> slot;
    worker.failNextTerminalCallbackSlotForTest();
    require(worker.makeTerminalCallbackSlot(
                state, ucx_callback_kind_t::DATA_CHUNK, slot) == NIXL_ERR_BACKEND,
            "injected callback-slot allocation failure was not returned");
    require(slot == nullptr && worker.activeTerminalCallbackCount() == 0 &&
                worker.getContinuationQueue()->producerCount() == 0 &&
                worker.terminalLifecycleDrained(),
            "callback-slot allocation failure leaked native ownership");
}

} // namespace

int
main() {
    try {
        testCallbackBeforePosterReturn();
        testCallbackCanScheduleBeforePosterArms();
        testNotificationAfterFlush();
        testCallbacksBeforePostingSeal();
        testCancellationAndFailure();
        testNotificationFailure();
        testCompositeFolding();
        testContinuationOverflowAndShutdown();
        testOwnerEnqueueDrainsToQuiescenceWithoutReentrantWake();
        testOverflowDrainsCallbackOwnership();
        testNoProgressOwnerIsUnsupported();
        testCallbackSlotAllocationFailureDoesNotRegisterProducer();
    }
    catch (const std::exception &error) {
        std::cerr << "ucx terminal progress test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "ucx terminal progress tests passed\n";
    return 0;
}
