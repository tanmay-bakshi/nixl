/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "terminal_event_api.h"

#include <atomic>
#include <barrier>
#include <memory>
#include <thread>

#include <gtest/gtest.h>

namespace {

class testBackendSubscription final : public nixlBackendEventSubscription {
public:
    nixl_status_t
    cancel() noexcept override {
        ++cancelCount;
        return cancelStatus;
    }

    std::atomic<int> cancelCount = 0;
    nixl_status_t cancelStatus = NIXL_SUCCESS;
};

TEST(TerminalEventApi, TransferAdapterPublishesExactTerminalOnce) {
    nixl::terminalEventChannel channel(2);
    const auto channel_subscription = channel.subscribe({
        .kind = nixl::terminal_event_kind_t::TRANSFER,
        .ownerCookie = 7,
        .identity = 11,
        .generation = 3,
    });
    ASSERT_NE(channel_subscription, nullptr);
    nixlTerminalTransferAdapter adapter({.handleIdentity = 11, .generation = 3},
                                        channel_subscription);
    std::atomic<int> terminal_count = 0;
    adapter.bindTerminalCallback([&terminal_count]() { ++terminal_count; });

    adapter.publish({
        .binding = {.handleIdentity = 11, .generation = 3},
        .status = NIXL_SUCCESS,
        .nativeTimestampNs = 1234,
    });
    adapter.publish({
        .binding = {.handleIdentity = 11, .generation = 3},
        .status = NIXL_ERR_CANCELED,
        .nativeTimestampNs = 1235,
    });

    const nixl::terminal_event_batch_t batch = channel.drain();
    ASSERT_EQ(batch.events.size(), 1U);
    EXPECT_EQ(std::get<nixl_status_t>(batch.events.front().result), NIXL_SUCCESS);
    EXPECT_EQ(batch.events.front().nativeTimestampNs, 1234U);
    EXPECT_EQ(terminal_count.load(), 1);
    EXPECT_EQ(batch.inventory.activeSubscriptions, 0U);
}

TEST(TerminalEventApi, MismatchedTransferPublicationFailsChannelClosed) {
    nixl::terminalEventChannel channel(1);
    const auto channel_subscription = channel.subscribe({
        .kind = nixl::terminal_event_kind_t::TRANSFER,
        .ownerCookie = 7,
        .identity = 11,
        .generation = 3,
    });
    nixlTerminalTransferAdapter adapter({.handleIdentity = 11, .generation = 3},
                                        channel_subscription);
    adapter.publish({
        .binding = {.handleIdentity = 12, .generation = 3},
        .status = NIXL_SUCCESS,
        .nativeTimestampNs = 1234,
    });

    const nixl::terminal_channel_inventory_t inventory = channel.inventory();
    EXPECT_TRUE(nixl::hasTerminalChannelFatal(
        inventory.health.fatal, nixl::terminal_channel_fatal_t::INVALID_PUBLICATION));
    EXPECT_EQ(inventory.activeSubscriptions, 0U);
}

TEST(TerminalEventApi, CapabilityAdapterStaysActiveUntilRetirement) {
    nixl::terminalEventChannel channel(4);
    const auto channel_subscription = channel.subscribe({
        .kind = nixl::terminal_event_kind_t::CAPABILITY,
        .ownerCookie = 8,
        .identity = 20,
        .generation = 4,
    });
    nixlTerminalCapabilityAdapter adapter(20, 4, channel_subscription);
    std::atomic<int> terminal_count = 0;
    adapter.bindTerminalCallback([&terminal_count]() { ++terminal_count; });

    adapter.publish({20, 4, nixl_backend_capability_state_t::READY, 1, 100});
    adapter.publish({20, 4, nixl_backend_capability_state_t::FAILED, 2, 101});
    EXPECT_EQ(channel.inventory().activeSubscriptions, 1U);
    adapter.publish({20, 4, nixl_backend_capability_state_t::RETIRED, 3, 102});

    const nixl::terminal_event_batch_t batch = channel.drain();
    ASSERT_EQ(batch.events.size(), 3U);
    EXPECT_EQ(batch.events[0].epoch, 1U);
    EXPECT_EQ(batch.events[1].epoch, 2U);
    EXPECT_EQ(batch.events[2].epoch, 3U);
    EXPECT_EQ(terminal_count.load(), 1);
    EXPECT_EQ(batch.inventory.activeSubscriptions, 0U);
}

TEST(TerminalEventApi, TerminalSnapshotAndBackendReleaseAreRaceSafe) {
    nixl::terminalEventChannel channel(1);
    const auto channel_subscription = channel.subscribe({
        .kind = nixl::terminal_event_kind_t::TRANSFER,
        .ownerCookie = 9,
        .identity = 30,
        .generation = 5,
    });
    const auto adapter = std::make_shared<nixlTerminalTransferAdapter>(
        nixlBackendTransferEventBinding{30, 5}, channel_subscription);
    auto backend_subscription = std::make_unique<testBackendSubscription>();
    testBackendSubscription *const backend_subscription_ptr = backend_subscription.get();
    auto handle = std::shared_ptr<nixlTerminalEventSubscriptionH>(
        new nixlTerminalEventSubscriptionH(
            1,
            1,
            {nixl_terminal_event_kind_t::TRANSFER, 9, 30, 5, true},
            nullptr,
            nullptr,
            std::move(backend_subscription),
            adapter));
    adapter->bindTerminalCallback([weak_handle = std::weak_ptr(handle)]() {
        if (const auto retained = weak_handle.lock(); retained != nullptr) {
            retained->finishRelease();
        }
    });
    std::barrier start(2);

    std::thread publisher([&]() {
        start.arrive_and_wait();
        adapter->publish({{30, 5}, NIXL_SUCCESS, 1000});
    });
    start.arrive_and_wait();
    for (int iteration = 0; iteration < 1000; ++iteration) {
        const nixl_terminal_subscription_info_t info = handle->snapshot();
        EXPECT_EQ(info.identity, 30U);
    }
    publisher.join();

    std::unique_ptr<nixlBackendEventSubscription> retained_backend =
        handle->takeBackendSubscription();
    ASSERT_NE(retained_backend, nullptr);
    EXPECT_EQ(retained_backend->cancel(), NIXL_SUCCESS);
    handle->finishRelease();
    EXPECT_FALSE(handle->snapshot().active);
    EXPECT_EQ(backend_subscription_ptr->cancelCount.load(), 1);
    EXPECT_EQ(channel.inventory().activeSubscriptions, 0U);
}

} // namespace
