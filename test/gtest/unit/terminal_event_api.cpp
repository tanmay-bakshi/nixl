/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "terminal_event_api.h"

#include <atomic>
#include <barrier>
#include <functional>
#include <memory>
#include <thread>

#include <gtest/gtest.h>

class nixlTerminalEventSubscriptionTestPeer {
public:
    static std::shared_ptr<nixlTerminalEventSubscriptionH>
    makeTransfer(
        std::unique_ptr<nixlBackendEventSubscription> backend_subscription,
        const std::shared_ptr<nixlTerminalTransferAdapter> &adapter) {
        return std::shared_ptr<nixlTerminalEventSubscriptionH>(
            new nixlTerminalEventSubscriptionH(
                1,
                1,
                {nixl_terminal_event_kind_t::TRANSFER, 9, 30, 5, true},
                nullptr,
                nullptr,
                std::move(backend_subscription),
                adapter));
    }

    static std::shared_ptr<nixlTerminalEventSubscriptionH>
    makeCapability(
        std::unique_ptr<nixlBackendEventSubscription> backend_subscription,
        const std::shared_ptr<nixlTerminalCapabilityAdapter> &adapter) {
        return std::shared_ptr<nixlTerminalEventSubscriptionH>(
            new nixlTerminalEventSubscriptionH(
                1,
                1,
                {nixl_terminal_event_kind_t::CAPABILITY, 9, 30, 5, true},
                nullptr,
                std::move(backend_subscription),
                adapter));
    }

    static void
    bind(const std::shared_ptr<nixlTerminalEventSubscriptionH> &handle,
         const std::shared_ptr<nixlTerminalTransferAdapter> &adapter) {
        adapter->bindTerminalCallback([weak_handle = std::weak_ptr(handle)]() noexcept {
            if (const std::shared_ptr<nixlTerminalEventSubscriptionH> retained =
                    weak_handle.lock();
                retained != nullptr) {
                retained->markTerminal();
            }
        });
    }

    static void
    bind(const std::shared_ptr<nixlTerminalEventSubscriptionH> &handle,
         const std::shared_ptr<nixlTerminalCapabilityAdapter> &adapter) {
        adapter->bindTerminalCallback([weak_handle = std::weak_ptr(handle)]() noexcept {
            if (const std::shared_ptr<nixlTerminalEventSubscriptionH> retained =
                    weak_handle.lock();
                retained != nullptr) {
                retained->markTerminal();
            }
        });
    }

    static nixl_status_t
    requestCancellation(const std::shared_ptr<nixlTerminalEventSubscriptionH> &handle) {
        return handle->requestCancellation();
    }

    static nixl_terminal_subscription_info_t
    snapshot(const std::shared_ptr<nixlTerminalEventSubscriptionH> &handle) {
        return handle->snapshot();
    }

    static nixlBackendEventSubscriptionInventory
    backendInventory(const std::shared_ptr<nixlTerminalEventSubscriptionH> &handle) {
        return handle->backendInventory();
    }
};

namespace {

class testBackendSubscription final : public nixlBackendEventSubscription {
public:
    explicit testBackendSubscription(
        std::function<nixl_status_t()> cancel = []() { return NIXL_SUCCESS; })
        : cancel_(std::move(cancel)) {}

    nixl_status_t
    cancel() noexcept override {
        ++cancelCount;
        return cancel_();
    }

    void
    queryInventory(nixlBackendEventSubscriptionInventory &inventory) const noexcept override {
        inventory = inventory_;
    }

    std::atomic<int> cancelCount = 0;
    nixlBackendEventSubscriptionInventory inventory_;

private:
    std::function<nixl_status_t()> cancel_;
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

TEST(TerminalEventApi, MismatchedTransferPublicationFailsChannelAndLifecycle) {
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
    const auto handle = nixlTerminalEventSubscriptionTestPeer::makeTransfer(
        std::move(backend_subscription), adapter);
    nixlTerminalEventSubscriptionTestPeer::bind(handle, adapter);

    adapter->publish({{31, 5}, NIXL_SUCCESS, 1234});

    const nixl::terminal_channel_inventory_t inventory = channel.inventory();
    EXPECT_TRUE(nixl::hasTerminalChannelFatal(
        inventory.health.fatal, nixl::terminal_channel_fatal_t::INVALID_PUBLICATION));
    EXPECT_EQ(inventory.activeSubscriptions, 0U);
    EXPECT_FALSE(nixlTerminalEventSubscriptionTestPeer::snapshot(handle).active);
}

TEST(TerminalEventApi, InvalidTransferStatusFailsChannelAndLifecycle) {
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
    const auto handle = nixlTerminalEventSubscriptionTestPeer::makeTransfer(
        std::move(backend_subscription), adapter);
    nixlTerminalEventSubscriptionTestPeer::bind(handle, adapter);

    adapter->publish({{30, 5}, NIXL_IN_PROG, 1234});

    const nixl::terminal_channel_inventory_t inventory = channel.inventory();
    EXPECT_TRUE(nixl::hasTerminalChannelFatal(
        inventory.health.fatal, nixl::terminal_channel_fatal_t::INVALID_PUBLICATION));
    EXPECT_EQ(inventory.activeSubscriptions, 0U);
    EXPECT_FALSE(nixlTerminalEventSubscriptionTestPeer::snapshot(handle).active);
}

TEST(TerminalEventApi, ReadyThenFailedAtSameEpochIsTerminal) {
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
    adapter.publish({20, 4, nixl_backend_capability_state_t::FAILED, 1, 101});
    adapter.publish({20, 4, nixl_backend_capability_state_t::RETIRED, 1, 102});

    const nixl::terminal_event_batch_t batch = channel.drain();
    ASSERT_EQ(batch.events.size(), 2U);
    EXPECT_EQ(batch.events[0].epoch, 1U);
    EXPECT_EQ(batch.events[1].epoch, 1U);
    EXPECT_EQ(std::get<nixl::terminal_capability_state_t>(batch.events[1].result),
              nixl::terminal_capability_state_t::FAILED);
    EXPECT_EQ(terminal_count.load(), 1);
    EXPECT_EQ(batch.inventory.activeSubscriptions, 0U);
}

TEST(TerminalEventApi, ReadyThenRetiredAtSameEpochIsTerminal) {
    nixl::terminalEventChannel channel(2);
    const auto channel_subscription = channel.subscribe({
        .kind = nixl::terminal_event_kind_t::CAPABILITY,
        .ownerCookie = 8,
        .identity = 20,
        .generation = 4,
    });
    nixlTerminalCapabilityAdapter adapter(20, 4, channel_subscription);

    adapter.publish({20, 4, nixl_backend_capability_state_t::READY, 21, 100});
    adapter.publish({20, 4, nixl_backend_capability_state_t::RETIRED, 21, 101});

    const nixl::terminal_event_batch_t batch = channel.drain();
    ASSERT_EQ(batch.events.size(), 2U);
    EXPECT_EQ(std::get<nixl::terminal_capability_state_t>(batch.events[1].result),
              nixl::terminal_capability_state_t::RETIRED);
    EXPECT_EQ(batch.events[1].epoch, 21U);
    EXPECT_EQ(batch.inventory.activeSubscriptions, 0U);
}

TEST(TerminalEventApi, ReadyEpochMayAdvanceBeforeTerminalState) {
    nixl::terminalEventChannel channel(3);
    const auto channel_subscription = channel.subscribe({
        .kind = nixl::terminal_event_kind_t::CAPABILITY,
        .ownerCookie = 8,
        .identity = 20,
        .generation = 4,
    });
    nixlTerminalCapabilityAdapter adapter(20, 4, channel_subscription);

    adapter.publish({20, 4, nixl_backend_capability_state_t::READY, 21, 100});
    adapter.publish({20, 4, nixl_backend_capability_state_t::READY, 22, 101});
    adapter.publish({20, 4, nixl_backend_capability_state_t::FAILED, 22, 102});

    const nixl::terminal_event_batch_t batch = channel.drain();
    ASSERT_EQ(batch.events.size(), 3U);
    EXPECT_EQ(batch.events[0].epoch, 21U);
    EXPECT_EQ(batch.events[1].epoch, 22U);
    EXPECT_EQ(batch.events[2].epoch, 22U);
    EXPECT_EQ(batch.inventory.activeSubscriptions, 0U);
}

TEST(TerminalEventApi, NonIncreasingCapabilityEpochFailsChannelAndLifecycle) {
    nixl::terminalEventChannel channel(4);
    const auto channel_subscription = channel.subscribe({
        .kind = nixl::terminal_event_kind_t::CAPABILITY,
        .ownerCookie = 9,
        .identity = 30,
        .generation = 5,
    });
    const auto adapter =
        std::make_shared<nixlTerminalCapabilityAdapter>(30, 5, channel_subscription);
    auto backend_subscription = std::make_unique<testBackendSubscription>();
    const auto handle = nixlTerminalEventSubscriptionTestPeer::makeCapability(
        std::move(backend_subscription), adapter);
    nixlTerminalEventSubscriptionTestPeer::bind(handle, adapter);

    adapter->publish({30, 5, nixl_backend_capability_state_t::READY, 2, 100});
    adapter->publish({30, 5, nixl_backend_capability_state_t::READY, 2, 101});

    const nixl::terminal_channel_inventory_t inventory = channel.inventory();
    EXPECT_TRUE(nixl::hasTerminalChannelFatal(
        inventory.health.fatal, nixl::terminal_channel_fatal_t::INVALID_PUBLICATION));
    EXPECT_EQ(inventory.activeSubscriptions, 0U);
    EXPECT_FALSE(nixlTerminalEventSubscriptionTestPeer::snapshot(handle).active);
}

TEST(TerminalEventApi, ReorderedTerminalEpochFailsChannelAndLifecycle) {
    nixl::terminalEventChannel channel(4);
    const auto channel_subscription = channel.subscribe({
        .kind = nixl::terminal_event_kind_t::CAPABILITY,
        .ownerCookie = 9,
        .identity = 30,
        .generation = 5,
    });
    const auto adapter =
        std::make_shared<nixlTerminalCapabilityAdapter>(30, 5, channel_subscription);
    auto backend_subscription = std::make_unique<testBackendSubscription>();
    const auto handle = nixlTerminalEventSubscriptionTestPeer::makeCapability(
        std::move(backend_subscription), adapter);
    nixlTerminalEventSubscriptionTestPeer::bind(handle, adapter);

    adapter->publish({30, 5, nixl_backend_capability_state_t::READY, 22, 100});
    adapter->publish({30, 5, nixl_backend_capability_state_t::FAILED, 21, 101});

    const nixl::terminal_channel_inventory_t inventory = channel.inventory();
    EXPECT_TRUE(nixl::hasTerminalChannelFatal(
        inventory.health.fatal, nixl::terminal_channel_fatal_t::INVALID_PUBLICATION));
    EXPECT_EQ(inventory.activeSubscriptions, 0U);
    EXPECT_FALSE(nixlTerminalEventSubscriptionTestPeer::snapshot(handle).active);
}

TEST(TerminalEventApi, SynchronousFailedSnapshotSurvivesLateCallbackBinding) {
    nixl::terminalEventChannel channel(1);
    const auto channel_subscription = channel.subscribe({
        .kind = nixl::terminal_event_kind_t::CAPABILITY,
        .ownerCookie = 9,
        .identity = 30,
        .generation = 5,
    });
    const auto adapter =
        std::make_shared<nixlTerminalCapabilityAdapter>(30, 5, channel_subscription);
    adapter->publish({30, 5, nixl_backend_capability_state_t::FAILED, 1, 100});

    auto backend_subscription = std::make_unique<testBackendSubscription>();
    const auto handle = nixlTerminalEventSubscriptionTestPeer::makeCapability(
        std::move(backend_subscription), adapter);
    nixlTerminalEventSubscriptionTestPeer::bind(handle, adapter);

    EXPECT_FALSE(nixlTerminalEventSubscriptionTestPeer::snapshot(handle).active);
    EXPECT_EQ(channel.inventory().activeSubscriptions, 0U);
    const nixl::terminal_event_batch_t batch = channel.drain();
    ASSERT_EQ(batch.events.size(), 1U);
    EXPECT_EQ(std::get<nixl::terminal_capability_state_t>(batch.events[0].result),
              nixl::terminal_capability_state_t::FAILED);
}

TEST(TerminalEventApi, ActiveTransferCancellationRetainsUntilTerminalDelivery) {
    nixl::terminalEventChannel channel(1);
    const auto channel_subscription = channel.subscribe({
        .kind = nixl::terminal_event_kind_t::TRANSFER,
        .ownerCookie = 9,
        .identity = 30,
        .generation = 5,
    });
    const auto adapter = std::make_shared<nixlTerminalTransferAdapter>(
        nixlBackendTransferEventBinding{30, 5}, channel_subscription);
    auto backend_subscription =
        std::make_unique<testBackendSubscription>([]() { return NIXL_IN_PROG; });
    testBackendSubscription *const backend_subscription_ptr = backend_subscription.get();
    const auto handle = nixlTerminalEventSubscriptionTestPeer::makeTransfer(
        std::move(backend_subscription), adapter);
    nixlTerminalEventSubscriptionTestPeer::bind(handle, adapter);

    EXPECT_EQ(nixlTerminalEventSubscriptionTestPeer::requestCancellation(handle),
              NIXL_IN_PROG);
    EXPECT_EQ(nixlTerminalEventSubscriptionTestPeer::requestCancellation(handle),
              NIXL_IN_PROG);
    EXPECT_TRUE(nixlTerminalEventSubscriptionTestPeer::snapshot(handle).active);
    EXPECT_EQ(channel.inventory().activeSubscriptions, 1U);
    EXPECT_EQ(backend_subscription_ptr->cancelCount.load(), 1);

    adapter->publish({{30, 5}, NIXL_ERR_CANCELED, 1000});

    EXPECT_FALSE(nixlTerminalEventSubscriptionTestPeer::snapshot(handle).active);
    EXPECT_EQ(channel.inventory().activeSubscriptions, 0U);
    EXPECT_EQ(nixlTerminalEventSubscriptionTestPeer::requestCancellation(handle),
              NIXL_SUCCESS);
    EXPECT_EQ(backend_subscription_ptr->cancelCount.load(), 1);
}

TEST(TerminalEventApi, SynchronousTransferCancellationCanCompleteInOneCall) {
    nixl::terminalEventChannel channel(1);
    const auto channel_subscription = channel.subscribe({
        .kind = nixl::terminal_event_kind_t::TRANSFER,
        .ownerCookie = 9,
        .identity = 30,
        .generation = 5,
    });
    const auto adapter = std::make_shared<nixlTerminalTransferAdapter>(
        nixlBackendTransferEventBinding{30, 5}, channel_subscription);
    auto backend_subscription = std::make_unique<testBackendSubscription>([adapter]() {
        adapter->publish({{30, 5}, NIXL_ERR_CANCELED, 1000});
        return NIXL_SUCCESS;
    });
    testBackendSubscription *const backend_subscription_ptr = backend_subscription.get();
    const auto handle = nixlTerminalEventSubscriptionTestPeer::makeTransfer(
        std::move(backend_subscription), adapter);
    nixlTerminalEventSubscriptionTestPeer::bind(handle, adapter);

    EXPECT_EQ(nixlTerminalEventSubscriptionTestPeer::requestCancellation(handle),
              NIXL_SUCCESS);
    EXPECT_FALSE(nixlTerminalEventSubscriptionTestPeer::snapshot(handle).active);
    EXPECT_EQ(channel.inventory().activeSubscriptions, 0U);
    EXPECT_EQ(backend_subscription_ptr->cancelCount.load(), 1);
}

TEST(TerminalEventApi, CapabilityCancellationMayTerminateImmediately) {
    nixl::terminalEventChannel channel(1);
    const auto channel_subscription = channel.subscribe({
        .kind = nixl::terminal_event_kind_t::CAPABILITY,
        .ownerCookie = 9,
        .identity = 30,
        .generation = 5,
    });
    const auto adapter =
        std::make_shared<nixlTerminalCapabilityAdapter>(30, 5, channel_subscription);
    auto backend_subscription = std::make_unique<testBackendSubscription>();
    testBackendSubscription *const backend_subscription_ptr = backend_subscription.get();
    const auto handle = nixlTerminalEventSubscriptionTestPeer::makeCapability(
        std::move(backend_subscription), adapter);
    nixlTerminalEventSubscriptionTestPeer::bind(handle, adapter);

    EXPECT_EQ(nixlTerminalEventSubscriptionTestPeer::requestCancellation(handle),
              NIXL_SUCCESS);
    EXPECT_FALSE(nixlTerminalEventSubscriptionTestPeer::snapshot(handle).active);
    EXPECT_EQ(channel.inventory().activeSubscriptions, 0U);
    EXPECT_EQ(backend_subscription_ptr->cancelCount.load(), 1);
}

TEST(TerminalEventApi, QueryAndTerminalDeliveryAreRaceSafe) {
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
    const auto handle = nixlTerminalEventSubscriptionTestPeer::makeTransfer(
        std::move(backend_subscription), adapter);
    nixlTerminalEventSubscriptionTestPeer::bind(handle, adapter);
    std::barrier start(2);

    std::thread publisher([&]() {
        start.arrive_and_wait();
        adapter->publish({{30, 5}, NIXL_SUCCESS, 1000});
    });
    start.arrive_and_wait();
    for (int iteration = 0; iteration < 1000; ++iteration) {
        const nixl_terminal_subscription_info_t info =
            nixlTerminalEventSubscriptionTestPeer::snapshot(handle);
        EXPECT_EQ(info.identity, 30U);
        EXPECT_EQ(info.generation, 5U);
    }
    publisher.join();

    EXPECT_FALSE(nixlTerminalEventSubscriptionTestPeer::snapshot(handle).active);
    EXPECT_EQ(channel.inventory().activeSubscriptions, 0U);
}

TEST(TerminalEventApi, BackendLifecycleInventoryKeepsDistinctCounts) {
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
    backend_subscription->inventory_ = {
        .backendProducers = 2,
        .activeCallbackSlots = 3,
        .queuedOwnerContinuations = 4,
    };
    const auto handle = nixlTerminalEventSubscriptionTestPeer::makeTransfer(
        std::move(backend_subscription), adapter);

    const nixlBackendEventSubscriptionInventory inventory =
        nixlTerminalEventSubscriptionTestPeer::backendInventory(handle);
    EXPECT_EQ(inventory.backendProducers, 2U);
    EXPECT_EQ(inventory.activeCallbackSlots, 3U);
    EXPECT_EQ(inventory.queuedOwnerContinuations, 4U);
}

} // namespace
