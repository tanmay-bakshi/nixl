/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "terminal_owner_producer.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

class nixlTerminalOwnerProducerTestPeer {
public:
    static std::shared_ptr<nixlTerminalOwnerTransferAdapter>
    arm(nixlTerminalOwnerProducerH &producer,
        const nixl_terminal_owner_binding_digest_t &owner_binding,
        nixlBackendTransferEventBinding transfer_binding) {
        if (producer.beginSubscription(owner_binding) != NIXL_SUCCESS) {
            return nullptr;
        }
        const auto adapter = std::make_shared<nixlTerminalOwnerTransferAdapter>(
            transfer_binding, owner_binding, producer.state_);
        producer.subscriptionRegistrationSucceeded(owner_binding);
        return adapter;
    }

    static std::shared_ptr<nixlTerminalOwnerTransferAdapter>
    beginRegistration(nixlTerminalOwnerProducerH &producer,
                      const nixl_terminal_owner_binding_digest_t &owner_binding,
                      nixlBackendTransferEventBinding transfer_binding) {
        if (producer.beginSubscription(owner_binding) != NIXL_SUCCESS) {
            return nullptr;
        }
        return std::make_shared<nixlTerminalOwnerTransferAdapter>(
            transfer_binding, owner_binding, producer.state_);
    }
};

namespace {

constexpr std::uint16_t source_native_terminal_event = 13;
constexpr std::uint16_t source_request_failed_event = 21;

struct fake_owner_t {
    static int
    submit(void *context, const sglang_terminal_owner_producer_event_v1 *event) noexcept {
        auto &owner = *static_cast<fake_owner_t *>(context);
        const std::lock_guard lock(owner.mutex);
        if (owner.submitStatus != 0) {
            return owner.submitStatus;
        }
        owner.events.push_back(*event);
        return 0;
    }

    static int
    retire(void *context) noexcept {
        auto &owner = *static_cast<fake_owner_t *>(context);
        ++owner.retireCount;
        return owner.retireStatus;
    }

    static int
    join(void *context, std::uint64_t timeout_ns) noexcept {
        auto &owner = *static_cast<fake_owner_t *>(context);
        owner.joinTimeoutNs = timeout_ns;
        ++owner.joinCount;
        return owner.joinStatus;
    }

    [[nodiscard]] sglang_terminal_owner_producer_api_v1
    api() const noexcept {
        return {
            .abi_version = SGLANG_TERMINAL_OWNER_PRODUCER_ABI_VERSION,
            .struct_size = sizeof(sglang_terminal_owner_producer_api_v1),
            .event_struct_size = sizeof(sglang_terminal_owner_producer_event_v1),
            .flags = SGLANG_TERMINAL_OWNER_PRODUCER_REQUIRED_FLAGS,
            .submit = submit,
            .retire = retire,
            .join = join,
        };
    }

    std::mutex mutex;
    std::vector<sglang_terminal_owner_producer_event_v1> events;
    std::atomic<int> retireCount = 0;
    std::atomic<int> joinCount = 0;
    std::atomic<std::uint64_t> joinTimeoutNs = 0;
    int submitStatus = 0;
    int retireStatus = 0;
    int joinStatus = 0;
};

[[nodiscard]] nixl_terminal_owner_binding_digest_t
makeBinding(std::uint8_t seed) {
    nixl_terminal_owner_binding_digest_t result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = static_cast<std::uint8_t>(seed + index);
    }
    return result;
}

void
expectBinding(const sglang_terminal_owner_producer_event_v1 &event,
              const nixl_terminal_owner_binding_digest_t &binding) {
    EXPECT_EQ(std::memcmp(event.binding_digest, binding.data(), binding.size()), 0);
}

void
retireAndClose(nixlTerminalOwnerProducerH &producer, fake_owner_t &owner) {
    producer.stopAdmission();
    EXPECT_EQ(producer.join(2'000'000'000ULL), NIXL_SUCCESS);
    EXPECT_EQ(producer.close(), NIXL_SUCCESS);
    EXPECT_EQ(owner.retireCount.load(), 1);
    EXPECT_EQ(owner.joinCount.load(), 1);
    EXPECT_EQ(owner.joinTimeoutNs.load(), 2'000'000'000ULL);
}

TEST(TerminalOwnerProducer, SuccessSubmitsNativeTerminalDirectly) {
    fake_owner_t owner;
    const sglang_terminal_owner_producer_api_v1 api = owner.api();
    nixlTerminalOwnerProducerH producer(&api, &owner);
    const nixl_terminal_owner_binding_digest_t owner_binding = makeBinding(11);
    constexpr nixlBackendTransferEventBinding transfer_binding{101, 7};
    const auto adapter =
        nixlTerminalOwnerProducerTestPeer::arm(producer, owner_binding, transfer_binding);
    ASSERT_NE(adapter, nullptr);
    std::atomic<int> terminal_count = 0;
    adapter->bindTerminalCallback([&terminal_count]() { ++terminal_count; });

    adapter->publish({
        .binding = transfer_binding,
        .status = NIXL_SUCCESS,
        .nativeTimestampNs = 1'234'567,
    });

    ASSERT_EQ(owner.events.size(), 1U);
    const sglang_terminal_owner_producer_event_v1 &event = owner.events.front();
    expectBinding(event, owner_binding);
    EXPECT_EQ(event.event_kind, source_native_terminal_event);
    EXPECT_EQ(event.enqueued_ns, 1'234'567U);
    EXPECT_EQ(event.reason_code, 0);
    EXPECT_EQ(event.backend_status, NIXL_SUCCESS);
    EXPECT_EQ(event.has_receipt, 0);
    EXPECT_EQ(terminal_count.load(), 1);

    const nixl_terminal_owner_producer_inventory_t inventory = producer.inventory();
    EXPECT_EQ(inventory.activeCallbacks, 0U);
    EXPECT_EQ(inventory.activeRegistrations, 0U);
    EXPECT_EQ(inventory.totalSubscriptions, 1U);
    EXPECT_EQ(inventory.totalDelivered, 1U);
    EXPECT_EQ(inventory.successfulTerminalEvents, 1U);
    EXPECT_EQ(inventory.failureTerminalEvents, 0U);
    EXPECT_EQ(inventory.fatal, nixl_terminal_owner_producer_fatal_t::NONE);
    retireAndClose(producer, owner);
}

TEST(TerminalOwnerProducer, BackendFailureSubmitsLocalRequestFailure) {
    fake_owner_t owner;
    const sglang_terminal_owner_producer_api_v1 api = owner.api();
    nixlTerminalOwnerProducerH producer(&api, &owner);
    const nixl_terminal_owner_binding_digest_t owner_binding = makeBinding(37);
    constexpr nixlBackendTransferEventBinding transfer_binding{203, 9};
    const auto adapter =
        nixlTerminalOwnerProducerTestPeer::arm(producer, owner_binding, transfer_binding);
    ASSERT_NE(adapter, nullptr);
    adapter->bindTerminalCallback([]() {});

    adapter->publish({
        .binding = transfer_binding,
        .status = NIXL_ERR_REMOTE_DISCONNECT,
        .nativeTimestampNs = 2'345'678,
    });

    ASSERT_EQ(owner.events.size(), 1U);
    const sglang_terminal_owner_producer_event_v1 &event = owner.events.front();
    expectBinding(event, owner_binding);
    EXPECT_EQ(event.event_kind, source_request_failed_event);
    EXPECT_EQ(event.reason_code, NIXL_TERMINAL_OWNER_REASON_TRANSFER_FAILED);
    EXPECT_EQ(event.backend_status, NIXL_ERR_REMOTE_DISCONNECT);
    const nixl_terminal_owner_producer_inventory_t inventory = producer.inventory();
    EXPECT_EQ(inventory.totalDelivered, 1U);
    EXPECT_EQ(inventory.successfulTerminalEvents, 0U);
    EXPECT_EQ(inventory.failureTerminalEvents, 1U);
    EXPECT_EQ(inventory.fatal, nixl_terminal_owner_producer_fatal_t::NONE);
    retireAndClose(producer, owner);
}

TEST(TerminalOwnerProducer, PendingTerminalWaitsForSubscriptionAuthority) {
    fake_owner_t owner;
    const sglang_terminal_owner_producer_api_v1 api = owner.api();
    nixlTerminalOwnerProducerH producer(&api, &owner);
    const nixl_terminal_owner_binding_digest_t owner_binding = makeBinding(51);
    constexpr nixlBackendTransferEventBinding transfer_binding{307, 11};
    const auto adapter =
        nixlTerminalOwnerProducerTestPeer::arm(producer, owner_binding, transfer_binding);
    ASSERT_NE(adapter, nullptr);

    adapter->publish({transfer_binding, NIXL_SUCCESS, 3'456'789});
    EXPECT_TRUE(owner.events.empty());
    EXPECT_EQ(producer.inventory().activeCallbacks, 1U);

    std::atomic<int> terminal_count = 0;
    adapter->bindTerminalCallback([&terminal_count]() { ++terminal_count; });
    EXPECT_EQ(owner.events.size(), 1U);
    EXPECT_EQ(terminal_count.load(), 1);
    EXPECT_EQ(producer.inventory().activeCallbacks, 0U);
    retireAndClose(producer, owner);
}

TEST(TerminalOwnerProducer, ConcurrentCallbacksLeaveOrderingToOwnerSubmission) {
    fake_owner_t owner;
    const sglang_terminal_owner_producer_api_v1 api = owner.api();
    nixlTerminalOwnerProducerH producer(&api, &owner);
    constexpr std::size_t count = 32;
    std::vector<nixl_terminal_owner_binding_digest_t> bindings;
    std::vector<std::shared_ptr<nixlTerminalOwnerTransferAdapter>> adapters;
    bindings.reserve(count);
    adapters.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        bindings.push_back(makeBinding(static_cast<std::uint8_t>(index + 1)));
        const auto adapter =
            nixlTerminalOwnerProducerTestPeer::arm(producer, bindings.back(), {400 + index, 13});
        ASSERT_NE(adapter, nullptr);
        adapter->bindTerminalCallback([]() {});
        adapters.push_back(adapter);
    }

    std::vector<std::thread> callbacks;
    callbacks.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        callbacks.emplace_back([&, index]() {
            adapters[index]->publish({
                .binding = {400 + index, 13},
                .status = NIXL_SUCCESS,
                .nativeTimestampNs = 4'000'000 + index,
            });
        });
    }
    for (std::thread &callback : callbacks) {
        callback.join();
    }

    ASSERT_EQ(owner.events.size(), count);
    for (const sglang_terminal_owner_producer_event_v1 &event : owner.events) {
        EXPECT_EQ(event.event_kind, source_native_terminal_event);
        EXPECT_EQ(event.struct_size, sizeof(event));
    }
    const nixl_terminal_owner_producer_inventory_t inventory = producer.inventory();
    EXPECT_EQ(inventory.totalSubscriptions, count);
    EXPECT_EQ(inventory.totalDelivered, count);
    EXPECT_EQ(inventory.activeCallbacks, 0U);
    EXPECT_EQ(inventory.fatal, nixl_terminal_owner_producer_fatal_t::NONE);
    retireAndClose(producer, owner);
}

TEST(TerminalOwnerProducer, CloseWithLiveCallbackFailsClosed) {
    fake_owner_t owner;
    const sglang_terminal_owner_producer_api_v1 api = owner.api();
    nixlTerminalOwnerProducerH producer(&api, &owner);
    const nixl_terminal_owner_binding_digest_t owner_binding = makeBinding(73);
    constexpr nixlBackendTransferEventBinding transfer_binding{509, 17};
    const auto adapter =
        nixlTerminalOwnerProducerTestPeer::arm(producer, owner_binding, transfer_binding);
    ASSERT_NE(adapter, nullptr);
    adapter->bindTerminalCallback([]() {});

    EXPECT_EQ(producer.join(1'000'000), NIXL_IN_PROG);
    EXPECT_EQ(producer.close(), NIXL_ERR_NOT_ALLOWED);
    EXPECT_EQ(producer.inventory().fatal,
              nixl_terminal_owner_producer_fatal_t::CLOSE_WITH_ACTIVE_CALLBACKS);

    adapter->publish({transfer_binding, NIXL_SUCCESS, 5'678'901});
    EXPECT_EQ(owner.events.size(), 1U);
    EXPECT_EQ(producer.inventory().activeCallbacks, 0U);
}

TEST(TerminalOwnerProducer, RegistrationAbortReleasesAllProducerAuthority) {
    fake_owner_t owner;
    const sglang_terminal_owner_producer_api_v1 api = owner.api();
    nixlTerminalOwnerProducerH producer(&api, &owner);
    const nixl_terminal_owner_binding_digest_t owner_binding = makeBinding(89);
    const auto adapter =
        nixlTerminalOwnerProducerTestPeer::beginRegistration(producer, owner_binding, {601, 19});
    ASSERT_NE(adapter, nullptr);

    adapter->abortRegistration(NIXL_ERR_NOT_SUPPORTED);

    const nixl_terminal_owner_producer_inventory_t inventory = producer.inventory();
    EXPECT_EQ(inventory.registeringBindings, 0U);
    EXPECT_EQ(inventory.submittedBindings, 0U);
    EXPECT_EQ(inventory.activeCallbacks, 0U);
    EXPECT_EQ(inventory.activeRegistrations, 0U);
    EXPECT_EQ(inventory.totalSubscriptions, 0U);
    EXPECT_EQ(inventory.fatal, nixl_terminal_owner_producer_fatal_t::NONE);
    retireAndClose(producer, owner);
}

TEST(TerminalOwnerProducer, InvalidTransitionFailsRequestAndPoisonsProducer) {
    fake_owner_t owner;
    const sglang_terminal_owner_producer_api_v1 api = owner.api();
    nixlTerminalOwnerProducerH producer(&api, &owner);
    const nixl_terminal_owner_binding_digest_t owner_binding = makeBinding(101);
    constexpr nixlBackendTransferEventBinding transfer_binding{701, 23};
    const auto adapter =
        nixlTerminalOwnerProducerTestPeer::arm(producer, owner_binding, transfer_binding);
    ASSERT_NE(adapter, nullptr);
    adapter->bindTerminalCallback([]() {});

    adapter->publish({transfer_binding, NIXL_IN_PROG, 6'789'012});

    ASSERT_EQ(owner.events.size(), 1U);
    EXPECT_EQ(owner.events.front().event_kind, source_request_failed_event);
    EXPECT_EQ(owner.events.front().reason_code, NIXL_TERMINAL_OWNER_REASON_INVALID_TRANSITION);
    EXPECT_EQ(producer.inventory().fatal, nixl_terminal_owner_producer_fatal_t::INVALID_TRANSITION);
    EXPECT_EQ(producer.inventory().activeCallbacks, 0U);
}

TEST(TerminalOwnerProducer, RejectsMismatchedOrIncompleteAbi) {
    fake_owner_t owner;
    sglang_terminal_owner_producer_api_v1 api = owner.api();
    ++api.abi_version;
    EXPECT_THROW(nixlTerminalOwnerProducerH(&api, &owner), std::invalid_argument);

    api = owner.api();
    api.flags = SGLANG_TERMINAL_OWNER_PRODUCER_FLAG_OWNER_ASSIGNED_SEQUENCE;
    EXPECT_THROW(nixlTerminalOwnerProducerH(&api, &owner), std::invalid_argument);
}

} // namespace
