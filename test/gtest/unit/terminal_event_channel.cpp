/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "terminal_event_channel.h"

#include <barrier>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <poll.h>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

namespace nixl {
namespace {

    [[nodiscard]] terminal_event_binding_t
    transferBinding(std::uint64_t cookie, std::uint64_t identity, std::uint64_t generation) {
        return {
            .kind = terminal_event_kind_t::TRANSFER,
            .ownerCookie = cookie,
            .identity = identity,
            .generation = generation,
        };
    }

    [[nodiscard]] terminal_event_binding_t
    capabilityBinding(std::uint64_t cookie, std::uint64_t identity, std::uint64_t generation) {
        return {
            .kind = terminal_event_kind_t::CAPABILITY,
            .ownerCookie = cookie,
            .identity = identity,
            .generation = generation,
        };
    }

    [[nodiscard]] int
    pollChannel(const terminalEventChannel &channel, int timeout_ms) {
        pollfd descriptor = {
            .fd = channel.fileno(),
            .events = POLLIN,
            .revents = 0,
        };
        return ::poll(&descriptor, 1, timeout_ms);
    }

    TEST(TerminalEventChannel, PreservesPublicationOrderAndEventFields) {
        terminalEventChannel channel(4);
        const auto transfer = channel.subscribe(transferBinding(11, 101, 7));
        const auto capability = channel.subscribe(capabilityBinding(12, 102, 8));
        ASSERT_NE(transfer, nullptr);
        ASSERT_NE(capability, nullptr);

        EXPECT_EQ(transfer->publishTransfer(NIXL_SUCCESS, 1'001),
                  terminal_event_publish_result_t::PUBLISHED);
        EXPECT_EQ(capability->publishCapability(terminal_capability_state_t::READY, 19, 1'002),
                  terminal_event_publish_result_t::PUBLISHED);

        const terminal_event_batch_t batch = channel.drain();
        ASSERT_EQ(batch.events.size(), 2U);
        EXPECT_EQ(batch.events[0].kind, terminal_event_kind_t::TRANSFER);
        EXPECT_EQ(batch.events[0].ownerCookie, 11U);
        EXPECT_EQ(batch.events[0].identity, 101U);
        EXPECT_EQ(batch.events[0].generation, 7U);
        EXPECT_EQ(std::get<nixl_status_t>(batch.events[0].result), NIXL_SUCCESS);
        EXPECT_EQ(batch.events[0].epoch, 0U);
        EXPECT_EQ(batch.events[0].nativeTimestampNs, 1'001U);
        EXPECT_EQ(batch.events[1].kind, terminal_event_kind_t::CAPABILITY);
        EXPECT_EQ(std::get<terminal_capability_state_t>(batch.events[1].result),
                  terminal_capability_state_t::READY);
        EXPECT_EQ(batch.events[1].epoch, 19U);
        EXPECT_EQ(batch.events[1].nativeTimestampNs, 1'002U);
    }

    TEST(TerminalEventChannel, CoalescesEventfdWakeupsAndDrainsExactlyOnce) {
        terminalEventChannel channel(4);
        const auto capability = channel.subscribe(capabilityBinding(21, 201, 1));
        ASSERT_NE(capability, nullptr);

        EXPECT_EQ(capability->publishCapability(terminal_capability_state_t::READY, 1),
                  terminal_event_publish_result_t::PUBLISHED);
        EXPECT_EQ(capability->publishCapability(terminal_capability_state_t::READY, 2),
                  terminal_event_publish_result_t::PUBLISHED);
        EXPECT_EQ(capability->publishCapability(terminal_capability_state_t::RETIRED, 3),
                  terminal_event_publish_result_t::PUBLISHED);
        EXPECT_EQ(pollChannel(channel, 0), 1);

        const terminal_event_batch_t batch = channel.drain();
        EXPECT_EQ(batch.wakeCount, 3U);
        EXPECT_EQ(batch.events.size(), 3U);
        EXPECT_EQ(pollChannel(channel, 0), 0);

        const terminal_event_batch_t empty_batch = channel.drain();
        EXPECT_EQ(empty_batch.wakeCount, 0U);
        EXPECT_TRUE(empty_batch.events.empty());
    }

    TEST(TerminalEventChannel, EventfdIsNonblockingCloseOnExecAndBorrowed) {
        terminalEventChannel channel(1);
        const int descriptor = channel.fileno();
        ASSERT_GE(descriptor, 0);
        EXPECT_NE(::fcntl(descriptor, F_GETFL) & O_NONBLOCK, 0);
        EXPECT_NE(::fcntl(descriptor, F_GETFD) & FD_CLOEXEC, 0);
        EXPECT_EQ(channel.fileno(), descriptor);
    }

    TEST(TerminalEventChannel, CapacityOverflowIsStickyFatalAndWakesConsumer) {
        terminalEventChannel channel(2);
        const auto first = channel.subscribe(transferBinding(31, 301, 1));
        const auto second = channel.subscribe(transferBinding(32, 302, 1));
        const auto overflow = channel.subscribe(transferBinding(33, 303, 1));
        ASSERT_NE(first, nullptr);
        ASSERT_NE(second, nullptr);
        ASSERT_NE(overflow, nullptr);

        EXPECT_EQ(first->publishTransfer(NIXL_SUCCESS), terminal_event_publish_result_t::PUBLISHED);
        EXPECT_EQ(second->publishTransfer(NIXL_SUCCESS),
                  terminal_event_publish_result_t::PUBLISHED);
        EXPECT_EQ(overflow->publishTransfer(NIXL_SUCCESS),
                  terminal_event_publish_result_t::CHANNEL_FATAL);
        EXPECT_EQ(overflow->publishTransfer(NIXL_SUCCESS),
                  terminal_event_publish_result_t::ALREADY_PUBLISHED);
        EXPECT_EQ(pollChannel(channel, 0), 1);

        const terminal_event_batch_t batch = channel.drain();
        EXPECT_EQ(batch.events.size(), 2U);
        EXPECT_EQ(batch.wakeCount, 3U);
        EXPECT_TRUE(hasTerminalChannelFatal(batch.inventory.health.fatal,
                                            terminal_channel_fatal_t::QUEUE_OVERFLOW));
        EXPECT_FALSE(batch.inventory.acceptingSubscriptions);
        EXPECT_EQ(channel.subscribe(transferBinding(34, 304, 1)), nullptr);
        EXPECT_TRUE(hasTerminalChannelFatal(channel.inventory().health.fatal,
                                            terminal_channel_fatal_t::QUEUE_OVERFLOW));
    }

    TEST(TerminalEventChannel, TransferPublicationAndConsumptionAreTakeOnce) {
        terminalEventChannel channel(2);
        const auto transfer = channel.subscribe(transferBinding(41, 401, 3));
        ASSERT_NE(transfer, nullptr);

        EXPECT_EQ(transfer->publishTransfer(NIXL_ERR_CANCELED),
                  terminal_event_publish_result_t::PUBLISHED);
        EXPECT_EQ(transfer->publishTransfer(NIXL_SUCCESS),
                  terminal_event_publish_result_t::ALREADY_PUBLISHED);

        const std::optional<terminal_event_t> event = channel.take();
        ASSERT_TRUE(event.has_value());
        EXPECT_EQ(std::get<nixl_status_t>(event->result), NIXL_ERR_CANCELED);
        EXPECT_GT(event->nativeTimestampNs, 0U);
        EXPECT_FALSE(channel.take().has_value());
        EXPECT_EQ(pollChannel(channel, 0), 0);
    }

    TEST(TerminalEventChannel, RejectsNonterminalOrMismatchedPublicationWithoutConsumingArm) {
        terminalEventChannel channel(2);
        const auto transfer = channel.subscribe(transferBinding(51, 501, 1));
        const auto capability = channel.subscribe(capabilityBinding(52, 502, 1));
        ASSERT_NE(transfer, nullptr);
        ASSERT_NE(capability, nullptr);

        EXPECT_EQ(transfer->publishTransfer(NIXL_ERR_NOT_READY),
                  terminal_event_publish_result_t::INVALID_EVENT);
        EXPECT_EQ(transfer->publishCapability(terminal_capability_state_t::READY, 1),
                  terminal_event_publish_result_t::INVALID_EVENT);
        EXPECT_EQ(transfer->publishTransfer(NIXL_SUCCESS),
                  terminal_event_publish_result_t::PUBLISHED);
        EXPECT_EQ(capability->publishTransfer(NIXL_SUCCESS),
                  terminal_event_publish_result_t::INVALID_EVENT);
        EXPECT_EQ(capability->publishCapability(terminal_capability_state_t::READY, 0),
                  terminal_event_publish_result_t::INVALID_EVENT);
        EXPECT_EQ(capability->publishCapability(terminal_capability_state_t::READY, 2),
                  terminal_event_publish_result_t::PUBLISHED);
    }

    TEST(TerminalEventChannel, ConcurrentProducersPreservePerSubscriptionOrder) {
        constexpr std::size_t producer_count = 8;
        constexpr std::size_t events_per_producer = 128;
        terminalEventChannel channel(producer_count * events_per_producer);
        std::vector<std::shared_ptr<terminalEventChannel::subscription>> subscriptions;
        subscriptions.reserve(producer_count);
        for (std::size_t producer = 0; producer < producer_count; ++producer) {
            subscriptions.push_back(
                channel.subscribe(capabilityBinding(60 + producer, 600 + producer, 1)));
            ASSERT_NE(subscriptions.back(), nullptr);
        }

        std::barrier start(static_cast<std::ptrdiff_t>(producer_count));
        std::vector<std::thread> producers;
        producers.reserve(producer_count);
        for (std::size_t producer = 0; producer < producer_count; ++producer) {
            producers.emplace_back([&, producer]() {
                start.arrive_and_wait();
                for (std::size_t event = 1; event <= events_per_producer; ++event) {
                    EXPECT_EQ(subscriptions[producer]->publishCapability(
                                  terminal_capability_state_t::READY, event),
                              terminal_event_publish_result_t::PUBLISHED);
                }
            });
        }
        for (std::thread &producer : producers) {
            producer.join();
        }

        const terminal_event_batch_t batch = channel.drain();
        ASSERT_EQ(batch.events.size(), producer_count * events_per_producer);
        EXPECT_EQ(batch.wakeCount, producer_count * events_per_producer);
        std::unordered_map<std::uint64_t, std::uint64_t> last_epoch;
        for (const terminal_event_t &event : batch.events) {
            EXPECT_EQ(event.epoch, ++last_epoch[event.identity]);
        }
    }

    TEST(TerminalEventChannel, CloseWithActiveSubscriptionsFailsClosedAndTracksInventory) {
        terminalEventChannel channel(2);
        const auto subscription = channel.subscribe(transferBinding(71, 701, 1));
        ASSERT_NE(subscription, nullptr);
        EXPECT_EQ(channel.inventory().activeSubscriptions, 1U);

        EXPECT_EQ(channel.close(), terminal_channel_close_result_t::ACTIVE_SUBSCRIPTIONS);
        terminal_channel_inventory_t inventory = channel.inventory();
        EXPECT_EQ(inventory.activeSubscriptions, 1U);
        EXPECT_FALSE(inventory.acceptingSubscriptions);
        EXPECT_FALSE(inventory.closed);
        EXPECT_TRUE(hasTerminalChannelFatal(
            inventory.health.fatal, terminal_channel_fatal_t::ACTIVE_SUBSCRIPTIONS_ON_CLOSE));
        EXPECT_EQ(pollChannel(channel, 0), 1);
        EXPECT_EQ(subscription->publishTransfer(NIXL_SUCCESS),
                  terminal_event_publish_result_t::CHANNEL_FATAL);

        subscription->release();
        inventory = channel.inventory();
        EXPECT_EQ(inventory.activeSubscriptions, 0U);
        EXPECT_TRUE(inventory.closed);
        EXPECT_EQ(channel.close(), terminal_channel_close_result_t::ALREADY_CLOSED);
    }

    TEST(TerminalEventChannel, CleanCloseDoesNotSignalTheDataEventfd) {
        terminalEventChannel channel(1);
        EXPECT_EQ(channel.close(), terminal_channel_close_result_t::CLOSED);
        EXPECT_EQ(pollChannel(channel, 0), 0);
    }

    TEST(TerminalEventChannel, SubscriptionKeepsCallbackStateAliveAfterWrapperDestruction) {
        std::shared_ptr<terminalEventChannel::subscription> subscription;
        {
            terminalEventChannel channel(1);
            subscription = channel.subscribe(transferBinding(81, 801, 1));
            ASSERT_NE(subscription, nullptr);
        }

        EXPECT_EQ(subscription->publishTransfer(NIXL_SUCCESS),
                  terminal_event_publish_result_t::CHANNEL_FATAL);
        subscription->release();
    }

    TEST(TerminalEventChannel, RejectsZeroCapacity) {
        EXPECT_THROW(static_cast<void>(terminalEventChannel(0)), std::invalid_argument);
    }

    TEST(TerminalEventChannel, RejectsInexactSubscriptionBindings) {
        terminalEventChannel channel(1);
        EXPECT_THROW(static_cast<void>(channel.subscribe(transferBinding(0, 1, 1))),
                     std::invalid_argument);
        EXPECT_THROW(static_cast<void>(channel.subscribe(transferBinding(1, 0, 1))),
                     std::invalid_argument);
        EXPECT_THROW(static_cast<void>(channel.subscribe(transferBinding(1, 1, 0))),
                     std::invalid_argument);
        EXPECT_EQ(channel.inventory().activeSubscriptions, 0U);
    }

} // namespace
} // namespace nixl
