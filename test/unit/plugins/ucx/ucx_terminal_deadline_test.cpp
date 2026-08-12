/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "ucx_terminal_deadline.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>

using namespace nixl::ucx;

namespace {

using namespace std::chrono_literals;

void
require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

terminal_deadline_key_t
key(std::uint64_t identity, std::uint64_t generation = 1) {
    return {.handleIdentity = identity, .generation = generation};
}

void
waitUntilOwnerIsAlive(terminal_deadline_owner_t &owner) {
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (!owner.inventory().ownerAlive && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    require(owner.inventory().ownerAlive, "deadline owner did not start");
}

void
testActiveIdentitySnapshotIsAtomicAndExact() {
    terminal_deadline_owner_t owner(4);
    const std::uint64_t anchor = terminal_deadline_owner_t::monotonicTimestampNs();
    const terminal_deadline_key_t first = key(11, 2);
    const terminal_deadline_key_t second = key(7, 3);
    const terminal_deadline_key_t third = key(11, 1);
    require(owner.arm(first, anchor, 1'000'000'000, [](const auto &) { return NIXL_SUCCESS; }) ==
                terminal_deadline_status_t::SUCCESS,
            "first snapshot deadline registration failed");
    require(owner.arm(second, anchor, 1'000'000'000, [](const auto &) { return NIXL_SUCCESS; }) ==
                terminal_deadline_status_t::SUCCESS,
            "second snapshot deadline registration failed");
    require(owner.arm(third, anchor, 1'000'000'000, [](const auto &) { return NIXL_SUCCESS; }) ==
                terminal_deadline_status_t::SUCCESS,
            "third snapshot deadline registration failed");

    terminal_deadline_snapshot_t snapshot = owner.snapshot();
    require(snapshot.inventory.active == 3 && snapshot.activeKeys.size() == 3,
            "active count and exact identity snapshot diverged");
    require(snapshot.activeKeys[0] == second && snapshot.activeKeys[1] == third &&
                snapshot.activeKeys[2] == first,
            "active identity snapshot was incomplete or noncanonical");
    require(owner.arm(first, anchor, 2'000'000'000, [](const auto &) { return NIXL_SUCCESS; }) ==
                terminal_deadline_status_t::DUPLICATE_DEADLINE,
            "duplicate deadline replaced an active identity");

    require(owner.retire(first) == terminal_deadline_status_t::SUCCESS,
            "first snapshot deadline retirement failed");
    snapshot = owner.snapshot();
    require(snapshot.inventory.active == 2 && snapshot.activeKeys == std::vector{second, third},
            "retirement did not atomically remove its exact active identity");
    require(owner.retire(first) == terminal_deadline_status_t::RETIREMENT_WON,
            "duplicate retirement did not report the original winner");
    require(owner.retire(second) == terminal_deadline_status_t::SUCCESS,
            "second snapshot deadline retirement failed");
    require(owner.retire(third) == terminal_deadline_status_t::SUCCESS,
            "third snapshot deadline retirement failed");
    snapshot = owner.snapshot();
    require(snapshot.inventory.active == 0 && snapshot.activeKeys.empty(),
            "empty active inventory retained an identity");
    require(owner.fatalStatus() == NIXL_SUCCESS, "healthy owner exposed a fatal status");
    require(owner.close() == NIXL_SUCCESS, "snapshot owner close failed");
}

void
testExactExpiryAndNoReset() {
    terminal_deadline_owner_t owner(4);
    waitUntilOwnerIsAlive(owner);
    std::mutex mutex;
    std::condition_variable delivered;
    std::size_t expiry_count = 0;
    const std::uint64_t anchor = terminal_deadline_owner_t::monotonicTimestampNs();
    require(owner.arm(key(1),
                      anchor,
                      20'000'000,
                      [&](const terminal_deadline_key_t &expired) {
                          {
                              const std::lock_guard lock(mutex);
                              ++expiry_count;
                          }
                          delivered.notify_all();
                          require(owner.acknowledgeExpiry(expired) ==
                                      terminal_deadline_status_t::SUCCESS,
                                  "expiry acknowledgement failed");
                          return NIXL_SUCCESS;
                      }) == terminal_deadline_status_t::SUCCESS,
            "deadline registration failed");
    require(owner.arm(key(1), anchor + 1, 60'000'000, [](const auto &) { return NIXL_SUCCESS; }) ==
                terminal_deadline_status_t::DUPLICATE_DEADLINE,
            "duplicate registration reset an existing deadline");
    {
        std::unique_lock lock(mutex);
        require(delivered.wait_for(lock, 2s, [&]() { return expiry_count == 1; }),
                "deadline did not expire");
    }
    std::this_thread::sleep_for(20ms);
    require(expiry_count == 1, "deadline expired more than once");
    const terminal_deadline_inventory_t inventory = owner.inventory();
    require(inventory.active == 0 && inventory.expired == 1 && inventory.retired == 0,
            "expired deadline remained live");
    require(owner.snapshot().activeKeys.empty(), "expired identity remained in exact snapshot");
    require(owner.retire(key(1)) == terminal_deadline_status_t::EXPIRY_WON,
            "late retirement did not report the expiry winner");
    require(owner.close() == NIXL_SUCCESS, "clean deadline owner close failed");
}

void
testRetirementWinsBeforeExpiry() {
    terminal_deadline_owner_t owner(2);
    std::atomic<std::size_t> expiry_count = 0;
    const std::uint64_t anchor = terminal_deadline_owner_t::monotonicTimestampNs();
    require(owner.arm(key(2),
                      anchor,
                      50'000'000,
                      [&](const auto &) {
                          ++expiry_count;
                          return NIXL_SUCCESS;
                      }) == terminal_deadline_status_t::SUCCESS,
            "retirement fixture registration failed");
    require(owner.retire(key(2)) == terminal_deadline_status_t::SUCCESS,
            "deadline retirement failed");
    std::this_thread::sleep_for(80ms);
    const terminal_deadline_inventory_t inventory = owner.inventory();
    require(expiry_count.load() == 0 && inventory.active == 0 && inventory.expired == 0 &&
                inventory.retired == 1,
            "retired deadline later expired");
    require(owner.close() == NIXL_SUCCESS, "retired owner close failed");
}

void
testCapacityAndExpiryFailureAreFatal() {
    terminal_deadline_owner_t capacity_owner(1);
    const std::uint64_t anchor = terminal_deadline_owner_t::monotonicTimestampNs();
    require(capacity_owner.arm(
                key(3), anchor, 1'000'000'000, [](const auto &) { return NIXL_SUCCESS; }) ==
                terminal_deadline_status_t::SUCCESS,
            "capacity fixture registration failed");
    require(capacity_owner.arm(
                key(4), anchor, 1'000'000'000, [](const auto &) { return NIXL_SUCCESS; }) ==
                terminal_deadline_status_t::CAPACITY_EXCEEDED,
            "deadline owner exceeded its process bound");
    require(capacity_owner.inventory().fatalStatus == NIXL_ERR_BACKEND,
            "capacity overflow was not process-fatal");
    require(capacity_owner.fatalStatus() == NIXL_ERR_BACKEND,
            "fatal-status accessor omitted capacity failure");
    require(capacity_owner.close() == NIXL_ERR_BACKEND, "capacity-failed owner closed as healthy");

    terminal_deadline_owner_t callback_owner(1);
    require(callback_owner.arm(key(5),
                               terminal_deadline_owner_t::monotonicTimestampNs(),
                               10'000'000,
                               [](const auto &) { return NIXL_ERR_NOT_ALLOWED; }) ==
                terminal_deadline_status_t::SUCCESS,
            "callback failure fixture registration failed");
    const auto failure_deadline = std::chrono::steady_clock::now() + 2s;
    while (callback_owner.inventory().fatalStatus == NIXL_SUCCESS &&
           std::chrono::steady_clock::now() < failure_deadline) {
        std::this_thread::yield();
    }
    require(callback_owner.inventory().fatalStatus == NIXL_ERR_NOT_ALLOWED,
            "expiry enqueue failure was not process-fatal");
    require(callback_owner.close() == NIXL_ERR_NOT_ALLOWED,
            "callback-failed owner closed as healthy");
}

void
testFatalCallbackCanInspectOwner() {
    terminal_deadline_owner_t *observed_owner = nullptr;
    std::promise<void> callback_completed_promise;
    std::future<void> callback_completed = callback_completed_promise.get_future();
    terminal_deadline_owner_t owner(1, [&](nixl_status_t status) {
        require(status == NIXL_ERR_BACKEND, "fatal callback changed exact status");
        require(observed_owner != nullptr, "fatal callback ran before owner publication");
        const terminal_deadline_snapshot_t snapshot = observed_owner->snapshot();
        require(snapshot.inventory.fatalStatus == NIXL_ERR_BACKEND &&
                    snapshot.inventory.active == 1 && snapshot.activeKeys.size() == 1,
                "fatal callback could not inspect exact owner inventory");
        require(observed_owner->fatalStatus() == NIXL_ERR_BACKEND,
                "fatal callback could not query owner status");
        callback_completed_promise.set_value();
    });
    observed_owner = &owner;
    const std::uint64_t anchor = terminal_deadline_owner_t::monotonicTimestampNs();
    require(owner.arm(key(30), anchor, 1'000'000'000, [](const auto &) { return NIXL_SUCCESS; }) ==
                terminal_deadline_status_t::SUCCESS,
            "fatal callback fixture registration failed");
    require(owner.arm(key(31), anchor, 1'000'000'000, [](const auto &) { return NIXL_SUCCESS; }) ==
                terminal_deadline_status_t::CAPACITY_EXCEEDED,
            "fatal callback fixture did not fail at capacity");
    require(callback_completed.wait_for(2s) == std::future_status::ready,
            "fatal callback deadlocked while inspecting its owner");
    require(owner.close() == NIXL_ERR_BACKEND,
            "fatal callback fixture closed without its sticky failure");

    terminal_deadline_owner_t *reactor_owner = nullptr;
    std::promise<void> reactor_callback_completed_promise;
    std::future<void> reactor_callback_completed = reactor_callback_completed_promise.get_future();
    terminal_deadline_owner_t owner_thread_failure(1, [&](nixl_status_t status) {
        require(status == NIXL_ERR_NOT_ALLOWED, "reactor fatal callback changed exact status");
        require(reactor_owner != nullptr, "reactor fatal callback ran before owner publication");
        const terminal_deadline_snapshot_t snapshot = reactor_owner->snapshot();
        require(snapshot.inventory.fatalStatus == NIXL_ERR_NOT_ALLOWED &&
                    snapshot.inventory.active == 1 && snapshot.activeKeys.size() == 1,
                "reactor fatal callback could not inspect exact owner inventory");
        reactor_callback_completed_promise.set_value();
    });
    reactor_owner = &owner_thread_failure;
    const terminal_deadline_key_t reactor_key = key(32);
    require(owner_thread_failure.arm(reactor_key,
                                     terminal_deadline_owner_t::monotonicTimestampNs(),
                                     10'000'000,
                                     [](const auto &) { return NIXL_ERR_NOT_ALLOWED; }) ==
                terminal_deadline_status_t::SUCCESS,
            "reactor fatal callback fixture registration failed");
    require(reactor_callback_completed.wait_for(2s) == std::future_status::ready,
            "reactor fatal callback deadlocked while inspecting its owner");
    require(owner_thread_failure.acknowledgeExpiry(reactor_key) ==
                terminal_deadline_status_t::SUCCESS,
            "reactor fatal callback fixture did not drain its claimed expiry");
    require(owner_thread_failure.close() == NIXL_ERR_NOT_ALLOWED,
            "reactor fatal callback fixture closed without its sticky failure");
}

void
testRetirementAndExpiryHaveOneWinner() {
    constexpr std::size_t population = 256;
    terminal_deadline_owner_t owner(population);
    std::array<std::atomic<std::size_t>, population> expiry_counts{};
    std::array<std::atomic<std::size_t>, population> retirement_counts{};
    std::array<std::atomic<std::size_t>, population> expiry_winner_counts{};
    std::atomic<std::size_t> ambiguous_counts = 0;
    const std::uint64_t anchor = terminal_deadline_owner_t::monotonicTimestampNs();
    for (std::size_t index = 0; index < population; ++index) {
        require(owner.arm(key(index + 100),
                          anchor,
                          1'000'000 + (index % 4) * 250'000,
                          [&, index](const terminal_deadline_key_t &expired) {
                              ++expiry_counts[index];
                              require(owner.acknowledgeExpiry(expired) ==
                                          terminal_deadline_status_t::SUCCESS,
                                      "race expiry acknowledgement failed");
                              return NIXL_SUCCESS;
                          }) == terminal_deadline_status_t::SUCCESS,
                "race fixture registration failed");
    }

    std::thread retirements([&]() {
        for (std::size_t index = 0; index < population; ++index) {
            const terminal_deadline_status_t status = owner.retire(key(index + 100));
            if (status == terminal_deadline_status_t::SUCCESS) {
                ++retirement_counts[index];
            } else if (status == terminal_deadline_status_t::EXPIRY_WON) {
                ++expiry_winner_counts[index];
            } else {
                ++ambiguous_counts;
            }
        }
    });
    retirements.join();
    const auto terminal = std::chrono::steady_clock::now() + 2s;
    auto allResolved = [&]() {
        for (std::size_t index = 0; index < population; ++index) {
            if (expiry_counts[index].load() + retirement_counts[index].load() != 1) {
                return false;
            }
        }
        return true;
    };
    while ((!allResolved() || owner.inventory().active != 0) &&
           std::chrono::steady_clock::now() < terminal) {
        std::this_thread::yield();
    }
    require(owner.inventory().active == 0, "race fixture retained a live deadline");
    require(ambiguous_counts.load() == 0, "deadline race returned an ambiguous loser status");
    for (std::size_t index = 0; index < population; ++index) {
        require(expiry_counts[index].load() + retirement_counts[index].load() == 1,
                "deadline expiry and retirement did not have exactly one winner");
        require(expiry_winner_counts[index].load() == expiry_counts[index].load(),
                "retirement did not observe the exact expiry winner");
    }
    require(owner.close() == NIXL_SUCCESS, "race fixture owner close failed");
}

void
testShutdownWaitsForForcedExpiryDispatch() {
    terminal_deadline_owner_t owner(1);
    waitUntilOwnerIsAlive(owner);
    std::promise<void> expiry_entered_promise;
    std::future<void> expiry_entered = expiry_entered_promise.get_future();
    std::promise<void> release_expiry_promise;
    const std::shared_future<void> release_expiry = release_expiry_promise.get_future().share();
    std::promise<void> expiry_returning_promise;
    std::future<void> expiry_returning = expiry_returning_promise.get_future();
    const terminal_deadline_key_t transfer = key(500);
    require(owner.arm(transfer,
                      terminal_deadline_owner_t::monotonicTimestampNs(),
                      60'000'000'000ULL,
                      [&](const terminal_deadline_key_t &expired) {
                          require(expired == transfer, "forced expiry changed its identity");
                          expiry_entered_promise.set_value();
                          release_expiry.wait();
                          expiry_returning_promise.set_value();
                          return NIXL_SUCCESS;
                      }) == terminal_deadline_status_t::SUCCESS,
            "shutdown fixture deadline registration failed");

    std::future<nixl_status_t> shutdown =
        std::async(std::launch::async, [&]() { return owner.beginShutdown(); });
    require(expiry_entered.wait_for(2s) == std::future_status::ready,
            "shutdown did not force its active deadline due");
    require(shutdown.wait_for(20ms) == std::future_status::timeout,
            "shutdown returned before the forced expiry callback completed");
    release_expiry_promise.set_value();
    require(expiry_returning.wait_for(2s) == std::future_status::ready,
            "forced expiry callback did not finish");
    require(shutdown.wait_for(20ms) == std::future_status::timeout,
            "shutdown returned before terminal disposition acknowledged expiry");
    require(owner.acknowledgeExpiry(transfer) == terminal_deadline_status_t::SUCCESS,
            "forced expiry acknowledgement failed");
    require(shutdown.get() == NIXL_SUCCESS,
            "shutdown did not finish after forced expiry dispatch completed");
    require(owner.snapshot().activeKeys.empty(),
            "forced shutdown expiry retained its exact identity");
    require(owner.close() == NIXL_SUCCESS, "forced-expiry owner close failed");
}

} // namespace

int
main() {
    try {
        testActiveIdentitySnapshotIsAtomicAndExact();
        testExactExpiryAndNoReset();
        testRetirementWinsBeforeExpiry();
        testCapacityAndExpiryFailureAreFatal();
        testFatalCallbackCanInspectOwner();
        testRetirementAndExpiryHaveOneWinner();
        testShutdownWaitsForForcedExpiryDispatch();
    }
    catch (const std::exception &error) {
        std::cerr << "ucx_terminal_deadline_test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "ucx_terminal_deadline_test passed\n";
    return EXIT_SUCCESS;
}
