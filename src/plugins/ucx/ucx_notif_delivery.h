/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_SRC_PLUGINS_UCX_UCX_NOTIF_DELIVERY_H
#define NIXL_SRC_PLUGINS_UCX_UCX_NOTIF_DELIVERY_H

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <nixl_types.h>

#include "ucx_notif_state.h"

namespace nixl::ucx {

struct notif_delivery_key_t {
    std::uint64_t identity = 0;
    std::uint64_t sourceHandleIdentity = 0;
    std::uint64_t sourceGeneration = 0;
    notif_route_key_t route;
    notif_wire_uuid_t capability;
    std::uint64_t capabilityEpoch = 0;
    std::uint64_t connectionIdentity = 0;
    notif_wire_uuid_t receiptWorkerIncarnation;
    std::uint64_t endpointIdentity = 0;

    bool
    operator==(const notif_delivery_key_t &) const = default;
};

struct notif_delivery_receipt_t {
    std::uint64_t identity = 0;
    std::uint64_t sourceHandleIdentity = 0;
    std::uint64_t sourceGeneration = 0;
    notif_route_key_t route;
    notif_wire_uuid_t capability;
    std::uint64_t capabilityEpoch = 0;
    std::uint64_t connectionIdentity = 0;
    notif_wire_uuid_t receiptWorkerIncarnation;
    std::uint64_t endpointIdentity = 0;
};

struct notif_delivery_inventory_t {
    std::size_t outstanding = 0;
    std::size_t completed = 0;
    bool accepting = false;
};

struct notif_delivery_inventory_record_t {
    notif_delivery_key_t key;
    bool localPending = false;
    bool receiptPending = false;
};

struct notif_delivery_snapshot_t {
    notif_delivery_inventory_t inventory;
    std::vector<notif_delivery_inventory_record_t> activeRecords;
};

enum class notif_delivery_status_t {
    SUCCESS,
    INVALID_ARGUMENT,
    UNKNOWN_DELIVERY,
    DUPLICATE_DELIVERY,
    IDENTITY_CONFLICT,
    CAPACITY_EXCEEDED,
    ROUTE_TERMINAL,
    REGISTRY_CLOSED,
};

/** Bounded exact-identity join between local AM completion and remote queue admission. */
class notif_delivery_registry_t final {
public:
    using terminal_t = std::function<void(nixl_status_t, std::uint64_t)>;
    using authority_validator_t = std::function<bool()>;

    explicit notif_delivery_registry_t(std::size_t capacity, std::size_t completed_capacity = 0);
    ~notif_delivery_registry_t() = default;

    notif_delivery_registry_t(const notif_delivery_registry_t &) = delete;
    notif_delivery_registry_t &
    operator=(const notif_delivery_registry_t &) = delete;

    [[nodiscard]] notif_delivery_status_t
    registerDelivery(const notif_delivery_key_t &key,
                     terminal_t terminal,
                     const authority_validator_t &validate_authority);
    [[nodiscard]] notif_delivery_status_t
    completeLocal(std::uint64_t identity, nixl_status_t status, std::uint64_t timestamp_ns);
    [[nodiscard]] notif_delivery_status_t
    acceptReceipt(const notif_delivery_receipt_t &receipt, std::uint64_t timestamp_ns);
    [[nodiscard]] notif_delivery_status_t
    failConnection(std::uint64_t connection_identity,
                   nixl_status_t status,
                   std::uint64_t timestamp_ns);
    [[nodiscard]] notif_delivery_status_t
    failRoute(const notif_route_key_t &route, nixl_status_t status, std::uint64_t timestamp_ns);
    [[nodiscard]] notif_delivery_status_t
    failSupersededAuthority(const notif_route_key_t &route,
                            const notif_wire_uuid_t &capability,
                            std::uint64_t capability_epoch,
                            nixl_status_t status,
                            std::uint64_t timestamp_ns);
    [[nodiscard]] notif_delivery_status_t
    failDelivery(std::uint64_t identity, nixl_status_t status, std::uint64_t timestamp_ns);
    [[nodiscard]] notif_delivery_status_t
    failAll(nixl_status_t status, std::uint64_t timestamp_ns);
    [[nodiscard]] notif_delivery_inventory_t
    inventory() const noexcept;
    [[nodiscard]] notif_delivery_snapshot_t
    snapshot() const;
    [[nodiscard]] std::vector<notif_delivery_inventory_record_t>
    activeRecords() const;

private:
    struct record_t {
        notif_delivery_key_t key;
        terminal_t terminal;
        bool localComplete = false;
        bool receiptComplete = false;
        std::uint64_t receiptTimestampNs = 0;
    };

    struct completion_t {
        terminal_t terminal;
        nixl_status_t status = NIXL_ERR_BACKEND;
        std::uint64_t timestampNs = 0;
    };

    struct tombstone_t {
        notif_delivery_key_t key;
        bool receiptAccepted = false;
    };

    [[nodiscard]] notif_delivery_status_t
    failMatchingLocked(const std::function<bool(const record_t &)> &matches,
                       nixl_status_t status,
                       std::uint64_t timestamp_ns,
                       std::vector<completion_t> &completions);
    void
    rememberCompletedLocked(const notif_delivery_key_t &key, bool receipt_accepted);
    static void
    dispatch(std::vector<completion_t> completions) noexcept;

    const std::size_t capacity_;
    const std::size_t completedCapacity_;
    mutable std::mutex mutex_;
    std::unordered_map<std::uint64_t, record_t> outstanding_;
    std::unordered_map<std::uint64_t, tombstone_t> completed_;
    std::deque<std::uint64_t> completedOrder_;
    bool accepting_ = true;
};

} // namespace nixl::ucx

#endif
