/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_SRC_PLUGINS_UCX_UCX_NOTIF_INGRESS_H
#define NIXL_SRC_PLUGINS_UCX_UCX_NOTIF_INGRESS_H

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nixl_types.h>

#include "backend/backend_aux.h"
#include "ucx_notif_state.h"

namespace nixl::ucx {

class ucx_callback_slot_t;

struct notif_ingress_key_t {
    notif_wire_uuid_t sourceBackendIncarnation;
    std::uint64_t sourceHandleIdentity = 0;
    std::uint64_t sourceGeneration = 0;
    std::uint64_t deliveryIdentity = 0;
    notif_wire_uuid_t destinationAgentIncarnation;
    notif_wire_uuid_t destinationBackendIncarnation;
    notif_route_key_t route;
    notif_wire_uuid_t capability;
    std::uint64_t capabilityEpoch = 0;
    std::uint64_t connectionIdentity = 0;
    notif_wire_uuid_t sourceWorkerIncarnation;
    notif_wire_uuid_t receiptWorkerIncarnation;
    std::uint64_t endpointIdentity = 0;

    bool
    operator==(const notif_ingress_key_t &) const = default;
};

struct notif_ingress_inventory_t {
    std::size_t pending = 0;
    std::size_t admitting = 0;
    std::size_t committed = 0;
    std::size_t replaying = 0;
    std::size_t completed = 0;
    std::size_t quarantined = 0;
};

enum class notif_ingress_inventory_phase_t {
    PENDING,
    ADMITTING,
    COMMITTED,
    REPLAYING,
    QUARANTINED,
};

struct notif_ingress_inventory_record_t {
    notif_ingress_key_t key;
    notif_ingress_inventory_phase_t phase = notif_ingress_inventory_phase_t::PENDING;
};

struct notif_ingress_snapshot_t {
    notif_ingress_inventory_t inventory;
    std::vector<notif_ingress_inventory_record_t> activeRecords;
};

enum class notif_ingress_status_t {
    SUCCESS,
    DUPLICATE_PENDING,
    DUPLICATE_COMMITTED,
    INVALID_ARGUMENT,
    UNKNOWN_DELIVERY,
    IDENTITY_CONFLICT,
    CAPACITY_EXCEEDED,
    ROUTE_TERMINAL,
    INVALID_TRANSITION,
    RECEIPT_FAILED,
    DRAIN_TIMEOUT,
    INTERNAL_ERROR,
};

/** Bounded destination authority for exactly-once attached-notification admission. */
class notif_ingress_registry_t final {
public:
    explicit notif_ingress_registry_t(std::size_t capacity, std::size_t completed_capacity = 0);

    [[nodiscard]] notif_ingress_status_t
    reserve(const notif_ingress_key_t &key,
            std::span<const std::uint8_t> payload,
            const notif_wire_envelope_t &receipt);
    [[nodiscard]] notif_ingress_status_t
    admit(const notif_ingress_key_t &key,
          std::span<const std::uint8_t> payload,
          nixlAuthenticatedNotification notification,
          const std::function<nixl_status_t(nixlAuthenticatedNotification &&)> &push,
          notif_wire_envelope_t &receipt);
    [[nodiscard]] notif_ingress_status_t
    completeReceipt(const notif_ingress_key_t &key, nixl_status_t status);
    [[nodiscard]] notif_ingress_status_t
    retainReceiptSlot(const notif_ingress_key_t &key, std::shared_ptr<ucx_callback_slot_t> slot);
    [[nodiscard]] notif_ingress_status_t
    failRoute(const notif_route_key_t &route);
    [[nodiscard]] notif_ingress_status_t
    failConnection(std::uint64_t connection_identity);
    [[nodiscard]] notif_ingress_status_t
    failAll();
    [[nodiscard]] notif_ingress_status_t
    drainRoute(const notif_route_key_t &route, std::chrono::nanoseconds timeout) const;
    [[nodiscard]] notif_ingress_status_t
    drainConnection(std::uint64_t connection_identity, std::chrono::nanoseconds timeout) const;
    [[nodiscard]] notif_ingress_status_t
    retireRoute(const notif_route_key_t &route);
    [[nodiscard]] notif_ingress_status_t
    retireConnection(std::uint64_t connection_identity);
    [[nodiscard]] notif_ingress_inventory_t
    inventory() const noexcept;
    [[nodiscard]] notif_ingress_snapshot_t
    snapshot() const;
    [[nodiscard]] std::vector<notif_ingress_inventory_record_t>
    activeRecords() const;

private:
    struct identity_t {
        notif_wire_uuid_t sourceBackendIncarnation;
        std::uint64_t sourceHandleIdentity = 0;
        std::uint64_t sourceGeneration = 0;
        std::uint64_t deliveryIdentity = 0;

        bool
        operator==(const identity_t &) const = default;
    };

    struct identity_hash_t {
        [[nodiscard]] std::size_t
        operator()(const identity_t &identity) const noexcept;
    };

    struct route_hash_t {
        [[nodiscard]] std::size_t
        operator()(const notif_route_key_t &route) const noexcept;
    };

    struct source_epoch_t {
        notif_wire_uuid_t sourceBackendIncarnation;
        std::uint64_t sourceHandleIdentity = 0;
        std::uint64_t sourceGeneration = 0;

        bool
        operator==(const source_epoch_t &) const = default;
    };

    struct source_epoch_hash_t {
        [[nodiscard]] std::size_t
        operator()(const source_epoch_t &source) const noexcept;
    };

    enum class phase_t {
        PENDING,
        ADMITTING,
        COMMITTED,
    };

    struct replay_evidence_t {
        notif_ingress_key_t key;
        std::array<std::uint8_t, 32> payloadDigest{};
        notif_wire_envelope_t receipt;
    };

    struct record_t : replay_evidence_t {
        std::shared_ptr<ucx_callback_slot_t> receiptSlot;
        phase_t phase = phase_t::PENDING;
        bool receiptClaimed = false;
    };

    struct replay_send_t {
        notif_ingress_key_t key;
        std::shared_ptr<ucx_callback_slot_t> receiptSlot;
    };

    [[nodiscard]] static identity_t
    identity(const notif_ingress_key_t &key) noexcept;
    [[nodiscard]] static source_epoch_t
    sourceEpoch(const notif_ingress_key_t &key) noexcept;
    [[nodiscard]] static bool
    digest(std::span<const std::uint8_t> payload, std::array<std::uint8_t, 32> &result) noexcept;
    [[nodiscard]] static bool
    valid(const notif_ingress_key_t &key, const notif_wire_envelope_t &receipt) noexcept;
    [[nodiscard]] static bool
    matches(const replay_evidence_t &evidence,
            const notif_ingress_key_t &key,
            const std::array<std::uint8_t, 32> &payload_digest,
            const notif_wire_envelope_t *receipt = nullptr) noexcept;
    [[nodiscard]] bool
    rememberCompletedLocked(const replay_evidence_t &evidence);
    [[nodiscard]] bool
    routeDrainedLocked(const notif_route_key_t &route) const noexcept;
    [[nodiscard]] bool
    connectionDrainedLocked(std::uint64_t connection_identity) const noexcept;

    const std::size_t capacity_;
    const std::size_t completedCapacity_;
    mutable std::mutex mutex_;
    mutable std::condition_variable drained_;
    std::unordered_map<identity_t, record_t, identity_hash_t> active_;
    std::unordered_map<identity_t, replay_evidence_t, identity_hash_t> completed_;
    std::unordered_map<identity_t, replay_evidence_t, identity_hash_t> quarantined_;
    std::unordered_map<identity_t, replay_send_t, identity_hash_t> replaySends_;
    std::deque<identity_t> completedOrder_;
    std::unordered_set<notif_route_key_t, route_hash_t> terminalRoutes_;
    std::unordered_set<std::uint64_t> terminalConnections_;
    std::unordered_map<source_epoch_t, std::uint64_t, source_epoch_hash_t> highWatermarks_;
    bool closed_ = false;
};

} // namespace nixl::ucx

#endif
