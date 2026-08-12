/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef NIXL_SRC_PLUGINS_UCX_UCX_NOTIF_WIRE_H
#define NIXL_SRC_PLUGINS_UCX_UCX_NOTIF_WIRE_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace nixl::ucx {

inline constexpr std::size_t notif_wire_uuid_size = 16;
inline constexpr std::size_t notif_wire_header_size = 144;
inline constexpr std::size_t notif_wire_max_frame_size = 64 * 1024;

enum class notif_wire_type_t : std::uint8_t {
    OFFER = 1,
    ACK = 2,
    DATA = 3,
    DATA_RECEIPT = 4,
    ATTACHED_DATA = 5,
};

enum class notif_wire_status_t {
    SUCCESS,
    INVALID_EPOCH,
    FRAME_TOO_LARGE,
    MALFORMED_FRAME,
    UNSUPPORTED_VERSION,
    INVALID_TYPE,
    NONZERO_RESERVED,
    INVALID_IDENTITY,
    INVALID_DELIVERY_IDENTITY,
    INVALID_SOURCE_TRANSFER,
    INVALID_PAYLOAD,
};

struct notif_wire_uuid_t {
    std::array<std::uint8_t, notif_wire_uuid_size> bytes{};

    bool
    operator==(const notif_wire_uuid_t &) const = default;
};

struct notif_wire_envelope_t {
    notif_wire_type_t type = notif_wire_type_t::OFFER;
    notif_wire_uuid_t senderAgentIncarnation;
    notif_wire_uuid_t recipientAgentIncarnation;
    notif_wire_uuid_t senderBackendIncarnation;
    notif_wire_uuid_t recipientBackendIncarnation;
    notif_wire_uuid_t senderWorkerIncarnation;
    notif_wire_uuid_t capability;
    std::uint64_t capabilityEpoch = 0;
    std::uint64_t deliveryIdentity = 0;
    std::uint64_t sourceHandleIdentity = 0;
    std::uint64_t sourceGeneration = 0;

    bool
    operator==(const notif_wire_envelope_t &) const = default;
};

struct notif_wire_frame_view_t {
    notif_wire_envelope_t envelope;
    std::span<const std::uint8_t> payload;
};

[[nodiscard]] bool
isCanonicalNotifWireUuid(const notif_wire_uuid_t &uuid) noexcept;

/*
 * Performs every size and structural check without allocating. A transfer path can
 * use this before submitting data, then encode the already-preflighted frame.
 */
[[nodiscard]] notif_wire_status_t
getNotifWireEncodedSize(const notif_wire_envelope_t &envelope,
                        std::size_t payload_size,
                        std::size_t &encoded_size) noexcept;

/*
 * On failure, output is unchanged. OFFER and ACK frames reject payload bytes;
 * DATA is the only frame kind that carries an opaque notification payload.
 */
[[nodiscard]] notif_wire_status_t
encodeNotifWireFrame(const notif_wire_envelope_t &envelope,
                     std::span<const std::uint8_t> payload,
                     std::vector<std::uint8_t> &output);

/*
 * On failure, output is unchanged. The decoded payload is a view into wire so route
 * validation can happen before any payload allocation or copy. The caller must keep
 * wire alive and immutable while using output. Frames larger than the symmetric
 * 64 KiB limit are rejected before any header field is inspected or copied.
 */
[[nodiscard]] notif_wire_status_t
decodeNotifWireFrame(std::span<const std::uint8_t> wire,
                     notif_wire_frame_view_t &output);

} // namespace nixl::ucx

#endif
