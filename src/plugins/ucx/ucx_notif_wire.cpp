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

#include "ucx_notif_wire.h"

#include <algorithm>
#include <array>

namespace nixl::ucx {
namespace {

constexpr std::array<std::uint8_t, 4> wire_magic = {'N', 'X', 'N', 'F'};
constexpr std::uint8_t wire_version = 2;

constexpr std::size_t magic_offset = 0;
constexpr std::size_t version_offset = 4;
constexpr std::size_t type_offset = 5;
constexpr std::size_t reserved_offset = 6;
constexpr std::size_t frame_size_offset = 8;
constexpr std::size_t payload_size_offset = 12;
constexpr std::size_t sender_agent_offset = 16;
constexpr std::size_t recipient_agent_offset = 32;
constexpr std::size_t sender_backend_offset = 48;
constexpr std::size_t recipient_backend_offset = 64;
constexpr std::size_t sender_worker_offset = 80;
constexpr std::size_t capability_offset = 96;
constexpr std::size_t capability_epoch_offset = 112;
constexpr std::size_t delivery_identity_offset = 120;
constexpr std::size_t source_handle_identity_offset = 128;
constexpr std::size_t source_generation_offset = 136;

static_assert(source_generation_offset + sizeof(std::uint64_t) ==
              notif_wire_header_size);

[[nodiscard]] bool
isKnownType(notif_wire_type_t type) noexcept {
    switch (type) {
    case notif_wire_type_t::OFFER:
    case notif_wire_type_t::ACK:
    case notif_wire_type_t::DATA:
    case notif_wire_type_t::DATA_RECEIPT:
    case notif_wire_type_t::ATTACHED_DATA:
        return true;
    }
    return false;
}

[[nodiscard]] bool
payloadIsValid(notif_wire_type_t type, std::size_t payload_size) noexcept {
    switch (type) {
    case notif_wire_type_t::OFFER:
    case notif_wire_type_t::ACK:
    case notif_wire_type_t::DATA_RECEIPT:
        return payload_size == 0;
    case notif_wire_type_t::DATA:
    case notif_wire_type_t::ATTACHED_DATA:
        return true;
    }
    return false;
}

[[nodiscard]] bool
hasAttachedDeliveryIdentity(notif_wire_type_t type) noexcept {
    return type == notif_wire_type_t::DATA_RECEIPT ||
        type == notif_wire_type_t::ATTACHED_DATA;
}

[[nodiscard]] bool
identitiesAreCanonical(const notif_wire_envelope_t &envelope) noexcept {
    return isCanonicalNotifWireUuid(envelope.senderAgentIncarnation) &&
        isCanonicalNotifWireUuid(envelope.recipientAgentIncarnation) &&
        isCanonicalNotifWireUuid(envelope.senderBackendIncarnation) &&
        isCanonicalNotifWireUuid(envelope.recipientBackendIncarnation) &&
        isCanonicalNotifWireUuid(envelope.senderWorkerIncarnation) &&
        isCanonicalNotifWireUuid(envelope.capability);
}

void
writeU16(std::vector<std::uint8_t> &output,
         std::size_t offset,
         std::uint16_t value) noexcept {
    output[offset] = static_cast<std::uint8_t>(value >> 8);
    output[offset + 1] = static_cast<std::uint8_t>(value);
}

void
writeU32(std::vector<std::uint8_t> &output,
         std::size_t offset,
         std::uint32_t value) noexcept {
    for (std::size_t index = 0; index < sizeof(value); ++index) {
        const std::size_t shift = (sizeof(value) - index - 1) * 8;
        output[offset + index] = static_cast<std::uint8_t>(value >> shift);
    }
}

void
writeU64(std::vector<std::uint8_t> &output,
         std::size_t offset,
         std::uint64_t value) noexcept {
    for (std::size_t index = 0; index < sizeof(value); ++index) {
        const std::size_t shift = (sizeof(value) - index - 1) * 8;
        output[offset + index] = static_cast<std::uint8_t>(value >> shift);
    }
}

[[nodiscard]] std::uint16_t
readU16(std::span<const std::uint8_t> wire, std::size_t offset) noexcept {
    return static_cast<std::uint16_t>(
        static_cast<std::uint16_t>(wire[offset]) << 8 |
        static_cast<std::uint16_t>(wire[offset + 1]));
}

[[nodiscard]] std::uint32_t
readU32(std::span<const std::uint8_t> wire, std::size_t offset) noexcept {
    std::uint32_t value = 0;
    for (std::size_t index = 0; index < sizeof(value); ++index) {
        value = static_cast<std::uint32_t>((value << 8) | wire[offset + index]);
    }
    return value;
}

[[nodiscard]] std::uint64_t
readU64(std::span<const std::uint8_t> wire, std::size_t offset) noexcept {
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < sizeof(value); ++index) {
        value = (value << 8) | wire[offset + index];
    }
    return value;
}

void
writeUuid(std::vector<std::uint8_t> &output,
          std::size_t offset,
          const notif_wire_uuid_t &uuid) noexcept {
    std::copy(uuid.bytes.begin(), uuid.bytes.end(), output.begin() + offset);
}

[[nodiscard]] notif_wire_uuid_t
readUuid(std::span<const std::uint8_t> wire, std::size_t offset) noexcept {
    notif_wire_uuid_t uuid;
    std::copy_n(wire.begin() + offset, uuid.bytes.size(), uuid.bytes.begin());
    return uuid;
}

[[nodiscard]] notif_wire_status_t
parseType(std::uint8_t encoded, notif_wire_type_t &type) noexcept {
    switch (encoded) {
    case static_cast<std::uint8_t>(notif_wire_type_t::OFFER):
        type = notif_wire_type_t::OFFER;
        return notif_wire_status_t::SUCCESS;
    case static_cast<std::uint8_t>(notif_wire_type_t::ACK):
        type = notif_wire_type_t::ACK;
        return notif_wire_status_t::SUCCESS;
    case static_cast<std::uint8_t>(notif_wire_type_t::DATA):
        type = notif_wire_type_t::DATA;
        return notif_wire_status_t::SUCCESS;
    case static_cast<std::uint8_t>(notif_wire_type_t::DATA_RECEIPT):
        type = notif_wire_type_t::DATA_RECEIPT;
        return notif_wire_status_t::SUCCESS;
    case static_cast<std::uint8_t>(notif_wire_type_t::ATTACHED_DATA):
        type = notif_wire_type_t::ATTACHED_DATA;
        return notif_wire_status_t::SUCCESS;
    default:
        return notif_wire_status_t::INVALID_TYPE;
    }
}

} // namespace

bool
isCanonicalNotifWireUuid(const notif_wire_uuid_t &uuid) noexcept {
    const bool is_version_four = (uuid.bytes[6] & 0xf0U) == 0x40U;
    const bool is_rfc_variant = (uuid.bytes[8] & 0xc0U) == 0x80U;
    return is_version_four && is_rfc_variant;
}

notif_wire_status_t
getNotifWireEncodedSize(const notif_wire_envelope_t &envelope,
                        std::size_t payload_size,
                        std::size_t &encoded_size) noexcept {
    if (payload_size > notif_wire_max_frame_size - notif_wire_header_size) {
        return notif_wire_status_t::FRAME_TOO_LARGE;
    }
    if (!isKnownType(envelope.type)) {
        return notif_wire_status_t::INVALID_TYPE;
    }
    if (!identitiesAreCanonical(envelope)) {
        return notif_wire_status_t::INVALID_IDENTITY;
    }
    if (envelope.capabilityEpoch == 0) {
        return notif_wire_status_t::INVALID_EPOCH;
    }
    const bool attached = hasAttachedDeliveryIdentity(envelope.type);
    if (attached && envelope.deliveryIdentity == 0) {
        return notif_wire_status_t::INVALID_DELIVERY_IDENTITY;
    }
    if (!attached && envelope.deliveryIdentity != 0) {
        return notif_wire_status_t::INVALID_DELIVERY_IDENTITY;
    }
    if (attached &&
        (envelope.sourceHandleIdentity == 0 || envelope.sourceGeneration == 0)) {
        return notif_wire_status_t::INVALID_SOURCE_TRANSFER;
    }
    if (!attached &&
        (envelope.sourceHandleIdentity != 0 || envelope.sourceGeneration != 0)) {
        return notif_wire_status_t::INVALID_SOURCE_TRANSFER;
    }
    if (!payloadIsValid(envelope.type, payload_size)) {
        return notif_wire_status_t::INVALID_PAYLOAD;
    }

    encoded_size = notif_wire_header_size + payload_size;
    return notif_wire_status_t::SUCCESS;
}

notif_wire_status_t
encodeNotifWireFrame(const notif_wire_envelope_t &envelope,
                     std::span<const std::uint8_t> payload,
                     std::vector<std::uint8_t> &output) {
    std::size_t encoded_size = 0;
    const notif_wire_status_t preflight_status =
        getNotifWireEncodedSize(envelope, payload.size(), encoded_size);
    if (preflight_status != notif_wire_status_t::SUCCESS) {
        return preflight_status;
    }

    std::vector<std::uint8_t> encoded(encoded_size, 0);
    std::copy(wire_magic.begin(), wire_magic.end(), encoded.begin() + magic_offset);
    encoded[version_offset] = wire_version;
    encoded[type_offset] = static_cast<std::uint8_t>(envelope.type);
    writeU16(encoded, reserved_offset, 0);
    writeU32(encoded, frame_size_offset, static_cast<std::uint32_t>(encoded_size));
    writeU32(encoded, payload_size_offset, static_cast<std::uint32_t>(payload.size()));
    writeUuid(encoded, sender_agent_offset, envelope.senderAgentIncarnation);
    writeUuid(encoded, recipient_agent_offset, envelope.recipientAgentIncarnation);
    writeUuid(encoded, sender_backend_offset, envelope.senderBackendIncarnation);
    writeUuid(encoded, recipient_backend_offset, envelope.recipientBackendIncarnation);
    writeUuid(encoded, sender_worker_offset, envelope.senderWorkerIncarnation);
    writeUuid(encoded, capability_offset, envelope.capability);
    writeU64(encoded, capability_epoch_offset, envelope.capabilityEpoch);
    writeU64(encoded, delivery_identity_offset, envelope.deliveryIdentity);
    writeU64(encoded, source_handle_identity_offset, envelope.sourceHandleIdentity);
    writeU64(encoded, source_generation_offset, envelope.sourceGeneration);
    std::copy(payload.begin(), payload.end(), encoded.begin() + notif_wire_header_size);

    output.swap(encoded);
    return notif_wire_status_t::SUCCESS;
}

notif_wire_status_t
decodeNotifWireFrame(std::span<const std::uint8_t> wire,
                     notif_wire_frame_view_t &output) {
    if (wire.size() > notif_wire_max_frame_size) {
        return notif_wire_status_t::FRAME_TOO_LARGE;
    }
    if (wire.size() < notif_wire_header_size) {
        return notif_wire_status_t::MALFORMED_FRAME;
    }
    if (!std::equal(wire_magic.begin(), wire_magic.end(), wire.begin() + magic_offset)) {
        return notif_wire_status_t::MALFORMED_FRAME;
    }
    if (wire[version_offset] != wire_version) {
        return notif_wire_status_t::UNSUPPORTED_VERSION;
    }

    notif_wire_frame_view_t decoded;
    const notif_wire_status_t type_status =
        parseType(wire[type_offset], decoded.envelope.type);
    if (type_status != notif_wire_status_t::SUCCESS) {
        return type_status;
    }
    if (readU16(wire, reserved_offset) != 0) {
        return notif_wire_status_t::NONZERO_RESERVED;
    }
    const std::uint32_t declared_frame_size = readU32(wire, frame_size_offset);
    const std::uint32_t declared_payload_size = readU32(wire, payload_size_offset);
    const std::size_t actual_payload_size = wire.size() - notif_wire_header_size;
    if (declared_frame_size != wire.size() ||
        declared_payload_size != actual_payload_size) {
        return notif_wire_status_t::MALFORMED_FRAME;
    }
    if (!payloadIsValid(decoded.envelope.type, actual_payload_size)) {
        return notif_wire_status_t::INVALID_PAYLOAD;
    }

    decoded.envelope.senderAgentIncarnation = readUuid(wire, sender_agent_offset);
    decoded.envelope.recipientAgentIncarnation = readUuid(wire, recipient_agent_offset);
    decoded.envelope.senderBackendIncarnation = readUuid(wire, sender_backend_offset);
    decoded.envelope.recipientBackendIncarnation = readUuid(wire, recipient_backend_offset);
    decoded.envelope.senderWorkerIncarnation = readUuid(wire, sender_worker_offset);
    decoded.envelope.capability = readUuid(wire, capability_offset);
    if (!identitiesAreCanonical(decoded.envelope)) {
        return notif_wire_status_t::INVALID_IDENTITY;
    }

    decoded.envelope.capabilityEpoch = readU64(wire, capability_epoch_offset);
    if (decoded.envelope.capabilityEpoch == 0) {
        return notif_wire_status_t::INVALID_EPOCH;
    }
    decoded.envelope.deliveryIdentity = readU64(wire, delivery_identity_offset);
    decoded.envelope.sourceHandleIdentity = readU64(wire, source_handle_identity_offset);
    decoded.envelope.sourceGeneration = readU64(wire, source_generation_offset);
    const bool attached = hasAttachedDeliveryIdentity(decoded.envelope.type);
    if (attached && decoded.envelope.deliveryIdentity == 0) {
        return notif_wire_status_t::INVALID_DELIVERY_IDENTITY;
    }
    if (!attached && decoded.envelope.deliveryIdentity != 0) {
        return notif_wire_status_t::INVALID_DELIVERY_IDENTITY;
    }
    if (attached &&
        (decoded.envelope.sourceHandleIdentity == 0 ||
         decoded.envelope.sourceGeneration == 0)) {
        return notif_wire_status_t::INVALID_SOURCE_TRANSFER;
    }
    if (!attached &&
        (decoded.envelope.sourceHandleIdentity != 0 ||
         decoded.envelope.sourceGeneration != 0)) {
        return notif_wire_status_t::INVALID_SOURCE_TRANSFER;
    }
    decoded.payload = wire.subspan(notif_wire_header_size);

    output = decoded;
    return notif_wire_status_t::SUCCESS;
}

} // namespace nixl::ucx
