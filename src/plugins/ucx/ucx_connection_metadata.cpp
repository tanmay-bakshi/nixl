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

#include "ucx_connection_metadata.h"

#include <algorithm>
#include <array>
#include <limits>

#include "common/uuid_v4.h"

namespace nixl::ucx {
namespace {

    constexpr std::array<std::uint8_t, 4> metadata_magic = {'N', 'X', 'C', 'M'};
    constexpr std::uint8_t metadata_version = 1;

    constexpr std::size_t magic_offset = 0;
    constexpr std::size_t version_offset = 4;
    constexpr std::size_t header_reserved_offset = 5;
    constexpr std::size_t header_reserved_size = 3;
    constexpr std::size_t metadata_size_offset = 8;
    constexpr std::size_t worker_count_offset = 12;
    constexpr std::size_t backend_incarnation_offset = 16;

    constexpr std::size_t worker_incarnation_offset = 0;
    constexpr std::size_t worker_address_size_offset = 16;
    constexpr std::size_t worker_reserved_offset = 20;
    constexpr std::size_t worker_reserved_size = 4;

    static_assert(backend_incarnation_offset + notif_wire_uuid_size ==
                  connection_metadata_header_size);
    static_assert(worker_reserved_offset + worker_reserved_size ==
                  connection_metadata_worker_header_size);
    static_assert(connection_metadata_max_size <= std::numeric_limits<std::uint32_t>::max());
    static_assert(connection_metadata_max_workers <= std::numeric_limits<std::uint32_t>::max());
    static_assert(connection_metadata_max_worker_address_size <=
                  std::numeric_limits<std::uint32_t>::max());

    void
    writeU32(std::string &output, std::size_t offset, std::uint32_t value) noexcept {
        for (std::size_t index = 0; index < sizeof(value); ++index) {
            const std::size_t shift = (sizeof(value) - index - 1) * 8;
            output[offset + index] = static_cast<char>(value >> shift);
        }
    }

    [[nodiscard]] std::uint32_t
    readU32(std::string_view wire, std::size_t offset) noexcept {
        std::uint32_t value = 0;
        for (std::size_t index = 0; index < sizeof(value); ++index) {
            const auto byte = static_cast<std::uint8_t>(wire[offset + index]);
            value = static_cast<std::uint32_t>((value << 8) | byte);
        }
        return value;
    }

    void
    writeUuid(std::string &output, std::size_t offset, const notif_wire_uuid_t &uuid) noexcept {
        for (std::size_t index = 0; index < uuid.bytes.size(); ++index) {
            output[offset + index] = static_cast<char>(uuid.bytes[index]);
        }
    }

    [[nodiscard]] notif_wire_uuid_t
    readUuid(std::string_view wire, std::size_t offset) noexcept {
        notif_wire_uuid_t uuid;
        for (std::size_t index = 0; index < uuid.bytes.size(); ++index) {
            uuid.bytes[index] = static_cast<std::uint8_t>(wire[offset + index]);
        }
        return uuid;
    }

    [[nodiscard]] bool
    containsWorker(const std::vector<connection_metadata_worker_t> &workers,
                   const notif_wire_uuid_t &incarnation) noexcept {
        return std::any_of(workers.begin(), workers.end(), [&](const auto &worker) {
            return worker.incarnation == incarnation;
        });
    }

    [[nodiscard]] connection_metadata_status_t
    validateMetadata(const connection_metadata_t &metadata, std::size_t &encoded_size) noexcept {
        if (!isCanonicalConnectionMetadataUuid(metadata.backendIncarnation)) {
            return connection_metadata_status_t::INVALID_IDENTITY;
        }
        if (metadata.workers.empty() || metadata.workers.size() > connection_metadata_max_workers) {
            return connection_metadata_status_t::INVALID_WORKER_COUNT;
        }

        std::size_t validated_size = connection_metadata_header_size;
        for (std::size_t index = 0; index < metadata.workers.size(); ++index) {
            const connection_metadata_worker_t &worker = metadata.workers[index];
            if (!isCanonicalConnectionMetadataUuid(worker.incarnation)) {
                return connection_metadata_status_t::INVALID_IDENTITY;
            }
            if (worker.endpointAddress.empty() ||
                worker.endpointAddress.size() > connection_metadata_max_worker_address_size) {
                return connection_metadata_status_t::INVALID_WORKER_ADDRESS;
            }
            if (std::any_of(metadata.workers.begin(),
                            metadata.workers.begin() + index,
                            [&](const auto &previous) {
                                return previous.incarnation == worker.incarnation;
                            })) {
                return connection_metadata_status_t::DUPLICATE_WORKER;
            }

            const std::size_t entry_size =
                connection_metadata_worker_header_size + worker.endpointAddress.size();
            if (entry_size > connection_metadata_max_size - validated_size) {
                return connection_metadata_status_t::METADATA_TOO_LARGE;
            }
            validated_size += entry_size;
        }

        encoded_size = validated_size;
        return connection_metadata_status_t::SUCCESS;
    }

    [[nodiscard]] bool
    hasNonzeroBytes(std::string_view wire, std::size_t offset, std::size_t size) noexcept {
        return std::any_of(wire.begin() + offset, wire.begin() + offset + size, [](char byte) {
            return byte != 0;
        });
    }

} // namespace

notif_wire_uuid_t
generateConnectionMetadataUuid() {
    const UUIDv4 generated;
    return {.bytes = generated.get_data()};
}

bool
isCanonicalConnectionMetadataUuid(const notif_wire_uuid_t &uuid) noexcept {
    return isCanonicalNotifWireUuid(uuid);
}

connection_metadata_status_t
getConnectionMetadataEncodedSize(const connection_metadata_t &metadata,
                                 std::size_t &encoded_size) noexcept {
    std::size_t validated_size = 0;
    const connection_metadata_status_t status = validateMetadata(metadata, validated_size);
    if (status != connection_metadata_status_t::SUCCESS) {
        return status;
    }

    encoded_size = validated_size;
    return connection_metadata_status_t::SUCCESS;
}

connection_metadata_status_t
encodeConnectionMetadata(const connection_metadata_t &metadata, std::string &output) {
    std::size_t encoded_size = 0;
    const connection_metadata_status_t status = validateMetadata(metadata, encoded_size);
    if (status != connection_metadata_status_t::SUCCESS) {
        return status;
    }

    std::string encoded(encoded_size, '\0');
    for (std::size_t index = 0; index < metadata_magic.size(); ++index) {
        encoded[magic_offset + index] = static_cast<char>(metadata_magic[index]);
    }
    encoded[version_offset] = static_cast<char>(metadata_version);
    writeU32(encoded, metadata_size_offset, static_cast<std::uint32_t>(encoded_size));
    writeU32(encoded, worker_count_offset, static_cast<std::uint32_t>(metadata.workers.size()));
    writeUuid(encoded, backend_incarnation_offset, metadata.backendIncarnation);

    std::size_t offset = connection_metadata_header_size;
    for (const connection_metadata_worker_t &worker : metadata.workers) {
        writeUuid(encoded, offset + worker_incarnation_offset, worker.incarnation);
        writeU32(encoded,
                 offset + worker_address_size_offset,
                 static_cast<std::uint32_t>(worker.endpointAddress.size()));
        offset += connection_metadata_worker_header_size;
        std::copy(
            worker.endpointAddress.begin(), worker.endpointAddress.end(), encoded.begin() + offset);
        offset += worker.endpointAddress.size();
    }

    output.swap(encoded);
    return connection_metadata_status_t::SUCCESS;
}

connection_metadata_status_t
decodeConnectionMetadata(std::string_view wire, connection_metadata_t &output) {
    if (wire.size() > connection_metadata_max_size) {
        return connection_metadata_status_t::METADATA_TOO_LARGE;
    }
    if (wire.size() < connection_metadata_header_size) {
        return connection_metadata_status_t::MALFORMED_METADATA;
    }
    for (std::size_t index = 0; index < metadata_magic.size(); ++index) {
        if (static_cast<std::uint8_t>(wire[magic_offset + index]) != metadata_magic[index]) {
            return connection_metadata_status_t::MALFORMED_METADATA;
        }
    }
    if (static_cast<std::uint8_t>(wire[version_offset]) != metadata_version) {
        return connection_metadata_status_t::UNSUPPORTED_VERSION;
    }
    if (hasNonzeroBytes(wire, header_reserved_offset, header_reserved_size)) {
        return connection_metadata_status_t::NONZERO_RESERVED;
    }
    if (readU32(wire, metadata_size_offset) != wire.size()) {
        return connection_metadata_status_t::MALFORMED_METADATA;
    }

    const std::uint32_t worker_count = readU32(wire, worker_count_offset);
    if (worker_count == 0 || worker_count > connection_metadata_max_workers) {
        return connection_metadata_status_t::INVALID_WORKER_COUNT;
    }

    connection_metadata_t decoded;
    decoded.backendIncarnation = readUuid(wire, backend_incarnation_offset);
    if (!isCanonicalConnectionMetadataUuid(decoded.backendIncarnation)) {
        return connection_metadata_status_t::INVALID_IDENTITY;
    }
    decoded.workers.reserve(worker_count);

    std::size_t offset = connection_metadata_header_size;
    for (std::uint32_t index = 0; index < worker_count; ++index) {
        if (wire.size() - offset < connection_metadata_worker_header_size) {
            return connection_metadata_status_t::MALFORMED_METADATA;
        }

        connection_metadata_worker_t worker;
        worker.incarnation = readUuid(wire, offset + worker_incarnation_offset);
        if (!isCanonicalConnectionMetadataUuid(worker.incarnation)) {
            return connection_metadata_status_t::INVALID_IDENTITY;
        }
        if (containsWorker(decoded.workers, worker.incarnation)) {
            return connection_metadata_status_t::DUPLICATE_WORKER;
        }

        const std::uint32_t address_size = readU32(wire, offset + worker_address_size_offset);
        if (address_size == 0 || address_size > connection_metadata_max_worker_address_size) {
            return connection_metadata_status_t::INVALID_WORKER_ADDRESS;
        }
        if (hasNonzeroBytes(wire, offset + worker_reserved_offset, worker_reserved_size)) {
            return connection_metadata_status_t::NONZERO_RESERVED;
        }

        offset += connection_metadata_worker_header_size;
        if (address_size > wire.size() - offset) {
            return connection_metadata_status_t::MALFORMED_METADATA;
        }
        worker.endpointAddress.assign(wire.substr(offset, address_size));
        offset += address_size;
        decoded.workers.push_back(std::move(worker));
    }
    if (offset != wire.size()) {
        return connection_metadata_status_t::MALFORMED_METADATA;
    }

    output = std::move(decoded);
    return connection_metadata_status_t::SUCCESS;
}

} // namespace nixl::ucx
