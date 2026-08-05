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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "ucx_connection_metadata.h"

namespace {

using namespace nixl::ucx;

static_assert(connection_metadata_header_size == 32);
static_assert(connection_metadata_worker_header_size == 24);
static_assert(connection_metadata_max_size == 65536);

void
require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

notif_wire_uuid_t
makeUuid(std::uint8_t seed) {
    notif_wire_uuid_t uuid;
    for (std::size_t index = 0; index < uuid.bytes.size(); ++index) {
        uuid.bytes[index] = static_cast<std::uint8_t>(seed + index * 13);
    }
    uuid.bytes[6] = static_cast<std::uint8_t>((uuid.bytes[6] & 0x0fU) | 0x40U);
    uuid.bytes[8] = static_cast<std::uint8_t>((uuid.bytes[8] & 0x3fU) | 0x80U);
    return uuid;
}

connection_metadata_t
makeMetadata() {
    return {
        .backendIncarnation = makeUuid(1),
        .workers =
            {
                {
                    .incarnation = makeUuid(2),
                    .endpointAddress = std::string("\0\x01\xff", 3),
                },
                {
                    .incarnation = makeUuid(3),
                    .endpointAddress = "ucx",
                },
            },
    };
}

std::string
encode(const connection_metadata_t &metadata) {
    std::string wire;
    require(encodeConnectionMetadata(metadata, wire) == connection_metadata_status_t::SUCCESS,
            "valid connection metadata was not encoded");
    return wire;
}

connection_metadata_t
makeDecodeSentinel() {
    return {
        .backendIncarnation = makeUuid(91),
        .workers = {{
            .incarnation = makeUuid(92),
            .endpointAddress = "sentinel",
        }},
    };
}

void
requireDecodeFailure(std::string_view wire,
                     connection_metadata_status_t expected_status,
                     std::string_view message) {
    connection_metadata_t output = makeDecodeSentinel();
    const connection_metadata_t expected_output = output;
    require(decodeConnectionMetadata(wire, output) == expected_status, message);
    require(output == expected_output, "failed decode mutated its output");
}

void
writeU32(std::string &wire, std::size_t offset, std::uint32_t value) {
    for (std::size_t index = 0; index < sizeof(value); ++index) {
        const std::size_t shift = (sizeof(value) - index - 1) * 8;
        wire[offset + index] = static_cast<char>(value >> shift);
    }
}

void
appendUuid(std::string &wire, const notif_wire_uuid_t &uuid) {
    for (const std::uint8_t byte : uuid.bytes) {
        wire.push_back(static_cast<char>(byte));
    }
}

void
appendU32(std::string &wire, std::uint32_t value) {
    const std::size_t offset = wire.size();
    wire.resize(offset + sizeof(value));
    writeU32(wire, offset, value);
}

void
testGeneratedIdentitiesAreCanonical() {
    for (std::size_t index = 0; index < 32; ++index) {
        require(isCanonicalConnectionMetadataUuid(generateConnectionMetadataUuid()),
                "generated connection UUID is not canonical UUID-v4");
    }
}

void
testStableCanonicalRoundTrip() {
    const connection_metadata_t metadata = makeMetadata();
    const std::string first = encode(metadata);
    const std::string duplicate = encode(metadata);
    require(first == duplicate, "encoding the same metadata did not produce identical bytes");

    connection_metadata_t decoded;
    require(decodeConnectionMetadata(first, decoded) == connection_metadata_status_t::SUCCESS,
            "valid connection metadata was not decoded");
    require(decoded == metadata, "connection metadata round trip changed values");
    require(decoded.workers[0].endpointAddress == std::string("\0\x01\xff", 3),
            "binary UCX worker address was not preserved");
    require(encode(decoded) == first, "decoded metadata did not re-encode canonically");
}

void
testCanonicalNetworkOrderLayout() {
    const connection_metadata_t metadata = makeMetadata();
    const std::string wire = encode(metadata);

    std::string expected = "NXCM";
    expected.push_back(1);
    expected.append(3, '\0');
    appendU32(expected, 86);
    appendU32(expected, 2);
    appendUuid(expected, metadata.backendIncarnation);
    for (const connection_metadata_worker_t &worker : metadata.workers) {
        appendUuid(expected, worker.incarnation);
        appendU32(expected, static_cast<std::uint32_t>(worker.endpointAddress.size()));
        expected.append(4, '\0');
        expected.append(worker.endpointAddress);
    }

    require(wire == expected, "connection metadata wire layout is not canonical network order");
}

connection_metadata_t
makeExactLimitMetadata() {
    return {
        .backendIncarnation = makeUuid(11),
        .workers = {{
            .incarnation = makeUuid(12),
            .endpointAddress = std::string(connection_metadata_max_worker_address_size, '\xa0'),
        }},
    };
}

void
testSymmetricSizeLimit() {
    const connection_metadata_t exact = makeExactLimitMetadata();
    std::size_t encoded_size = 17;
    require(getConnectionMetadataEncodedSize(exact, encoded_size) ==
                connection_metadata_status_t::SUCCESS,
            "exact-limit connection metadata failed preflight");
    require(encoded_size == connection_metadata_max_size,
            "exact-limit preflight returned the wrong size");

    const std::string exact_wire = encode(exact);
    require(exact_wire.size() == connection_metadata_max_size,
            "exact-limit metadata did not encode to exactly 64 KiB");
    connection_metadata_t decoded;
    require(decodeConnectionMetadata(exact_wire, decoded) ==
                    connection_metadata_status_t::SUCCESS &&
                decoded == exact,
            "exact-limit metadata did not round trip");

    constexpr std::size_t oversized_address_size =
        (connection_metadata_max_size - connection_metadata_header_size -
         2 * connection_metadata_worker_header_size) /
            2 +
        1;
    connection_metadata_t oversized{
        .backendIncarnation = makeUuid(21),
        .workers =
            {
                {.incarnation = makeUuid(22),
                 .endpointAddress = std::string(oversized_address_size, 'a')},
                {.incarnation = makeUuid(23),
                 .endpointAddress = std::string(oversized_address_size, 'b')},
            },
    };
    encoded_size = 19;
    require(getConnectionMetadataEncodedSize(oversized, encoded_size) ==
                connection_metadata_status_t::METADATA_TOO_LARGE,
            "over-limit connection metadata passed preflight");
    require(encoded_size == 19, "failed preflight mutated its output");

    std::string encoded_output = "unchanged";
    require(encodeConnectionMetadata(oversized, encoded_output) ==
                connection_metadata_status_t::METADATA_TOO_LARGE,
            "over-limit connection metadata was encoded");
    require(encoded_output == "unchanged", "failed over-limit encode mutated its output");

    const std::string oversized_wire(connection_metadata_max_size + 1, '\xff');
    requireDecodeFailure(oversized_wire,
                         connection_metadata_status_t::METADATA_TOO_LARGE,
                         "over-limit connection metadata was parsed");
}

void
requireEncodeFailure(const connection_metadata_t &metadata,
                     connection_metadata_status_t expected_status,
                     std::string_view message) {
    std::string output = "unchanged";
    require(encodeConnectionMetadata(metadata, output) == expected_status, message);
    require(output == "unchanged", "failed encode mutated its output");

    std::size_t encoded_size = 23;
    require(getConnectionMetadataEncodedSize(metadata, encoded_size) == expected_status,
            "encode and preflight failure statuses disagree");
    require(encoded_size == 23, "failed preflight mutated its output");
}

void
testEncodeValidation() {
    connection_metadata_t invalid = makeMetadata();
    invalid.backendIncarnation = {};
    requireEncodeFailure(invalid,
                         connection_metadata_status_t::INVALID_IDENTITY,
                         "invalid backend UUID was encoded");

    invalid = makeMetadata();
    invalid.workers.front().incarnation = {};
    requireEncodeFailure(
        invalid, connection_metadata_status_t::INVALID_IDENTITY, "invalid worker UUID was encoded");

    invalid = makeMetadata();
    invalid.workers[1].incarnation = invalid.workers[0].incarnation;
    requireEncodeFailure(invalid,
                         connection_metadata_status_t::DUPLICATE_WORKER,
                         "duplicate worker UUID was encoded");

    invalid = makeMetadata();
    invalid.workers.clear();
    requireEncodeFailure(invalid,
                         connection_metadata_status_t::INVALID_WORKER_COUNT,
                         "empty worker manifest was encoded");

    invalid = makeMetadata();
    invalid.workers.assign(connection_metadata_max_workers + 1, invalid.workers.front());
    requireEncodeFailure(invalid,
                         connection_metadata_status_t::INVALID_WORKER_COUNT,
                         "oversized worker manifest was encoded");

    invalid = makeMetadata();
    invalid.workers.front().endpointAddress.clear();
    requireEncodeFailure(invalid,
                         connection_metadata_status_t::INVALID_WORKER_ADDRESS,
                         "empty worker address was encoded");

    invalid = makeMetadata();
    invalid.workers.front().endpointAddress.assign(connection_metadata_max_worker_address_size + 1,
                                                   'x');
    requireEncodeFailure(invalid,
                         connection_metadata_status_t::INVALID_WORKER_ADDRESS,
                         "oversized worker address was encoded");
}

void
testDecodeHeaderValidation() {
    const std::string valid = encode(makeMetadata());

    for (std::size_t size = 0; size < valid.size(); ++size) {
        requireDecodeFailure(std::string_view(valid).substr(0, size),
                             connection_metadata_status_t::MALFORMED_METADATA,
                             "truncated connection metadata was decoded");
    }

    std::string malformed = valid;
    malformed[0] ^= static_cast<char>(0xffU);
    requireDecodeFailure(malformed,
                         connection_metadata_status_t::MALFORMED_METADATA,
                         "invalid metadata magic was decoded");

    for (const std::uint8_t version : {0U, 2U, 255U}) {
        malformed = valid;
        malformed[4] = static_cast<char>(version);
        requireDecodeFailure(malformed,
                             connection_metadata_status_t::UNSUPPORTED_VERSION,
                             "unsupported metadata version was decoded");
    }
    for (const std::size_t reserved_offset : {5U, 6U, 7U}) {
        malformed = valid;
        malformed[reserved_offset] = 1;
        requireDecodeFailure(malformed,
                             connection_metadata_status_t::NONZERO_RESERVED,
                             "nonzero header reserved byte was decoded");
    }

    malformed = valid;
    writeU32(malformed, 8, static_cast<std::uint32_t>(valid.size() - 1));
    requireDecodeFailure(malformed,
                         connection_metadata_status_t::MALFORMED_METADATA,
                         "incorrect declared metadata size was decoded");

    malformed = valid;
    writeU32(malformed, 12, 0);
    requireDecodeFailure(malformed,
                         connection_metadata_status_t::INVALID_WORKER_COUNT,
                         "zero worker count was decoded");

    malformed = valid;
    writeU32(malformed, 12, static_cast<std::uint32_t>(connection_metadata_max_workers + 1));
    requireDecodeFailure(malformed,
                         connection_metadata_status_t::INVALID_WORKER_COUNT,
                         "oversized worker count was decoded");

    for (const std::size_t invalid_uuid_byte : {16U, 22U, 24U, 32U, 38U, 40U}) {
        malformed = valid;
        malformed[invalid_uuid_byte] = 0;
        if (invalid_uuid_byte == 16 || invalid_uuid_byte == 32) {
            malformed[invalid_uuid_byte + 6] = 0;
        }
        requireDecodeFailure(malformed,
                             connection_metadata_status_t::INVALID_IDENTITY,
                             "noncanonical backend or worker UUID was decoded");
    }
}

void
testDecodeWorkerValidation() {
    const std::string valid = encode(makeMetadata());
    std::string malformed;

    malformed = valid;
    writeU32(malformed, 48, 0);
    requireDecodeFailure(malformed,
                         connection_metadata_status_t::INVALID_WORKER_ADDRESS,
                         "zero-length worker address was decoded");

    malformed = valid;
    writeU32(
        malformed, 48, static_cast<std::uint32_t>(connection_metadata_max_worker_address_size + 1));
    requireDecodeFailure(malformed,
                         connection_metadata_status_t::INVALID_WORKER_ADDRESS,
                         "oversized worker address was decoded");

    malformed = valid;
    writeU32(malformed, 48, 100);
    requireDecodeFailure(malformed,
                         connection_metadata_status_t::MALFORMED_METADATA,
                         "worker address extending past the frame was decoded");

    for (const std::size_t reserved_offset : {52U, 53U, 54U, 55U}) {
        malformed = valid;
        malformed[reserved_offset] = 1;
        requireDecodeFailure(malformed,
                             connection_metadata_status_t::NONZERO_RESERVED,
                             "nonzero worker reserved byte was decoded");
    }

    malformed = valid;
    std::copy_n(malformed.begin() + 32, notif_wire_uuid_size, malformed.begin() + 59);
    requireDecodeFailure(malformed,
                         connection_metadata_status_t::DUPLICATE_WORKER,
                         "duplicate worker UUID was decoded");

    malformed = valid;
    malformed.push_back('x');
    writeU32(malformed, 8, static_cast<std::uint32_t>(malformed.size()));
    requireDecodeFailure(malformed,
                         connection_metadata_status_t::MALFORMED_METADATA,
                         "trailing metadata bytes were decoded");
}

} // namespace

int
main() {
    try {
        testGeneratedIdentitiesAreCanonical();
        testStableCanonicalRoundTrip();
        testCanonicalNetworkOrderLayout();
        testSymmetricSizeLimit();
        testEncodeValidation();
        testDecodeHeaderValidation();
        testDecodeWorkerValidation();
    }
    catch (const std::exception &error) {
        std::cerr << "ucx_connection_metadata_test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
