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
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "ucx_notif_wire.h"

namespace {

using namespace nixl::ucx;

static_assert(notif_wire_header_size == 128);
static_assert(notif_wire_max_frame_size == 65536);

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

notif_wire_envelope_t
makeEnvelope(notif_wire_type_t type) {
    return {
        .type = type,
        .senderAgentIncarnation = makeUuid(1),
        .recipientAgentIncarnation = makeUuid(2),
        .senderBackendIncarnation = makeUuid(3),
        .recipientBackendIncarnation = makeUuid(4),
        .senderWorkerIncarnation = makeUuid(5),
        .capability = makeUuid(6),
        .capabilityEpoch = 0x0102030405060708ULL,
    };
}

std::vector<std::uint8_t>
encode(const notif_wire_envelope_t &envelope,
       const std::vector<std::uint8_t> &payload) {
    std::vector<std::uint8_t> wire;
    require(encodeNotifWireFrame(envelope, payload, wire) ==
                notif_wire_status_t::SUCCESS,
            "valid notification wire frame was not encoded");
    return wire;
}

notif_wire_frame_view_t
makeDecodeSentinel() {
    static constexpr std::array<std::uint8_t, 4> sentinel_payload = {
        0xde, 0xad, 0xbe, 0xef};
    return {
        .envelope = makeEnvelope(notif_wire_type_t::DATA),
        .payload = sentinel_payload,
    };
}

bool
sameView(const notif_wire_frame_view_t &left,
         const notif_wire_frame_view_t &right) {
    return left.envelope == right.envelope &&
        left.payload.data() == right.payload.data() &&
        left.payload.size() == right.payload.size();
}

void
requireDecodeFailure(const std::vector<std::uint8_t> &wire,
                     notif_wire_status_t expected_status,
                     std::string_view message) {
    notif_wire_frame_view_t output = makeDecodeSentinel();
    const notif_wire_frame_view_t expected_output = output;
    require(decodeNotifWireFrame(wire, output) == expected_status, message);
    require(sameView(output, expected_output), "failed decode mutated its output");
}

void
testRoundTrips() {
    for (notif_wire_type_t type : {notif_wire_type_t::OFFER,
                                   notif_wire_type_t::ACK,
                                   notif_wire_type_t::DATA}) {
        const std::vector<std::uint8_t> payload =
            type == notif_wire_type_t::DATA ?
            std::vector<std::uint8_t>{0, 1, 2, 127, 128, 254, 255} :
            std::vector<std::uint8_t>{};
        const notif_wire_envelope_t envelope = makeEnvelope(type);
        const std::vector<std::uint8_t> wire = encode(envelope, payload);

        notif_wire_frame_view_t decoded;
        require(decodeNotifWireFrame(wire, decoded) ==
                    notif_wire_status_t::SUCCESS,
                "valid notification wire frame was not decoded");
        require(decoded.envelope == envelope &&
                    std::equal(decoded.payload.begin(),
                               decoded.payload.end(),
                               payload.begin(),
                               payload.end()),
                "notification wire round trip changed the frame");
        require(decoded.payload.data() == wire.data() + notif_wire_header_size,
                "decoded payload is not a zero-copy wire view");
    }
}

void
testCanonicalNetworkOrderLayout() {
    const notif_wire_envelope_t envelope = makeEnvelope(notif_wire_type_t::OFFER);
    const std::vector<std::uint8_t> wire = encode(envelope, {});

    const std::vector<std::uint8_t> expected_prefix = {
        'N', 'X', 'N', 'F',
        1,
        static_cast<std::uint8_t>(notif_wire_type_t::OFFER),
        0, 0,
        0, 0, 0, 128,
        0, 0, 0, 0,
    };
    require(std::equal(expected_prefix.begin(), expected_prefix.end(), wire.begin()),
            "wire prefix is not canonical");

    const std::array<std::pair<std::size_t, const notif_wire_uuid_t *>, 6>
        identity_offsets = {{
            {16, &envelope.senderAgentIncarnation},
            {32, &envelope.recipientAgentIncarnation},
            {48, &envelope.senderBackendIncarnation},
            {64, &envelope.recipientBackendIncarnation},
            {80, &envelope.senderWorkerIncarnation},
            {96, &envelope.capability},
        }};
    for (const auto &[offset, uuid] : identity_offsets) {
        require(std::equal(uuid->bytes.begin(),
                           uuid->bytes.end(),
                           wire.begin() + offset),
                "binary UUID identity moved on the wire");
    }

    const std::array<std::uint8_t, 16> expected_epoch_and_reserved = {
        1, 2, 3, 4, 5, 6, 7, 8,
        0, 0, 0, 0, 0, 0, 0, 0,
    };
    require(std::equal(expected_epoch_and_reserved.begin(),
                       expected_epoch_and_reserved.end(),
                       wire.begin() + 112),
            "capability epoch or reserved tail is not canonical");
}

void
testSymmetricFrameLimit() {
    const std::size_t maximum_payload_size =
        notif_wire_max_frame_size - notif_wire_header_size;
    const notif_wire_envelope_t envelope = makeEnvelope(notif_wire_type_t::DATA);
    std::vector<std::uint8_t> maximum_payload(maximum_payload_size);
    for (std::size_t index = 0; index < maximum_payload.size(); ++index) {
        maximum_payload[index] = static_cast<std::uint8_t>(index);
    }

    std::size_t encoded_size = 17;
    require(getNotifWireEncodedSize(
                envelope, maximum_payload.size(), encoded_size) ==
                notif_wire_status_t::SUCCESS,
            "exact-limit frame failed preflight");
    require(encoded_size == notif_wire_max_frame_size,
            "exact-limit preflight returned the wrong size");

    const std::vector<std::uint8_t> maximum_wire = encode(envelope, maximum_payload);
    require(maximum_wire.size() == notif_wire_max_frame_size,
            "exact-limit frame did not encode to exactly 64 KiB");
    notif_wire_frame_view_t decoded;
    require(decodeNotifWireFrame(maximum_wire, decoded) ==
                notif_wire_status_t::SUCCESS,
            "exact-limit frame was not decoded");
    require(std::equal(decoded.payload.begin(),
                       decoded.payload.end(),
                       maximum_payload.begin(),
                       maximum_payload.end()),
            "exact-limit frame payload changed");

    const std::vector<std::uint8_t> oversized_payload(maximum_payload_size + 1, 0xaa);
    encoded_size = 19;
    require(getNotifWireEncodedSize(
                envelope, oversized_payload.size(), encoded_size) ==
                notif_wire_status_t::FRAME_TOO_LARGE,
            "over-limit frame passed preflight");
    require(encoded_size == 19, "failed preflight mutated its size output");

    std::vector<std::uint8_t> encoded_output = {0xca, 0xfe};
    const std::vector<std::uint8_t> expected_encoded_output = encoded_output;
    require(encodeNotifWireFrame(envelope, oversized_payload, encoded_output) ==
                notif_wire_status_t::FRAME_TOO_LARGE,
            "over-limit frame was encoded");
    require(encoded_output == expected_encoded_output,
            "failed encode mutated its output");

    const std::vector<std::uint8_t> oversized_wire(
        notif_wire_max_frame_size + 1, 0xff);
    requireDecodeFailure(oversized_wire,
                         notif_wire_status_t::FRAME_TOO_LARGE,
                         "over-limit frame was parsed");
}

void
testEncodeRejectsInvalidFramesWithoutMutation() {
    const std::vector<std::uint8_t> sentinel = {0xba, 0xad, 0xf0, 0x0d};

    notif_wire_envelope_t invalid_type = makeEnvelope(notif_wire_type_t::DATA);
    invalid_type.type = static_cast<notif_wire_type_t>(255);
    std::vector<std::uint8_t> output = sentinel;
    require(encodeNotifWireFrame(invalid_type, {}, output) ==
                notif_wire_status_t::INVALID_TYPE,
            "invalid frame type was encoded");
    require(output == sentinel, "invalid-type encode mutated its output");

    const std::array<notif_wire_uuid_t notif_wire_envelope_t::*, 6> identities = {
        &notif_wire_envelope_t::senderAgentIncarnation,
        &notif_wire_envelope_t::recipientAgentIncarnation,
        &notif_wire_envelope_t::senderBackendIncarnation,
        &notif_wire_envelope_t::recipientBackendIncarnation,
        &notif_wire_envelope_t::senderWorkerIncarnation,
        &notif_wire_envelope_t::capability,
    };
    for (const auto identity : identities) {
        notif_wire_envelope_t invalid_identity =
            makeEnvelope(notif_wire_type_t::DATA);
        invalid_identity.*identity = {};
        output = sentinel;
        require(encodeNotifWireFrame(invalid_identity, {}, output) ==
                    notif_wire_status_t::INVALID_IDENTITY,
                "noncanonical binary identity was encoded");
        require(output == sentinel, "invalid-identity encode mutated its output");
    }

    notif_wire_envelope_t zero_epoch = makeEnvelope(notif_wire_type_t::DATA);
    zero_epoch.capabilityEpoch = 0;
    output = sentinel;
    require(encodeNotifWireFrame(zero_epoch, {}, output) ==
                notif_wire_status_t::INVALID_EPOCH,
            "zero capability epoch was encoded");
    require(output == sentinel, "zero-epoch encode mutated its output");

    for (const notif_wire_type_t type : {
             notif_wire_type_t::OFFER, notif_wire_type_t::ACK}) {
        output = sentinel;
        require(encodeNotifWireFrame(makeEnvelope(type), {sentinel}, output) ==
                    notif_wire_status_t::INVALID_PAYLOAD,
                "control frame accepted an opaque payload");
        require(output == sentinel, "invalid-payload encode mutated its output");
    }
}

void
testDecodeRejectsInvalidHeaderFields() {
    const std::vector<std::uint8_t> valid =
        encode(makeEnvelope(notif_wire_type_t::DATA), {1, 2, 3});

    std::vector<std::uint8_t> malformed = valid;
    malformed[0] ^= 0xffU;
    requireDecodeFailure(malformed,
                         notif_wire_status_t::MALFORMED_FRAME,
                         "invalid magic was decoded");

    for (const std::uint8_t version : {0U, 2U, 255U}) {
        malformed = valid;
        malformed[4] = version;
        requireDecodeFailure(malformed,
                             notif_wire_status_t::UNSUPPORTED_VERSION,
                             "unsupported version was decoded");
    }
    for (const std::uint8_t type : {0U, 4U, 255U}) {
        malformed = valid;
        malformed[5] = type;
        requireDecodeFailure(malformed,
                             notif_wire_status_t::INVALID_TYPE,
                             "invalid frame type was decoded");
    }
    for (const std::size_t reserved_byte : {
             6U, 7U, 120U, 121U, 122U, 123U, 124U, 125U, 126U, 127U}) {
        malformed = valid;
        malformed[reserved_byte] = 1;
        requireDecodeFailure(malformed,
                             notif_wire_status_t::NONZERO_RESERVED,
                             "nonzero reserved field was decoded");
    }

    malformed = valid;
    std::fill_n(malformed.begin() + 112, sizeof(std::uint64_t), 0);
    requireDecodeFailure(malformed,
                         notif_wire_status_t::INVALID_EPOCH,
                         "zero capability epoch was decoded");

    malformed = valid;
    malformed[11] ^= 1U;
    requireDecodeFailure(malformed,
                         notif_wire_status_t::MALFORMED_FRAME,
                         "incorrect frame length was decoded");
    malformed = valid;
    malformed[15] ^= 1U;
    requireDecodeFailure(malformed,
                         notif_wire_status_t::MALFORMED_FRAME,
                         "incorrect payload length was decoded");
    malformed = valid;
    malformed.push_back(0);
    requireDecodeFailure(malformed,
                         notif_wire_status_t::MALFORMED_FRAME,
                         "unaccounted trailing byte was decoded");

    malformed = valid;
    malformed[5] = static_cast<std::uint8_t>(notif_wire_type_t::OFFER);
    requireDecodeFailure(malformed,
                         notif_wire_status_t::INVALID_PAYLOAD,
                         "OFFER frame with payload was decoded");
    malformed[5] = static_cast<std::uint8_t>(notif_wire_type_t::ACK);
    requireDecodeFailure(malformed,
                         notif_wire_status_t::INVALID_PAYLOAD,
                         "ACK frame with payload was decoded");
}

void
testDecodeRejectsTruncationAndInvalidIdentities() {
    const std::vector<std::uint8_t> valid =
        encode(makeEnvelope(notif_wire_type_t::DATA), {1, 2, 3});
    for (std::size_t length = 0; length < valid.size(); ++length) {
        const std::vector<std::uint8_t> truncated(valid.begin(), valid.begin() + length);
        requireDecodeFailure(truncated,
                             notif_wire_status_t::MALFORMED_FRAME,
                             "truncated frame was decoded");
    }

    for (const std::size_t identity_offset : {16U, 32U, 48U, 64U, 80U, 96U}) {
        std::vector<std::uint8_t> malformed = valid;
        std::fill_n(malformed.begin() + identity_offset,
                    notif_wire_uuid_size,
                    0);
        requireDecodeFailure(malformed,
                             notif_wire_status_t::INVALID_IDENTITY,
                             "noncanonical binary identity was decoded");
    }
}

} // namespace

int
main() {
    try {
        testRoundTrips();
        testCanonicalNetworkOrderLayout();
        testSymmetricFrameLimit();
        testEncodeRejectsInvalidFramesWithoutMutation();
        testDecodeRejectsInvalidHeaderFields();
        testDecodeRejectsTruncationAndInvalidIdentities();
    } catch (const std::exception &error) {
        std::cerr << "ucx_notif_wire_test failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "ucx_notif_wire_test passed\n";
    return 0;
}
