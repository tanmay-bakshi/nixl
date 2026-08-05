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
#ifndef NIXL_SRC_PLUGINS_UCX_UCX_CONNECTION_METADATA_H
#define NIXL_SRC_PLUGINS_UCX_UCX_CONNECTION_METADATA_H

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "ucx_notif_wire.h"

namespace nixl::ucx {

inline constexpr std::size_t connection_metadata_header_size = 32;
inline constexpr std::size_t connection_metadata_worker_header_size = 24;
inline constexpr std::size_t connection_metadata_max_size = 64 * 1024;
inline constexpr std::size_t connection_metadata_max_workers = 256;
inline constexpr std::size_t connection_metadata_max_worker_address_size =
    connection_metadata_max_size - connection_metadata_header_size -
    connection_metadata_worker_header_size;

enum class connection_metadata_status_t {
    SUCCESS,
    METADATA_TOO_LARGE,
    MALFORMED_METADATA,
    UNSUPPORTED_VERSION,
    NONZERO_RESERVED,
    INVALID_IDENTITY,
    INVALID_WORKER_COUNT,
    INVALID_WORKER_ADDRESS,
    DUPLICATE_WORKER,
};

struct connection_metadata_worker_t {
    notif_wire_uuid_t incarnation;
    std::string endpointAddress;

    bool
    operator==(const connection_metadata_worker_t &) const = default;
};

struct connection_metadata_t {
    notif_wire_uuid_t backendIncarnation;
    std::vector<connection_metadata_worker_t> workers;

    bool
    operator==(const connection_metadata_t &) const = default;
};

/* Generates a canonical RFC 9562 UUID-v4 for a backend or worker incarnation. */
[[nodiscard]] notif_wire_uuid_t
generateConnectionMetadataUuid();

[[nodiscard]] bool
isCanonicalConnectionMetadataUuid(const notif_wire_uuid_t &uuid) noexcept;

/* On failure, encoded_size is unchanged. */
[[nodiscard]] connection_metadata_status_t
getConnectionMetadataEncodedSize(const connection_metadata_t &metadata,
                                 std::size_t &encoded_size) noexcept;

/* On failure, output is unchanged. */
[[nodiscard]] connection_metadata_status_t
encodeConnectionMetadata(const connection_metadata_t &metadata, std::string &output);

/*
 * On failure, output is unchanged. Worker order is preserved because it is the
 * remote worker-index manifest, not an unordered set of addresses.
 */
[[nodiscard]] connection_metadata_status_t
decodeConnectionMetadata(std::string_view wire, connection_metadata_t &output);

} // namespace nixl::ucx

#endif
