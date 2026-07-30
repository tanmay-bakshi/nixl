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

#include "ucx_attestation.h"

#include <dlfcn.h>
#include <elf.h>
#include <link.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <tuple>
#include <utility>

#include <openssl/evp.h>
#include <ucp/api/ucp.h>

nixl_status_t
nixlUcxGetTransportModuleRequirements(
    const std::vector<nixl_xfer_attestation_transport_t> &transports,
    std::vector<nixlUcxTransportModuleRequirement> &requirements,
    std::string &error) {
    static const std::set<std::string> core_transports = {
        "posix",
        "self",
        "sysv",
        "tcp",
    };
    static const std::set<std::string> mlx5_transports = {
        "dc_mlx5",
        "gga_mlx5",
        "rc_mlx5",
        "ud_mlx5",
    };

    std::set<std::pair<std::string, std::string>> canonical_requirements;
    for (const auto &transport : transports) {
        const std::string &name = transport.transport;
        if (core_transports.find(name) != core_transports.end()) {
            continue;
        }

        if (name == "cuda_copy" || name == "cuda_ipc") {
            canonical_requirements.emplace("libuct_cuda", "libuct_cuda.so");
            continue;
        }

        if (name == "gdr_copy") {
            canonical_requirements.emplace("libuct_cuda", "libuct_cuda.so");
            canonical_requirements.emplace(
                "libuct_cuda_gdrcopy", "libuct_cuda_gdrcopy.so");
            continue;
        }

        if (name == "rc_gda") {
            canonical_requirements.emplace("libuct_ib", "libuct_ib.so");
            canonical_requirements.emplace("libuct_ib_mlx5", "libuct_ib_mlx5.so");
            canonical_requirements.emplace(
                "libuct_ib_mlx5_gda", "libuct_ib_mlx5_gda.so");
            continue;
        }

        if (mlx5_transports.find(name) != mlx5_transports.end()) {
            canonical_requirements.emplace("libuct_ib", "libuct_ib.so");
            canonical_requirements.emplace("libuct_ib_mlx5", "libuct_ib_mlx5.so");
            continue;
        }

        if (name == "rc_verbs" || name == "ud_verbs") {
            canonical_requirements.emplace("libuct_ib", "libuct_ib.so");
            continue;
        }

        if (name == "srd") {
            canonical_requirements.emplace("libuct_ib", "libuct_ib.so");
            canonical_requirements.emplace("libuct_ib_efa", "libuct_ib_efa.so");
            continue;
        }

        if (name == "cma" || name == "knem" || name == "xpmem") {
            canonical_requirements.emplace("libuct_" + name, "libuct_" + name + ".so");
            continue;
        }

        requirements.clear();
        error = "selected UCX transport has no runtime module identity rule: " + name;
        return NIXL_ERR_BACKEND;
    }

    requirements.clear();
    requirements.reserve(canonical_requirements.size());
    for (const auto &[component, soname] : canonical_requirements) {
        requirements.push_back({
            .component = component,
            .soname = soname,
        });
    }
    error.clear();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxCanonicalizeLoadedPath(const std::string &component,
                              const std::string &loaded_path,
                              std::string &canonical_path,
                              std::string &error) {
    canonical_path.clear();
    const std::filesystem::path unresolved_path(loaded_path);
    if (!unresolved_path.is_absolute()) {
        error = "loaded path is not absolute for " + component;
        return NIXL_ERR_BACKEND;
    }

    std::error_code path_error;
    const std::filesystem::path path =
        std::filesystem::canonical(unresolved_path, path_error);
    if (path_error) {
        error = "loaded path canonicalization failed for " + component;
        return NIXL_ERR_BACKEND;
    }

    canonical_path = path.string();
    error.clear();
    return NIXL_SUCCESS;
}

namespace {

constexpr size_t elf_note_alignment = 4;

struct build_id_search_t {
    ElfW(Addr) baseAddress = 0;
    bool objectFound = false;
    bool malformedNotes = false;
    std::vector<std::string> buildIds;
};

struct dynamic_library_handle_deleter {
    void
    operator()(void *handle) const noexcept {
        if (handle != nullptr) {
            (void)dlclose(handle);
        }
    }
};

struct runtime_artifact_resolution_t {
    nixl_status_t status = NIXL_ERR_BACKEND;
    std::vector<nixl_runtime_artifact_t> artifacts;
    std::string error;
};

struct loaded_object_t {
    ElfW(Addr) baseAddress = 0;
    std::string path;
};

struct loaded_object_search_t {
    std::string soname;
    std::vector<loaded_object_t> objects;
};

[[gnu::noinline]] void
ucxPluginIdentityAnchor() {}

[[nodiscard]] const void *
functionAddress(auto function) {
    return reinterpret_cast<const void *>(reinterpret_cast<uintptr_t>(function));
}

[[nodiscard]] size_t
alignElfNoteSize(size_t size) {
    if (size > std::numeric_limits<size_t>::max() - (elf_note_alignment - 1)) {
        return std::numeric_limits<size_t>::max();
    }
    return (size + elf_note_alignment - 1) & ~(elf_note_alignment - 1);
}

[[nodiscard]] std::string
hexEncode(const unsigned char *bytes, size_t size) {
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (size_t index = 0; index < size; ++index) {
        output << std::setw(2) << static_cast<unsigned int>(bytes[index]);
    }
    return output.str();
}

[[nodiscard]] bool
collectBuildIdsFromNote(const unsigned char *note_data,
                        size_t note_size,
                        build_id_search_t &search) {
    const unsigned char *cursor = note_data;
    size_t remaining = note_size;
    while (remaining >= sizeof(ElfW(Nhdr))) {
        ElfW(Nhdr) header;
        std::memcpy(&header, cursor, sizeof(header));
        cursor += sizeof(header);
        remaining -= sizeof(header);

        const size_t name_size = alignElfNoteSize(header.n_namesz);
        const size_t descriptor_size = alignElfNoteSize(header.n_descsz);
        if (name_size == std::numeric_limits<size_t>::max() ||
            descriptor_size == std::numeric_limits<size_t>::max() ||
            name_size > remaining || descriptor_size > remaining - name_size) {
            return false;
        }

        const unsigned char *name = cursor;
        const unsigned char *descriptor = cursor + name_size;
        if (header.n_type == NT_GNU_BUILD_ID && header.n_namesz == 4 &&
            std::memcmp(name, "GNU", 4) == 0) {
            if (header.n_descsz == 0) {
                return false;
            }
            search.buildIds.push_back(hexEncode(descriptor, header.n_descsz));
        }

        cursor += name_size + descriptor_size;
        remaining -= name_size + descriptor_size;
    }
    return true;
}

int
collectBuildId(dl_phdr_info *info, size_t, void *context) {
    auto &search = *static_cast<build_id_search_t *>(context);
    if (info->dlpi_addr != search.baseAddress) {
        return 0;
    }

    search.objectFound = true;
    for (ElfW(Half) index = 0; index < info->dlpi_phnum; ++index) {
        const ElfW(Phdr) &header = info->dlpi_phdr[index];
        if (header.p_type != PT_NOTE) {
            continue;
        }

        const auto *note_data = reinterpret_cast<const unsigned char *>(
            info->dlpi_addr + header.p_vaddr);
        if (!collectBuildIdsFromNote(note_data, header.p_memsz, search)) {
            search.malformedNotes = true;
            return 1;
        }
    }
    return 1;
}

[[nodiscard]] bool
matchesSoname(const std::filesystem::path &path, const std::string &soname) {
    const std::string filename = path.filename().string();
    return filename == soname ||
        (filename.size() > soname.size() &&
         filename.compare(0, soname.size(), soname) == 0 &&
         filename[soname.size()] == '.');
}

int
collectLoadedObject(dl_phdr_info *info, size_t, void *context) {
    auto &search = *static_cast<loaded_object_search_t *>(context);
    if (info->dlpi_name == nullptr || info->dlpi_name[0] == '\0' ||
        !matchesSoname(info->dlpi_name, search.soname)) {
        return 0;
    }

    search.objects.push_back({
        .baseAddress = info->dlpi_addr,
        .path = info->dlpi_name,
    });
    return 0;
}

[[nodiscard]] nixl_status_t
resolveRuntimeArtifactAt(const std::string &component,
                         ElfW(Addr) base_address,
                         const std::string &loaded_path,
                         const std::string &version,
                         nixl_runtime_artifact_t &artifact,
                         std::string &error) {
    if (base_address == 0 || loaded_path.empty() || version.empty()) {
        error = "runtime artifact identity is unavailable for " + component;
        return NIXL_ERR_BACKEND;
    }

    std::string canonical_path;
    const nixl_status_t path_status =
        nixlUcxCanonicalizeLoadedPath(component, loaded_path, canonical_path, error);
    if (path_status != NIXL_SUCCESS) {
        return path_status;
    }

    build_id_search_t search{
        .baseAddress = base_address,
    };
    (void)dl_iterate_phdr(collectBuildId, &search);
    if (!search.objectFound || search.malformedNotes || search.buildIds.empty()) {
        error = "in-memory GNU build ID resolution failed for " + component;
        return NIXL_ERR_BACKEND;
    }

    std::sort(search.buildIds.begin(), search.buildIds.end());
    search.buildIds.erase(
        std::unique(search.buildIds.begin(), search.buildIds.end()), search.buildIds.end());
    if (search.buildIds.size() != 1) {
        error = "loaded object has ambiguous GNU build IDs for " + component;
        return NIXL_ERR_BACKEND;
    }

    artifact = {
        .component = component,
        .path = std::move(canonical_path),
        .buildId = search.buildIds.front(),
        .version = version,
    };
    return NIXL_SUCCESS;
}

[[nodiscard]] nixl_status_t
resolveRuntimeArtifact(const std::string &component,
                       const void *symbol,
                       const std::string &version,
                       nixl_runtime_artifact_t &artifact,
                       std::string &error) {
    if (symbol == nullptr) {
        error = "runtime artifact symbol is unavailable for " + component;
        return NIXL_ERR_BACKEND;
    }

    Dl_info dynamic_info{};
    if (dladdr(symbol, &dynamic_info) == 0 || dynamic_info.dli_fbase == nullptr ||
        dynamic_info.dli_fname == nullptr || dynamic_info.dli_fname[0] == '\0') {
        error = "dladdr failed for " + component;
        return NIXL_ERR_BACKEND;
    }

    return resolveRuntimeArtifactAt(component,
                                    reinterpret_cast<ElfW(Addr)>(dynamic_info.dli_fbase),
                                    dynamic_info.dli_fname,
                                    version,
                                    artifact,
                                    error);
}

[[nodiscard]] nixl_status_t
resolveLoadedRuntimeArtifact(const std::string &component,
                             const std::string &soname,
                             const std::string &version,
                             nixl_runtime_artifact_t &artifact,
                             std::string &error) {
    loaded_object_search_t search{.soname = soname};
    (void)dl_iterate_phdr(collectLoadedObject, &search);
    if (search.objects.empty()) {
        error = "selected transport module is not loaded for " + component;
        return NIXL_ERR_BACKEND;
    }
    if (search.objects.size() != 1) {
        error = "selected transport module is ambiguous for " + component;
        return NIXL_ERR_BACKEND;
    }

    return resolveRuntimeArtifactAt(component,
                                    search.objects.front().baseAddress,
                                    search.objects.front().path,
                                    version,
                                    artifact,
                                    error);
}

[[nodiscard]] nixl_status_t
resolveBaseRuntimeArtifacts(std::vector<nixl_runtime_artifact_t> &artifacts,
                            std::string &error) {
    const char *ucx_version = ucp_get_version_string();
    if (ucx_version == nullptr || ucx_version[0] == '\0') {
        error = "UCX runtime version is unavailable";
        return NIXL_ERR_BACKEND;
    }

    using status_str_identity_resolver_t = const void *(*)();
    std::unique_ptr<void, dynamic_library_handle_deleter> libnixl_handle(
        dlopen("libnixl.so", RTLD_NOW | RTLD_NOLOAD));
    if (libnixl_handle == nullptr) {
        error = "loaded libnixl handle is unavailable";
        return NIXL_ERR_BACKEND;
    }
    void *resolver_symbol =
        dlsym(libnixl_handle.get(), "nixl_status_str_identity_symbol");
    if (resolver_symbol == nullptr) {
        error = "libnixl status-string identity resolver is unavailable";
        return NIXL_ERR_BACKEND;
    }
    const auto status_str_identity_resolver =
        reinterpret_cast<status_str_identity_resolver_t>(resolver_symbol);
    const void *status_str_symbol = status_str_identity_resolver();
    if (status_str_symbol == nullptr) {
        error = "libnixl status-string identity symbol is unavailable";
        return NIXL_ERR_BACKEND;
    }

    const std::array<std::tuple<std::string, const void *, std::string>, 3> inputs = {{
        {"libnixl",
         status_str_symbol,
         NIXL_VERSION},
        {"ucx-plugin",
         functionAddress(&ucxPluginIdentityAnchor),
         NIXL_UCX_PLUGIN_VERSION},
        {"libucp",
         functionAddress(&ucp_get_version_string),
         ucx_version},
    }};

    artifacts.clear();
    artifacts.reserve(inputs.size());
    for (const auto &[component, symbol, version] : inputs) {
        nixl_runtime_artifact_t artifact;
        const nixl_status_t status =
            resolveRuntimeArtifact(component, symbol, version, artifact, error);
        if (status != NIXL_SUCCESS) {
            artifacts.clear();
            return status;
        }
        artifacts.push_back(std::move(artifact));
    }
    return NIXL_SUCCESS;
}

[[nodiscard]] nixl_status_t
makeBaseRuntimeArtifacts(std::vector<nixl_runtime_artifact_t> &artifacts,
                         std::string &error) {
    static const runtime_artifact_resolution_t resolution = []() {
        runtime_artifact_resolution_t result;
        result.status = resolveBaseRuntimeArtifacts(result.artifacts, result.error);
        return result;
    }();

    artifacts = resolution.artifacts;
    error = resolution.error;
    return resolution.status;
}

[[nodiscard]] nixl_status_t
appendTransportRuntimeArtifacts(
    const std::vector<nixl_xfer_attestation_endpoint_t> &endpoints,
    std::vector<nixl_runtime_artifact_t> &runtime_artifacts,
    std::string &error) {
    const char *ucx_version = ucp_get_version_string();
    if (ucx_version == nullptr || ucx_version[0] == '\0') {
        error = "UCX runtime version is unavailable";
        return NIXL_ERR_BACKEND;
    }

    std::vector<nixl_xfer_attestation_transport_t> transports;
    for (const auto &endpoint : endpoints) {
        transports.insert(
            transports.end(), endpoint.transports.begin(), endpoint.transports.end());
    }

    std::vector<nixlUcxTransportModuleRequirement> requirements;
    const nixl_status_t requirements_status =
        nixlUcxGetTransportModuleRequirements(transports, requirements, error);
    if (requirements_status != NIXL_SUCCESS) {
        return requirements_status;
    }

    std::vector<nixl_runtime_artifact_t> transport_artifacts;
    transport_artifacts.reserve(requirements.size());
    for (const auto &requirement : requirements) {
        nixl_runtime_artifact_t artifact;
        const nixl_status_t status = resolveLoadedRuntimeArtifact(
            requirement.component, requirement.soname, ucx_version, artifact, error);
        if (status != NIXL_SUCCESS) {
            return status;
        }
        transport_artifacts.push_back(std::move(artifact));
    }

    runtime_artifacts.insert(runtime_artifacts.end(),
                             std::make_move_iterator(transport_artifacts.begin()),
                             std::make_move_iterator(transport_artifacts.end()));
    std::sort(runtime_artifacts.begin(),
              runtime_artifacts.end(),
              [](const auto &left, const auto &right) {
                  return left.component < right.component;
              });
    const auto duplicate = std::adjacent_find(
        runtime_artifacts.begin(),
        runtime_artifacts.end(),
        [](const auto &left, const auto &right) {
            return left.component == right.component;
        });
    if (duplicate != runtime_artifacts.end()) {
        error = "duplicate runtime artifact component: " + duplicate->component;
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

void
appendUint64(std::string &buffer, uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8) {
        buffer.push_back(static_cast<char>((value >> shift) & 0xff));
    }
}

void
appendString(std::string &buffer, const std::string &value) {
    appendUint64(buffer, value.size());
    buffer.append(value);
}

[[nodiscard]] nixl_status_t
sha256(const std::string &input, std::string &digest) {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
        EVP_MD_CTX_new(), &EVP_MD_CTX_free);
    if (context == nullptr) {
        return NIXL_ERR_BACKEND;
    }

    if (EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(context.get(), input.data(), input.size()) != 1) {
        return NIXL_ERR_BACKEND;
    }

    std::array<unsigned char, EVP_MAX_MD_SIZE> bytes{};
    unsigned int size = 0;
    if (EVP_DigestFinal_ex(context.get(), bytes.data(), &size) != 1) {
        return NIXL_ERR_BACKEND;
    }

    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < size; ++i) {
        output << std::setw(2) << static_cast<unsigned int>(bytes[i]);
    }
    digest = output.str();
    return NIXL_SUCCESS;
}

[[nodiscard]] nixl_status_t
makeDescriptorDigest(const nixl_xfer_attestation_t &attestation, std::string &digest) {
    std::string input("nixl-descriptor-attestation-v1", 30);
    appendUint64(input, attestation.handleIdentity);
    appendString(input, attestation.backend);
    appendString(input, attestation.localAgent);
    appendString(input, attestation.remoteAgent);
    appendUint64(input, static_cast<uint64_t>(attestation.operation));
    appendUint64(input, static_cast<uint64_t>(attestation.localMemoryType));
    appendUint64(input, static_cast<uint64_t>(attestation.remoteMemoryType));
    appendUint64(input, attestation.segments.size());

    for (const auto &segment : attestation.segments) {
        appendUint64(input, segment.index);
        appendUint64(input, segment.localAddress);
        appendUint64(input, segment.remoteAddress);
        appendUint64(input, segment.localDeviceId);
        appendUint64(input, segment.remoteDeviceId);
        appendUint64(input, segment.length);
    }

    return sha256(input, digest);
}

[[nodiscard]] nixl_status_t
makeEvidenceDigest(const nixl_xfer_attestation_t &attestation, std::string &digest) {
    std::string input("nixl-transfer-evidence-v1", 25);
    appendUint64(input, attestation.handleIdentity);
    appendUint64(input, attestation.generation);
    appendString(input, attestation.descriptorDigest);
    appendUint64(input, attestation.segments.size());

    for (const auto &segment : attestation.segments) {
        appendUint64(input, segment.index);
        appendUint64(input, segment.workerId);
        appendUint64(input, segment.workerIdentity);
        appendUint64(input, segment.endpointIdentity);
        appendString(input, segment.requestInfo);
    }

    auto endpoints = attestation.endpoints;
    std::sort(endpoints.begin(), endpoints.end(), [](const auto &left, const auto &right) {
        return std::tie(left.workerIdentity, left.endpointIdentity) <
            std::tie(right.workerIdentity, right.endpointIdentity);
    });
    appendUint64(input, endpoints.size());
    for (auto &endpoint : endpoints) {
        std::sort(endpoint.segmentIndices.begin(), endpoint.segmentIndices.end());
        std::sort(endpoint.transports.begin(),
                  endpoint.transports.end(),
                  [](const auto &left, const auto &right) {
                      return std::tie(left.transport, left.device) <
                          std::tie(right.transport, right.device);
                  });
        appendUint64(input, endpoint.workerId);
        appendUint64(input, endpoint.workerIdentity);
        appendUint64(input, endpoint.endpointIdentity);
        appendUint64(input, endpoint.segmentIndices.size());
        for (size_t index : endpoint.segmentIndices) {
            appendUint64(input, index);
        }
        appendUint64(input, endpoint.transports.size());
        for (const auto &transport : endpoint.transports) {
            appendString(input, transport.transport);
            appendString(input, transport.device);
        }
        appendUint64(input, endpoint.flushPosted ? 1 : 0);
        appendUint64(input, endpoint.remoteFlushed ? 1 : 0);
    }

    auto runtime_artifacts = attestation.runtimeArtifacts;
    std::sort(runtime_artifacts.begin(),
              runtime_artifacts.end(),
              [](const auto &left, const auto &right) {
                  return std::tie(left.component, left.path, left.buildId, left.version) <
                      std::tie(right.component, right.path, right.buildId, right.version);
              });
    appendUint64(input, runtime_artifacts.size());
    for (const auto &artifact : runtime_artifacts) {
        appendString(input, artifact.component);
        appendString(input, artifact.path);
        appendString(input, artifact.buildId);
        appendString(input, artifact.version);
    }

    return sha256(input, digest);
}

} // namespace

nixlUcxAttestationState::nixlUcxAttestationState(uint64_t handle_identity) {
    attestation_.handleIdentity = handle_identity;
    attestation_.backend = "UCX";
}

nixl_status_t
nixlUcxAttestationState::prepare(nixl_xfer_op_t operation,
                                 const nixl_meta_dlist_t &local,
                                 const nixl_meta_dlist_t &remote,
                                 const std::string &local_agent,
                                 const std::string &remote_agent) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (local.descCount() == 0 || local.descCount() != remote.descCount()) {
        return failLocked(NIXL_ERR_INVALID_PARAM, "invalid descriptor count");
    }

    attestation_.localAgent = local_agent;
    attestation_.remoteAgent = remote_agent;
    attestation_.operation = operation;
    attestation_.localMemoryType = local.getType();
    attestation_.remoteMemoryType = remote.getType();
    attestation_.segments.clear();
    attestation_.segments.reserve(local.descCount());

    for (size_t i = 0; i < static_cast<size_t>(local.descCount()); ++i) {
        if (local[i].len != remote[i].len) {
            return failLocked(NIXL_ERR_MISMATCH, "local and remote segment lengths differ");
        }

        attestation_.segments.push_back({
            .index = i,
            .localAddress = local[i].addr,
            .remoteAddress = remote[i].addr,
            .localDeviceId = local[i].devId,
            .remoteDeviceId = remote[i].devId,
            .length = local[i].len,
        });
    }

    nixl_status_t status = makeDescriptorDigest(attestation_, attestation_.descriptorDigest);
    if (status != NIXL_SUCCESS) {
        return failLocked(status, "failed to digest transfer descriptors");
    }

    std::string runtime_error;
    status = makeBaseRuntimeArtifacts(baseRuntimeArtifacts_, runtime_error);
    if (status != NIXL_SUCCESS) {
        return failLocked(status, runtime_error);
    }

    attestation_.generation = 0;
    attestation_.state = nixl_xfer_attestation_state_t::PREPARED;
    attestation_.status = NIXL_ERR_NOT_POSTED;
    attestation_.submissionSealed = false;
    attestation_.completionClaimed = false;
    attestation_.endpoints.clear();
    attestation_.runtimeArtifacts = baseRuntimeArtifacts_;
    attestation_.evidenceDigest.clear();
    attestation_.error.clear();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxAttestationState::beginSubmission() {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (attestation_.state == nixl_xfer_attestation_state_t::FAILED) {
        return attestation_.status;
    }
    if (attestation_.state == nixl_xfer_attestation_state_t::POSTING ||
        attestation_.state == nixl_xfer_attestation_state_t::IN_PROGRESS) {
        return NIXL_ERR_REPOST_ACTIVE;
    }
    if (attestation_.state == nixl_xfer_attestation_state_t::REMOTE_FLUSHED &&
        !attestation_.completionClaimed) {
        return NIXL_ERR_NOT_ALLOWED;
    }
    if (attestation_.state != nixl_xfer_attestation_state_t::PREPARED &&
        attestation_.state != nixl_xfer_attestation_state_t::REMOTE_FLUSHED) {
        return failLocked(NIXL_ERR_BACKEND, "invalid state before submission");
    }
    if (attestation_.generation == std::numeric_limits<uint64_t>::max()) {
        return failLocked(NIXL_ERR_NOT_ALLOWED, "submission generation exhausted");
    }

    ++attestation_.generation;
    attestation_.state = nixl_xfer_attestation_state_t::POSTING;
    attestation_.status = NIXL_IN_PROG;
    attestation_.submissionSealed = false;
    attestation_.completionClaimed = false;
    attestation_.endpoints.clear();
    attestation_.runtimeArtifacts = baseRuntimeArtifacts_;
    attestation_.evidenceDigest.clear();
    attestation_.error.clear();
    for (auto &segment : attestation_.segments) {
        segment.workerId = 0;
        segment.workerIdentity = 0;
        segment.endpointIdentity = 0;
        segment.requestInfo.clear();
        segment.posted = false;
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxAttestationState::recordSegment(
    size_t index,
    size_t worker_id,
    uint64_t worker_identity,
    uint64_t endpoint_identity,
    const std::vector<nixl_xfer_attestation_transport_t> &transports,
    const std::string &request_info) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (attestation_.state == nixl_xfer_attestation_state_t::FAILED) {
        return attestation_.status;
    }
    if (attestation_.state != nixl_xfer_attestation_state_t::POSTING ||
        index >= attestation_.segments.size() || request_info.empty() || transports.empty()) {
        return failLocked(NIXL_ERR_BACKEND, "invalid segment evidence");
    }

    auto &segment = attestation_.segments[index];
    if (segment.posted) {
        return failLocked(NIXL_ERR_BACKEND, "segment evidence was recorded more than once");
    }

    auto canonical_transports = transports;
    std::sort(canonical_transports.begin(),
              canonical_transports.end(),
              [](const auto &left, const auto &right) {
                  return std::tie(left.transport, left.device) <
                      std::tie(right.transport, right.device);
              });

    segment.workerId = worker_id;
    segment.workerIdentity = worker_identity;
    segment.endpointIdentity = endpoint_identity;
    segment.requestInfo = request_info;
    segment.posted = true;

    auto *endpoint = findEndpointLocked(endpoint_identity);
    if (endpoint == nullptr) {
        attestation_.endpoints.push_back({
            .workerId = worker_id,
            .workerIdentity = worker_identity,
            .endpointIdentity = endpoint_identity,
            .segmentIndices = {index},
            .transports = std::move(canonical_transports),
        });
        return NIXL_SUCCESS;
    }

    if (endpoint->workerId != worker_id || endpoint->workerIdentity != worker_identity ||
        endpoint->transports != canonical_transports) {
        return failLocked(NIXL_ERR_BACKEND, "endpoint context changed during submission");
    }
    endpoint->segmentIndices.push_back(index);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxAttestationState::recordFlush(size_t worker_id,
                                     uint64_t worker_identity,
                                     uint64_t endpoint_identity,
                                     nixl_status_t status) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (attestation_.state == nixl_xfer_attestation_state_t::FAILED) {
        return attestation_.status;
    }
    if (attestation_.state != nixl_xfer_attestation_state_t::POSTING) {
        return failLocked(NIXL_ERR_BACKEND, "endpoint flush was recorded outside posting");
    }

    auto *endpoint = findEndpointLocked(endpoint_identity);
    if (endpoint == nullptr || endpoint->workerId != worker_id ||
        endpoint->workerIdentity != worker_identity || endpoint->flushPosted) {
        return failLocked(NIXL_ERR_BACKEND, "invalid endpoint flush evidence");
    }

    if (status != NIXL_SUCCESS && status != NIXL_IN_PROG) {
        return failLocked(status, "endpoint flush post failed");
    }

    endpoint->flushPosted = true;
    endpoint->remoteFlushed = status == NIXL_SUCCESS;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxAttestationState::finishSubmission() {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (attestation_.state == nixl_xfer_attestation_state_t::FAILED) {
        return attestation_.status;
    }
    if (attestation_.state != nixl_xfer_attestation_state_t::POSTING ||
        attestation_.endpoints.empty()) {
        return failLocked(NIXL_ERR_BACKEND, "submission did not produce endpoint evidence");
    }

    const bool all_segments_posted =
        std::all_of(attestation_.segments.begin(), attestation_.segments.end(),
                    [](const auto &segment) { return segment.posted; });
    const bool all_flushes_posted =
        std::all_of(attestation_.endpoints.begin(), attestation_.endpoints.end(),
                    [](const auto &endpoint) { return endpoint.flushPosted; });
    if (!all_segments_posted || !all_flushes_posted) {
        return failLocked(NIXL_ERR_BACKEND, "submission evidence is incomplete");
    }

    std::string runtime_error;
    const nixl_status_t runtime_status = appendTransportRuntimeArtifacts(
        attestation_.endpoints, attestation_.runtimeArtifacts, runtime_error);
    if (runtime_status != NIXL_SUCCESS) {
        return failLocked(runtime_status, runtime_error);
    }

    attestation_.submissionSealed = true;
    if (allRemoteFlushedLocked()) {
        return sealCompletionLocked();
    }

    attestation_.state = nixl_xfer_attestation_state_t::IN_PROGRESS;
    attestation_.status = NIXL_IN_PROG;
    return NIXL_SUCCESS;
}

void
nixlUcxAttestationState::completeFlush(uint64_t generation, uint64_t endpoint_identity) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (attestation_.state == nixl_xfer_attestation_state_t::FAILED) {
        return;
    }
    if (generation != attestation_.generation) {
        (void)failLocked(NIXL_ERR_BACKEND, "stale endpoint flush completion");
        return;
    }
    if (attestation_.state != nixl_xfer_attestation_state_t::POSTING &&
        attestation_.state != nixl_xfer_attestation_state_t::IN_PROGRESS) {
        (void)failLocked(NIXL_ERR_BACKEND, "endpoint flush completed outside an active submission");
        return;
    }

    auto *endpoint = findEndpointLocked(endpoint_identity);
    if (endpoint == nullptr || !endpoint->flushPosted) {
        (void)failLocked(NIXL_ERR_BACKEND, "unknown endpoint flush completion");
        return;
    }
    if (endpoint->remoteFlushed) {
        (void)failLocked(NIXL_ERR_BACKEND, "endpoint flush completed more than once");
        return;
    }

    endpoint->remoteFlushed = true;
    if (attestation_.submissionSealed && allRemoteFlushedLocked()) {
        (void)sealCompletionLocked();
    }
}

void
nixlUcxAttestationState::fail(uint64_t generation,
                              nixl_status_t status,
                              const std::string &error) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (attestation_.state == nixl_xfer_attestation_state_t::FAILED) {
        return;
    }
    if (generation != attestation_.generation) {
        (void)failLocked(NIXL_ERR_BACKEND, "stale submission failure");
        return;
    }
    (void)failLocked(status, error);
}

uint64_t
nixlUcxAttestationState::getGeneration() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return attestation_.generation;
}

nixl_xfer_attestation_t
nixlUcxAttestationState::snapshot() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    nixl_xfer_attestation_t result = attestation_;
    std::sort(result.endpoints.begin(), result.endpoints.end(), [](const auto &left, const auto &right) {
        return std::tie(left.workerIdentity, left.endpointIdentity) <
            std::tie(right.workerIdentity, right.endpointIdentity);
    });
    for (auto &endpoint : result.endpoints) {
        std::sort(endpoint.segmentIndices.begin(), endpoint.segmentIndices.end());
        std::sort(endpoint.transports.begin(),
                  endpoint.transports.end(),
                  [](const auto &left, const auto &right) {
                      return std::tie(left.transport, left.device) <
                          std::tie(right.transport, right.device);
                  });
    }
    std::sort(result.runtimeArtifacts.begin(),
              result.runtimeArtifacts.end(),
              [](const auto &left, const auto &right) {
                  return std::tie(left.component, left.path, left.buildId, left.version) <
                      std::tie(right.component, right.path, right.buildId, right.version);
              });
    return result;
}

nixl_status_t
nixlUcxAttestationState::takeCompletion(nixl_xfer_attestation_t &attestation) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (attestation_.state != nixl_xfer_attestation_state_t::REMOTE_FLUSHED ||
        attestation_.status != NIXL_SUCCESS || !attestation_.submissionSealed ||
        attestation_.completionClaimed || attestation_.descriptorDigest.empty() ||
        attestation_.evidenceDigest.empty()) {
        return NIXL_ERR_NOT_ALLOWED;
    }

    attestation_.completionClaimed = true;
    attestation = attestation_;
    return NIXL_SUCCESS;
}

nixl_xfer_attestation_endpoint_t *
nixlUcxAttestationState::findEndpointLocked(uint64_t endpoint_identity) {
    auto endpoint = std::find_if(
        attestation_.endpoints.begin(), attestation_.endpoints.end(),
        [endpoint_identity](const auto &candidate) {
            return candidate.endpointIdentity == endpoint_identity;
        });
    return endpoint == attestation_.endpoints.end() ? nullptr : &*endpoint;
}

bool
nixlUcxAttestationState::allRemoteFlushedLocked() const {
    return std::all_of(attestation_.endpoints.begin(), attestation_.endpoints.end(),
                       [](const auto &endpoint) { return endpoint.remoteFlushed; });
}

nixl_status_t
nixlUcxAttestationState::failLocked(nixl_status_t status, const std::string &error) {
    if (attestation_.state == nixl_xfer_attestation_state_t::FAILED) {
        return attestation_.status;
    }
    attestation_.state = nixl_xfer_attestation_state_t::FAILED;
    attestation_.status = status < 0 ? status : NIXL_ERR_BACKEND;
    attestation_.evidenceDigest.clear();
    attestation_.error = error;
    return attestation_.status;
}

nixl_status_t
nixlUcxAttestationState::sealCompletionLocked() {
    if (attestation_.state == nixl_xfer_attestation_state_t::FAILED) {
        return attestation_.status;
    }
    if (!attestation_.submissionSealed || attestation_.endpoints.empty() ||
        !allRemoteFlushedLocked()) {
        return failLocked(NIXL_ERR_BACKEND, "completion evidence is incomplete");
    }
    attestation_.state = nixl_xfer_attestation_state_t::REMOTE_FLUSHED;
    attestation_.status = NIXL_SUCCESS;
    nixl_status_t status = makeEvidenceDigest(attestation_, attestation_.evidenceDigest);
    if (status != NIXL_SUCCESS) {
        return failLocked(status, "failed to digest transfer evidence");
    }
    return NIXL_SUCCESS;
}
