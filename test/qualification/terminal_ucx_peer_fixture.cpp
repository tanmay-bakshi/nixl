/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "terminal_ucx_peer_fixture.h"

#include "terminal_ucx_api_adapter.h"

#include "backend/backend_aux.h"

#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

extern char **environ;

namespace nixl::qualification {
namespace {

    constexpr std::string_view worker_argument = "--terminal-ucx-peer-worker";
    constexpr int worker_socket_fd = 198;
    constexpr int event_timeout_ms = 30000;
    constexpr std::size_t endpoint_failure_bytes = 512U * 1024U * 1024U;
    constexpr std::size_t notification_failure_bytes = 64U * 1024U * 1024U;
    constexpr std::size_t notification_bytes = 60U * 1024U;
    constexpr std::size_t maximum_frame_bytes = 16U * 1024U * 1024U;
    constexpr std::uint64_t capability_cookie = 9001;
    constexpr std::uint64_t transfer_cookie = 9002;
    constexpr std::uint64_t unequal_worker_cookie_base = 9100;
    constexpr std::uint64_t device_id = 0;

    enum class frame_kind_t : std::uint64_t {
        HELLO = 1,
        LOAD_REMOTE_METADATA = 2,
        REMOTE_METADATA_LOADED = 3,
        SHUTDOWN = 4,
        ERROR = 5,
        ARM_EXIT_AFTER_WRITE = 6,
        EXIT_WATCH_ARMED = 7,
        DRAIN_NOTIFICATIONS = 8,
        NOTIFICATIONS_DRAINED = 9,
        ARM_ADMISSION_RECEIPT_HOLD = 10,
        ADMISSION_RECEIPT_HOLD_ARMED = 11,
        ADMISSION_RECEIPT_HELD = 12,
        RELEASE_ADMISSION_RECEIPT = 13,
        ADMISSION_RECEIPT_RELEASED = 14,
    };

    class peer_fixture_error final : public std::runtime_error {
    public:
        using std::runtime_error::runtime_error;
    };

    struct frame_t {
        frame_kind_t kind = frame_kind_t::ERROR;
        std::string payload;
    };

    struct peer_hello_t {
        std::uintptr_t address = 0;
        std::size_t capacity = 0;
        nixl_blob_t metadata;
    };

    struct observed_event_t {
        nixl_terminal_event_t event;
        bool ownerWoken = false;
    };

    struct admission_receipt_authority_t {
        std::uint64_t sourceHandleIdentity = 0;
        std::uint64_t sourceGeneration = 0;
        std::uint64_t deliveryIdentity = 0;
    };

    [[nodiscard]] std::uint64_t
    monotonicRawNs() {
        timespec timestamp = {};
        if (::clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp) != 0) {
            throw peer_fixture_error("CLOCK_MONOTONIC_RAW failed");
        }
        return static_cast<std::uint64_t>(timestamp.tv_sec) * 1000000000ULL +
            static_cast<std::uint64_t>(timestamp.tv_nsec);
    }

    void
    require(bool condition, std::string_view message) {
        if (!condition) {
            throw peer_fixture_error(std::string(message));
        }
    }

    void
    requireStatus(nixl_status_t actual, nixl_status_t expected, std::string_view operation) {
        if (actual == expected) {
            return;
        }
        throw peer_fixture_error(std::string(operation) + " failed with " +
                                 nixlEnumStrings::statusStr(actual));
    }

    [[nodiscard]] std::string
    encodeAdmissionReceiptAuthority(const admission_receipt_authority_t &authority) {
        std::string payload(sizeof(authority.sourceHandleIdentity) +
                                sizeof(authority.sourceGeneration) +
                                sizeof(authority.deliveryIdentity),
                            '\0');
        std::size_t offset = 0;
        std::memcpy(payload.data() + offset,
                    &authority.sourceHandleIdentity,
                    sizeof(authority.sourceHandleIdentity));
        offset += sizeof(authority.sourceHandleIdentity);
        std::memcpy(payload.data() + offset,
                    &authority.sourceGeneration,
                    sizeof(authority.sourceGeneration));
        offset += sizeof(authority.sourceGeneration);
        std::memcpy(payload.data() + offset,
                    &authority.deliveryIdentity,
                    sizeof(authority.deliveryIdentity));
        return payload;
    }

    [[nodiscard]] admission_receipt_authority_t
    decodeAdmissionReceiptAuthority(std::string_view payload) {
        admission_receipt_authority_t authority;
        require(payload.size() ==
                    sizeof(authority.sourceHandleIdentity) + sizeof(authority.sourceGeneration) +
                        sizeof(authority.deliveryIdentity),
                "admission-receipt authority has an invalid size");
        std::size_t offset = 0;
        std::memcpy(&authority.sourceHandleIdentity,
                    payload.data() + offset,
                    sizeof(authority.sourceHandleIdentity));
        offset += sizeof(authority.sourceHandleIdentity);
        std::memcpy(&authority.sourceGeneration,
                    payload.data() + offset,
                    sizeof(authority.sourceGeneration));
        offset += sizeof(authority.sourceGeneration);
        std::memcpy(&authority.deliveryIdentity,
                    payload.data() + offset,
                    sizeof(authority.deliveryIdentity));
        require(authority.sourceHandleIdentity != 0 && authority.sourceGeneration != 0 &&
                    authority.deliveryIdentity != 0,
                "admission-receipt authority is incomplete");
        return authority;
    }

    void
    sendBytes(int fd, const void *data, std::size_t size) {
        const auto *position = static_cast<const std::uint8_t *>(data);
        std::size_t remaining = size;
        while (remaining > 0) {
            const ssize_t sent = ::send(fd, position, remaining, MSG_NOSIGNAL);
            if (sent > 0) {
                position += sent;
                remaining -= static_cast<std::size_t>(sent);
                continue;
            }
            if (sent < 0 && errno == EINTR) {
                continue;
            }
            throw peer_fixture_error("peer control socket write failed");
        }
    }

    void
    receiveBytes(int fd, void *data, std::size_t size) {
        auto *position = static_cast<std::uint8_t *>(data);
        std::size_t remaining = size;
        while (remaining > 0) {
            const ssize_t received = ::recv(fd, position, remaining, 0);
            if (received > 0) {
                position += received;
                remaining -= static_cast<std::size_t>(received);
                continue;
            }
            if (received < 0 && errno == EINTR) {
                continue;
            }
            throw peer_fixture_error(received == 0 ? "peer control socket closed unexpectedly" :
                                                     "peer control socket read failed");
        }
    }

    void
    sendFrame(int fd, frame_kind_t kind, std::string_view payload = {}) {
        require(payload.size() <= maximum_frame_bytes, "peer control frame exceeds its bound");
        const std::array<std::uint64_t, 2> header = {
            static_cast<std::uint64_t>(kind),
            static_cast<std::uint64_t>(payload.size()),
        };
        sendBytes(fd, header.data(), sizeof(header));
        if (!payload.empty()) {
            sendBytes(fd, payload.data(), payload.size());
        }
    }

    [[nodiscard]] frame_t
    receiveFrame(int fd) {
        std::array<std::uint64_t, 2> header = {};
        receiveBytes(fd, header.data(), sizeof(header));
        require(header[1] <= maximum_frame_bytes, "peer control frame declared an invalid length");
        frame_t frame{
            .kind = static_cast<frame_kind_t>(header[0]),
            .payload = std::string(static_cast<std::size_t>(header[1]), '\0'),
        };
        if (!frame.payload.empty()) {
            receiveBytes(fd, frame.payload.data(), frame.payload.size());
        }
        if (frame.kind == frame_kind_t::ERROR) {
            throw peer_fixture_error("peer worker failed: " + frame.payload);
        }
        return frame;
    }

    [[nodiscard]] std::string
    encodeHello(std::uintptr_t address, std::size_t capacity, std::string_view metadata) {
        require(address != 0 && capacity > 0, "peer registration is invalid");
        std::string payload(sizeof(std::uint64_t) * 2 + metadata.size(), '\0');
        const std::uint64_t address_value = static_cast<std::uint64_t>(address);
        const std::uint64_t capacity_value = static_cast<std::uint64_t>(capacity);
        std::memcpy(payload.data(), &address_value, sizeof(address_value));
        std::memcpy(
            payload.data() + sizeof(address_value), &capacity_value, sizeof(capacity_value));
        std::memcpy(payload.data() + sizeof(address_value) + sizeof(capacity_value),
                    metadata.data(),
                    metadata.size());
        return payload;
    }

    [[nodiscard]] peer_hello_t
    decodeHello(std::string_view payload) {
        require(payload.size() > sizeof(std::uint64_t) * 2, "peer hello is incomplete");
        std::uint64_t address = 0;
        std::uint64_t capacity = 0;
        std::memcpy(&address, payload.data(), sizeof(address));
        std::memcpy(&capacity, payload.data() + sizeof(address), sizeof(capacity));
        require(address != 0 && capacity > 0 && capacity <= std::numeric_limits<std::size_t>::max(),
                "peer hello contains an invalid registration");
        return {
            .address = static_cast<std::uintptr_t>(address),
            .capacity = static_cast<std::size_t>(capacity),
            .metadata = std::string(payload.substr(sizeof(address) + sizeof(capacity))),
        };
    }

    [[nodiscard]] std::string
    encodeExitWatch(std::size_t offset, std::uint8_t expected) {
        const std::array<std::uint64_t, 2> values = {
            static_cast<std::uint64_t>(offset),
            static_cast<std::uint64_t>(expected),
        };
        return std::string(reinterpret_cast<const char *>(values.data()), sizeof(values));
    }

    [[nodiscard]] std::pair<std::size_t, std::uint8_t>
    decodeExitWatch(std::string_view payload) {
        require(payload.size() == sizeof(std::uint64_t) * 2,
                "peer exit-watch frame has an invalid length");
        std::array<std::uint64_t, 2> values = {};
        std::memcpy(values.data(), payload.data(), payload.size());
        require(values[0] <= std::numeric_limits<std::size_t>::max() &&
                    values[1] <= std::numeric_limits<std::uint8_t>::max(),
                "peer exit-watch frame is invalid");
        return {
            static_cast<std::size_t>(values[0]),
            static_cast<std::uint8_t>(values[1]),
        };
    }

    [[nodiscard]] std::string
    encodeNotifications(const std::vector<std::string> &notifications) {
        std::size_t encoded_size = sizeof(std::uint64_t);
        for (const std::string &notification : notifications) {
            require(notification.size() <= maximum_frame_bytes,
                    "peer notification exceeds its control-frame bound");
            require(encoded_size <=
                        maximum_frame_bytes - sizeof(std::uint64_t) - notification.size(),
                    "peer notification batch exceeds its control-frame bound");
            encoded_size += sizeof(std::uint64_t) + notification.size();
        }
        std::string payload(encoded_size, '\0');
        const std::uint64_t count = notifications.size();
        std::size_t offset = 0;
        std::memcpy(payload.data() + offset, &count, sizeof(count));
        offset += sizeof(count);
        for (const std::string &notification : notifications) {
            const std::uint64_t size = notification.size();
            std::memcpy(payload.data() + offset, &size, sizeof(size));
            offset += sizeof(size);
            std::memcpy(payload.data() + offset, notification.data(), notification.size());
            offset += notification.size();
        }
        return payload;
    }

    [[nodiscard]] std::vector<std::string>
    decodeNotifications(std::string_view payload) {
        require(payload.size() >= sizeof(std::uint64_t), "peer notification batch is incomplete");
        std::uint64_t count = 0;
        std::memcpy(&count, payload.data(), sizeof(count));
        require(count <= maximum_frame_bytes / sizeof(std::uint64_t),
                "peer notification count exceeds its bound");
        std::size_t offset = sizeof(count);
        std::vector<std::string> notifications;
        notifications.reserve(static_cast<std::size_t>(count));
        for (std::uint64_t index = 0; index < count; ++index) {
            require(offset <= payload.size() && payload.size() - offset >= sizeof(std::uint64_t),
                    "peer notification batch omitted a length");
            std::uint64_t size = 0;
            std::memcpy(&size, payload.data() + offset, sizeof(size));
            offset += sizeof(size);
            require(size <= payload.size() - offset,
                    "peer notification batch declared a truncated payload");
            notifications.emplace_back(payload.substr(offset, static_cast<std::size_t>(size)));
            offset += static_cast<std::size_t>(size);
        }
        require(offset == payload.size(), "peer notification batch retained trailing bytes");
        return notifications;
    }

    [[nodiscard]] nixlAgentConfig
    agentConfig(bool use_progress_thread = true) {
        nixlAgentConfig config;
        config.useProgThread = use_progress_thread;
        config.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
        return config;
    }

    [[nodiscard]] nixl_b_params_t
    backendParameters(const std::string &engine,
                      std::optional<std::size_t> worker_count = std::nullopt) {
        require(engine == "shared" || engine == "thread_pool", "peer fixture engine is invalid");
        nixl_b_params_t parameters;
        parameters["ucx_error_handling_mode"] = "peer";
        parameters["split_batch_size"] = "2";
        if (engine == "shared") {
            const std::size_t workers = worker_count.value_or(2);
            require(workers > 0, "shared peer fixture requires at least one worker");
            parameters["num_workers"] = std::to_string(workers);
            parameters["num_threads"] = "0";
            return parameters;
        }
        const std::size_t workers = worker_count.value_or(3);
        require(workers >= 2, "thread-pool peer fixture requires at least two workers");
        parameters["num_workers"] = std::to_string(workers);
        parameters["num_threads"] = std::to_string(workers - 1);
        return parameters;
    }

    class registered_dram_t final {
    public:
        registered_dram_t(nixlAgent &agent, nixlBackendH *backend, std::size_t capacity)
            : agent_(agent),
              backend_(backend),
              bytes_(capacity, 0) {
            nixl_opt_args_t arguments;
            arguments.backends = {backend_};
            requireStatus(agent_.registerMem(registrationList(), &arguments),
                          NIXL_SUCCESS,
                          "register peer DRAM");
            registered_ = true;
        }

        registered_dram_t(const registered_dram_t &) = delete;
        registered_dram_t &
        operator=(const registered_dram_t &) = delete;

        ~registered_dram_t() {
            if (!registered_) {
                return;
            }
            nixl_opt_args_t arguments;
            arguments.backends = {backend_};
            static_cast<void>(agent_.deregisterMem(registrationList(), &arguments));
        }

        void
        fill(std::uint8_t seed) {
            for (std::size_t index = 0; index < bytes_.size(); ++index) {
                bytes_[index] = static_cast<std::uint8_t>(seed + index * 29U);
            }
        }

        [[nodiscard]] std::uintptr_t
        address() const noexcept {
            return reinterpret_cast<std::uintptr_t>(bytes_.data());
        }

        [[nodiscard]] std::size_t
        capacity() const noexcept {
            return bytes_.size();
        }

        [[nodiscard]] std::uint8_t
        byteAt(std::size_t offset) const {
            require(offset < bytes_.size(), "peer DRAM byte offset is out of range");
            return bytes_[offset];
        }

        void
        waitForByte(std::size_t offset, std::uint8_t expected) const {
            require(offset < bytes_.size(), "peer exit-watch offset is out of range");
            const volatile std::uint8_t *const observed = bytes_.data() + offset;
            while (*observed != expected) {
                std::this_thread::yield();
            }
        }

        [[nodiscard]] nixl_xfer_dlist_t
        transferList(std::size_t size, std::size_t descriptor_count = 1) const {
            require(size > 0 && size <= bytes_.size(), "peer transfer exceeds registered DRAM");
            require(descriptor_count > 0 && descriptor_count <= size,
                    "peer transfer descriptor geometry is invalid");
            nixl_xfer_dlist_t descriptors(DRAM_SEG);
            const std::size_t base_size = size / descriptor_count;
            const std::size_t remainder = size % descriptor_count;
            std::size_t offset = 0;
            for (std::size_t index = 0; index < descriptor_count; ++index) {
                const std::size_t descriptor_size = base_size + (index < remainder ? 1U : 0U);
                descriptors.addDesc(nixlBasicDesc(address() + offset, descriptor_size, device_id));
                offset += descriptor_size;
            }
            return descriptors;
        }

    private:
        [[nodiscard]] nixl_reg_dlist_t
        registrationList() const {
            nixl_reg_dlist_t descriptors(DRAM_SEG);
            descriptors.addDesc(nixlBlobDesc(address(), bytes_.size(), device_id));
            return descriptors;
        }

        nixlAgent &agent_;
        nixlBackendH *backend_;
        std::vector<std::uint8_t> bytes_;
        bool registered_ = false;
    };

    class admission_receipt_barrier_t final : public nixlBackendAdmissionReceiptBarrier {
    public:
        admission_receipt_barrier_t() {
            descriptor_ = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
            if (descriptor_ < 0) {
                throw peer_fixture_error("failed to create admission-receipt eventfd");
            }
        }

        admission_receipt_barrier_t(const admission_receipt_barrier_t &) = delete;
        admission_receipt_barrier_t &
        operator=(const admission_receipt_barrier_t &) = delete;

        ~admission_receipt_barrier_t() override {
            if (descriptor_ >= 0) {
                static_cast<void>(::close(descriptor_));
            }
        }

        [[nodiscard]] nixl_status_t
        deferAfterAdmission(const nixlBackendAdmissionReceiptAuthority &authority,
                            schedule_receipt_t schedule_receipt) noexcept override {
            if (authority.sourceHandleIdentity == 0 || authority.sourceGeneration == 0 ||
                authority.deliveryIdentity == 0 || !schedule_receipt) {
                return NIXL_ERR_INVALID_PARAM;
            }
            {
                const std::lock_guard lock(mutex_);
                if (held_ || released_ || scheduleReceipt_) {
                    return NIXL_ERR_NOT_ALLOWED;
                }
                authority_ = {
                    .sourceHandleIdentity = authority.sourceHandleIdentity,
                    .sourceGeneration = authority.sourceGeneration,
                    .deliveryIdentity = authority.deliveryIdentity,
                };
                scheduleReceipt_ = std::move(schedule_receipt);
                held_ = true;
            }
            const std::uint64_t signal = 1;
            ssize_t written = -1;
            do {
                written = ::write(descriptor_, &signal, sizeof(signal));
            } while (written < 0 && errno == EINTR);
            return written == static_cast<ssize_t>(sizeof(signal)) ? NIXL_SUCCESS :
                                                                     NIXL_ERR_BACKEND;
        }

        [[nodiscard]] int
        descriptor() const noexcept {
            return descriptor_;
        }

        [[nodiscard]] admission_receipt_authority_t
        takeHeldAuthority() {
            std::uint64_t signal = 0;
            ssize_t read_status = -1;
            do {
                read_status = ::read(descriptor_, &signal, sizeof(signal));
            } while (read_status < 0 && errno == EINTR);
            require(read_status == static_cast<ssize_t>(sizeof(signal)) && signal == 1,
                    "admission-receipt eventfd carried an invalid signal");
            const std::lock_guard lock(mutex_);
            require(held_ && !released_ && scheduleReceipt_,
                    "admission-receipt eventfd lost its held authority");
            return authority_;
        }

        [[nodiscard]] nixl_status_t
        release() {
            schedule_receipt_t schedule_receipt;
            {
                const std::lock_guard lock(mutex_);
                if (!held_ || released_ || !scheduleReceipt_) {
                    return NIXL_ERR_NOT_ALLOWED;
                }
                released_ = true;
                schedule_receipt = std::move(scheduleReceipt_);
            }
            return schedule_receipt();
        }

        [[nodiscard]] bool
        outstanding() const noexcept {
            const std::lock_guard lock(mutex_);
            return held_ && !released_;
        }

    private:
        int descriptor_ = -1;
        mutable std::mutex mutex_;
        admission_receipt_authority_t authority_;
        schedule_receipt_t scheduleReceipt_;
        bool held_ = false;
        bool released_ = false;
    };

    class peer_process_t final {
    public:
        peer_process_t(const std::string &engine,
                       std::optional<std::size_t> worker_count = std::nullopt) {
            const std::filesystem::path executable = std::filesystem::canonical("/proc/self/exe");
            const std::string executable_string = executable.string();
            const std::string fd_string = std::to_string(worker_socket_fd);
            const std::string worker_count_string = std::to_string(worker_count.value_or(0));
            std::array<char *, 9> arguments = {
                const_cast<char *>(executable_string.c_str()),
                const_cast<char *>(worker_argument.data()),
                const_cast<char *>("--fd"),
                const_cast<char *>(fd_string.c_str()),
                const_cast<char *>("--engine"),
                const_cast<char *>(engine.c_str()),
                const_cast<char *>("--worker-count"),
                const_cast<char *>(worker_count_string.c_str()),
                nullptr,
            };

            std::array<int, 2> sockets = {-1, -1};
            if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets.data()) != 0) {
                throw peer_fixture_error("failed to create peer control socket");
            }

            posix_spawn_file_actions_t actions = {};
            int status = ::posix_spawn_file_actions_init(&actions);
            const bool actions_initialized = status == 0;
            if (status == 0) {
                status = ::posix_spawn_file_actions_addclose(&actions, sockets[0]);
            }
            if (status == 0) {
                status = ::posix_spawn_file_actions_adddup2(&actions, sockets[1], worker_socket_fd);
            }
            if (status == 0 && sockets[1] != worker_socket_fd) {
                status = ::posix_spawn_file_actions_addclose(&actions, sockets[1]);
            }
            if (status == 0) {
                status = ::posix_spawn(
                    &pid_, executable_string.c_str(), &actions, nullptr, arguments.data(), environ);
            }
            if (actions_initialized) {
                static_cast<void>(::posix_spawn_file_actions_destroy(&actions));
            }
            static_cast<void>(::close(sockets[1]));
            if (status != 0) {
                static_cast<void>(::close(sockets[0]));
                throw peer_fixture_error("failed to spawn TCP peer worker: " +
                                         std::string(std::strerror(status)));
            }
            socket_ = sockets[0];
            try {
                hello_ = decodeHello(expect(frame_kind_t::HELLO).payload);
            }
            catch (...) {
                terminateNoexcept();
                throw;
            }
        }

        peer_process_t(const peer_process_t &) = delete;
        peer_process_t &
        operator=(const peer_process_t &) = delete;

        ~peer_process_t() {
            terminateNoexcept();
        }

        [[nodiscard]] const peer_hello_t &
        hello() const noexcept {
            return hello_;
        }

        void
        loadRemoteMetadata(std::string_view metadata) {
            sendFrame(socket_, frame_kind_t::LOAD_REMOTE_METADATA, metadata);
            static_cast<void>(expect(frame_kind_t::REMOTE_METADATA_LOADED));
        }

        void
        armExitAfterWrite(std::size_t offset, std::uint8_t expected) {
            sendFrame(
                socket_, frame_kind_t::ARM_EXIT_AFTER_WRITE, encodeExitWatch(offset, expected));
            static_cast<void>(expect(frame_kind_t::EXIT_WATCH_ARMED));
        }

        [[nodiscard]] std::vector<std::string>
        drainNotifications() {
            sendFrame(socket_, frame_kind_t::DRAIN_NOTIFICATIONS);
            return decodeNotifications(expect(frame_kind_t::NOTIFICATIONS_DRAINED).payload);
        }

        void
        armAdmissionReceiptHold() {
            sendFrame(socket_, frame_kind_t::ARM_ADMISSION_RECEIPT_HOLD);
            static_cast<void>(expect(frame_kind_t::ADMISSION_RECEIPT_HOLD_ARMED));
        }

        [[nodiscard]] admission_receipt_authority_t
        waitForAdmissionReceiptHeld() const {
            return decodeAdmissionReceiptAuthority(
                expect(frame_kind_t::ADMISSION_RECEIPT_HELD).payload);
        }

        void
        releaseAdmissionReceipt() {
            sendFrame(socket_, frame_kind_t::RELEASE_ADMISSION_RECEIPT);
            static_cast<void>(expect(frame_kind_t::ADMISSION_RECEIPT_RELEASED));
        }

        [[nodiscard]] bool
        shutdownAndWait() {
            require(pid_ > 0 && !waited_, "peer worker cannot shut down");
            sendFrame(socket_, frame_kind_t::SHUTDOWN);
            int status = 0;
            while (::waitpid(pid_, &status, 0) < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw peer_fixture_error("failed to reap shut-down TCP peer worker");
            }
            waited_ = true;
            static_cast<void>(::close(socket_));
            socket_ = -1;
            return WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS;
        }

        [[nodiscard]] bool
        waitForExitSignal(int expected_signal) {
            require(pid_ > 0 && !waited_, "peer worker is not live");
            int status = 0;
            while (::waitpid(pid_, &status, 0) < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw peer_fixture_error("failed to reap self-terminated TCP peer worker");
            }
            waited_ = true;
            static_cast<void>(::close(socket_));
            socket_ = -1;
            return WIFSIGNALED(status) && WTERMSIG(status) == expected_signal;
        }

        [[nodiscard]] bool
        killAndWait() {
            require(pid_ > 0 && !waited_, "peer worker is not live");
            if (::kill(pid_, SIGKILL) != 0) {
                throw peer_fixture_error("failed to kill TCP peer worker");
            }
            int status = 0;
            while (::waitpid(pid_, &status, 0) < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw peer_fixture_error("failed to reap TCP peer worker");
            }
            waited_ = true;
            static_cast<void>(::close(socket_));
            socket_ = -1;
            return WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL;
        }

    private:
        void
        terminateNoexcept() noexcept {
            if (socket_ >= 0) {
                static_cast<void>(::close(socket_));
                socket_ = -1;
            }
            if (pid_ <= 0 || waited_) {
                return;
            }
            static_cast<void>(::kill(pid_, SIGKILL));
            int status = 0;
            while (::waitpid(pid_, &status, 0) < 0 && errno == EINTR) {}
            waited_ = true;
        }

        [[nodiscard]] frame_t
        expect(frame_kind_t expected) const {
            frame_t frame = receiveFrame(socket_);
            require(frame.kind == expected, "peer control protocol changed state unexpectedly");
            return frame;
        }

        pid_t pid_ = -1;
        int socket_ = -1;
        bool waited_ = false;
        peer_hello_t hello_;
    };

    class terminal_inbox_t final {
    public:
        explicit terminal_inbox_t(terminal_ucx_api_adapter_t &adapter) : adapter_(adapter) {
            requireStatus(adapter_.fileno(fd_), NIXL_SUCCESS, "get peer terminal fd");
        }

        [[nodiscard]] observed_event_t
        take(nixl_terminal_event_kind_t kind, std::uint64_t owner_cookie) {
            const std::uint64_t deadline =
                monotonicRawNs() + static_cast<std::uint64_t>(event_timeout_ms) * 1000000ULL;
            while (true) {
                const auto event = std::find_if(
                    pending_.begin(), pending_.end(), [kind, owner_cookie](const auto &candidate) {
                        return candidate.event.kind == kind &&
                            candidate.event.ownerCookie == owner_cookie;
                    });
                if (event != pending_.end()) {
                    observed_event_t result = *event;
                    pending_.erase(event);
                    return result;
                }
                const std::uint64_t now = monotonicRawNs();
                require(now < deadline, "peer terminal event timed out");
                const std::uint64_t remaining_ns = deadline - now;
                const std::uint64_t remaining_ms = (remaining_ns + 999999ULL) / 1000000ULL;
                const int poll_timeout = static_cast<int>(std::min<std::uint64_t>(
                    remaining_ms, static_cast<std::uint64_t>(std::numeric_limits<int>::max())));
                pollfd descriptor = {.fd = fd_, .events = POLLIN, .revents = 0};
                int status = 0;
                do {
                    status = ::poll(&descriptor, 1, poll_timeout);
                } while (status < 0 && errno == EINTR);
                require(status == 1 && (descriptor.revents & POLLIN) != 0,
                        "peer terminal event poll failed");
                nixl_terminal_event_batch_t batch;
                requireStatus(adapter_.drain(batch), NIXL_SUCCESS, "drain peer terminal events");
                require(!batch.events.empty(), "peer terminal wake contained no event");
                for (nixl_terminal_event_t &candidate : batch.events) {
                    pending_.push_back({
                        .event = std::move(candidate),
                        .ownerWoken = batch.wakeCount > 0,
                    });
                }
            }
        }

        [[nodiscard]] bool
        empty() const noexcept {
            return pending_.empty();
        }

    private:
        terminal_ucx_api_adapter_t &adapter_;
        int fd_ = -1;
        std::vector<observed_event_t> pending_;
    };

    [[nodiscard]] tcp_peer_channel_observation_t
    observeChannel(const terminal_channel_inventory_t &inventory) {
        return {
            .capacity = inventory.capacity,
            .queuedChannelEvents = inventory.queuedChannelEvents,
            .activeChannelSubscriptions = inventory.activeChannelSubscriptions,
            .retainedPublicSubscriptions = inventory.retainedPublicSubscriptions,
            .backendProducers = inventory.backendProducers,
            .activeCallbackSlots = inventory.activeCallbackSlots,
            .queuedOwnerContinuations = inventory.queuedOwnerContinuations,
            .acceptingSubscriptions = inventory.acceptingSubscriptions,
            .closed = inventory.closed,
            .fatal = inventory.fatal,
            .eventfdError = inventory.eventfdError,
        };
    }

    [[nodiscard]] tcp_peer_channel_observation_t
    closeAndObserveChannel(terminal_ucx_api_adapter_t &adapter) {
        requireStatus(adapter.close(), NIXL_SUCCESS, "close peer terminal channel");
        terminal_channel_inventory_t inventory;
        requireStatus(adapter.queryInventory(inventory),
                      NIXL_SUCCESS,
                      "query peer terminal channel inventory");
        const bool clean = inventory.queuedChannelEvents == 0 &&
            inventory.activeChannelSubscriptions == 0 &&
            inventory.retainedPublicSubscriptions == 0 && inventory.backendProducers == 0 &&
            inventory.activeCallbackSlots == 0 && inventory.queuedOwnerContinuations == 0 &&
            !inventory.acceptingSubscriptions && inventory.closed && inventory.fatal == 0 &&
            inventory.eventfdError == 0;
        require(clean, "peer terminal channel retained lifecycle state");
        return observeChannel(inventory);
    }

    struct source_fixture_t {
        source_fixture_t(const std::string &engine,
                         std::size_t bytes,
                         std::optional<std::size_t> worker_count = std::nullopt)
            : agent("qualification-peer-source-" +
                        std::to_string(static_cast<unsigned long long>(::getpid())),
                    agentConfig()) {
            requireStatus(
                agent.createBackend("UCX", backendParameters(engine, worker_count), backend),
                NIXL_SUCCESS,
                "create source UCX backend");
            require(backend != nullptr, "source UCX backend is null");
            memory = std::make_unique<registered_dram_t>(agent, backend, bytes);
            memory->fill(37);
        }

        nixlAgent agent;
        nixlBackendH *backend = nullptr;
        std::unique_ptr<registered_dram_t> memory;
    };

    [[nodiscard]] nixl_xfer_dlist_t
    remoteTransferList(const peer_hello_t &hello,
                       std::size_t size,
                       std::size_t descriptor_count = 1) {
        require(size > 0 && size <= hello.capacity, "fault transfer exceeds peer registration");
        require(descriptor_count > 0 && descriptor_count <= size,
                "fault transfer descriptor geometry is invalid");
        nixl_xfer_dlist_t descriptors(DRAM_SEG);
        const std::size_t base_size = size / descriptor_count;
        const std::size_t remainder = size % descriptor_count;
        std::size_t offset = 0;
        for (std::size_t index = 0; index < descriptor_count; ++index) {
            const std::size_t descriptor_size = base_size + (index < remainder ? 1U : 0U);
            descriptors.addDesc(nixlBasicDesc(hello.address + offset, descriptor_size, device_id));
            offset += descriptor_size;
        }
        return descriptors;
    }

    [[nodiscard]] nixlXferReqH *
    createTransfer(source_fixture_t &source,
                   const peer_hello_t &peer,
                   const nixlRemoteAgentH *remote,
                   std::size_t bytes,
                   bool attached_notification,
                   std::size_t descriptor_count = 1,
                   std::optional<std::size_t> worker_id = std::nullopt,
                   std::string notification = std::string(notification_bytes, 'N')) {
        nixl_opt_args_t arguments;
        arguments.backends = {source.backend};
        if (attached_notification) {
            arguments.notif = std::move(notification);
        }
        if (worker_id.has_value()) {
            arguments.customParam = "worker_id=" + std::to_string(*worker_id);
        }
        nixlXferReqH *request = nullptr;
        requireStatus(
            source.agent.createXferReq(NIXL_WRITE,
                                       source.memory->transferList(bytes, descriptor_count),
                                       remoteTransferList(peer, bytes, descriptor_count),
                                       remote,
                                       request,
                                       &arguments),
            NIXL_SUCCESS,
            "create peer fault transfer");
        require(request != nullptr, "peer fault transfer request is null");
        return request;
    }

    [[nodiscard]] tcp_peer_capability_observation_t
    makeCapabilityReady(source_fixture_t &source,
                        peer_process_t &peer,
                        terminal_ucx_api_adapter_t &adapter,
                        terminal_inbox_t &inbox,
                        nixlRemoteAgentH *remote,
                        nixlTerminalEventSubscriptionH *&subscription) {
        requireStatus(
            adapter.subscribeCapability(remote, source.backend, capability_cookie, subscription),
            NIXL_SUCCESS,
            "subscribe peer capability");
        require(subscription != nullptr, "peer capability subscription is null");
        nixl_terminal_subscription_info_t info;
        requireStatus(adapter.querySubscription(subscription, info),
                      NIXL_SUCCESS,
                      "query armed peer capability");
        require(info.active, "peer capability subscription was not armed");

        nixl_blob_t source_metadata;
        requireStatus(
            source.agent.getLocalMD(source_metadata), NIXL_SUCCESS, "get source peer metadata");
        peer.loadRemoteMetadata(source_metadata);
        const observed_event_t ready =
            inbox.take(nixl_terminal_event_kind_t::CAPABILITY, capability_cookie);
        require(ready.event.capabilityState == nixl_terminal_capability_state_t::READY,
                "peer capability did not become ready");
        require(ready.event.identity == info.identity && ready.event.generation == info.generation,
                "peer readiness changed exact route generation");
        require(ready.ownerWoken, "peer readiness did not wake the owner");
        return {
            .handleIdentity = info.identity,
            .handleGeneration = info.generation,
            .states = {ready.event.capabilityState},
            .epochs = {ready.event.capabilityEpoch},
        };
    }

    [[nodiscard]] nixlRemoteAgentH *
    loadPeer(source_fixture_t &source, const peer_process_t &peer) {
        nixlRemoteAgentH *remote = nullptr;
        requireStatus(source.agent.loadRemoteMD(peer.hello().metadata, remote),
                      NIXL_SUCCESS,
                      "load TCP peer metadata");
        require(remote != nullptr, "TCP peer remote handle is null");
        return remote;
    }

    [[nodiscard]] bool
    attestationIsRemoteFlushed(const nixl_xfer_attestation_t &attestation) {
        return attestation.state == nixl_xfer_attestation_state_t::REMOTE_FLUSHED &&
            !attestation.endpoints.empty() &&
            std::all_of(attestation.endpoints.begin(),
                        attestation.endpoints.end(),
                        [](const auto &endpoint) { return endpoint.remoteFlushed; });
    }

    [[nodiscard]] int
    parseWorkerFd(int argc,
                  char **argv,
                  std::string &engine,
                  std::optional<std::size_t> &worker_count) {
        int fd = -1;
        for (int index = 2; index < argc; ++index) {
            const std::string_view argument(argv[index]);
            if (argument == "--fd" && index + 1 < argc) {
                fd = std::stoi(argv[++index]);
                continue;
            }
            if (argument == "--engine" && index + 1 < argc) {
                engine = argv[++index];
                continue;
            }
            if (argument == "--worker-count" && index + 1 < argc) {
                const std::size_t parsed = std::stoull(argv[++index]);
                worker_count = parsed == 0 ? std::nullopt : std::optional(parsed);
                continue;
            }
            throw peer_fixture_error("unknown peer worker argument");
        }
        require(fd >= 0, "peer worker fd is missing");
        require(engine == "shared" || engine == "thread_pool", "peer worker engine is invalid");
        return fd;
    }

    int
    runWorker(int fd, const std::string &engine, std::optional<std::size_t> worker_count) {
        nixlAgent agent("qualification-peer-destination-" +
                            std::to_string(static_cast<unsigned long long>(::getpid())),
                        agentConfig());
        nixlBackendH *backend = nullptr;
        requireStatus(agent.createBackend("UCX", backendParameters(engine, worker_count), backend),
                      NIXL_SUCCESS,
                      "create peer worker UCX backend");
        require(backend != nullptr, "peer worker UCX backend is null");
        registered_dram_t memory(agent, backend, endpoint_failure_bytes);
        nixl_blob_t metadata;
        requireStatus(agent.getLocalMD(metadata), NIXL_SUCCESS, "get peer worker metadata");
        sendFrame(
            fd, frame_kind_t::HELLO, encodeHello(memory.address(), memory.capacity(), metadata));

        frame_t frame = receiveFrame(fd);
        require(frame.kind == frame_kind_t::LOAD_REMOTE_METADATA,
                "peer worker expected remote metadata");
        nixlRemoteAgentH *remote = nullptr;
        requireStatus(agent.loadRemoteMD(frame.payload, remote),
                      NIXL_SUCCESS,
                      "load source metadata in peer worker");
        require(remote != nullptr, "peer worker source handle is null");
        sendFrame(fd, frame_kind_t::REMOTE_METADATA_LOADED);

        std::optional<frame_t> pending_frame = receiveFrame(fd);
        std::unique_ptr<admission_receipt_barrier_t> admission_barrier;
        while (true) {
            if (!pending_frame.has_value()) {
                std::array<pollfd, 2> descriptors = {
                    pollfd{.fd = fd, .events = POLLIN, .revents = 0},
                    pollfd{
                        .fd = admission_barrier == nullptr ? -1 : admission_barrier->descriptor(),
                        .events = POLLIN,
                        .revents = 0,
                    },
                };
                int poll_status = -1;
                do {
                    poll_status = ::poll(descriptors.data(), descriptors.size(), -1);
                } while (poll_status < 0 && errno == EINTR);
                require(poll_status > 0, "poll peer worker control and admission authority");
                if ((descriptors[1].revents & POLLIN) != 0) {
                    const admission_receipt_authority_t authority =
                        admission_barrier->takeHeldAuthority();
                    sendFrame(fd,
                              frame_kind_t::ADMISSION_RECEIPT_HELD,
                              encodeAdmissionReceiptAuthority(authority));
                    continue;
                }
                require((descriptors[0].revents & POLLIN) != 0,
                        "peer worker control socket failed");
                pending_frame = receiveFrame(fd);
            }
            frame = std::move(*pending_frame);
            pending_frame.reset();
            if (frame.kind == frame_kind_t::SHUTDOWN) {
                break;
            }
            if (frame.kind == frame_kind_t::ARM_ADMISSION_RECEIPT_HOLD) {
                require(admission_barrier == nullptr,
                        "peer admission-receipt hold was armed more than once");
                admission_barrier = std::make_unique<admission_receipt_barrier_t>();
                requireStatus(terminal_ucx_api_adapter_t::installAdmissionReceiptBarrier(
                                  agent, backend, admission_barrier.get()),
                              NIXL_SUCCESS,
                              "install peer admission-receipt hold");
                sendFrame(fd, frame_kind_t::ADMISSION_RECEIPT_HOLD_ARMED);
                continue;
            }
            if (frame.kind == frame_kind_t::RELEASE_ADMISSION_RECEIPT) {
                require(admission_barrier != nullptr,
                        "peer admission-receipt release has no barrier");
                requireStatus(
                    admission_barrier->release(), NIXL_SUCCESS, "release peer admission receipt");
                sendFrame(fd, frame_kind_t::ADMISSION_RECEIPT_RELEASED);
                continue;
            }
            if (frame.kind == frame_kind_t::ARM_EXIT_AFTER_WRITE) {
                const auto [offset, expected] = decodeExitWatch(frame.payload);
                require(offset < memory.capacity(), "peer exit-watch exceeds registered DRAM");
                sendFrame(fd, frame_kind_t::EXIT_WATCH_ARMED);
                memory.waitForByte(offset, expected);
                requireStatus(agent.invalidateRemoteMD(remote),
                              NIXL_SUCCESS,
                              "retire source route at data boundary");
                if (::raise(SIGKILL) != 0) {
                    throw peer_fixture_error("peer worker failed to exit at the data boundary");
                }
                throw peer_fixture_error("peer exit-watch survived SIGKILL");
            }
            if (frame.kind == frame_kind_t::DRAIN_NOTIFICATIONS) {
                nixl_remote_notifs_t notifications;
                requireStatus(agent.getRemoteNotifs(notifications),
                              NIXL_SUCCESS,
                              "drain peer authenticated notifications");
                std::vector<std::string> payloads;
                for (const auto &[handle, messages] : notifications) {
                    require(handle == remote,
                            "peer notification resolved to an unexpected source handle");
                    payloads.insert(payloads.end(), messages.begin(), messages.end());
                }
                std::sort(payloads.begin(), payloads.end());
                sendFrame(fd, frame_kind_t::NOTIFICATIONS_DRAINED, encodeNotifications(payloads));
                continue;
            }
            throw peer_fixture_error("peer worker received an invalid command");
        }
        if (admission_barrier != nullptr) {
            require(!admission_barrier->outstanding(),
                    "peer worker shut down with held admission-receipt authority");
            requireStatus(
                terminal_ucx_api_adapter_t::installAdmissionReceiptBarrier(agent, backend, nullptr),
                NIXL_SUCCESS,
                "remove peer admission-receipt hold");
        }
        requireStatus(
            agent.invalidateRemoteMD(remote), NIXL_SUCCESS, "retire peer worker source handle");
        return EXIT_SUCCESS;
    }

} // namespace

bool
isTerminalUcxPeerWorkerInvocation(int argc, char **argv) noexcept {
    return argc > 1 && std::string_view(argv[1]) == worker_argument;
}

int
runTerminalUcxPeerWorker(int argc, char **argv) {
    int fd = -1;
    try {
        std::string engine;
        std::optional<std::size_t> worker_count;
        fd = parseWorkerFd(argc, argv, engine, worker_count);
        return runWorker(fd, engine, worker_count);
    }
    catch (const std::exception &error) {
        if (fd >= 0) {
            try {
                sendFrame(fd, frame_kind_t::ERROR, error.what());
            }
            catch (const std::exception &) {
            }
        }
        std::cerr << "terminal UCX peer worker failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

tcp_endpoint_failure_observation_t
runTcpEndpointFailureFixture(const std::string &engine) {
    peer_process_t peer(engine);
    source_fixture_t source(engine, endpoint_failure_bytes);
    nixlRemoteAgentH *remote = loadPeer(source, peer);
    terminal_ucx_api_adapter_t adapter(source.agent, 8);
    terminal_inbox_t inbox(adapter);
    nixlTerminalEventSubscriptionH *capability_subscription = nullptr;
    tcp_peer_capability_observation_t capability =
        makeCapabilityReady(source, peer, adapter, inbox, remote, capability_subscription);

    nixlXferReqH *request =
        createTransfer(source, peer.hello(), remote, endpoint_failure_bytes, false);
    nixl_xfer_attestation_t prepared_attestation;
    requireStatus(source.agent.queryXferAttestation(request, prepared_attestation),
                  NIXL_SUCCESS,
                  "query prepared endpoint-failure transfer");
    nixlTerminalEventSubscriptionH *transfer_subscription = nullptr;
    requireStatus(adapter.subscribeTransfer(request, transfer_cookie, transfer_subscription),
                  NIXL_SUCCESS,
                  "subscribe endpoint-failure transfer");
    require(transfer_subscription != nullptr, "endpoint-failure transfer subscription is null");
    nixl_terminal_subscription_info_t transfer_info;
    requireStatus(adapter.querySubscription(transfer_subscription, transfer_info),
                  NIXL_SUCCESS,
                  "query endpoint-failure transfer subscription");
    require(transfer_info.active && transfer_info.identity == prepared_attestation.handleIdentity &&
                transfer_info.generation == prepared_attestation.generation + 1,
            "endpoint-failure subscription changed transfer generation");
    peer.armExitAfterWrite(0, source.memory->byteAt(0));
    const nixl_status_t post_status = source.agent.postXferReq(request);
    require(post_status == NIXL_IN_PROG, "endpoint-failure transfer completed before peer death");
    const bool peer_exited_by_signal = peer.waitForExitSignal(SIGKILL);
    require(peer_exited_by_signal, "TCP endpoint peer did not exit by SIGKILL");

    const observed_event_t transfer =
        inbox.take(nixl_terminal_event_kind_t::TRANSFER, transfer_cookie);
    const observed_event_t failed =
        inbox.take(nixl_terminal_event_kind_t::CAPABILITY, capability_cookie);
    requireStatus(transfer.event.transferStatus,
                  NIXL_ERR_REMOTE_DISCONNECT,
                  "endpoint-failure terminal status");
    require(transfer.event.identity == transfer_info.identity &&
                transfer.event.generation == transfer_info.generation,
            "endpoint failure changed transfer generation");
    require(transfer.ownerWoken, "endpoint failure did not wake the owner");
    nixl_xfer_attestation_t failed_attestation;
    requireStatus(source.agent.queryXferAttestation(request, failed_attestation),
                  NIXL_SUCCESS,
                  "query failed endpoint transfer");
    require(failed_attestation.state == nixl_xfer_attestation_state_t::FAILED &&
                failed_attestation.status == NIXL_ERR_REMOTE_DISCONNECT,
            "endpoint failure did not seal failed attestation");
    require(failed.event.identity == capability.handleIdentity &&
                failed.event.generation == capability.handleGeneration &&
                failed.event.capabilityState == nixl_terminal_capability_state_t::FAILED &&
                failed.event.capabilityEpoch >= capability.epochs.back(),
            "endpoint failure changed capability route semantics");
    require(failed.ownerWoken, "failed capability did not wake the owner");
    capability.states.push_back(failed.event.capabilityState);
    capability.epochs.push_back(failed.event.capabilityEpoch);
    nixl_terminal_subscription_info_t capability_info;
    requireStatus(adapter.querySubscription(capability_subscription, capability_info),
                  NIXL_SUCCESS,
                  "query failed peer capability");
    capability.subscriptionTerminal = !capability_info.active;
    capability.releaseStatus = adapter.release(capability_subscription);
    requireStatus(capability.releaseStatus, NIXL_SUCCESS, "release endpoint-failure capability");
    nixl_terminal_subscription_info_t terminal_transfer_info;
    requireStatus(adapter.querySubscription(transfer_subscription, terminal_transfer_info),
                  NIXL_SUCCESS,
                  "query terminal endpoint-failure transfer");
    require(!terminal_transfer_info.active, "endpoint-failure transfer subscription stayed active");
    requireStatus(adapter.release(transfer_subscription),
                  NIXL_SUCCESS,
                  "release endpoint-failure transfer subscription");
    requireStatus(
        source.agent.releaseXferReq(request), NIXL_SUCCESS, "release endpoint-failure request");
    requireStatus(
        source.agent.invalidateRemoteMD(remote), NIXL_SUCCESS, "invalidate failed TCP peer route");
    require(inbox.empty(), "endpoint-failure fixture retained a drained event");
    const tcp_peer_channel_observation_t channel = closeAndObserveChannel(adapter);

    return {
        .transfer =
            {
                .terminalStatus = transfer.event.transferStatus,
                .terminalEventCount = 1,
                .ownerWoken = transfer.ownerWoken,
            },
        .capability = std::move(capability),
        .channel = channel,
        .peerExitedBySignal = peer_exited_by_signal,
    };
}

tcp_notification_failure_observation_t
runTcpNotificationFailureFixture(const std::string &engine) {
    const bool composite = engine == "thread_pool";
    peer_process_t peer(engine);
    source_fixture_t source(engine, notification_failure_bytes);
    nixlRemoteAgentH *remote = loadPeer(source, peer);
    terminal_ucx_api_adapter_t adapter(source.agent, 8);
    terminal_inbox_t inbox(adapter);
    nixlTerminalEventSubscriptionH *capability_subscription = nullptr;
    static_cast<void>(
        makeCapabilityReady(source, peer, adapter, inbox, remote, capability_subscription));
    requireStatus(adapter.release(capability_subscription),
                  NIXL_SUCCESS,
                  "release notification-failure capability");
    peer.armAdmissionReceiptHold();

    const std::size_t descriptor_count = composite ? 2U : 1U;
    nixlXferReqH *request = createTransfer(
        source, peer.hello(), remote, notification_failure_bytes, true, descriptor_count);
    nixl_xfer_attestation_t prepared_attestation;
    requireStatus(source.agent.queryXferAttestation(request, prepared_attestation),
                  NIXL_SUCCESS,
                  "query prepared notification-failure transfer");
    require(prepared_attestation.segments.size() == descriptor_count,
            "notification-failure transfer did not preserve its descriptor geometry");
    nixlTerminalEventSubscriptionH *transfer_subscription = nullptr;
    requireStatus(adapter.subscribeTransfer(request, transfer_cookie, transfer_subscription),
                  NIXL_SUCCESS,
                  "subscribe notification-failure transfer");
    require(transfer_subscription != nullptr, "notification-failure transfer subscription is null");
    nixl_terminal_subscription_info_t transfer_info;
    requireStatus(adapter.querySubscription(transfer_subscription, transfer_info),
                  NIXL_SUCCESS,
                  "query notification-failure transfer subscription");
    require(transfer_info.active && transfer_info.identity == prepared_attestation.handleIdentity &&
                transfer_info.generation == prepared_attestation.generation + 1,
            "notification-failure subscription changed transfer generation");
    const nixl_status_t post_status = source.agent.postXferReq(request);
    require(post_status == NIXL_IN_PROG, "notification-failure transfer completed during post");

    const admission_receipt_authority_t held = peer.waitForAdmissionReceiptHeld();
    nixl_xfer_attestation_t pre_failure_attestation;
    requireStatus(source.agent.queryXferAttestation(request, pre_failure_attestation),
                  NIXL_SUCCESS,
                  "query held-receipt remote-flush boundary");
    require(attestationIsRemoteFlushed(pre_failure_attestation),
            "notification transfer lost remote-flush authority before failure");
    require(held.sourceHandleIdentity == pre_failure_attestation.handleIdentity &&
                held.sourceGeneration == pre_failure_attestation.generation &&
                held.deliveryIdentity != 0,
            "notification failure changed held delivery authority");
    nixl_terminal_subscription_info_t pre_failure_info;
    requireStatus(adapter.querySubscription(transfer_subscription, pre_failure_info),
                  NIXL_SUCCESS,
                  "query notification subscription at held-receipt boundary");
    require(pre_failure_info.active, "notification became terminal before the fault boundary");
    terminal_channel_inventory_t pre_failure_inventory;
    requireStatus(adapter.queryInventory(pre_failure_inventory),
                  NIXL_SUCCESS,
                  "query notification inventory at held-receipt boundary");
    require(pre_failure_inventory.backendProducers == 1,
            "notification failure boundary exposed no live delivery producer");

    const bool peer_exited_by_signal = peer.killAndWait();
    require(peer_exited_by_signal, "TCP notification peer did not exit by SIGKILL");
    const observed_event_t transfer =
        inbox.take(nixl_terminal_event_kind_t::TRANSFER, transfer_cookie);
    requireStatus(transfer.event.transferStatus,
                  NIXL_ERR_REMOTE_DISCONNECT,
                  "notification-failure terminal status");
    require(transfer.event.identity == transfer_info.identity &&
                transfer.event.generation == transfer_info.generation,
            "notification failure changed transfer generation");
    require(transfer.ownerWoken, "notification failure did not wake the owner");
    nixl_xfer_attestation_t failed_attestation;
    requireStatus(source.agent.queryXferAttestation(request, failed_attestation),
                  NIXL_SUCCESS,
                  "query failed notification transfer");
    require(failed_attestation.state == nixl_xfer_attestation_state_t::FAILED &&
                failed_attestation.status == NIXL_ERR_REMOTE_DISCONNECT,
            "notification failure did not seal failed attestation");
    const bool data_remote_flushed = !failed_attestation.endpoints.empty() &&
        std::all_of(failed_attestation.endpoints.begin(),
                    failed_attestation.endpoints.end(),
                    [](const auto &endpoint) { return endpoint.remoteFlushed; });
    require(data_remote_flushed,
            "notification failure discarded established remote-flush authority");
    const nixl_xfer_terminal_progress_t &progress = failed_attestation.terminalProgress;
    const bool notification_failure_after_remote_flush =
        progress.lastFlushCallbackTimestampNs > 0 &&
        progress.notificationCallbackTimestampNs > progress.lastFlushCallbackTimestampNs;
    require(notification_failure_after_remote_flush,
            "notification failure did not follow the final remote-flush callback");
    nixl_terminal_subscription_info_t terminal_transfer_info;
    requireStatus(adapter.querySubscription(transfer_subscription, terminal_transfer_info),
                  NIXL_SUCCESS,
                  "query terminal notification-failure transfer");
    require(!terminal_transfer_info.active,
            "notification-failure transfer subscription stayed active");
    requireStatus(adapter.release(transfer_subscription),
                  NIXL_SUCCESS,
                  "release notification-failure transfer subscription");
    requireStatus(
        source.agent.releaseXferReq(request), NIXL_SUCCESS, "release notification-failure request");
    requireStatus(source.agent.invalidateRemoteMD(remote),
                  NIXL_SUCCESS,
                  "invalidate failed notification peer route");
    require(inbox.empty(), "notification-failure fixture retained a drained event");
    const tcp_peer_channel_observation_t channel = closeAndObserveChannel(adapter);

    return {
        .transfer =
            {
                .terminalStatus = transfer.event.transferStatus,
                .terminalEventCount = 1,
                .ownerWoken = transfer.ownerWoken,
            },
        .channel = channel,
        .dataRemoteFlushedBeforeFailure = data_remote_flushed,
        .notificationFailureAfterRemoteFlush = notification_failure_after_remote_flush,
        .faultPeerEngine = engine,
        .faultPeerAdmissionReceiptHeld = true,
        .peerExitedBySignal = peer_exited_by_signal,
    };
}

tcp_shutdown_cancellation_observation_t
runTcpShutdownCancellationFixture(const std::string &engine) {
    peer_process_t peer(engine);
    source_fixture_t source(engine, endpoint_failure_bytes);
    nixlRemoteAgentH *remote = loadPeer(source, peer);
    terminal_ucx_api_adapter_t adapter(source.agent, 8);
    terminal_inbox_t inbox(adapter);
    nixlTerminalEventSubscriptionH *capability_subscription = nullptr;
    static_cast<void>(
        makeCapabilityReady(source, peer, adapter, inbox, remote, capability_subscription));
    requireStatus(adapter.release(capability_subscription),
                  NIXL_SUCCESS,
                  "release shutdown-cancellation capability");

    nixlXferReqH *request =
        createTransfer(source, peer.hello(), remote, endpoint_failure_bytes, false);
    nixlTerminalEventSubscriptionH *transfer_subscription = nullptr;
    requireStatus(adapter.subscribeTransfer(request, transfer_cookie, transfer_subscription),
                  NIXL_SUCCESS,
                  "subscribe shutdown-cancellation transfer");
    const nixl_status_t post_status = source.agent.postXferReq(request);
    require(post_status == NIXL_IN_PROG,
            "shutdown-cancellation transfer completed before cancellation");

    terminal_channel_inventory_t pre_cancel_inventory;
    requireStatus(adapter.queryInventory(pre_cancel_inventory),
                  NIXL_SUCCESS,
                  "query in-flight shutdown-cancellation inventory");
    require(pre_cancel_inventory.backendProducers > 0 ||
                pre_cancel_inventory.activeCallbackSlots > 0 ||
                pre_cancel_inventory.queuedOwnerContinuations > 0,
            "shutdown-cancellation transfer exposed no native in-flight inventory");

    nixl_status_t cancel_status = adapter.release(transfer_subscription);
    require(cancel_status == NIXL_SUCCESS || cancel_status == NIXL_IN_PROG,
            "cancel posted shutdown-cancellation transfer failed");
    const observed_event_t transfer =
        inbox.take(nixl_terminal_event_kind_t::TRANSFER, transfer_cookie);
    requireStatus(
        transfer.event.transferStatus, NIXL_ERR_CANCELED, "shutdown-cancellation terminal status");
    if (cancel_status == NIXL_IN_PROG) {
        cancel_status = adapter.release(transfer_subscription);
    }
    requireStatus(
        cancel_status, NIXL_SUCCESS, "release terminal shutdown-cancellation subscription");
    requireStatus(source.agent.releaseXferReq(request),
                  NIXL_SUCCESS,
                  "release shutdown-cancellation request");
    require(inbox.empty(), "shutdown-cancellation fixture retained a drained event");
    const tcp_peer_channel_observation_t post_close = closeAndObserveChannel(adapter);
    const bool drained = post_close.queuedChannelEvents == 0 &&
        post_close.activeChannelSubscriptions == 0 && post_close.retainedPublicSubscriptions == 0 &&
        post_close.backendProducers == 0 && post_close.activeCallbackSlots == 0 &&
        post_close.queuedOwnerContinuations == 0 && !post_close.acceptingSubscriptions &&
        post_close.closed && post_close.fatal == 0 && post_close.eventfdError == 0;
    requireStatus(source.agent.invalidateRemoteMD(remote),
                  NIXL_SUCCESS,
                  "invalidate shutdown-cancellation peer route");
    const bool peer_exited_by_signal = peer.killAndWait();

    return {
        .transfer =
            {
                .terminalStatus = transfer.event.transferStatus,
                .terminalEventCount = 1,
                .ownerWoken = transfer.ownerWoken,
            },
        .inventoryBeforeCancellation = observeChannel(pre_cancel_inventory),
        .inventoryAfterClose = post_close,
        .cancelStatus = cancel_status,
        .postedInFlight = true,
        .drained = drained,
        .peerExitedBySignal = peer_exited_by_signal,
    };
}

tcp_admission_receipt_observation_t
runTcpAdmissionReceiptReleaseFixture(const std::string &engine) {
    peer_process_t peer(engine);
    source_fixture_t source(engine, notification_failure_bytes);
    nixlRemoteAgentH *remote = loadPeer(source, peer);
    terminal_ucx_api_adapter_t adapter(source.agent, 8);
    terminal_inbox_t inbox(adapter);
    nixlTerminalEventSubscriptionH *capability_subscription = nullptr;
    static_cast<void>(
        makeCapabilityReady(source, peer, adapter, inbox, remote, capability_subscription));
    requireStatus(adapter.release(capability_subscription),
                  NIXL_SUCCESS,
                  "release admission-receipt capability");

    peer.armAdmissionReceiptHold();
    const std::string notification = "admission-receipt-release";
    nixlXferReqH *request = createTransfer(
        source, peer.hello(), remote, notification_failure_bytes, true, 1, 0, notification);
    nixlTerminalEventSubscriptionH *transfer_subscription = nullptr;
    requireStatus(adapter.subscribeTransfer(request, transfer_cookie, transfer_subscription),
                  NIXL_SUCCESS,
                  "subscribe held admission-receipt transfer");
    const nixl_status_t post_status = source.agent.postXferReq(request);
    require(post_status == NIXL_IN_PROG, "held admission-receipt transfer completed during post");

    const admission_receipt_authority_t held = peer.waitForAdmissionReceiptHeld();
    nixl_xfer_attestation_t held_attestation;
    requireStatus(source.agent.queryXferAttestation(request, held_attestation),
                  NIXL_SUCCESS,
                  "query held admission-receipt transfer");
    require(attestationIsRemoteFlushed(held_attestation),
            "held admission receipt lost remote-flush authority");
    require(held.sourceHandleIdentity == held_attestation.handleIdentity &&
                held.sourceGeneration == held_attestation.generation && held.deliveryIdentity != 0,
            "held admission receipt changed immutable delivery authority");
    nixl_terminal_subscription_info_t held_subscription;
    requireStatus(adapter.querySubscription(transfer_subscription, held_subscription),
                  NIXL_SUCCESS,
                  "query held admission-receipt subscription");
    require(held_subscription.active, "held admission receipt published source terminality early");

    peer.releaseAdmissionReceipt();
    const observed_event_t transfer =
        inbox.take(nixl_terminal_event_kind_t::TRANSFER, transfer_cookie);
    requireStatus(
        transfer.event.transferStatus, NIXL_SUCCESS, "released admission-receipt terminal status");
    require(transfer.ownerWoken, "released admission receipt did not wake the owner");
    const std::vector<std::string> notifications = peer.drainNotifications();
    require(notifications == std::vector<std::string>{notification},
            "released admission receipt exposed the wrong notification");

    requireStatus(adapter.release(transfer_subscription),
                  NIXL_SUCCESS,
                  "release admission-receipt subscription");
    requireStatus(
        source.agent.releaseXferReq(request), NIXL_SUCCESS, "release admission-receipt request");
    require(inbox.empty(), "admission-receipt release retained a drained event");
    requireStatus(source.agent.invalidateRemoteMD(remote),
                  NIXL_SUCCESS,
                  "invalidate admission-receipt peer route");
    const tcp_peer_channel_observation_t channel = closeAndObserveChannel(adapter);
    const bool peer_exited_cleanly = peer.shutdownAndWait();
    require(peer_exited_cleanly, "admission-receipt peer did not shut down cleanly");

    return {
        .transfer =
            {
                .terminalStatus = transfer.event.transferStatus,
                .terminalEventCount = 1,
                .ownerWoken = transfer.ownerWoken,
            },
        .channel = channel,
        .stateWhileHeld = held_attestation.state,
        .heldSourceHandleIdentity = held.sourceHandleIdentity,
        .heldSourceGeneration = held.sourceGeneration,
        .heldDeliveryIdentity = held.deliveryIdentity,
        .notificationCount = notifications.size(),
        .subscriptionActiveWhileHeld = held_subscription.active,
        .remoteFlushedWhileHeld = true,
        .peerExitedCleanly = peer_exited_cleanly,
    };
}

tcp_admission_receipt_observation_t
runTcpAdmissionReceiptPeerDeathFixture(const std::string &engine) {
    peer_process_t peer(engine);
    source_fixture_t source(engine, notification_failure_bytes);
    nixlRemoteAgentH *remote = loadPeer(source, peer);
    terminal_ucx_api_adapter_t adapter(source.agent, 8);
    terminal_inbox_t inbox(adapter);
    nixlTerminalEventSubscriptionH *capability_subscription = nullptr;
    static_cast<void>(
        makeCapabilityReady(source, peer, adapter, inbox, remote, capability_subscription));
    requireStatus(adapter.release(capability_subscription),
                  NIXL_SUCCESS,
                  "release admission-death capability");

    peer.armAdmissionReceiptHold();
    nixlXferReqH *request = createTransfer(source,
                                           peer.hello(),
                                           remote,
                                           notification_failure_bytes,
                                           true,
                                           1,
                                           0,
                                           "admission-receipt-peer-death");
    nixlTerminalEventSubscriptionH *transfer_subscription = nullptr;
    requireStatus(adapter.subscribeTransfer(request, transfer_cookie, transfer_subscription),
                  NIXL_SUCCESS,
                  "subscribe admission-death transfer");
    const nixl_status_t post_status = source.agent.postXferReq(request);
    require(post_status == NIXL_IN_PROG, "admission-death transfer completed during post");

    const admission_receipt_authority_t held = peer.waitForAdmissionReceiptHeld();
    nixl_xfer_attestation_t held_attestation;
    requireStatus(source.agent.queryXferAttestation(request, held_attestation),
                  NIXL_SUCCESS,
                  "query held admission-death transfer");
    require(attestationIsRemoteFlushed(held_attestation),
            "admission-death transfer lost remote-flush authority");
    require(held.sourceHandleIdentity == held_attestation.handleIdentity &&
                held.sourceGeneration == held_attestation.generation && held.deliveryIdentity != 0,
            "admission-death hold changed immutable delivery authority");
    nixl_terminal_subscription_info_t held_subscription;
    requireStatus(adapter.querySubscription(transfer_subscription, held_subscription),
                  NIXL_SUCCESS,
                  "query held admission-death subscription");
    require(held_subscription.active,
            "admission-death transfer published source terminality early");

    const bool peer_exited_by_signal = peer.killAndWait();
    require(peer_exited_by_signal, "held admission peer did not exit by SIGKILL");
    const observed_event_t transfer =
        inbox.take(nixl_terminal_event_kind_t::TRANSFER, transfer_cookie);
    requireStatus(transfer.event.transferStatus,
                  NIXL_ERR_REMOTE_DISCONNECT,
                  "held admission peer-death terminal status");
    require(transfer.ownerWoken, "held admission peer death did not wake the owner");

    requireStatus(adapter.release(transfer_subscription),
                  NIXL_SUCCESS,
                  "release admission-death subscription");
    requireStatus(
        source.agent.releaseXferReq(request), NIXL_SUCCESS, "release admission-death request");
    requireStatus(source.agent.invalidateRemoteMD(remote),
                  NIXL_SUCCESS,
                  "invalidate admission-death peer route");
    require(inbox.empty(), "admission-death fixture retained a drained event");
    const tcp_peer_channel_observation_t channel = closeAndObserveChannel(adapter);

    return {
        .transfer =
            {
                .terminalStatus = transfer.event.transferStatus,
                .terminalEventCount = 1,
                .ownerWoken = transfer.ownerWoken,
            },
        .channel = channel,
        .stateWhileHeld = held_attestation.state,
        .heldSourceHandleIdentity = held.sourceHandleIdentity,
        .heldSourceGeneration = held.sourceGeneration,
        .heldDeliveryIdentity = held.deliveryIdentity,
        .subscriptionActiveWhileHeld = held_subscription.active,
        .remoteFlushedWhileHeld = true,
        .peerExitedBySignal = peer_exited_by_signal,
    };
}

tcp_unequal_worker_observation_t
runTcpUnequalWorkerFixture(std::size_t source_worker_count, std::size_t destination_worker_count) {
    require(source_worker_count > 0 && destination_worker_count > 0,
            "unequal-worker fixture requires positive worker counts");
    require(source_worker_count != destination_worker_count,
            "unequal-worker fixture requires asymmetric worker counts");

    peer_process_t peer("shared", destination_worker_count);
    source_fixture_t source("shared", notification_failure_bytes, source_worker_count);
    nixlRemoteAgentH *remote = loadPeer(source, peer);
    terminal_ucx_api_adapter_t adapter(source.agent, source_worker_count + 4);
    terminal_inbox_t inbox(adapter);
    nixlTerminalEventSubscriptionH *capability_subscription = nullptr;
    static_cast<void>(
        makeCapabilityReady(source, peer, adapter, inbox, remote, capability_subscription));
    requireStatus(adapter.release(capability_subscription),
                  NIXL_SUCCESS,
                  "release unequal-worker capability");

    std::vector<std::string> expected_notifications;
    std::vector<std::size_t> exercised_workers;
    expected_notifications.reserve(source_worker_count);
    exercised_workers.reserve(source_worker_count);
    for (std::size_t worker_id = 0; worker_id < source_worker_count; ++worker_id) {
        const std::string notification = "worker-" + std::to_string(worker_id);
        nixlXferReqH *request = createTransfer(source,
                                               peer.hello(),
                                               remote,
                                               notification_failure_bytes,
                                               true,
                                               1,
                                               worker_id,
                                               notification);
        nixlTerminalEventSubscriptionH *subscription = nullptr;
        requireStatus(adapter.subscribeTransfer(
                          request, unequal_worker_cookie_base + worker_id, subscription),
                      NIXL_SUCCESS,
                      "subscribe unequal-worker transfer");
        const nixl_status_t post_status = source.agent.postXferReq(request);
        require(post_status == NIXL_SUCCESS || post_status == NIXL_IN_PROG,
                "post unequal-worker transfer failed");
        const observed_event_t transfer = inbox.take(nixl_terminal_event_kind_t::TRANSFER,
                                                     unequal_worker_cookie_base + worker_id);
        requireStatus(
            transfer.event.transferStatus, NIXL_SUCCESS, "unequal-worker terminal status");
        require(transfer.ownerWoken, "unequal-worker completion did not wake the owner");
        requireStatus(
            adapter.release(subscription), NIXL_SUCCESS, "release unequal-worker subscription");
        requireStatus(
            source.agent.releaseXferReq(request), NIXL_SUCCESS, "release unequal-worker request");
        expected_notifications.push_back(notification);
        exercised_workers.push_back(worker_id);
    }

    std::sort(expected_notifications.begin(), expected_notifications.end());
    const std::vector<std::string> notifications = peer.drainNotifications();
    require(notifications == expected_notifications,
            "unequal-worker peer observed the wrong authenticated notification set");
    require(inbox.empty(), "unequal-worker fixture retained a drained event");
    requireStatus(source.agent.invalidateRemoteMD(remote),
                  NIXL_SUCCESS,
                  "invalidate unequal-worker peer route");
    const tcp_peer_channel_observation_t channel = closeAndObserveChannel(adapter);
    const bool peer_exited_cleanly = peer.shutdownAndWait();
    require(peer_exited_cleanly, "unequal-worker peer did not shut down cleanly");

    return {
        .channel = channel,
        .sourceWorkerCount = source_worker_count,
        .destinationWorkerCount = destination_worker_count,
        .exercisedSourceWorkers = std::move(exercised_workers),
        .completedTransferCount = source_worker_count,
        .notificationCount = notifications.size(),
        .peerExitedCleanly = peer_exited_cleanly,
    };
}

} // namespace nixl::qualification
