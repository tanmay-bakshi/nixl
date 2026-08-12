/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "terminal_ucx_peer_fixture.h"

#include "terminal_ucx_api_adapter.h"

#include <poll.h>
#include <signal.h>
#include <spawn.h>
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
    constexpr std::uint64_t device_id = 0;

    enum class frame_kind_t : std::uint64_t {
        HELLO = 1,
        LOAD_REMOTE_METADATA = 2,
        REMOTE_METADATA_LOADED = 3,
        SHUTDOWN = 4,
        ERROR = 5,
        ARM_STOP_AFTER_WRITE = 6,
        STOP_WATCH_ARMED = 7,
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
    encodeStopWatch(std::size_t offset, std::uint8_t expected) {
        const std::array<std::uint64_t, 2> values = {
            static_cast<std::uint64_t>(offset),
            static_cast<std::uint64_t>(expected),
        };
        return std::string(reinterpret_cast<const char *>(values.data()), sizeof(values));
    }

    [[nodiscard]] std::pair<std::size_t, std::uint8_t>
    decodeStopWatch(std::string_view payload) {
        require(payload.size() == sizeof(std::uint64_t) * 2,
                "peer stop-watch frame has an invalid length");
        std::array<std::uint64_t, 2> values = {};
        std::memcpy(values.data(), payload.data(), payload.size());
        require(values[0] <= std::numeric_limits<std::size_t>::max() &&
                    values[1] <= std::numeric_limits<std::uint8_t>::max(),
                "peer stop-watch frame is invalid");
        return {
            static_cast<std::size_t>(values[0]),
            static_cast<std::uint8_t>(values[1]),
        };
    }

    [[nodiscard]] nixlAgentConfig
    agentConfig() {
        nixlAgentConfig config;
        config.useProgThread = true;
        config.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
        return config;
    }

    [[nodiscard]] nixl_b_params_t
    backendParameters(const std::string &engine) {
        require(engine == "shared" || engine == "thread_pool", "peer fixture engine is invalid");
        nixl_b_params_t parameters;
        parameters["ucx_error_handling_mode"] = "peer";
        parameters["split_batch_size"] = "2";
        if (engine == "shared") {
            parameters["num_workers"] = "2";
            parameters["num_threads"] = "0";
            return parameters;
        }
        parameters["num_workers"] = "3";
        parameters["num_threads"] = "2";
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
        stopProcessAfterByte(std::size_t offset, std::uint8_t expected) const {
            require(offset < bytes_.size(), "peer stop-watch offset is out of range");
            const volatile std::uint8_t *const observed = bytes_.data() + offset;
            while (*observed != expected) {
                std::this_thread::yield();
            }
            if (::raise(SIGSTOP) != 0) {
                throw peer_fixture_error("peer worker failed to stop at the data boundary");
            }
        }

        [[nodiscard]] nixl_xfer_dlist_t
        transferList(std::size_t size) const {
            require(size > 0 && size <= bytes_.size(), "peer transfer exceeds registered DRAM");
            nixl_xfer_dlist_t descriptors(DRAM_SEG);
            descriptors.addDesc(nixlBasicDesc(address(), size, device_id));
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

    class peer_process_t final {
    public:
        explicit peer_process_t(const std::string &engine) {
            const std::filesystem::path executable = std::filesystem::canonical("/proc/self/exe");
            const std::string executable_string = executable.string();
            const std::string fd_string = std::to_string(worker_socket_fd);
            std::array<char *, 7> arguments = {
                const_cast<char *>(executable_string.c_str()),
                const_cast<char *>(worker_argument.data()),
                const_cast<char *>("--fd"),
                const_cast<char *>(fd_string.c_str()),
                const_cast<char *>("--engine"),
                const_cast<char *>(engine.c_str()),
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
        armStopAfterWrite(std::size_t offset, std::uint8_t expected) {
            require(!stopped_, "peer worker is already stopped");
            sendFrame(socket_, frame_kind_t::ARM_STOP_AFTER_WRITE, encodeStopWatch(offset, expected));
            static_cast<void>(expect(frame_kind_t::STOP_WATCH_ARMED));
        }

        void
        waitUntilStopped() {
            require(pid_ > 0 && !waited_ && !stopped_, "peer worker cannot enter stop boundary");
            int status = 0;
            while (::waitpid(pid_, &status, WUNTRACED) < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw peer_fixture_error("failed to observe stopped TCP peer worker");
            }
            require(WIFSTOPPED(status) && WSTOPSIG(status) == SIGSTOP,
                    "TCP peer did not stop at the data boundary");
            stopped_ = true;
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
            stopped_ = false;
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
        bool stopped_ = false;
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
        source_fixture_t(const std::string &engine, std::size_t bytes)
            : agent("qualification-peer-source-" +
                        std::to_string(static_cast<unsigned long long>(::getpid())),
                    agentConfig()) {
            requireStatus(agent.createBackend("UCX", backendParameters(engine), backend),
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
    remoteTransferList(const peer_hello_t &hello, std::size_t size) {
        require(size > 0 && size <= hello.capacity, "fault transfer exceeds peer registration");
        nixl_xfer_dlist_t descriptors(DRAM_SEG);
        descriptors.addDesc(nixlBasicDesc(hello.address, size, device_id));
        return descriptors;
    }

    [[nodiscard]] nixlXferReqH *
    createTransfer(source_fixture_t &source,
                   const peer_hello_t &peer,
                   const nixlRemoteAgentH *remote,
                   std::size_t bytes,
                   bool attached_notification) {
        nixl_opt_args_t arguments;
        arguments.backends = {source.backend};
        if (attached_notification) {
            arguments.notif = std::string(notification_bytes, 'N');
        }
        nixlXferReqH *request = nullptr;
        requireStatus(source.agent.createXferReq(NIXL_WRITE,
                                                 source.memory->transferList(bytes),
                                                 remoteTransferList(peer, bytes),
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
    parseWorkerFd(int argc, char **argv, std::string &engine) {
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
            throw peer_fixture_error("unknown peer worker argument");
        }
        require(fd >= 0, "peer worker fd is missing");
        require(engine == "shared" || engine == "thread_pool", "peer worker engine is invalid");
        return fd;
    }

    int
    runWorker(int fd, const std::string &engine) {
        nixlAgent agent("qualification-peer-destination-" +
                            std::to_string(static_cast<unsigned long long>(::getpid())),
                        agentConfig());
        nixlBackendH *backend = nullptr;
        requireStatus(agent.createBackend("UCX", backendParameters(engine), backend),
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

        frame = receiveFrame(fd);
        if (frame.kind == frame_kind_t::ARM_STOP_AFTER_WRITE) {
            const auto [offset, expected] = decodeStopWatch(frame.payload);
            require(offset < memory.capacity(), "peer stop-watch exceeds registered DRAM");
            sendFrame(fd, frame_kind_t::STOP_WATCH_ARMED);
            memory.stopProcessAfterByte(offset, expected);
            throw peer_fixture_error("peer stop-watch resumed without termination");
        }
        require(frame.kind == frame_kind_t::SHUTDOWN, "peer worker received an invalid command");
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
        fd = parseWorkerFd(argc, argv, engine);
        return runWorker(fd, engine);
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
    const nixl_status_t post_status = source.agent.postXferReq(request);
    require(post_status == NIXL_IN_PROG, "endpoint-failure transfer completed before peer death");
    const bool peer_exited_by_signal = peer.killAndWait();
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

    nixlXferReqH *request =
        createTransfer(source, peer.hello(), remote, notification_failure_bytes, true);
    nixl_xfer_attestation_t prepared_attestation;
    requireStatus(source.agent.queryXferAttestation(request, prepared_attestation),
                  NIXL_SUCCESS,
                  "query prepared notification-failure transfer");
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
    const std::size_t terminal_byte_offset = notification_failure_bytes - 1;
    peer.armStopAfterWrite(terminal_byte_offset, source.memory->byteAt(terminal_byte_offset));
    const nixl_status_t post_status = source.agent.postXferReq(request);
    require(post_status == NIXL_IN_PROG, "notification-failure transfer completed during post");
    peer.waitUntilStopped();

    const std::uint64_t deadline =
        monotonicRawNs() + static_cast<std::uint64_t>(event_timeout_ms) * 1000000ULL;
    bool remote_flushed = false;
    while (monotonicRawNs() < deadline) {
        nixl_xfer_attestation_t attestation;
        requireStatus(source.agent.queryXferAttestation(request, attestation),
                      NIXL_SUCCESS,
                      "query pre-notification remote flush");
        if (attestationIsRemoteFlushed(attestation)) {
            remote_flushed = true;
            break;
        }
        std::this_thread::yield();
    }
    require(remote_flushed, "notification-failure transfer never exposed remote flush");
    nixl_terminal_subscription_info_t pre_failure_info;
    requireStatus(adapter.querySubscription(transfer_subscription, pre_failure_info),
                  NIXL_SUCCESS,
                  "query notification transfer at the stopped-peer boundary");
    require(pre_failure_info.active,
            "notification completed before the stopped-peer failure boundary");
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
        .dataRemoteFlushedBeforeFailure =
            remote_flushed && transfer.event.transferStatus == NIXL_ERR_REMOTE_DISCONNECT,
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

    const nixl_status_t cancel_status = adapter.release(transfer_subscription);
    requireStatus(cancel_status, NIXL_SUCCESS, "cancel posted shutdown-cancellation transfer");
    const observed_event_t transfer =
        inbox.take(nixl_terminal_event_kind_t::TRANSFER, transfer_cookie);
    requireStatus(
        transfer.event.transferStatus, NIXL_ERR_CANCELED, "shutdown-cancellation terminal status");
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

} // namespace nixl::qualification
