/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "terminal_event_channel.h"

#include <cerrno>
#include <ctime>
#include <stdexcept>
#include <system_error>
#include <utility>

#include <sys/eventfd.h>
#include <unistd.h>

namespace nixl {

namespace {

    [[nodiscard]] int
    createEventFd(std::size_t capacity) {
        if (capacity == 0) {
            throw std::invalid_argument("terminal event channel capacity must be positive");
        }

        const int event_fd = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
        if (event_fd < 0) {
            throw std::system_error(errno, std::generic_category(), "failed to create eventfd");
        }
        return event_fd;
    }

    [[nodiscard]] std::uint64_t
    monotonicRawTimestampNs() noexcept {
        timespec timestamp{};
        if (::clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp) != 0) {
            return 0;
        }

        return static_cast<std::uint64_t>(timestamp.tv_sec) * 1'000'000'000ULL +
            static_cast<std::uint64_t>(timestamp.tv_nsec);
    }

    [[nodiscard]] bool
    isTerminalTransferStatus(nixl_status_t status) noexcept {
        return status != NIXL_IN_PROG && status != NIXL_ERR_NOT_POSTED &&
            status != NIXL_ERR_NOT_READY;
    }

} // namespace

class terminalEventChannelState {
public:
    explicit terminalEventChannelState(std::size_t capacity)
        : capacity_(capacity),
          eventFd_(createEventFd(capacity)),
          events_(capacity) {}

    terminalEventChannelState(const terminalEventChannelState &) = delete;
    terminalEventChannelState &
    operator=(const terminalEventChannelState &) = delete;

    ~terminalEventChannelState() {
        if (eventFd_ >= 0) {
            static_cast<void>(::close(eventFd_));
        }
    }

    [[nodiscard]] bool
    acquireSubscription() noexcept {
        const std::lock_guard lock(mutex_);
        if (!acceptingSubscriptions_ || closed_ || fatal_ != terminal_channel_fatal_t::NONE) {
            return false;
        }

        ++activeSubscriptions_;
        return true;
    }

    void
    releaseSubscription() noexcept {
        const std::lock_guard lock(mutex_);
        if (activeSubscriptions_ == 0) {
            return;
        }

        --activeSubscriptions_;
        if (closeRequested_ && activeSubscriptions_ == 0) {
            closed_ = true;
        }
    }

    [[nodiscard]] terminal_event_publish_result_t
    publish(terminal_event_t event) noexcept {
        const std::lock_guard lock(mutex_);
        if (fatal_ != terminal_channel_fatal_t::NONE) {
            return terminal_event_publish_result_t::CHANNEL_FATAL;
        }
        if (closeRequested_ || closed_) {
            return terminal_event_publish_result_t::CHANNEL_CLOSED;
        }
        if (queuedEvents_ == capacity_) {
            setFatalLocked(terminal_channel_fatal_t::QUEUE_OVERFLOW, 0);
            static_cast<void>(signalLocked());
            return terminal_event_publish_result_t::CHANNEL_FATAL;
        }

        const std::size_t tail = (head_ + queuedEvents_) % capacity_;
        events_[tail] = std::move(event);
        ++queuedEvents_;
        if (!signalLocked()) {
            return terminal_event_publish_result_t::CHANNEL_FATAL;
        }
        return terminal_event_publish_result_t::PUBLISHED;
    }

    void
    fail(terminal_channel_fatal_t fatal) noexcept {
        const std::lock_guard lock(mutex_);
        setFatalLocked(fatal, 0);
        static_cast<void>(signalLocked());
    }

    [[nodiscard]] int
    fileno() const noexcept {
        return eventFd_;
    }

    [[nodiscard]] std::optional<terminal_event_t>
    take() noexcept {
        const std::lock_guard lock(mutex_);
        static_cast<void>(consumeWakeLocked());
        if (queuedEvents_ == 0) {
            if (fatal_ != terminal_channel_fatal_t::NONE) {
                static_cast<void>(signalLocked());
            }
            return std::nullopt;
        }

        terminal_event_t event = std::move(events_[head_]);
        head_ = (head_ + 1) % capacity_;
        --queuedEvents_;
        if (queuedEvents_ != 0 || fatal_ != terminal_channel_fatal_t::NONE) {
            static_cast<void>(signalLocked());
        }
        return event;
    }

    [[nodiscard]] terminal_event_batch_t
    drain() {
        const std::lock_guard lock(mutex_);
        terminal_event_batch_t batch;
        batch.events.reserve(queuedEvents_);
        // Allocation happens before consuming the wake so an allocation failure leaves the fd
        // readable and cannot strand queued terminal authority.
        batch.wakeCount = consumeWakeLocked();
        while (queuedEvents_ != 0) {
            batch.events.push_back(std::move(events_[head_]));
            head_ = (head_ + 1) % capacity_;
            --queuedEvents_;
        }
        batch.inventory = makeInventoryLocked();
        return batch;
    }

    [[nodiscard]] terminal_channel_inventory_t
    inventory() const noexcept {
        const std::lock_guard lock(mutex_);
        return makeInventoryLocked();
    }

    [[nodiscard]] terminal_channel_close_result_t
    close() noexcept {
        const std::lock_guard lock(mutex_);
        if (closed_) {
            return terminal_channel_close_result_t::ALREADY_CLOSED;
        }
        if (closeRequested_) {
            return activeSubscriptions_ == 0 ?
                terminal_channel_close_result_t::CLOSED :
                terminal_channel_close_result_t::ACTIVE_SUBSCRIPTIONS;
        }

        acceptingSubscriptions_ = false;
        closeRequested_ = true;
        if (activeSubscriptions_ != 0) {
            setFatalLocked(terminal_channel_fatal_t::ACTIVE_SUBSCRIPTIONS_ON_CLOSE, 0);
            static_cast<void>(signalLocked());
            return terminal_channel_close_result_t::ACTIVE_SUBSCRIPTIONS;
        }

        closed_ = true;
        return terminal_channel_close_result_t::CLOSED;
    }

private:
    [[nodiscard]] bool
    signalLocked() noexcept {
        constexpr std::uint64_t signal = 1;
        while (true) {
            const ssize_t written = ::write(eventFd_, &signal, sizeof(signal));
            if (written == static_cast<ssize_t>(sizeof(signal))) {
                return true;
            }
            if (written < 0 && errno == EINTR) {
                continue;
            }

            const int error = written < 0 ? errno : EIO;
            setFatalLocked(terminal_channel_fatal_t::EVENTFD_FAILURE, error);
            return false;
        }
    }

    [[nodiscard]] std::uint64_t
    consumeWakeLocked() noexcept {
        std::uint64_t wake_count = 0;
        while (true) {
            const ssize_t read_size = ::read(eventFd_, &wake_count, sizeof(wake_count));
            if (read_size == static_cast<ssize_t>(sizeof(wake_count))) {
                return wake_count;
            }
            if (read_size < 0 && errno == EINTR) {
                continue;
            }
            if (read_size < 0 && errno == EAGAIN) {
                return 0;
            }

            const int error = read_size < 0 ? errno : EIO;
            setFatalLocked(terminal_channel_fatal_t::EVENTFD_FAILURE, error);
            return 0;
        }
    }

    void
    setFatalLocked(terminal_channel_fatal_t fatal, int eventfd_error) noexcept {
        fatal_ = fatal_ | fatal;
        acceptingSubscriptions_ = false;
        if (eventfd_error != 0 && eventfdError_ == 0) {
            eventfdError_ = eventfd_error;
        }
    }

    [[nodiscard]] terminal_channel_inventory_t
    makeInventoryLocked() const noexcept {
        return {
            .capacity = capacity_,
            .queuedEvents = queuedEvents_,
            .activeSubscriptions = activeSubscriptions_,
            .acceptingSubscriptions = acceptingSubscriptions_,
            .closed = closed_,
            .health = {.fatal = fatal_, .eventfdError = eventfdError_},
        };
    }

    const std::size_t capacity_;
    const int eventFd_;
    mutable std::mutex mutex_;
    std::vector<terminal_event_t> events_;
    std::size_t head_ = 0;
    std::size_t queuedEvents_ = 0;
    std::size_t activeSubscriptions_ = 0;
    bool acceptingSubscriptions_ = true;
    bool closeRequested_ = false;
    bool closed_ = false;
    terminal_channel_fatal_t fatal_ = terminal_channel_fatal_t::NONE;
    int eventfdError_ = 0;
};

terminalEventChannel::subscription::subscription(std::shared_ptr<terminalEventChannelState> state,
                                                 terminal_event_binding_t binding)
    : state_(std::move(state)),
      binding_(binding) {}

terminalEventChannel::subscription::~subscription() {
    release();
}

terminal_event_publish_result_t
terminalEventChannel::subscription::publishTransfer(nixl_status_t status,
                                                    std::uint64_t native_timestamp_ns) noexcept {
    const std::lock_guard lock(mutex_);
    if (released_) {
        return terminal_event_publish_result_t::CHANNEL_CLOSED;
    }
    if (binding_.kind != terminal_event_kind_t::TRANSFER || !isTerminalTransferStatus(status)) {
        return terminal_event_publish_result_t::INVALID_EVENT;
    }
    if (transferPublished_) {
        return terminal_event_publish_result_t::ALREADY_PUBLISHED;
    }

    transferPublished_ = true;
    const std::uint64_t timestamp =
        native_timestamp_ns == 0 ? monotonicRawTimestampNs() : native_timestamp_ns;
    return state_->publish({
        .kind = binding_.kind,
        .ownerCookie = binding_.ownerCookie,
        .identity = binding_.identity,
        .generation = binding_.generation,
        .result = status,
        .epoch = 0,
        .nativeTimestampNs = timestamp,
    });
}

terminal_event_publish_result_t
terminalEventChannel::subscription::publishCapability(terminal_capability_state_t capability_state,
                                                      std::uint64_t epoch,
                                                      std::uint64_t native_timestamp_ns) noexcept {
    const std::lock_guard lock(mutex_);
    if (released_) {
        return terminal_event_publish_result_t::CHANNEL_CLOSED;
    }
    if (binding_.kind != terminal_event_kind_t::CAPABILITY || epoch == 0) {
        return terminal_event_publish_result_t::INVALID_EVENT;
    }

    const std::uint64_t timestamp =
        native_timestamp_ns == 0 ? monotonicRawTimestampNs() : native_timestamp_ns;
    return state_->publish({
        .kind = binding_.kind,
        .ownerCookie = binding_.ownerCookie,
        .identity = binding_.identity,
        .generation = binding_.generation,
        .result = capability_state,
        .epoch = epoch,
        .nativeTimestampNs = timestamp,
    });
}

void
terminalEventChannel::subscription::release() noexcept {
    std::shared_ptr<terminalEventChannelState> state;
    {
        const std::lock_guard lock(mutex_);
        if (released_) {
            return;
        }

        released_ = true;
        state = std::move(state_);
    }
    state->releaseSubscription();
}

void
terminalEventChannel::subscription::failInvalidPublication() noexcept {
    const std::lock_guard lock(mutex_);
    if (released_) {
        return;
    }
    state_->fail(terminal_channel_fatal_t::INVALID_PUBLICATION);
}

terminalEventChannel::terminalEventChannel(std::size_t capacity)
    : state_(std::make_shared<terminalEventChannelState>(capacity)) {}

terminalEventChannel::~terminalEventChannel() {
    static_cast<void>(close());
}

std::shared_ptr<terminalEventChannel::subscription>
terminalEventChannel::subscribe(const terminal_event_binding_t &binding) {
    if (binding.ownerCookie == 0 || binding.identity == 0 || binding.generation == 0) {
        throw std::invalid_argument(
            "terminal event subscription requires exact nonzero identity fields");
    }
    if (!state_->acquireSubscription()) {
        return nullptr;
    }

    try {
        return std::shared_ptr<subscription>(new subscription(state_, binding));
    }
    catch (...) {
        state_->releaseSubscription();
        throw;
    }
}

int
terminalEventChannel::fileno() const noexcept {
    return state_->fileno();
}

std::optional<terminal_event_t>
terminalEventChannel::take() noexcept {
    return state_->take();
}

terminal_event_batch_t
terminalEventChannel::drain() {
    return state_->drain();
}

terminal_channel_inventory_t
terminalEventChannel::inventory() const noexcept {
    return state_->inventory();
}

terminal_channel_close_result_t
terminalEventChannel::close() noexcept {
    return state_->close();
}

} // namespace nixl
