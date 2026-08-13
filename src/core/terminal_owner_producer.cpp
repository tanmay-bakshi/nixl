/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "terminal_owner_producer.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <time.h>
#include <unordered_map>
#include <utility>

namespace {

constexpr std::uint16_t source_native_terminal_event = 13;
constexpr std::uint16_t source_request_failed_event = 21;

enum class binding_phase_t : std::uint8_t {
    REGISTERING = 1,
    SUBMITTED = 2,
    DELIVERING = 3,
};

struct binding_digest_hash_t {
    std::size_t
    operator()(const nixl_terminal_owner_binding_digest_t &value) const noexcept {
        std::size_t hash = 1469598103934665603ULL;
        for (const std::uint8_t byte : value) {
            hash ^= byte;
            hash *= 1099511628211ULL;
        }
        return hash;
    }
};

[[nodiscard]] bool
isZeroBinding(const nixl_terminal_owner_binding_digest_t &binding) noexcept {
    return std::all_of(binding.begin(), binding.end(), [](std::uint8_t byte) { return byte == 0; });
}

[[nodiscard]] std::uint64_t
monotonicRawNs() noexcept {
    timespec value{};
    if (::clock_gettime(CLOCK_MONOTONIC_RAW, &value) != 0) {
        return 1;
    }
    return static_cast<std::uint64_t>(value.tv_sec) * 1'000'000'000ULL +
        static_cast<std::uint64_t>(value.tv_nsec);
}

[[nodiscard]] bool
isTerminalStatus(nixl_status_t status) noexcept {
    return status == NIXL_SUCCESS || (status < NIXL_SUCCESS && status != NIXL_ERR_NOT_READY);
}

} // namespace

class nixlTerminalOwnerProducerState final {
public:
    nixlTerminalOwnerProducerState(const sglang_terminal_owner_producer_api_v1 *api, void *context)
        : api_(api),
          context_(context) {
        if (api_ == nullptr || context_ == nullptr ||
            api_->abi_version != SGLANG_TERMINAL_OWNER_PRODUCER_ABI_VERSION ||
            api_->struct_size != sizeof(sglang_terminal_owner_producer_api_v1) ||
            api_->event_struct_size != sizeof(sglang_terminal_owner_producer_event_v1) ||
            (api_->flags & SGLANG_TERMINAL_OWNER_PRODUCER_REQUIRED_FLAGS) !=
                SGLANG_TERMINAL_OWNER_PRODUCER_REQUIRED_FLAGS ||
            api_->submit == nullptr || api_->retire == nullptr || api_->join == nullptr) {
            throw std::invalid_argument("terminal owner producer ABI mismatch");
        }
    }

    [[nodiscard]] nixl_status_t
    beginSubscription(const nixl_terminal_owner_binding_digest_t &binding) noexcept {
        const std::lock_guard lock(mutex_);
        if (fatal_ != nixl_terminal_owner_producer_fatal_t::NONE || closed_) {
            return NIXL_ERR_BACKEND;
        }
        if (!admissionOpen_) {
            setFatalLocked(
                nixl_terminal_owner_producer_fatal_t::SUBMISSION_AFTER_STOP, ESHUTDOWN, binding);
            return NIXL_ERR_NOT_ALLOWED;
        }
        if (isZeroBinding(binding)) {
            setFatalLocked(
                nixl_terminal_owner_producer_fatal_t::INVALID_BINDING_STATE, EINVAL, binding);
            return NIXL_ERR_INVALID_PARAM;
        }
        const auto [iterator, inserted] = bindings_.emplace(binding, binding_phase_t::REGISTERING);
        static_cast<void>(iterator);
        if (!inserted) {
            setFatalLocked(
                nixl_terminal_owner_producer_fatal_t::DUPLICATE_BINDING, EEXIST, binding);
            return NIXL_ERR_NOT_ALLOWED;
        }
        ++activeCallbacks_;
        ++activeRegistrations_;
        return NIXL_SUCCESS;
    }

    void
    registrationSucceeded(const nixl_terminal_owner_binding_digest_t &binding) noexcept {
        const std::lock_guard lock(mutex_);
        const auto iterator = bindings_.find(binding);
        if (iterator == bindings_.end() || iterator->second != binding_phase_t::REGISTERING) {
            setFatalLocked(
                nixl_terminal_owner_producer_fatal_t::INVALID_BINDING_STATE, EPROTO, binding);
            return;
        }
        iterator->second = binding_phase_t::SUBMITTED;
        --activeRegistrations_;
        ++totalSubscriptions_;
    }

    void
    registrationFailed(const nixl_terminal_owner_binding_digest_t &binding,
                       nixl_status_t status) noexcept {
        const std::lock_guard lock(mutex_);
        const auto iterator = bindings_.find(binding);
        if (iterator == bindings_.end() || iterator->second != binding_phase_t::REGISTERING) {
            setFatalLocked(nixl_terminal_owner_producer_fatal_t::INVALID_BINDING_STATE,
                           static_cast<int>(status),
                           binding);
            return;
        }
        bindings_.erase(iterator);
        --activeCallbacks_;
        --activeRegistrations_;
    }

    void
    publish(const nixl_terminal_owner_binding_digest_t &owner_binding,
            const nixlBackendTransferEventBinding &expected_transfer_binding,
            const nixlBackendTransferTransition &transition) noexcept {
        bool invalid_transition = false;
        {
            const std::lock_guard lock(mutex_);
            const auto iterator = bindings_.find(owner_binding);
            if (iterator == bindings_.end()) {
                setFatalLocked(
                    nixl_terminal_owner_producer_fatal_t::UNKNOWN_BINDING, ENOENT, owner_binding);
                return;
            }
            if (iterator->second != binding_phase_t::SUBMITTED) {
                setFatalLocked(nixl_terminal_owner_producer_fatal_t::INVALID_BINDING_STATE,
                               EALREADY,
                               owner_binding);
                finishBindingLocked(iterator);
                return;
            }
            iterator->second = binding_phase_t::DELIVERING;
            invalid_transition =
                transition.binding.handleIdentity != expected_transfer_binding.handleIdentity ||
                transition.binding.generation != expected_transfer_binding.generation ||
                transition.nativeTimestampNs == 0 || !isTerminalStatus(transition.status);
            if (invalid_transition) {
                setFatalLocked(nixl_terminal_owner_producer_fatal_t::INVALID_TRANSITION,
                               static_cast<int>(transition.status),
                               owner_binding);
            }
        }

        sglang_terminal_owner_producer_event_v1 event{};
        event.abi_version = SGLANG_TERMINAL_OWNER_PRODUCER_ABI_VERSION;
        event.struct_size = sizeof(event);
        std::memcpy(event.binding_digest, owner_binding.data(), owner_binding.size());
        event.enqueued_ns =
            transition.nativeTimestampNs == 0 ? monotonicRawNs() : transition.nativeTimestampNs;
        if (invalid_transition) {
            event.event_kind = source_request_failed_event;
            event.reason_code = NIXL_TERMINAL_OWNER_REASON_INVALID_TRANSITION;
            event.backend_status = static_cast<std::int64_t>(transition.status);
        } else if (transition.status == NIXL_SUCCESS) {
            event.event_kind = source_native_terminal_event;
            event.backend_status = NIXL_SUCCESS;
        } else {
            event.event_kind = source_request_failed_event;
            event.reason_code = NIXL_TERMINAL_OWNER_REASON_TRANSFER_FAILED;
            event.backend_status = static_cast<std::int64_t>(transition.status);
        }

        const int submit_status = api_->submit(context_, &event);
        {
            const std::lock_guard lock(mutex_);
            const auto iterator = bindings_.find(owner_binding);
            if (iterator == bindings_.end() || iterator->second != binding_phase_t::DELIVERING) {
                setFatalLocked(nixl_terminal_owner_producer_fatal_t::INVALID_BINDING_STATE,
                               EPROTO,
                               owner_binding);
                return;
            }
            if (submit_status != 0) {
                ++ownerSubmissionFailures_;
                setFatalLocked(nixl_terminal_owner_producer_fatal_t::OWNER_SUBMISSION_FAILURE,
                               submit_status,
                               owner_binding);
            } else {
                ++totalDelivered_;
                if (event.event_kind == source_native_terminal_event) {
                    ++successfulTerminalEvents_;
                } else {
                    ++failureTerminalEvents_;
                }
            }
            finishBindingLocked(iterator);
        }
    }

    void
    stopAdmission() noexcept {
        const std::lock_guard lock(mutex_);
        admissionOpen_ = false;
    }

    [[nodiscard]] nixl_status_t
    join(std::uint64_t timeout_ns) noexcept {
        if (timeout_ns == 0) {
            return NIXL_ERR_INVALID_PARAM;
        }
        {
            const std::lock_guard lock(mutex_);
            admissionOpen_ = false;
            if (fatal_ != nixl_terminal_owner_producer_fatal_t::NONE || closed_) {
                return NIXL_ERR_BACKEND;
            }
            if (activeCallbacks_ != 0 || activeRegistrations_ != 0) {
                return NIXL_IN_PROG;
            }
            if (!bindings_.empty()) {
                setFatalLocked(nixl_terminal_owner_producer_fatal_t::CLOSE_WITH_RETAINED_BINDINGS,
                               EBUSY,
                               bindings_.begin()->first);
                return NIXL_ERR_BACKEND;
            }
            if (!retirementRequested_) {
                const int retire_status = api_->retire(context_);
                if (retire_status != 0) {
                    setFatalLocked(nixl_terminal_owner_producer_fatal_t::RETIREMENT_FAILURE,
                                   retire_status,
                                   {});
                    return NIXL_ERR_BACKEND;
                }
                retirementRequested_ = true;
            }
        }

        const int join_status = api_->join(context_, timeout_ns);
        const std::lock_guard lock(mutex_);
        if (join_status == ETIMEDOUT) {
            return NIXL_IN_PROG;
        }
        if (join_status != 0) {
            setFatalLocked(nixl_terminal_owner_producer_fatal_t::JOIN_FAILURE, join_status, {});
            return NIXL_ERR_BACKEND;
        }
        joined_ = true;
        return NIXL_SUCCESS;
    }

    [[nodiscard]] nixl_status_t
    close() noexcept {
        const std::lock_guard lock(mutex_);
        admissionOpen_ = false;
        if (closed_) {
            return NIXL_SUCCESS;
        }
        if (activeCallbacks_ != 0 || activeRegistrations_ != 0) {
            setFatalLocked(
                nixl_terminal_owner_producer_fatal_t::CLOSE_WITH_ACTIVE_CALLBACKS, EBUSY, {});
            return NIXL_ERR_NOT_ALLOWED;
        }
        if (!bindings_.empty()) {
            setFatalLocked(nixl_terminal_owner_producer_fatal_t::CLOSE_WITH_RETAINED_BINDINGS,
                           EBUSY,
                           bindings_.begin()->first);
            return NIXL_ERR_NOT_ALLOWED;
        }
        if (!joined_) {
            setFatalLocked(
                nixl_terminal_owner_producer_fatal_t::CLOSE_BEFORE_PRODUCER_JOIN, EBUSY, {});
            return NIXL_ERR_NOT_ALLOWED;
        }
        if (fatal_ != nixl_terminal_owner_producer_fatal_t::NONE) {
            return NIXL_ERR_BACKEND;
        }
        closed_ = true;
        return NIXL_SUCCESS;
    }

    [[nodiscard]] nixl_terminal_owner_producer_inventory_t
    inventory() const noexcept {
        const std::lock_guard lock(mutex_);
        nixl_terminal_owner_producer_inventory_t result;
        for (const auto &[binding, phase] : bindings_) {
            static_cast<void>(binding);
            if (phase == binding_phase_t::REGISTERING) {
                ++result.registeringBindings;
            } else {
                ++result.submittedBindings;
            }
        }
        result.activeCallbacks = activeCallbacks_;
        result.activeRegistrations = activeRegistrations_;
        result.totalSubscriptions = totalSubscriptions_;
        result.totalDelivered = totalDelivered_;
        result.successfulTerminalEvents = successfulTerminalEvents_;
        result.failureTerminalEvents = failureTerminalEvents_;
        result.ownerSubmissionFailures = ownerSubmissionFailures_;
        result.admissionOpen = admissionOpen_;
        result.retirementRequested = retirementRequested_;
        result.joined = joined_;
        result.closed = closed_;
        result.fatal = fatal_;
        result.fatalStatus = fatalStatus_;
        result.fatalBinding = fatalBinding_;
        result.fatalHasBinding = fatalHasBinding_;
        return result;
    }

private:
    using binding_map_t = std::
        unordered_map<nixl_terminal_owner_binding_digest_t, binding_phase_t, binding_digest_hash_t>;

    void
    finishBindingLocked(binding_map_t::iterator iterator) noexcept {
        bindings_.erase(iterator);
        if (activeCallbacks_ > 0) {
            --activeCallbacks_;
        }
    }

    void
    setFatalLocked(nixl_terminal_owner_producer_fatal_t fatal,
                   int status,
                   const nixl_terminal_owner_binding_digest_t &binding) noexcept {
        if (fatal_ != nixl_terminal_owner_producer_fatal_t::NONE) {
            return;
        }
        fatal_ = fatal;
        fatalStatus_ = status;
        fatalBinding_ = binding;
        fatalHasBinding_ = !isZeroBinding(binding);
        admissionOpen_ = false;
    }

    const sglang_terminal_owner_producer_api_v1 *api_;
    void *context_;
    mutable std::mutex mutex_;
    binding_map_t bindings_;
    std::size_t activeCallbacks_ = 0;
    std::size_t activeRegistrations_ = 0;
    std::uint64_t totalSubscriptions_ = 0;
    std::uint64_t totalDelivered_ = 0;
    std::uint64_t successfulTerminalEvents_ = 0;
    std::uint64_t failureTerminalEvents_ = 0;
    std::uint64_t ownerSubmissionFailures_ = 0;
    bool admissionOpen_ = true;
    bool retirementRequested_ = false;
    bool joined_ = false;
    bool closed_ = false;
    nixl_terminal_owner_producer_fatal_t fatal_ = nixl_terminal_owner_producer_fatal_t::NONE;
    int fatalStatus_ = 0;
    nixl_terminal_owner_binding_digest_t fatalBinding_{};
    bool fatalHasBinding_ = false;
};

nixlTerminalOwnerProducerH::nixlTerminalOwnerProducerH(
    const sglang_terminal_owner_producer_api_v1 *api,
    void *context)
    : state_(std::make_shared<nixlTerminalOwnerProducerState>(api, context)) {}

void
nixlTerminalOwnerProducerH::stopAdmission() noexcept {
    state_->stopAdmission();
}

nixl_status_t
nixlTerminalOwnerProducerH::join(std::uint64_t timeout_ns) noexcept {
    return state_->join(timeout_ns);
}

nixl_status_t
nixlTerminalOwnerProducerH::close() noexcept {
    return state_->close();
}

nixl_terminal_owner_producer_inventory_t
nixlTerminalOwnerProducerH::inventory() const noexcept {
    return state_->inventory();
}

nixl_status_t
nixlTerminalOwnerProducerH::bindAgent(std::uint64_t agent_identity) noexcept {
    if (agent_identity == 0) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const std::lock_guard lock(agentMutex_);
    if (agentIdentity_ == 0) {
        agentIdentity_ = agent_identity;
        return NIXL_SUCCESS;
    }
    return agentIdentity_ == agent_identity ? NIXL_SUCCESS : NIXL_ERR_INVALID_PARAM;
}

nixl_status_t
nixlTerminalOwnerProducerH::beginSubscription(
    const nixl_terminal_owner_binding_digest_t &binding) noexcept {
    return state_->beginSubscription(binding);
}

void
nixlTerminalOwnerProducerH::subscriptionRegistrationSucceeded(
    const nixl_terminal_owner_binding_digest_t &binding) noexcept {
    state_->registrationSucceeded(binding);
}

nixlTerminalOwnerTransferAdapter::nixlTerminalOwnerTransferAdapter(
    nixlBackendTransferEventBinding transfer_binding,
    nixl_terminal_owner_binding_digest_t owner_binding,
    std::shared_ptr<nixlTerminalOwnerProducerState> producer)
    : transferBinding_(transfer_binding),
      ownerBinding_(owner_binding),
      producer_(std::move(producer)) {}

void
nixlTerminalOwnerTransferAdapter::bindTerminalCallback(terminal_callback_t terminal_callback) {
    std::optional<nixlBackendTransferTransition> pending;
    {
        const std::lock_guard lock(mutex_);
        if (terminalCallback_ != nullptr || terminal_) {
            return;
        }
        terminalCallback_ = std::move(terminal_callback);
        pending = std::move(pendingTransition_);
        pendingTransition_.reset();
    }
    if (pending.has_value()) {
        publishBound(*pending);
    }
}

bool
nixlTerminalOwnerTransferAdapter::isTerminal() const noexcept {
    const std::lock_guard lock(mutex_);
    return terminal_;
}

void
nixlTerminalOwnerTransferAdapter::publish(
    const nixlBackendTransferTransition &transition) noexcept {
    {
        const std::lock_guard lock(mutex_);
        if (terminal_) {
            return;
        }
        if (terminalCallback_ == nullptr) {
            if (pendingTransition_.has_value()) {
                pendingTransition_ = nixlBackendTransferTransition{
                    .binding = transferBinding_,
                    .status = NIXL_ERR_BACKEND,
                    .nativeTimestampNs = 0,
                };
            } else {
                pendingTransition_ = transition;
            }
            return;
        }
    }
    publishBound(transition);
}

void
nixlTerminalOwnerTransferAdapter::release() noexcept {
    abortRegistration(NIXL_ERR_CANCELED);
}

void
nixlTerminalOwnerTransferAdapter::abortRegistration(nixl_status_t status) noexcept {
    std::shared_ptr<nixlTerminalOwnerProducerState> producer;
    {
        const std::lock_guard lock(mutex_);
        if (terminal_) {
            return;
        }
        terminal_ = true;
        pendingTransition_.reset();
        producer = std::move(producer_);
    }
    if (producer != nullptr) {
        producer->registrationFailed(ownerBinding_, status);
    }
}

void
nixlTerminalOwnerTransferAdapter::publishBound(
    const nixlBackendTransferTransition &transition) noexcept {
    std::shared_ptr<nixlTerminalOwnerProducerState> producer;
    terminal_callback_t terminal_callback;
    {
        const std::lock_guard lock(mutex_);
        if (terminal_ || terminalCallback_ == nullptr || producer_ == nullptr) {
            return;
        }
        terminal_ = true;
        producer = producer_;
        terminal_callback = terminalCallback_;
    }

    producer->publish(ownerBinding_, transferBinding_, transition);
    terminal_callback();

    const std::lock_guard lock(mutex_);
    producer_.reset();
}

const char *
nixlTerminalOwnerProducerFatalName(nixl_terminal_owner_producer_fatal_t fatal) noexcept {
    switch (fatal) {
    case nixl_terminal_owner_producer_fatal_t::NONE:
        return "none";
    case nixl_terminal_owner_producer_fatal_t::DUPLICATE_BINDING:
        return "duplicate_binding";
    case nixl_terminal_owner_producer_fatal_t::UNKNOWN_BINDING:
        return "unknown_binding";
    case nixl_terminal_owner_producer_fatal_t::INVALID_BINDING_STATE:
        return "invalid_binding_state";
    case nixl_terminal_owner_producer_fatal_t::INVALID_TRANSITION:
        return "invalid_transition";
    case nixl_terminal_owner_producer_fatal_t::OWNER_SUBMISSION_FAILURE:
        return "owner_submission_failure";
    case nixl_terminal_owner_producer_fatal_t::SUBMISSION_AFTER_STOP:
        return "submission_after_stop";
    case nixl_terminal_owner_producer_fatal_t::RETIREMENT_FAILURE:
        return "retirement_failure";
    case nixl_terminal_owner_producer_fatal_t::JOIN_FAILURE:
        return "join_failure";
    case nixl_terminal_owner_producer_fatal_t::CLOSE_WITH_ACTIVE_CALLBACKS:
        return "close_with_active_callbacks";
    case nixl_terminal_owner_producer_fatal_t::CLOSE_WITH_RETAINED_BINDINGS:
        return "close_with_retained_bindings";
    case nixl_terminal_owner_producer_fatal_t::CLOSE_BEFORE_PRODUCER_JOIN:
        return "close_before_producer_join";
    }
    return "unknown";
}
