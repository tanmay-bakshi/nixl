/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_TEST_QUALIFICATION_TERMINAL_UCX_API_ADAPTER_H
#define NIXL_TEST_QUALIFICATION_TERMINAL_UCX_API_ADAPTER_H

#include <cstddef>
#include <cstdint>

#include "nixl.h"

class nixlBackendAdmissionReceiptBarrier;

namespace nixl::qualification {

struct terminal_channel_inventory_t {
    std::size_t capacity = 0;
    std::size_t queuedChannelEvents = 0;
    std::size_t activeChannelSubscriptions = 0;
    std::size_t retainedPublicSubscriptions = 0;
    std::size_t backendProducers = 0;
    std::size_t activeCallbackSlots = 0;
    std::size_t queuedOwnerContinuations = 0;
    bool acceptingSubscriptions = false;
    bool closed = false;
    std::uint32_t fatal = 0;
    int eventfdError = 0;
};

/**
 * Narrow qualification wrapper around the public terminal-event API.
 */
class terminal_ucx_api_adapter_t final {
public:
    terminal_ucx_api_adapter_t(nixlAgent &agent, std::size_t capacity);

    terminal_ucx_api_adapter_t(const terminal_ucx_api_adapter_t &) = delete;
    terminal_ucx_api_adapter_t &
    operator=(const terminal_ucx_api_adapter_t &) = delete;

    [[nodiscard]] nixl_status_t
    subscribeTransfer(nixlXferReqH *request,
                      std::uint64_t owner_cookie,
                      nixlTerminalEventSubscriptionH *&subscription);

    [[nodiscard]] nixl_status_t
    subscribeCapability(const nixlRemoteAgentH *remote_agent,
                        const nixlBackendH *backend,
                        std::uint64_t owner_cookie,
                        nixlTerminalEventSubscriptionH *&subscription);

    [[nodiscard]] nixl_status_t
    fileno(int &fd) const;

    [[nodiscard]] nixl_status_t
    drain(nixl_terminal_event_batch_t &batch);

    [[nodiscard]] nixl_status_t
    release(nixlTerminalEventSubscriptionH *subscription);

    [[nodiscard]] nixl_status_t
    querySubscription(const nixlTerminalEventSubscriptionH *subscription,
                      nixl_terminal_subscription_info_t &info) const;

    [[nodiscard]] nixl_status_t
    queryInventory(terminal_channel_inventory_t &inventory);

    [[nodiscard]] static nixl_status_t
    installAdmissionReceiptBarrier(nixlAgent &agent,
                                   const nixlBackendH *backend,
                                   nixlBackendAdmissionReceiptBarrier *barrier) noexcept;

    [[nodiscard]] nixl_status_t
    close();

private:
    nixlAgent &agent_;
    nixlTerminalEventChannelH *channel_ = nullptr;
};

} // namespace nixl::qualification

#endif // NIXL_TEST_QUALIFICATION_TERMINAL_UCX_API_ADAPTER_H
