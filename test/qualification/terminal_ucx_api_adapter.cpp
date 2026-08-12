/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "terminal_ucx_api_adapter.h"

#include <stdexcept>

namespace nixl::qualification {

terminal_ucx_api_adapter_t::terminal_ucx_api_adapter_t(nixlAgent &agent, std::size_t capacity)
    : agent_(agent) {
    const nixl_status_t status = agent_.createTerminalEventChannel(capacity, channel_);
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error("failed to create terminal event channel");
    }
}

nixl_status_t
terminal_ucx_api_adapter_t::subscribeTransfer(nixlXferReqH *request,
                                              std::uint64_t owner_cookie,
                                              nixlTerminalEventSubscriptionH *&subscription) {
    return agent_.subscribeXferTerminal(channel_, request, owner_cookie, subscription);
}

nixl_status_t
terminal_ucx_api_adapter_t::subscribeCapability(const nixlRemoteAgentH *remote_agent,
                                                const nixlBackendH *backend,
                                                std::uint64_t owner_cookie,
                                                nixlTerminalEventSubscriptionH *&subscription) {
    return agent_.subscribeRemoteNotificationState(
        channel_, remote_agent, backend, owner_cookie, subscription);
}

nixl_status_t
terminal_ucx_api_adapter_t::fileno(int &fd) const {
    return agent_.getTerminalEventChannelFd(channel_, fd);
}

nixl_status_t
terminal_ucx_api_adapter_t::drain(nixl_terminal_event_batch_t &batch) {
    return agent_.drainTerminalEvents(channel_, batch);
}

nixl_status_t
terminal_ucx_api_adapter_t::release(nixlTerminalEventSubscriptionH *subscription) {
    return agent_.releaseTerminalEventSubscription(subscription);
}

nixl_status_t
terminal_ucx_api_adapter_t::querySubscription(const nixlTerminalEventSubscriptionH *subscription,
                                              nixl_terminal_subscription_info_t &info) const {
    return agent_.queryTerminalEventSubscription(subscription, info);
}

nixl_status_t
terminal_ucx_api_adapter_t::queryInventory(terminal_channel_inventory_t &inventory) {
    nixl_terminal_channel_inventory_t channel_inventory;
    const nixl_status_t status = agent_.queryTerminalEventChannel(channel_, channel_inventory);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    std::size_t subscription_count = 0;
    const nixl_status_t count_status =
        agent_.getTerminalEventSubscriptionCount(channel_, subscription_count);
    if (count_status != NIXL_SUCCESS) {
        return count_status;
    }

    inventory = {
        .activeSubscriptions = subscription_count,
        .activeProducers = channel_inventory.activeSubscriptions,
        .queuedEvents = channel_inventory.queuedEvents,
        .closed = channel_inventory.closed,
        .fatal = static_cast<std::uint32_t>(channel_inventory.fatal),
    };
    return NIXL_SUCCESS;
}

nixl_status_t
terminal_ucx_api_adapter_t::close() {
    return agent_.closeTerminalEventChannel(channel_);
}

} // namespace nixl::qualification
