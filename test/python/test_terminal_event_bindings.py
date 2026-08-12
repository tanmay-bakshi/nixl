# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import fcntl
import os
import select
from dataclasses import dataclass

import pytest

import nixl._bindings as bindings
import nixl._api as api


def _raw_agent(name: str) -> bindings.nixlAgent:
    """Create a native agent without a listener or progress thread.

    :param name: Unique test agent name.
    :returns: Native test agent.
    """
    config = bindings.nixlAgentConfig()
    config.useProgThread = False
    config.useListenThread = False
    return bindings.nixlAgent(name, config)


def test_terminal_channel_binding_exposes_borrowed_fd_and_inventory() -> None:
    """Expose a poll-only fd and distinct immutable lifecycle counters."""
    agent = _raw_agent("terminal-binding-inventory")
    channel = agent.createTerminalEventChannel(8)

    fd = agent.getTerminalEventChannelFd(channel)
    assert fd >= 0
    assert fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_NONBLOCK
    assert fcntl.fcntl(fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
    assert select.select([fd], [], [], 0)[0] == []

    inventory = agent.queryTerminalEventChannel(channel)
    assert inventory.capacity == 8
    assert inventory.queuedChannelEvents == 0
    assert inventory.activeChannelSubscriptions == 0
    assert inventory.retainedPublicSubscriptions == 0
    assert inventory.backendProducers == 0
    assert inventory.activeCallbackSlots == 0
    assert inventory.queuedOwnerContinuations == 0
    assert inventory.acceptingSubscriptions
    assert not inventory.closed
    assert inventory.fatal == bindings.nixl_terminal_channel_fatal_t.NONE
    assert inventory.eventfdError == 0

    with pytest.raises(AttributeError):
        inventory.queuedChannelEvents = 1

    batch = agent.drainTerminalEvents(channel)
    assert batch.events == ()
    assert batch.wakeCount == 0
    assert batch.inventory.queuedChannelEvents == 0
    with pytest.raises(AttributeError):
        batch.wakeCount = 1

    assert agent.closeTerminalEventChannel(channel) == bindings.NIXL_SUCCESS
    assert agent.queryTerminalEventChannel(channel).closed


def test_terminal_channel_binding_rejects_duplicate_and_foreign_handles() -> None:
    """Preserve exact agent ownership and the one-channel lifetime invariant."""
    owner = _raw_agent("terminal-binding-owner")
    foreign = _raw_agent("terminal-binding-foreign")
    channel = owner.createTerminalEventChannel(2)

    with pytest.raises(bindings.nixlNotAllowedError):
        owner.createTerminalEventChannel(2)
    with pytest.raises(bindings.nixlInvalidParamError):
        foreign.queryTerminalEventChannel(channel)

    assert owner.closeTerminalEventChannel(channel) == bindings.NIXL_SUCCESS


def test_terminal_types_are_read_only_and_cover_both_terminal_route_states() -> None:
    """Bind FAILED and RETIRED distinctly without Python construction authority."""
    assert bindings.nixl_terminal_capability_state_t.FAILED.name == "FAILED"
    assert bindings.nixl_terminal_capability_state_t.RETIRED.name == "RETIRED"
    assert bindings.nixl_terminal_event_kind_t.TRANSFER.name == "TRANSFER"
    assert bindings.nixl_terminal_event_kind_t.CAPABILITY.name == "CAPABILITY"

    with pytest.raises(TypeError):
        bindings.nixlTerminalEvent()
    with pytest.raises(TypeError):
        bindings.nixlTerminalSubscriptionInfo()


def test_high_level_channel_wraps_native_fd_drain_and_close() -> None:
    """Keep the public wrapper thin over the native nonblocking channel."""
    owner = object.__new__(api.nixl_agent)
    owner.agent = _raw_agent("terminal-wrapper-channel")
    owner._terminal_event_channel = None
    owner._leaked_xfer_handles = []

    channel = owner.create_terminal_event_channel(4)
    assert isinstance(channel, api.nixl_terminal_event_channel)
    assert channel.fileno() >= 0
    assert channel.query_inventory().capacity == 4
    assert channel.drain().events == ()
    with pytest.raises(RuntimeError, match="already owns"):
        owner.create_terminal_event_channel(4)
    assert channel.close() == bindings.NIXL_SUCCESS


@dataclass(frozen=True)
class _FakeSubscriptionInfo:
    """Immutable native query result used to qualify wrapper lifecycle."""

    kind: bindings.nixl_terminal_event_kind_t
    ownerCookie: int
    identity: int
    generation: int
    active: bool


class _FakeTerminalAgent:
    """Deterministic native-agent surface for wrapper-only lifecycle tests."""

    def __init__(self, release_statuses: list[bindings.nixl_status_t]) -> None:
        """Create a fake with prospective release outcomes.

        :param release_statuses: Status returned by each release call.
        """
        self._release_statuses = list(release_statuses)
        self._active = True
        self._close_fails = True

    def queryTerminalEventSubscription(
        self, native_handle: object
    ) -> _FakeSubscriptionInfo:
        """Return the exact fake subscription state.

        :param native_handle: Opaque fake native handle.
        :returns: Immutable lifecycle snapshot.
        """
        assert native_handle is not None
        return _FakeSubscriptionInfo(
            kind=bindings.nixl_terminal_event_kind_t.TRANSFER,
            ownerCookie=7,
            identity=11,
            generation=3,
            active=self._active,
        )

    def releaseTerminalEventSubscription(
        self, native_handle: object
    ) -> bindings.nixl_status_t:
        """Return the next prospective cancellation outcome.

        :param native_handle: Opaque fake native handle.
        :returns: Native-style release status.
        """
        assert native_handle is not None
        status = self._release_statuses.pop(0)
        if status == bindings.NIXL_SUCCESS:
            self._active = False
            self._close_fails = False
        return status

    def closeTerminalEventChannel(
        self, native_handle: object
    ) -> bindings.nixl_status_t:
        """Fail closed until the retained public handle is consumed.

        :param native_handle: Opaque fake native channel.
        :returns: Native success after release.
        :raises bindings.nixlNotAllowedError: While a subscription is live.
        """
        assert native_handle is not None
        if self._close_fails:
            raise bindings.nixlNotAllowedError("live terminal subscription")
        return bindings.NIXL_SUCCESS


def test_subscription_wrapper_retains_async_cancellation_until_terminal() -> None:
    """Do not consume or finalize a handle on NIXL_IN_PROG cancellation."""
    native_agent = _FakeTerminalAgent(
        [bindings.NIXL_IN_PROG, bindings.NIXL_SUCCESS]
    )
    owner = object.__new__(api.nixl_agent)
    owner.agent = native_agent
    channel = api.nixl_terminal_event_channel(
        owner,
        object(),
        api._TERMINAL_CHANNEL_CONSTRUCTION_TOKEN,
    )
    subscription = channel._retain(object())

    with pytest.raises(bindings.nixlNotAllowedError):
        channel.close()
    assert subscription.release() == bindings.NIXL_IN_PROG
    assert not subscription._released
    assert len(channel._subscriptions) == 1
    assert subscription.query().active

    assert subscription.release() == bindings.NIXL_SUCCESS
    assert subscription._released
    assert len(channel._subscriptions) == 0
    assert channel.close() == bindings.NIXL_SUCCESS
    with pytest.raises(RuntimeError, match="already been released"):
        subscription.release()


@pytest.mark.parametrize(
    "state",
    [
        bindings.nixl_terminal_capability_state_t.FAILED,
        bindings.nixl_terminal_capability_state_t.RETIRED,
    ],
)
def test_terminal_capability_states_remain_typed(
    state: bindings.nixl_terminal_capability_state_t,
) -> None:
    """Keep both terminal route outcomes as exact enum values.

    :param state: Terminal capability state under test.
    """
    assert isinstance(state, bindings.nixl_terminal_capability_state_t)
    assert state not in (bindings.nixl_terminal_capability_state_t.READY,)
