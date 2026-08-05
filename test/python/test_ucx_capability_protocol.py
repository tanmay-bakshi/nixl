# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import time
import uuid

import pytest

import nixl._bindings as bindings
import nixl._utils as utils
from nixl._api import (
    nixl_agent,
    nixl_agent_config,
    nixl_remote_agent_handle,
    nixl_xfer_handle,
)


_DEADLINE_SECONDS = 10.0


def _agent(role: str) -> nixl_agent:
    """Create one UCX agent with its normal production progress thread.

    :param role: Human-readable role used in the unique agent name.
    :returns: Configured agent.
    """
    config = nixl_agent_config(backends=["UCX"], num_threads=0)
    return nixl_agent(f"{role}-{uuid.uuid4()}", config)


def _progress(*agents: nixl_agent) -> None:
    """Drive control-plane progress without retaining notifications.

    :param agents: Agents whose backend queues should be progressed.
    """
    for agent in agents:
        notifications = agent.get_new_notifs()
        assert len(notifications) == 0


def _send_when_ready(
    sender: nixl_agent,
    receiver: nixl_agent,
    remote: nixl_remote_agent_handle,
    payload: bytes,
) -> None:
    """Retry a standalone notification while its exact capability converges.

    :param sender: Notification sender.
    :param receiver: Notification receiver.
    :param remote: Receiver handle owned by ``sender``.
    :param payload: Notification bytes.
    """
    deadline = time.monotonic() + _DEADLINE_SECONDS
    while True:
        try:
            sender.send_notif(remote, payload)
            return
        except bindings.nixlNotReadyError:
            _progress(sender, receiver)
            if time.monotonic() >= deadline:
                pytest.fail("notification capability did not become ready")
            time.sleep(0.001)


def _wait_for_notification(
    receiver: nixl_agent,
    sender: nixl_remote_agent_handle,
) -> list[bytes]:
    """Wait for notifications attributed to one exact sender handle.

    :param receiver: Agent receiving the notification.
    :param sender: Canonical sender handle owned by ``receiver``.
    :returns: Messages drained for ``sender``.
    """
    deadline = time.monotonic() + _DEADLINE_SECONDS
    while True:
        notifications = receiver.get_new_notifs()
        if sender in notifications:
            assert next(iter(notifications)) is sender
            return notifications[sender]
        if time.monotonic() >= deadline:
            pytest.fail("notification did not arrive on the exact remote handle")
        time.sleep(0.001)


def _wait_for_transfer(agent: nixl_agent, handle: nixl_xfer_handle) -> None:
    """Wait for one successfully submitted transfer.

    :param agent: Transfer owner.
    :param handle: Submitted transfer handle.
    """
    deadline = time.monotonic() + _DEADLINE_SECONDS
    while True:
        state = agent.check_xfer_state(handle)
        if state == "DONE":
            return
        assert state == "PROC"
        if time.monotonic() >= deadline:
            pytest.fail("transfer did not complete")
        time.sleep(0.001)


def test_standalone_notifications_converge_bidirectionally() -> None:
    """Deliver standalone notifications through reciprocal exact capabilities."""
    left = _agent("left")
    right = _agent("right")
    right_handle = left.add_remote_agent(right.get_agent_metadata())
    left_handle = right.add_remote_agent(left.get_agent_metadata())

    try:
        _send_when_ready(left, right, right_handle, b"left-to-right")
        assert _wait_for_notification(right, left_handle) == [b"left-to-right"]

        _send_when_ready(right, left, left_handle, b"right-to-left")
        assert _wait_for_notification(left, right_handle) == [b"right-to-left"]
    finally:
        left.remove_remote_agent(right_handle)
        right.remove_remote_agent(left_handle)


def test_transfer_data_waits_for_notification_capability() -> None:
    """Return NOT_READY without moving bytes, then retry the same request."""
    initiator = _agent("initiator")
    target = _agent("target")
    size = 4096
    source = utils.malloc_passthru(size)
    destination = utils.malloc_passthru(size)
    source_registration = None
    destination_registration = None
    target_handle = None
    initiator_handle = None
    transfer_handle = None

    try:
        utils.ba_buf(source, size)
        ctypes.memset(destination, 0xA5, size)
        source_bytes = ctypes.string_at(source, size)
        untouched_bytes = bytes([0xA5]) * size
        assert source_bytes != untouched_bytes

        source_registration = initiator.register_memory(
            [(source, size, 0, "source")], mem_type="DRAM"
        )
        destination_registration = target.register_memory(
            [(destination, size, 0, "destination")], mem_type="DRAM"
        )

        target_handle = initiator.add_remote_agent(target.get_agent_metadata())
        source_descriptors = initiator.get_xfer_descs(
            [(source, size, 0)], mem_type="DRAM"
        )
        destination_descriptors = initiator.get_xfer_descs(
            [(destination, size, 0)], mem_type="DRAM"
        )
        transfer_handle = initiator.initialize_xfer(
            "WRITE",
            source_descriptors,
            destination_descriptors,
            target_handle,
            b"transfer-complete",
        )

        assert initiator.transfer(transfer_handle) == "NOT_READY"
        assert ctypes.string_at(destination, size) == untouched_bytes

        initiator_handle = target.add_remote_agent(initiator.get_agent_metadata())
        deadline = time.monotonic() + _DEADLINE_SECONDS
        while True:
            state = initiator.transfer(transfer_handle)
            if state != "NOT_READY":
                break
            _progress(initiator, target)
            if time.monotonic() >= deadline:
                pytest.fail("transfer notification capability did not become ready")
            time.sleep(0.001)

        assert state in {"DONE", "PROC"}
        if state == "PROC":
            _wait_for_transfer(initiator, transfer_handle)
        assert ctypes.string_at(destination, size) == source_bytes
        assert _wait_for_notification(target, initiator_handle) == [
            b"transfer-complete"
        ]
    finally:
        if transfer_handle is not None:
            transfer_handle.release()
        if source_registration is not None:
            initiator.deregister_memory(source_registration)
        if destination_registration is not None:
            target.deregister_memory(destination_registration)
        if target_handle is not None:
            initiator.remove_remote_agent(target_handle)
        if initiator_handle is not None:
            target.remove_remote_agent(initiator_handle)
        utils.free_passthru(source)
        utils.free_passthru(destination)
