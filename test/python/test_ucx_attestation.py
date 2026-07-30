# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import os
import pickle
import time
import uuid

import pytest

import nixl._bindings as bindings
import nixl._utils as utils
from nixl._api import nixl_agent, nixl_agent_config, nixl_xfer_handle


def _make_agents(num_threads: int) -> tuple[nixl_agent, nixl_agent]:
    """Create two UCX agents.

    :param num_threads: Dedicated UCX thread-pool workers per agent.
    :returns: Initiator and target agents.
    """
    config = nixl_agent_config(backends=["UCX"], num_threads=num_threads)
    initiator = nixl_agent(f"initiator-{uuid.uuid4()}", config)
    target = nixl_agent(f"target-{uuid.uuid4()}", config)
    return initiator, target


def _wait_for_done(agent: nixl_agent, handle: nixl_xfer_handle) -> None:
    """Poll one transfer to top-level completion.

    :param agent: Agent that owns the transfer handle.
    :param handle: Transfer handle to poll.
    """
    deadline = time.monotonic() + 30
    while agent.check_xfer_state(handle) != "DONE":
        if time.monotonic() >= deadline:
            pytest.fail("transfer did not complete before the deadline")
        time.sleep(0.001)


def _assert_non_constructible_and_read_only(
    snapshot: bindings.nixlXferAttestationSnapshot,
) -> None:
    """Verify Python evidence objects expose no forging or mutation surface.

    :param snapshot: Diagnostic evidence snapshot to inspect.
    """
    evidence_types = (
        bindings.nixlXferAttestationSnapshot,
        bindings.nixlXferCompletionReceipt,
        bindings.nixlXferAttestationTransport,
        bindings.nixlXferAttestationSegment,
        bindings.nixlXferAttestationEndpoint,
        bindings.nixlRuntimeArtifact,
    )
    for evidence_type in evidence_types:
        with pytest.raises(TypeError):
            evidence_type()

    with pytest.raises(TypeError):
        pickle.dumps(snapshot)
    with pytest.raises(AttributeError):
        snapshot.generation = snapshot.generation + 1
    with pytest.raises(AttributeError):
        snapshot.segments[0].posted = False
    with pytest.raises(AttributeError):
        snapshot.endpoints[0].remoteFlushed = False
    with pytest.raises(AttributeError):
        snapshot.endpoints[0].transports[0].device = "forged"
    with pytest.raises(AttributeError):
        snapshot.runtimeArtifacts[0].buildId = "forged"


def test_completion_receipt_waits_for_notification_terminal_status() -> None:
    """Gate unique completion authority on the whole native request handle."""
    initiator, target = _make_agents(num_threads=0)
    size = 16 * 1024 * 1024
    source = utils.malloc_passthru(size)
    destination = utils.malloc_passthru(size)
    handle: nixl_xfer_handle | None = None
    source_registration: bindings.nixlRegDList | None = None
    destination_registration: bindings.nixlRegDList | None = None
    initiator_connected = False
    target_connected = False

    try:
        utils.ba_buf(source, size)
        ctypes.memset(destination, 0, size)
        source_registration = initiator.register_memory(
            [(source, size, 0, "source")], mem_type="DRAM"
        )
        destination_registration = target.register_memory(
            [(destination, size, 0, "destination")], mem_type="DRAM"
        )
        initiator.add_remote_agent(target.get_agent_metadata())
        initiator_connected = True
        target.add_remote_agent(initiator.get_agent_metadata())
        target_connected = True

        source_descriptors = initiator.get_xfer_descs(
            [(source, size, 0)], mem_type="DRAM"
        )
        destination_descriptors = initiator.get_xfer_descs(
            [(destination, size, 0)], mem_type="DRAM"
        )
        notification = b"completion-gate:" + b"x" * (4 * 1024 * 1024)
        handle = initiator.initialize_xfer(
            "WRITE",
            source_descriptors,
            destination_descriptors,
            target.name,
            notification,
        )
        assert initiator.transfer(handle) == "PROC"

        pending_snapshot = None
        deadline = time.monotonic() + 30
        while pending_snapshot is None:
            state = initiator.check_xfer_state(handle)
            snapshot = initiator.query_xfer_attestation(handle)
            if (
                state == "PROC"
                and snapshot.state
                == bindings.NIXL_XFER_ATTESTATION_REMOTE_FLUSHED
            ):
                pending_snapshot = snapshot
                break
            if state == "DONE":
                pytest.fail(
                    "notification completed before the terminal receipt gate was observable"
                )
            if time.monotonic() >= deadline:
                pytest.fail("remote flush did not seal before the deadline")
            time.sleep(0.001)

        assert pending_snapshot.submissionSealed
        assert pending_snapshot.status == bindings.NIXL_SUCCESS
        assert pending_snapshot.completionClaimed is False
        assert initiator.take_xfer_completion_receipt(handle) is None
        _assert_non_constructible_and_read_only(pending_snapshot)

        _wait_for_done(initiator, handle)
        receipt = initiator.take_xfer_completion_receipt(handle)
        assert isinstance(receipt, bindings.nixlXferCompletionReceipt)
        assert type(receipt) is not type(pending_snapshot)
        assert receipt.completionClaimed
        assert receipt.generation == pending_snapshot.generation
        assert receipt.descriptorDigest == pending_snapshot.descriptorDigest
        assert receipt.evidenceDigest == pending_snapshot.evidenceDigest
        assert len(receipt.runtimeArtifacts) == 3
        assert {artifact.component for artifact in receipt.runtimeArtifacts} == {
            "libnixl",
            "libucp",
            "ucx-plugin",
        }
        for artifact in receipt.runtimeArtifacts:
            assert os.path.realpath(artifact.path) == artifact.path
            assert len(artifact.buildId) > 0
            assert len(artifact.buildId) % 2 == 0
            int(artifact.buildId, 16)
            assert len(artifact.version) > 0

        with pytest.raises(bindings.nixlNotAllowedError):
            initiator.take_xfer_completion_receipt(handle)
        assert initiator.query_xfer_attestation(handle).completionClaimed
        utils.verify_transfer(source, destination, size)

        notification_deadline = time.monotonic() + 30
        notifications: dict[str, list[bytes]] = {}
        while initiator.name not in notifications:
            notifications = target.get_new_notifs()
            if time.monotonic() >= notification_deadline:
                pytest.fail("completion notification did not arrive before the deadline")
            time.sleep(0.001)
        assert notifications[initiator.name] == [notification]
    finally:
        if handle is not None:
            handle.release()
        if source_registration is not None:
            initiator.deregister_memory(source_registration)
        if destination_registration is not None:
            target.deregister_memory(destination_registration)
        if initiator_connected:
            initiator.remove_remote_agent(target.name)
        if target_connected:
            target.remove_remote_agent(initiator.name)
        utils.free_passthru(source)
        utils.free_passthru(destination)


def test_thread_pool_completion_receipt_covers_every_chunk() -> None:
    """Bind one receipt to every segment and endpoint in a composite request."""
    initiator, target = _make_agents(num_threads=2)
    segment_count = 2048
    segment_length = 64
    segment_stride = 128
    region_size = (segment_count - 1) * segment_stride + segment_length
    source = utils.malloc_passthru(region_size)
    destination = utils.malloc_passthru(region_size)
    handle: nixl_xfer_handle | None = None
    source_registration: bindings.nixlRegDList | None = None
    destination_registration: bindings.nixlRegDList | None = None
    initiator_connected = False
    target_connected = False

    try:
        utils.ba_buf(source, region_size)
        ctypes.memset(destination, 0, region_size)
        source_registration = initiator.register_memory(
            [(source, region_size, 0, "source")], mem_type="DRAM"
        )
        destination_registration = target.register_memory(
            [(destination, region_size, 0, "destination")], mem_type="DRAM"
        )
        initiator.add_remote_agent(target.get_agent_metadata())
        initiator_connected = True
        target.add_remote_agent(initiator.get_agent_metadata())
        target_connected = True

        source_descriptors = initiator.get_xfer_descs(
            [
                (source + index * segment_stride, segment_length, 0)
                for index in range(segment_count)
            ],
            mem_type="DRAM",
        )
        destination_descriptors = initiator.get_xfer_descs(
            [
                (destination + index * segment_stride, segment_length, 0)
                for index in range(segment_count)
            ],
            mem_type="DRAM",
        )
        handle = initiator.initialize_xfer(
            "WRITE",
            source_descriptors,
            destination_descriptors,
            target.name,
        )
        assert initiator.transfer(handle) in {"DONE", "PROC"}
        _wait_for_done(initiator, handle)

        receipt = initiator.take_xfer_completion_receipt(handle)
        assert isinstance(receipt, bindings.nixlXferCompletionReceipt)
        assert len(receipt.segments) == segment_count
        assert all(segment.posted for segment in receipt.segments)
        assert len(receipt.endpoints) == 2
        assert len({endpoint.workerId for endpoint in receipt.endpoints}) == 2
        assert all(endpoint.flushPosted for endpoint in receipt.endpoints)
        assert all(endpoint.remoteFlushed for endpoint in receipt.endpoints)
        assert sorted(
            index
            for endpoint in receipt.endpoints
            for index in endpoint.segmentIndices
        ) == list(range(segment_count))

        source_bytes = b"".join(
            ctypes.string_at(source + index * segment_stride, segment_length)
            for index in range(segment_count)
        )
        destination_bytes = b"".join(
            ctypes.string_at(destination + index * segment_stride, segment_length)
            for index in range(segment_count)
        )
        assert destination_bytes == source_bytes
    finally:
        if handle is not None:
            handle.release()
        if source_registration is not None:
            initiator.deregister_memory(source_registration)
        if destination_registration is not None:
            target.deregister_memory(destination_registration)
        if initiator_connected:
            initiator.remove_remote_agent(target.name)
        if target_connected:
            target.remove_remote_agent(initiator.name)
        utils.free_passthru(source)
        utils.free_passthru(destination)
