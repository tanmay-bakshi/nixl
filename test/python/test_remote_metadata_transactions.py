# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid

import nixl._bindings as nixl
import nixl._utils as utils


def _agent(name: str) -> nixl.nixlAgent:
    """Create one non-threaded agent for metadata transaction tests.

    :param name: Agent name.
    :returns: Configured native agent.
    """
    config = nixl.nixlAgentConfig()
    config.useProgThread = False
    config.useListenThread = False
    return nixl.nixlAgent(name, config)


def test_partial_metadata_merge_and_generation() -> None:
    """Merge additive partials into one handle and advance only after invalidation."""
    suffix = str(uuid.uuid4())
    source = _agent(f"source-{suffix}")
    receiver = _agent(f"receiver-{suffix}")
    source_backend = source.createBackend("UCX", {})
    receiver_backend = receiver.createBackend("UCX", {})
    size = 256
    address = utils.malloc_passthru(size)
    registration = nixl.nixlRegDList(nixl.DRAM_SEG)
    registration.addDesc((address, size, 0, ""))
    remote = None
    reloaded = None

    try:
        assert source.registerMem(registration, [source_backend]) == nixl.NIXL_SUCCESS
        empty = nixl.nixlRegDList(nixl.DRAM_SEG)
        connection_document = source.getLocalPartialMD(empty, True, [source_backend])
        descriptor_document = source.getLocalPartialMD(
            registration, False, [source_backend]
        )

        remote = receiver.loadRemoteMD(connection_document)
        merged = receiver.loadRemoteMD(descriptor_document)
        assert merged == remote
        assert merged.identity == remote.identity
        assert merged.generation == remote.generation
        assert receiver.loadRemoteMD(descriptor_document) == remote

        remote_descs = nixl.nixlXferDList(nixl.DRAM_SEG)
        remote_descs.addDesc((address, size, 0))
        prepared = receiver.prepXferDlist(remote, remote_descs, [receiver_backend])
        assert prepared != 0
        assert receiver.releasedDlistH(prepared) == nixl.NIXL_SUCCESS

        receiver.invalidateRemoteMD(remote)
        remote = None
        reloaded = receiver.loadRemoteMD(connection_document)
        assert reloaded.identity != merged.identity
        assert reloaded.generation == merged.generation + 1
        assert receiver.loadRemoteMD(connection_document) == reloaded
    finally:
        if remote is not None:
            receiver.invalidateRemoteMD(remote)
        if reloaded is not None:
            receiver.invalidateRemoteMD(reloaded)
        source.deregisterMem(registration, [source_backend])
        utils.free_passthru(address)


def test_same_name_different_incarnation_is_rejected() -> None:
    """Reject a second live incarnation before changing the active session."""
    shared_name = f"shared-{uuid.uuid4()}"
    receiver = _agent(f"receiver-{uuid.uuid4()}")
    first_source = _agent(shared_name)
    second_source = _agent(shared_name)
    receiver.createBackend("UCX", {})
    first_backend = first_source.createBackend("UCX", {})
    second_backend = second_source.createBackend("UCX", {})
    empty = nixl.nixlRegDList(nixl.DRAM_SEG)
    first_document = first_source.getLocalPartialMD(empty, True, [first_backend])
    second_document = second_source.getLocalPartialMD(empty, True, [second_backend])
    remote = receiver.loadRemoteMD(first_document)

    try:
        try:
            receiver.loadRemoteMD(second_document)
        except nixl.nixlNotAllowedError:
            pass
        else:
            raise AssertionError("same-name metadata from a different incarnation was accepted")
        assert receiver.loadRemoteMD(first_document) == remote
    finally:
        receiver.invalidateRemoteMD(remote)


def test_conflicting_resource_preserves_active_handle() -> None:
    """Reject one conflicting resource while retaining the committed descriptor."""
    suffix = str(uuid.uuid4())
    source = _agent(f"source-{suffix}")
    receiver = _agent(f"receiver-{suffix}")
    source_backend = source.createBackend("UCX", {})
    receiver_backend = receiver.createBackend("UCX", {})
    size = 256
    address = utils.malloc_passthru(size)
    registration = nixl.nixlRegDList(nixl.DRAM_SEG)
    registration.addDesc((address, size, 0, ""))
    remote = None

    try:
        assert source.registerMem(registration, [source_backend]) == nixl.NIXL_SUCCESS
        empty = nixl.nixlRegDList(nixl.DRAM_SEG)
        remote = receiver.loadRemoteMD(
            source.getLocalPartialMD(empty, True, [source_backend])
        )
        valid_document = source.getLocalPartialMD(
            registration, False, [source_backend]
        )
        assert receiver.loadRemoteMD(valid_document) == remote

        conflicting_document = bytearray(valid_document)
        assert conflicting_document[-1] == ord("|")
        conflicting_document[-2] ^= 1
        try:
            receiver.loadRemoteMD(bytes(conflicting_document))
        except nixl.nixlNotAllowedError:
            pass
        else:
            raise AssertionError("conflicting descriptor metadata was accepted")

        assert receiver.loadRemoteMD(valid_document) == remote
        remote_descs = nixl.nixlXferDList(nixl.DRAM_SEG)
        remote_descs.addDesc((address, size, 0))
        prepared = receiver.prepXferDlist(remote, remote_descs, [receiver_backend])
        assert prepared != 0
        assert receiver.releasedDlistH(prepared) == nixl.NIXL_SUCCESS
    finally:
        if remote is not None:
            receiver.invalidateRemoteMD(remote)
        source.deregisterMem(registration, [source_backend])
        utils.free_passthru(address)
