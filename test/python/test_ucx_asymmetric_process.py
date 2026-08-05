# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import hashlib
import multiprocessing
import time
import traceback
import uuid
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess

import nixl._bindings as bindings
import nixl._utils as utils
from nixl._api import (
    nixl_agent,
    nixl_agent_config,
    nixl_remote_agent_handle,
    nixl_xfer_handle,
)


_SEGMENT_COUNT = 2048
_SEGMENT_LENGTH = 64
_SEGMENT_STRIDE = 128
_REGION_SIZE = (_SEGMENT_COUNT - 1) * _SEGMENT_STRIDE + _SEGMENT_LENGTH
_TIMEOUT_SECONDS = 30.0


def _wait_for_exact_notification(
    agent: nixl_agent,
    remote_handle: nixl_remote_agent_handle,
    expected_message: bytes,
) -> tuple[int, int]:
    """Wait for one notification delivered under the canonical remote handle.

    :param agent: Agent receiving the notification.
    :param remote_handle: Canonical handle that must own the notification.
    :param expected_message: Exact notification payload.
    :returns: Observed handle identity and generation.
    """
    deadline = time.monotonic() + _TIMEOUT_SECONDS
    observed_messages: list[bytes] = []
    while len(observed_messages) == 0:
        notifications = agent.get_new_notifs()
        for observed_handle, messages in notifications.items():
            if observed_handle is not remote_handle:
                raise AssertionError(
                    "notification was not resolved to the canonical remote handle"
                )
            observed_messages.extend(messages)
        if len(observed_messages) > 0:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("notification did not arrive before the deadline")
        time.sleep(0.001)

    if observed_messages != [expected_message]:
        raise AssertionError(
            f"unexpected notification payloads: {observed_messages!r}"
        )
    return remote_handle.identity, remote_handle.generation


def _send_notification_with_retry(
    agent: nixl_agent,
    remote_handle: nixl_remote_agent_handle,
    message: bytes,
) -> int:
    """Send after the exact capability route becomes ready.

    :param agent: Sending agent.
    :param remote_handle: Exact destination handle.
    :param message: Notification payload.
    :returns: Number of ``NOT_READY`` retries observed.
    """
    deadline = time.monotonic() + _TIMEOUT_SECONDS
    not_ready_count = 0
    while True:
        try:
            agent.send_notif(remote_handle, message)
            return not_ready_count
        except bindings.nixlNotReadyError as error:
            not_ready_count += 1
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "notification capability did not converge before the deadline"
                ) from error
            time.sleep(0.001)


def _target_process(connection: Connection, suffix: str) -> None:
    """Run the one-worker transfer target in an isolated process.

    :param connection: Parent control connection.
    :param suffix: Unique agent-name suffix.
    """
    target: nixl_agent | None = None
    initiator_handle: nixl_remote_agent_handle | None = None
    destination_registration: bindings.nixlRegDList | None = None
    destination = 0
    try:
        config = nixl_agent_config(
            enable_prog_thread=True,
            enable_listen_thread=False,
            num_threads=0,
            backends=["UCX"],
        )
        target = nixl_agent(f"asymmetric-target-{suffix}", config)
        destination = utils.malloc_passthru(_REGION_SIZE)
        ctypes.memset(destination, 0, _REGION_SIZE)
        destination_registration = target.register_memory(
            [(destination, _REGION_SIZE, 0, "destination")], mem_type="DRAM"
        )
        connection.send(("ready", target.get_agent_metadata(), destination))

        while True:
            command = connection.recv()
            if not isinstance(command, tuple) or len(command) == 0:
                raise TypeError("target command must be a non-empty tuple")
            command_name = command[0]
            if command_name == "verify_zero":
                contents = ctypes.string_at(destination, _REGION_SIZE)
                connection.send(
                    (
                        "zero_verified",
                        contents == bytes(_REGION_SIZE),
                        hashlib.sha256(contents).hexdigest(),
                    )
                )
                continue
            if command_name == "load_remote":
                if len(command) != 2 or not isinstance(command[1], bytes):
                    raise TypeError("load_remote requires metadata bytes")
                initiator_handle = target.add_remote_agent(command[1])
                connection.send(
                    (
                        "remote_loaded",
                        initiator_handle.identity,
                        initiator_handle.generation,
                    )
                )
                continue
            if command_name == "receive_standalone":
                if initiator_handle is None:
                    raise RuntimeError("initiator metadata has not been loaded")
                if len(command) != 2 or not isinstance(command[1], bytes):
                    raise TypeError("receive_standalone requires notification bytes")
                identity, generation = _wait_for_exact_notification(
                    target, initiator_handle, command[1]
                )
                connection.send(("standalone_received", identity, generation))
                continue
            if command_name == "send_standalone":
                if initiator_handle is None:
                    raise RuntimeError("initiator metadata has not been loaded")
                if len(command) != 2 or not isinstance(command[1], bytes):
                    raise TypeError("send_standalone requires notification bytes")
                not_ready_count = _send_notification_with_retry(
                    target, initiator_handle, command[1]
                )
                connection.send(("standalone_sent", not_ready_count))
                continue
            if command_name == "verify_transfer":
                if initiator_handle is None:
                    raise RuntimeError("initiator metadata has not been loaded")
                if (
                    len(command) != 3
                    or not isinstance(command[1], bytes)
                    or not isinstance(command[2], str)
                ):
                    raise TypeError(
                        "verify_transfer requires notification bytes and a digest"
                    )
                identity, generation = _wait_for_exact_notification(
                    target, initiator_handle, command[1]
                )
                contents = ctypes.string_at(destination, _REGION_SIZE)
                connection.send(
                    (
                        "transfer_verified",
                        hashlib.sha256(contents).hexdigest() == command[2],
                        identity,
                        generation,
                    )
                )
                continue
            if command_name == "shutdown":
                break
            raise ValueError(f"unknown target command: {command_name!r}")

        if initiator_handle is not None:
            target.remove_remote_agent(initiator_handle)
            initiator_handle = None
        if destination_registration is not None:
            target.deregister_memory(destination_registration)
            destination_registration = None
        utils.free_passthru(destination)
        destination = 0
        connection.send(("stopped",))
    except Exception:
        connection.send(("error", traceback.format_exc()))
    finally:
        if target is not None and initiator_handle is not None:
            target.remove_remote_agent(initiator_handle)
        if target is not None and destination_registration is not None:
            target.deregister_memory(destination_registration)
        if destination != 0:
            utils.free_passthru(destination)
        connection.close()


def _receive(connection: Connection, expected_kind: str) -> tuple[object, ...]:
    """Receive one bounded child response and surface child tracebacks.

    :param connection: Child control connection.
    :param expected_kind: Required response discriminator.
    :returns: Response payload after the discriminator.
    """
    if not connection.poll(_TIMEOUT_SECONDS):
        raise TimeoutError(f"timed out waiting for {expected_kind!r}")
    response = connection.recv()
    if not isinstance(response, tuple) or len(response) == 0:
        raise TypeError("target response must be a non-empty tuple")
    if response[0] == "error":
        raise AssertionError(f"target process failed:\n{response[1]}")
    if response[0] != expected_kind:
        raise AssertionError(
            f"expected target response {expected_kind!r}, got {response[0]!r}"
        )
    return response[1:]


def _post_transfer_with_retry(
    agent: nixl_agent, handle: nixl_xfer_handle
) -> tuple[str, int]:
    """Retry one unsubmitted handle until its capability route is ready.

    :param agent: Transfer initiator.
    :param handle: Transfer handle whose first post returned ``NOT_READY``.
    :returns: Posted state and number of additional ``NOT_READY`` results.
    """
    deadline = time.monotonic() + _TIMEOUT_SECONDS
    not_ready_count = 0
    while True:
        state = agent.transfer(handle)
        if state == "NOT_READY":
            not_ready_count += 1
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "transfer capability did not converge before the deadline"
                )
            time.sleep(0.001)
            continue
        if state == "ERR":
            raise AssertionError("transfer failed after capability convergence")
        return state, not_ready_count


def _wait_for_done(agent: nixl_agent, handle: nixl_xfer_handle) -> None:
    """Wait for a posted transfer to finish.

    :param agent: Transfer initiator.
    :param handle: Posted transfer handle.
    """
    deadline = time.monotonic() + _TIMEOUT_SECONDS
    while agent.check_xfer_state(handle) != "DONE":
        if time.monotonic() >= deadline:
            raise TimeoutError("transfer did not complete before the deadline")
        time.sleep(0.001)


def test_three_worker_initiator_to_one_worker_target() -> None:
    """Retry an exact-handle attached WRITE from three UCX workers to one."""
    suffix = str(uuid.uuid4())
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=True)
    target_process: BaseProcess = context.Process(
        target=_target_process,
        args=(child_connection, suffix),
        name=f"nixl-asymmetric-target-{suffix}",
    )
    target_process.start()
    child_connection.close()

    initiator: nixl_agent | None = None
    target_handle: nixl_remote_agent_handle | None = None
    transfer_handle: nixl_xfer_handle | None = None
    source_registration: bindings.nixlRegDList | None = None
    source = 0
    clean_shutdown = False
    try:
        ready = _receive(parent_connection, "ready")
        target_metadata, destination = ready
        if not isinstance(target_metadata, bytes) or not isinstance(destination, int):
            raise TypeError("target readiness payload is malformed")

        config = nixl_agent_config(
            enable_prog_thread=True,
            enable_listen_thread=False,
            num_threads=2,
            backends=["UCX"],
        )
        initiator = nixl_agent(f"asymmetric-initiator-{suffix}", config)
        source = utils.malloc_passthru(_REGION_SIZE)
        source_contents = bytes(range(256)) * (_REGION_SIZE // 256) + bytes(
            range(_REGION_SIZE % 256)
        )
        ctypes.memmove(source, source_contents, _REGION_SIZE)
        source_registration = initiator.register_memory(
            [(source, _REGION_SIZE, 0, "source")], mem_type="DRAM"
        )
        target_handle = initiator.add_remote_agent(target_metadata)
        if target_handle.identity <= 0 or target_handle.generation <= 0:
            raise AssertionError("initiator received an invalid exact target handle")

        source_descriptors = initiator.get_xfer_descs(
            [
                (source + index * _SEGMENT_STRIDE, _SEGMENT_LENGTH, 0)
                for index in range(_SEGMENT_COUNT)
            ],
            mem_type="DRAM",
        )
        destination_descriptors = initiator.get_xfer_descs(
            [
                (destination + index * _SEGMENT_STRIDE, _SEGMENT_LENGTH, 0)
                for index in range(_SEGMENT_COUNT)
            ],
            mem_type="DRAM",
        )
        transfer_notification = b"asymmetric-transfer-attached"
        transfer_handle = initiator.initialize_xfer(
            "WRITE",
            source_descriptors,
            destination_descriptors,
            target_handle,
            transfer_notification,
        )

        initial_state = initiator.transfer(transfer_handle)

        parent_connection.send(("verify_zero",))
        zero_verified = _receive(parent_connection, "zero_verified")
        expected_zero_digest = hashlib.sha256(bytes(_REGION_SIZE)).hexdigest()
        if initial_state != "NOT_READY":
            raise AssertionError(
                "transfer posted before the target admitted the initiator route: "
                f"state={initial_state}, zero_verification={zero_verified!r}"
            )
        if zero_verified != (True, expected_zero_digest):
            raise AssertionError(
                "destination bytes moved before notification capability readiness"
            )

        parent_connection.send(("load_remote", initiator.get_agent_metadata()))
        remote_loaded = _receive(parent_connection, "remote_loaded")
        initiator_identity, initiator_generation = remote_loaded
        if (
            not isinstance(initiator_identity, int)
            or not isinstance(initiator_generation, int)
            or initiator_identity <= 0
            or initiator_generation <= 0
        ):
            raise AssertionError("target received an invalid exact initiator handle")

        outbound_standalone = b"asymmetric-standalone-initiator-to-target"
        _send_notification_with_retry(
            initiator, target_handle, outbound_standalone
        )
        parent_connection.send(("receive_standalone", outbound_standalone))
        standalone_received = _receive(parent_connection, "standalone_received")
        if standalone_received != (initiator_identity, initiator_generation):
            raise AssertionError(
                "target standalone notification was attributed to the wrong handle"
            )

        inbound_standalone = b"asymmetric-standalone-target-to-initiator"
        parent_connection.send(("send_standalone", inbound_standalone))
        _receive(parent_connection, "standalone_sent")
        observed_identity, observed_generation = _wait_for_exact_notification(
            initiator, target_handle, inbound_standalone
        )
        if (
            observed_identity != target_handle.identity
            or observed_generation != target_handle.generation
        ):
            raise AssertionError(
                "initiator standalone notification was attributed to the wrong handle"
            )

        posted_state, _ = _post_transfer_with_retry(initiator, transfer_handle)
        if posted_state == "PROC":
            _wait_for_done(initiator, transfer_handle)
        receipt = initiator.take_xfer_completion_receipt(transfer_handle)
        if receipt is None:
            raise AssertionError("completed asymmetric transfer has no receipt")
        if len(receipt.segments) != _SEGMENT_COUNT:
            raise AssertionError("receipt does not cover every transferred segment")
        worker_ids = {endpoint.workerId for endpoint in receipt.endpoints}
        if len(receipt.endpoints) != 2 or len(worker_ids) != 2:
            raise AssertionError(
                "asymmetric transfer did not exercise both dedicated initiator workers"
            )

        expected_destination = bytearray(_REGION_SIZE)
        for index in range(_SEGMENT_COUNT):
            offset = index * _SEGMENT_STRIDE
            expected_destination[offset : offset + _SEGMENT_LENGTH] = source_contents[
                offset : offset + _SEGMENT_LENGTH
            ]
        expected_digest = hashlib.sha256(expected_destination).hexdigest()
        parent_connection.send(
            ("verify_transfer", transfer_notification, expected_digest)
        )
        transfer_verified = _receive(parent_connection, "transfer_verified")
        if transfer_verified != (
            True,
            initiator_identity,
            initiator_generation,
        ):
            raise AssertionError(
                "transfer bytes or exact-handle notification attribution were incorrect"
            )

        parent_connection.send(("shutdown",))
        _receive(parent_connection, "stopped")
        clean_shutdown = True
    finally:
        if transfer_handle is not None:
            transfer_handle.release()
        if initiator is not None and source_registration is not None:
            initiator.deregister_memory(source_registration)
        if initiator is not None and target_handle is not None:
            initiator.remove_remote_agent(target_handle)
        if source != 0:
            utils.free_passthru(source)
        if not clean_shutdown and target_process.is_alive():
            try:
                parent_connection.send(("shutdown",))
            except (BrokenPipeError, EOFError, OSError):
                pass
        target_process.join(timeout=10)
        if target_process.is_alive():
            target_process.terminate()
            target_process.join(timeout=10)
        parent_connection.close()
        if target_process.exitcode != 0:
            raise AssertionError(
                f"target process exited with status {target_process.exitcode}"
            )
