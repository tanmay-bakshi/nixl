# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
from pathlib import Path

import pytest
from terminal_ucx_receipt import seal_receipt, validate_receipt


def _inventory() -> dict[str, object]:
    """Build one clean native lifecycle inventory.

    :returns: Valid zero-inventory receipt.
    """
    return {
        "active_subscriptions": 0,
        "active_backend_producers": 0,
        "active_callback_slots": 0,
        "queued_continuations": 0,
        "queued_events": 0,
        "channel_closed": True,
        "fatal": "NONE",
    }


def _population(
    transport: str,
    engine: str,
    completion_mode: str,
) -> dict[str, object]:
    """Build one valid completion-population fixture.

    :param transport: Observed UCX transport.
    :param engine: UCX progress-engine path.
    :param completion_mode: Immediate or asynchronous population.
    :returns: Valid completion-population receipt.
    """
    digest = hashlib.sha256(
        f"{transport}:{engine}:{completion_mode}".encode()
    ).hexdigest()
    immediate = completion_mode == "immediate"
    return {
        "completion_mode": completion_mode,
        "byte_count": 4096,
        "destination_byte_count": 4096,
        "source_sha256": digest,
        "destination_sha256": digest,
        "bytes_verified": True,
        "terminal_status": "NIXL_SUCCESS",
        "terminal_event_count": 1,
        "attestation_state": "REMOTE_FLUSHED",
        "attestation_status": "NIXL_SUCCESS",
        "attestation_sha256": hashlib.sha256(
            f"attestation:{digest}".encode()
        ).hexdigest(),
        "completion_claimed": True,
        "take_once_second_status": "NIXL_ERR_NOT_ALLOWED",
        "selected_transports": [transport],
        "subscription_before_post": True,
        "notification_completion_timestamp_ns": 10,
        "terminal_native_timestamp_ns": 11,
        "drain_timestamp_ns": 12,
        "callbacks_before_return": 3 if immediate else 0,
        "callbacks_after_return": 0 if immediate else 3,
        "callback_count": 3,
    }


def _capability() -> dict[str, object]:
    """Build complete capability-transition evidence.

    :returns: Valid capability receipt.
    """
    return {
        "subscribe_before_ready": True,
        "snapshot_after_ready": True,
        "ready_state": "READY",
        "ready_epoch": 7,
        "epoch_transition_observed": True,
        "next_epoch": 8,
        "failed_state": "FAILED",
        "failed_subscription_terminal": True,
        "retired_state": "RETIRED",
        "retired_subscription_terminal": True,
    }


def _faults() -> dict[str, object]:
    """Build complete isolated failure-path evidence.

    :returns: Valid failure-path receipt.
    """
    return {
        "transfer_cancellation": {
            "terminal_status": "NIXL_ERR_CANCELED",
            "terminal_event_count": 1,
            "owner_woken": True,
        },
        "remote_failure": {
            "terminal_status": "NIXL_ERR_REMOTE_DISCONNECT",
            "terminal_event_count": 1,
            "owner_woken": True,
        },
        "notification_failure": {
            "terminal_status": "NIXL_ERR_REMOTE_DISCONNECT",
            "terminal_event_count": 1,
            "owner_woken": True,
            "data_remote_flushed_before_failure": True,
        },
        "queue_overflow": {
            "fatal": "QUEUE_OVERFLOW",
            "owner_woken": True,
            "admitted_event_preserved": True,
        },
        "shutdown_cancellation_drain": {
            "cancel_status": "NIXL_SUCCESS",
            "terminal_status": "NIXL_ERR_CANCELED",
            "terminal_event_count": 1,
            "owner_woken": True,
            "drained": True,
        },
    }


def _case(transport: str, engine: str, identity_seed: int) -> dict[str, object]:
    """Build one valid transport and engine coordinate.

    :param transport: Observed UCX transport.
    :param engine: UCX progress-engine path.
    :param identity_seed: Positive base for distinct native identities.
    :returns: Valid matrix-case receipt.
    """
    return {
        "transport": transport,
        "engine": engine,
        "memory_type": "DRAM",
        "endpoint_identities": [identity_seed, identity_seed + 1],
        "source_registration_identity": identity_seed + 2,
        "destination_registration_identity": identity_seed + 3,
        "completion_populations": [
            _population(transport, engine, "immediate"),
            _population(transport, engine, "asynchronous"),
        ],
        "capability": _capability(),
        "faults": _faults(),
        "shutdown": _inventory(),
    }


def _invocation(transport: str, engine: str) -> dict[str, object]:
    """Build one exact qualification invocation.

    :param transport: Requested UCX transport.
    :param engine: Requested progress engine.
    :returns: Valid invocation receipt.
    """
    return {
        "transport": transport,
        "engine": engine,
        "argv": [
            "/workspace/build/terminal_ucx_qualification",
            "--transport",
            transport,
            "--engine",
            engine,
        ],
        "environment": {
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "void",
            "UCX_TLS": transport,
            "UCX_NET_DEVICES": "lo" if transport == "tcp" else None,
        },
    }


def _receipt() -> dict[str, object]:
    """Build a complete valid receipt fixture.

    :returns: Valid receipt object.
    """
    coordinates = [
        ("self", "shared"),
        ("self", "thread_pool"),
        ("tcp", "shared"),
        ("tcp", "thread_pool"),
    ]
    return {
        "schema": "nixl-terminal-ucx-qualification/v2",
        "status": "pass",
        "nixl_revision": "1" * 40,
        "ucx_revision": "2" * 40,
        "executable_sha256": "3" * 64,
        "invocations": [
            _invocation(transport, engine) for transport, engine in coordinates
        ],
        "runtime_artifacts": [
            {"component": "libnixl", "path": "/tmp/libnixl.so", "build_id": "aa"},
            {"component": "libucp", "path": "/tmp/libucp.so", "build_id": "bb"},
            {
                "component": "ucx-plugin",
                "path": "/tmp/libplugin_UCX.so",
                "build_id": "cc",
            },
        ],
        "zero_gpu": {
            "cuda_visible_devices": "",
            "nvidia_visible_devices": "void",
            "nvidia_device_open_count": 0,
            "driver_client_delta": [],
            "gpu_api_used": False,
        },
        "cases": [
            _case(transport, engine, 100 + index * 10)
            for index, (transport, engine) in enumerate(coordinates)
        ],
        "shutdown": _inventory(),
    }


def _replace_path(
    receipt: dict[str, object],
    path: tuple[str | int, ...],
    value: object,
) -> None:
    """Replace one nested fixture value.

    :param receipt: Mutable receipt fixture.
    :param path: Nested dictionary and list path.
    :param value: Replacement value.
    """
    target: object = receipt
    for component in path[:-1]:
        target = target[component]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]


def test_validate_receipt_accepts_complete_matrix() -> None:
    """Accept the exact bounded qualification matrix."""
    validate_receipt(_receipt())


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("zero_gpu", "nvidia_device_open_count"), 1),
        (("shutdown", "active_callback_slots"), 1),
        (("cases", 0, "shutdown", "queued_continuations"), 1),
        (("cases", 0, "endpoint_identities"), [100, 100]),
        (("cases", 0, "destination_registration_identity"), 102),
        (("cases", 0, "completion_populations", 0, "destination_sha256"), "0" * 64),
        (("cases", 0, "completion_populations", 0, "selected_transports"), ["tcp"]),
        (
            ("cases", 0, "completion_populations", 0, "take_once_second_status"),
            "NIXL_SUCCESS",
        ),
        (
            (
                "cases",
                0,
                "completion_populations",
                0,
                "notification_completion_timestamp_ns",
            ),
            12,
        ),
        (("cases", 0, "completion_populations", 0, "callbacks_before_return"), 0),
        (("cases", 0, "completion_populations", 1, "callbacks_after_return"), 0),
        (("cases", 0, "capability", "failed_subscription_terminal"), False),
        (("cases", 0, "capability", "next_epoch"), 7),
        (("cases", 0, "faults", "remote_failure", "owner_woken"), False),
        (
            ("cases", 0, "faults", "notification_failure", "terminal_status"),
            "NIXL_SUCCESS",
        ),
        (("cases", 0, "faults", "queue_overflow", "fatal"), "NONE"),
        (("runtime_artifacts", 0, "build_id"), "not-hex"),
    ],
)
def test_validate_receipt_rejects_false_authority(
    path: tuple[str | int, ...], value: object
) -> None:
    """Reject evidence which weakens a checkpoint invariant.

    :param path: Nested fixture path to corrupt.
    :param value: Contradictory value.
    """
    receipt = copy.deepcopy(_receipt())
    _replace_path(receipt, path, value)
    with pytest.raises(ValueError):
        validate_receipt(receipt)


def test_validate_receipt_rejects_old_schema() -> None:
    """Forbid the permissive predecessor schema from sealing Stage 1."""
    receipt = _receipt()
    receipt["schema"] = "nixl-terminal-ucx-qualification/v1"
    with pytest.raises(ValueError, match="schema"):
        validate_receipt(receipt)


def test_validate_receipt_rejects_non_object_runtime_artifact() -> None:
    """Report malformed runtime evidence as validation failure."""
    receipt = _receipt()
    receipt["runtime_artifacts"] = [None]
    with pytest.raises(ValueError, match="runtime artifact"):
        validate_receipt(receipt)


def test_validate_receipt_rejects_duplicate_coordinate() -> None:
    """Require every transport and engine coordinate exactly once."""
    receipt = _receipt()
    cases = receipt["cases"]
    assert isinstance(cases, list)
    cases[-1] = copy.deepcopy(cases[0])
    with pytest.raises(ValueError, match="coordinates"):
        validate_receipt(receipt)


def test_seal_receipt_is_deterministic(tmp_path: Path) -> None:
    """Canonicalize and digest identical receipts deterministically.

    :param tmp_path: Pytest-provided temporary directory.
    """
    source = tmp_path / "native.json"
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    source.write_text(json.dumps(_receipt()), encoding="utf-8")
    seal_receipt(source, first)
    seal_receipt(source, second)
    assert first.read_bytes() == second.read_bytes()
    assert (
        first.with_suffix(".json.sha256").read_text().split()[0]
        == second.with_suffix(".json.sha256").read_text().split()[0]
    )
