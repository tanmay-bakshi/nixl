# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
from pathlib import Path

import pytest
from terminal_ucx_receipt import seal_receipt, validate_receipt


def _case(transport: str, engine: str, completion_mode: str) -> dict[str, object]:
    """Build one valid case fixture.

    :param transport: Observed UCX transport.
    :param engine: UCX progress engine path.
    :param completion_mode: Immediate or asynchronous completion population.
    :returns: Valid case receipt.
    """
    digest = hashlib.sha256(f"{transport}:{engine}".encode()).hexdigest()
    return {
        "transport": transport,
        "engine": engine,
        "completion_mode": completion_mode,
        "memory_type": "DRAM",
        "byte_count": 4096,
        "source_sha256": digest,
        "destination_sha256": digest,
        "bytes_verified": True,
        "terminal_status": "NIXL_SUCCESS",
        "attestation_state": "REMOTE_FLUSHED",
        "attestation_status": "NIXL_SUCCESS",
        "completion_claimed": True,
        "take_once_second_status": "NIXL_ERR_NOT_ALLOWED",
        "selected_transports": [transport],
        "terminal_native_timestamp_ns": 10,
        "drain_timestamp_ns": 11,
        "subscription_before_post": True,
        "terminal_after_notification_completion": True,
        "callback_before_return_observed": transport == "self" and engine == "shared",
        "capability_snapshot_ready": True,
        "capability_retired": True,
    }


def _receipt() -> dict[str, object]:
    """Build a complete valid receipt fixture.

    :returns: Valid receipt object.
    """
    return {
        "schema": "nixl-terminal-ucx-qualification/v1",
        "status": "pass",
        "nixl_revision": "1" * 40,
        "ucx_revision": "2" * 40,
        "executable_sha256": "3" * 64,
        "commands": [
            ["terminal_ucx_qualification", "--transport", "self", "--engine", "shared"],
            [
                "terminal_ucx_qualification",
                "--transport",
                "self",
                "--engine",
                "thread_pool",
            ],
            ["terminal_ucx_qualification", "--transport", "tcp", "--engine", "shared"],
            [
                "terminal_ucx_qualification",
                "--transport",
                "tcp",
                "--engine",
                "thread_pool",
            ],
        ],
        "environment": {
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "void",
        },
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
            _case("self", "shared", "immediate"),
            _case("self", "thread_pool", "asynchronous"),
            _case("tcp", "shared", "asynchronous"),
            _case("tcp", "thread_pool", "asynchronous"),
        ],
        "shutdown": {
            "active_subscriptions": 0,
            "active_producers": 0,
            "queued_events": 0,
            "channel_closed": True,
            "fatal": "NONE",
        },
    }


def test_validate_receipt_accepts_complete_matrix() -> None:
    """Accept the exact bounded qualification matrix."""
    validate_receipt(_receipt())


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("zero_gpu", "nvidia_device_open_count"), 1),
        (("shutdown", "active_subscriptions"), 1),
        (("shutdown", "active_producers"), 1),
        (("cases", 0, "destination_sha256"), "0" * 64),
        (("cases", 0, "selected_transports"), ["tcp"]),
        (("cases", 0, "take_once_second_status"), "NIXL_SUCCESS"),
        (("cases", 0, "terminal_after_notification_completion"), False),
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
    target: object = receipt
    for component in path[:-1]:
        target = target[component]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    with pytest.raises(ValueError):
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
