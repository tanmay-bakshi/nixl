# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
from pathlib import Path

import pytest
from terminal_ucx_receipt import seal_receipt, validate_receipt

_SELF_REASON = (
    "ucx_self_is_same_worker_only_and_nixl_local_routes_have_no_remote_agent_handle"
)
_SELF_ANCHORS = [
    "ucx/src/uct/sm/self/self.c:144",
    "ucx/src/ucp/core/ucp_ep.c:1098",
    "nixl/src/core/nixl_agent.cpp:2449",
]


def _not_applicable() -> dict[str, object]:
    """Build one structural self-transport non-applicability record.

    :returns: Typed non-applicability evidence.
    """
    return {
        "applicability": "not_applicable",
        "reason": _SELF_REASON,
        "evidence_anchors": list(_SELF_ANCHORS),
    }


def _inventory() -> dict[str, object]:
    """Build one clean public lifecycle inventory.

    :returns: Valid zero-inventory receipt.
    """
    return {
        "capacity": 64,
        "queued_channel_events": 0,
        "active_channel_subscriptions": 0,
        "retained_public_subscriptions": 0,
        "backend_producers": 0,
        "active_callback_slots": 0,
        "queued_owner_continuations": 0,
        "accepting_subscriptions": False,
        "closed": True,
        "fatal": "NONE",
        "eventfd_error": 0,
    }


def _terminal_progress(
    transport: str, population: str, endpoint_count: int
) -> dict[str, object]:
    """Build native terminal-progress evidence.

    :param transport: Self or TCP coordinate.
    :param population: Small or large population.
    :param endpoint_count: Number of endpoint flushes in the attestation.
    :returns: Conserved callback and ordering observations.
    """
    self_transport = transport == "self"
    small = population == "small"
    data_callbacks = 1 if small else 4
    flush_callbacks = 0 if self_transport else endpoint_count
    notification_callbacks = 0 if self_transport else 1
    callback_count = data_callbacks + flush_callbacks + notification_callbacks
    before_return = data_callbacks if self_transport else 0
    return {
        "autonomous": True,
        "data_callbacks": data_callbacks,
        "endpoint_flush_callbacks": flush_callbacks,
        "notification_callbacks": notification_callbacks,
        "asynchronous_requests": 0 if self_transport else callback_count,
        "immediate_completions": (
            data_callbacks + endpoint_count if self_transport else 0
        ),
        "callbacks_before_poster_return": before_return,
        "peak_continuation_depth": 2,
        "active_callback_slots_at_terminal": 0,
        "continuation_depth_at_terminal": 0,
        "last_data_callback_timestamp_ns": 10,
        "last_flush_callback_timestamp_ns": 0 if self_transport else 11,
        "notification_callback_timestamp_ns": 0 if self_transport else 12,
        "terminal_publish_timestamp_ns": 13,
        "terminal_status": "NIXL_SUCCESS",
    }


def _population(
    transport: str,
    engine: str,
    population: str,
    handle_identity: int,
    generation: int,
) -> dict[str, object]:
    """Build one valid completion population.

    :param transport: Observed UCX transport.
    :param engine: Progress-engine path.
    :param population: Small or large population.
    :param handle_identity: Transfer-request identity in the attestation.
    :param generation: Transfer generation in the attestation.
    :returns: Valid population receipt.
    """
    digest = hashlib.sha256(f"{transport}:{engine}:{population}".encode()).hexdigest()
    byte_count = 4096 if population == "small" else 16 * 1024 * 1024
    endpoint_count = 2 if engine == "thread_pool" and population != "small" else 1
    endpoint_flushes = [
        {
            "worker_id": index,
            "worker_identity": handle_identity * 10 + index,
            "endpoint_identity": handle_identity * 100 + index,
            "flush_posted": True,
            "remote_flushed": True,
        }
        for index in range(endpoint_count)
    ]
    return {
        "population": population,
        "descriptor_count": 1 if population == "small" else 4,
        "byte_count": byte_count,
        "destination_byte_count": byte_count,
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
        "attestation_handle_identity": handle_identity,
        "attestation_generation": generation,
        "completion_claimed": True,
        "take_once_second_status": "NIXL_ERR_NOT_ALLOWED",
        "selected_transports": [transport],
        "endpoint_flushes": endpoint_flushes,
        "subscription_before_post": True,
        "event_native_timestamp_ns": 13,
        "drain_timestamp_ns": 14,
        "terminal_progress": _terminal_progress(transport, population, endpoint_count),
    }


def _capability_route(name: str, identity: int) -> dict[str, object]:
    """Build one exact-route lifecycle record.

    :param name: Epoch, failure, or retirement route.
    :param identity: Distinct handle identity.
    :returns: Valid route lifecycle evidence.
    """
    states = {
        "epoch_advance": ["READY", "READY"],
        "endpoint_failure": ["READY", "FAILED"],
        "retirement": ["READY", "RETIRED"],
    }[name]
    return {
        "name": name,
        "handle_identity": identity,
        "handle_generation": 1,
        "states": states,
        "epochs": [7, 8] if name == "epoch_advance" else [7, 7],
        "subscription_terminal": states[-1] in {"FAILED", "RETIRED"},
        "release_status": "NIXL_SUCCESS",
    }


def _capability() -> dict[str, object]:
    """Build complete TCP capability-transition evidence.

    :returns: Valid distinct-route capability receipt.
    """
    return {
        "applicability": "applicable",
        "subscribe_before_ready": True,
        "snapshot_after_ready": True,
        "routes": [
            _capability_route("epoch_advance", 501),
            _capability_route("endpoint_failure", 502),
            _capability_route("retirement", 503),
        ],
    }


def _terminal_fault(status: str) -> dict[str, object]:
    """Build one exactly-once terminal failure record.

    :param status: Required NIXL terminal status.
    :returns: Valid failure observation.
    """
    return {
        "applicability": "applicable",
        "terminal_status": status,
        "terminal_event_count": 1,
        "owner_woken": True,
    }


def _faults(transport: str) -> dict[str, object]:
    """Build transport-aware failure-path evidence.

    :param transport: Self or TCP coordinate.
    :returns: Complete common and remote failure receipt.
    """
    cancellation = _terminal_fault("NIXL_ERR_CANCELED")
    shutdown = dict(cancellation)
    shutdown.update(
        {
            "cancel_status": "NIXL_SUCCESS",
            "posted_in_flight": transport == "tcp",
            "backend_producers_before_cancel": 1 if transport == "tcp" else 0,
            "active_callback_slots_before_cancel": 2 if transport == "tcp" else 0,
            "queued_owner_continuations_before_cancel": 0,
            "drained": True,
        }
    )
    faults: dict[str, object] = {
        "transfer_cancellation": cancellation,
        "queue_overflow": {
            "applicability": "applicable",
            "fatal": "QUEUE_OVERFLOW",
            "owner_woken": True,
            "admitted_event_preserved": True,
        },
        "shutdown_cancellation_drain": shutdown,
    }
    if transport == "self":
        faults["remote_failure"] = _not_applicable()
        faults["notification_failure"] = _not_applicable()
        return faults
    faults["remote_failure"] = _terminal_fault("NIXL_ERR_REMOTE_DISCONNECT")
    notification_failure = _terminal_fault("NIXL_ERR_REMOTE_DISCONNECT")
    notification_failure["data_remote_flushed_before_failure"] = True
    faults["notification_failure"] = notification_failure
    return faults


def _case(transport: str, engine: str, address_seed: int) -> dict[str, object]:
    """Build one transport-aware coordinate.

    :param transport: Self or TCP transport.
    :param engine: Shared or thread-pool progress engine.
    :param address_seed: Positive base for distinct buffer addresses.
    :returns: Valid matrix case.
    """
    self_transport = transport == "self"
    populations = [
        _population(transport, engine, "small", address_seed + 1, 1),
        _population(transport, engine, "large", address_seed + 2, 1),
    ]
    if engine == "thread_pool":
        repost_identity = address_seed + 3
        populations.extend(
            [
                _population(
                    transport,
                    engine,
                    "thread_pool_repost_generation_1",
                    repost_identity,
                    7,
                ),
                _population(
                    transport,
                    engine,
                    "thread_pool_repost_generation_2",
                    repost_identity,
                    8,
                ),
            ]
        )
    return {
        "transport": transport,
        "engine": engine,
        "agent_shape": (
            "one_agent_local_route" if self_transport else "two_distinct_agents"
        ),
        "agent_count": 1 if self_transport else 2,
        "remote_agent_handle_present": not self_transport,
        "registrations": {
            "source": {
                "status": "NIXL_SUCCESS",
                "memory_type": "DRAM",
                "base_address": address_seed,
                "byte_capacity": 16 * 1024 * 1024,
            },
            "destination": {
                "status": "NIXL_SUCCESS",
                "memory_type": "DRAM",
                "base_address": address_seed + 4096,
                "byte_capacity": 16 * 1024 * 1024,
            },
        },
        "completion_populations": populations,
        "remote_route_capability": (
            _not_applicable() if self_transport else _capability()
        ),
        "attached_authenticated_notification": (
            _not_applicable()
            if self_transport
            else {"applicability": "applicable", "success_count": 2}
        ),
        "faults": _faults(transport),
        "runtime_artifacts": [
            {"component": "libnixl", "path": "/tmp/libnixl.so", "build_id": "aa"},
            {"component": "libucp", "path": "/tmp/libucp.so", "build_id": "bb"},
            {
                "component": "ucx-plugin",
                "path": "/tmp/libplugin_UCX.so",
                "build_id": "cc",
            },
        ],
        "shutdown": _inventory(),
    }


def _invocation(transport: str, engine: str) -> dict[str, object]:
    """Build one exact qualification invocation.

    :param transport: Requested transport.
    :param engine: Requested progress engine.
    :returns: Valid invocation receipt.
    """
    argv = [
        "/workspace/build/terminal_ucx_qualification",
        "--transport",
        transport,
        "--engine",
        engine,
    ]
    return {
        "transport": transport,
        "engine": engine,
        "argv": argv,
        "trace_argv": ["/usr/bin/strace", "-f", *argv],
        "environment": {
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "void",
            "UCX_TLS": transport,
            "UCX_NET_DEVICES": "lo" if transport == "tcp" else None,
            "NIXL_PLUGIN_DIR": "/workspace/build/src/plugins",
            "LD_LIBRARY_PATH": "/workspace/build/src/core",
        },
    }


def _receipt() -> dict[str, object]:
    """Build a complete schema-v4 receipt.

    :returns: Valid qualification receipt.
    """
    coordinates = [
        ("self", "shared"),
        ("self", "thread_pool"),
        ("tcp", "shared"),
        ("tcp", "thread_pool"),
    ]
    cases = [
        _case(transport, engine, 100_000 + index * 100_000)
        for index, (transport, engine) in enumerate(coordinates)
    ]
    return {
        "schema": "nixl-terminal-ucx-qualification/v4",
        "status": "pass",
        "nixl_revision": "1" * 40,
        "ucx_revision": "2" * 40,
        "executable_sha256": "3" * 64,
        "invocations": [
            _invocation(transport, engine) for transport, engine in coordinates
        ],
        "runtime_artifacts": cases[0]["runtime_artifacts"],
        "zero_gpu": {
            "cuda_visible_devices": "",
            "nvidia_visible_devices": "void",
            "nvidia_device_open_count": 0,
            "driver_client_delta": [],
            "gpu_api_used": False,
            "evidence_sources": ["strace", "/proc/<pid>/fd"],
        },
        "cases": cases,
        "shutdown": _inventory(),
    }


def _replace_path(
    receipt: dict[str, object],
    path: tuple[str | int, ...],
    value: object,
) -> None:
    """Replace one nested fixture value.

    :param receipt: Mutable fixture.
    :param path: Nested dictionary and list path.
    :param value: Replacement value.
    """
    target: object = receipt
    for component in path[:-1]:
        target = target[component]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]


def test_validate_receipt_accepts_transport_aware_matrix() -> None:
    """Accept the exact frozen Stage-1 coverage split."""
    validate_receipt(_receipt())


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("zero_gpu", "nvidia_device_open_count"), 1),
        (("invocations", 0, "environment", "NIXL_PLUGIN_DIR"), "relative/plugins"),
        (("cases", 0, "agent_count"), 2),
        (("cases", 0, "remote_agent_handle_present"), True),
        (("cases", 0, "registrations", "destination", "base_address"), 100_000),
        (("cases", 0, "remote_route_capability", "applicability"), "applicable"),
        (("cases", 0, "remote_route_capability", "fabricated_pass"), True),
        (("cases", 0, "completion_populations", 0, "selected_transports"), ["tcp"]),
        (("cases", 0, "completion_populations", 0, "destination_sha256"), "0" * 64),
        (
            (
                "cases",
                0,
                "completion_populations",
                0,
                "terminal_progress",
                "endpoint_flush_callbacks",
            ),
            1,
        ),
        (
            (
                "cases",
                0,
                "completion_populations",
                0,
                "terminal_progress",
                "callbacks_before_poster_return",
            ),
            99,
        ),
        (
            (
                "cases",
                0,
                "completion_populations",
                0,
                "terminal_progress",
                "immediate_completions",
            ),
            1,
        ),
        (
            (
                "cases",
                0,
                "completion_populations",
                0,
                "endpoint_flushes",
                0,
                "remote_flushed",
            ),
            False,
        ),
        (
            (
                "cases",
                1,
                "completion_populations",
                2,
                "population",
            ),
            "thread_pool_repost_generation_2",
        ),
        (
            (
                "cases",
                1,
                "completion_populations",
                3,
                "attestation_handle_identity",
            ),
            999_999,
        ),
        (
            (
                "cases",
                1,
                "completion_populations",
                3,
                "attestation_generation",
            ),
            7,
        ),
        (
            (
                "cases",
                2,
                "completion_populations",
                0,
                "terminal_progress",
                "endpoint_flush_callbacks",
            ),
            0,
        ),
        (
            (
                "cases",
                2,
                "completion_populations",
                0,
                "terminal_progress",
                "notification_callbacks",
            ),
            0,
        ),
        (("cases", 2, "remote_route_capability", "routes", 1, "states"), ["READY"]),
        (("cases", 2, "remote_route_capability", "routes", 2, "handle_identity"), 502),
        (("cases", 2, "faults", "remote_failure", "owner_woken"), False),
        (
            ("cases", 2, "faults", "shutdown_cancellation_drain", "posted_in_flight"),
            False,
        ),
        (("cases", 2, "shutdown", "active_callback_slots"), 1),
        (("runtime_artifacts", 0, "build_id"), "not-hex"),
    ],
)
def test_validate_receipt_rejects_false_authority(
    path: tuple[str | int, ...], value: object
) -> None:
    """Reject evidence which weakens the frozen authority split.

    :param path: Nested path to corrupt.
    :param value: Contradictory replacement.
    """
    receipt = copy.deepcopy(_receipt())
    _replace_path(receipt, path, value)
    with pytest.raises(ValueError):
        validate_receipt(receipt)


def test_validate_receipt_rejects_old_schema() -> None:
    """Forbid sealing the physically over-constrained predecessor schema."""
    receipt = _receipt()
    receipt["schema"] = "nixl-terminal-ucx-qualification/v3"
    with pytest.raises(ValueError, match="schema"):
        validate_receipt(receipt)


def test_validate_receipt_rejects_remote_na_on_tcp() -> None:
    """Require TCP to exercise rather than waive remote semantics."""
    receipt = _receipt()
    cases = receipt["cases"]
    assert isinstance(cases, list)
    tcp_case = cases[2]
    assert isinstance(tcp_case, dict)
    tcp_case["remote_route_capability"] = _not_applicable()
    with pytest.raises(ValueError, match="capability"):
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
