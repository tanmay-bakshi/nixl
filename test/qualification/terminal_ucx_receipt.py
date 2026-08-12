# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
from pathlib import Path

_SCHEMA = "nixl-terminal-ucx-qualification/v1"
_TRANSPORTS = {"self", "tcp"}
_ENGINES = {"shared", "thread_pool"}


def _require(condition: bool, message: str) -> None:
    """Raise when one qualification invariant is not satisfied.

    :param condition: Condition which must be true.
    :param message: Validation error message.
    :raises ValueError: If ``condition`` is false.
    """
    if not condition:
        raise ValueError(message)


def _validate_case(case: dict[str, object]) -> None:
    """Validate one real-UCX matrix case.

    :param case: Case receipt emitted by the native executable.
    :raises ValueError: If the case is incomplete or contradictory.
    """
    _require(case.get("transport") in _TRANSPORTS, "unknown observed transport")
    _require(case.get("engine") in _ENGINES, "unknown engine path")
    _require(case.get("memory_type") == "DRAM", "qualification used non-DRAM memory")
    _require(
        case.get("source_sha256") == case.get("destination_sha256"), "digest mismatch"
    )
    _require(int(case.get("byte_count", 0)) > 0, "case transferred no bytes")
    _require(case.get("bytes_verified") is True, "destination bytes were not verified")
    _require(case.get("terminal_status") == "NIXL_SUCCESS", "transfer did not succeed")
    _require(
        case.get("attestation_state") == "REMOTE_FLUSHED", "receipt preceded flush"
    )
    _require(case.get("attestation_status") == "NIXL_SUCCESS", "attestation failed")
    _require(
        case.get("completion_claimed") is True, "completion authority was not claimed"
    )
    _require(
        case.get("take_once_second_status") == "NIXL_ERR_NOT_ALLOWED",
        "receipt was reusable",
    )
    _require(
        case.get("selected_transports") == [case.get("transport")],
        "transport intent was not observed",
    )
    _require(
        int(case.get("terminal_native_timestamp_ns", 0)) > 0,
        "terminal timestamp is missing",
    )
    _require(
        int(case.get("drain_timestamp_ns", 0))
        >= int(case.get("terminal_native_timestamp_ns", 0)),
        "terminal event was drained before publication",
    )
    _require(
        case.get("subscription_before_post") is True,
        "subscription was not armed before post",
    )
    _require(
        case.get("notification_after_terminal") is True, "notification ordering failed"
    )
    _require(
        case.get("capability_snapshot_ready") is True,
        "ready snapshot race was not covered",
    )
    _require(
        case.get("capability_retired") is True, "route retirement was not observed"
    )


def validate_receipt(receipt: dict[str, object]) -> None:
    """Validate one complete zero-GPU UCX qualification receipt.

    :param receipt: Parsed receipt object.
    :raises ValueError: If any checkpoint invariant is absent or false.
    """
    _require(receipt.get("schema") == _SCHEMA, "unexpected receipt schema")
    _require(receipt.get("status") == "pass", "native qualification did not pass")
    _require(
        isinstance(receipt.get("nixl_revision"), str)
        and len(str(receipt["nixl_revision"])) == 40,
        "invalid NIXL revision",
    )
    _require(
        isinstance(receipt.get("ucx_revision"), str)
        and len(str(receipt["ucx_revision"])) == 40,
        "invalid UCX revision",
    )
    _require(
        isinstance(receipt.get("executable_sha256"), str)
        and len(str(receipt["executable_sha256"])) == 64,
        "invalid executable digest",
    )

    zero_gpu = receipt.get("zero_gpu")
    _require(isinstance(zero_gpu, dict), "zero-GPU evidence is missing")
    _require(zero_gpu.get("cuda_visible_devices") == "", "GPU visibility was not empty")
    _require(
        zero_gpu.get("nvidia_visible_devices") == "void",
        "container GPU visibility was not void",
    )
    _require(
        zero_gpu.get("nvidia_device_open_count") == 0, "a NVIDIA device was opened"
    )
    _require(
        zero_gpu.get("driver_client_delta") == [],
        "NVIDIA driver client inventory changed",
    )
    _require(
        zero_gpu.get("gpu_api_used") is False, "GPU API was used to construct evidence"
    )

    cases = receipt.get("cases")
    _require(isinstance(cases, list), "case matrix is missing")
    _require(
        len(cases) == len(_TRANSPORTS) * len(_ENGINES), "case matrix is incomplete"
    )
    observed = set()
    for case in cases:
        _require(isinstance(case, dict), "case receipt is not an object")
        _validate_case(case)
        observed.add((case["transport"], case["engine"]))
    _require(
        observed
        == {(transport, engine) for transport in _TRANSPORTS for engine in _ENGINES},
        "matrix coordinates are not unique and complete",
    )
    _require(
        {case["completion_mode"] for case in cases} == {"immediate", "asynchronous"},
        "completion populations are incomplete",
    )

    shutdown = receipt.get("shutdown")
    _require(isinstance(shutdown, dict), "shutdown inventory is missing")
    _require(shutdown.get("active_subscriptions") == 0, "subscriptions remain active")
    _require(shutdown.get("active_producers") == 0, "native producers remain active")
    _require(shutdown.get("queued_events") == 0, "terminal events remain queued")
    _require(shutdown.get("channel_closed") is True, "terminal channel was not closed")
    _require(
        shutdown.get("fatal") == "NONE", "terminal channel reported a fatal condition"
    )


def seal_receipt(input_path: Path, output_path: Path) -> None:
    """Validate, canonicalize, and digest a native receipt.

    :param input_path: Native JSON receipt path.
    :param output_path: Destination for the sealed receipt.
    """
    receipt = json.loads(input_path.read_text(encoding="utf-8"))
    validate_receipt(receipt)
    canonical = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    output_path.write_text(canonical, encoding="utf-8")
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{digest}  {output_path.name}\n", encoding="utf-8"
    )


def main() -> None:
    """Validate and seal a native qualification receipt."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    seal_receipt(arguments.input, arguments.output)


if __name__ == "__main__":
    main()
