# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
from pathlib import Path

_SCHEMA = "nixl-terminal-ucx-qualification/v2"
_TRANSPORTS = {"self", "tcp"}
_ENGINES = {"shared", "thread_pool"}
_COMPLETION_MODES = {"immediate", "asynchronous"}
_RUNTIME_COMPONENTS = {"libnixl", "libucp", "ucx-plugin"}
_ZERO_INVENTORY_FIELDS = {
    "active_subscriptions",
    "active_backend_producers",
    "active_callback_slots",
    "queued_continuations",
    "queued_events",
}


def _require(condition: bool, message: str) -> None:
    """Raise when one qualification invariant is not satisfied.

    :param condition: Condition which must be true.
    :param message: Validation error message.
    :raises ValueError: If ``condition`` is false.
    """
    if not condition:
        raise ValueError(message)


def _is_positive_int(value: object) -> bool:
    """Return whether a value is a positive integer rather than a boolean.

    :param value: Value to inspect.
    :returns: Whether the value is a positive integer.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonnegative_int(value: object) -> bool:
    """Return whether a value is a nonnegative integer rather than a boolean.

    :param value: Value to inspect.
    :returns: Whether the value is a nonnegative integer.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_sha256(value: object) -> bool:
    """Return whether a value is a lowercase hexadecimal SHA-256 digest.

    :param value: Value to inspect.
    :returns: Whether the value is a canonical digest.
    """
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _validate_zero_inventory(inventory: object, context: str) -> None:
    """Validate one clean native lifecycle inventory.

    :param inventory: Inventory object emitted after draining native work.
    :param context: Human-readable validation context.
    :raises ValueError: If resources or fatal state remain.
    """
    _require(isinstance(inventory, dict), f"{context} inventory is missing")
    for field in _ZERO_INVENTORY_FIELDS:
        _require(inventory.get(field) == 0, f"{context} retained {field}")
    _require(inventory.get("channel_closed") is True, f"{context} channel stayed open")
    _require(inventory.get("fatal") == "NONE", f"{context} reported a fatal condition")


def _validate_invocation(invocation: object) -> tuple[str, str]:
    """Validate one exact real-UCX process invocation.

    :param invocation: Invocation receipt.
    :returns: Transport and engine coordinate.
    :raises ValueError: If argv or relevant environment is incomplete.
    """
    _require(isinstance(invocation, dict), "invocation is not an object")
    transport = invocation.get("transport")
    engine = invocation.get("engine")
    _require(transport in _TRANSPORTS, "invocation transport is invalid")
    _require(engine in _ENGINES, "invocation engine is invalid")

    argv = invocation.get("argv")
    _require(isinstance(argv, list) and len(argv) > 0, "invocation argv is missing")
    _require(
        all(isinstance(argument, str) and len(argument) > 0 for argument in argv),
        "invocation argv contains an invalid argument",
    )
    _require(
        Path(str(argv[0])).is_absolute(),
        "qualification executable path is not absolute",
    )

    environment = invocation.get("environment")
    _require(isinstance(environment, dict), "invocation environment is missing")
    _require(
        environment.get("CUDA_VISIBLE_DEVICES") == "", "CUDA visibility was not empty"
    )
    _require(
        environment.get("NVIDIA_VISIBLE_DEVICES") == "void",
        "NVIDIA visibility was not void",
    )
    _require(
        environment.get("UCX_TLS") == transport, "UCX_TLS does not match transport"
    )
    expected_device = "lo" if transport == "tcp" else None
    _require(
        environment.get("UCX_NET_DEVICES") == expected_device,
        "UCX_NET_DEVICES does not match transport",
    )
    return str(transport), str(engine)


def _validate_completion_population(population: object, transport: str) -> str:
    """Validate one immediate or asynchronous completion population.

    :param population: Population receipt.
    :param transport: Observed coordinate transport.
    :returns: Completion mode.
    :raises ValueError: If byte, callback, notification, or authority evidence is weak.
    """
    _require(isinstance(population, dict), "completion population is not an object")
    mode = population.get("completion_mode")
    _require(mode in _COMPLETION_MODES, "completion mode is invalid")
    _require(
        _is_positive_int(population.get("byte_count")),
        "population transferred no bytes",
    )
    _require(
        population.get("destination_byte_count") == population.get("byte_count"),
        "destination byte count differs from source",
    )
    _require(_is_sha256(population.get("source_sha256")), "source digest is malformed")
    _require(
        population.get("source_sha256") == population.get("destination_sha256"),
        "destination digest mismatch",
    )
    _require(
        population.get("bytes_verified") is True, "destination bytes were not verified"
    )
    _require(
        population.get("terminal_status") == "NIXL_SUCCESS", "transfer did not succeed"
    )
    _require(
        population.get("terminal_event_count") == 1,
        "transfer did not publish exactly one terminal event",
    )
    _require(
        population.get("attestation_state") == "REMOTE_FLUSHED",
        "terminal event preceded remote flush",
    )
    _require(
        population.get("attestation_status") == "NIXL_SUCCESS",
        "completion attestation failed",
    )
    _require(
        _is_sha256(population.get("attestation_sha256")),
        "completion attestation digest is malformed",
    )
    _require(population.get("completion_claimed") is True, "completion was not claimed")
    _require(
        population.get("take_once_second_status") == "NIXL_ERR_NOT_ALLOWED",
        "completion authority was reusable",
    )
    _require(
        population.get("selected_transports") == [transport],
        "observed transport differs from the coordinate",
    )
    _require(
        population.get("subscription_before_post") is True,
        "transfer subscription was not armed before post",
    )

    notification_timestamp = population.get("notification_completion_timestamp_ns")
    terminal_timestamp = population.get("terminal_native_timestamp_ns")
    drain_timestamp = population.get("drain_timestamp_ns")
    _require(
        _is_positive_int(notification_timestamp), "notification timestamp is missing"
    )
    _require(_is_positive_int(terminal_timestamp), "terminal timestamp is missing")
    _require(_is_positive_int(drain_timestamp), "drain timestamp is missing")
    _require(
        int(terminal_timestamp) >= int(notification_timestamp),
        "terminal event preceded notification completion",
    )
    _require(
        int(drain_timestamp) >= int(terminal_timestamp),
        "terminal event was drained before publication",
    )

    before_return = population.get("callbacks_before_return")
    after_return = population.get("callbacks_after_return")
    callback_count = population.get("callback_count")
    _require(
        _is_nonnegative_int(before_return), "before-return callback count is invalid"
    )
    _require(
        _is_nonnegative_int(after_return), "after-return callback count is invalid"
    )
    _require(
        _is_positive_int(callback_count), "population observed no native callbacks"
    )
    _require(
        int(callback_count) == int(before_return) + int(after_return),
        "callback ordering counts do not conserve",
    )
    if mode == "immediate":
        _require(
            int(before_return) > 0, "immediate population saw no callback before return"
        )
    else:
        _require(
            int(after_return) > 0,
            "asynchronous population saw no callback after return",
        )
    return str(mode)


def _validate_capability(capability: object) -> None:
    """Validate exact-route readiness, failure, retirement, and epoch evidence.

    :param capability: Capability-path receipt.
    :raises ValueError: If a transition or terminal subscription outcome is absent.
    """
    _require(isinstance(capability, dict), "capability evidence is missing")
    _require(
        capability.get("subscribe_before_ready") is True,
        "pre-ready race was not covered",
    )
    _require(
        capability.get("snapshot_after_ready") is True,
        "ready snapshot race was not covered",
    )
    _require(capability.get("ready_state") == "READY", "READY was not observed")
    _require(_is_positive_int(capability.get("ready_epoch")), "READY epoch is missing")
    _require(
        capability.get("epoch_transition_observed") is True,
        "capability epoch transition was not observed",
    )
    _require(
        _is_positive_int(capability.get("next_epoch")),
        "next capability epoch is missing",
    )
    _require(
        int(capability["next_epoch"]) > int(capability["ready_epoch"]),
        "capability epoch did not advance",
    )
    _require(capability.get("failed_state") == "FAILED", "FAILED was not observed")
    _require(
        capability.get("failed_subscription_terminal") is True,
        "FAILED did not terminate its exact subscription",
    )
    _require(capability.get("retired_state") == "RETIRED", "RETIRED was not observed")
    _require(
        capability.get("retired_subscription_terminal") is True,
        "RETIRED did not terminate its exact subscription",
    )


def _validate_faults(faults: object) -> None:
    """Validate the bounded native failure-path matrix.

    :param faults: Failure receipts from isolated real-UCX subcases.
    :raises ValueError: If any required terminal or overflow outcome is absent.
    """
    _require(isinstance(faults, dict), "failure-path evidence is missing")
    expected_statuses = {
        "transfer_cancellation": "NIXL_ERR_CANCELED",
        "remote_failure": "NIXL_ERR_REMOTE_DISCONNECT",
        "notification_failure": "NIXL_ERR_REMOTE_DISCONNECT",
    }
    for name, expected_status in expected_statuses.items():
        fault = faults.get(name)
        _require(isinstance(fault, dict), f"{name} evidence is missing")
        _require(
            fault.get("terminal_status") == expected_status, f"{name} status changed"
        )
        _require(fault.get("terminal_event_count") == 1, f"{name} was not exactly-once")
        _require(fault.get("owner_woken") is True, f"{name} did not wake the owner")
    notification_failure = faults["notification_failure"]
    _require(
        notification_failure.get("data_remote_flushed_before_failure") is True,
        "notification failure did not isolate post-flush ordering",
    )

    overflow = faults.get("queue_overflow")
    _require(isinstance(overflow, dict), "queue-overflow evidence is missing")
    _require(overflow.get("fatal") == "QUEUE_OVERFLOW", "queue overflow was not fatal")
    _require(
        overflow.get("owner_woken") is True, "queue overflow did not wake the owner"
    )
    _require(
        overflow.get("admitted_event_preserved") is True,
        "queue overflow damaged an admitted terminal event",
    )

    shutdown = faults.get("shutdown_cancellation_drain")
    _require(isinstance(shutdown, dict), "shutdown cancellation evidence is missing")
    _require(shutdown.get("cancel_status") == "NIXL_SUCCESS", "shutdown cancel failed")
    _require(
        shutdown.get("terminal_status") == "NIXL_ERR_CANCELED",
        "shutdown did not publish cancellation",
    )
    _require(
        shutdown.get("terminal_event_count") == 1, "shutdown cancellation was not exact"
    )
    _require(
        shutdown.get("owner_woken") is True, "shutdown cancellation did not wake owner"
    )
    _require(shutdown.get("drained") is True, "shutdown left callback work undrained")


def _validate_case(case: object) -> tuple[str, str]:
    """Validate one transport and progress-engine coordinate.

    :param case: Coordinate receipt.
    :returns: Transport and engine coordinate.
    :raises ValueError: If success, failure, or lifecycle evidence is incomplete.
    """
    _require(isinstance(case, dict), "matrix case is not an object")
    transport = case.get("transport")
    engine = case.get("engine")
    _require(transport in _TRANSPORTS, "unknown observed transport")
    _require(engine in _ENGINES, "unknown engine path")
    _require(case.get("memory_type") == "DRAM", "qualification used non-DRAM memory")

    endpoint_identities = case.get("endpoint_identities")
    _require(
        isinstance(endpoint_identities, list)
        and len(endpoint_identities) == 2
        and all(_is_positive_int(identity) for identity in endpoint_identities)
        and len(set(endpoint_identities)) == 2,
        "case did not exercise two exact endpoints",
    )
    source_registration = case.get("source_registration_identity")
    destination_registration = case.get("destination_registration_identity")
    _require(
        _is_positive_int(source_registration), "source registration identity is missing"
    )
    _require(
        _is_positive_int(destination_registration),
        "destination registration identity is missing",
    )
    _require(
        source_registration != destination_registration,
        "source and destination did not use separate registrations",
    )

    populations = case.get("completion_populations")
    _require(isinstance(populations, list), "completion populations are missing")
    _require(
        len(populations) == len(_COMPLETION_MODES),
        "completion populations are incomplete",
    )
    modes = {
        _validate_completion_population(population, str(transport))
        for population in populations
    }
    _require(
        modes == _COMPLETION_MODES, "completion populations are not unique and complete"
    )
    _validate_capability(case.get("capability"))
    _validate_faults(case.get("faults"))
    _validate_zero_inventory(case.get("shutdown"), f"{transport}/{engine}")
    return str(transport), str(engine)


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
    _require(_is_sha256(receipt.get("executable_sha256")), "invalid executable digest")

    invocations = receipt.get("invocations")
    _require(isinstance(invocations, list), "invocation matrix is missing")
    _require(
        len(invocations) == len(_TRANSPORTS) * len(_ENGINES),
        "invocation matrix is incomplete",
    )
    invocation_coordinates = {
        _validate_invocation(invocation) for invocation in invocations
    }
    expected_coordinates = {
        (transport, engine) for transport in _TRANSPORTS for engine in _ENGINES
    }
    _require(
        invocation_coordinates == expected_coordinates,
        "invocation coordinates are not unique and complete",
    )

    runtime_artifacts = receipt.get("runtime_artifacts")
    _require(
        isinstance(runtime_artifacts, list), "runtime artifact inventory is missing"
    )
    for artifact in runtime_artifacts:
        _require(isinstance(artifact, dict), "runtime artifact is not an object")
    _require(
        {artifact.get("component") for artifact in runtime_artifacts}
        == _RUNTIME_COMPONENTS,
        "runtime artifact components are incomplete",
    )
    for artifact in runtime_artifacts:
        _require(
            isinstance(artifact.get("path"), str)
            and Path(str(artifact["path"])).is_absolute(),
            "runtime artifact path is not absolute",
        )
        build_id = artifact.get("build_id")
        _require(
            isinstance(build_id, str) and len(build_id) > 0 and len(build_id) % 2 == 0,
            "runtime artifact build ID is malformed",
        )
        try:
            int(str(build_id), 16)
        except ValueError as error:
            raise ValueError("runtime artifact build ID is not hexadecimal") from error

    zero_gpu = receipt.get("zero_gpu")
    _require(isinstance(zero_gpu, dict), "zero-GPU evidence is missing")
    _require(zero_gpu.get("cuda_visible_devices") == "", "GPU visibility was not empty")
    _require(
        zero_gpu.get("nvidia_visible_devices") == "void",
        "container GPU visibility was not void",
    )
    _require(
        zero_gpu.get("nvidia_device_open_count") == 0, "an NVIDIA device was opened"
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
    _require(len(cases) == len(expected_coordinates), "case matrix is incomplete")
    case_coordinates = {_validate_case(case) for case in cases}
    _require(
        case_coordinates == expected_coordinates,
        "case coordinates are not unique and complete",
    )
    _require(
        case_coordinates == invocation_coordinates,
        "executed and evidenced coordinates differ",
    )
    _validate_zero_inventory(receipt.get("shutdown"), "aggregate shutdown")


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
