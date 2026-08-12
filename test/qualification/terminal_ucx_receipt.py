# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
from pathlib import Path

_SCHEMA = "nixl-terminal-ucx-qualification/v4"
_TRANSPORTS = {"self", "tcp"}
_ENGINES = {"shared", "thread_pool"}
_BASE_POPULATIONS = ("small", "large")
_THREAD_POOL_REPOST_POPULATIONS = (
    "thread_pool_repost_generation_1",
    "thread_pool_repost_generation_2",
)
_POPULATIONS = set(_BASE_POPULATIONS + _THREAD_POOL_REPOST_POPULATIONS)
_RUNTIME_COMPONENTS = {"libnixl", "libucp", "ucx-plugin"}
_ARENA_BYTES = 64 * 1024 * 1024
_POPULATION_GEOMETRY = {
    "small": (1, 1024),
    "large": (8, _ARENA_BYTES),
    "thread_pool_repost_generation_1": (8, _ARENA_BYTES),
    "thread_pool_repost_generation_2": (8, _ARENA_BYTES),
}
_COMMON_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "NVIDIA_VISIBLE_DEVICES": "void",
    "NIXL_TELEMETRY_ENABLE": "n",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
}
_SELF_NA_REASON = (
    "ucx_self_is_same_worker_only_and_nixl_local_routes_have_no_remote_agent_handle"
)
_SELF_NA_ANCHORS = {
    "ucx/src/uct/sm/self/self.c:144",
    "ucx/src/ucp/core/ucp_ep.c:1098",
    "nixl/src/core/nixl_agent.cpp:2449",
}
_INVENTORY_ZERO_FIELDS = {
    "queued_channel_events",
    "active_channel_subscriptions",
    "retained_public_subscriptions",
    "backend_producers",
    "active_callback_slots",
    "queued_owner_continuations",
}
_INVENTORY_FIELDS = _INVENTORY_ZERO_FIELDS | {
    "capacity",
    "accepting_subscriptions",
    "closed",
    "fatal",
    "eventfd_error",
}


def _require(condition: bool, message: str) -> None:
    """Raise when one qualification invariant is not satisfied.

    :param condition: Condition which must be true.
    :param message: Validation error message.
    :raises ValueError: If ``condition`` is false.
    """
    if not condition:
        raise ValueError(message)


def _is_int(value: object, minimum: int = 0) -> bool:
    """Return whether a value is an integer at or above a minimum.

    :param value: Value to inspect.
    :param minimum: Inclusive lower bound.
    :returns: Whether the value is a qualifying integer.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _is_sha256(value: object) -> bool:
    """Return whether a value is a canonical SHA-256 digest.

    :param value: Value to inspect.
    :returns: Whether the value is a lowercase hexadecimal digest.
    """
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _is_git_revision(value: object) -> bool:
    """Return whether a value is a full hexadecimal Git object identifier.

    :param value: Value to inspect.
    :returns: Whether the value is a full SHA-1 object identifier.
    """
    if not isinstance(value, str) or len(value) != 40 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_not_applicable(value: object, context: str) -> None:
    """Validate one structurally inapplicable self-transport semantic.

    :param value: Typed coverage record.
    :param context: Human-readable semantic name.
    :raises ValueError: If the record fabricates evidence or loses its rationale.
    """
    _require(isinstance(value, dict), f"{context} applicability is missing")
    _require(
        value.get("applicability") == "not_applicable",
        f"{context} was not marked structurally inapplicable",
    )
    _require(value.get("reason") == _SELF_NA_REASON, f"{context} reason changed")
    anchors = value.get("evidence_anchors")
    _require(
        isinstance(anchors, list)
        and all(isinstance(anchor, str) for anchor in anchors)
        and set(anchors) == _SELF_NA_ANCHORS,
        f"{context} source anchors are incomplete",
    )
    _require(
        set(value) == {"applicability", "reason", "evidence_anchors"},
        f"{context} contains fabricated pass evidence",
    )


def _validate_zero_inventory(inventory: object, context: str) -> None:
    """Validate one public native lifecycle inventory after shutdown.

    :param inventory: Inventory emitted by the public terminal-event API.
    :param context: Human-readable validation context.
    :raises ValueError: If work, subscriptions, or fatal state remain.
    """
    _require(isinstance(inventory, dict), f"{context} inventory is missing")
    _require(
        set(inventory) == _INVENTORY_FIELDS,
        f"{context} inventory fields differ from the public lifecycle schema",
    )
    _require(_is_int(inventory.get("capacity"), 1), f"{context} capacity is invalid")
    for field in _INVENTORY_ZERO_FIELDS:
        _require(inventory.get(field) == 0, f"{context} retained {field}")
    _require(
        inventory.get("accepting_subscriptions") is False,
        f"{context} still accepts subscriptions",
    )
    _require(inventory.get("closed") is True, f"{context} channel stayed open")
    _require(inventory.get("fatal") == "NONE", f"{context} reported fatal state")
    _require(inventory.get("eventfd_error") == 0, f"{context} eventfd failed")


def _validate_invocation(invocation: object) -> tuple[str, str, str]:
    """Validate one exact real-UCX process invocation.

    :param invocation: Invocation receipt.
    :returns: Transport, engine, and qualification executable path.
    :raises ValueError: If argv or process environment is incomplete.
    """
    _require(isinstance(invocation, dict), "invocation is not an object")
    transport = invocation.get("transport")
    engine = invocation.get("engine")
    _require(transport in _TRANSPORTS, "invocation transport is invalid")
    _require(engine in _ENGINES, "invocation engine is invalid")
    _require(
        set(invocation)
        == {
            "transport",
            "engine",
            "argv",
            "trace_argv",
            "environment",
            "stdout_path",
            "stderr_path",
            "strace_path",
        },
        "invocation fields differ from the sealed runner schema",
    )
    argv = invocation.get("argv")
    trace_argv = invocation.get("trace_argv")
    _require(
        isinstance(argv, list)
        and len(argv) == 7
        and all(isinstance(argument, str) and len(argument) > 0 for argument in argv),
        "invocation argv is missing",
    )
    executable = str(argv[0])
    output_path = str(argv[6])
    _require(
        Path(executable).is_absolute(), "qualification executable is not absolute"
    )
    _require(
        argv[1:6] == ["--transport", transport, "--engine", engine, "--output"],
        "qualification argv differs from its coordinate",
    )
    _require(
        Path(output_path).is_absolute(), "qualification output path is not absolute"
    )
    strace_path = invocation.get("strace_path")
    _require(
        isinstance(trace_argv, list)
        and len(trace_argv) == len(argv) + 7
        and Path(str(trace_argv[0])).is_absolute(),
        "device-open trace invocation is missing",
    )
    _require(
        trace_argv[1:6] == ["-f", "-qq", "-e", "trace=open,openat,openat2", "-o"]
        and trace_argv[6] == strace_path
        and trace_argv[7:] == argv,
        "device-open trace invocation differs from the native invocation",
    )
    for field in ("stdout_path", "stderr_path", "strace_path"):
        value = invocation.get(field)
        _require(
            isinstance(value, str) and Path(value).is_absolute(),
            f"invocation {field} is not an absolute path",
        )
    environment = invocation.get("environment")
    _require(isinstance(environment, dict), "invocation environment is missing")
    expected_environment = dict(_COMMON_ENVIRONMENT)
    expected_environment["UCX_TLS"] = transport
    if transport == "tcp":
        expected_environment["UCX_NET_DEVICES"] = "lo"
    _require(
        set(environment) == set(expected_environment) | {"NIXL_PLUGIN_DIR", "LD_LIBRARY_PATH"},
        "invocation environment contains unbound state",
    )
    for name, expected in expected_environment.items():
        _require(environment.get(name) == expected, f"invocation {name} changed")
    plugin_directory = environment.get("NIXL_PLUGIN_DIR")
    _require(
        isinstance(plugin_directory, str) and Path(plugin_directory).is_absolute(),
        "NIXL_PLUGIN_DIR is not bound to an absolute path",
    )
    _require(
        isinstance(environment.get("LD_LIBRARY_PATH"), str)
        and len(str(environment["LD_LIBRARY_PATH"])) > 0,
        "LD_LIBRARY_PATH is missing",
    )
    return str(transport), str(engine), executable


def _validate_registration(registration: object, context: str) -> tuple[int, int]:
    """Validate one real DRAM registration observation.

    :param registration: Native registration evidence.
    :param context: Source or destination label.
    :returns: Base address and registered byte capacity.
    :raises ValueError: If the registration was not observed successfully.
    """
    _require(isinstance(registration, dict), f"{context} registration is missing")
    _require(
        registration.get("status") == "NIXL_SUCCESS", f"{context} registration failed"
    )
    _require(registration.get("memory_type") == "DRAM", f"{context} was not DRAM")
    address = registration.get("base_address")
    byte_capacity = registration.get("byte_capacity")
    _require(_is_int(address, 1), f"{context} address is invalid")
    _require(_is_int(byte_capacity, 1), f"{context} byte capacity is invalid")
    return int(address), int(byte_capacity)


def _validate_endpoint_flushes(endpoint_flushes: object) -> int:
    """Validate immutable endpoint identities and completed flush authority.

    :param endpoint_flushes: Endpoint evidence copied from the transfer attestation.
    :returns: Number of distinct endpoints flushed by the submission.
    :raises ValueError: If endpoint identity aliases or flush authority is incomplete.
    """
    _require(
        isinstance(endpoint_flushes, list) and len(endpoint_flushes) > 0,
        "endpoint flush evidence is missing",
    )
    identities: set[tuple[int, int, int]] = set()
    for endpoint in endpoint_flushes:
        _require(isinstance(endpoint, dict), "endpoint flush is not an object")
        worker_id = endpoint.get("worker_id")
        worker_identity = endpoint.get("worker_identity")
        endpoint_identity = endpoint.get("endpoint_identity")
        _require(_is_int(worker_id), "endpoint worker ID is invalid")
        _require(_is_int(worker_identity, 1), "endpoint worker identity is invalid")
        _require(_is_int(endpoint_identity, 1), "endpoint identity is invalid")
        _require(endpoint.get("flush_posted") is True, "endpoint flush was not posted")
        _require(
            endpoint.get("remote_flushed") is True, "endpoint was not remote-flushed"
        )
        identities.add((int(worker_id), int(worker_identity), int(endpoint_identity)))
    _require(
        len(identities) == len(endpoint_flushes),
        "endpoint flush identities are not unique",
    )
    return len(endpoint_flushes)


def _validate_terminal_progress(
    progress: object,
    transport: str,
    endpoint_count: int,
) -> tuple[int, int]:
    """Validate native callback, flush, notification, and queue evidence.

    :param progress: Attested terminal-progress observation.
    :param transport: Observed UCX transport.
    :param endpoint_count: Number of distinct attested endpoint flushes.
    :returns: Before-return and after-return callback counts.
    :raises ValueError: If callback accounting or terminal ordering is unsound.
    """
    _require(isinstance(progress, dict), "terminal-progress evidence is missing")
    _require(
        progress.get("autonomous") is True, "transfer was not autonomously completed"
    )
    data_callbacks = progress.get("data_callbacks")
    flush_callbacks = progress.get("endpoint_flush_callbacks")
    notification_callbacks = progress.get("notification_callbacks")
    asynchronous_requests = progress.get("asynchronous_requests")
    immediate_completions = progress.get("immediate_completions")
    before_return = progress.get("callbacks_before_poster_return")
    for value, name in (
        (data_callbacks, "data callbacks"),
        (flush_callbacks, "endpoint-flush callbacks"),
        (notification_callbacks, "notification callbacks"),
        (asynchronous_requests, "asynchronous requests"),
        (immediate_completions, "immediate completions"),
        (before_return, "callbacks before poster return"),
    ):
        _require(_is_int(value), f"{name} count is invalid")
    _require(int(data_callbacks) > 0, "no data completion callback was observed")
    expected_notifications = 0 if transport == "self" else 1
    _require(
        int(notification_callbacks) == expected_notifications,
        "notification callback count differs from transport authority",
    )
    callback_count = (
        int(data_callbacks) + int(flush_callbacks) + int(notification_callbacks)
    )
    post_count = int(asynchronous_requests) + int(immediate_completions)
    _require(
        post_count >= callback_count,
        "callback observations exceed native post observations",
    )
    _require(
        int(before_return) <= callback_count, "before-return callbacks exceed total"
    )
    after_return = callback_count - int(before_return)
    _require(
        _is_int(progress.get("peak_continuation_depth")),
        "peak continuation depth is invalid",
    )
    _require(
        progress.get("active_callback_slots_at_terminal") == 0,
        "callback slots remained active at terminal publication",
    )
    _require(
        progress.get("continuation_depth_at_terminal") == 0,
        "owner continuations remained queued at terminal publication",
    )
    data_timestamp = progress.get("last_data_callback_timestamp_ns")
    flush_timestamp = progress.get("last_flush_callback_timestamp_ns")
    notification_timestamp = progress.get("notification_callback_timestamp_ns")
    terminal_timestamp = progress.get("terminal_publish_timestamp_ns")
    _require(_is_int(data_timestamp, 1), "data callback timestamp is missing")
    _require(
        _is_int(terminal_timestamp, 1), "terminal publication timestamp is missing"
    )
    if transport == "self":
        _require(int(asynchronous_requests) == 0, "self completion was not immediate")
        _require(
            int(flush_callbacks) == 0, "self unexpectedly invoked a flush callback"
        )
        _require(flush_timestamp == 0, "self emitted a flush callback timestamp")
        _require(
            post_count - callback_count == endpoint_count,
            "self immediate flush completions differ from endpoint authority",
        )
        _require(
            notification_timestamp == 0, "self emitted a remote notification timestamp"
        )
        _require(
            int(terminal_timestamp) >= int(data_timestamp),
            "self terminal publication preceded data completion",
        )
    else:
        _require(
            int(flush_callbacks) == endpoint_count,
            "TCP flush callbacks differ from endpoint authority",
        )
        _require(
            post_count == callback_count,
            "TCP completion bypassed its native callback",
        )
        _require(
            int(asynchronous_requests) > 0,
            "TCP did not exercise asynchronous completion",
        )
        _require(_is_int(flush_timestamp, 1), "flush callback timestamp is missing")
        _require(
            int(flush_timestamp) >= int(data_timestamp),
            "flush preceded data completion",
        )
        _require(
            _is_int(notification_timestamp, 1), "notification timestamp is missing"
        )
        _require(
            int(notification_timestamp) >= int(flush_timestamp),
            "notification completed before endpoint flush",
        )
        _require(
            int(terminal_timestamp) >= int(notification_timestamp),
            "terminal publication preceded notification completion",
        )
    _require(
        progress.get("terminal_status") == "NIXL_SUCCESS",
        "native terminal status failed",
    )
    return int(before_return), after_return


def _validate_population(
    population: object, transport: str
) -> tuple[str, int, int, int, int]:
    """Validate one small or large transfer population.

    :param population: Population receipt.
    :param transport: Observed UCX transport.
    :returns: Population name, callback ordering counts, handle identity, and generation.
    :raises ValueError: If byte integrity or completion authority is incomplete.
    """
    _require(isinstance(population, dict), "completion population is not an object")
    name = population.get("population")
    _require(name in _POPULATIONS, "population name is invalid")
    expected_descriptor_count, expected_byte_count = _POPULATION_GEOMETRY[str(name)]
    _require(
        population.get("descriptor_count") == expected_descriptor_count,
        f"{name} descriptor count differs from the frozen geometry",
    )
    _require(
        population.get("byte_count") == expected_byte_count,
        f"{name} byte count differs from the frozen geometry",
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
        population.get("terminal_event_count") == 1, "terminal event was not exact"
    )
    _require(
        population.get("attestation_state") == "REMOTE_FLUSHED", "remote flush absent"
    )
    _require(
        population.get("attestation_status") == "NIXL_SUCCESS", "attestation failed"
    )
    _require(
        _is_sha256(population.get("attestation_sha256")),
        "attestation digest is malformed",
    )
    handle_identity = population.get("attestation_handle_identity")
    generation = population.get("attestation_generation")
    _require(_is_int(handle_identity, 1), "attestation handle identity is invalid")
    _require(_is_int(generation, 1), "attestation generation is invalid")
    _require(population.get("completion_claimed") is True, "completion was not claimed")
    _require(
        population.get("take_once_second_status") == "NIXL_ERR_NOT_ALLOWED",
        "completion authority was reusable",
    )
    _require(
        population.get("selected_transports") == [transport],
        "transport attestation differs",
    )
    _require(
        population.get("subscription_before_post") is True,
        "subscription armed too late",
    )
    _require(
        _is_int(population.get("event_native_timestamp_ns"), 1),
        "terminal event timestamp is missing",
    )
    _require(
        _is_int(population.get("drain_timestamp_ns"), 1)
        and int(population["drain_timestamp_ns"])
        >= int(population["event_native_timestamp_ns"]),
        "terminal event drain ordering is invalid",
    )
    endpoint_count = _validate_endpoint_flushes(population.get("endpoint_flushes"))
    before, after = _validate_terminal_progress(
        population.get("terminal_progress"), transport, endpoint_count
    )
    return str(name), before, after, int(handle_identity), int(generation)


def _validate_capability_route(
    route: object,
    expected_name: str,
    expected_states: list[str],
) -> tuple[int, int]:
    """Validate one exact remote-route lifecycle subscription.

    :param route: Route lifecycle evidence.
    :param expected_name: Semantic route name.
    :param expected_states: Exact ordered state sequence.
    :returns: Handle identity and generation.
    :raises ValueError: If identity, state, epoch, or release evidence is incomplete.
    """
    _require(isinstance(route, dict), f"{expected_name} capability route is missing")
    _require(route.get("name") == expected_name, f"{expected_name} route name changed")
    identity = route.get("handle_identity")
    generation = route.get("handle_generation")
    _require(_is_int(identity, 1), f"{expected_name} handle identity is invalid")
    _require(_is_int(generation, 1), f"{expected_name} handle generation is invalid")
    _require(
        route.get("states") == expected_states, f"{expected_name} states are incomplete"
    )
    epochs = route.get("epochs")
    _require(
        isinstance(epochs, list)
        and len(epochs) == len(expected_states)
        and all(_is_int(epoch, 1) for epoch in epochs),
        f"{expected_name} epochs are incomplete",
    )
    if expected_name == "epoch_advance":
        _require(int(epochs[1]) > int(epochs[0]), "capability epoch did not advance")
    _require(
        route.get("release_status") == "NIXL_SUCCESS", f"{expected_name} release failed"
    )
    expected_terminal = expected_states[-1] in {"FAILED", "RETIRED"}
    _require(
        route.get("subscription_terminal") is expected_terminal,
        f"{expected_name} subscription terminality changed",
    )
    return int(identity), int(generation)


def _validate_capability(capability: object) -> None:
    """Validate TCP exact-route capability coverage on distinct generations.

    :param capability: Applicable remote capability evidence.
    :raises ValueError: If races, lifecycle states, or route isolation are absent.
    """
    _require(isinstance(capability, dict), "capability evidence is missing")
    _require(
        capability.get("applicability") == "applicable", "capability was not exercised"
    )
    _require(
        capability.get("subscribe_before_ready") is True,
        "pre-ready race was not exercised",
    )
    _require(
        capability.get("snapshot_after_ready") is True,
        "ready snapshot was not exercised",
    )
    routes = capability.get("routes")
    _require(
        isinstance(routes, list) and len(routes) == 3,
        "capability routes are incomplete",
    )
    expected = {
        "epoch_advance": ["READY", "READY"],
        "endpoint_failure": ["READY", "FAILED"],
        "retirement": ["READY", "RETIRED"],
    }
    identities: set[tuple[int, int]] = set()
    names: set[str] = set()
    for route in routes:
        _require(isinstance(route, dict), "capability route is not an object")
        name = route.get("name")
        _require(name in expected, "unknown capability route")
        names.add(str(name))
        identities.add(
            _validate_capability_route(route, str(name), expected[str(name)])
        )
    _require(names == set(expected), "capability route coverage is incomplete")
    _require(
        len(identities) == 3, "capability states reused one exact route generation"
    )


def _validate_terminal_fault(fault: object, expected_status: str, context: str) -> None:
    """Validate one exactly-once terminal failure observation.

    :param fault: Failure record.
    :param expected_status: Required terminal status.
    :param context: Failure-path name.
    :raises ValueError: If terminal delivery or owner wakeup is absent.
    """
    _require(isinstance(fault, dict), f"{context} evidence is missing")
    _require(fault.get("applicability") == "applicable", f"{context} was not exercised")
    _require(
        fault.get("terminal_status") == expected_status, f"{context} status changed"
    )
    _require(fault.get("terminal_event_count") == 1, f"{context} was not exact")
    _require(fault.get("owner_woken") is True, f"{context} did not wake the owner")


def _validate_faults(faults: object, transport: str) -> None:
    """Validate transport-aware isolated failure coverage.

    :param faults: Failure-path evidence.
    :param transport: Coordinate transport.
    :raises ValueError: If a common fault or applicable remote fault is absent.
    """
    _require(isinstance(faults, dict), "failure-path evidence is missing")
    _validate_terminal_fault(
        faults.get("transfer_cancellation"),
        "NIXL_ERR_CANCELED",
        "transfer cancellation",
    )
    overflow = faults.get("queue_overflow")
    _require(isinstance(overflow, dict), "queue overflow evidence is missing")
    _require(
        overflow.get("applicability") == "applicable",
        "queue overflow was not exercised",
    )
    _require(overflow.get("fatal") == "QUEUE_OVERFLOW", "queue overflow was not fatal")
    _require(
        overflow.get("owner_woken") is True, "queue overflow did not wake the owner"
    )
    _require(
        overflow.get("admitted_event_preserved") is True,
        "queue overflow damaged an admitted event",
    )
    shutdown = faults.get("shutdown_cancellation_drain")
    _validate_terminal_fault(shutdown, "NIXL_ERR_CANCELED", "shutdown cancellation")
    _require(
        isinstance(shutdown, dict) and shutdown.get("cancel_status") == "NIXL_SUCCESS",
        "shutdown cancel failed",
    )
    if transport == "tcp":
        _require(
            shutdown.get("posted_in_flight") is True,
            "shutdown cancellation never reached an in-flight native transfer",
        )
        pre_cancel_inventory = (
            shutdown.get("backend_producers_before_cancel"),
            shutdown.get("active_callback_slots_before_cancel"),
            shutdown.get("queued_owner_continuations_before_cancel"),
        )
        _require(
            all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in pre_cancel_inventory
            )
            and sum(pre_cancel_inventory) > 0,
            "shutdown cancellation exposed no in-flight native inventory",
        )
    _require(shutdown.get("drained") is True, "shutdown left native work undrained")
    if transport == "self":
        _validate_not_applicable(faults.get("remote_failure"), "self remote failure")
        _validate_not_applicable(
            faults.get("notification_failure"), "self notification failure"
        )
        return
    _validate_terminal_fault(
        faults.get("remote_failure"), "NIXL_ERR_REMOTE_DISCONNECT", "remote failure"
    )
    notification_failure = faults.get("notification_failure")
    _validate_terminal_fault(
        notification_failure,
        "NIXL_ERR_REMOTE_DISCONNECT",
        "notification failure",
    )
    _require(
        isinstance(notification_failure, dict)
        and notification_failure.get("data_remote_flushed_before_failure") is True,
        "notification failure lacked all-endpoint remote-flush authority",
    )
    _require(
        notification_failure.get("notification_failed_after_remote_flush") is True,
        "notification failure did not follow the final remote-flush callback",
    )
    _require(
        notification_failure.get("source_progress_mode") == "production",
        "notification fixture did not retain the production source progress mode",
    )
    _require(
        notification_failure.get("fault_peer_engine") == "thread_pool"
        and notification_failure.get("fault_peer_shared_worker_quiesced") is True,
        "notification fixture did not prove controlled fault-peer quiescence",
    )


def _validate_case(case: object) -> tuple[str, str]:
    """Validate one transport and progress-engine coordinate.

    :param case: Coordinate receipt.
    :returns: Transport and engine coordinate.
    :raises ValueError: If authority, coverage, or lifecycle evidence is incomplete.
    """
    _require(isinstance(case, dict), "matrix case is not an object")
    transport = case.get("transport")
    engine = case.get("engine")
    _require(transport in _TRANSPORTS, "unknown observed transport")
    _require(engine in _ENGINES, "unknown engine path")
    expected_shape = (
        "one_agent_local_route" if transport == "self" else "two_distinct_agents"
    )
    _require(
        case.get("agent_shape") == expected_shape, "agent shape differs from transport"
    )
    _require(
        case.get("agent_count") == (1 if transport == "self" else 2),
        "agent count changed",
    )
    _require(
        case.get("remote_agent_handle_present") is (transport == "tcp"),
        "remote handle presence differs from transport authority",
    )
    registrations = case.get("registrations")
    _require(isinstance(registrations, dict), "registration evidence is missing")
    source_address, source_capacity = _validate_registration(
        registrations.get("source"), "source"
    )
    destination_address, destination_capacity = _validate_registration(
        registrations.get("destination"), "destination"
    )
    _require(
        source_address != destination_address, "source and destination buffers alias"
    )
    _require(source_capacity == destination_capacity, "registration capacities differ")
    _require(
        source_capacity == _ARENA_BYTES,
        "registration capacity differs from the frozen arena geometry",
    )
    populations = case.get("completion_populations")
    expected_populations = list(_BASE_POPULATIONS)
    if engine == "thread_pool":
        expected_populations.extend(_THREAD_POOL_REPOST_POPULATIONS)
    _require(
        isinstance(populations, list) and len(populations) == len(expected_populations),
        "population matrix is incomplete",
    )
    _require(
        [
            population.get("population")
            for population in populations
            if isinstance(population, dict)
        ]
        == expected_populations,
        "population order or names differ from the engine contract",
    )
    before_return = 0
    after_return = 0
    population_authority: list[tuple[str, int, int]] = []
    for population in populations:
        name, before, after, handle_identity, generation = _validate_population(
            population, str(transport)
        )
        population_authority.append((name, handle_identity, generation))
        before_return += before
        after_return += after
    if engine == "thread_pool":
        first_repost = population_authority[-2]
        second_repost = population_authority[-1]
        _require(
            first_repost[1] == second_repost[1],
            "thread-pool repost changed request identity",
        )
        _require(
            second_repost[2] > first_repost[2],
            "thread-pool repost did not advance transfer generation",
        )
    if transport == "self":
        _require(before_return > 0, "self did not prove callback-before-poster-return")
    else:
        _require(after_return > 0, "TCP did not prove callback-after-poster-return")
    capability = case.get("remote_route_capability")
    attached_notification = case.get("attached_authenticated_notification")
    if transport == "self":
        _validate_not_applicable(capability, "self remote capability")
        _validate_not_applicable(attached_notification, "self attached notification")
    else:
        _validate_capability(capability)
        _require(
            isinstance(attached_notification, dict)
            and attached_notification.get("applicability") == "applicable"
            and attached_notification.get("success_count") == 2,
            "TCP attached-notification success coverage is incomplete",
        )
    _validate_faults(case.get("faults"), str(transport))
    _validate_runtime_artifacts(case.get("runtime_artifacts"))
    _validate_zero_inventory(case.get("shutdown"), f"{transport}/{engine}")
    return str(transport), str(engine)


def _validate_runtime_artifacts(artifacts: object) -> None:
    """Validate exact loaded NIXL, UCX, and UCX-plugin identities.

    :param artifacts: Runtime artifact inventory.
    :raises ValueError: If paths, build IDs, or components are incomplete.
    """
    _require(
        isinstance(artifacts, list) and len(artifacts) == len(_RUNTIME_COMPONENTS),
        "runtime artifact inventory is missing",
    )
    _require(
        all(isinstance(artifact, dict) for artifact in artifacts),
        "runtime artifact is not an object",
    )
    _require(
        {artifact.get("component") for artifact in artifacts} == _RUNTIME_COMPONENTS,
        "runtime artifact components are incomplete",
    )
    for artifact in artifacts:
        _require(
            set(artifact) == {"component", "path", "build_id", "version"},
            "runtime artifact fields differ from the sealed schema",
        )
        path = artifact.get("path")
        build_id = artifact.get("build_id")
        version = artifact.get("version")
        _require(
            isinstance(path, str) and Path(path).is_absolute(),
            "runtime artifact path is not absolute",
        )
        _require(
            isinstance(build_id, str) and len(build_id) > 0 and len(build_id) % 2 == 0,
            "runtime artifact build ID is malformed",
        )
        _require(
            isinstance(version, str) and len(version) > 0,
            "runtime artifact version is missing",
        )
        try:
            int(build_id, 16)
        except ValueError as error:
            raise ValueError("runtime artifact build ID is not hexadecimal") from error


def validate_receipt(receipt: dict[str, object]) -> None:
    """Validate one complete zero-GPU transport-aware UCX qualification.

    :param receipt: Parsed receipt object.
    :raises ValueError: If any Stage-1 invariant is absent or false.
    """
    _require(receipt.get("schema") == _SCHEMA, "unexpected receipt schema")
    _require(receipt.get("status") == "pass", "native qualification did not pass")
    _require(_is_git_revision(receipt.get("nixl_revision")), "invalid NIXL revision")
    _require(_is_git_revision(receipt.get("ucx_revision")), "invalid UCX revision")
    _require(_is_sha256(receipt.get("executable_sha256")), "invalid executable digest")
    invocations = receipt.get("invocations")
    _require(
        isinstance(invocations, list) and len(invocations) == 4,
        "invocation matrix is incomplete",
    )
    invocation_results = [_validate_invocation(invocation) for invocation in invocations]
    invocation_coordinates = {
        (transport, engine) for transport, engine, executable in invocation_results
    }
    executable_paths = {executable for _, _, executable in invocation_results}
    _require(
        len(executable_paths) == 1,
        "qualification executable changed across coordinates",
    )
    expected_coordinates = {
        (transport, engine) for transport in _TRANSPORTS for engine in _ENGINES
    }
    _require(
        invocation_coordinates == expected_coordinates, "invocation coordinates differ"
    )
    runtime_artifacts = receipt.get("runtime_artifacts")
    _validate_runtime_artifacts(runtime_artifacts)
    zero_gpu = receipt.get("zero_gpu")
    _require(isinstance(zero_gpu, dict), "zero-GPU evidence is missing")
    _require(zero_gpu.get("cuda_visible_devices") == "", "GPU visibility was not empty")
    _require(
        zero_gpu.get("nvidia_visible_devices") == "void",
        "NVIDIA visibility was not void",
    )
    _require(
        zero_gpu.get("nvidia_device_open_count") == 0, "an NVIDIA device was opened"
    )
    _require(zero_gpu.get("driver_client_delta") == [], "NVIDIA driver clients changed")
    _require(
        zero_gpu.get("gpu_api_used") is False, "GPU API constructed zero-GPU evidence"
    )
    _require(
        zero_gpu.get("evidence_sources") == ["strace", "/proc/<pid>/fd"],
        "zero-GPU evidence sources changed",
    )
    cases = receipt.get("cases")
    _require(isinstance(cases, list) and len(cases) == 4, "case matrix is incomplete")
    case_coordinates: set[tuple[str, str]] = set()
    for case in cases:
        case_coordinates.add(_validate_case(case))
        _require(
            isinstance(case, dict) and case.get("runtime_artifacts") == runtime_artifacts,
            "runtime artifact identity changed across coordinates",
        )
    _require(case_coordinates == expected_coordinates, "case coordinates differ")
    _require(
        case_coordinates == invocation_coordinates,
        "executed and evidenced cases differ",
    )
    aggregate_shutdown = receipt.get("shutdown")
    _validate_zero_inventory(aggregate_shutdown, "aggregate shutdown")
    _require(isinstance(aggregate_shutdown, dict), "aggregate shutdown is missing")
    coordinate_capacity = sum(
        int(case["shutdown"]["capacity"])
        for case in cases
        if isinstance(case, dict) and isinstance(case.get("shutdown"), dict)
    )
    _require(
        aggregate_shutdown.get("capacity") == coordinate_capacity,
        "aggregate shutdown capacity does not conserve coordinate inventories",
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
    digest = hashlib.sha256(canonical.encode()).hexdigest()
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
