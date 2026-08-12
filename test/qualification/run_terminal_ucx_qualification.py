# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from terminal_ucx_receipt import seal_receipt

_COORDINATES = (
    ("self", "shared"),
    ("self", "thread_pool"),
    ("tcp", "shared"),
    ("tcp", "thread_pool"),
)
_NVIDIA_DEVICE_PATTERN = re.compile(r'"(/dev/nvidia[^" ]*)"')
_ZERO_INVENTORY_FIELDS = (
    "queued_channel_events",
    "active_channel_subscriptions",
    "retained_public_subscriptions",
    "backend_producers",
    "active_callback_slots",
    "queued_owner_continuations",
)


@dataclass(frozen=True)
class RunnerPaths:
    """Resolved paths used by the qualification process matrix.

    :ivar executable: Native qualification executable.
    :ivar plugin_directory: Build-tree UCX plugin directory.
    :ivar library_path: Exact dynamic-library search path.
    :ivar nixl_source: NIXL Git source tree.
    :ivar ucx_source: UCX Git source tree.
    :ivar strace: System-call tracer executable.
    :ivar output_root: Durable output directory.
    """

    executable: Path
    plugin_directory: Path
    library_path: str
    nixl_source: Path
    ucx_source: Path
    strace: Path
    output_root: Path


def _sha256(path: Path) -> str:
    """Compute one file's SHA-256 digest.

    :param path: File to digest.
    :returns: Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if len(block) == 0:
                break
            digest.update(block)
    return digest.hexdigest()


def _git_revision(source: Path) -> str:
    """Resolve and validate the exact source revision.

    :param source: Git worktree.
    :returns: Full commit identifier.
    :raises RuntimeError: If the worktree is dirty or the revision is malformed.
    """
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(revision) != 40:
        raise RuntimeError(f"invalid Git revision from {source}")
    dirty = subprocess.run(
        ["git", "-C", str(source), "status", "--short", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(dirty) > 0:
        raise RuntimeError(f"source worktree is dirty: {source}")
    return revision


def _driver_clients() -> set[str]:
    """Inventory open NVIDIA device descriptors without invoking a GPU API.

    :returns: Stable process/fd/device identities visible through procfs.
    """
    clients: set[str] = set()
    for process_directory in Path("/proc").glob("[0-9]*"):
        fd_directory = process_directory / "fd"
        try:
            descriptors = tuple(fd_directory.iterdir())
        except (FileNotFoundError, PermissionError):
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except (FileNotFoundError, PermissionError, OSError):
                continue
            if not target.startswith("/dev/nvidia"):
                continue
            clients.add(f"{process_directory.name}:{descriptor.name}:{target}")
    return clients


def _environment(paths: RunnerPaths, transport: str) -> dict[str, str]:
    """Build the complete clean environment for one process coordinate.

    :param paths: Qualification paths.
    :param transport: Requested and subsequently attested UCX transport.
    :returns: Complete subprocess environment.
    """
    environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "void",
        "UCX_TLS": transport,
        "NIXL_PLUGIN_DIR": str(paths.plugin_directory),
        "LD_LIBRARY_PATH": paths.library_path,
        "NIXL_TELEMETRY_ENABLE": "n",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    if transport == "tcp":
        environment["UCX_NET_DEVICES"] = "lo"
    return environment


def _run_coordinate(
    paths: RunnerPaths, transport: str, engine: str
) -> tuple[dict[str, object], dict[str, object], int]:
    """Run one traced real-UCX coordinate in a fresh process.

    :param paths: Qualification paths.
    :param transport: UCX transport coordinate.
    :param engine: UCX progress-engine coordinate.
    :returns: Native case, exact invocation receipt, and NVIDIA device-open count.
    :raises RuntimeError: If the native process or trace evidence fails.
    """
    coordinate_name = f"{transport}-{engine}"
    case_path = paths.output_root / f"{coordinate_name}.json"
    trace_path = paths.output_root / f"{coordinate_name}.strace"
    stdout_path = paths.output_root / f"{coordinate_name}.stdout.log"
    stderr_path = paths.output_root / f"{coordinate_name}.stderr.log"
    argv = [
        str(paths.executable),
        "--transport",
        transport,
        "--engine",
        engine,
        "--output",
        str(case_path),
    ]
    traced_argv = [
        str(paths.strace),
        "-f",
        "-qq",
        "-e",
        "trace=open,openat,openat2",
        "-o",
        str(trace_path),
        *argv,
    ]
    environment = _environment(paths, transport)
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        result = subprocess.run(
            traced_argv,
            check=False,
            env=environment,
            stdout=stdout,
            stderr=stderr,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"{coordinate_name} failed with {result.returncode}; see {stderr_path}"
        )
    if not case_path.is_file():
        raise RuntimeError(f"{coordinate_name} produced no case receipt")

    trace = trace_path.read_text(encoding="utf-8", errors="replace")
    nvidia_device_opens = len(_NVIDIA_DEVICE_PATTERN.findall(trace))
    if nvidia_device_opens != 0:
        raise RuntimeError(f"{coordinate_name} opened an NVIDIA device")
    native_case = json.loads(case_path.read_text(encoding="utf-8"))
    if native_case.get("transport") != transport or native_case.get("engine") != engine:
        raise RuntimeError(f"{coordinate_name} receipt does not match its invocation")
    invocation: dict[str, object] = {
        "transport": transport,
        "engine": engine,
        "argv": argv,
        "trace_argv": traced_argv,
        "environment": environment,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "strace_path": str(trace_path),
    }
    return native_case, invocation, nvidia_device_opens


def _runtime_artifacts(cases: list[dict[str, object]]) -> list[dict[str, object]]:
    """Require identical runtime artifact identity across all coordinates.

    :param cases: Native coordinate receipts.
    :returns: Canonical shared runtime artifact inventory.
    :raises RuntimeError: If any coordinate used another runtime.
    """
    first = cases[0].get("runtime_artifacts")
    if not isinstance(first, list):
        raise RuntimeError("first coordinate has no runtime artifact inventory")
    for case in cases[1:]:
        if case.get("runtime_artifacts") != first:
            raise RuntimeError("runtime artifact identity changed across coordinates")
    return first


def _aggregate_shutdown(cases: list[dict[str, object]]) -> dict[str, object]:
    """Construct the exact aggregate-zero shutdown receipt.

    :param cases: Native coordinate receipts.
    :returns: Aggregate lifecycle inventory.
    :raises RuntimeError: If a coordinate lacks a typed shutdown inventory.
    """
    aggregate: dict[str, object] = {field: 0 for field in _ZERO_INVENTORY_FIELDS}
    aggregate["capacity"] = 0
    aggregate["accepting_subscriptions"] = False
    aggregate["closed"] = True
    aggregate["fatal"] = "NONE"
    aggregate["eventfd_error"] = 0
    for case in cases:
        shutdown = case.get("shutdown")
        if not isinstance(shutdown, dict):
            raise RuntimeError("coordinate has no shutdown inventory")
        for field in _ZERO_INVENTORY_FIELDS:
            value = shutdown.get(field)
            if not isinstance(value, int) or isinstance(value, bool):
                raise RuntimeError(f"coordinate shutdown field is malformed: {field}")
            aggregate[field] = int(aggregate[field]) + value
        capacity = shutdown.get("capacity")
        if not isinstance(capacity, int) or isinstance(capacity, bool):
            raise RuntimeError("coordinate shutdown capacity is malformed")
        aggregate["capacity"] = int(aggregate["capacity"]) + capacity
        if shutdown.get("accepting_subscriptions") is not False:
            aggregate["accepting_subscriptions"] = True
        if shutdown.get("closed") is not True:
            aggregate["closed"] = False
        if shutdown.get("fatal") != "NONE":
            aggregate["fatal"] = shutdown.get("fatal")
        eventfd_error = shutdown.get("eventfd_error")
        if not isinstance(eventfd_error, int) or isinstance(eventfd_error, bool):
            raise RuntimeError("coordinate shutdown eventfd error is malformed")
        if eventfd_error != 0:
            aggregate["eventfd_error"] = eventfd_error
    return aggregate


def run(paths: RunnerPaths) -> Path:
    """Run, validate, and seal the complete zero-GPU matrix.

    :param paths: Qualification paths.
    :returns: Sealed receipt path.
    """
    paths.output_root.mkdir(parents=True, exist_ok=False)
    before_clients = _driver_clients()
    cases: list[dict[str, object]] = []
    invocations: list[dict[str, object]] = []
    device_open_count = 0
    for transport, engine in _COORDINATES:
        case, invocation, opens = _run_coordinate(paths, transport, engine)
        cases.append(case)
        invocations.append(invocation)
        device_open_count += opens
    after_clients = _driver_clients()

    native_receipt = paths.output_root / "terminal-ucx-native.json"
    sealed_receipt = paths.output_root / "terminal-ucx-sealed.json"
    receipt: dict[str, object] = {
        "schema": "nixl-terminal-ucx-qualification/v3",
        "status": "pass",
        "nixl_revision": _git_revision(paths.nixl_source),
        "ucx_revision": _git_revision(paths.ucx_source),
        "executable_sha256": _sha256(paths.executable),
        "invocations": invocations,
        "runtime_artifacts": _runtime_artifacts(cases),
        "zero_gpu": {
            "cuda_visible_devices": "",
            "nvidia_visible_devices": "void",
            "nvidia_device_open_count": device_open_count,
            "driver_client_delta": sorted(after_clients - before_clients),
            "gpu_api_used": False,
            "evidence_sources": ["strace", "/proc/<pid>/fd"],
        },
        "cases": cases,
        "shutdown": _aggregate_shutdown(cases),
    }
    native_receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    seal_receipt(native_receipt, sealed_receipt)
    return sealed_receipt


def main() -> None:
    """Run the terminal UCX qualification matrix."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--plugin-directory", type=Path, required=True)
    parser.add_argument("--library-path", required=True)
    parser.add_argument("--nixl-source", type=Path, required=True)
    parser.add_argument("--ucx-source", type=Path, required=True)
    parser.add_argument("--strace", type=Path, default=Path("/usr/bin/strace"))
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    paths = RunnerPaths(
        executable=arguments.executable.resolve(strict=True),
        plugin_directory=arguments.plugin_directory.resolve(strict=True),
        library_path=arguments.library_path,
        nixl_source=arguments.nixl_source.resolve(strict=True),
        ucx_source=arguments.ucx_source.resolve(strict=True),
        strace=arguments.strace.resolve(strict=True),
        output_root=arguments.output_root.resolve(strict=False),
    )
    print(run(paths))


if __name__ == "__main__":
    main()
