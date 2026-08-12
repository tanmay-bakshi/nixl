# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from pathlib import Path

import pytest
from run_terminal_ucx_qualification import (
    RunnerPaths,
    _aggregate_shutdown,
    _environment,
    _run_coordinate,
)


def _paths(tmp_path: Path) -> RunnerPaths:
    """Build isolated runner paths for unit qualification.

    :param tmp_path: Pytest temporary directory.
    :returns: Runner paths rooted in the temporary directory.
    """
    executable = tmp_path / "terminal_ucx_qualification"
    executable.touch()
    plugin_directory = tmp_path / "plugins"
    plugin_directory.mkdir()
    strace = tmp_path / "strace"
    strace.touch()
    output_root = tmp_path / "receipt"
    output_root.mkdir()
    return RunnerPaths(
        executable=executable,
        plugin_directory=plugin_directory,
        library_path=str(tmp_path / "lib"),
        nixl_source=tmp_path / "nixl",
        ucx_source=tmp_path / "ucx",
        strace=strace,
        output_root=output_root,
    )


def _shutdown(**changes: object) -> dict[str, object]:
    """Build one public terminal-channel shutdown inventory.

    :param changes: Inventory fields to replace.
    :returns: Complete shutdown inventory.
    """
    inventory: dict[str, object] = {
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
    inventory.update(changes)
    return inventory


def test_environment_is_clean_and_binds_exact_plugin_directory(tmp_path: Path) -> None:
    """Bind the requested transport and build-tree plugin without ambient state."""
    paths = _paths(tmp_path)

    self_environment = _environment(paths, "self")
    tcp_environment = _environment(paths, "tcp")

    assert self_environment == {
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "void",
        "UCX_TLS": "self",
        "NIXL_PLUGIN_DIR": str(paths.plugin_directory),
        "LD_LIBRARY_PATH": paths.library_path,
        "NIXL_TELEMETRY_ENABLE": "n",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    assert tcp_environment["UCX_TLS"] == "tcp"
    assert tcp_environment["UCX_NET_DEVICES"] == "lo"
    assert tcp_environment["NIXL_PLUGIN_DIR"] == str(paths.plugin_directory)


def test_aggregate_shutdown_conserves_public_inventory() -> None:
    """Aggregate capacities while preserving an exact terminal zero inventory."""
    aggregate = _aggregate_shutdown(
        [{"shutdown": _shutdown()}, {"shutdown": _shutdown()}]
    )

    assert aggregate == _shutdown(capacity=128)


def test_aggregate_shutdown_propagates_nonzero_lifecycle_state() -> None:
    """Prevent coordinate residue from being hidden by aggregate construction."""
    aggregate = _aggregate_shutdown(
        [
            {"shutdown": _shutdown()},
            {
                "shutdown": _shutdown(
                    backend_producers=1,
                    fatal="QUEUE_OVERFLOW",
                    eventfd_error=5,
                )
            },
        ]
    )

    assert aggregate["backend_producers"] == 1
    assert aggregate["fatal"] == "QUEUE_OVERFLOW"
    assert aggregate["eventfd_error"] == 5


def test_run_coordinate_rejects_native_coordinate_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a native receipt that does not attest its exact invocation."""
    paths = _paths(tmp_path)

    def fake_run(
        argv: list[str],
        *,
        check: bool,
        env: dict[str, str],
        stdout: object,
        stderr: object,
    ) -> subprocess.CompletedProcess[str]:
        """Emit a successful but mismatched native coordinate.

        :param argv: Traced process argument vector.
        :param check: Whether subprocess errors should raise.
        :param env: Clean child environment.
        :param stdout: Captured stdout file.
        :param stderr: Captured stderr file.
        :returns: Successful process result.
        """
        assert check is False
        assert env["NIXL_PLUGIN_DIR"] == str(paths.plugin_directory)
        assert stdout is not None
        assert stderr is not None
        output_index = argv.index("--output") + 1
        Path(argv[output_index]).write_text(
            json.dumps({"transport": "tcp", "engine": "shared"}),
            encoding="utf-8",
        )
        trace_index = argv.index("-o") + 1
        Path(argv[trace_index]).write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="does not match its invocation"):
        _run_coordinate(paths, "self", "shared")
