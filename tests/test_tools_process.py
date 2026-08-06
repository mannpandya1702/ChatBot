"""T-2.7 verification: sys.process.top and sys.process.find."""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from jarvis.tools import sys_process
from jarvis.tools.registry import registry
from jarvis.tools.sys_process import (
    ProcessListInput,
    ProcessLookupInput,
    find_process,
    top_processes,
)
from jarvis.util.errors import ToolExecutionError


@pytest.fixture(autouse=True)
def _fast_priming(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the CPU priming sleep so the suite stays quick."""
    monkeypatch.setattr(sys_process.time, "sleep", lambda _s: None)


@pytest.fixture
def own_name() -> str:
    """Name of the process running the tests.

    Derived rather than hardcoded: under "uv run pytest" this is "pytest", under
    "uv run python -m pytest" it is "python3", and on Windows it is "python.exe".
    """
    return sys_process.psutil.Process(os.getpid()).name()


class TestTopProcesses:
    def test_returns_processes(self) -> None:
        result = top_processes(ProcessListInput())
        assert result.processes
        assert result.total_processes > 0

    def test_capped_at_five(self) -> None:
        """§5 caps spoken list results at five items."""
        assert len(top_processes(ProcessListInput()).processes) <= 5

    def test_limit_is_bounded_by_the_schema(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ProcessListInput(limit=50)

    def test_sorted_by_cpu(self) -> None:
        rows = top_processes(ProcessListInput(sort_by="cpu")).processes
        assert rows == sorted(rows, key=lambda r: r.cpu_percent, reverse=True)

    def test_sorted_by_memory(self) -> None:
        result = top_processes(ProcessListInput(sort_by="memory"))
        assert result.sorted_by == "memory"
        assert result.processes == sorted(
            result.processes, key=lambda r: r.memory_mb, reverse=True
        )

    def test_cpu_percent_is_normalised_across_cores(self) -> None:
        """psutil reports per-core percent. A user expects the Task Manager number."""
        for row in top_processes(ProcessListInput()).processes:
            assert 0.0 <= row.cpu_percent <= 100.0

    def test_fields_are_populated(self) -> None:
        for row in top_processes(ProcessListInput()).processes:
            assert row.name
            assert row.pid > 0
            assert row.memory_mb >= 0
            assert row.status

    def test_dead_process_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Dying:
            pid = 999999

            def oneshot(self) -> Any:
                raise sys_process.psutil.NoSuchProcess(pid=999999)

            def cpu_percent(self, interval: Any = None) -> float:
                return 0.0

        real = sys_process.psutil.process_iter
        monkeypatch.setattr(
            sys_process.psutil, "process_iter", lambda *a, **k: [Dying(), *list(real())]
        )
        assert top_processes(ProcessListInput()).processes

    def test_failure_becomes_tool_execution_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("no process table")

        monkeypatch.setattr(sys_process.psutil, "process_iter", boom)
        with pytest.raises(ToolExecutionError) as excinfo:
            top_processes(ProcessListInput())
        assert "running processes" in excinfo.value.speakable


class TestFindProcess:
    def test_finds_the_current_process(self, own_name: str) -> None:
        """This very process is running, so searching for its name must find it."""
        result = find_process(ProcessLookupInput(name=own_name))
        assert result.running is True
        assert result.instance_count >= 1
        assert any(own_name.lower() in row.name.lower() for row in result.instances)

    def test_missing_process_reports_not_running(self) -> None:
        result = find_process(ProcessLookupInput(name="definitely-not-a-real-process-xyz"))
        assert result.running is False
        assert result.instance_count == 0
        assert result.instances == []
        assert result.total_memory_mb == 0

    def test_matching_is_case_insensitive(self, own_name: str) -> None:
        assert find_process(ProcessLookupInput(name=own_name.upper())).running is True

    def test_matching_is_partial(self, own_name: str) -> None:
        assert find_process(ProcessLookupInput(name=own_name[:3])).running is True

    def test_instances_capped_at_five(self) -> None:
        """A single letter matches many processes; the result is still capped."""
        assert len(find_process(ProcessLookupInput(name="e")).instances) <= 5

    def test_empty_name_is_rejected_by_the_schema(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ProcessLookupInput(name="")

    def test_totals_are_summed(self, own_name: str) -> None:
        result = find_process(ProcessLookupInput(name=own_name))
        assert result.total_memory_mb >= max(
            (row.memory_mb for row in result.instances), default=0
        )

    def test_query_is_echoed(self) -> None:
        assert find_process(ProcessLookupInput(name="anything")).query == "anything"

    def test_failure_becomes_tool_execution_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("no process table")

        monkeypatch.setattr(sys_process.psutil, "process_iter", boom)
        with pytest.raises(ToolExecutionError):
            find_process(ProcessLookupInput(name="anything"))


class TestSafety:
    def test_module_never_terminates_a_process(self) -> None:
        """§6 lists no process-control tool. This module must be read-only."""
        from pathlib import Path

        source = Path(sys_process.__file__).read_text(encoding="utf-8")
        for forbidden in (".kill(", ".terminate(", ".suspend(", ".nice(", ".resume("):
            assert forbidden not in source, f"{forbidden} is a mutating call"

    def test_both_tools_are_registered_read_only(self) -> None:
        for name in ("sys.process.top", "sys.process.find"):
            spec = registry.get(name)
            assert spec is not None, name
            assert spec.read_only is True
            assert spec.requires_confirmation is False


class TestRegistration:
    def test_dispatch_top(self) -> None:
        result = registry.dispatch("sys.process.top", {"sort_by": "memory"})
        assert result.ok is True
        json.dumps(result.data)

    def test_dispatch_find(self, own_name: str) -> None:
        result = registry.dispatch("sys.process.find", {"name": own_name})
        assert result.ok is True
        assert result.data["running"] is True

    def test_current_pid_is_visible(self, own_name: str) -> None:
        """Sanity check that we are reading the real process table."""
        result = find_process(ProcessLookupInput(name=own_name))
        assert os.getpid() in {row.pid for row in result.instances}
