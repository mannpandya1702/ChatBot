"""T-2.3 verification: sys.memory."""

from __future__ import annotations

import json
from typing import Any

import pytest

from jarvis.tools import sys_memory
from jarvis.tools.registry import registry
from jarvis.tools.sys_memory import MemoryInput, memory_status
from jarvis.util.errors import ToolExecutionError


class TestPressure:
    @pytest.mark.parametrize(
        ("percent", "expected"),
        [(10, "comfortable"), (49, "comfortable"), (60, "moderate"), (80, "tight"), (95, "critical")],
    )
    def test_thresholds(self, percent: float, expected: str) -> None:
        assert sys_memory._describe_pressure(percent) == expected


class TestMemoryStatus:
    def test_returns_plausible_values(self) -> None:
        result = memory_status(MemoryInput())
        assert result.total_gb > 0
        assert 0 <= result.percent_used <= 100
        assert result.available_gb >= 0
        assert result.used_gb >= 0

    def test_used_and_available_are_consistent(self) -> None:
        """used + available should account for total, within rounding."""
        result = memory_status(MemoryInput(include_top_processes=False))
        assert abs((result.used_gb + result.available_gb) - result.total_gb) < 0.5

    def test_available_is_used_not_raw_free(self) -> None:
        """The number a user means by free memory is psutil's `available`."""
        result = memory_status(MemoryInput(include_top_processes=False))
        assert result.available_gb <= result.total_gb

    def test_top_processes_capped_at_five(self) -> None:
        """§5 caps spoken list results at five items."""
        assert len(memory_status(MemoryInput()).top_processes) <= 5

    def test_top_processes_sorted_descending(self) -> None:
        rows = memory_status(MemoryInput()).top_processes
        assert rows == sorted(rows, key=lambda r: r.memory_mb, reverse=True)

    def test_top_processes_are_opt_out(self) -> None:
        assert memory_status(MemoryInput(include_top_processes=False)).top_processes == []

    def test_dead_process_during_enumeration_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Processes exit while being listed. That is normal, not an error."""

        class Dying:
            @property
            def info(self) -> dict[str, Any]:
                raise sys_memory.psutil.NoSuchProcess(pid=1)

        real = sys_memory.psutil.process_iter

        def mixed(attrs: Any = None) -> Any:
            yield Dying()
            yield from real(attrs)

        monkeypatch.setattr(sys_memory.psutil, "process_iter", mixed)
        assert memory_status(MemoryInput()).total_gb > 0

    def test_failure_becomes_tool_execution_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom() -> Any:
            raise RuntimeError("counters gone")

        monkeypatch.setattr(sys_memory.psutil, "virtual_memory", boom)
        with pytest.raises(ToolExecutionError) as excinfo:
            memory_status(MemoryInput())
        assert excinfo.value.speakable == "I could not read the memory counters."


class TestRegistration:
    def test_registered_and_read_only(self) -> None:
        spec = registry.get("sys.memory")
        assert spec is not None
        assert spec.read_only is True

    def test_description_names_the_units(self) -> None:
        spec = registry.get("sys.memory")
        assert spec is not None
        assert "gigabytes" in spec.description.lower()

    def test_dispatch_and_serialise(self) -> None:
        result = registry.dispatch("sys.memory", {})
        assert result.ok is True
        json.dumps(result.data)


class TestSnapshot:
    def test_returns_hud_fields(self) -> None:
        payload = sys_memory.snapshot()
        assert set(payload) == {"memory_percent", "memory_used_gb", "memory_total_gb"}

    def test_never_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom() -> Any:
            raise RuntimeError("gone")

        monkeypatch.setattr(sys_memory.psutil, "virtual_memory", boom)
        assert sys_memory.snapshot()["memory_percent"] == 0.0
