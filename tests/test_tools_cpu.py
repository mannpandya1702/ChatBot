"""T-2.2 verification: sys.cpu."""

from __future__ import annotations

from typing import Any

import pytest

from jarvis.tools import sys_cpu
from jarvis.tools.registry import registry
from jarvis.tools.sys_cpu import CpuInput, cpu_status, format_uptime
from jarvis.util.errors import ToolExecutionError


class TestFormatUptime:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0, "less than a minute"),
            (30, "less than a minute"),
            (60, "1 minute"),
            (120, "2 minutes"),
            (3600, "1 hour"),
            (7200, "2 hours"),
            (3660, "1 hour, 1 minute"),
            (86400, "1 day"),
            (172800, "2 days"),
            (273600, "3 days, 4 hours"),
        ],
    )
    def test_phrasing(self, seconds: float, expected: str) -> None:
        assert format_uptime(seconds) == expected

    def test_negative_is_clamped(self) -> None:
        assert format_uptime(-100) == "less than a minute"

    def test_days_suppress_minutes(self) -> None:
        """Nobody wants "3 days, 4 hours, 17 minutes" read aloud."""
        assert "minute" not in format_uptime(273600 + 1020)


class TestLoadTrend:
    @pytest.mark.parametrize(
        ("percent", "expected"),
        [
            (0, "idle"), (5, "idle"), (20, "light"),
            (45, "moderate"), (70, "heavy"), (99, "saturated"),
        ],
    )
    def test_thresholds(self, percent: float, expected: str) -> None:
        assert sys_cpu._describe_load(percent) == expected


class TestCpuStatus:
    def test_returns_plausible_values(self) -> None:
        result = cpu_status(CpuInput(interval_s=0.1))
        assert 0.0 <= result.cpu_percent <= 100.0
        assert result.logical_cores >= 1
        assert result.physical_cores >= 1
        assert result.logical_cores >= result.physical_cores
        assert result.uptime_seconds > 0
        assert result.uptime_human

    def test_does_not_report_a_confident_zero(self) -> None:
        """psutil returns 0.0 on a first non-blocking call. The interval avoids that."""
        result = cpu_status(CpuInput(interval_s=0.1))
        assert result.load_trend in {"idle", "light", "moderate", "heavy", "saturated"}

    def test_per_core_is_opt_in(self) -> None:
        assert cpu_status(CpuInput(interval_s=0.1)).busiest_cores == []

    def test_per_core_returns_at_most_five(self) -> None:
        result = cpu_status(CpuInput(interval_s=0.1, include_per_core=True))
        assert len(result.busiest_cores) <= 5

    def test_per_core_is_sorted_descending(self) -> None:
        cores = cpu_status(CpuInput(interval_s=0.1, include_per_core=True)).busiest_cores
        assert cores == sorted(cores, key=lambda c: c.percent, reverse=True)

    def test_missing_frequency_is_tolerated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Containers and some VMs do not expose cpu_freq. That is not a failure."""

        def boom() -> Any:
            raise OSError("not supported")

        monkeypatch.setattr(sys_cpu.psutil, "cpu_freq", boom)
        result = cpu_status(CpuInput(interval_s=0.1))
        assert result.frequency_mhz is None

    def test_zero_max_frequency_becomes_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Freq:
            current = 2400.0
            max = 0.0

        monkeypatch.setattr(sys_cpu.psutil, "cpu_freq", Freq)
        assert cpu_status(CpuInput(interval_s=0.1)).max_frequency_mhz is None

    def test_failure_becomes_tool_execution_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args: Any, **kwargs: Any) -> float:
            raise RuntimeError("counters gone")

        monkeypatch.setattr(sys_cpu.psutil, "cpu_percent", boom)
        with pytest.raises(ToolExecutionError) as excinfo:
            cpu_status(CpuInput(interval_s=0.1))
        assert excinfo.value.speakable == "I could not read the processor counters."


class TestRegistration:
    def test_registered_and_read_only(self) -> None:
        spec = registry.get("sys.cpu")
        assert spec is not None
        assert spec.read_only is True
        assert spec.requires_confirmation is False

    def test_description_names_the_units(self) -> None:
        """§5: tool descriptions must carry units."""
        spec = registry.get("sys.cpu")
        assert spec is not None
        lowered = spec.description.lower()
        assert "percent" in lowered
        assert "megahertz" in lowered

    def test_dispatches_through_the_registry(self) -> None:
        result = registry.dispatch("sys.cpu", {"interval_s": 0.1})
        assert result.ok is True
        assert "cpu_percent" in result.data

    def test_output_is_json_serialisable(self) -> None:
        import json

        json.dumps(registry.dispatch("sys.cpu", {"interval_s": 0.1}).data)


class TestSnapshot:
    def test_returns_hud_fields(self) -> None:
        payload = sys_cpu.snapshot()
        assert "cpu_percent" in payload
        assert "logical_cores" in payload

    def test_never_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args: Any, **kwargs: Any) -> float:
            raise RuntimeError("gone")

        monkeypatch.setattr(sys_cpu.psutil, "cpu_percent", boom)
        assert sys_cpu.snapshot()["cpu_percent"] == 0.0
