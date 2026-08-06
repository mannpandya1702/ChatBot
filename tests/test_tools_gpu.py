"""T-2.5 verification: sys.gpu.

Written against a fake pynvml so it runs on any host. The graceful "no NVIDIA
GPU" path is the one that matters most, since it is what the build host and many
target machines actually take.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any

import pytest

from jarvis.tools import sys_gpu
from jarvis.tools.registry import registry
from jarvis.tools.sys_gpu import GpuInput, gpu_status
from jarvis.util.errors import ToolExecutionError


class _NvmlError(Exception):
    """Stands in for pynvml.NVMLError."""


def _make_pynvml(
    *,
    device_count: int = 1,
    name: Any = b"NVIDIA GeForce RTX 4070 Ti",
    util: float | None = 42.0,
    mem: tuple[int, int] | None = (6 * 1024**2 * 1024, 12 * 1024**2 * 1024),
    temp: float | None = 61.0,
    power_mw: float | None = 145_000.0,
    limit_mw: float | None = 285_000.0,
    graphics_mhz: int | None = 2610,
    memory_mhz: int | None = 10501,
    fan: float | None = 38.0,
    init_raises: bool = False,
    processes: list[tuple[int, int]] | None = None,
) -> types.ModuleType:
    """Build a fake pynvml module with configurable sensor availability."""
    module = types.ModuleType("pynvml")
    module.NVML_TEMPERATURE_GPU = 0  # type: ignore[attr-defined]
    module.NVML_CLOCK_GRAPHICS = 0  # type: ignore[attr-defined]
    module.NVML_CLOCK_MEM = 1  # type: ignore[attr-defined]
    state = {"shutdown_calls": 0}

    def nvmlInit() -> None:
        if init_raises:
            raise _NvmlError("driver not loaded")

    def nvmlShutdown() -> None:
        state["shutdown_calls"] += 1

    def unsupported(*_args: Any, **_kwargs: Any) -> Any:
        raise _NvmlError("NVML_ERROR_NOT_SUPPORTED")

    module.nvmlInit = nvmlInit  # type: ignore[attr-defined]
    module.nvmlShutdown = nvmlShutdown  # type: ignore[attr-defined]
    module.nvmlDeviceGetCount = lambda: device_count  # type: ignore[attr-defined]
    module.nvmlDeviceGetHandleByIndex = lambda index: f"handle{index}"  # type: ignore[attr-defined]
    module.nvmlDeviceGetName = lambda _h: name  # type: ignore[attr-defined]

    if util is None:
        module.nvmlDeviceGetUtilizationRates = unsupported  # type: ignore[attr-defined]
    else:
        module.nvmlDeviceGetUtilizationRates = lambda _h: types.SimpleNamespace(  # type: ignore[attr-defined]
            gpu=util, memory=0
        )

    if mem is None:
        module.nvmlDeviceGetMemoryInfo = unsupported  # type: ignore[attr-defined]
    else:
        module.nvmlDeviceGetMemoryInfo = lambda _h: types.SimpleNamespace(  # type: ignore[attr-defined]
            used=mem[0], total=mem[1], free=mem[1] - mem[0]
        )

    module.nvmlDeviceGetTemperature = (  # type: ignore[attr-defined]
        unsupported if temp is None else (lambda _h, _s: temp)
    )
    module.nvmlDeviceGetPowerUsage = (  # type: ignore[attr-defined]
        unsupported if power_mw is None else (lambda _h: power_mw)
    )
    module.nvmlDeviceGetEnforcedPowerLimit = (  # type: ignore[attr-defined]
        unsupported if limit_mw is None else (lambda _h: limit_mw)
    )

    def clock(_h: Any, kind: int) -> int:
        value = graphics_mhz if kind == 0 else memory_mhz
        if value is None:
            raise _NvmlError("NVML_ERROR_NOT_SUPPORTED")
        return value

    module.nvmlDeviceGetClockInfo = clock  # type: ignore[attr-defined]
    module.nvmlDeviceGetFanSpeed = (  # type: ignore[attr-defined]
        unsupported if fan is None else (lambda _h: fan)
    )
    module.nvmlDeviceGetComputeRunningProcesses = lambda _h: [  # type: ignore[attr-defined]
        types.SimpleNamespace(pid=pid, usedGpuMemory=used) for pid, used in (processes or [])
    ]
    module._state = state  # type: ignore[attr-defined]
    return module


@pytest.fixture
def with_nvml(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install a fake pynvml and make has_module report it as present."""

    def _install(**kwargs: Any) -> types.ModuleType:
        module = _make_pynvml(**kwargs)
        monkeypatch.setitem(sys.modules, "pynvml", module)
        monkeypatch.setattr(sys_gpu, "has_module", lambda _name: True)
        return module

    return _install


class TestNoGpu:
    def test_missing_library_is_not_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The build host has no pynvml. That must be a calm spoken answer."""
        monkeypatch.setattr(sys_gpu, "has_module", lambda _name: False)
        result = gpu_status(GpuInput())
        assert result.available is False
        assert result.reason is not None
        assert "not installed" in result.reason
        assert result.temperature_c is None

    def test_no_driver_is_not_an_error(self, with_nvml: Any) -> None:
        with_nvml(init_raises=True)
        result = gpu_status(GpuInput())
        assert result.available is False
        assert result.reason is not None
        assert "driver" in result.reason

    def test_zero_devices_is_not_an_error(self, with_nvml: Any) -> None:
        with_nvml(device_count=0)
        result = gpu_status(GpuInput())
        assert result.available is False
        assert result.reason is not None
        assert "no NVIDIA GPU" in result.reason

    def test_out_of_range_device_explains_itself(self, with_nvml: Any) -> None:
        with_nvml(device_count=1)
        result = gpu_status(GpuInput(device_index=3))
        assert result.available is False
        assert result.reason is not None
        assert "1 NVIDIA GPU" in result.reason

    def test_unavailable_result_dispatches_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An absent GPU is a successful tool call reporting absence."""
        monkeypatch.setattr(sys_gpu, "has_module", lambda _name: False)
        result = registry.dispatch("sys.gpu", {})
        assert result.ok is True
        assert result.data["available"] is False


class TestHappyPath:
    def test_reads_every_sensor(self, with_nvml: Any) -> None:
        with_nvml()
        result = gpu_status(GpuInput())
        assert result.available is True
        assert result.name == "NVIDIA GeForce RTX 4070 Ti"
        assert result.utilization_percent == 42.0
        assert result.memory_used_mb == pytest.approx(6144.0, rel=0.01)
        assert result.memory_total_mb == pytest.approx(12288.0, rel=0.01)
        assert result.memory_percent == pytest.approx(50.0, rel=0.01)
        assert result.temperature_c == 61.0
        assert result.power_draw_w == 145.0
        assert result.power_limit_w == 285.0
        assert result.graphics_clock_mhz == 2610
        assert result.memory_clock_mhz == 10501
        assert result.fan_percent == 38.0

    def test_decodes_a_str_name(self, with_nvml: Any) -> None:
        """Newer drivers return str, older ones return bytes."""
        with_nvml(name="NVIDIA RTX A4000")
        assert gpu_status(GpuInput()).name == "NVIDIA RTX A4000"

    @pytest.mark.parametrize(
        ("celsius", "expected"),
        [(35, "cool"), (49, "cool"), (60, "warm"), (75, "hot"), (90, "critical")],
    )
    def test_thermal_state(self, with_nvml: Any, celsius: float, expected: str) -> None:
        with_nvml(temp=celsius)
        assert gpu_status(GpuInput()).thermal_state == expected

    def test_nvml_is_always_shut_down(self, with_nvml: Any) -> None:
        """Leaking the NVML handle across calls would eventually fail."""
        module = with_nvml()
        gpu_status(GpuInput())
        assert module._state["shutdown_calls"] == 1

    def test_nvml_is_shut_down_even_on_failure(self, with_nvml: Any) -> None:
        module = with_nvml()
        module.nvmlDeviceGetHandleByIndex = lambda _i: (_ for _ in ()).throw(RuntimeError("boom"))
        with pytest.raises(ToolExecutionError):
            gpu_status(GpuInput())
        assert module._state["shutdown_calls"] == 1


class TestPartialSensorSupport:
    def test_missing_temperature_does_not_lose_the_rest(self, with_nvml: Any) -> None:
        """Older cards raise NOT_SUPPORTED per sensor. Keep what did work."""
        with_nvml(temp=None)
        result = gpu_status(GpuInput())
        assert result.available is True
        assert result.temperature_c is None
        assert result.thermal_state is None
        assert result.utilization_percent == 42.0

    def test_missing_fan_does_not_lose_the_rest(self, with_nvml: Any) -> None:
        with_nvml(fan=None)
        result = gpu_status(GpuInput())
        assert result.fan_percent is None
        assert result.memory_used_mb is not None

    def test_missing_clocks_do_not_lose_the_rest(self, with_nvml: Any) -> None:
        with_nvml(graphics_mhz=None, memory_mhz=None)
        result = gpu_status(GpuInput())
        assert result.graphics_clock_mhz is None
        assert result.memory_clock_mhz is None
        assert result.temperature_c == 61.0

    def test_all_sensors_missing_still_reports_available(self, with_nvml: Any) -> None:
        with_nvml(util=None, mem=None, temp=None, power_mw=None, limit_mw=None, fan=None)
        result = gpu_status(GpuInput())
        assert result.available is True
        assert result.name is not None


class TestProcesses:
    def test_processes_are_opt_in(self, with_nvml: Any) -> None:
        with_nvml(processes=[(1, 1024**3)])
        assert gpu_status(GpuInput()).processes == []

    def test_processes_capped_and_sorted(self, with_nvml: Any) -> None:
        with_nvml(processes=[(pid, pid * 1024**2 * 100) for pid in range(1, 9)])
        rows = gpu_status(GpuInput(include_processes=True)).processes
        assert len(rows) <= 5
        assert rows == sorted(rows, key=lambda r: r.vram_mb, reverse=True)

    def test_unresolvable_pid_still_listed(self, with_nvml: Any) -> None:
        with_nvml(processes=[(999_999, 512 * 1024**2)])
        rows = gpu_status(GpuInput(include_processes=True)).processes
        assert len(rows) == 1
        assert rows[0].pid == 999_999

    def test_unsupported_process_list_is_tolerated(self, with_nvml: Any) -> None:
        module = with_nvml()
        module.nvmlDeviceGetComputeRunningProcesses = lambda _h: (_ for _ in ()).throw(
            _NvmlError("NOT_SUPPORTED")
        )
        assert gpu_status(GpuInput(include_processes=True)).processes == []


class TestRegistration:
    def test_registered_and_read_only(self) -> None:
        spec = registry.get("sys.gpu")
        assert spec is not None
        assert spec.read_only is True

    def test_description_names_the_units(self) -> None:
        spec = registry.get("sys.gpu")
        assert spec is not None
        lowered = spec.description.lower()
        assert "celsius" in lowered
        assert "megabytes" in lowered
        assert "watts" in lowered

    def test_description_warns_against_guessing(self) -> None:
        """§5 anti-hallucination: the model must not invent a temperature."""
        spec = registry.get("sys.gpu")
        assert spec is not None
        assert "do not guess" in spec.description.lower()

    def test_dispatch_and_serialise(self, with_nvml: Any) -> None:
        with_nvml()
        result = registry.dispatch("sys.gpu", {})
        assert result.ok is True
        json.dumps(result.data)


class TestSnapshot:
    def test_returns_hud_fields(self) -> None:
        assert set(sys_gpu.snapshot()) == {"gpu_percent", "gpu_temp_c", "gpu_vram_percent"}

    def test_none_when_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys_gpu, "has_module", lambda _name: False)
        assert sys_gpu.snapshot()["gpu_percent"] is None

    def test_populated_when_available(self, with_nvml: Any) -> None:
        with_nvml()
        assert sys_gpu.snapshot()["gpu_temp_c"] == 61.0
