"""NVIDIA GPU telemetry via NVML.

The graceful "no NVIDIA GPU detected" path matters more than the happy path
here. A machine with an AMD card, an integrated GPU, or no discrete GPU at all
must get a calm spoken answer, not an exception. NVML is imported lazily so this
module still loads on a machine with no NVIDIA driver at all (§0b).
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from pydantic import Field

from jarvis.tools.registry import ToolCategory, ToolInput, ToolOutput, tool
from jarvis.util.errors import ToolExecutionError
from jarvis.util.platform import has_module

__all__ = ["GpuInput", "GpuOutput", "GpuProcess", "gpu_status"]

_log = logging.getLogger(__name__)

_MAX_PROCESSES = 5


class GpuProcess(ToolOutput):
    """One process holding GPU memory."""

    name: str = Field(description="Process executable name.")
    pid: int = Field(description="Process identifier.")
    vram_mb: float = Field(description="Video memory held by this process, in megabytes.")


class GpuInput(ToolInput):
    """Arguments for the GPU status tool."""

    include_processes: bool = Field(
        default=False,
        description="Include the processes currently using video memory, at most 5.",
    )
    device_index: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Which GPU to read when the machine has more than one. Zero is the first.",
    )


class GpuOutput(ToolOutput):
    """Current GPU state, or an explanation of why it is unavailable."""

    available: bool = Field(
        description="False when there is no NVIDIA GPU or no driver. Every other field is then "
        "unset and the reason field explains why."
    )
    reason: str | None = Field(
        default=None, description="Why GPU telemetry is unavailable, when it is."
    )
    name: str | None = Field(default=None, description="GPU model name.")
    device_count: int = Field(default=0, description="How many NVIDIA GPUs are present.")
    utilization_percent: float | None = Field(
        default=None, description="Core utilisation, 0 to 100."
    )
    memory_used_mb: float | None = Field(default=None, description="Video memory used, megabytes.")
    memory_total_mb: float | None = Field(
        default=None, description="Total video memory, megabytes."
    )
    memory_percent: float | None = Field(
        default=None, description="Share of video memory in use, 0 to 100."
    )
    temperature_c: float | None = Field(
        default=None, description="Core temperature in degrees Celsius."
    )
    power_draw_w: float | None = Field(default=None, description="Current power draw in watts.")
    power_limit_w: float | None = Field(default=None, description="Power limit in watts.")
    graphics_clock_mhz: int | None = Field(
        default=None, description="Current graphics clock in megahertz."
    )
    memory_clock_mhz: int | None = Field(
        default=None, description="Current memory clock in megahertz."
    )
    fan_percent: float | None = Field(
        default=None, description="Fan speed as a percentage of maximum, when the card reports it."
    )
    thermal_state: str | None = Field(
        default=None, description="One of cool, warm, hot, critical."
    )
    processes: list[GpuProcess] = Field(
        default_factory=list, description="Up to 5 processes using video memory."
    )


def _describe_temperature(celsius: float) -> str:
    """Classify a GPU core temperature for speech."""
    if celsius < 50:
        return "cool"
    if celsius < 70:
        return "warm"
    if celsius < 84:
        return "hot"
    return "critical"


def _unavailable(reason: str) -> GpuOutput:
    """Build the calm no-GPU answer."""
    return GpuOutput(available=False, reason=reason)


def _decode(value: Any) -> str:
    """NVML returns bytes on some driver versions and str on others."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


@tool(
    name="sys.gpu",
    description=(
        "NVIDIA GPU utilisation, video memory, temperature, power draw, and clocks. "
        "Use this for questions like how hot is my GPU, how much VRAM is free, is the "
        "graphics card busy, or what is using my GPU. Temperature is in degrees Celsius, "
        "memory in megabytes, power in watts, clocks in megahertz. If the machine has no "
        "NVIDIA GPU the result says so; do not guess a number in that case. Read-only."
    ),
    category=ToolCategory.SYSTEM,
    read_only=True,
)
def gpu_status(params: GpuInput) -> GpuOutput:
    """Read NVIDIA GPU telemetry.

    Args:
        params: Which device to read and whether to list processes.

    Returns:
        Populated telemetry, or ``available=False`` with a reason. A machine
        without an NVIDIA GPU is an ordinary outcome, not an error.

    Raises:
        ToolExecutionError: NVML was present and initialised but then failed in
            a way that is not simply "no GPU here".
    """
    if not has_module("pynvml"):
        return _unavailable("the NVIDIA management library is not installed")

    try:
        import pynvml
    except ImportError:
        return _unavailable("the NVIDIA management library is not installed")

    try:
        pynvml.nvmlInit()
    except Exception as exc:  # noqa: BLE001 - any init failure means no telemetry
        _log.debug("nvmlInit failed: %s", exc)
        return _unavailable("no NVIDIA driver is loaded on this machine")

    try:
        count = int(pynvml.nvmlDeviceGetCount())
        if count == 0:
            return _unavailable("no NVIDIA GPU was detected")
        if params.device_index >= count:
            return _unavailable(
                f"this machine has {count} NVIDIA GPU"
                f"{'s' if count != 1 else ''}, so there is no device {params.device_index}"
            )

        handle = pynvml.nvmlDeviceGetHandleByIndex(params.device_index)
        result = GpuOutput(available=True, device_count=count)
        result.name = _decode(pynvml.nvmlDeviceGetName(handle))

        # Each reading is optional. Older cards and some driver versions raise
        # NVML_ERROR_NOT_SUPPORTED for individual sensors, which must not lose
        # the readings that did work.
        def attempt(label: str, fn: Any) -> Any:
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - per-sensor degradation
                _log.debug("gpu sensor %s unavailable: %s", label, exc)
                return None

        util = attempt("utilization", lambda: pynvml.nvmlDeviceGetUtilizationRates(handle))
        if util is not None:
            result.utilization_percent = round(float(util.gpu), 1)

        mem = attempt("memory", lambda: pynvml.nvmlDeviceGetMemoryInfo(handle))
        if mem is not None:
            used_mb = mem.used / (1024**2)
            total_mb = mem.total / (1024**2)
            result.memory_used_mb = round(used_mb, 1)
            result.memory_total_mb = round(total_mb, 1)
            result.memory_percent = round(100.0 * used_mb / total_mb, 1) if total_mb else None

        temp = attempt(
            "temperature",
            lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
        )
        if temp is not None:
            result.temperature_c = round(float(temp), 1)
            result.thermal_state = _describe_temperature(float(temp))

        power = attempt("power", lambda: pynvml.nvmlDeviceGetPowerUsage(handle))
        if power is not None:
            result.power_draw_w = round(float(power) / 1000.0, 1)

        limit = attempt("power_limit", lambda: pynvml.nvmlDeviceGetEnforcedPowerLimit(handle))
        if limit is not None:
            result.power_limit_w = round(float(limit) / 1000.0, 1)

        gclock = attempt(
            "graphics_clock",
            lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS),
        )
        if gclock is not None:
            result.graphics_clock_mhz = int(gclock)

        mclock = attempt(
            "memory_clock", lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        )
        if mclock is not None:
            result.memory_clock_mhz = int(mclock)

        fan = attempt("fan", lambda: pynvml.nvmlDeviceGetFanSpeed(handle))
        if fan is not None:
            result.fan_percent = round(float(fan), 1)

        if params.include_processes:
            result.processes = _gpu_processes(pynvml, handle)

        return result
    except Exception as exc:
        _log.exception("sys.gpu failed after NVML initialised")
        raise ToolExecutionError(
            "sys.gpu",
            f"NVML initialised but reading the device failed: {exc}",
            speakable="I could not read the graphics card sensors.",
        ) from exc
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001 - shutdown failure is not interesting
            _log.debug("nvmlShutdown failed", exc_info=True)


def _gpu_processes(pynvml: Any, handle: Any) -> list[GpuProcess]:
    """Processes holding video memory, resolved to names via psutil."""
    import psutil

    rows: list[GpuProcess] = []
    try:
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    except Exception:  # noqa: BLE001 - not supported on every card
        _log.debug("gpu process list unavailable", exc_info=True)
        return rows

    for entry in procs:
        name = f"pid {entry.pid}"
        with contextlib.suppress(
            psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError
        ):
            name = psutil.Process(entry.pid).name()
        used = getattr(entry, "usedGpuMemory", None)
        rows.append(
            GpuProcess(
                name=name,
                pid=int(entry.pid),
                vram_mb=round(float(used) / (1024**2), 1) if used else 0.0,
            )
        )
    rows.sort(key=lambda row: row.vram_mb, reverse=True)
    return rows[:_MAX_PROCESSES]


def snapshot() -> dict[str, Any]:
    """Cheap sample for the HUD metrics stream."""
    try:
        result = gpu_status(GpuInput())
        if not result.available:
            return {"gpu_percent": None, "gpu_temp_c": None, "gpu_vram_percent": None}
        return {
            "gpu_percent": result.utilization_percent,
            "gpu_temp_c": result.temperature_c,
            "gpu_vram_percent": result.memory_percent,
        }
    except Exception:  # noqa: BLE001 - the HUD must never take down the loop
        _log.debug("gpu snapshot failed", exc_info=True)
        return {"gpu_percent": None, "gpu_temp_c": None, "gpu_vram_percent": None}
