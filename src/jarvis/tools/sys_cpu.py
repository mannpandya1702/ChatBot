"""CPU telemetry.

Reference implementation for the ``sys_*`` tool family. The shape here is the
one every other system tool follows:

* A pydantic input model with described, unit-bearing fields.
* A pydantic output model that is small and phrasable. CLAUDE.md §5 is explicit:
  return ``{"cpu_percent": 43.2, "top_process": "chrome.exe"}``, not a table.
* A ``@tool`` registration whose description tells the model exactly when to call
  it and what units come back, because schema quality drives tool-calling
  reliability more than model size does.
* Read-only. No writes, no kills, no config changes (§6).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import psutil
from pydantic import Field

from jarvis.tools.registry import ToolCategory, ToolInput, ToolOutput, tool
from jarvis.util.errors import ToolExecutionError

__all__ = ["CoreLoad", "CpuInput", "CpuOutput", "cpu_status", "format_uptime"]

_log = logging.getLogger(__name__)

#: psutil returns 0.0 for the first non-blocking cpu_percent() call because it
#: has no previous sample to diff against. A short blocking interval avoids
#: reporting a confident, wrong zero.
_DEFAULT_INTERVAL_S = 0.3


class CoreLoad(ToolOutput):
    """Utilisation of one logical core."""

    core: int = Field(description="Logical core index, zero based.")
    percent: float = Field(description="Utilisation of this core, 0 to 100.")


class CpuInput(ToolInput):
    """Arguments for the CPU status tool."""

    include_per_core: bool = Field(
        default=False,
        description=(
            "Include the busiest individual cores. Only ask for this when the user "
            "specifically wants per-core detail; the overall percentage is usually enough."
        ),
    )
    interval_s: float = Field(
        default=_DEFAULT_INTERVAL_S,
        ge=0.1,
        le=2.0,
        description="Sampling window in seconds. Longer is more accurate but slower.",
    )


class CpuOutput(ToolOutput):
    """Current processor state. All percentages are 0 to 100."""

    cpu_percent: float = Field(description="Overall CPU utilisation, percent.")
    physical_cores: int = Field(description="Number of physical cores.")
    logical_cores: int = Field(description="Number of logical cores, including SMT threads.")
    frequency_mhz: float | None = Field(
        default=None, description="Current clock speed in megahertz, if the platform reports it."
    )
    max_frequency_mhz: float | None = Field(
        default=None, description="Maximum rated clock speed in megahertz."
    )
    load_trend: str = Field(
        description="One of idle, light, moderate, heavy, saturated. A plain summary to speak."
    )
    uptime_seconds: float = Field(description="Seconds since the machine booted.")
    uptime_human: str = Field(description="Uptime phrased for speech, for example 3 days, 4 hours.")
    busiest_cores: list[CoreLoad] = Field(
        default_factory=list,
        description="Up to 5 busiest logical cores, only when per-core detail was requested.",
    )


def _describe_load(percent: float) -> str:
    """Turn a percentage into a word the assistant can say."""
    if percent < 10:
        return "idle"
    if percent < 30:
        return "light"
    if percent < 60:
        return "moderate"
    if percent < 85:
        return "heavy"
    return "saturated"


def format_uptime(seconds: float) -> str:
    """Phrase a duration the way a person says it.

    Deliberately coarse. "3 days, 4 hours" is what someone wants to hear; the
    seconds are noise once you are past a minute.
    """
    seconds = max(0.0, seconds)
    days, rest = divmod(int(seconds), 86_400)
    hours, rest = divmod(rest, 3_600)
    minutes = rest // 60

    parts: list[str] = []
    if days:
        parts.append(f"{days} day" if days == 1 else f"{days} days")
    if hours:
        parts.append(f"{hours} hour" if hours == 1 else f"{hours} hours")
    if minutes and not days:
        parts.append(f"{minutes} minute" if minutes == 1 else f"{minutes} minutes")
    if not parts:
        return "less than a minute"
    return ", ".join(parts)


@tool(
    name="sys.cpu",
    description=(
        "Current CPU utilisation, core count, clock speed, and system uptime. "
        "Use this for questions like how busy is the CPU, what is the processor doing, "
        "how fast is the CPU running, or how long has the machine been up. "
        "Percentages are 0 to 100, frequency is in megahertz, uptime is in seconds "
        "with a spoken form alongside it. Read-only."
    ),
    category=ToolCategory.SYSTEM,
    read_only=True,
)
def cpu_status(params: CpuInput) -> CpuOutput:
    """Read current CPU state.

    Args:
        params: Sampling options.

    Returns:
        A speakable summary of processor load, topology, clock, and uptime.

    Raises:
        ToolExecutionError: psutil could not read the processor counters.
    """
    try:
        overall = float(psutil.cpu_percent(interval=params.interval_s))

        logical = psutil.cpu_count(logical=True) or 0
        physical = psutil.cpu_count(logical=False) or logical

        frequency_mhz: float | None = None
        max_frequency_mhz: float | None = None
        try:
            freq = psutil.cpu_freq()
            if freq is not None:
                frequency_mhz = round(float(freq.current), 1)
                # A reported max of 0 means "unknown", not "zero megahertz".
                max_frequency_mhz = round(float(freq.max), 1) or None
        except (OSError, AttributeError, NotImplementedError):
            # Common in containers and on some virtualised hosts. Not fatal.
            _log.debug("cpu frequency is unavailable on this platform")

        uptime = max(0.0, time.time() - psutil.boot_time())

        busiest: list[CoreLoad] = []
        if params.include_per_core:
            per_core = psutil.cpu_percent(interval=None, percpu=True)
            ranked = sorted(enumerate(per_core), key=lambda item: item[1], reverse=True)
            busiest = [
                CoreLoad(core=index, percent=round(float(value), 1)) for index, value in ranked[:5]
            ]

        return CpuOutput(
            cpu_percent=round(overall, 1),
            physical_cores=physical,
            logical_cores=logical,
            frequency_mhz=frequency_mhz,
            max_frequency_mhz=max_frequency_mhz,
            load_trend=_describe_load(overall),
            uptime_seconds=round(uptime, 1),
            uptime_human=format_uptime(uptime),
            busiest_cores=busiest,
        )
    except ToolExecutionError:
        raise
    except Exception as exc:
        _log.exception("sys.cpu failed")
        raise ToolExecutionError(
            "sys.cpu",
            f"psutil could not read processor counters: {exc}",
            speakable="I could not read the processor counters.",
        ) from exc


def snapshot() -> dict[str, Any]:
    """Cheap non-blocking sample for the HUD metrics stream.

    Deliberately separate from the tool: the HUD polls at 30 Hz and must never
    pay the blocking sampling interval the tool uses.
    """
    try:
        return {
            "cpu_percent": round(float(psutil.cpu_percent(interval=None)), 1),
            "logical_cores": psutil.cpu_count(logical=True) or 0,
        }
    except Exception:  # noqa: BLE001 - the HUD must never take down the loop
        _log.debug("cpu snapshot failed", exc_info=True)
        return {"cpu_percent": 0.0, "logical_cores": 0}
