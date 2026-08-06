"""RAM and swap telemetry."""

from __future__ import annotations

import logging
from typing import Any

import psutil
from pydantic import Field

from jarvis.tools.registry import ToolCategory, ToolInput, ToolOutput, tool
from jarvis.util.errors import ToolExecutionError

__all__ = ["MemoryInput", "MemoryOutput", "ProcessMemory", "memory_status"]

_log = logging.getLogger(__name__)

#: §5 caps spoken list results at five items.
_MAX_PROCESSES = 5


class ProcessMemory(ToolOutput):
    """One process's memory footprint."""

    name: str = Field(description="Process executable name, for example chrome.exe.")
    pid: int = Field(description="Process identifier.")
    memory_mb: float = Field(description="Resident set size in megabytes.")
    memory_percent: float = Field(description="Share of total RAM, 0 to 100.")


class MemoryInput(ToolInput):
    """Arguments for the memory status tool."""

    include_top_processes: bool = Field(
        default=True,
        description="Include the largest memory consumers, at most 5.",
    )


class MemoryOutput(ToolOutput):
    """Current memory state. Sizes are gigabytes unless a field says otherwise."""

    total_gb: float = Field(description="Installed RAM in gigabytes.")
    used_gb: float = Field(description="RAM in use in gigabytes.")
    available_gb: float = Field(
        description="RAM available for new allocations in gigabytes. This is the number a "
        "user means by free memory, not the raw free counter."
    )
    percent_used: float = Field(description="Share of RAM in use, 0 to 100.")
    pressure: str = Field(
        description="One of comfortable, moderate, tight, critical. A plain summary to speak."
    )
    swap_total_gb: float = Field(description="Total swap or page file size in gigabytes.")
    swap_used_gb: float = Field(description="Swap in use in gigabytes.")
    swap_percent: float = Field(description="Share of swap in use, 0 to 100.")
    top_processes: list[ProcessMemory] = Field(
        default_factory=list, description="Up to 5 largest memory consumers, biggest first."
    )


def _gb(value: float) -> float:
    """Bytes to gigabytes, rounded for speech."""
    return round(value / (1024**3), 2)


def _describe_pressure(percent: float) -> str:
    """Turn a memory percentage into a word the assistant can say."""
    if percent < 50:
        return "comfortable"
    if percent < 75:
        return "moderate"
    if percent < 90:
        return "tight"
    return "critical"


def _top_memory_processes(limit: int = _MAX_PROCESSES) -> list[ProcessMemory]:
    """Largest processes by resident memory.

    Processes die while being enumerated, which is normal and not an error, so
    per-process failures are skipped rather than propagated.
    """
    rows: list[ProcessMemory] = []
    for proc in psutil.process_iter(["name", "pid", "memory_info", "memory_percent"]):
        try:
            info = proc.info
            mem = info.get("memory_info")
            if mem is None:
                continue
            rows.append(
                ProcessMemory(
                    name=info.get("name") or "unknown",
                    pid=int(info.get("pid", 0)),
                    memory_mb=round(mem.rss / (1024**2), 1),
                    memory_percent=round(float(info.get("memory_percent") or 0.0), 1),
                )
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except (ValueError, TypeError, AttributeError):
            continue
    rows.sort(key=lambda row: row.memory_mb, reverse=True)
    return rows[:limit]


@tool(
    name="sys.memory",
    description=(
        "Current RAM and swap usage, plus the largest memory consumers. "
        "Use this for questions like how much memory is free, am I running out of RAM, "
        "what is using my memory, or how much RAM do I have. "
        "Sizes are in gigabytes, process sizes in megabytes, percentages 0 to 100. "
        "Read-only."
    ),
    category=ToolCategory.SYSTEM,
    read_only=True,
)
def memory_status(params: MemoryInput) -> MemoryOutput:
    """Read current RAM and swap state.

    Args:
        params: Whether to include the top consumers.

    Returns:
        A speakable summary of memory usage.

    Raises:
        ToolExecutionError: psutil could not read the memory counters.
    """
    try:
        virtual = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return MemoryOutput(
            total_gb=_gb(virtual.total),
            used_gb=_gb(virtual.total - virtual.available),
            available_gb=_gb(virtual.available),
            percent_used=round(float(virtual.percent), 1),
            pressure=_describe_pressure(float(virtual.percent)),
            swap_total_gb=_gb(swap.total),
            swap_used_gb=_gb(swap.used),
            swap_percent=round(float(swap.percent), 1),
            top_processes=_top_memory_processes() if params.include_top_processes else [],
        )
    except Exception as exc:
        _log.exception("sys.memory failed")
        raise ToolExecutionError(
            "sys.memory",
            f"psutil could not read memory counters: {exc}",
            speakable="I could not read the memory counters.",
        ) from exc


def snapshot() -> dict[str, Any]:
    """Cheap sample for the HUD metrics stream."""
    try:
        virtual = psutil.virtual_memory()
        return {
            "memory_percent": round(float(virtual.percent), 1),
            "memory_used_gb": _gb(virtual.total - virtual.available),
            "memory_total_gb": _gb(virtual.total),
        }
    except Exception:  # noqa: BLE001 - the HUD must never take down the loop
        _log.debug("memory snapshot failed", exc_info=True)
        return {"memory_percent": 0.0, "memory_used_gb": 0.0, "memory_total_gb": 0.0}
