"""Process inspection.

Read-only throughout. This module never kills, suspends, or reprioritises a
process; §6 lists no process-control tool on the mutating allowlist.
"""

from __future__ import annotations

import logging
import time
from typing import Literal

import psutil
from pydantic import Field

from jarvis.tools.registry import ToolCategory, ToolInput, ToolOutput, tool
from jarvis.util.errors import ToolExecutionError

__all__ = [
    "ProcessInfo",
    "ProcessListInput",
    "ProcessListOutput",
    "ProcessLookupInput",
    "ProcessLookupOutput",
    "find_process",
    "top_processes",
]

_log = logging.getLogger(__name__)

_MAX_RESULTS = 5
#: psutil needs two samples to compute CPU percent. The first call primes the
#: counters, the interval elapses, then the second call is meaningful.
_PRIME_INTERVAL_S = 0.25


class ProcessInfo(ToolOutput):
    """One process, summarised for speech."""

    name: str = Field(description="Executable name, for example chrome.exe.")
    pid: int = Field(description="Process identifier.")
    cpu_percent: float = Field(
        description="Share of total CPU capacity, 0 to 100, normalised across all cores."
    )
    memory_mb: float = Field(description="Resident set size in megabytes.")
    status: str = Field(description="Process state, for example running or sleeping.")
    running_for_seconds: float | None = Field(
        default=None, description="Seconds since the process started, when readable."
    )


class ProcessListInput(ToolInput):
    """Arguments for listing the busiest processes."""

    sort_by: Literal["cpu", "memory"] = Field(
        default="cpu",
        description=(
            "Rank by cpu for questions like what is using my CPU, or by memory for "
            "questions like what is eating my RAM."
        ),
    )
    limit: int = Field(
        default=_MAX_RESULTS,
        ge=1,
        le=_MAX_RESULTS,
        description="How many processes to return. Capped at 5 because the answer is spoken.",
    )


class ProcessListOutput(ToolOutput):
    """The busiest processes right now."""

    sorted_by: str = Field(description="Which metric the list was ranked by, cpu or memory.")
    total_processes: int = Field(description="Total number of running processes.")
    processes: list[ProcessInfo] = Field(
        default_factory=list, description="Up to 5 processes, busiest first."
    )


class ProcessLookupInput(ToolInput):
    """Arguments for looking up a named process."""

    name: str = Field(
        min_length=1,
        description=(
            "Process name or a fragment of it, case insensitive. For example chrome, "
            "spotify, or code."
        ),
    )


class ProcessLookupOutput(ToolOutput):
    """Result of a named process lookup."""

    query: str = Field(description="The name that was searched for.")
    running: bool = Field(description="Whether any matching process is running.")
    instance_count: int = Field(description="How many matching processes are running.")
    total_cpu_percent: float = Field(description="Combined CPU share of all matches, 0 to 100.")
    total_memory_mb: float = Field(description="Combined resident memory of all matches, in MB.")
    instances: list[ProcessInfo] = Field(
        default_factory=list, description="Up to 5 matching processes, busiest first."
    )


def _iter_processes(prime: bool) -> list[psutil.Process]:
    """Snapshot the process table, optionally priming CPU counters first.

    psutil's ``cpu_percent()`` returns 0.0 on its first call for a process
    because there is no previous sample to diff against. Priming avoids
    reporting a confident, wrong zero for every process.
    """
    procs = list(psutil.process_iter())
    if prime:
        for proc in procs:
            try:
                proc.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        time.sleep(_PRIME_INTERVAL_S)
    return procs


def _describe(proc: psutil.Process, core_count: int) -> ProcessInfo | None:
    """Read one process, or None when it vanished or is not readable."""
    try:
        with proc.oneshot():
            raw_cpu = proc.cpu_percent(interval=None)
            # psutil reports per-core percent, so a fully busy core on an 8-core
            # box reads as 100. Normalise so the number matches what a person
            # sees in Task Manager.
            cpu = raw_cpu / core_count if core_count else raw_cpu
            memory = proc.memory_info().rss / (1024**2)
            started: float | None
            try:
                started = max(0.0, time.time() - proc.create_time())
            except (psutil.AccessDenied, OSError):
                started = None
            return ProcessInfo(
                name=proc.name() or "unknown",
                pid=proc.pid,
                cpu_percent=round(min(100.0, cpu), 1),
                memory_mb=round(memory, 1),
                status=proc.status(),
                running_for_seconds=round(started, 1) if started is not None else None,
            )
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None
    except (OSError, ValueError, TypeError):
        return None


@tool(
    name="sys.process.top",
    description=(
        "The busiest running processes, ranked by CPU or by memory. "
        "Use this for questions like what is using my CPU, what is slowing my machine down, "
        "or what is eating my RAM. CPU percent is normalised across all cores so it matches "
        "Task Manager. Memory is in megabytes. Returns at most 5 processes. Read-only, it "
        "never stops or changes a process."
    ),
    category=ToolCategory.SYSTEM,
    read_only=True,
)
def top_processes(params: ProcessListInput) -> ProcessListOutput:
    """List the busiest processes.

    Args:
        params: Ranking metric and result count.

    Returns:
        The top processes, capped at five.

    Raises:
        ToolExecutionError: The process table could not be read.
    """
    try:
        core_count = psutil.cpu_count(logical=True) or 1
        procs = _iter_processes(prime=params.sort_by == "cpu")
        rows = [row for row in (_describe(p, core_count) for p in procs) if row is not None]
        key = (lambda r: r.cpu_percent) if params.sort_by == "cpu" else (lambda r: r.memory_mb)
        rows.sort(key=key, reverse=True)
        return ProcessListOutput(
            sorted_by=params.sort_by,
            total_processes=len(procs),
            processes=rows[: params.limit],
        )
    except Exception as exc:
        _log.exception("sys.process.top failed")
        raise ToolExecutionError(
            "sys.process.top",
            f"could not read the process table: {exc}",
            speakable="I could not read the list of running processes.",
        ) from exc


@tool(
    name="sys.process.find",
    description=(
        "Check whether a named program is running and how much CPU and memory it is using. "
        "Use this for questions like is Chrome running, is Spotify open, or how much memory "
        "is Discord using. Matching is case insensitive and matches partial names. CPU is a "
        "percent of total capacity, memory is in megabytes, and uptime is in seconds. "
        "Returns at most 5 matching instances. Read-only, it never stops or changes a process."
    ),
    category=ToolCategory.SYSTEM,
    read_only=True,
)
def find_process(params: ProcessLookupInput) -> ProcessLookupOutput:
    """Look up processes by name.

    Args:
        params: The name or fragment to search for.

    Returns:
        Whether it is running, how many instances, and their combined usage.

    Raises:
        ToolExecutionError: The process table could not be read.
    """
    try:
        needle = params.name.strip().lower()
        core_count = psutil.cpu_count(logical=True) or 1
        matches: list[ProcessInfo] = []

        procs = _iter_processes(prime=True)
        for proc in procs:
            try:
                name = (proc.name() or "").lower()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            if needle not in name:
                continue
            row = _describe(proc, core_count)
            if row is not None:
                matches.append(row)

        matches.sort(key=lambda row: row.cpu_percent, reverse=True)
        return ProcessLookupOutput(
            query=params.name,
            running=bool(matches),
            instance_count=len(matches),
            total_cpu_percent=round(min(100.0, sum(m.cpu_percent for m in matches)), 1),
            total_memory_mb=round(sum(m.memory_mb for m in matches), 1),
            instances=matches[:_MAX_RESULTS],
        )
    except Exception as exc:
        _log.exception("sys.process.find failed")
        raise ToolExecutionError(
            "sys.process.find",
            f"could not search the process table: {exc}",
            speakable="I could not search the list of running processes.",
        ) from exc
