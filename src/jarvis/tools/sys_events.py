"""Recent Windows event log errors and warnings.

T-2.10: System and Application logs, last 24 hours, capped at 5, summarised.
Read-only. This module reads the event log and never clears, writes, or
subscribes to it.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import Field

from jarvis.tools.registry import ToolCategory, ToolInput, ToolOutput, tool
from jarvis.util.errors import ToolExecutionError
from jarvis.util.platform import is_windows

__all__ = ["EventEntry", "EventsInput", "EventsOutput", "recent_events"]

_log = logging.getLogger(__name__)

_MAX_EVENTS = 5
_QUERY_TIMEOUT_S = 30.0

#: Get-WinEvent levels: 1 critical, 2 error, 3 warning.
_LEVEL_NAMES = {1: "critical", 2: "error", 3: "warning", 4: "information"}


class EventEntry(ToolOutput):
    """One event log record, trimmed for speech."""

    log: str = Field(description="Which log it came from, System or Application.")
    level: str = Field(description="One of critical, error, warning.")
    source: str = Field(
        description="The provider that logged it, for example disk or Kernel-Power."
    )
    event_id: int = Field(description="Numeric event identifier.")
    when: str = Field(description="ISO 8601 timestamp of the event.")
    message: str = Field(description="First line of the message, truncated for speech.")


class EventsInput(ToolInput):
    """Arguments for the event log query."""

    hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="How far back to look, in hours. Defaults to the last 24 hours.",
    )
    level: Literal["error", "warning", "all"] = Field(
        default="all",
        description=(
            "Filter by severity. Use error for questions about crashes and failures, "
            "all to include warnings."
        ),
    )


class EventsOutput(ToolOutput):
    """Recent event log activity, summarised."""

    available: bool = Field(
        description="False when the event log cannot be read, for example off Windows."
    )
    reason: str | None = Field(default=None, description="Why the query is unavailable.")
    window_hours: int = Field(default=24, description="How far back the query looked.")
    error_count: int = Field(default=0, description="Errors and criticals in the window.")
    warning_count: int = Field(default=0, description="Warnings in the window.")
    summary: str = Field(
        default="",
        description="One spoken sentence describing what was found, safe to read aloud verbatim.",
    )
    top_sources: list[str] = Field(
        default_factory=list, description="Up to 3 providers responsible for the most entries."
    )
    events: list[EventEntry] = Field(
        default_factory=list, description="Up to 5 most recent entries, newest first."
    )


# Fixed, code-authored PowerShell. Only the integers below are interpolated, and
# both are schema-bounded ints, never strings from an LLM (§6).
_QUERY_TEMPLATE = (
    "$since = (Get-Date).AddHours(-{hours}); "
    "Get-WinEvent -FilterHashtable @{{ LogName=@('System','Application'); "
    "Level=@({levels}); StartTime=$since }} -MaxEvents 200 -ErrorAction SilentlyContinue "
    "| Select-Object LogName,Id,LevelDisplayName,Level,ProviderName,TimeCreated,Message "
    "| ConvertTo-Json -Compress -Depth 3"
)


def _levels_for(level: str) -> str:
    """Map the filter choice onto Get-WinEvent level numbers."""
    if level == "error":
        return "1,2"
    if level == "warning":
        return "3"
    return "1,2,3"


def _run_query(hours: int, level: str) -> str | None:
    """Execute the event log query. Returns stdout, or None on failure."""
    import shutil

    exe = shutil.which("powershell.exe") or shutil.which("pwsh")
    if exe is None:
        return None

    script = _QUERY_TEMPLATE.format(hours=int(hours), levels=_levels_for(level))
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, literal script, int-only interpolation
            [exe, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=_QUERY_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _log.debug("event log query failed: %s", exc)
        return None
    if completed.returncode != 0:
        _log.debug("event log query returned %s", completed.returncode)
        return None
    return completed.stdout


def _parse(raw: str) -> list[dict[str, Any]]:
    """ConvertTo-Json emits an object for one record and an array for many."""
    text = raw.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        _log.debug("event log output was not valid JSON")
        return []
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def _timestamp(value: Any) -> str:
    """Normalise PowerShell's date representation to ISO 8601.

    ConvertTo-Json renders DateTime as "/Date(1699999999999)/" in Windows
    PowerShell and as an ISO string in PowerShell 7. Handle both.
    """
    if isinstance(value, str):
        if value.startswith("/Date("):
            digits = "".join(ch for ch in value if ch.isdigit() or ch == "-")
            try:
                return datetime.fromtimestamp(int(digits) / 1000, tz=UTC).isoformat()
            except (ValueError, OSError, OverflowError):
                return value
        return value
    return str(value)


def _summarise(errors: int, warnings: int, sources: list[str], hours: int) -> str:
    """Build the sentence the assistant reads aloud."""
    window = "the last day" if hours == 24 else f"the last {hours} hours"
    if errors == 0 and warnings == 0:
        return f"No errors or warnings in {window}."

    parts: list[str] = []
    if errors:
        parts.append(f"{errors} error{'s' if errors != 1 else ''}")
    if warnings:
        parts.append(f"{warnings} warning{'s' if warnings != 1 else ''}")
    sentence = f"{' and '.join(parts)} in {window}."
    if sources:
        sentence += f" Mostly from {sources[0]}."
    return sentence


@tool(
    name="sys.events",
    description=(
        "Recent Windows System and Application event log errors and warnings, by default from "
        "the last 24 hours. Use this for questions like has anything crashed, are there any "
        "errors, why did my machine restart, or is anything wrong. Returns counts, the "
        "providers responsible, and at most 5 recent entries. Read-only, it never clears the "
        "log."
    ),
    category=ToolCategory.SYSTEM,
    read_only=True,
    requires_windows=True,
)
def recent_events(params: EventsInput) -> EventsOutput:
    """Query the Windows event log.

    Args:
        params: Time window and severity filter.

    Returns:
        A summarised view, or ``available=False`` with a reason.

    Raises:
        ToolExecutionError: The query result could not be processed at all.
    """
    if not is_windows():
        return EventsOutput(
            available=False,
            reason="the Windows event log only exists on Windows",
            window_hours=params.hours,
            summary="I can only read the event log on Windows.",
        )

    raw = _run_query(params.hours, params.level)
    if raw is None:
        return EventsOutput(
            available=False,
            reason="the event log query did not run",
            window_hours=params.hours,
            summary="I could not read the event log.",
        )

    try:
        records = _parse(raw)
        # Newest first. PowerShell returns newest first already, but sorting
        # makes that independent of the backend's behaviour.
        records.sort(key=lambda r: str(r.get("TimeCreated", "")), reverse=True)

        errors = 0
        warnings = 0
        sources: Counter[str] = Counter()
        entries: list[EventEntry] = []

        for record in records:
            level_num = int(record.get("Level", 0) or 0)
            name = str(record.get("LevelDisplayName") or _LEVEL_NAMES.get(level_num, "unknown"))
            lowered = name.lower()
            if level_num in (1, 2) or lowered in ("error", "critical"):
                errors += 1
            elif level_num == 3 or lowered == "warning":
                warnings += 1

            provider = str(record.get("ProviderName") or "unknown")
            sources[provider] += 1

            if len(entries) < _MAX_EVENTS:
                message = str(record.get("Message") or "").strip()
                first_line = message.splitlines()[0] if message else "no message"
                entries.append(
                    EventEntry(
                        log=str(record.get("LogName") or "unknown"),
                        level=lowered if lowered != "unknown" else "error",
                        source=provider,
                        event_id=int(record.get("Id", 0) or 0),
                        when=_timestamp(record.get("TimeCreated")),
                        message=first_line[:200],
                    )
                )

        top = [name for name, _count in sources.most_common(3)]
        return EventsOutput(
            available=True,
            window_hours=params.hours,
            error_count=errors,
            warning_count=warnings,
            summary=_summarise(errors, warnings, top, params.hours),
            top_sources=top,
            events=entries,
        )
    except Exception as exc:
        _log.exception("sys.events failed")
        raise ToolExecutionError(
            "sys.events",
            f"could not process the event log output: {exc}",
            speakable="I could not read the event log.",
        ) from exc


def cutoff(hours: int, now: datetime | None = None) -> datetime:
    """Start of the query window. Exposed for testing the boundary arithmetic."""
    return (now or datetime.now(tz=UTC)) - timedelta(hours=hours)
