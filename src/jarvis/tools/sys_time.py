"""The current date and time.

The §7 Phase 1 gate is "say 'Hey Jarvis, what time is it' and hear a spoken
answer", and no tool could answer it. A language model has no clock, so it
either declined or invented one, which is precisely what §5 forbids: never
invent a metric that did not come from a tool.

Read-only. Reports the machine's local time, so it matches the clock the user
is looking at rather than UTC.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from pydantic import Field

from jarvis.tools.registry import ToolCategory, ToolInput, ToolOutput, tool
from jarvis.util.errors import ToolExecutionError

__all__ = ["TimeInput", "TimeOutput", "current_time"]

_log = logging.getLogger(__name__)


class TimeInput(ToolInput):
    """Arguments for the clock tool."""

    include_date: bool = Field(
        default=True,
        description="Include the date and day of the week as well as the time.",
    )


class TimeOutput(ToolOutput):
    """The current local date and time, phrased for speech."""

    time_spoken: str = Field(
        description=(
            "The time as a person would say it on a 12 hour clock, for example "
            "twenty past three in the afternoon. Read this out rather than the 24 hour form."
        )
    )
    time_24h: str = Field(description="The time as HH:MM on a 24 hour clock.")
    date_spoken: str | None = Field(
        default=None,
        description="The date as a person would say it, for example Tuesday the 6th of August.",
    )
    iso: str = Field(description="Full local timestamp in ISO 8601 form, with offset.")
    day_of_week: str | None = Field(default=None, description="Weekday name, for example Tuesday.")
    timezone_name: str = Field(description="Local timezone abbreviation, for example GMT or IST.")
    utc_offset_hours: float = Field(description="Hours ahead of UTC, negative when behind.")


#: Spoken forms, so the model never has to read digits aloud.
_HOUR_WORDS = (
    "twelve", "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten", "eleven",
)

_MINUTE_WORDS = {
    0: "o'clock", 15: "quarter past", 30: "half past", 45: "quarter to",
}

_ORDINALS = {1: "st", 2: "nd", 3: "rd", 21: "st", 22: "nd", 23: "rd", 31: "st"}


def _part_of_day(hour: int) -> str:
    """Morning, afternoon, evening, or night, by the usual conventions."""
    if hour < 12:
        return "in the morning"
    if hour < 17:
        return "in the afternoon"
    if hour < 21:
        return "in the evening"
    return "at night"


def _hour_phrase(hour: int) -> tuple[str, bool]:
    """Name an hour, and say whether it needs a part of day after it.

    Midnight and midday name themselves. "twelve o'clock in the morning" is
    not how anyone says it, and "twelve in the afternoon" for noon is worse.
    """
    hour %= 24
    if hour == 0:
        return "midnight", False
    if hour == 12:
        return "midday", False
    return _HOUR_WORDS[hour % 12], True


def speak_time(moment: datetime) -> str:
    """Phrase a time the way a person says it.

    ``15:20`` becomes "20 minutes past three in the afternoon". Whole hours,
    quarters, and half hours get their idiomatic forms; everything else is
    "<minutes> past" or "<minutes> to" the nearest hour. Midnight and midday
    are named rather than being called twelve.
    """
    hour, minute = moment.hour, moment.minute

    if minute > 30:
        # Counted towards the next hour, so that is the one to name.
        name, needs_period = _hour_phrase(hour + 1)
        tail = f" {_part_of_day((hour + 1) % 24)}" if needs_period else ""
        remaining = 60 - minute
        lead = _MINUTE_WORDS[45] if minute == 45 else f"{remaining} minutes to"
        return f"{lead} {name}{tail}"

    name, needs_period = _hour_phrase(hour)
    tail = f" {_part_of_day(hour)}" if needs_period else ""
    if minute == 0:
        return f"{name} {_MINUTE_WORDS[0]}{tail}" if needs_period else name
    if minute in _MINUTE_WORDS:
        return f"{_MINUTE_WORDS[minute]} {name}{tail}"
    return f"{minute} minutes past {name}{tail}"


def speak_date(moment: datetime) -> str:
    """Phrase a date the way a person says it, for example Tuesday the 6th of August."""
    day = moment.day
    suffix = _ORDINALS.get(day, "th")
    return f"{moment.strftime('%A')} the {day}{suffix} of {moment.strftime('%B')}"


@tool(
    name="sys.time",
    description=(
        "The current date and time on this machine, in local time. Use this for any question "
        "about what time or day it is, such as what time is it, what is the date, or what day "
        "is it today. Always call this rather than answering from memory: you have no clock of "
        "your own and a guess will be wrong. Hours and minutes are given twice, once phrased "
        "for speech on a 12 hour clock and once as HH:MM on a 24 hour clock, and the timezone "
        "offset is in hours ahead of UTC. Speak the time_spoken and date_spoken fields as "
        "written; they are already phrased for speech. Read-only."
    ),
    category=ToolCategory.SYSTEM,
    read_only=True,
)
def current_time(params: TimeInput) -> TimeOutput:
    """Report the current local date and time.

    Args:
        params: Whether to include the date as well as the time.

    Returns:
        The time phrased for speech, plus machine-readable forms.

    Raises:
        ToolExecutionError: The system clock could not be read.
    """
    try:
        moment = datetime.now().astimezone()
        offset = moment.utcoffset()
        offset_hours = offset.total_seconds() / 3600.0 if offset is not None else 0.0
        name = moment.tzname() or time.tzname[0] or "local time"

        return TimeOutput(
            time_spoken=speak_time(moment),
            time_24h=moment.strftime("%H:%M"),
            date_spoken=speak_date(moment) if params.include_date else None,
            iso=moment.isoformat(timespec="seconds"),
            day_of_week=moment.strftime("%A") if params.include_date else None,
            timezone_name=name,
            utc_offset_hours=round(offset_hours, 2),
        )
    except Exception as exc:
        _log.exception("sys.time failed")
        raise ToolExecutionError(
            "sys.time",
            f"could not read the system clock: {exc}",
            speakable="I could not read the clock.",
        ) from exc


def snapshot() -> dict[str, Any]:
    """Cheap sample for the HUD metrics stream. Never raises."""
    try:
        return {"local_time": datetime.now().astimezone().strftime("%H:%M")}
    except Exception:  # noqa: BLE001 - the HUD must never take down the loop
        return {"local_time": None}


_ = timezone  # re-exported for tests that build fixed-offset moments
