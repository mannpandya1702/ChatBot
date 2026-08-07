"""SQLite-backed reminders and timers with Windows toast notifications.

T-4.6, gated: ``reminders.create`` and ``reminders.delete`` are on the §6
mutating allowlist. Listing is read-only and is not gated.

The scheduler runs on one background thread that sleeps until the next due
reminder rather than polling, so an idle assistant costs nothing.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import Field

from jarvis.config import JarvisConfig, get_config
from jarvis.tools.registry import ToolCategory, ToolInput, ToolOutput, tool
from jarvis.util.errors import ToolExecutionError
from jarvis.util.platform import is_windows

__all__ = [
    "Reminder",
    "ReminderStore",
    "create_reminder",
    "delete_reminder",
    "list_reminders",
    "parse_when",
]

_log = logging.getLogger(__name__)

_MAX_RESULTS = 5
_MAX_TEXT_CHARS = 200

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reminders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT    NOT NULL,
    due_at      REAL    NOT NULL,
    created_at  REAL    NOT NULL,
    fired       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(due_at, fired);
"""

#: "in 5 minutes", "in an hour", "in 90 seconds".
_RELATIVE = re.compile(
    r"^in\s+(?:(?P<count>\d+(?:\.\d+)?)|(?P<article>an?))\s*"
    r"(?P<unit>second|sec|minute|min|hour|hr|day)s?$",
    re.IGNORECASE,
)
#: "at 17:30", "at 5:30pm", "at 5pm".
_ABSOLUTE = re.compile(
    r"^at\s+(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<meridiem>am|pm)?$",
    re.IGNORECASE,
)

_UNIT_SECONDS = {
    "second": 1, "sec": 1,
    "minute": 60, "min": 60,
    "hour": 3600, "hr": 3600,
    "day": 86400,
}


class Reminder(ToolOutput):
    """One reminder."""

    id: int = Field(description="Identifier, used to delete it.")
    text: str = Field(description="What to be reminded about.")
    due_at: str = Field(description="When it fires, ISO 8601.")
    due_in_seconds: float = Field(description="Seconds until it fires. Negative when overdue.")
    fired: bool = Field(default=False, description="Whether it has already fired.")


def parse_when(when: str, now: datetime | None = None) -> datetime:
    """Turn a spoken time expression into an absolute time.

    Handles "in 5 minutes", "in an hour", and "at 5:30pm". An absolute time that
    has already passed today is taken to mean tomorrow, which is what a person
    means by "remind me at 8".

    A clock time is a *local* clock time. The reference used to be
    ``datetime.now(tz=UTC)``, so ``replace(hour=17)`` produced 17:00 UTC and
    "remind me at 5pm" fired at 13:00 for a user in New York and 22:30 for one
    in India. Nobody says "at five" and means somewhere else's five.

    Args:
        when: The spoken expression.
        now: Reference time, injected for tests. Naive values are read as local.

    Returns:
        The absolute due time, timezone aware.

    Raises:
        ValueError: The expression could not be understood.
    """
    reference = now if now is not None else datetime.now().astimezone()
    if reference.tzinfo is None:
        reference = reference.astimezone()
    text = when.strip().lower()

    match = _RELATIVE.match(text)
    if match:
        count = float(match.group("count")) if match.group("count") else 1.0
        unit = match.group("unit").lower()
        return reference + timedelta(seconds=count * _UNIT_SECONDS[unit])

    match = _ABSOLUTE.match(text)
    if match:
        hour = int(match.group("hour"))
        minute = int(match.group("minute") or 0)
        meridiem = (match.group("meridiem") or "").lower()
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            msg = f"{when!r} is not a valid time"
            raise ValueError(msg)
        # In the reference's own zone, which is the user's, so the hour they
        # said is the hour they get.
        candidate = reference.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= reference:
            candidate += timedelta(days=1)
        return candidate

    msg = f"could not understand the time {when!r}"
    raise ValueError(msg)


class ReminderStore:
    """SQLite persistence plus the scheduling thread."""

    def __init__(
        self,
        db_path: Path,
        *,
        notifier: Callable[[str], None] | None = None,
    ) -> None:
        self._path = db_path
        self._notifier = notifier or _toast
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- persistence -------------------------------------------------------

    def add(self, text: str, due_at: datetime) -> Reminder:
        """Store a reminder and wake the scheduler."""
        now = time.time()
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO reminders (text, due_at, created_at, fired) VALUES (?, ?, ?, 0)",
                (text[:_MAX_TEXT_CHARS], due_at.timestamp(), now),
            )
            self._conn.commit()
            reminder_id = int(cursor.lastrowid or 0)
        self._wake.set()
        return Reminder(
            id=reminder_id,
            text=text[:_MAX_TEXT_CHARS],
            due_at=due_at.isoformat(),
            due_in_seconds=round(due_at.timestamp() - now, 1),
        )

    def delete(self, reminder_id: int) -> bool:
        """Remove a reminder. Returns whether it existed."""
        with self._lock:
            cursor = self._conn.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
            self._conn.commit()
            deleted = cursor.rowcount > 0
        self._wake.set()
        return deleted

    def pending(self, limit: int = _MAX_RESULTS) -> list[Reminder]:
        """Reminders that have not fired yet, soonest first."""
        now = time.time()
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM reminders WHERE fired = 0 ORDER BY due_at ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            Reminder(
                id=int(row["id"]),
                text=str(row["text"]),
                due_at=datetime.fromtimestamp(row["due_at"], tz=UTC).isoformat(),
                due_in_seconds=round(float(row["due_at"]) - now, 1),
                fired=bool(row["fired"]),
            )
            for row in rows
        ]

    def pending_count(self) -> int:
        """How many reminders are outstanding."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM reminders WHERE fired = 0"
            ).fetchone()
        return int(row["n"])

    def due(self, now: float | None = None) -> list[sqlite3.Row]:
        """Reminders whose time has come."""
        stamp = now if now is not None else time.time()
        with self._lock:
            return list(
                self._conn.execute(
                    "SELECT * FROM reminders WHERE fired = 0 AND due_at <= ? ORDER BY due_at",
                    (stamp,),
                ).fetchall()
            )

    def mark_fired(self, reminder_id: int) -> None:
        """Record that a reminder has been delivered."""
        with self._lock:
            self._conn.execute("UPDATE reminders SET fired = 1 WHERE id = ?", (reminder_id,))
            self._conn.commit()

    def close(self) -> None:
        """Stop the scheduler and close the database."""
        self.stop()
        with self._lock:
            self._conn.close()

    # -- scheduling --------------------------------------------------------

    def fire_due(self, now: float | None = None) -> int:
        """Deliver every due reminder. Returns how many fired."""
        fired = 0
        for row in self.due(now):
            try:
                self._notifier(str(row["text"]))
            except Exception:  # noqa: BLE001 - a failed toast must not block the queue
                _log.exception("could not deliver a reminder notification")
            self.mark_fired(int(row["id"]))
            fired += 1
        return fired

    def start(self) -> None:
        """Start the scheduling thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="jarvis-reminders", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the scheduling thread."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def _run(self) -> None:
        """Sleep until the next reminder is due rather than polling."""
        while not self._stop.is_set():
            self.fire_due()
            with self._lock:
                row = self._conn.execute(
                    "SELECT MIN(due_at) AS next FROM reminders WHERE fired = 0"
                ).fetchone()
            next_due = row["next"] if row else None
            # Cap the sleep so a clock change or a missed wake cannot strand the
            # thread indefinitely.
            delay = 60.0 if next_due is None else max(0.05, min(60.0, next_due - time.time()))
            self._wake.wait(delay)
            self._wake.clear()


def _toast(text: str) -> None:
    """Show a Windows toast, falling back to a log line elsewhere."""
    if not is_windows():
        _log.info("reminder due: %s", text)
        return
    try:
        from windows_toasts import Toast, WindowsToaster

        toaster = WindowsToaster("Jarvis")
        notification = Toast()
        notification.text_fields = ["Jarvis", text]
        toaster.show_toast(notification)
    except Exception:  # noqa: BLE001 - notification is best effort
        _log.warning("could not show a toast notification", exc_info=True)
        _log.info("reminder due: %s", text)


_store: ReminderStore | None = None


def get_store(config: JarvisConfig | None = None) -> ReminderStore:
    """Process-wide reminder store, created on first use."""
    global _store  # noqa: PLW0603 - one store owns the database and the thread
    if _store is None:
        cfg = config or get_config()
        _store = ReminderStore(cfg.resolve_path(cfg.tools.reminders_db_path))
        _store.start()
    return _store


def set_store(store: ReminderStore | None) -> None:
    """Replace the shared store. Used by the orchestrator and by tests."""
    global _store  # noqa: PLW0603
    _store = store


class CreateInput(ToolInput):
    """Arguments for creating a reminder."""

    text: str = Field(
        min_length=1,
        max_length=_MAX_TEXT_CHARS,
        description="What to be reminded about, for example take the pizza out.",
    )
    when: str = Field(
        min_length=1,
        max_length=60,
        description=(
            "When to fire it. Either a relative time such as in 5 minutes or in an hour, "
            "or an absolute time such as at 17:30 or at 5:30pm. An absolute time that has "
            "already passed today means tomorrow."
        ),
    )


class CreateOutput(ToolOutput):
    """Outcome of creating a reminder."""

    created: bool = Field(description="Whether the reminder was stored.")
    reminder: Reminder | None = Field(default=None, description="The stored reminder.")
    detail: str = Field(description="A short spoken confirmation.")


class DeleteInput(ToolInput):
    """Arguments for deleting a reminder."""

    reminder_id: int = Field(ge=1, description="Identifier from the reminder list.")


class DeleteOutput(ToolOutput):
    """Outcome of deleting a reminder."""

    deleted: bool = Field(description="Whether a reminder was removed.")
    detail: str = Field(description="A short spoken confirmation.")


class ListInput(ToolInput):
    """Arguments for listing reminders."""


class ListOutput(ToolOutput):
    """Outstanding reminders."""

    count: int = Field(description="How many reminders are outstanding.")
    reminders: list[Reminder] = Field(
        default_factory=list, description="Up to 5 reminders, soonest first."
    )


@tool(
    name="reminders.create",
    description=(
        "Set a reminder or a timer. Use this for requests like remind me to take the pizza "
        "out in 12 minutes, or remind me to call mum at 6pm. The time can be relative, such "
        "as in 5 minutes, or absolute, such as at 17:30. This changes system state, so it "
        "requires the user to confirm out loud first."
    ),
    category=ToolCategory.REMINDERS,
    read_only=False,
)
def create_reminder(params: CreateInput) -> CreateOutput:
    """Create a reminder.

    Args:
        params: The text and the time expression.

    Returns:
        The stored reminder, or an explanation of why the time was not understood.
    """
    try:
        due = parse_when(params.when)
    except ValueError:
        return CreateOutput(
            created=False,
            detail=f"I did not understand the time {params.when}.",
        )

    try:
        reminder = get_store().add(params.text, due)
    except sqlite3.Error as exc:
        _log.exception("reminders.create failed")
        raise ToolExecutionError(
            "reminders.create",
            f"could not store the reminder: {exc}",
            speakable="I could not save that reminder.",
        ) from exc

    minutes = reminder.due_in_seconds / 60
    when_phrase = (
        f"in {round(reminder.due_in_seconds)} seconds"
        if minutes < 1
        else f"in {round(minutes)} minutes"
        if minutes < 90
        else f"at {due.strftime('%H:%M')}"
    )
    return CreateOutput(
        created=True,
        reminder=reminder,
        detail=f"I will remind you {when_phrase}.",
    )


@tool(
    name="reminders.delete",
    description=(
        "Cancel a reminder by its identifier. List the reminders first to get the "
        "identifier. This changes system state, so it requires the user to confirm out loud "
        "first."
    ),
    category=ToolCategory.REMINDERS,
    read_only=False,
)
def delete_reminder(params: DeleteInput) -> DeleteOutput:
    """Delete a reminder.

    Args:
        params: The reminder identifier.

    Returns:
        Whether it was removed.
    """
    try:
        removed = get_store().delete(params.reminder_id)
    except sqlite3.Error as exc:
        _log.exception("reminders.delete failed")
        raise ToolExecutionError(
            "reminders.delete",
            f"could not delete the reminder: {exc}",
            speakable="I could not cancel that reminder.",
        ) from exc
    return DeleteOutput(
        deleted=removed,
        detail="Cancelled." if removed else "I could not find that reminder.",
    )


@tool(
    name="reminders.list",
    description=(
        "List outstanding reminders and timers with the time remaining. Use this for "
        "questions like what reminders do I have, or how long is left on my timer. Returns "
        "at most 5, soonest first. Read-only."
    ),
    category=ToolCategory.REMINDERS,
    read_only=True,
)
def list_reminders(params: ListInput) -> ListOutput:
    """List outstanding reminders.

    Args:
        params: No arguments.

    Returns:
        Up to five reminders, soonest first.
    """
    try:
        store = get_store()
        return ListOutput(count=store.pending_count(), reminders=store.pending())
    except sqlite3.Error as exc:
        _log.exception("reminders.list failed")
        raise ToolExecutionError(
            "reminders.list",
            f"could not read the reminders: {exc}",
            speakable="I could not read your reminders.",
        ) from exc
