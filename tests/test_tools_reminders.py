"""T-4.6 verification: reminders and timers."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from jarvis.config import JarvisConfig
from jarvis.tools.registry import registry
from jarvis.tools.reminders import (
    CreateInput,
    DeleteInput,
    ListInput,
    ReminderStore,
    create_reminder,
    delete_reminder,
    list_reminders,
    parse_when,
    set_store,
)

NOW = datetime(2026, 8, 6, 14, 0, 0, tzinfo=UTC)


class TestParseWhen:
    @pytest.mark.parametrize(
        ("expression", "seconds"),
        [
            ("in 30 seconds", 30),
            ("in 1 second", 1),
            ("in 5 minutes", 300),
            ("in 90 minutes", 5400),
            ("in 2 hours", 7200),
            ("in 1 day", 86400),
            ("in an hour", 3600),
            ("in a minute", 60),
            ("in 1.5 hours", 5400),
        ],
    )
    def test_relative(self, expression: str, seconds: float) -> None:
        assert parse_when(expression, NOW) == NOW + timedelta(seconds=seconds)

    def test_relative_is_case_insensitive(self) -> None:
        assert parse_when("IN 5 MINUTES", NOW) == NOW + timedelta(minutes=5)

    @pytest.mark.parametrize(
        ("expression", "hour", "minute"),
        [
            ("at 17:30", 17, 30),
            ("at 5:30pm", 17, 30),
            ("at 5pm", 17, 0),
            ("at 9am", 9, 0),
            ("at 12am", 0, 0),
            ("at 12pm", 12, 0),
        ],
    )
    def test_absolute(self, expression: str, hour: int, minute: int) -> None:
        result = parse_when(expression, NOW)
        assert (result.hour, result.minute) == (hour, minute)

    def test_absolute_time_already_past_means_tomorrow(self) -> None:
        """"remind me at 8" said at 2pm means 8am tomorrow, not 8am today."""
        result = parse_when("at 8am", NOW)
        assert result.date() == (NOW + timedelta(days=1)).date()

    def test_absolute_time_still_ahead_means_today(self) -> None:
        assert parse_when("at 6pm", NOW).date() == NOW.date()

    @pytest.mark.parametrize(
        "expression", ["tomorrow", "next tuesday", "", "in", "at 99:99", "soon", "in 5 fortnights"]
    )
    def test_unparseable_raises(self, expression: str) -> None:
        with pytest.raises(ValueError, match=r"could not understand|not a valid"):
            parse_when(expression, NOW)


@pytest.fixture
def store(tmp_path: Path) -> Any:
    """A store with a captured notifier and no background thread."""
    delivered: list[str] = []
    instance = ReminderStore(tmp_path / "reminders.db", notifier=delivered.append)
    instance.delivered = delivered  # type: ignore[attr-defined]
    yield instance
    instance.close()


class TestStore:
    def test_add_and_list(self, store: Any) -> None:
        store.add("take the pizza out", datetime.now(tz=UTC) + timedelta(minutes=12))
        pending = store.pending()
        assert len(pending) == 1
        assert pending[0].text == "take the pizza out"
        assert pending[0].due_in_seconds > 0

    def test_delete(self, store: Any) -> None:
        reminder = store.add("x", datetime.now(tz=UTC) + timedelta(minutes=5))
        assert store.delete(reminder.id) is True
        assert store.pending() == []

    def test_delete_missing_returns_false(self, store: Any) -> None:
        assert store.delete(9999) is False

    def test_pending_sorted_soonest_first(self, store: Any) -> None:
        now = datetime.now(tz=UTC)
        store.add("later", now + timedelta(hours=2))
        store.add("sooner", now + timedelta(minutes=1))
        assert [r.text for r in store.pending()] == ["sooner", "later"]

    def test_pending_capped_at_five(self, store: Any) -> None:
        now = datetime.now(tz=UTC)
        for index in range(10):
            store.add(f"r{index}", now + timedelta(minutes=index + 1))
        assert len(store.pending()) == 5
        assert store.pending_count() == 10

    def test_text_is_truncated(self, store: Any) -> None:
        reminder = store.add("x" * 500, datetime.now(tz=UTC) + timedelta(minutes=1))
        assert len(reminder.text) <= 200

    def test_persists_across_instances(self, tmp_path: Path) -> None:
        """Reminders must survive a restart."""
        path = tmp_path / "reminders.db"
        first = ReminderStore(path)
        first.add("survive me", datetime.now(tz=UTC) + timedelta(hours=1))
        first.close()

        second = ReminderStore(path)
        try:
            assert [r.text for r in second.pending()] == ["survive me"]
        finally:
            second.close()

    def test_due_reminders_fire(self, store: Any) -> None:
        store.add("fire now", datetime.now(tz=UTC) - timedelta(seconds=1))
        assert store.fire_due() == 1
        assert store.delivered == ["fire now"]
        assert store.pending() == []

    def test_future_reminders_do_not_fire(self, store: Any) -> None:
        store.add("later", datetime.now(tz=UTC) + timedelta(hours=1))
        assert store.fire_due() == 0
        assert store.delivered == []

    def test_a_reminder_fires_only_once(self, store: Any) -> None:
        store.add("once", datetime.now(tz=UTC) - timedelta(seconds=1))
        store.fire_due()
        store.fire_due()
        assert store.delivered == ["once"]

    def test_a_failing_notifier_does_not_block_the_queue(self, tmp_path: Path) -> None:
        """One broken toast must not strand every later reminder."""
        attempts: list[str] = []

        def flaky(text: str) -> None:
            attempts.append(text)
            if text == "bad":
                raise RuntimeError("toast failed")

        instance = ReminderStore(tmp_path / "r.db", notifier=flaky)
        try:
            past = datetime.now(tz=UTC) - timedelta(seconds=1)
            instance.add("bad", past)
            instance.add("good", past)
            assert instance.fire_due() == 2
            assert attempts == ["bad", "good"]
            assert instance.pending() == []
        finally:
            instance.close()

    def test_scheduler_thread_fires(self, tmp_path: Path) -> None:
        delivered: list[str] = []
        instance = ReminderStore(tmp_path / "r.db", notifier=delivered.append)
        try:
            instance.start()
            instance.add("soon", datetime.now(tz=UTC) + timedelta(milliseconds=100))
            deadline = time.monotonic() + 5
            while not delivered and time.monotonic() < deadline:
                time.sleep(0.05)
            assert delivered == ["soon"]
        finally:
            instance.close()

    def test_start_is_idempotent(self, store: Any) -> None:
        store.start()
        store.start()
        store.stop()


@pytest.fixture
def wired(store: Any) -> Any:
    """Point the tool functions at the test store."""
    set_store(store)
    yield store
    set_store(None)


class TestTools:
    def test_create(self, wired: Any) -> None:
        result = create_reminder(CreateInput(text="check the oven", when="in 10 minutes"))
        assert result.created is True
        assert result.reminder is not None
        assert "10 minutes" in result.detail

    def test_create_with_seconds_phrasing(self, wired: Any) -> None:
        result = create_reminder(CreateInput(text="x", when="in 30 seconds"))
        assert "seconds" in result.detail

    def test_create_with_unparseable_time_is_not_an_exception(self, wired: Any) -> None:
        result = create_reminder(CreateInput(text="x", when="next thursday"))
        assert result.created is False
        assert "did not understand" in result.detail

    def test_list(self, wired: Any) -> None:
        create_reminder(CreateInput(text="one", when="in 5 minutes"))
        create_reminder(CreateInput(text="two", when="in 10 minutes"))
        result = list_reminders(ListInput())
        assert result.count == 2
        assert [r.text for r in result.reminders] == ["one", "two"]

    def test_delete(self, wired: Any) -> None:
        created = create_reminder(CreateInput(text="cancel me", when="in 5 minutes"))
        assert created.reminder is not None
        result = delete_reminder(DeleteInput(reminder_id=created.reminder.id))
        assert result.deleted is True
        assert list_reminders(ListInput()).count == 0

    def test_delete_missing(self, wired: Any) -> None:
        result = delete_reminder(DeleteInput(reminder_id=4242))
        assert result.deleted is False
        assert "could not find" in result.detail


class TestGating:
    def test_create_and_delete_are_mutating(self) -> None:
        """§6 lists reminders.create and reminders.delete on the allowlist."""
        for name in ("reminders.create", "reminders.delete"):
            spec = registry.get(name)
            assert spec is not None
            assert spec.read_only is False
            assert spec.requires_confirmation is True

    def test_list_is_read_only(self) -> None:
        spec = registry.get("reminders.list")
        assert spec is not None
        assert spec.read_only is True

    def test_unconfirmed_create_is_refused(self, wired: Any) -> None:
        result = registry.dispatch("reminders.create", {"text": "x", "when": "in 5 minutes"})
        assert result.ok is False
        assert wired.pending() == []

    def test_confirmed_create_runs(self, wired: Any) -> None:
        result = registry.dispatch(
            "reminders.create", {"text": "x", "when": "in 5 minutes"}, confirmed=True
        )
        assert result.ok is True
        assert len(wired.pending()) == 1

    def test_allowlist_matches_the_contract(self, cfg: JarvisConfig) -> None:
        assert "reminders.create" in cfg.gate.mutating_allowlist
        assert "reminders.delete" in cfg.gate.mutating_allowlist
        assert "reminders.list" not in cfg.gate.mutating_allowlist


class TestClockTimesAreLocal:
    """"At five" means the user's five.

    parse_when used datetime.now(tz=UTC) as its reference, so replace(hour=17)
    produced 17:00 UTC. On the target host, a Windows machine in a local zone,
    "remind me at 5pm" fired at 13:00 in New York and 22:30 in India, and the
    spoken confirmation repeated the hour the user asked for, so nothing about
    it looked wrong until the reminder did not arrive.
    """

    @pytest.mark.parametrize(
        "zone", ["America/New_York", "Asia/Kolkata", "Europe/London", "Pacific/Auckland"]
    )
    @pytest.mark.parametrize(
        ("spoken", "hour", "minute"),
        [("at 5pm", 17, 0), ("at 8am", 8, 0), ("at 17:30", 17, 30), ("at 12am", 0, 0)],
    )
    def test_the_hour_asked_for_is_the_hour_stored(
        self,
        monkeypatch: pytest.MonkeyPatch,
        zone: str,
        spoken: str,
        hour: int,
        minute: int,
    ) -> None:
        import time as _time

        monkeypatch.setenv("TZ", zone)
        if hasattr(_time, "tzset"):
            _time.tzset()
        try:
            due = parse_when(spoken).astimezone()
            assert (due.hour, due.minute) == (hour, minute)
        finally:
            monkeypatch.delenv("TZ", raising=False)
            if hasattr(_time, "tzset"):
                _time.tzset()

    def test_a_time_already_past_moves_to_tomorrow_in_local_terms(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import time as _time

        monkeypatch.setenv("TZ", "America/New_York")
        if hasattr(_time, "tzset"):
            _time.tzset()
        try:
            now = datetime.now().astimezone()
            asked = (now.hour - 1) % 24
            due = parse_when(f"at {asked}:00").astimezone()
            assert due.hour == asked
            assert due > now
        finally:
            monkeypatch.delenv("TZ", raising=False)
            if hasattr(_time, "tzset"):
                _time.tzset()

    def test_relative_times_are_unaffected(self) -> None:
        before = datetime.now().astimezone()
        due = parse_when("in 5 minutes")
        assert 4.9 <= (due - before).total_seconds() / 60 <= 5.1

    def test_a_naive_reference_is_read_as_local(self) -> None:
        """Callers passing a bare datetime meant local, not UTC."""
        naive = datetime(2026, 8, 7, 9, 0)
        assert parse_when("at 5pm", naive).astimezone().hour == 17
