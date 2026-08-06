"""The clock tool, which the Phase 1 gate depends on.

The gate is "say 'Hey Jarvis, what time is it' and hear a spoken answer", and
until this tool existed nothing could answer it. A language model has no clock,
so it either declined or invented one, which §5 forbids outright.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from jarvis.tools.registry import registry
from jarvis.tools.sys_time import TimeInput, current_time, speak_date, speak_time


class TestSpokenTime:
    """Output is read aloud, so it has to be phrased the way a person speaks."""

    @pytest.mark.parametrize(
        ("hour", "minute", "expected"),
        [
            (15, 0, "three o'clock in the afternoon"),
            (15, 15, "quarter past three in the afternoon"),
            (15, 30, "half past three in the afternoon"),
            (15, 45, "quarter to four in the afternoon"),
            (15, 20, "20 minutes past three in the afternoon"),
            (13, 40, "20 minutes to two in the afternoon"),
            (9, 5, "5 minutes past nine in the morning"),
            (19, 0, "seven o'clock in the evening"),
            (22, 0, "ten o'clock at night"),
        ],
    )
    def test_ordinary_times(self, hour: int, minute: int, expected: str) -> None:
        assert speak_time(datetime(2026, 8, 6, hour, minute)) == expected

    @pytest.mark.parametrize(
        ("hour", "minute", "expected"),
        [
            (0, 0, "midnight"),
            (12, 0, "midday"),
            (23, 58, "2 minutes to midnight"),
            (11, 50, "10 minutes to midday"),
            (0, 15, "quarter past midnight"),
            (12, 30, "half past midday"),
        ],
    )
    def test_midnight_and_midday_name_themselves(
        self, hour: int, minute: int, expected: str
    ) -> None:
        """"twelve o'clock in the morning" is not how anyone says midnight."""
        assert speak_time(datetime(2026, 8, 6, hour, minute)) == expected

    def test_no_time_is_phrased_as_a_bare_number(self) -> None:
        """A 24 hour reading would be spoken as digits, which sounds robotic."""
        for hour in range(24):
            for minute in (0, 1, 15, 29, 30, 31, 45, 59):
                spoken = speak_time(datetime(2026, 8, 6, hour, minute))
                assert not spoken[0].isdigit() or "minutes" in spoken
                assert ":" not in spoken


class TestSpokenDate:
    @pytest.mark.parametrize(
        ("day", "expected"),
        [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"), (21, "21st"), (31, "31st")],
    )
    def test_ordinal_suffixes(self, day: int, expected: str) -> None:
        assert expected in speak_date(datetime(2026, 8, day))

    def test_the_weekday_and_month_are_named(self) -> None:
        assert speak_date(datetime(2026, 8, 6)) == "Thursday the 6th of August"


class TestTheTool:
    def test_it_reports_a_time_and_a_date(self) -> None:
        result = current_time(TimeInput())

        assert result.time_spoken
        assert result.date_spoken
        assert result.day_of_week
        assert ":" in result.time_24h
        assert result.iso

    def test_the_date_can_be_left_out(self) -> None:
        result = current_time(TimeInput(include_date=False))

        assert result.time_spoken
        assert result.date_spoken is None
        assert result.day_of_week is None

    def test_the_time_matches_the_clock(self) -> None:
        """It reports local time, which is the clock the user is looking at."""
        before = datetime.now().astimezone()
        result = current_time(TimeInput())
        after = datetime.now().astimezone()

        reported = datetime.fromisoformat(result.iso)
        assert before - timedelta(seconds=5) <= reported <= after + timedelta(seconds=5)

    def test_the_offset_matches_the_timestamp(self) -> None:
        result = current_time(TimeInput())
        reported = datetime.fromisoformat(result.iso)
        offset = reported.utcoffset() or timedelta(0)
        assert result.utc_offset_hours == pytest.approx(offset.total_seconds() / 3600.0, abs=0.01)

    def test_utc_is_handled(self) -> None:
        """A machine on UTC has a zero offset, not a missing one."""
        moment = datetime(2026, 8, 6, 15, 20, tzinfo=UTC)
        assert speak_time(moment) == "20 minutes past three in the afternoon"


class TestRegistration:
    def test_it_is_registered_and_read_only(self) -> None:
        import jarvis.main as main_module
        from jarvis.config import load_defaults

        main_module._register_tools(load_defaults())
        spec = registry.get("sys.time")

        assert spec is not None, "the Phase 1 gate question has no tool to answer it"
        assert spec.read_only is True

    def test_the_description_tells_the_model_not_to_guess(self) -> None:
        """§5: never invent a metric that did not come from a tool."""
        import jarvis.main as main_module
        from jarvis.config import load_defaults

        main_module._register_tools(load_defaults())
        spec = registry.get("sys.time")
        assert spec is not None
        description = spec.description.lower()
        assert "no clock" in description or "rather than answering from memory" in description
        assert "what time is it" in description

    def test_it_is_offered_to_the_model(self) -> None:
        import jarvis.main as main_module
        from jarvis.config import load_defaults

        config = load_defaults()
        main_module._register_tools(config)
        assert "sys.time" in [spec.name for spec in registry.select(config=config)]
