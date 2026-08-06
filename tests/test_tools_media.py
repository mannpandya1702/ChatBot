"""T-4.3 verification: media.control."""

from __future__ import annotations

import pytest

from jarvis.config import JarvisConfig
from jarvis.tools import media as media_module
from jarvis.tools.media import MediaInput, media_control
from jarvis.tools.registry import registry


class TestKeyCodes:
    def test_every_action_has_a_key_code(self) -> None:
        actions = [
            "play_pause", "next", "previous", "stop", "volume_up", "volume_down", "mute"
        ]
        for action in actions:
            assert action in media_module._VK_CODES

    def test_codes_are_the_documented_windows_values(self) -> None:
        """Wrong codes would silently do nothing, or the wrong thing."""
        assert media_module._VK_CODES["play_pause"] == 0xB3
        assert media_module._VK_CODES["next"] == 0xB0
        assert media_module._VK_CODES["previous"] == 0xB1
        assert media_module._VK_CODES["stop"] == 0xB2
        assert media_module._VK_CODES["volume_up"] == 0xAF
        assert media_module._VK_CODES["volume_down"] == 0xAE
        assert media_module._VK_CODES["mute"] == 0xAD


class TestOffWindows:
    def test_reports_platform_rather_than_failing(self) -> None:
        result = media_control(MediaInput(action="play_pause"))
        assert result.performed is False
        assert "Windows" in result.detail


class TestOnWindows:
    @pytest.fixture
    def sent(self, monkeypatch: pytest.MonkeyPatch) -> list[int]:
        keys: list[int] = []
        monkeypatch.setattr(media_module, "is_windows", lambda: True)
        monkeypatch.setattr(media_module, "_send_key", keys.append)
        return keys

    def test_sends_the_right_key(self, sent: list[int]) -> None:
        result = media_control(MediaInput(action="next"))
        assert result.performed is True
        assert sent == [0xB0]

    def test_repeat_sends_multiple(self, sent: list[int]) -> None:
        media_control(MediaInput(action="next", repeat=3))
        assert sent == [0xB0, 0xB0, 0xB0]

    def test_volume_taps_several_times_per_step(self, sent: list[int]) -> None:
        """One volume tap is about two percent, which nobody would notice."""
        media_control(MediaInput(action="volume_up"))
        assert len(sent) == media_module._VOLUME_TAPS
        assert set(sent) == {0xAF}

    def test_volume_repeat_multiplies(self, sent: list[int]) -> None:
        media_control(MediaInput(action="volume_down", repeat=2))
        assert len(sent) == media_module._VOLUME_TAPS * 2

    def test_detail_is_speakable(self, sent: list[int]) -> None:
        assert media_control(MediaInput(action="play_pause")).detail == "Toggled playback."

    def test_detail_mentions_repeats(self, sent: list[int]) -> None:
        assert "3 times" in media_control(MediaInput(action="next", repeat=3)).detail

    def test_key_failure_becomes_a_tool_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from jarvis.util.errors import ToolExecutionError

        monkeypatch.setattr(media_module, "is_windows", lambda: True)

        def boom(_code: int) -> None:
            raise OSError("no input desktop")

        monkeypatch.setattr(media_module, "_send_key", boom)
        with pytest.raises(ToolExecutionError) as excinfo:
            media_control(MediaInput(action="next"))
        assert excinfo.value.speakable == "I could not reach the media controls."


class TestValidation:
    def test_unknown_action_is_rejected_by_the_schema(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MediaInput(action="self_destruct")  # type: ignore[arg-type]

    def test_repeat_is_bounded(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MediaInput(action="next", repeat=999)


class TestRegistration:
    def test_is_mutating_and_gated(self) -> None:
        spec = registry.get("media.control")
        assert spec is not None
        assert spec.read_only is False
        assert spec.requires_confirmation is True

    def test_unconfirmed_call_is_refused(self) -> None:
        assert registry.dispatch("media.control", {"action": "next"}).ok is False

    def test_allowlist_matches_the_contract(self, cfg: JarvisConfig) -> None:
        assert "media.control" in cfg.gate.mutating_allowlist
