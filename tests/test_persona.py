"""T-1.8 verification: the system prompt."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.brain.persona import (
    ANTI_HALLUCINATION,
    build_confirmation_prompt,
    build_error_response,
    build_summary_prompt,
    build_system_prompt,
)
from jarvis.config import JarvisConfig, load_config


class TestIdentity:
    def test_names_the_assistant(self, cfg: JarvisConfig) -> None:
        assert "Jarvis" in build_system_prompt(cfg)

    def test_a_renamed_assistant_is_used(self, tmp_path: Path) -> None:
        config = load_config(tmp_path / "absent.yaml", persona={"assistant_name": "Friday"})
        prompt = build_system_prompt(config)
        assert "Friday" in prompt
        assert "You are Friday" in prompt

    def test_includes_the_address_form(self, cfg: JarvisConfig) -> None:
        assert "sir" in build_system_prompt(cfg)

    def test_empty_address_form_forbids_inventing_one(self, tmp_path: Path) -> None:
        """Without this the model invents "sir" or a name from nowhere."""
        config = load_config(tmp_path / "absent.yaml", persona={"user_address_form": ""})
        prompt = build_system_prompt(config)
        assert "sir" not in prompt.lower()
        assert "Do not address the user by any name" in prompt

    def test_includes_the_response_style(self, cfg: JarvisConfig) -> None:
        assert cfg.persona.response_style in build_system_prompt(cfg)


class TestRules:
    def test_demands_brevity(self, cfg: JarvisConfig) -> None:
        prompt = build_system_prompt(cfg)
        assert "one or two short spoken sentences" in prompt

    def test_forbids_inventing_measurements(self, cfg: JarvisConfig) -> None:
        """The single most important rule in the prompt."""
        prompt = build_system_prompt(cfg)
        assert "Never state a measurement you did not receive from a tool" in prompt
        assert ANTI_HALLUCINATION in prompt

    def test_requires_a_tool_for_machine_state(self, cfg: JarvisConfig) -> None:
        prompt = build_system_prompt(cfg)
        assert "must be answered from a tool call" in prompt

    def test_forbids_reading_json_aloud(self, cfg: JarvisConfig) -> None:
        assert "Never read out JSON" in build_system_prompt(cfg)

    def test_forbids_markdown(self, cfg: JarvisConfig) -> None:
        prompt = build_system_prompt(cfg)
        assert "No markdown" in prompt
        assert "bullet points" in prompt

    def test_asks_for_spoken_numbers_with_units(self, cfg: JarvisConfig) -> None:
        prompt = build_system_prompt(cfg)
        assert "forty three percent" in prompt
        assert "include the unit" in prompt.lower()

    def test_covers_the_confirmation_rule(self, cfg: JarvisConfig) -> None:
        assert "confirm out loud" in build_system_prompt(cfg)

    def test_forbids_em_dashes_in_speech(self, cfg: JarvisConfig) -> None:
        assert "No em dashes" in build_system_prompt(cfg)

    def test_the_prompt_itself_has_no_em_dash(self, cfg: JarvisConfig) -> None:
        prompt = build_system_prompt(cfg)
        assert "\u2014" not in prompt
        assert "\u2013" not in prompt


class TestTools:
    def test_tool_names_are_listed(self, cfg: JarvisConfig) -> None:
        prompt = build_system_prompt(cfg, tools=["sys.cpu", "sys.memory"])
        assert "sys.cpu" in prompt
        assert "sys.memory" in prompt

    def test_no_tool_section_when_none_offered(self, cfg: JarvisConfig) -> None:
        assert "available to you right now" not in build_system_prompt(cfg)

    def test_tool_names_are_sorted(self, cfg: JarvisConfig) -> None:
        """Stable ordering keeps the prompt cacheable across turns."""
        first = build_system_prompt(cfg, tools=["b.two", "a.one"])
        second = build_system_prompt(cfg, tools=["a.one", "b.two"])
        assert first == second


class TestExtras:
    def test_extra_is_appended(self, cfg: JarvisConfig) -> None:
        assert "SPECIAL RULE" in build_system_prompt(cfg, extra="SPECIAL RULE")

    def test_config_extra_instructions_are_appended(self, tmp_path: Path) -> None:
        config = load_config(
            tmp_path / "absent.yaml", persona={"extra_instructions": "Prefer metric units."}
        )
        assert "Prefer metric units." in build_system_prompt(config)

    def test_both_extras_appear(self, tmp_path: Path) -> None:
        config = load_config(
            tmp_path / "absent.yaml", persona={"extra_instructions": "FROM CONFIG"}
        )
        prompt = build_system_prompt(config, extra="FROM CALL")
        assert "FROM CONFIG" in prompt
        assert "FROM CALL" in prompt


class TestStability:
    def test_deterministic(self, cfg: JarvisConfig) -> None:
        assert build_system_prompt(cfg) == build_system_prompt(cfg)

    def test_changes_with_the_persona(self, cfg: JarvisConfig, tmp_path: Path) -> None:
        other = load_config(tmp_path / "absent.yaml", persona={"assistant_name": "Friday"})
        assert build_system_prompt(cfg) != build_system_prompt(other)

    def test_stays_a_reasonable_size(self, cfg: JarvisConfig) -> None:
        """A bloated system prompt eats the context window on a small model."""
        prompt = build_system_prompt(cfg, tools=[f"tool.{i}" for i in range(20)])
        assert len(prompt) < 4_000, f"the prompt is {len(prompt)} characters"

    def test_no_blank_run_between_blocks(self, cfg: JarvisConfig) -> None:
        assert "\n\n\n" not in build_system_prompt(cfg)


class TestConfirmationPrompt:
    def test_names_the_action_and_asks(self, cfg: JarvisConfig) -> None:
        sentence = build_confirmation_prompt("You want me to launch Spotify", cfg)
        assert "Spotify" in sentence
        assert sentence.endswith("?")
        assert "Shall I go ahead" in sentence

    def test_uses_the_address_form(self, cfg: JarvisConfig) -> None:
        assert "sir" in build_confirmation_prompt("You want me to do a thing", cfg)

    def test_omits_the_address_when_unset(self, tmp_path: Path) -> None:
        config = load_config(tmp_path / "absent.yaml", persona={"user_address_form": ""})
        sentence = build_confirmation_prompt("You want me to do a thing", config)
        assert "sir" not in sentence
        assert sentence.endswith("?")

    def test_a_trailing_stop_is_not_doubled(self, cfg: JarvisConfig) -> None:
        assert ".." not in build_confirmation_prompt("You want me to do a thing.", cfg)


class TestSummaryPrompt:
    def test_asks_for_brevity(self) -> None:
        assert "six short lines" in build_summary_prompt()

    def test_excludes_stale_readings(self) -> None:
        """A summary that carries old sensor values invites stale answers."""
        prompt = build_summary_prompt()
        assert "Do not keep specific" in prompt
        assert "go stale" in prompt

    def test_forbids_markdown(self) -> None:
        assert "no markdown" in build_summary_prompt().lower()


class TestErrorResponse:
    def test_uses_the_speakable(self, cfg: JarvisConfig) -> None:
        assert "I could not read that sensor" in build_error_response(
            "I could not read that sensor.", cfg
        )

    def test_appends_the_address_form(self, cfg: JarvisConfig) -> None:
        assert build_error_response("Something failed.", cfg).endswith("sir.")

    def test_empty_speakable_gets_a_default(self, cfg: JarvisConfig) -> None:
        assert build_error_response("   ", cfg)

    def test_no_address_form_leaves_it_alone(self, tmp_path: Path) -> None:
        config = load_config(tmp_path / "absent.yaml", persona={"user_address_form": ""})
        assert build_error_response("It failed.", config) == "It failed."


class TestNoLeakage:
    @pytest.mark.parametrize(
        "secret", ["C:/Users/mann", "token=abc123", "password", "/home/user"]
    )
    def test_the_prompt_contains_no_machine_specifics(
        self, cfg: JarvisConfig, secret: str
    ) -> None:
        """§0.6: nothing machine-specific belongs in a committed prompt."""
        assert secret not in build_system_prompt(cfg)
