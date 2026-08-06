"""T-4.1 verification: the confirmation gate.

CLAUDE.md §6 is the section most likely to be got wrong, so these tests are
written as adversarial probes rather than a happy-path walkthrough. Every test
here asks the same question: can a mutating call reach the system without an
explicit, current, matching spoken yes?
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis.config import JarvisConfig, load_config
from jarvis.tools.gate import (
    ConfirmationGate,
    ConfirmationOutcome,
    ConfirmationRequired,
    GateDecision,
    classify_response,
    describe_action,
)
from jarvis.tools.registry import ToolInput, ToolOutput, ToolRegistry, ToolResult, tool


class ActionInput(ToolInput):
    name: str = "notepad"


class ActionOutput(ToolOutput):
    launched: str


class _Clock:
    """Controllable clock so timeouts are tested without sleeping."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


@pytest.fixture
def reg() -> ToolRegistry:
    registry = ToolRegistry()
    calls: list[str] = []

    @tool(name="apps.launch", description="Launch an app.", read_only=False, target=registry)
    def launch(params: ActionInput) -> ActionOutput:
        calls.append(params.name)
        return ActionOutput(launched=params.name)

    @tool(name="sys.cpu", description="Read CPU usage.", target=registry)
    def cpu(params: ActionInput) -> ActionOutput:
        calls.append("cpu")
        return ActionOutput(launched="cpu")

    @tool(name="evil.wipe", description="Not allowlisted.", read_only=False, target=registry)
    def wipe(params: ActionInput) -> ActionOutput:
        calls.append("WIPED")
        return ActionOutput(launched="wiped")

    registry.calls = calls  # type: ignore[attr-defined]
    return registry


@pytest.fixture
def gate(cfg: JarvisConfig, reg: ToolRegistry, clock: _Clock, tmp_path: Path) -> ConfirmationGate:
    return ConfirmationGate(
        cfg, registry=reg, audit_path=tmp_path / "gate_audit.jsonl", clock=clock
    )


class TestClassifyResponse:
    @pytest.mark.parametrize(
        "utterance",
        ["yes", "Yes.", "yeah", "yep", "confirm", "do it", "go ahead", "please do", "proceed"],
    )
    def test_affirmatives(self, utterance: str, cfg: JarvisConfig) -> None:
        assert classify_response(utterance, cfg) is ConfirmationOutcome.AFFIRMATIVE

    @pytest.mark.parametrize(
        "utterance",
        ["no", "nope", "cancel", "stop", "abort", "never mind", "don't", "do not"],
    )
    def test_negatives(self, utterance: str, cfg: JarvisConfig) -> None:
        assert classify_response(utterance, cfg) is ConfirmationOutcome.NEGATIVE

    def test_negative_wins_when_both_appear(self, cfg: JarvisConfig) -> None:
        """"no, don't do it" contains the affirmative phrase "do it"."""
        assert classify_response("no, don't do it", cfg) is ConfirmationOutcome.NEGATIVE
        assert classify_response("actually cancel, do it later", cfg) is (
            ConfirmationOutcome.NEGATIVE
        )

    def test_substring_matches_do_not_fire(self, cfg: JarvisConfig) -> None:
        """"no" must not match inside "now" or "know"."""
        assert classify_response("now do it", cfg) is ConfirmationOutcome.AFFIRMATIVE
        assert classify_response("I know, go ahead", cfg) is ConfirmationOutcome.AFFIRMATIVE

    @pytest.mark.parametrize(
        "utterance",
        ["", "   ", "what time is it", "maybe", "hmm", "open the pod bay doors"],
    )
    def test_unclear_is_not_a_yes(self, utterance: str, cfg: JarvisConfig) -> None:
        assert classify_response(utterance, cfg) is not ConfirmationOutcome.AFFIRMATIVE

    @pytest.mark.parametrize(
        "utterance",
        [
            "you can't do it",
            "I can't let you do it",
            "I'd rather you didn't",
            "please don't do it",
            "no, do not do it",
            "not now",
            "maybe later",
            "nah",
            "leave it",
            "skip it",
            "hold on",
            "wait",
            "I won't allow that",
            "that isn't what I meant",
            "forget it",
            "do it later instead",
        ],
    )
    def test_a_negated_reply_is_never_a_yes(
        self, utterance: str, cfg: JarvisConfig
    ) -> None:
        """The dangerous direction.

        An affirmative substring search reads "you can't do it" as a yes,
        because it contains "do it". Negation is decided first and decided
        broadly: a false refusal costs one repeat, a false acceptance costs the
        action itself.
        """
        assert classify_response(utterance, cfg) is not ConfirmationOutcome.AFFIRMATIVE

    @pytest.mark.parametrize(
        "utterance",
        ["yes", "yeah", "yep", "do it", "go ahead", "confirm", "please do", "yes please"],
    )
    def test_a_clean_yes_still_works(self, utterance: str, cfg: JarvisConfig) -> None:
        """The guard must not be so broad that confirming becomes impossible."""
        assert classify_response(utterance, cfg) is ConfirmationOutcome.AFFIRMATIVE

    @pytest.mark.parametrize(
        ("tool", "arguments"),
        [
            ("apps.launch", {"name": "notepad"}),
            ("shell.run", {"command": "Get-Volume"}),
            ("media.control", {"action": "next"}),
            ("reminders.create", {"when": "in 5 minutes", "text": "x"}),
        ],
    )
    def test_the_prompt_is_not_a_valid_answer_to_itself(
        self, tool: str, arguments: dict[str, object], cfg: JarvisConfig
    ) -> None:
        """A microphone that picks up the prompt must not thereby confirm it."""
        prompt = describe_action(tool, arguments, cfg)
        assert classify_response(prompt, cfg) is not ConfirmationOutcome.AFFIRMATIVE

    def test_typographic_apostrophe(self, cfg: JarvisConfig) -> None:
        assert classify_response("don\u2019t", cfg) is ConfirmationOutcome.NEGATIVE


class TestDecision:
    def test_read_only_tool_runs_without_prompting(self, gate: ConfirmationGate) -> None:
        result = gate.check("sys.cpu", {})
        assert result.decision is GateDecision.ALLOWED
        assert result.allowed is True

    def test_mutating_tool_requires_confirmation(self, gate: ConfirmationGate) -> None:
        result = gate.check("apps.launch", {"name": "notepad"})
        assert result.decision is GateDecision.CONFIRMATION_REQUIRED
        assert result.confirmation is not None
        assert "notepad" in result.confirmation.prompt

    def test_tool_off_the_allowlist_is_denied_outright(self, gate: ConfirmationGate) -> None:
        """§6: only the six named tools may mutate, and only in Phase 4."""
        result = gate.check("evil.wipe", {})
        assert result.decision is GateDecision.DENIED
        assert "allowlist" in result.reason

    def test_unknown_tool_is_denied(self, gate: ConfirmationGate) -> None:
        assert gate.check("nope.nothing", {}).decision is GateDecision.DENIED

    def test_shell_denied_when_disabled(
        self, reg: ToolRegistry, clock: _Clock, tmp_path: Path
    ) -> None:
        config = load_config(tmp_path / "absent.yaml")

        @tool(name="shell.run", description="Run a command.", read_only=False, target=reg)
        def run(params: ActionInput) -> ActionOutput:
            return ActionOutput(launched="ran")

        gate = ConfirmationGate(
            config, registry=reg, audit_path=tmp_path / "audit.jsonl", clock=clock
        )
        result = gate.check("shell.run", {"command": "Get-Date"})
        assert result.decision is GateDecision.DENIED
        assert "disabled" in result.reason


class TestConfirmationLifecycle:
    def test_affirmative_grants_execution(self, gate: ConfirmationGate, reg: ToolRegistry) -> None:
        pending = gate.dispatch("apps.launch", {"name": "notepad"})
        assert isinstance(pending, ConfirmationRequired)
        assert reg.calls == []  # type: ignore[attr-defined]

        assert gate.resolve(pending.token, "yes") is ConfirmationOutcome.AFFIRMATIVE
        result = gate.dispatch(
            "apps.launch", {"name": "notepad"}, confirmation_token=pending.token
        )
        assert isinstance(result, ToolResult)
        assert result.ok is True
        assert reg.calls == ["notepad"]  # type: ignore[attr-defined]

    def test_negative_blocks_execution(self, gate: ConfirmationGate, reg: ToolRegistry) -> None:
        pending = gate.dispatch("apps.launch", {"name": "notepad"})
        assert isinstance(pending, ConfirmationRequired)
        assert gate.resolve(pending.token, "no") is ConfirmationOutcome.NEGATIVE
        result = gate.dispatch(
            "apps.launch", {"name": "notepad"}, confirmation_token=pending.token
        )
        assert isinstance(result, ToolResult)
        assert result.ok is False
        assert reg.calls == []  # type: ignore[attr-defined]

    def test_unclear_blocks_execution(self, gate: ConfirmationGate, reg: ToolRegistry) -> None:
        pending = gate.dispatch("apps.launch", {"name": "notepad"})
        assert isinstance(pending, ConfirmationRequired)
        assert gate.resolve(pending.token, "what") is ConfirmationOutcome.UNCLEAR
        assert reg.calls == []  # type: ignore[attr-defined]

    def test_timeout_equals_denial(
        self, gate: ConfirmationGate, reg: ToolRegistry, clock: _Clock, cfg: JarvisConfig
    ) -> None:
        """§6: timeout equals denial. Even a later yes must not run the tool."""
        pending = gate.dispatch("apps.launch", {"name": "notepad"})
        assert isinstance(pending, ConfirmationRequired)
        clock.advance(cfg.gate.confirmation_timeout_s + 0.1)
        assert gate.resolve(pending.token, "yes") is ConfirmationOutcome.TIMEOUT
        result = gate.dispatch(
            "apps.launch", {"name": "notepad"}, confirmation_token=pending.token
        )
        assert isinstance(result, ToolResult)
        assert result.ok is False
        assert reg.calls == []  # type: ignore[attr-defined]

    def test_explicit_expire(self, gate: ConfirmationGate) -> None:
        pending = gate.dispatch("apps.launch", {})
        assert isinstance(pending, ConfirmationRequired)
        assert gate.expire(pending.token) is ConfirmationOutcome.TIMEOUT
        assert gate.is_pending(pending.token) is False

    def test_expire_revokes_an_already_issued_grant(
        self, gate: ConfirmationGate, reg: ToolRegistry
    ) -> None:
        """Reporting a timeout while leaving the grant usable would run the
        very action the caller just cancelled."""
        pending = gate.dispatch("apps.launch", {"name": "notepad"})
        assert isinstance(pending, ConfirmationRequired)
        gate.resolve(pending.token, "yes")
        gate.expire(pending.token)

        result = gate.dispatch(
            "apps.launch", {"name": "notepad"}, confirmation_token=pending.token
        )
        assert isinstance(result, ToolResult)
        assert result.ok is False
        assert reg.calls == []  # type: ignore[attr-defined]


class TestTokenForgery:
    """The gate must not trust a token it did not issue for this exact call."""

    def test_fabricated_token_is_rejected(
        self, gate: ConfirmationGate, reg: ToolRegistry
    ) -> None:
        result = gate.dispatch(
            "apps.launch", {"name": "notepad"}, confirmation_token="deadbeefdeadbeef"
        )
        assert isinstance(result, ToolResult)
        assert result.ok is False
        assert reg.calls == []  # type: ignore[attr-defined]

    def test_token_is_single_use(self, gate: ConfirmationGate, reg: ToolRegistry) -> None:
        pending = gate.dispatch("apps.launch", {"name": "notepad"})
        assert isinstance(pending, ConfirmationRequired)
        gate.resolve(pending.token, "yes")
        gate.dispatch("apps.launch", {"name": "notepad"}, confirmation_token=pending.token)
        gate.dispatch("apps.launch", {"name": "notepad"}, confirmation_token=pending.token)
        assert reg.calls == ["notepad"], "a replayed token must not run the tool twice"  # type: ignore[attr-defined]

    def test_token_does_not_transfer_to_another_tool(
        self, gate: ConfirmationGate, reg: ToolRegistry
    ) -> None:
        """A yes for launching notepad is not a yes for anything else."""
        pending = gate.dispatch("apps.launch", {"name": "notepad"})
        assert isinstance(pending, ConfirmationRequired)
        gate.resolve(pending.token, "yes")
        result = gate.dispatch("evil.wipe", {}, confirmation_token=pending.token)
        assert isinstance(result, ToolResult)
        assert result.ok is False
        assert "WIPED" not in reg.calls  # type: ignore[attr-defined]

    def test_token_does_not_transfer_to_different_arguments(
        self, gate: ConfirmationGate, reg: ToolRegistry
    ) -> None:
        """Confirming "launch notepad" must not authorise "launch cmd"."""
        pending = gate.dispatch("apps.launch", {"name": "notepad"})
        assert isinstance(pending, ConfirmationRequired)
        gate.resolve(pending.token, "yes")
        result = gate.dispatch(
            "apps.launch", {"name": "cmd.exe"}, confirmation_token=pending.token
        )
        assert isinstance(result, ToolResult)
        assert result.ok is False
        assert reg.calls == []  # type: ignore[attr-defined]

    def test_grant_expires(
        self, gate: ConfirmationGate, reg: ToolRegistry, clock: _Clock
    ) -> None:
        pending = gate.dispatch("apps.launch", {"name": "notepad"})
        assert isinstance(pending, ConfirmationRequired)
        gate.resolve(pending.token, "yes")
        clock.advance(120.0)
        result = gate.dispatch(
            "apps.launch", {"name": "notepad"}, confirmation_token=pending.token
        )
        assert isinstance(result, ToolResult)
        assert result.ok is False
        assert reg.calls == []  # type: ignore[attr-defined]


class TestRegistryBackstop:
    def test_registry_refuses_even_if_the_gate_is_bypassed(self, reg: ToolRegistry) -> None:
        """Defence in depth: calling the registry directly must still refuse."""
        result = reg.dispatch("apps.launch", {"name": "notepad"})
        assert result.ok is False
        assert reg.calls == []  # type: ignore[attr-defined]


class TestClockSafety:
    def test_deadlines_use_the_monotonic_clock(self, cfg: JarvisConfig) -> None:
        """A wall clock can step backwards, silently stretching the §6 window."""
        import time as time_module

        gate = ConfirmationGate(cfg)
        assert gate._monotonic_source is time_module.monotonic

    def test_a_backwards_wall_clock_does_not_extend_the_window(
        self, cfg: JarvisConfig, reg: ToolRegistry, tmp_path: Path
    ) -> None:
        wall = [1_000.0]
        mono = [500.0]
        gate = ConfirmationGate(
            cfg,
            registry=reg,
            audit_path=tmp_path / "audit.jsonl",
            clock=lambda: wall[0],
            monotonic=lambda: mono[0],
        )
        pending = gate.dispatch("apps.launch", {"name": "notepad"})
        assert isinstance(pending, ConfirmationRequired)

        wall[0] -= 3_600                                      # the wall clock jumps back
        mono[0] += cfg.gate.confirmation_timeout_s + 1         # real time still elapses

        assert gate.resolve(pending.token, "yes") is ConfirmationOutcome.TIMEOUT
        assert reg.calls == []  # type: ignore[attr-defined]


class TestGrantHousekeeping:
    def test_grants_do_not_accumulate_forever(
        self, cfg: JarvisConfig, reg: ToolRegistry, tmp_path: Path
    ) -> None:
        mono = [0.0]
        gate = ConfirmationGate(
            cfg,
            registry=reg,
            audit_path=tmp_path / "audit.jsonl",
            clock=lambda: 1_000.0,
            monotonic=lambda: mono[0],
        )
        for index in range(300):
            pending = gate.dispatch("apps.launch", {"name": f"app{index}"})
            if isinstance(pending, ConfirmationRequired):
                gate.resolve(pending.token, "yes")
            mono[0] += 1

        assert len(gate._grants) < 100, "grants are never swept"


class TestAuditLog:
    def test_every_decision_is_recorded(
        self, gate: ConfirmationGate, tmp_path: Path
    ) -> None:
        pending = gate.dispatch("apps.launch", {"name": "notepad"})
        assert isinstance(pending, ConfirmationRequired)
        gate.resolve(pending.token, "yes")
        gate.dispatch("apps.launch", {"name": "notepad"}, confirmation_token=pending.token)
        gate.check("evil.wipe", {})

        lines = (tmp_path / "gate_audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
        events = [json.loads(line)["event"] for line in lines]
        assert "confirmation_requested" in events
        assert "confirmation_resolved" in events
        assert "executed" in events
        assert "denied" in events

    def test_records_are_valid_json_with_the_required_fields(
        self, gate: ConfirmationGate, tmp_path: Path
    ) -> None:
        pending = gate.dispatch("apps.launch", {"name": "notepad"})
        assert isinstance(pending, ConfirmationRequired)
        gate.resolve(pending.token, "yes")
        for line in (tmp_path / "gate_audit.jsonl").read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            assert {"ts", "event", "tool", "confirmed", "reason"} <= set(record)

    def test_a_timed_out_confirmation_leaves_a_record(
        self, cfg: JarvisConfig, reg: ToolRegistry, tmp_path: Path
    ) -> None:
        """A confirmation that simply expired must not vanish from the log."""
        mono = [0.0]
        gate = ConfirmationGate(
            cfg,
            registry=reg,
            audit_path=tmp_path / "audit.jsonl",
            clock=lambda: 1_000.0,
            monotonic=lambda: mono[0],
        )
        gate.dispatch("apps.launch", {"name": "notepad"})
        mono[0] += cfg.gate.confirmation_timeout_s + 1
        _ = gate.pending_count

        events = [
            json.loads(line)["event"]
            for line in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert "confirmation_timeout" in events

    def test_execution_is_recorded_after_it_happens(
        self, gate: ConfirmationGate, tmp_path: Path
    ) -> None:
        """Logging before dispatch recorded runs that never happened."""
        pending = gate.dispatch("apps.launch", {"name": "notepad"})
        assert isinstance(pending, ConfirmationRequired)
        gate.resolve(pending.token, "yes")
        gate.dispatch("apps.launch", {"name": "notepad"}, confirmation_token=pending.token)

        records = [
            json.loads(line)
            for line in (tmp_path / "gate_audit.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert records[-1]["event"] == "executed"

    def test_credentials_are_redacted(self, gate: ConfirmationGate, tmp_path: Path) -> None:
        gate.check("evil.wipe", {"api_token": "sk-secret", "name": "x"})
        body = (tmp_path / "gate_audit.jsonl").read_text(encoding="utf-8")
        assert "sk-secret" not in body
        assert "<redacted>" in body

    def test_unwritable_audit_path_does_not_raise(
        self, cfg: JarvisConfig, reg: ToolRegistry
    ) -> None:
        """Logging must never be the thing that breaks a safety decision."""
        gate = ConfirmationGate(cfg, registry=reg, audit_path=Path("/proc/nope/audit.jsonl"))
        assert gate.check("evil.wipe", {}).decision is GateDecision.DENIED


class TestDescribeAction:
    def test_names_the_action_and_asks(self, cfg: JarvisConfig) -> None:
        sentence = describe_action("apps.launch", {"name": "Spotify"}, cfg)
        assert "Spotify" in sentence
        assert sentence.rstrip().endswith("?")

    def test_uses_the_address_form(self, cfg: JarvisConfig) -> None:
        assert "sir" in describe_action("apps.launch", {"name": "x"}, cfg)

    def test_omits_address_when_empty(self, tmp_path: Path) -> None:
        config = load_config(tmp_path / "absent.yaml", persona={"user_address_form": ""})
        sentence = describe_action("apps.launch", {"name": "x"}, config)
        assert ", ." not in sentence
        assert sentence.count(",") == 0 or "sir" not in sentence

    def test_material_arguments_are_spoken(self, cfg: JarvisConfig) -> None:
        """The user must hear everything that changes what the call does."""
        sentence = describe_action(
            "shell.run", {"command": "Get-ChildItem", "working_dir": "C:/Users/me/Docs"}, cfg
        )
        assert "Get-ChildItem" in sentence
        assert "C:/Users/me/Docs" in sentence

        repeated = describe_action("media.control", {"action": "volume_down", "repeat": 10}, cfg)
        assert "10" in repeated

    def test_shell_command_is_spoken_back_verbatim(self, cfg: JarvisConfig) -> None:
        """The user must hear exactly what would run."""
        sentence = describe_action("shell.run", {"command": "Get-Volume"}, cfg)
        assert "Get-Volume" in sentence

    def test_unknown_tool_still_produces_a_sentence(self, cfg: JarvisConfig) -> None:
        assert describe_action("some.tool", {"a": 1}, cfg)


class TestPendingBookkeeping:
    def test_pending_count_tracks_open_requests(self, gate: ConfirmationGate) -> None:
        assert gate.pending_count == 0
        gate.check("apps.launch", {"name": "a"})
        assert gate.pending_count == 1

    def test_expired_requests_are_swept(
        self, gate: ConfirmationGate, clock: _Clock, cfg: JarvisConfig
    ) -> None:
        gate.check("apps.launch", {"name": "a"})
        clock.advance(cfg.gate.confirmation_timeout_s + 1)
        assert gate.pending_count == 0

    def test_tokens_are_unique(self, gate: ConfirmationGate) -> None:
        first = gate.check("apps.launch", {"name": "a"}).confirmation
        second = gate.check("apps.launch", {"name": "b"}).confirmation
        assert first is not None
        assert second is not None
        assert first.token != second.token

    def test_seconds_remaining_never_negative(self, gate: ConfirmationGate) -> None:
        pending = gate.check("apps.launch", {}).confirmation
        assert pending is not None
        assert pending.seconds_remaining >= 0
