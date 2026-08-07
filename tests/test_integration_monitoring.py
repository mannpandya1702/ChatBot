"""T-2.11 and T-1.9 verification: the turn loop end to end.

This is the test that proves the pieces fit. A scripted Ollama drives the real
orchestrator, the real registry, the real gate, and the real memory, over the
real system tools. Nothing about the conversational core is stubbed except the
model itself and the speaker.

The headline case is the one the ledger names: "how's my system doing" should
chain CPU, memory, disk, and GPU into one spoken summary.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from jarvis.brain.llm import ChatChunk, ToolCall
from jarvis.brain.memory import ConversationMemory
from jarvis.brain.orchestrator import Orchestrator
from jarvis.config import JarvisConfig, load_config
from jarvis.state import EventBus, EventType
from jarvis.tools import (  # noqa: F401 - imported for their registration side effect
    sys_cpu,
    sys_disk,
    sys_gpu,
    sys_memory,
    sys_network,
    sys_process,
)
from jarvis.tools.registry import registry


class ScriptedLlm:
    """An Ollama stand-in that replays a fixed script of rounds.

    Each round is either a list of content strings or a list of ToolCall
    objects. Records the messages it was given so the test can assert on what
    the model actually saw.
    """

    def __init__(self, rounds: list[list[Any]]) -> None:
        self.rounds = rounds
        self.calls: list[list[dict[str, Any]]] = []
        self.tools_offered: list[list[dict[str, Any]]] = []
        self.round_index = 0

    def chat_stream(
        self,
        messages: Any,
        *,
        tools: Any = None,
        model: Any = None,
        think: Any = None,
    ) -> Iterator[ChatChunk]:
        self.calls.append(list(messages))
        self.tools_offered.append(list(tools or []))

        if self.round_index >= len(self.rounds):
            yield ChatChunk(content="", done=True, metrics={})
            return

        script = self.rounds[self.round_index]
        self.round_index += 1

        tool_calls = [item for item in script if isinstance(item, ToolCall)]
        contents = [item for item in script if isinstance(item, str)]

        for piece in contents:
            yield ChatChunk(content=piece)
        if tool_calls:
            yield ChatChunk(tool_calls=tool_calls)
        yield ChatChunk(content="", done=True, metrics={"eval_count": 12})

    def chat(self, messages: Any, *, tools: Any = None, model: Any = None) -> Any:
        from jarvis.brain.llm import ChatResponse

        content = "".join(chunk.content for chunk in self.chat_stream(messages, tools=tools))
        return ChatResponse(content=content)

    def close(self) -> None:
        return None


@pytest.fixture
def config(tmp_path: Path) -> JarvisConfig:
    return load_config(
        tmp_path / "absent.yaml",
        memory={"db_path": str(tmp_path / "memory.db"), "session_id": "test"},
        gate={"audit_log": str(tmp_path / "gate.jsonl")},
        paths={"log_dir": str(tmp_path / "logs"), "data_dir": str(tmp_path / "data")},
    )


def _build(config: JarvisConfig, llm: ScriptedLlm, bus: EventBus) -> Orchestrator:
    memory = ConversationMemory(config, db_path=Path(config.memory.db_path))
    return Orchestrator(config, bus, llm=llm, memory=memory)  # type: ignore[arg-type]


class TestSimpleTurn:
    def test_plain_answer_round_trips(self, config: JarvisConfig, bus: EventBus) -> None:
        llm = ScriptedLlm([["It is half past four."]])
        spoken: list[str] = []
        with _build(config, llm, bus) as orchestrator:
            result = orchestrator.run_turn("what time is it", speak=spoken.append)

        assert result.ok
        assert result.text == "It is half past four."
        assert spoken == ["It is half past four."]

    def test_empty_input_is_a_no_op(self, config: JarvisConfig, bus: EventBus) -> None:
        llm = ScriptedLlm([["should not be reached"]])
        with _build(config, llm, bus) as orchestrator:
            assert orchestrator.run_turn("   ").text == ""
        assert llm.calls == []

    def test_the_system_prompt_leads(self, config: JarvisConfig, bus: EventBus) -> None:
        llm = ScriptedLlm([["Fine."]])
        with _build(config, llm, bus) as orchestrator:
            orchestrator.run_turn("hello")
        first = llm.calls[0][0]
        assert first["role"] == "system"
        assert "Jarvis" in first["content"]

    def test_tools_are_offered_to_the_model(self, config: JarvisConfig, bus: EventBus) -> None:
        llm = ScriptedLlm([["Fine."]])
        with _build(config, llm, bus) as orchestrator:
            orchestrator.run_turn("hello")
        names = {tool["function"]["name"] for tool in llm.tools_offered[0]}
        assert "sys.cpu" in names
        assert "sys.memory" in names

    def test_streaming_chunks_are_spoken_as_they_arrive(
        self, config: JarvisConfig, bus: EventBus
    ) -> None:
        """§3: playback must start on the first sentence, not at the end."""
        llm = ScriptedLlm(
            [["The processor is at forty three percent. ", "Memory is comfortable. "]]
        )
        spoken: list[str] = []
        with _build(config, llm, bus) as orchestrator:
            orchestrator.run_turn("how is the cpu", speak=spoken.append)
        assert len(spoken) >= 2
        assert spoken[0].startswith("The processor")

    def test_conversation_is_remembered(self, config: JarvisConfig, bus: EventBus) -> None:
        llm = ScriptedLlm([["Noted."], ["Your name is Mann."]])
        with _build(config, llm, bus) as orchestrator:
            orchestrator.run_turn("my name is Mann")
            orchestrator.run_turn("what is my name")

        second = llm.calls[1]
        contents = " ".join(str(message.get("content", "")) for message in second)
        assert "my name is Mann" in contents


class TestToolChaining:
    def test_system_health_chains_four_tools(
        self, config: JarvisConfig, bus: EventBus
    ) -> None:
        """The T-2.11 headline case: one question, four tools, one summary."""
        llm = ScriptedLlm(
            [
                [
                    ToolCall(name="sys.cpu", arguments={"interval_s": 0.1}),
                    ToolCall(name="sys.memory", arguments={}),
                    ToolCall(name="sys.disk", arguments={}),
                    ToolCall(name="sys.gpu", arguments={}),
                ],
                [
                    "Everything looks healthy. The processor is light, memory is "
                    "comfortable, and there is plenty of disk space."
                ],
            ]
        )
        spoken: list[str] = []
        with _build(config, llm, bus) as orchestrator:
            result = orchestrator.run_turn("how's my system doing", speak=spoken.append)

        assert result.ok
        assert result.tool_calls == ["sys.cpu", "sys.memory", "sys.disk", "sys.gpu"]
        assert "healthy" in result.text
        assert spoken, "the summary was never spoken"

    def test_tool_results_reach_the_model(self, config: JarvisConfig, bus: EventBus) -> None:
        """The second round must actually contain the readings."""
        llm = ScriptedLlm(
            [[ToolCall(name="sys.cpu", arguments={"interval_s": 0.1})], ["Light load."]]
        )
        with _build(config, llm, bus) as orchestrator:
            orchestrator.run_turn("how busy is the cpu")

        second = llm.calls[1]
        tool_messages = [m for m in second if m.get("role") == "tool"]
        assert tool_messages, "no tool result was fed back"
        assert "cpu_percent" in tool_messages[0]["content"]

    def test_real_readings_are_returned(self, config: JarvisConfig, bus: EventBus) -> None:
        """Not a stub: the numbers come from psutil on this machine."""
        llm = ScriptedLlm([[ToolCall(name="sys.memory", arguments={})], ["Fine."]])
        with _build(config, llm, bus) as orchestrator:
            orchestrator.run_turn("how much memory")

        tool_message = next(m for m in llm.calls[1] if m.get("role") == "tool")
        payload = json.loads(tool_message["content"])
        assert payload["total_gb"] > 0
        assert 0 <= payload["percent_used"] <= 100

    def test_a_failing_tool_does_not_kill_the_turn(
        self, config: JarvisConfig, bus: EventBus
    ) -> None:
        """§5: a tool exception must never crash the loop."""
        llm = ScriptedLlm(
            [
                [ToolCall(name="sys.nonexistent", arguments={})],
                ["I could not read that."],
            ]
        )
        with _build(config, llm, bus) as orchestrator:
            result = orchestrator.run_turn("do something impossible")
        assert result.ok
        assert result.text == "I could not read that."

    def test_tool_events_are_published(self, config: JarvisConfig, bus: EventBus) -> None:
        seen: list[str] = []
        bus.subscribe(
            lambda event: seen.append(str(event.type)),
            [EventType.TOOL_CALL, EventType.TOOL_RESULT],
        )
        llm = ScriptedLlm([[ToolCall(name="sys.cpu", arguments={"interval_s": 0.1})], ["Ok."]])
        with _build(config, llm, bus) as orchestrator:
            orchestrator.run_turn("cpu")
        assert "tool_call" in seen
        assert "tool_result" in seen

    def test_iteration_ceiling_is_enforced(self, config: JarvisConfig, bus: EventBus) -> None:
        """A model that only ever calls tools must not loop forever.

        The cap is on tool rounds. One further round runs with the tools taken
        away, to get an answer out of the results already gathered, so the
        ceiling costs the user a worse answer rather than no answer.
        """
        rounds = [[ToolCall(name="sys.cpu", arguments={"interval_s": 0.1})] for _ in range(20)]
        llm = ScriptedLlm(rounds)
        with _build(config, llm, bus) as orchestrator:
            result = orchestrator.run_turn("loop forever")
        assert llm.round_index <= config.llm.max_tool_iterations + 1
        assert result.ok

    def test_the_ceiling_still_says_something(
        self, config: JarvisConfig, bus: EventBus
    ) -> None:
        """It used to end the turn silently and report success.

        No speech, no RESPONSE event, ok=True, and a memory window full of tool
        results nobody spoke. From the user's side the assistant simply said
        nothing and looked fine, which is the hardest kind of failure to report.
        """
        rounds = [[ToolCall(name="sys.cpu", arguments={"interval_s": 0.1})] for _ in range(20)]
        spoken: list[str] = []
        with _build(config, ScriptedLlm(rounds), bus) as orchestrator:
            orchestrator.run_turn("loop forever", speak=spoken.append)
        assert spoken, "the turn ended without the assistant saying anything"

    def test_the_ceiling_falls_back_when_the_model_says_nothing(
        self, config: JarvisConfig, bus: EventBus
    ) -> None:
        """Even the final round can come back empty."""

        class Mute(ScriptedLlm):
            def chat(self, messages: Any, **kwargs: Any) -> Any:
                from jarvis.brain.llm import ChatResponse

                return ChatResponse(content="")

        rounds = [[ToolCall(name="sys.cpu", arguments={"interval_s": 0.1})] for _ in range(20)]
        spoken: list[str] = []
        with _build(config, Mute(rounds), bus) as orchestrator:
            orchestrator.run_turn("loop forever", speak=spoken.append)
        assert any("could not" in line for line in spoken), spoken


class TestToolBoundary:
    """Regressions for text that straddles a tool call."""

    def _preamble_llm(self) -> Any:
        class Preamble(ScriptedLlm):
            def chat_stream(self, messages: Any, **kwargs: Any) -> Iterator[ChatChunk]:
                self.calls.append(list(messages))
                self.round_index += 1
                if self.round_index == 1:
                    yield ChatChunk(content="Let me check that for you. ")
                    yield ChatChunk(
                        tool_calls=[ToolCall(name="sys.cpu", arguments={"interval_s": 0.1})]
                    )
                    yield ChatChunk(content="", done=True)
                else:
                    yield ChatChunk(content="The processor is light. ")
                    yield ChatChunk(content="", done=True)

        return Preamble([])

    def test_preamble_is_not_recorded_twice(
        self, config: JarvisConfig, bus: EventBus
    ) -> None:
        """Text before a tool call must reach memory once, not once per round.

        The chunker buffers until it sees a sentence boundary plus more text, so
        without an explicit flush at the tool boundary the preamble is carried
        into the next round and written again with the final answer.
        """
        memory = ConversationMemory(config, db_path=Path(config.memory.db_path))
        with Orchestrator(config, bus, llm=self._preamble_llm(), memory=memory) as orch:
            orch.run_turn("how busy is the cpu")

        assistant = [m.content for m in memory.window() if m.role == "assistant"]
        assert sum("Let me check" in line for line in assistant) == 1, assistant

    def test_preamble_is_spoken_before_the_tool_runs(
        self, config: JarvisConfig, bus: EventBus
    ) -> None:
        """Otherwise the user hears nothing until the tool has returned."""
        memory = ConversationMemory(config, db_path=Path(config.memory.db_path))
        order: list[str] = []

        def record(chunk: str) -> None:
            order.append(f"speak:{chunk[:12]}")

        original_subscribe = bus.subscribe
        bus.subscribe(
            lambda event: order.append("tool"), [EventType.TOOL_CALL]
        )
        assert original_subscribe is not None

        with Orchestrator(config, bus, llm=self._preamble_llm(), memory=memory) as orch:
            orch.run_turn("how busy is the cpu", speak=record)

        assert order[0].startswith("speak:Let me check"), order
        assert "tool" in order
        assert order.index("tool") > 0


class TestMutatingToolsAreGated:
    def test_a_mutating_call_without_a_listener_is_refused(
        self, config: JarvisConfig, bus: EventBus
    ) -> None:
        """§6: no listener means no confirmation, which means no."""
        from jarvis.tools import apps  # noqa: F401 - registers apps.launch

        llm = ScriptedLlm(
            [
                [ToolCall(name="apps.launch", arguments={"name": "notepad"})],
                ["I have not done that."],
            ]
        )
        with _build(config, llm, bus) as orchestrator:
            result = orchestrator.run_turn("open notepad")

        tool_messages = [m for m in llm.calls[1] if m.get("role") == "tool"]
        assert tool_messages
        assert "not confirm" in tool_messages[0]["content"] or "error" in tool_messages[0][
            "content"
        ]
        assert result.ok

    def test_a_spoken_no_blocks_the_action(
        self, config: JarvisConfig, bus: EventBus
    ) -> None:
        from jarvis.tools import apps  # noqa: F401

        llm = ScriptedLlm(
            [[ToolCall(name="apps.launch", arguments={"name": "notepad"})], ["Understood."]]
        )
        with _build(config, llm, bus) as orchestrator:
            orchestrator.set_listener(lambda _timeout: "no, cancel that")
            result = orchestrator.run_turn("open notepad")
        assert result.ok

    def test_the_confirmation_prompt_is_spoken(
        self, config: JarvisConfig, bus: EventBus
    ) -> None:
        from jarvis.tools import apps  # noqa: F401

        spoken: list[str] = []
        llm = ScriptedLlm(
            [[ToolCall(name="apps.launch", arguments={"name": "notepad"})], ["Done."]]
        )
        with _build(config, llm, bus) as orchestrator:
            orchestrator.set_listener(lambda _timeout: "no")
            orchestrator.run_turn("open notepad", speak=spoken.append)

        assert any("notepad" in line and "?" in line for line in spoken), spoken

    def test_confirmation_events_are_published(
        self, config: JarvisConfig, bus: EventBus
    ) -> None:
        from jarvis.tools import apps  # noqa: F401

        seen: list[str] = []
        bus.subscribe(
            lambda event: seen.append(str(event.type)),
            [EventType.CONFIRMATION_REQUIRED, EventType.CONFIRMATION_RESOLVED],
        )
        llm = ScriptedLlm(
            [[ToolCall(name="apps.launch", arguments={"name": "notepad"})], ["Ok."]]
        )
        with _build(config, llm, bus) as orchestrator:
            orchestrator.set_listener(lambda _timeout: "no")
            orchestrator.run_turn("open notepad")
        assert "confirmation_required" in seen
        assert "confirmation_resolved" in seen


class TestBargeIn:
    def test_interrupting_stops_the_turn(self, config: JarvisConfig, bus: EventBus) -> None:
        """T-1.9: user speech kills the reply in progress."""
        orchestrator_ref: dict[str, Any] = {}

        class Interrupting(ScriptedLlm):
            def chat_stream(self, messages: Any, **kwargs: Any) -> Iterator[ChatChunk]:
                yield ChatChunk(content="This is a long answer. ")
                orchestrator_ref["it"].interrupt()
                yield ChatChunk(content="You should never hear this part. ")
                yield ChatChunk(content="", done=True)

        llm = Interrupting([])
        with _build(config, llm, bus) as orchestrator:
            orchestrator_ref["it"] = orchestrator
            spoken: list[str] = []
            result = orchestrator.run_turn("tell me a story", speak=spoken.append)

        assert result.interrupted is True
        assert "never hear this" not in " ".join(spoken)

    def test_barge_in_publishes_an_event(self, config: JarvisConfig, bus: EventBus) -> None:
        seen: list[str] = []
        bus.subscribe(lambda event: seen.append(str(event.type)), [EventType.BARGE_IN])
        llm = ScriptedLlm([["Answer."]])
        with _build(config, llm, bus) as orchestrator:
            orchestrator.interrupt()
            orchestrator.run_turn("anything")
        assert "barge_in" in seen


class TestFailureHandling:
    def test_an_llm_failure_becomes_a_spoken_apology(
        self, config: JarvisConfig, bus: EventBus
    ) -> None:
        from jarvis.util.errors import LlmError

        class Broken(ScriptedLlm):
            def chat_stream(self, messages: Any, **kwargs: Any) -> Iterator[ChatChunk]:
                raise LlmError("connection refused", speakable="Ollama is not running.")
                yield  # pragma: no cover

        spoken: list[str] = []
        with _build(config, Broken([]), bus) as orchestrator:
            result = orchestrator.run_turn("hello", speak=spoken.append)

        assert result.ok is False
        assert "Ollama is not running" in result.text
        assert spoken

    def test_an_unexpected_error_does_not_leak_internals(
        self, config: JarvisConfig, bus: EventBus
    ) -> None:
        """§5: a foreign exception's text must never be spoken."""

        class Exploding(ScriptedLlm):
            def chat_stream(self, messages: Any, **kwargs: Any) -> Iterator[ChatChunk]:
                raise RuntimeError("C:/Users/mann/secret/token=abc123")
                yield  # pragma: no cover

        with _build(config, Exploding([]), bus) as orchestrator:
            result = orchestrator.run_turn("hello")

        assert result.ok is False
        assert "abc123" not in result.text
        assert "secret" not in result.text

    def test_latency_is_recorded(self, config: JarvisConfig, bus: EventBus) -> None:
        llm = ScriptedLlm([["Quick answer."]])
        with _build(config, llm, bus) as orchestrator:
            result = orchestrator.run_turn("hi", speak=lambda _c: None)

        assert result.latency is not None
        breakdown = result.latency.breakdown()
        stages = {row["stage"] for row in breakdown["stages"]}
        assert "llm_first_token" in stages
        assert "turn_total" in stages


class TestRegistryCoverage:
    def test_every_registered_tool_produces_a_valid_schema(self) -> None:
        """A malformed schema would break tool calling for every model."""
        for spec in registry:
            schema = spec.json_schema()
            assert schema["type"] == "function"
            assert schema["function"]["name"] == spec.name
            assert schema["function"]["description"].strip()
            json.dumps(schema)

    def test_no_ref_survives_in_any_schema(self) -> None:
        payload = json.dumps([spec.json_schema() for spec in registry])
        assert "$ref" not in payload
        assert "$defs" not in payload

    def test_mutating_tools_are_exactly_the_allowlist(self, config: JarvisConfig) -> None:
        """§6 names six mutating tools. Nothing else may be registered as one."""
        import jarvis.tools.apps
        import jarvis.tools.media
        import jarvis.tools.reminders
        import jarvis.tools.shell  # noqa: F401

        mutating = {spec.name for spec in registry if not spec.read_only}
        assert mutating <= set(config.gate.mutating_allowlist), (
            f"{mutating - set(config.gate.mutating_allowlist)} mutate but are not allowlisted"
        )


class TestTheModelSeesWhatItAskedFor:
    """The assistant message that requested a tool must reach the next round.

    Ollama's Message struct is Role, Content, Thinking, Images, ToolCalls,
    ToolName, ToolCallID, and Go's JSON decoder drops unknown keys in silence.
    Qwen3's template gives a tool result no identity of its own, so the only
    thing binding a result to the question that produced it is the tool_calls
    block on the assistant message before it.

    Send neither and round two is the system prompt, the user's question, and
    then loose JSON with nothing saying the assistant went and measured
    anything. The model then answers as though reading a document rather than
    reporting its own instruments, which is what was reported from the Windows
    host as "it felt like it was reading instructions".
    """

    def _round_two(self, config: JarvisConfig) -> list[dict[str, Any]]:
        llm = ScriptedLlm(
            [
                [ToolCall(name="sys.cpu", arguments={"interval_s": 0.05})],
                ["Forty one percent, sir."],
            ]
        )
        orchestrator = _build(config, llm, EventBus())
        try:
            orchestrator.run_turn("how busy is the cpu")
        finally:
            orchestrator.close()
        assert len(llm.calls) >= 2, "the tool round never happened"
        return [dict(message) for message in llm.calls[1]]

    def test_the_assistant_message_carries_the_call_it_made(
        self, config: JarvisConfig
    ) -> None:
        assistant = [m for m in self._round_two(config) if m["role"] == "assistant"]
        assert assistant, "round two had no assistant message at all"
        names = [c["function"]["name"] for m in assistant for c in m.get("tool_calls", [])]
        assert names == ["sys.cpu"]

    def test_it_is_recorded_even_though_nothing_was_said_first(
        self, config: JarvisConfig
    ) -> None:
        """The usual case: the model calls the tool without any preamble.

        Storing only non-empty content is what dropped the call block, so an
        empty assistant turn is exactly the case that has to survive.
        """
        assistant = [m for m in self._round_two(config) if m["role"] == "assistant"]
        assert assistant[0]["content"] == ""
        assert assistant[0]["tool_calls"]

    def test_the_arguments_go_back_as_they_came(self, config: JarvisConfig) -> None:
        calls = [
            c
            for m in self._round_two(config)
            if m["role"] == "assistant"
            for c in m.get("tool_calls", [])
        ]
        assert calls[0]["function"]["arguments"] == {"interval_s": 0.05}

    def test_the_call_comes_before_its_result(self, config: JarvisConfig) -> None:
        roles = [m["role"] for m in self._round_two(config)]
        assert roles.index("assistant") < roles.index("tool")

    def test_the_result_names_itself_under_the_key_ollama_reads(
        self, config: JarvisConfig
    ) -> None:
        results = [m for m in self._round_two(config) if m["role"] == "tool"]
        assert [m.get("tool_name") for m in results] == ["sys.cpu"]

    def test_nothing_is_sent_under_the_key_ollama_discards(
        self, config: JarvisConfig
    ) -> None:
        assert not any("name" in message for message in self._round_two(config))

    def test_parallel_calls_all_come_back(self, config: JarvisConfig) -> None:
        llm = ScriptedLlm(
            [
                [
                    ToolCall(name="sys.cpu", arguments={"interval_s": 0.05}),
                    ToolCall(name="sys.memory", arguments={}),
                ],
                ["All quiet, sir."],
            ]
        )
        orchestrator = _build(config, llm, EventBus())
        try:
            orchestrator.run_turn("how is the machine")
        finally:
            orchestrator.close()

        round_two = [dict(m) for m in llm.calls[1]]
        requested = [
            c["function"]["name"]
            for m in round_two
            if m["role"] == "assistant"
            for c in m.get("tool_calls", [])
        ]
        answered = [m["tool_name"] for m in round_two if m["role"] == "tool"]
        assert sorted(requested) == sorted(answered) == ["sys.cpu", "sys.memory"]


class TestAnUnheardPromptAuthorisesNothing:
    """§6 defines the gate as speaking the action back and waiting for a yes.

    The speaking half was never checked. _emit_speech returns early when there
    is no speech callback and swallows any exception from one, so whenever
    tts.enabled is false or Kokoro fails, the assistant opened a fifteen second
    microphone window having asked nothing. Any affirmative in earshot, someone
    on a phone call saying "go ahead", then authorised apps.launch,
    media.control, reminders.delete or shell.run, and the user was never told
    what they had approved.

    Neither condition is exotic: tts.enabled is a shipped toggle and
    main._speak is written to log a synthesis failure and carry on.
    """

    def _turn(self, config: JarvisConfig, bus: EventBus, speak: Any) -> tuple[Any, list[str]]:
        from jarvis.tools import apps  # noqa: F401 - registers apps.launch

        llm = ScriptedLlm(
            [[ToolCall(name="apps.launch", arguments={"name": "notepad"})], ["Done."]]
        )
        heard: list[str] = []
        with _build(config, llm, bus) as orchestrator:
            orchestrator.set_listener(lambda _timeout: heard.append("listened") or "yes")
            orchestrator.run_turn("open notepad", speak=speak)
        payloads = [m["content"] for m in llm.calls[1] if m.get("role") == "tool"]
        return payloads, heard

    def test_no_speech_callback_means_no_confirmation(
        self, config: JarvisConfig, bus: EventBus
    ) -> None:
        """The tts.enabled false path, where main._speak returns immediately."""
        payloads, heard = self._turn(config, bus, None)
        assert payloads and "could not be spoken" in payloads[0]
        assert not heard, "the microphone was opened for a question nobody was asked"

    def test_a_synthesis_failure_means_no_confirmation(
        self, config: JarvisConfig, bus: EventBus
    ) -> None:
        """The Kokoro failure path, which main._speak logs and swallows."""

        def broken(_text: str) -> None:
            from jarvis.util.errors import TtsError

            raise TtsError("no espeak backend", speakable="I could not speak that.")

        payloads, heard = self._turn(config, bus, broken)
        assert payloads and "could not be spoken" in payloads[0]
        assert not heard, "the microphone was opened for a question nobody heard"

    def test_a_working_voice_still_confirms(
        self, config: JarvisConfig, bus: EventBus
    ) -> None:
        """The guard must refuse silence, not refuse everything."""
        payloads, heard = self._turn(config, bus, lambda _text: None)
        assert heard, "the confirmation window never opened"
        assert payloads and "could not be spoken" not in payloads[0]

    def test_the_refusal_is_published(self, config: JarvisConfig, bus: EventBus) -> None:
        outcomes: list[str] = []
        bus.subscribe(
            lambda event: outcomes.append(str(event.payload.get("outcome"))),
            [EventType.CONFIRMATION_RESOLVED],
        )
        self._turn(config, bus, None)
        assert "unspoken" in outcomes


class TestTheUserGetsTheWholeWindow:
    """§6 promises a window to answer the question the user was just asked.

    The deadline was armed inside the gate's check(), which runs before the
    prompt has been synthesised, let alone played. Speaking it then ate into the
    same window: on the cpu tier routinely five to eight seconds of it. The user
    said yes, heard nothing happen, and was told nothing.

    shell.run failed closed outright. Its prompt embeds the whole command and
    ShellInput.command allows 512 characters, which takes far longer to speak
    than the window, so that call could never be confirmed at all.
    """

    def test_speaking_the_prompt_does_not_eat_the_window(
        self, config: JarvisConfig, bus: EventBus
    ) -> None:
        from jarvis.tools import apps  # noqa: F401

        clock = {"now": 1_000.0}
        llm = ScriptedLlm(
            [[ToolCall(name="apps.launch", arguments={"name": "notepad"})], ["Done."]]
        )
        with _build(config, llm, bus) as orchestrator:
            orchestrator.gate._monotonic_source = lambda: clock["now"]

            def slow_speak(_text: str) -> None:
                # A long prompt on a slow tier, longer than the whole window.
                clock["now"] += config.gate.confirmation_timeout_s + 5.0

            def answer(_timeout: float) -> str:
                clock["now"] += 1.0  # the user replies promptly
                return "yes"

            orchestrator.set_listener(answer)
            orchestrator.run_turn("open notepad", speak=slow_speak)

        payloads = [m["content"] for m in llm.calls[1] if m.get("role") == "tool"]
        assert payloads
        assert "timeout" not in payloads[0], (
            f"a prompt answer was discarded as a timeout: {payloads[0]}"
        )

    def test_a_slow_answer_still_times_out(
        self, config: JarvisConfig, bus: EventBus
    ) -> None:
        """Rearming must move the window, not remove it."""
        from jarvis.tools import apps  # noqa: F401

        clock = {"now": 1_000.0}
        llm = ScriptedLlm(
            [[ToolCall(name="apps.launch", arguments={"name": "notepad"})], ["Done."]]
        )
        with _build(config, llm, bus) as orchestrator:
            orchestrator.gate._monotonic_source = lambda: clock["now"]
            orchestrator.set_listener(
                lambda _t: (clock.__setitem__("now", clock["now"] + 60.0), "yes")[1]
            )
            orchestrator.run_turn("open notepad", speak=lambda _t: None)

        payloads = [m["content"] for m in llm.calls[1] if m.get("role") == "tool"]
        assert payloads and "timeout" in payloads[0]

    def test_arm_reports_an_unknown_token(self, config: JarvisConfig) -> None:
        from jarvis.tools.gate import ConfirmationGate

        assert ConfirmationGate(config).arm("not-a-token") is False
