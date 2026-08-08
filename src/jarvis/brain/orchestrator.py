"""The turn loop.

T-1.9. Wires wake word, endpointing, transcription, the LLM, tools, and speech
into one conversation, across threads, with barge-in.

The design decision that matters most for testability: :meth:`Orchestrator.run_turn`
takes text and returns text. Every dependency is injected, so the whole
conversational core, tool dispatch, the confirmation gate, memory, chunking, and
the streaming contract, is exercised without a microphone, a GPU, or Ollama. The
audio path on top of it is thin by design.

Barge-in works because capture never stops. The microphone runs into a ring
buffer continuously (§5), so during playback a second reader is already watching
for speech; when it fires, playback is cut and the turn is abandoned.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from jarvis.brain.llm import ChatChunk, OllamaClient, ToolCall, timing_seconds
from jarvis.brain.memory import ConversationMemory, MemoryMessage
from jarvis.brain.persona import (
    build_error_response,
    build_summary_prompt,
    build_system_prompt,
)
from jarvis.config import JarvisConfig
from jarvis.state import AssistantState, EventBus, EventType, StateMachine
from jarvis.tools.gate import ConfirmationGate, ConfirmationOutcome, ConfirmationRequired
from jarvis.tools.registry import ToolRegistry
from jarvis.tools.registry import registry as default_registry
from jarvis.util.errors import JarvisError, as_speakable
from jarvis.util.latency import Stage, TurnLatency

__all__ = ["Orchestrator", "TurnResult", "turn_budgets"]

_log = logging.getLogger(__name__)


def _log_model_timings(rounds: list[dict[str, Any]]) -> None:
    """Report what Ollama says it did, once per turn.

    ``prompt_eval_count`` is the number to read on a slow machine: it counts the
    prompt tokens Ollama evaluated rather than reused from its cache. The
    preamble here is roughly 3,750 tokens of system prompt and tool schemas, so
    a first turn reporting all of them is expected and a *later* turn reporting
    all of them again means the prompt prefix is moving between turns and the
    machine is paying for it every time. Nothing else in the logs can tell those
    two apart, and on the ``cpu`` tier they are seconds apart.
    """
    if not rounds:
        return
    prompt_tokens = sum(int(r.get("prompt_eval_count") or 0) for r in rounds)
    output_tokens = sum(int(r.get("eval_count") or 0) for r in rounds)
    seconds: dict[str, float] = {}
    for served in rounds:
        for key, value in timing_seconds(served).items():
            seconds[key] = round(seconds.get(key, 0.0) + value, 2)
    generated = seconds.get("eval_s", 0.0)
    _log.info(
        "model timings",
        extra={
            "context": {
                "round_trips": len(rounds),
                "prompt_tokens_evaluated": prompt_tokens,
                "output_tokens": output_tokens,
                "tokens_per_second": round(output_tokens / generated, 1) if generated else None,
                **seconds,
            }
        },
    )

#: Speech callback: receives each speakable chunk as soon as it is ready.
SpeakFn = Callable[[str], None]

#: Spoken when the tool loop runs out of rounds and the model still will not
#: answer. Says what happened without reading the tool results aloud.
_CEILING_SPEAKABLE = "I checked, but I could not put an answer together."


def _wire_call(call: ToolCall) -> dict[str, Any]:
    """Render a tool call back into the shape Ollama sent it in.

    It has to go back the way it came. Qwen3's chat template renders a tool
    result with no identity of its own, so the only thing binding a result to
    the question that produced it is this block on the preceding assistant
    message.
    """
    function: dict[str, Any] = {"name": call.name, "arguments": call.arguments}
    wire: dict[str, Any] = {"type": "function", "function": function}
    if call.id:
        wire["id"] = call.id
    return wire


#: Listener used to collect a spoken confirmation. Returns the transcript.
ListenFn = Callable[[float], str]


@dataclass
class TurnResult:
    """Everything one turn produced."""

    text: str
    tool_calls: list[str] = field(default_factory=list)
    interrupted: bool = False
    error: str | None = None
    latency: TurnLatency | None = None

    @property
    def ok(self) -> bool:
        """Whether the turn completed without an error."""
        return self.error is None


class Orchestrator:
    """Runs conversational turns.

    Every collaborator is injectable so the turn loop can be tested end to end
    without hardware.

    Args:
        config: Resolved configuration.
        bus: Event bus. One is created when omitted.
        registry: Tool registry. Defaults to the process-wide one.
        gate: Confirmation gate. Built from the registry when omitted.
        llm: Ollama client. Built from the config when omitted.
        memory: Conversation memory. Built from the config when omitted.
    """

    def __init__(
        self,
        config: JarvisConfig,
        bus: EventBus | None = None,
        *,
        registry: ToolRegistry | None = None,
        gate: ConfirmationGate | None = None,
        llm: OllamaClient | None = None,
        memory: ConversationMemory | None = None,
    ) -> None:
        self._config = config
        self.bus = bus or EventBus()
        self.state = StateMachine(self.bus)
        self._registry = default_registry if registry is None else registry
        self._llm = llm if llm is not None else OllamaClient(config)
        self._gate = gate or ConfirmationGate(config, registry=self._registry)
        self._memory = memory if memory is not None else ConversationMemory(
            config, summariser=self._summarise
        )

        self._interrupt = threading.Event()
        self._turn_lock = threading.Lock()
        self._listen: ListenFn | None = None

    # -- accessors ---------------------------------------------------------

    @property
    def memory(self) -> ConversationMemory:
        """The conversation memory."""
        return self._memory

    @property
    def gate(self) -> ConfirmationGate:
        """The confirmation gate."""
        return self._gate

    def set_listener(self, listener: ListenFn | None) -> None:
        """Install the function used to collect a spoken confirmation."""
        self._listen = listener

    def system_prompt(self) -> str:
        """The system prompt for the currently available tools."""
        names = [spec.name for spec in self._registry.select(config=self._config)]
        return build_system_prompt(self._config, tools=names)

    def warmup(self) -> dict[str, Any]:
        """Load the model and evaluate the preamble before the first question.

        Startup already warms the endpointer, the transcriber and the
        synthesiser, and the language model was the one left cold, so the first
        question of every session paid for loading the weights and for
        evaluating the whole preamble on top of everything else. On the ``cpu``
        tier that is the larger half of the wait.

        The prompt and the tools are built exactly as :meth:`run_turn` builds
        them, because Ollama's cache only helps a prefix it has already seen: a
        warmup with a different preamble buys the weights and nothing else.

        Returns:
            Ollama's timings, empty when it could not be reached. Never raises.
        """
        try:
            return self._llm.warmup(
                self.system_prompt(), tools=self._registry.ollama_tools(config=self._config)
            )
        except Exception:  # noqa: BLE001 - a cold model is slow, not broken
            _log.debug("could not warm the language model", exc_info=True)
            return {}

    # -- interruption ------------------------------------------------------

    def interrupt(self) -> None:
        """Abandon the turn in progress. Called by the barge-in detector."""
        if not self._interrupt.is_set():
            self._interrupt.set()
            self.bus.emit(EventType.BARGE_IN)
            _log.info("turn interrupted by the user")

    @property
    def interrupted(self) -> bool:
        """Whether the current turn has been interrupted."""
        return self._interrupt.is_set()

    # -- the turn ----------------------------------------------------------

    def run_turn(
        self,
        text: str,
        *,
        speak: SpeakFn | None = None,
        turn: TurnLatency | None = None,
    ) -> TurnResult:
        """Run one conversational turn.

        Args:
            text: What the user said.
            speak: Called with each speakable chunk as soon as it is ready, so
                playback starts on the first sentence rather than at the end.
            turn: Latency recorder. One is created when omitted.

        Returns:
            The full spoken reply plus what happened along the way. Never raises:
            a failure becomes a spoken apology, because §5 says a tool exception
            must not crash the loop.
        """
        from jarvis.audio.tts import SentenceChunker

        latency = turn or TurnLatency(budgets_ms=turn_budgets(self._config))
        # Compaction is a full LLM generation. Held until the turn is
        # over, where it overlaps with the reply already playing.
        with self._turn_lock, self._memory.deferred_compaction():
            self._interrupt.clear()
            spoken = text.strip()
            if not spoken:
                return TurnResult(text="", latency=latency)

            self.bus.emit(EventType.TRANSCRIPT, text=spoken)
            self._memory.add_user(spoken)
            self.state.set(AssistantState.THINKING)

            chunker = SentenceChunker(self._config)
            # Everything spoken this turn, which is what the caller gets back.
            spoken_all: list[str] = []
            # Only what has NOT yet been written to memory. A tool round writes
            # the text that preceded the call, so it is cleared afterwards;
            # otherwise the final write would record that text a second time.
            reply: list[str] = []
            called: list[str] = []
            first_token_marked = False
            # Ollama's own timings, one set per round trip. Worth logging: they
            # are the only place that says how much of the prompt was actually
            # evaluated rather than reused from the KV cache, which on a slow
            # machine is the difference between a pause and a wait.
            served: list[dict[str, Any]] = []

            try:
                for _round in range(self._config.llm.max_tool_iterations):
                    if self._interrupt.is_set():
                        return self._interrupted_result(spoken_all, called, latency)

                    messages = self._memory.messages(self.system_prompt())
                    tools = self._registry.ollama_tools(config=self._config)

                    content_parts: list[str] = []
                    calls: list[ToolCall] = []

                    for chunk in self._stream(messages, tools):
                        if self._interrupt.is_set():
                            return self._interrupted_result(spoken_all, called, latency)

                        if chunk.content:
                            if not first_token_marked:
                                latency.mark(Stage.LLM_FIRST_TOKEN)
                                first_token_marked = True
                            content_parts.append(chunk.content)
                            self.bus.emit(EventType.RESPONSE_CHUNK, text=chunk.content)
                            for ready in chunker.feed(chunk.content):
                                reply.append(ready)
                                spoken_all.append(ready)
                                self._emit_speech(ready, speak, latency)

                        calls.extend(chunk.tool_calls)
                        if chunk.metrics:
                            served.append(chunk.metrics)

                    if not calls:
                        for ready in chunker.flush():
                            reply.append(ready)
                            spoken_all.append(ready)
                            self._emit_speech(ready, speak, latency)
                        break

                    # A tool round. Flush the chunker here rather than letting
                    # it carry text into the next round: otherwise "Let me check
                    # that" sits in the buffer until after the tool returns, so
                    # it is spoken late and then merged into the final answer,
                    # which also records it in memory twice.
                    for ready in chunker.flush():
                        spoken_all.append(ready)
                        self._emit_speech(ready, speak, latency)

                    # The assistant message goes in even when it said nothing,
                    # because it is what carries the tool_calls. Without it the
                    # results that follow are anonymous JSON appearing after the
                    # user's question with nothing that asked for them, and the
                    # model answers as though it were reading a document rather
                    # than its own instruments.
                    said = "".join(content_parts).strip()
                    self._memory.add_assistant(said, tool_calls=[_wire_call(c) for c in calls])
                    # Recorded above, so it must not be recorded again at the end.
                    reply.clear()
                    called.extend(self._dispatch(calls, latency, speak))
                else:
                    # Falling through here used to end the turn: no speech, no
                    # RESPONSE event, ok=True, and a memory window full of tool
                    # results nobody ever spoke. From the user's side the
                    # assistant simply said nothing and looked fine. The
                    # results are already gathered, so ask once more with the
                    # tools taken away, which forces an answer out of them.
                    _log.warning(
                        "hit the tool iteration ceiling, answering from what was gathered",
                        extra={"context": {"limit": self._config.llm.max_tool_iterations}},
                    )
                    for ready in self._final_answer(latency):
                        reply.append(ready)
                        spoken_all.append(ready)
                        self._emit_speech(ready, speak, latency)

            except JarvisError as exc:
                return self._error_result(exc, spoken_all, called, latency, speak)
            except Exception as exc:  # noqa: BLE001 - §5, never crash the loop
                _log.exception("the turn failed")
                return self._error_result(exc, spoken_all, called, latency, speak)

            unrecorded = " ".join(part.strip() for part in reply if part.strip()).strip()
            if unrecorded:
                self._memory.add_assistant(unrecorded)

            full = " ".join(part.strip() for part in spoken_all if part.strip()).strip()
            if full:
                self.bus.emit(EventType.RESPONSE, text=full)

            latency.finish()
            latency.log(_log)
            _log_model_timings(served)
            self.bus.emit(EventType.LATENCY, **latency.breakdown())
            self.state.set(AssistantState.IDLE)
            return TurnResult(text=full, tool_calls=called, latency=latency)

    def _stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Iterator[ChatChunk]:
        """Stream a completion, timing the whole call."""
        started = time.perf_counter()
        try:
            yield from self._llm.chat_stream(messages, tools=tools or None)
        finally:
            _log.debug(
                "llm round complete",
                extra={"context": {"ms": round((time.perf_counter() - started) * 1000, 1)}},
            )

    def _final_answer(self, latency: TurnLatency) -> list[str]:
        """Ask for an answer with the tools taken away.

        Used when the tool loop runs out of rounds. The model has usually called
        the same tool over and over, so its results are sitting in memory
        unspoken; without a tool list it has nothing to do but say what they
        mean. Never raises: this already runs on a degraded path, and a failure
        here should leave the caller with the fallback sentence rather than an
        exception.
        """
        try:
            response = self._llm.chat(self._memory.messages(self.system_prompt()))
        except Exception:  # noqa: BLE001 - §5, never crash the loop
            _log.exception("could not summarise after the tool ceiling")
            return [build_error_response(_CEILING_SPEAKABLE, self._config)]

        text = str(response.content).strip()
        if not text:
            return [build_error_response(_CEILING_SPEAKABLE, self._config)]
        self._memory.add_assistant(text)
        return [text]

    def _emit_speech(self, text: str, speak: SpeakFn | None, latency: TurnLatency) -> bool:
        """Hand one chunk to the speech callback and record first audio.

        Returns:
            Whether the chunk actually reached the speech callback. For an
            ordinary sentence nobody looks: a failure there is logged and the
            turn goes on, because losing one sentence is better than losing the
            turn. The confirmation prompt is the exception, and §6 is the reason:
            it defines the gate as speaking the action back and waiting for an
            affirmative, so a prompt that was never rendered cannot be answered
            and must not open a microphone.
        """
        if not text.strip() or speak is None:
            return False
        self.state.set(AssistantState.SPEAKING)
        try:
            speak(text)
        except Exception:  # noqa: BLE001 - a speech failure must not end the turn
            _log.exception("speaking a chunk failed")
            return False
        # After, not before. The callback is what synthesises and queues the
        # audio, so marking first the way this used to measured time to hand
        # the text over, which is not a stage anyone budgeted. §3 budgets time
        # to first audio out, and a chunk that failed to speak produced none.
        if latency.get(Stage.TTS_FIRST_AUDIO) is None:
            latency.mark(Stage.TTS_FIRST_AUDIO)
        return True

    # -- tools -------------------------------------------------------------

    def _dispatch(
        self,
        calls: list[ToolCall],
        latency: TurnLatency,
        speak: SpeakFn | None,
    ) -> list[str]:
        """Run the requested tools, routing mutating ones through the gate."""
        names: list[str] = []
        for call in calls:
            if self._interrupt.is_set():
                break
            names.append(call.name)
            self.bus.emit(EventType.TOOL_CALL, name=call.name, arguments=call.arguments)

            with latency.stage(Stage.TOOL, tool=call.name):
                outcome = self._gate.dispatch(call.name, call.arguments)

                if isinstance(outcome, ConfirmationRequired):
                    outcome = self._confirm_and_run(outcome, speak)

            payload = outcome.for_llm()
            self._memory.add_tool(call.name, payload)
            self.bus.emit(EventType.TOOL_RESULT, name=call.name, ok=outcome.ok)
        return names

    def _confirm_and_run(self, pending: ConfirmationRequired, speak: SpeakFn | None) -> Any:
        """Speak the confirmation prompt, wait for a yes, then run or refuse."""
        from jarvis.tools.registry import ToolResult

        self.bus.emit(
            EventType.CONFIRMATION_REQUIRED, prompt=pending.prompt, tool=pending.tool
        )

        # §6 defines the gate as speaking the action back and waiting for an
        # explicit affirmative. If the prompt never reached a speaker there is
        # no question in the room, so opening the microphone would let any
        # affirmative in earshot, someone on a phone call saying "go ahead",
        # authorise a mutating tool the user was never told about. Silence is
        # not exotic: tts.enabled is a shipped toggle and main._speak swallows
        # a Kokoro failure by design.
        if not self._emit_speech(pending.prompt, speak, TurnLatency()):
            _log.warning(
                "refusing a mutating call whose confirmation prompt could not be spoken",
                extra={"context": {"tool": pending.tool}},
            )
            self._gate.expire(pending.token)
            self.bus.emit(
                EventType.CONFIRMATION_RESOLVED, tool=pending.tool, outcome="unspoken"
            )
            return ToolResult(
                tool=pending.tool,
                ok=False,
                error="the confirmation prompt could not be spoken, so nothing was confirmed",
                speakable="I have not done that.",
            )

        # A user talking over the confirmation prompt is interrupting, not
        # answering. Treating their speech as the reply would let a barge-in
        # authorise the very action they cut in to stop.
        if self._interrupt.is_set():
            self._gate.expire(pending.token)
            self.bus.emit(
                EventType.CONFIRMATION_RESOLVED, tool=pending.tool, outcome="interrupted"
            )
            return ToolResult(
                tool=pending.tool,
                ok=False,
                error="the user interrupted before confirming",
                speakable="I have not done that.",
            )

        answer = ""
        if self._listen is not None:
            # Start the answering window now, not when the gate was checked.
            # The listener blocks until playback has drained, so by this point
            # the question has actually been asked and the user has the full
            # window §6 promises rather than what is left of it.
            self._gate.arm(pending.token)
            self.state.set(AssistantState.LISTENING)
            try:
                answer = self._listen(self._config.gate.confirmation_timeout_s)
            except Exception:  # noqa: BLE001 - a listen failure is a denial
                _log.exception("listening for a confirmation failed")
                answer = ""

        if self._interrupt.is_set():
            self._gate.expire(pending.token)
            self.bus.emit(
                EventType.CONFIRMATION_RESOLVED, tool=pending.tool, outcome="interrupted"
            )
            return ToolResult(
                tool=pending.tool,
                ok=False,
                error="the user interrupted during confirmation",
                speakable="I have not done that.",
            )

        outcome = self._gate.resolve(pending.token, answer)
        self.bus.emit(
            EventType.CONFIRMATION_RESOLVED, tool=pending.tool, outcome=str(outcome)
        )

        if outcome is not ConfirmationOutcome.AFFIRMATIVE:
            # §6: anything that is not a clear yes is a no, including silence.
            _log.info(
                "confirmation not granted",
                extra={"context": {"tool": pending.tool, "outcome": str(outcome)}},
            )
            return ToolResult(
                tool=pending.tool,
                ok=False,
                error=f"the user did not confirm ({outcome})",
                speakable="I have not done that.",
            )

        result = self._gate.dispatch(
            pending.tool, pending.arguments, confirmation_token=pending.token
        )
        if isinstance(result, ConfirmationRequired):  # pragma: no cover - invariant
            return ToolResult(
                tool=pending.tool,
                ok=False,
                error="the gate asked for confirmation twice",
                speakable="I have not done that.",
            )
        return result

    # -- results -----------------------------------------------------------

    def _interrupted_result(
        self, reply: list[str], called: list[str], latency: TurnLatency
    ) -> TurnResult:
        """Close out a turn the user talked over."""
        partial = " ".join(reply).strip()
        if partial:
            self._memory.add_assistant(partial + " (interrupted)")
        latency.finish()
        self.state.set(AssistantState.LISTENING)
        return TurnResult(text=partial, tool_calls=called, interrupted=True, latency=latency)

    def _error_result(
        self,
        exc: BaseException,
        reply: list[str],
        called: list[str],
        latency: TurnLatency,
        speak: SpeakFn | None,
    ) -> TurnResult:
        """Turn a failure into a spoken apology."""
        spoken = build_error_response(as_speakable(exc), self._config)
        self.bus.emit(EventType.ERROR, message=str(exc), speakable=spoken)
        if speak is not None:
            self._emit_speech(spoken, speak, latency)
        latency.finish()
        self.state.set(AssistantState.ERROR)
        return TurnResult(
            text=spoken,
            tool_calls=called,
            error=f"{type(exc).__name__}: {exc}",
            latency=latency,
        )

    # -- memory ------------------------------------------------------------

    def _summarise(self, messages: list[MemoryMessage], previous: str) -> str:
        """Summariser handed to memory. Uses the configured summary model."""
        transcript = "\n".join(
            f"{message.role}: {message.content[:400]}" for message in messages
        )
        prompt = build_summary_prompt()
        if previous:
            prompt = f"{prompt}\n\nThe summary so far:\n{previous}"

        response = self._llm.chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": transcript},
            ],
            model=self._config.memory.summary_model or None,
        )
        return response.content.strip()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Release the LLM client and the memory database."""
        try:
            self._llm.close()
        except Exception:  # noqa: BLE001
            _log.debug("closing the llm client failed", exc_info=True)
        try:
            self._memory.close()
        except Exception:  # noqa: BLE001
            _log.debug("closing memory failed", exc_info=True)

    def __enter__(self) -> Orchestrator:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def turn_budgets(config: JarvisConfig) -> dict[Stage, float]:
    """Latency budgets from config, keyed by Stage.

    Public because ``main`` builds the turn's recorder now: §3 measures from the
    end of the user's speech, which is before the orchestrator sees the turn at
    all, so the two have to agree on the budgets.
    """
    out: dict[Stage, float] = {}
    for stage in Stage:
        value = config.latency.budgets_ms.get(str(stage))
        if value is not None:
            out[stage] = float(value)
    return out
