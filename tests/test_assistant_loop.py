"""End-to-end test of the threaded assistant loop in main.py.

This is the last path the other tests do not reach: the wake word firing, the
turn thread endpointing and transcribing, the reply being synthesised and
played, and everything shutting down cleanly. It runs with a fake capture
stream, a fake transcriber, a fake synthesiser, and a scripted model, so no
hardware is involved, but the threading, the supervisor, the shutdown ordering,
and the wiring in ``Assistant`` are all real.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from jarvis.audio.ring import StreamSpec
from jarvis.audio.stt import Transcript
from jarvis.brain.llm import ChatChunk
from jarvis.config import JarvisConfig, load_config
from jarvis.main import Assistant
from jarvis.state import AssistantState, EventType


class FakeStream:
    """Feeds silence into the capture callback on its own thread."""

    def __init__(self, spec: StreamSpec) -> None:
        self.spec = spec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.blocks_delivered = 0

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2)
            self._thread = None

    def close(self) -> None:
        self.stop()

    def _pump(self) -> None:
        block = np.zeros((self.spec.block_samples, self.spec.channels), dtype=np.float32)
        period = self.spec.block_samples / self.spec.sample_rate
        while not self._stop.is_set():
            with contextlib.suppress(Exception):
                # PortAudio swallows exceptions from the callback, so the fake
                # must too, otherwise a bug here looks like a capture failure.
                self.spec.callback(block, self.spec.block_samples, None, None)
                self.blocks_delivered += 1
            time.sleep(period)


class FakeTranscriber:
    """Returns a scripted utterance the first time, then silence."""

    def __init__(self, utterances: list[str]) -> None:
        self.utterances = list(utterances)
        self.calls = 0

    def transcribe(self, audio: Any, sample_rate: int = 16_000) -> Transcript:
        self.calls += 1
        text = self.utterances.pop(0) if self.utterances else ""
        return Transcript(
            text=text, language="en", duration_s=1.0, rtf=0.1, segments=(), no_speech_prob=0.0
        )


class FakeSynth:
    """Produces a short buffer per chunk and records what it was asked to say."""

    def __init__(self) -> None:
        self.said: list[str] = []

    def synthesize(self, text: str) -> Any:
        self.said.append(text)
        return np.zeros(2400, dtype=np.float32)


class ScriptedLlm:
    """Answers every turn with one fixed sentence."""

    def __init__(self, answer: str = "The processor is light.") -> None:
        self.answer = answer
        self.turns = 0

    def chat_stream(self, messages: Any, **kwargs: Any) -> Iterator[ChatChunk]:
        self.turns += 1
        yield ChatChunk(content=self.answer)
        yield ChatChunk(content="", done=True, metrics={})

    def chat(self, messages: Any, **kwargs: Any) -> Any:
        from jarvis.brain.llm import ChatResponse

        return ChatResponse(content="")

    def close(self) -> None:
        return None


class ImmediateEndpointer:
    """Closes an utterance as soon as it is asked, once per reset."""

    def __init__(self, sample_rate: int = 16_000) -> None:
        self._fired = False
        self.sample_rate = sample_rate

    def reset(self) -> None:
        self._fired = False

    def process(self, frame: Any) -> Any:
        if self._fired:
            return None
        self._fired = True
        from jarvis.audio.vad import Endpoint

        return Endpoint(
            audio=np.zeros(self.sample_rate, dtype=np.float32),
            duration_s=1.0,
            decision_ms=50.0,
            reason="trailing_silence",
        )


class NeverBargesIn:
    def reset(self) -> None:
        return None

    def process(self, frame: Any) -> bool:
        return False


@pytest.fixture
def config(tmp_path: Path) -> JarvisConfig:
    return load_config(
        tmp_path / "absent.yaml",
        paths={"log_dir": str(tmp_path / "logs"), "data_dir": str(tmp_path / "data")},
        memory={"db_path": str(tmp_path / "memory.db")},
        gate={"audit_log": str(tmp_path / "gate.jsonl")},
        ui={"enabled": False},
        wake={"enabled": False},
        orchestrator={"always_listening": True, "idle_timeout_s": 2.0},
        tools={"reminders_db_path": str(tmp_path / "reminders.db")},
    )


def _wire(assistant: Assistant, config: JarvisConfig, utterances: list[str]) -> dict[str, Any]:
    """Build the assistant with every engine replaced by a fake."""
    from jarvis.audio.player import NullSink, StreamingPlayer
    from jarvis.audio.ring import AudioCapture
    from jarvis.brain.memory import ConversationMemory
    from jarvis.brain.orchestrator import Orchestrator

    streams: list[FakeStream] = []

    def factory(spec: StreamSpec) -> FakeStream:
        stream = FakeStream(spec)
        streams.append(stream)
        return stream

    llm = ScriptedLlm()
    synth = FakeSynth()
    transcriber = FakeTranscriber(utterances)
    sink = NullSink()

    assistant._capture = AudioCapture(config, assistant.bus, stream_factory=factory)
    assistant.shutdown.register("capture", assistant._capture.stop)
    assistant._endpointer = ImmediateEndpointer(config.audio.sample_rate)
    assistant._barge = NeverBargesIn()
    assistant._transcriber = transcriber
    assistant._synth = synth
    assistant._player = StreamingPlayer(config, sink=sink)
    assistant.shutdown.register("player", assistant._player.close)
    assistant._orchestrator = Orchestrator(
        config,
        assistant.bus,
        llm=llm,  # type: ignore[arg-type]
        memory=ConversationMemory(config, db_path=Path(config.memory.db_path)),
    )
    assistant._orchestrator.set_listener(assistant._listen_for_confirmation)
    assistant.shutdown.register("orchestrator", assistant._orchestrator.close)

    return {"llm": llm, "synth": synth, "transcriber": transcriber, "streams": streams}


class TestFullLoop:
    def test_a_spoken_question_produces_a_spoken_answer(self, config: JarvisConfig) -> None:
        """The whole path: capture, endpoint, transcribe, think, speak."""
        assistant = Assistant(config, headless=True)
        parts = _wire(assistant, config, ["how busy is the cpu"])

        answered = threading.Event()
        assistant.bus.subscribe(lambda _e: answered.set(), [EventType.RESPONSE])

        try:
            assistant.start()
            assert answered.wait(15), "no response was produced"
            assert parts["llm"].turns >= 1
            assert parts["synth"].said, "nothing was handed to the synthesiser"
            assert "processor" in parts["synth"].said[0]
        finally:
            assistant.stop()

    def test_the_transcript_reaches_the_bus(self, config: JarvisConfig) -> None:
        assistant = Assistant(config, headless=True)
        _wire(assistant, config, ["what is the time"])

        seen: list[str] = []
        assistant.bus.subscribe(
            lambda event: seen.append(str(event.payload.get("text", ""))),
            [EventType.TRANSCRIPT],
        )
        try:
            assistant.start()
            deadline = time.monotonic() + 15
            while not seen and time.monotonic() < deadline:
                time.sleep(0.05)
            assert seen == ["what is the time"]
        finally:
            assistant.stop()

    def test_states_progress_through_the_turn(self, config: JarvisConfig) -> None:
        assistant = Assistant(config, headless=True)
        _wire(assistant, config, ["how busy is the cpu"])

        states: list[str] = []
        assistant.bus.subscribe(
            lambda event: states.append(str(event.payload.get("state", ""))),
            [EventType.STATE_CHANGED],
        )
        try:
            assistant.start()
            deadline = time.monotonic() + 15
            while AssistantState.SPEAKING.value not in states and time.monotonic() < deadline:
                time.sleep(0.05)
            assert AssistantState.LISTENING.value in states
            assert AssistantState.THINKING.value in states
            assert AssistantState.SPEAKING.value in states
        finally:
            assistant.stop()

    def test_capture_actually_runs(self, config: JarvisConfig) -> None:
        assistant = Assistant(config, headless=True)
        parts = _wire(assistant, config, [""])
        try:
            assistant.start()
            time.sleep(0.5)
            assert parts["streams"], "no capture stream was opened"
            assert parts["streams"][0].blocks_delivered > 0
        finally:
            assistant.stop()

    def test_metrics_are_published(self, config: JarvisConfig) -> None:
        assistant = Assistant(config, headless=True)
        _wire(assistant, config, [""])

        seen: list[dict[str, Any]] = []
        assistant.bus.subscribe(lambda event: seen.append(event.payload), [EventType.METRICS])
        try:
            assistant.start()
            deadline = time.monotonic() + 10
            while not seen and time.monotonic() < deadline:
                time.sleep(0.05)
            assert seen, "the metrics loop never published"
            assert "cpu_percent" in seen[0]
        finally:
            assistant.stop()


class TestLifecycle:
    def test_shutdown_stops_every_thread(self, config: JarvisConfig) -> None:
        assistant = Assistant(config, headless=True)
        _wire(assistant, config, [""])
        assistant.start()
        time.sleep(0.3)

        before = {t.name for t in threading.enumerate() if t.name.startswith("jarvis-")}
        assert before, "no jarvis threads were running"

        assistant.stop()
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            alive = {t.name for t in threading.enumerate() if t.name.startswith("jarvis-")}
            if not alive:
                break
            time.sleep(0.1)
        alive = {t.name for t in threading.enumerate() if t.name.startswith("jarvis-")}
        assert alive == set(), f"threads still running after stop: {alive}"

    def test_a_signal_shutdown_stops_the_loops_first(self, config: JarvisConfig) -> None:
        """Shutdown steps run in reverse, so the flag-setter must be last
        registered. Otherwise a SIGINT tears down capture and the player
        underneath a turn worker that is still using them."""
        assistant = Assistant(config, headless=True)
        _wire(assistant, config, [""])
        assistant.start()
        try:
            # Exactly what the signal handler does: the coordinator only.
            assistant.shutdown.shutdown()
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                alive = {t.name for t in threading.enumerate() if t.name.startswith("jarvis-")}
                if not alive:
                    break
                time.sleep(0.1)
            alive = {t.name for t in threading.enumerate() if t.name.startswith("jarvis-")}
            assert alive == set(), f"a bare coordinator shutdown left {alive} running"
        finally:
            assistant.stop()

    def test_double_stop_is_safe(self, config: JarvisConfig) -> None:
        assistant = Assistant(config, headless=True)
        _wire(assistant, config, [""])
        assistant.start()
        assistant.stop()
        assistant.stop()

    def test_stop_without_start_is_safe(self, config: JarvisConfig) -> None:
        Assistant(config, headless=True).stop()

    def test_health_is_reported(self, config: JarvisConfig) -> None:
        assistant = Assistant(config, headless=True)
        _wire(assistant, config, [""])
        try:
            assistant.start()
            time.sleep(0.3)
            report = assistant.supervisor.health()
            assert report.healthy is True
            assert "turn-loop" in report.workers
            assert "metrics" in report.workers
        finally:
            assistant.stop()

    def test_a_transcriber_failure_does_not_kill_the_loop(self, config: JarvisConfig) -> None:
        """§5: a component failure must not take the assistant down."""
        assistant = Assistant(config, headless=True)
        _wire(assistant, config, [""])

        from jarvis.util.errors import SttError

        class Broken:
            def transcribe(self, audio: Any, sample_rate: int = 16_000) -> Transcript:
                raise SttError("the model is gone", speakable="I did not catch that.")

        assistant._transcriber = Broken()
        try:
            assistant.start()
            time.sleep(1.0)
            assert assistant.supervisor.health().healthy is True
        finally:
            assistant.stop()

    def test_a_synthesis_failure_does_not_kill_the_loop(self, config: JarvisConfig) -> None:
        assistant = Assistant(config, headless=True)
        _wire(assistant, config, ["say something"])

        from jarvis.util.errors import TtsError

        class Broken:
            def synthesize(self, text: str) -> Any:
                raise TtsError("no voice", speakable="I could not speak that.")

        assistant._synth = Broken()
        try:
            assistant.start()
            time.sleep(1.5)
            assert assistant.supervisor.health().healthy is True
        finally:
            assistant.stop()
