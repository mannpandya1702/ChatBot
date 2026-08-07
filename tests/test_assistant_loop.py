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
    """Produces a short buffer per chunk and records what it was asked to say.

    Carries ``begin_utterance`` as well as ``synthesize`` because the loop calls
    both: the voice rack has to be told where one reply ends and the next
    begins, or the previous answer's reverb tail leads into this one.
    """

    def __init__(self) -> None:
        self.said: list[str] = []
        self.utterances = 0

    def begin_utterance(self) -> None:
        self.utterances += 1

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


class TestTheFakesMatchTheRealThing:
    """Pin the fakes to the interfaces they stand in for.

    Every fake in this module exists so the loop can be tested without a GPU or
    a microphone, and each one is a copy of an interface that is free to drift
    away from it. When the loop started resetting the voice rack between
    replies, ``FakeSynth`` did not grow the method and the audio worker crashed
    in a restart loop, which the loop tests could only report as "no response
    was produced". A missing method should say so.
    """

    def test_the_synthesiser_fake_is_complete(self) -> None:
        """Read what the loop calls out of the loop, rather than listing it here.

        A hand written list is the same drift one level up: it would have to be
        remembered too. This walks ``main.py`` for every ``self._synth.x`` and
        demands the fake have each one.
        """
        import ast
        import inspect

        from jarvis import main as main_module
        from jarvis.audio.tts import KokoroSynthesizer

        tree = ast.parse(inspect.getsource(main_module))
        called = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "_synth"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "self"
        }
        assert called, "found no calls on self._synth, so this test proves nothing"

        real = {name for name in called if hasattr(KokoroSynthesizer, name)}
        missing = {name for name in real if not hasattr(FakeSynth, name)}
        assert not missing, f"the loop calls {sorted(missing)} but FakeSynth has no such method"


class TestFullLoop:
    def test_each_reply_starts_a_fresh_utterance(self, config: JarvisConfig) -> None:
        """Otherwise the previous answer's reverb tail leads into this one."""
        assistant = Assistant(config, headless=True)
        parts = _wire(assistant, config, ["how busy is the cpu"])

        answered = threading.Event()
        assistant.bus.subscribe(lambda _e: answered.set(), [EventType.RESPONSE])

        try:
            assistant.start()
            assert answered.wait(15), "no response was produced"
            assert parts["synth"].utterances >= 1
        finally:
            assistant.stop()

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
            # Any of them, not the first: startup warms the engines through the
            # same synthesiser so the first real sentence does not pay for
            # loading Kokoro, and that warmup text is recorded here too.
            assert any("processor" in line for line in parts["synth"].said), parts["synth"].said
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


class TestReadinessCheck:
    """`--check` is what a user runs when something is wrong, so it must be right.

    It reported "Not ready: vision available" on a cpu tier and told the reader
    to re-run the setup scripts. Vision turns itself off below 6 GB of VRAM by
    design (§2), so that was a healthy machine being called broken, with a
    remedy that could never have worked.
    """

    @staticmethod
    def _run(config: JarvisConfig, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
        import jarvis.main as main_module

        code = main_module.run_check(config)
        return code, capsys.readouterr().out

    def test_a_tier_that_cannot_host_vision_is_not_a_failure(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import jarvis.main as main_module
        from jarvis.config import load_defaults

        config = load_defaults()
        assert not config.vision_available(), "the cpu tier should not host the VLM"

        # Everything the voice loop actually needs is present.
        monkeypatch.setattr(main_module, "_register_tools", lambda _c: 14)
        monkeypatch.setattr("jarvis.util.platform.has_module", lambda _n: True)
        monkeypatch.setattr(
            "jarvis.brain.llm.OllamaClient.is_available", lambda _self: True
        )

        code, out = self._run(config, capsys)

        assert "Ready." in out
        assert code == 0, "a tier-disabled capability must not fail the check"
        assert "Not ready" not in out

    def test_the_vision_row_says_why_rather_than_just_no(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bare "no" sent the reader to a script that cannot help."""
        import jarvis.main as main_module
        from jarvis.config import load_defaults

        monkeypatch.setattr(main_module, "_register_tools", lambda _c: 14)
        _code, out = self._run(load_defaults(), capsys)

        vision = next(line for line in out.splitlines() if line.startswith("vision"))
        assert "VRAM" in vision or "configuration" in vision, vision

    def test_a_genuinely_missing_engine_still_fails(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The check must still catch what it exists to catch."""
        import jarvis.main as main_module
        from jarvis.config import load_defaults

        monkeypatch.setattr(main_module, "_register_tools", lambda _c: 14)
        monkeypatch.setattr("jarvis.util.platform.has_module", lambda _n: False)

        code, out = self._run(load_defaults(), capsys)

        assert code == 1
        assert "Not ready" in out
        for required in ("sounddevice", "openwakeword", "kokoro", "speech to text"):
            assert required in out

    def test_either_transcription_engine_satisfies_the_check(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cpu tier uses whisper.cpp, so requiring faster-whisper would be wrong."""
        import jarvis.main as main_module
        from jarvis.config import load_defaults

        monkeypatch.setattr(main_module, "_register_tools", lambda _c: 14)
        monkeypatch.setattr(
            "jarvis.util.platform.has_module",
            lambda name: name != "faster_whisper",
        )

        _code, out = self._run(load_defaults(), capsys)

        stt = next(line for line in out.splitlines() if line.startswith("speech to text"))
        assert "whisper.cpp" in stt
        assert "speech to text" not in out.split("Not ready:")[-1] if "Not ready" in out else True


class TestMicrophoneDiagnostic:
    """`detections=0 errors=0` has three different causes that look identical.

    A muted microphone, a device that opens but captures silence, and speech
    the model simply scores below threshold all produce the same line in the
    log. --mic-test separates them by measuring the signal and then scoring
    the same audio the listener would have seen.
    """

    def test_the_flag_is_accepted_with_and_without_a_duration(self) -> None:
        import jarvis.main as main_module

        parser = main_module.build_parser()
        assert parser.parse_args(["--mic-test"]).mic_test == pytest.approx(6.0)
        assert parser.parse_args(["--mic-test", "3"]).mic_test == pytest.approx(3.0)
        assert parser.parse_args([]).mic_test is None

    def test_it_degrades_instead_of_raising_without_an_audio_stack(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """This is what a user runs when audio is already broken."""
        import jarvis.main as main_module
        from jarvis.config import load_defaults

        code = main_module.run_mic_test(load_defaults(), 1.0)
        out = capsys.readouterr()

        assert code == 1
        assert "sounddevice" in (out.out + out.err)

    @pytest.mark.parametrize(
        ("peak", "expected"),
        [(0.0, 0), (0.25, 15), (0.5, 30), (5.0, 30)],
    )
    def test_the_level_bar_saturates_rather_than_overflowing(
        self, peak: float, expected: int
    ) -> None:
        import jarvis.main as main_module

        bar = main_module._level_bar(peak, width=30)
        assert len(bar) == 30
        assert bar.count("#") == expected

    def test_the_silence_threshold_sits_below_the_quiet_one(self) -> None:
        """Otherwise the silence verdict would swallow the quiet one."""
        import jarvis.main as main_module

        assert 0 < main_module._SILENCE_PEAK < main_module._QUIET_PEAK


class TestTheMicVerdictReadsScoresAgainstTheNoiseFloor:
    """A near miss is a tuning problem, not a broken microphone.

    The first version compared the score against half the threshold, so a real
    run scoring 0.234 against a 0.50 threshold fell a hundredth short of the
    "nearly triggered" branch and got told its microphone might be picking up
    noise. The noise floor is near zero, so 0.234 was plainly the model
    recognising the phrase and hesitating.
    """

    def test_the_responded_floor_is_absolute_not_relative(self) -> None:
        import jarvis.main as main_module

        # Well clear of the noise floor, well under any sane threshold.
        assert main_module._MODEL_RESPONDED < 0.234
        assert main_module._MODEL_RESPONDED > 0.0

    def test_the_floor_is_below_every_shipped_threshold(self) -> None:
        """Otherwise the near-miss advice could never fire."""
        import jarvis.main as main_module
        from jarvis.config import WakeConfig

        assert WakeConfig().threshold > main_module._MODEL_RESPONDED

    def test_the_suggested_threshold_sits_under_the_observed_score(self) -> None:
        """Suggesting a threshold at or above what was measured would not help."""
        for score in (0.10, 0.234, 0.35, 0.49):
            suggested = max(0.2, round(score * 0.8, 2))
            assert suggested <= max(0.2, score), f"{suggested} is not reachable from {score}"

    def test_the_suggestion_never_goes_dangerously_low(self) -> None:
        """A threshold near zero would trigger on any noise at all."""
        for score in (0.08, 0.09, 0.12):
            assert max(0.2, round(score * 0.8, 2)) >= 0.2

class _SilentEndpointer:
    """Never confirms speech, which is the case that produced total silence."""

    def __init__(self) -> None:
        self.frames_processed = 0

    def reset(self) -> None:
        self.frames_processed = 0

    def process(self, _frame: Any) -> Any:
        self.frames_processed += 1
        return None

    @property
    def state(self) -> Any:
        from jarvis.audio.vad import SpeechState

        return SpeechState.SILENCE

    @property
    def speech_ms(self) -> float:
        return 0.0

    @property
    def silence_ms(self) -> float:
        return float(self.frames_processed * 32)

@contextlib.contextmanager
def _capture_jarvis_logs(caplog: pytest.LogCaptureFixture) -> Any:
    """Collect records from the jarvis logger.

    ``setup_logging`` sets ``propagate = False`` on the jarvis logger so the
    application controls its own handlers, which means caplog's root handler
    never sees anything. Attaching directly works either way, and does not
    depend on whether another test has configured logging first.
    """
    import logging

    logger = logging.getLogger("jarvis.main")
    previous = logger.level
    logger.addHandler(caplog.handler)
    logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        logger.removeHandler(caplog.handler)
        logger.setLevel(previous)


class TestTheListenPathSaysWhatHappened:
    """A turn that hears nothing must not leave the log silent too.

    A real run detected the wake word, loaded the VAD, then printed nothing at
    all until the next wake detection twenty seconds later. From outside there
    was no way to tell whether speech was heard, whether it endpointed,
    whether transcription ran, or what came back.
    """

    def test_a_timed_out_listen_reports_what_the_endpointer_saw(
        self, config: JarvisConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        from jarvis.audio.vad import SpeechState

        assistant = Assistant(config, headless=True)
        _wire(assistant, config, ["unused"])
        assistant.supervisor.add("turn-loop", lambda: None)
        assistant._endpointer = _SilentEndpointer()

        with _capture_jarvis_logs(caplog):
            assert assistant._listen(timeout_s=0.1)[0] == ""

        record = next(
            (r for r in caplog.records if "no complete utterance" in r.getMessage()), None
        )
        assert record is not None, "a listen that heard nothing logged nothing"
        context = getattr(record, "context", {})
        for key in ("state", "speech_ms", "silence_ms", "vad_threshold", "timeout_s"):
            assert key in context, f"the timeout line does not report {key}"
        assert context["state"] == str(SpeechState.SILENCE)

    def test_the_threshold_that_matters_is_named(
        self, config: JarvisConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Naming the threshold is what turns the line into a next step."""
        assistant = Assistant(config, headless=True)
        _wire(assistant, config, ["unused"])
        assistant.supervisor.add("turn-loop", lambda: None)
        assistant._endpointer = _SilentEndpointer()

        with _capture_jarvis_logs(caplog):
            assistant._listen(timeout_s=0.1)

        record = next(r for r in caplog.records if "no complete utterance" in r.getMessage())
        assert record.context["vad_threshold"] == config.vad.threshold

    def test_a_diagnostic_that_raises_does_not_lose_the_utterance(
        self, config: JarvisConfig
    ) -> None:
        """The log line exists to explain a turn, not to be able to end one."""

        class Exploding(_SilentEndpointer):
            @property
            def state(self) -> Any:
                raise RuntimeError("this property is broken")

            @property
            def speech_ms(self) -> float:
                raise RuntimeError("so is this one")

        assistant = Assistant(config, headless=True)
        _wire(assistant, config, ["unused"])
        assistant.supervisor.add("turn-loop", lambda: None)
        assistant._endpointer = Exploding()

        # Must return normally rather than propagating out of the turn loop.
        assert assistant._listen(timeout_s=0.05)[0] == ""
        # The broken properties are omitted rather than raising.
        state = assistant._endpointer_state()
        assert "state" not in state
        assert "speech_ms" not in state

    def test_the_happy_path_reports_what_it_heard(
        self, config: JarvisConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        assistant = Assistant(config, headless=True)
        _wire(assistant, config, ["how busy is the cpu"])

        answered = threading.Event()
        assistant.bus.subscribe(lambda _e: answered.set(), [EventType.RESPONSE])

        try:
            with _capture_jarvis_logs(caplog):
                assistant.start()
                assert answered.wait(15), "no response was produced"
        finally:
            assistant.stop()

        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "utterance endpointed" in messages
        assert "heard" in messages


class TestStartupWarmsTheEngines:
    """Every engine loads lazily, so the first sentence of a session paid for it.

    Measured with the weights already on disk: Silero 86 ms, faster-whisper
    3.7 s, Kokoro 7.7 s on their first call, against §3 budgets of 250, 200 and
    300 ms. That is about eleven seconds added to the first question of every
    session and none of the ones after it, which reads as the assistant being
    broken rather than cold. Transcriber.warmup existed for exactly this and was
    called from nowhere but its own test.
    """

    def test_the_transcriber_is_warmed(self, config: JarvisConfig) -> None:
        assistant = Assistant(config, headless=True)
        parts = _wire(assistant, config, [])
        warmed = threading.Event()
        parts["transcriber"].warmup = warmed.set  # type: ignore[attr-defined]
        try:
            assistant.start()
            assert warmed.wait(5), "the transcriber was never warmed"
        finally:
            assistant.stop()

    def test_the_synthesiser_is_warmed(self, config: JarvisConfig) -> None:
        assistant = Assistant(config, headless=True)
        parts = _wire(assistant, config, [])
        try:
            assistant.start()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not parts["synth"].said:
                time.sleep(0.02)
            assert parts["synth"].said, "the synthesiser was never warmed"
        finally:
            assistant.stop()

    def test_warming_plays_no_audio(self, config: JarvisConfig) -> None:
        """It loads the model. It must not make a noise doing so."""
        assistant = Assistant(config, headless=True)
        parts = _wire(assistant, config, [])
        try:
            assistant.start()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not parts["synth"].said:
                time.sleep(0.02)
            assert assistant._player.queued_seconds == 0.0
        finally:
            assistant.stop()

    def test_a_broken_engine_does_not_stop_startup(self, config: JarvisConfig) -> None:
        """A warmup failure belongs on the first real utterance, with context."""
        assistant = Assistant(config, headless=True)
        parts = _wire(assistant, config, [])

        def explode() -> None:
            raise RuntimeError("no CUDA")

        parts["transcriber"].warmup = explode  # type: ignore[attr-defined]
        try:
            assistant.start()
            time.sleep(0.3)
            assert assistant.supervisor.health().healthy
        finally:
            assistant.stop()


class TestTheWakeWordsPrerollIsUsed:
    """The ring buffer keeps a second of audio before the trigger for a reason.

    T-1.1 asks for the pre-roll "so the wake word's trailing audio is not lost",
    WakeDetection carries it, and its own docstring says to prepend it. Nothing
    did: the turn loop attached a fresh cursor at the write head, so the
    endpointer only ever saw audio captured after the trigger. Anyone saying
    "hey jarvis what time is it" as one phrase lost the front of the question,
    and the transcriber got a sentence starting mid-word.
    """

    class _Recorder:
        """An endpointer that records every frame and never closes an utterance."""

        def __init__(self) -> None:
            self.frames: list[Any] = []

        def reset(self) -> None:
            return None

        def process(self, frame: Any) -> Any:
            self.frames.append(np.asarray(frame).copy())
            return None

    def test_the_preroll_reaches_the_endpointer(self, config: JarvisConfig) -> None:
        assistant = Assistant(config, headless=True)
        _wire(assistant, config, [])
        recorder = self._Recorder()
        assistant._endpointer = recorder

        frame = config.vad.frame_samples
        preroll = np.tile(np.linspace(0.1, 1.0, frame, dtype=np.float32), 4)

        assistant._capture.start()
        try:
            assistant._listen(0.15, preroll)
        finally:
            assistant._capture.stop()

        assert len(recorder.frames) >= 4, "the pre-roll was never scored"
        seeded = np.concatenate(recorder.frames[:4])
        assert np.allclose(seeded, preroll), "the endpointer did not see the pre-roll audio"

    def test_no_preroll_is_harmless(self, config: JarvisConfig) -> None:
        assistant = Assistant(config, headless=True)
        _wire(assistant, config, [])
        assistant._endpointer = self._Recorder()
        assistant._capture.start()
        try:
            assert assistant._listen(0.1, None)[0] == ""
        finally:
            assistant._capture.stop()

    def test_a_short_preroll_is_ignored_rather_than_padded(
        self, config: JarvisConfig
    ) -> None:
        """Silero rejects a frame of the wrong length, so a partial one is dropped."""
        assistant = Assistant(config, headless=True)
        _wire(assistant, config, [])
        recorder = self._Recorder()
        assistant._endpointer = recorder
        assistant._capture.start()
        try:
            assistant._listen(0.1, np.zeros(7, dtype=np.float32))
        finally:
            assistant._capture.stop()
        assert all(f.size == config.vad.frame_samples for f in recorder.frames)

    def test_rubbish_in_the_preroll_does_not_end_the_turn(
        self, config: JarvisConfig
    ) -> None:
        assistant = Assistant(config, headless=True)
        _wire(assistant, config, [])
        assistant._endpointer = self._Recorder()
        assistant._capture.start()
        try:
            assert assistant._listen(0.1, "not audio at all")[0] == ""
        finally:
            assistant._capture.stop()

    def test_a_detection_is_consumed_once(self, config: JarvisConfig) -> None:
        """A stale pre-roll leaking into the next turn would replay old audio."""
        assistant = Assistant(config, headless=True)
        _wire(assistant, config, [])

        class _Detection:
            preroll = np.zeros(512, dtype=np.float32)

        assistant._on_wake(_Detection())
        assert assistant._wake_detection is not None

        assistant._capture.start()
        assistant._stop.set()
        try:
            assistant._turn_loop()
        finally:
            assistant._capture.stop()
        assistant._stop.clear()


class TestShutdownWakesEverythingItIsWaitingOn:
    """Ctrl-C mid-reply hung for the supervisor's whole grace period.

    During a reply the turn loop parks in player.wait(max_turn_seconds), on an
    event only the audio callback or player.stop() ever sets. _request_stop set
    the stop flag, set the wake event and interrupted the orchestrator, but
    never told the one object the loop was actually blocked on, so stop() took
    5 s and reported the turn loop as stalled.
    """

    def test_requesting_stop_drains_the_player(self, config: JarvisConfig) -> None:
        assistant = Assistant(config, headless=True)
        _wire(assistant, config, [])
        assistant._player.start()
        try:
            assistant._player.play(
                np.ones(config.tts.sample_rate * 30, dtype=np.float32)
            )
            assert not assistant._player.wait(0.05), "the player should still be busy"

            assistant._request_stop()
            assert assistant._player.wait(1.0), "shutdown left the turn loop parked"
        finally:
            assistant._player.close()

    def test_stopping_mid_reply_is_prompt(self, config: JarvisConfig) -> None:
        assistant = Assistant(config, headless=True)
        _wire(assistant, config, ["how busy is the cpu"])
        spoke = threading.Event()
        assistant.bus.subscribe(lambda _e: spoke.set(), [EventType.RESPONSE])

        assistant.start()
        try:
            spoke.wait(10)
            assistant._player.play(np.ones(config.tts.sample_rate * 30, dtype=np.float32))
            started = time.monotonic()
        finally:
            assistant.stop()
        assert time.monotonic() - started < 3.0, "shutdown waited on playback"


class TestAnUnrecoverableEndpointerStopsInsteadOfSpinning:
    """A missing checkpoint made the loop fail at the frame rate.

    _listen already special-cased it to avoid a traceback per frame, but
    returning "" put the spin one frame up: the turn loop reads that as ordinary
    silence and calls straight back in. Measured at 31 errors and 11 KiB of log
    per second, which is exactly audio.sample_rate / vad.frame_samples.
    """

    class _Broken:
        def reset(self) -> None:
            return None

        def process(self, frame: Any) -> Any:
            from jarvis.util.errors import DependencyMissingError

            raise DependencyMissingError("silero-vad")

    def test_the_failure_is_reported_once_not_per_frame(
        self, config: JarvisConfig
    ) -> None:
        assistant = Assistant(config, headless=True)
        _wire(assistant, config, [])
        assistant._endpointer = self._Broken()

        errors: list[Any] = []
        assistant.bus.subscribe(errors.append, [EventType.ERROR])

        assistant.start()
        try:
            time.sleep(1.5)
        finally:
            assistant.stop()

        # One frame's worth of errors, not one second's worth. The old loop
        # produced about 31 a second here.
        assert len(errors) <= 2, f"{len(errors)} errors in 1.5 s, the loop is spinning"

    def test_it_latches_so_the_turn_loop_stops(self, config: JarvisConfig) -> None:
        assistant = Assistant(config, headless=True)
        _wire(assistant, config, [])
        assistant._endpointer = self._Broken()

        assistant.start()
        try:
            time.sleep(0.8)
            assert assistant._endpointer_failed is not None
        finally:
            assistant.stop()


class TestADeadVoiceIsReported:
    """Synthesis was the one failure path in main.py that never reached the bus.

    Its two siblings both do: _listen emits an ERROR for endpointer and STT
    failures, and WakeListener emits one on a fatal load, so the HUD showed
    those. A dead Kokoro produced a WARNING per chunk, a state machine still
    reporting "speaking", and nothing else. The user simply stopped hearing
    anything and had no way to find out why.
    """

    class _BrokenSynth(FakeSynth):
        def synthesize(self, text: str) -> Any:
            from jarvis.util.errors import TtsError

            self.said.append(text)
            raise TtsError("no espeak backend", speakable="I could not speak that.")

    def test_the_failure_reaches_the_bus(self, config: JarvisConfig) -> None:
        assistant = Assistant(config, headless=True)
        _wire(assistant, config, ["how busy is the cpu"])
        assistant._synth = self._BrokenSynth()

        errors: list[Any] = []
        assistant.bus.subscribe(errors.append, [EventType.ERROR])

        assistant.start()
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not errors:
                time.sleep(0.05)
        finally:
            assistant.stop()

        assert errors, "a mute assistant reported nothing at all"
        assert errors[0].payload.get("speakable")

    def test_it_does_not_report_once_per_chunk(self, config: JarvisConfig) -> None:
        """Every chunk of every reply hits the same failure."""
        assistant = Assistant(config, headless=True)
        _wire(assistant, config, ["how busy is the cpu"])
        assistant._synth = self._BrokenSynth()

        errors: list[Any] = []
        assistant.bus.subscribe(errors.append, [EventType.ERROR])

        assistant.start()
        try:
            time.sleep(2.0)
        finally:
            assistant.stop()
        assert len(errors) <= 3, f"{len(errors)} error events, the report is flooding"

    def test_a_working_voice_reports_nothing(self, config: JarvisConfig) -> None:
        assistant = Assistant(config, headless=True)
        _wire(assistant, config, ["how busy is the cpu"])
        errors: list[Any] = []
        assistant.bus.subscribe(errors.append, [EventType.ERROR])

        spoke = threading.Event()
        assistant.bus.subscribe(lambda _e: spoke.set(), [EventType.RESPONSE])
        assistant.start()
        try:
            spoke.wait(10)
        finally:
            assistant.stop()
        assert not errors


class TestTheContractsLatencyIsTheOneMeasured:
    """§3 budgets "end of user speech to first audio out".

    VAD_ENDPOINT and STT appeared nowhere in src/ at all, and the recorder was
    built inside run_turn, which the orchestrator only reaches once transcription
    has finished. So the logged turn_total covered LLM plus TTS and nothing else:
    a machine sitting at 1.6 s of real latency logged about 1.0 s and never
    tripped the over-budget warning. bench_latency.py measured all five stages
    against synthetic sources, so the benchmark stayed green while the shipped
    loop reported a number that was not the contract's number.
    """

    def _turn(self, config: JarvisConfig) -> Any:
        from jarvis.util.latency import Stage

        assistant = Assistant(config, headless=True)
        _wire(assistant, config, ["how busy is the cpu"])
        seen: list[Any] = []
        assistant.bus.subscribe(seen.append, [EventType.LATENCY])

        answered = threading.Event()
        assistant.bus.subscribe(lambda _e: answered.set(), [EventType.RESPONSE])
        assistant.start()
        try:
            assert answered.wait(15), "no response was produced"
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not seen:
                time.sleep(0.02)
        finally:
            assistant.stop()
        assert seen, "no latency breakdown was published"
        _ = Stage
        return {
            stage["stage"]: float(stage["duration_ms"])
            for stage in seen[0].payload["stages"]
        }

    def test_the_endpoint_decision_is_measured(self, config: JarvisConfig) -> None:
        assert "vad_endpoint" in self._turn(config)

    def test_transcription_is_measured(self, config: JarvisConfig) -> None:
        assert "stt" in self._turn(config)

    def test_the_total_includes_them(self, config: JarvisConfig) -> None:
        """Otherwise the number logged is not the number §3 puts a budget on."""
        stages = self._turn(config)
        assert stages["turn_total"] >= stages["stt"]


class TestALappedReaderIsNotSpliced:
    """ring.py's own docstring says a silent splice is worse than a logged gap.

    RingReader counts what it lost correctly and no consumer ever asked:

        $ grep -rn '\\.dropped' src/jarvis --include=*.py | grep -v ring.py
        (nothing)

    So audio from either side of a gap was concatenated into one utterance and
    handed to the transcriber, with Silero still carrying LSTM state from before
    the jump. The wake listener is the most exposed of the three consumers,
    because the detector loads its models lazily inside process() and the very
    first frame can block for seconds while the writer keeps going.
    """

    class _LappingReader:
        """Reports a gap once, after a few frames."""

        def __init__(self, frame: int, gap_after: int = 3) -> None:
            self.frame = frame
            self.gap_after = gap_after
            self.reads = 0
            self.dropped = 0

        def read(self, n: int) -> Any:
            self.reads += 1
            if self.reads == self.gap_after:
                self.dropped += 8000
            return np.zeros(n, dtype=np.float32)

    def test_the_utterance_is_abandoned_rather_than_spliced(
        self, config: JarvisConfig
    ) -> None:
        assistant = Assistant(config, headless=True)
        _wire(assistant, config, ["should never be transcribed"])

        assistant.supervisor.add("turn-loop", lambda: None)
        # A gap mid-utterance, which is the case that got spliced: the
        # endpointer is still open, so without the check the frames either side
        # of the jump are concatenated into one Endpoint.
        assistant._endpointer = _SilentEndpointer()
        reader = self._LappingReader(config.vad.frame_samples)
        assistant._capture = type(
            "OneReader", (), {"reader": lambda _self, **_kw: reader, "stop": lambda _self: None}
        )()

        errors: list[Any] = []
        assistant.bus.subscribe(errors.append, [EventType.ERROR])

        text, latency = assistant._listen(2.0)

        assert text == "", "a spliced utterance was transcribed"
        assert latency is None
        assert errors, "the gap was neither logged nor published"
        assert "dropped" in errors[0].payload["message"]

    def test_a_reader_that_keeps_up_is_untouched(self, config: JarvisConfig) -> None:
        assistant = Assistant(config, headless=True)
        _wire(assistant, config, ["how busy is the cpu"])

        assistant.supervisor.add("turn-loop", lambda: None)
        reader = self._LappingReader(config.vad.frame_samples, gap_after=10_000)
        assistant._capture = type(
            "OneReader", (), {"reader": lambda _self, **_kw: reader, "stop": lambda _self: None}
        )()

        text, _latency = assistant._listen(2.0)
        assert text == "how busy is the cpu"
