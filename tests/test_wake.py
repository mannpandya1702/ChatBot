"""T-1.2: openWakeWord detection, cooldown, pre-roll, and the listener thread.

Everything here runs on Linux with no microphone, no GPU, and no openWakeWord
install (§0b). The model is injected: :class:`FakeModel` returns a scripted score
per frame, so threshold, cooldown, framing, and pre-roll behaviour are all
exercised deterministically. The two checks that genuinely need a human and a
microphone are marked ``manual`` and skip themselves off the target host.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from jarvis.audio import wake as wake_module
from jarvis.audio.ring import RingBuffer
from jarvis.audio.wake import (
    WakeDetection,
    WakeListener,
    WakeWordDetector,
    float_to_int16,
)
from jarvis.config import JarvisConfig
from jarvis.state import EventBus, EventType
from jarvis.util.errors import AudioError, DependencyMissingError
from jarvis.util.platform import has_module, is_windows

MODEL_NAME = "hey_jarvis_v0.1"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeModel:
    """Stand-in for ``openwakeword.model.Model`` with a scripted score per frame.

    A script entry may be a float, a dict of per-model scores, or an exception
    instance, which is raised instead of returning. Once the script runs out the
    model reports silence forever, so a listener loop can keep spinning
    harmlessly after the interesting part of a test. With ``raw`` the script
    value is returned untouched, which is how the odd return shapes are covered.
    """

    def __init__(
        self,
        script: list[Any] | None = None,
        *,
        name: str = MODEL_NAME,
        raw: bool = False,
    ) -> None:
        self.script: list[Any] = list(script or [])
        self.name = name
        self.raw = raw
        self.frames: list[np.ndarray[Any, Any]] = []
        self.resets = 0

    def predict(self, frame: np.ndarray[Any, Any]) -> Any:
        self.frames.append(np.asarray(frame))
        value: Any = self.script.pop(0) if self.script else 0.0
        if isinstance(value, BaseException):
            raise value
        if self.raw or isinstance(value, dict):
            return value
        return {self.name: float(value)}

    def reset(self) -> None:
        self.resets += 1


class FakeClock:
    """Monotonic clock the test drives by hand, so cooldowns need no sleeping."""

    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class ScriptedReader:
    """Frame reader that hands out prepared blocks, then starves.

    ``failures`` reads raise before any block is returned, which is how the
    "reader blows up" path is exercised.
    """

    def __init__(self, blocks: list[np.ndarray[Any, Any]], *, failures: int = 0) -> None:
        self.blocks = list(blocks)
        self.failures = failures
        self.reads = 0

    def read(self, n: int) -> np.ndarray[Any, Any] | None:
        self.reads += 1
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("ring buffer exploded")
        if not self.blocks:
            return None
        return self.blocks.pop(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def tune(config: JarvisConfig, **wake: Any) -> JarvisConfig:
    """Apply wake-section overrides in place and return the config."""
    for key, value in wake.items():
        setattr(config.wake, key, value)
    return config


def ramp(count: int) -> np.ndarray[Any, Any]:
    """Distinct float32 samples in -0.9..0.9, so slices compare meaningfully."""
    return np.linspace(-0.9, 0.9, count, dtype=np.float32)


def wait_until(predicate: Any, timeout: float = 3.0) -> bool:
    """Poll ``predicate`` until true or the timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


def detector_for(
    config: JarvisConfig,
    model: FakeModel,
    clock: FakeClock | None = None,
    **wake: Any,
) -> WakeWordDetector:
    """Build a detector over a fake model with wake overrides applied."""
    tune(config, **wake)
    return WakeWordDetector(config, model, clock=clock or FakeClock())


# ---------------------------------------------------------------------------
# int16 conversion
# ---------------------------------------------------------------------------


def test_float_to_int16_covers_the_range_without_wrapping() -> None:
    out = float_to_int16(np.array([0.0, 1.0, -1.0, 0.5, -0.5], dtype=np.float32))
    assert out.dtype == np.int16
    # 1.0 maps to the positive limit, not to 32768, which would wrap negative.
    assert out.tolist() == [0, 32767, -32767, 16384, -16384]


def test_float_to_int16_clips_instead_of_wrapping() -> None:
    out = float_to_int16(np.array([2.0, -2.0, 1e9, -1e9], dtype=np.float32))
    assert out.tolist() == [32767, -32768, 32767, -32768]
    assert out.dtype == np.int16


def test_float_to_int16_preserves_shape_and_passes_int16_through() -> None:
    assert float_to_int16(np.zeros((2, 3), dtype=np.float32)).shape == (2, 3)
    already = np.array([100, -200, 32767, -32768], dtype=np.int16)
    assert float_to_int16(already).tolist() == already.tolist()


def test_the_model_is_handed_int16_audio(cfg: JarvisConfig) -> None:
    model = FakeModel([0.0])
    detector = detector_for(cfg, model, frame_samples=160)
    detector.process(np.ones(160, dtype=np.float32))
    assert model.frames[0].dtype == np.int16
    assert int(model.frames[0].max()) == 32767


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------


def test_detects_when_the_score_reaches_the_threshold(cfg: JarvisConfig) -> None:
    clock = FakeClock()
    model = FakeModel([0.5])
    detector = detector_for(cfg, model, clock, frame_samples=160, threshold=0.5)

    detection = detector.process(ramp(160))

    assert isinstance(detection, WakeDetection)
    assert detection.model == MODEL_NAME
    assert detection.score == pytest.approx(0.5)
    assert detection.timestamp == clock.now


def test_no_detection_just_below_the_threshold(cfg: JarvisConfig) -> None:
    model = FakeModel([0.4999])
    detector = detector_for(cfg, model, frame_samples=160, threshold=0.5)

    assert detector.process(ramp(160)) is None
    assert detector.scores[MODEL_NAME] == pytest.approx(0.4999)
    assert detector.frames_processed == 1


def test_the_highest_scoring_model_wins(cfg: JarvisConfig) -> None:
    model = FakeModel([{"alexa": 0.6, MODEL_NAME: 0.91}])
    detector = detector_for(cfg, model, frame_samples=160, threshold=0.5)

    detection = detector.process(ramp(160))

    assert detection is not None
    assert detection.model == MODEL_NAME
    assert detector.scores == {"alexa": pytest.approx(0.6), MODEL_NAME: pytest.approx(0.91)}


def test_scores_are_a_copy(cfg: JarvisConfig) -> None:
    detector = detector_for(cfg, FakeModel([0.2]), frame_samples=160)
    detector.process(ramp(160))

    snapshot = detector.scores
    snapshot[MODEL_NAME] = 99.0

    assert detector.scores[MODEL_NAME] == pytest.approx(0.2)


def test_a_bare_number_is_accepted_as_a_score(cfg: JarvisConfig) -> None:
    tune(cfg, frame_samples=160, threshold=0.5, model="hey_jarvis")
    model = FakeModel([0.8], raw=True)  # a bare float, not a dict of scores
    detector = WakeWordDetector(cfg, model, clock=FakeClock())

    detection = detector.process(ramp(160))

    assert detection is not None
    assert detection.model == "hey_jarvis"


def test_an_unusable_score_raises_audio_error(cfg: JarvisConfig) -> None:
    model = FakeModel(["not a number"], raw=True)
    detector = detector_for(cfg, model, frame_samples=160)

    with pytest.raises(AudioError, match="expected a score"):
        detector.process(ramp(160))


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------


def test_cooldown_suppresses_then_expires(cfg: JarvisConfig) -> None:
    clock = FakeClock()
    model = FakeModel([0.9, 0.9, 0.9])
    detector = detector_for(cfg, model, clock, frame_samples=160, threshold=0.5, cooldown_s=2.0)
    frame = ramp(160)

    assert detector.process(frame) is not None
    assert detector.cooldown_remaining == pytest.approx(2.0)

    clock.advance(1.9)
    assert detector.process(frame) is None
    assert detector.cooldown_remaining == pytest.approx(0.1)

    clock.advance(0.2)
    assert detector.cooldown_remaining == 0.0
    assert detector.process(frame) is not None
    # Every frame still reached the model; suppression is a policy, not a skip.
    assert len(model.frames) == 3


def test_a_suppressed_hit_does_not_extend_the_cooldown(cfg: JarvisConfig) -> None:
    clock = FakeClock()
    model = FakeModel([0.9, 0.9, 0.9])
    detector = detector_for(cfg, model, clock, frame_samples=160, threshold=0.5, cooldown_s=2.0)
    frame = ramp(160)

    assert detector.process(frame) is not None
    clock.advance(1.0)
    assert detector.process(frame) is None  # suppressed, must not restart the timer
    clock.advance(1.0)
    assert detector.process(frame) is not None


def test_zero_cooldown_allows_back_to_back_detections(cfg: JarvisConfig) -> None:
    model = FakeModel([0.9, 0.9])
    detector = detector_for(cfg, model, frame_samples=160, threshold=0.5, cooldown_s=0.0)

    assert detector.process(ramp(160)) is not None
    assert detector.process(ramp(160)) is not None


# ---------------------------------------------------------------------------
# Frame accumulation
# ---------------------------------------------------------------------------


def test_partial_writes_accumulate_into_whole_frames(cfg: JarvisConfig) -> None:
    model = FakeModel()
    detector = detector_for(cfg, model, frame_samples=1280)
    fed: list[np.ndarray[Any, Any]] = []

    # 200 is deliberately not a divisor of 1280, so every boundary is ragged.
    for index in range(7):
        chunk = ramp(200) + float(index)
        fed.append(chunk)
        assert detector.process(chunk) is None

    assert len(model.frames) == 1, "inference must only run on whole frames"
    assert detector.pending_samples == 7 * 200 - 1280
    expected = np.concatenate(fed)[:1280]
    assert np.array_equal(model.frames[0], float_to_int16(expected))


def test_a_long_block_is_split_into_consecutive_frames(cfg: JarvisConfig) -> None:
    model = FakeModel()
    detector = detector_for(cfg, model, frame_samples=160)
    data = ramp(500)

    assert detector.process(data) is None

    assert len(model.frames) == 3
    assert detector.pending_samples == 20
    assert np.array_equal(model.frames[2], float_to_int16(data[320:480]))


def test_a_detection_stops_the_block_and_buffers_the_remainder(cfg: JarvisConfig) -> None:
    model = FakeModel([0.9, 0.9, 0.9])
    detector = detector_for(cfg, model, frame_samples=160, threshold=0.5)

    detection = detector.process(ramp(500))

    assert detection is not None
    assert len(model.frames) == 1, "scoring must stop at the trigger point"
    assert detector.pending_samples == 340


def test_an_empty_block_is_a_no_op(cfg: JarvisConfig) -> None:
    model = FakeModel([0.9])
    detector = detector_for(cfg, model, frame_samples=160)

    assert detector.process(np.zeros(0, dtype=np.float32)) is None
    assert model.frames == []


def test_two_dimensional_and_int16_input_are_accepted(cfg: JarvisConfig) -> None:
    model = FakeModel([0.9])
    detector = detector_for(cfg, model, frame_samples=160, threshold=0.5)

    block = np.full((160, 1), 8000, dtype=np.int16)
    detection = detector.process(block)

    assert detection is not None
    # 8000/32768 back out through 32767 lands within one LSB of the original.
    assert abs(int(model.frames[0][0]) - 8000) <= 1


# ---------------------------------------------------------------------------
# Pre-roll
# ---------------------------------------------------------------------------


def test_preroll_is_the_audio_immediately_before_the_trigger(cfg: JarvisConfig) -> None:
    # 160 sample frames, 320 samples of pre-roll: exactly two frames of history.
    detector = detector_for(
        cfg,
        FakeModel([0.0, 0.0, 0.0, 0.9]),
        frame_samples=160,
        preroll_s=0.02,
        threshold=0.5,
    )
    data = ramp(640)

    detection = detector.process(data)

    assert detection is not None
    assert detection.preroll_samples == 320
    assert np.array_equal(detection.preroll, data[320:640])


def test_preroll_excludes_audio_after_the_trigger_point(cfg: JarvisConfig) -> None:
    detector = detector_for(
        cfg,
        FakeModel([0.9]),
        frame_samples=160,
        preroll_s=0.01,
        threshold=0.5,
    )
    data = ramp(300)

    detection = detector.process(data)

    assert detection is not None
    # The trailing 140 samples arrived in the same call but after the trigger.
    assert np.array_equal(detection.preroll, data[:160])


def test_preroll_is_front_padded_when_the_history_is_short(cfg: JarvisConfig) -> None:
    detector = detector_for(
        cfg,
        FakeModel([0.0, 0.9]),
        frame_samples=160,
        preroll_s=0.05,  # 800 samples wanted, only 320 will exist
        threshold=0.5,
    )
    data = ramp(320)

    detection = detector.process(data)

    assert detection is not None
    assert detection.preroll_samples == 800
    assert np.all(detection.preroll[:480] == 0.0), "padding belongs at the front"
    assert np.array_equal(detection.preroll[480:], data)


def test_preroll_is_empty_when_disabled(cfg: JarvisConfig) -> None:
    detector = detector_for(
        cfg, FakeModel([0.9]), frame_samples=160, preroll_s=0.0, threshold=0.5
    )

    detection = detector.process(ramp(160))

    assert detection is not None
    assert detection.preroll_samples == 0


def test_preroll_accessor_reflects_scored_audio_only(cfg: JarvisConfig) -> None:
    detector = detector_for(cfg, FakeModel(), frame_samples=160, preroll_s=0.01)

    assert np.all(detector.preroll() == 0.0)
    detector.process(ramp(100))
    assert np.all(detector.preroll() == 0.0), "a partial frame is not scored yet"
    detector.process(np.full(60, 0.5, dtype=np.float32))
    assert detector.preroll().size == 160
    assert not np.all(detector.preroll() == 0.0)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def test_reset_clears_pending_history_scores_and_cooldown(cfg: JarvisConfig) -> None:
    clock = FakeClock()
    model = FakeModel([0.9, 0.9])
    detector = detector_for(
        cfg, model, clock, frame_samples=160, threshold=0.5, cooldown_s=5.0, preroll_s=0.02
    )
    detector.process(ramp(250))
    assert detector.pending_samples == 90
    assert detector.cooldown_remaining > 0

    detector.reset()

    assert detector.pending_samples == 0
    assert detector.scores == {}
    assert detector.frames_processed == 0
    assert detector.cooldown_remaining == 0.0
    assert np.all(detector.preroll() == 0.0)
    assert model.resets == 1
    # The cooldown is gone, so the very next frame may trigger again.
    assert detector.process(ramp(160)) is not None


def test_reset_survives_a_model_without_reset(cfg: JarvisConfig) -> None:
    class Bare:
        def predict(self, frame: np.ndarray[Any, Any]) -> dict[str, float]:
            return {MODEL_NAME: 0.0}

    detector = WakeWordDetector(cfg, Bare())
    detector.reset()  # must not raise


def test_reset_survives_a_model_whose_reset_raises(cfg: JarvisConfig) -> None:
    class Angry(FakeModel):
        def reset(self) -> None:
            raise RuntimeError("no")

    detector = WakeWordDetector(cfg, Angry())
    detector.reset()  # logged, swallowed


# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------


def fake_openwakeword(recorder: dict[str, Any], *, boom: bool = False) -> types.ModuleType:
    """A module object exposing a ``Model`` class that records its kwargs."""

    class Model:
        def __init__(self, **kwargs: Any) -> None:
            recorder.update(kwargs)
            if boom:
                raise RuntimeError("onnxruntime is broken")

        def predict(self, frame: np.ndarray[Any, Any]) -> dict[str, float]:
            return {MODEL_NAME: 0.0}

    module = types.ModuleType("openwakeword")
    module.Model = Model  # type: ignore[attr-defined]
    return module


def test_the_model_is_loaded_lazily_and_only_once(
    cfg: JarvisConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder: dict[str, Any] = {}
    calls = {"n": 0}

    def _require(name: str, **_: Any) -> types.ModuleType:
        calls["n"] += 1
        assert name == "openwakeword"
        return fake_openwakeword(recorder)

    monkeypatch.setattr(wake_module, "require_module", _require)
    detector = WakeWordDetector(tune(cfg, frame_samples=160))

    assert not detector.is_loaded, "constructing must not import anything"
    detector.process(ramp(320))

    assert detector.is_loaded
    assert calls["n"] == 1, "the model must be built once, not per frame"


def test_a_local_checkpoint_beats_the_bundled_one(
    cfg: JarvisConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = cfg.models_dir
    models.mkdir(parents=True, exist_ok=True)
    (models / "hey_jarvis_v0.1.onnx").write_bytes(b"stub")
    (models / "hey_jarvis_v0.2.onnx").write_bytes(b"stub")
    (models / "hey_jarvis_v0.3.tflite").write_bytes(b"wrong framework")
    recorder: dict[str, Any] = {}
    monkeypatch.setattr(
        wake_module, "require_module", lambda *a, **k: fake_openwakeword(recorder)
    )

    detector = WakeWordDetector(tune(cfg, frame_samples=160, inference_framework="onnx"))
    detector.process(ramp(160))

    assert recorder["wakeword_models"] == [str(models / "hey_jarvis_v0.2.onnx")]
    assert recorder["inference_framework"] == "onnx"
    assert recorder["vad_threshold"] == pytest.approx(cfg.wake.vad_threshold)


def test_without_a_local_checkpoint_the_bare_name_is_used(
    cfg: JarvisConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder: dict[str, Any] = {}
    monkeypatch.setattr(
        wake_module, "require_module", lambda *a, **k: fake_openwakeword(recorder)
    )

    detector = WakeWordDetector(tune(cfg, frame_samples=160, model="hey_jarvis"))
    detector.process(ramp(160))

    assert recorder["wakeword_models"] == ["hey_jarvis"]


def test_a_missing_dependency_is_reported_with_an_install_hint(
    cfg: JarvisConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _missing(name: str, **_: Any) -> types.ModuleType:
        raise DependencyMissingError(name, extra="audio")

    monkeypatch.setattr(wake_module, "require_module", _missing)
    detector = WakeWordDetector(tune(cfg, frame_samples=160))

    with pytest.raises(DependencyMissingError) as info:
        detector.process(ramp(160))

    assert "uv sync --extra audio" in str(info.value)


def test_a_loader_failure_becomes_a_speakable_audio_error(
    cfg: JarvisConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        wake_module, "require_module", lambda *a, **k: fake_openwakeword({}, boom=True)
    )
    detector = WakeWordDetector(tune(cfg, frame_samples=160))

    with pytest.raises(AudioError) as info:
        detector.process(ramp(160))

    assert "onnxruntime is broken" in str(info.value)
    assert info.value.speakable == "I could not load my wake word model."


def test_a_package_without_a_model_class_is_reported(
    cfg: JarvisConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = types.ModuleType("openwakeword")
    monkeypatch.setattr(wake_module, "require_module", lambda *a, **k: empty)
    detector = WakeWordDetector(tune(cfg, frame_samples=160))

    with pytest.raises((AudioError, ModuleNotFoundError, ImportError)):
        detector.process(ramp(160))


# ---------------------------------------------------------------------------
# Listener
# ---------------------------------------------------------------------------


def ring_with(samples: np.ndarray[Any, Any]) -> tuple[RingBuffer, Any]:
    """A ring buffer preloaded with ``samples`` plus a reader at the oldest."""
    buffer = RingBuffer(max(len(samples) * 2, 1024))
    reader = buffer.reader()
    buffer.write(samples)
    return buffer, reader


def test_listener_detects_and_publishes(cfg: JarvisConfig, bus: EventBus) -> None:
    tune(cfg, frame_samples=160, preroll_s=0.02, threshold=0.5)
    data = ramp(480)
    _buffer, reader = ring_with(data)
    detector = WakeWordDetector(cfg, FakeModel([0.0, 0.0, 0.93]), clock=FakeClock())
    events, _ = bus.subscribe_queue([EventType.WAKE_DETECTED])
    seen: list[WakeDetection] = []
    fired = threading.Event()

    def _on(detection: WakeDetection) -> None:
        seen.append(detection)
        fired.set()

    listener = WakeListener(cfg, reader, bus, detector, on_detected=_on)
    with listener:
        assert fired.wait(3.0), "the wake detection never reached the callback"
    assert not listener.is_running

    assert listener.detections == 1
    assert listener.errors == 0
    assert seen[0].score == pytest.approx(0.93)
    assert np.array_equal(seen[0].preroll, data[160:480])

    event = events.get(timeout=1.0)
    assert event.payload["model"] == MODEL_NAME
    assert event.payload["score"] == pytest.approx(0.93)
    assert event.payload["preroll_samples"] == 320
    assert event.payload["preroll_seconds"] == pytest.approx(0.02)
    assert np.array_equal(event.payload["preroll"], data[160:480])


def test_listener_start_and_stop_are_idempotent(cfg: JarvisConfig) -> None:
    reader = ScriptedReader([])
    listener = WakeListener(cfg, reader, None, WakeWordDetector(cfg, FakeModel()))

    listener.stop()  # before ever starting
    listener.start()
    listener.start()
    assert listener.is_running

    listener.stop()
    listener.stop()
    assert not listener.is_running
    assert listener.detections == 0


def test_listener_survives_a_model_that_raises(cfg: JarvisConfig) -> None:
    tune(cfg, frame_samples=160, threshold=0.5)
    _buffer, reader = ring_with(ramp(480))
    model = FakeModel([RuntimeError("onnx session died"), ValueError("bad shape"), 0.9])
    detector = WakeWordDetector(cfg, model, clock=FakeClock())
    fired = threading.Event()

    listener = WakeListener(cfg, reader, None, detector, on_detected=lambda _d: fired.set())
    with listener:
        assert fired.wait(3.0), "the listener died on the first model exception"
        assert listener.is_running

    assert listener.errors == 2
    assert listener.detections == 1


def test_listener_survives_a_callback_that_raises(cfg: JarvisConfig) -> None:
    tune(cfg, frame_samples=160, threshold=0.5, cooldown_s=0.0)
    _buffer, reader = ring_with(ramp(320))
    detector = WakeWordDetector(cfg, FakeModel([0.9, 0.9]), clock=FakeClock())
    good: list[WakeDetection] = []

    def _angry(_detection: WakeDetection) -> None:
        raise RuntimeError("consumer is broken")

    listener = WakeListener(cfg, reader, None, detector)
    listener.add_listener(_angry)
    listener.add_listener(good.append)
    with listener:
        assert wait_until(lambda: len(good) == 2)
        assert listener.is_running

    assert listener.errors == 2, "each bad callback counts once"


def test_listener_survives_a_reader_that_raises(cfg: JarvisConfig) -> None:
    tune(cfg, frame_samples=160, threshold=0.5)
    reader = ScriptedReader([ramp(160)], failures=2)
    detector = WakeWordDetector(cfg, FakeModel([0.9]), clock=FakeClock())
    fired = threading.Event()

    listener = WakeListener(
        cfg, reader, None, detector, on_detected=lambda _d: fired.set(), poll_interval_s=0.0
    )
    with listener:
        assert fired.wait(3.0)

    assert listener.errors == 2


def test_listener_stops_when_the_dependency_is_missing(
    cfg: JarvisConfig, bus: EventBus, monkeypatch: pytest.MonkeyPatch
) -> None:
    tune(cfg, frame_samples=160)

    def _missing(name: str, **_: Any) -> types.ModuleType:
        raise DependencyMissingError(name, extra="audio")

    monkeypatch.setattr(wake_module, "require_module", _missing)
    _buffer, reader = ring_with(ramp(1600))
    errors, _ = bus.subscribe_queue([EventType.ERROR])

    listener = WakeListener(cfg, reader, bus)
    listener.start()
    assert wait_until(lambda: not listener.is_running), "the loop must not retry forever"

    event = errors.get(timeout=1.0)
    assert event.payload["source"] == "wake"
    assert event.payload["error"] == "DependencyMissingError"
    assert "openwakeword" in event.payload["speakable"]
    listener.stop()


def test_unsubscribing_a_callback_stops_delivery(cfg: JarvisConfig) -> None:
    tune(cfg, frame_samples=160, threshold=0.5, cooldown_s=0.0)
    _buffer, reader = ring_with(ramp(320))
    detector = WakeWordDetector(cfg, FakeModel([0.9, 0.9]), clock=FakeClock())
    removed: list[WakeDetection] = []
    kept: list[WakeDetection] = []

    listener = WakeListener(cfg, reader, None, detector)
    unsubscribe = listener.add_listener(removed.append)
    listener.add_listener(kept.append)
    unsubscribe()

    with listener:
        assert wait_until(lambda: len(kept) == 2)

    assert removed == [], "the removed callback still received detections"


def test_listener_exposes_the_detector_and_its_scores(cfg: JarvisConfig) -> None:
    tune(cfg, frame_samples=160)
    detector = WakeWordDetector(cfg, FakeModel([0.3]))
    listener = WakeListener(cfg, ScriptedReader([]), None, detector)

    assert listener.detector is detector
    detector.process(ramp(160))
    assert listener.scores[MODEL_NAME] == pytest.approx(0.3)


def test_listener_builds_its_own_detector_when_none_is_given(cfg: JarvisConfig) -> None:
    listener = WakeListener(cfg, ScriptedReader([]), None)

    assert isinstance(listener.detector, WakeWordDetector)
    assert not listener.detector.is_loaded, "construction must stay import free"


def test_listener_reads_whole_frames_only(cfg: JarvisConfig) -> None:
    tune(cfg, frame_samples=1280)
    reader = ScriptedReader([])
    listener = WakeListener(cfg, reader, None, WakeWordDetector(cfg, FakeModel()))

    with listener:
        assert wait_until(lambda: reader.reads > 0)

    # A starved reader must be polled, not spun on, and never asked for a
    # partial frame: the detector's framing depends on fixed size reads.
    assert reader.reads >= 1


# ---------------------------------------------------------------------------
# Manual verification on the Windows target host (§7 T-1.2)
# ---------------------------------------------------------------------------


def require_windows_audio_host() -> Any:
    """Skip unless this is the Windows host with openWakeWord and a human present."""
    if not is_windows():
        pytest.skip("wake word hardware checks only run on the Windows target host")
    if not has_module("openwakeword"):
        pytest.skip("openwakeword is not installed; run: uv sync --extra audio")
    if not sys.stdin.isatty():
        pytest.skip("needs an interactive terminal; run: uv run pytest -m manual -s")
    from jarvis.config import load_config

    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    return load_config(config_path if config_path.is_file() else None)


def say(message: str) -> None:
    """Write an instruction to the terminal without tripping the no-print rule."""
    sys.stdout.write(f"\n{message}\n")
    sys.stdout.flush()


@pytest.mark.manual
def test_ten_spoken_utterances_are_detected() -> None:
    """Say the wake word ten times, expect at least nine detections (§7 T-1.2).

    Procedure, on the Windows 11 target host with the usual microphone:

    1. ``uv sync --extra audio`` and ``powershell scripts/pull_models.ps1`` so the
       ``hey_jarvis`` checkpoint is in ``models/``.
    2. Sit at your normal desk position, roughly 0.5 to 1 metre from the mic,
       with whatever background noise is typical (fans, music off).
    3. Run ``uv run pytest tests/test_wake.py -m manual -s``.
    4. When prompted, say "hey jarvis" ten times in a normal speaking voice,
       leaving about three seconds between each so the cooldown expires.
    5. The test prints a running count. It passes at nine or more detections out
       of ten utterances, which is the T-1.2 acceptance criterion.

    If it fails, lower ``wake.threshold`` in ``config/config.yaml`` in steps of
    0.05 and rerun, then rerun the false-accept check below at the new value.
    Record the final threshold in ``PROGRESS.md``.
    """
    config = require_windows_audio_host()
    from jarvis.audio.ring import AudioCapture

    detections: list[WakeDetection] = []
    capture = AudioCapture(config)
    listener = WakeListener(
        config, capture.reader(), None, on_detected=detections.append
    )
    say("Say 'hey jarvis' ten times, about three seconds apart. Press Enter to start.")
    input()
    with capture, listener:
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline and len(detections) < 10:
            time.sleep(0.25)
    say(f"detections: {len(detections)} of 10 utterances")

    assert len(detections) >= 9, f"only {len(detections)} of 10 utterances were detected"
    assert all(d.preroll.size == listener.detector.preroll_samples for d in detections)


@pytest.mark.manual
def test_ambient_audio_produces_at_most_one_false_accept() -> None:
    """Ten minutes of ambient room audio, at most one false accept (§7 T-1.2).

    Procedure, on the Windows 11 target host:

    1. Same setup as the detection-rate check above.
    2. Run ``uv run pytest tests/test_wake.py -m manual -s`` and leave the room
       in its normal state: talk, type, play a video, but never say the wake
       word. Ten minutes of wall clock.
    3. The test passes at one or fewer detections.

    A failure means the threshold is too low for this microphone. Raise
    ``wake.threshold`` by 0.05, rerun the detection-rate check to confirm it is
    still at nine of ten, and record both numbers in ``PROGRESS.md``.
    """
    config = require_windows_audio_host()
    from jarvis.audio.ring import AudioCapture

    detections: list[WakeDetection] = []
    capture = AudioCapture(config)
    listener = WakeListener(
        config, capture.reader(), None, on_detected=detections.append
    )
    say("Ambient check running for 10 minutes. Do not say the wake word. Press Enter.")
    input()
    with capture, listener:
        time.sleep(600.0)
    say(f"false accepts: {len(detections)}")

    assert len(detections) <= 1, f"{len(detections)} false accepts in ten minutes"
