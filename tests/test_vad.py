"""T-1.3: Silero VAD wrapper, utterance endpointing, and barge-in detection.

Nothing here needs ONNX Runtime, a model checkpoint, a microphone, or Windows.
The ONNX session is faked and the endpointing policy is driven from scripted
probability sequences, which is the only way to test a boundary condition such
as "exactly one frame short of the trailing silence threshold" deterministically.

The one thing that genuinely needs the real model is marked ``manual`` and is
written so it passes on the Windows host once ``scripts/pull_models.ps1`` has
run.

Frame arithmetic used throughout, all from the shipped defaults:

    frame_samples 512 at 16000 Hz  ->  32.0 ms per frame
    min_speech_ms 250              ->   8 frames (256 ms)
    trailing_silence_ms 500        ->  16 frames (512 ms)
    speech_pad_ms 100              ->   4 frames (128 ms)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from jarvis.audio.vad import (
    _CONTEXT_SAMPLES,
    BargeInDetector,
    Endpoint,
    Endpointer,
    SileroVad,
    SpeechState,
)
from jarvis.config import JarvisConfig, load_config
from jarvis.util.errors import AudioError, DependencyMissingError
from jarvis.util.platform import has_module

RATE = 16_000
FRAME = 512
FRAME_MS = 32.0

#: CLAUDE.md §3 budgets the VAD endpoint decision at this many milliseconds.
BUDGET_MS = 250.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(tmp_path: Path, *, audio: dict[str, Any] | None = None, **vad: Any) -> JarvisConfig:
    """A config with ``vad`` (and optionally ``audio``) overrides, paths in tmp."""
    return load_config(
        tmp_path / "absent.yaml",
        paths={
            "data_dir": str(tmp_path / "data"),
            "log_dir": str(tmp_path / "logs"),
            "models_dir": str(tmp_path / "models"),
            "vendor_dir": str(tmp_path / "vendor"),
        },
        vad=vad,
        audio=audio or {},
    )


class ScriptedVad:
    """A probability source that reads its answers off a list.

    Records every frame it was handed, so a test can prove that a rejected
    frame never reached the model, and counts resets so a test can prove that
    :meth:`Endpointer.reset` propagates.
    """

    def __init__(self, probabilities: Sequence[float]) -> None:
        self.script = list(probabilities)
        self.index = 0
        self.frames: list[np.ndarray[Any, Any]] = []
        self.resets = 0

    def probability(self, frame: np.ndarray[Any, Any]) -> float:
        if self.index >= len(self.script):
            raise AssertionError("scripted vad ran out of probabilities")
        self.frames.append(np.asarray(frame).copy())
        value = self.script[self.index]
        self.index += 1
        return value

    def reset(self) -> None:
        self.resets += 1


class _Named:
    """Stands in for an ONNX ``NodeArg``, which only its ``name`` is read from."""

    def __init__(self, name: str) -> None:
        self.name = name


class FakeSession:
    """Stands in for ``onnxruntime.InferenceSession`` for the Silero graph.

    The state tensor it returns is filled with the call number, so a test can
    read the feed of call N and prove it holds what call N-1 produced. That is
    the whole point of the wrapper and cannot be checked any other way.
    """

    def __init__(
        self,
        probabilities: Sequence[float] = (0.9,),
        *,
        inputs: tuple[str, ...] = ("input", "state", "sr"),
        outputs: tuple[str, ...] = ("output", "stateN"),
        fail: bool = False,
        empty_output: bool = False,
    ) -> None:
        self.script = list(probabilities)
        self.inputs = inputs
        self.outputs = outputs
        self.fail = fail
        self.empty_output = empty_output
        self.calls: list[dict[str, Any]] = []
        self.requested: list[list[str]] = []

    def get_inputs(self) -> list[_Named]:
        return [_Named(name) for name in self.inputs]

    def get_outputs(self) -> list[_Named]:
        return [_Named(name) for name in self.outputs]

    def run(self, output_names: Sequence[str], feed: dict[str, Any]) -> list[Any]:
        if self.fail:
            raise RuntimeError("onnxruntime exploded")
        self.calls.append({key: np.array(value, copy=True) for key, value in feed.items()})
        self.requested.append(list(output_names))
        if self.empty_output:
            return []
        n = len(self.calls)
        prob = self.script[min(n - 1, len(self.script) - 1)]
        out = np.array([[prob]], dtype=np.float32)
        if "h" in self.inputs:
            return [
                out,
                np.full((1, 1, 128), float(n), dtype=np.float32),
                np.full((1, 1, 128), float(n) + 0.5, dtype=np.float32),
            ]
        return [out, np.full((2, 1, 128), float(n), dtype=np.float32)]


class BareSession:
    """A session that refuses to describe itself, exercising the name fallback."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(self, output_names: Sequence[str], feed: dict[str, Any]) -> list[Any]:
        self.calls.append(dict(feed))
        return [np.array([[0.75]], dtype=np.float32), np.zeros((2, 1, 128), dtype=np.float32)]


def zeros_frame() -> np.ndarray[Any, Any]:
    """One frame of silence."""
    return np.zeros(FRAME, dtype=np.float32)


def run_script(
    config: JarvisConfig,
    probabilities: Sequence[float],
    frames: Sequence[np.ndarray[Any, Any]] | None = None,
) -> tuple[Endpointer, ScriptedVad, list[tuple[int, Endpoint]]]:
    """Drive an endpointer one frame per probability.

    Returns:
        The endpointer, the scripted source, and every endpoint with the index
        of the frame that produced it.
    """
    source = ScriptedVad(probabilities)
    endpointer = Endpointer(config, vad=source, sample_rate=RATE)
    supply = list(frames) if frames is not None else [zeros_frame()] * len(probabilities)
    results: list[tuple[int, Endpoint]] = []
    for index, frame in enumerate(supply):
        endpoint = endpointer.process(frame)
        if endpoint is not None:
            results.append((index, endpoint))
    return endpointer, source, results


def voiced_frames(
    probabilities: Sequence[float],
    speech: np.ndarray[Any, Any],
    quiet: np.ndarray[Any, Any],
    threshold: float = 0.5,
) -> list[np.ndarray[Any, Any]]:
    """Audio matching a probability script: tone where speech, zeros elsewhere."""
    return [speech if p >= threshold else quiet for p in probabilities]


# ---------------------------------------------------------------------------
# SileroVad
# ---------------------------------------------------------------------------


def test_silero_threads_recurrent_state_between_frames(cfg: JarvisConfig) -> None:
    session = FakeSession([0.1, 0.2, 0.3])
    vad = SileroVad(cfg, session=session)

    values = [vad.probability(zeros_frame()) for _ in range(3)]

    assert values == [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3)]
    assert len(session.calls) == 3
    # Frame 1 starts from a zeroed state, then every frame is handed exactly
    # what the previous call returned. Without this the model is stateless and
    # its probabilities are meaningless.
    assert np.all(session.calls[0]["state"] == 0.0)
    assert np.all(session.calls[1]["state"] == 1.0)
    assert np.all(session.calls[2]["state"] == 2.0)
    assert vad.frames_processed == 3
    assert vad.is_loaded


def test_silero_feeds_the_right_shapes_and_sample_rate(cfg: JarvisConfig) -> None:
    session = FakeSession([0.4])
    vad = SileroVad(cfg, session=session)

    vad.probability(zeros_frame())

    feed = session.calls[0]
    # Frame plus the context window Silero requires. This asserted a bare
    # (1, FRAME) for a long time, which is exactly the shape that makes the
    # model return near zero for speech, so the test held the defect in place.
    assert feed["input"].shape == (1, FRAME + _CONTEXT_SAMPLES[RATE])
    assert feed["input"].dtype == np.float32
    assert int(feed["sr"]) == RATE
    assert feed["state"].shape == (2, 1, 128)
    assert session.requested[0] == ["output", "stateN"]
    assert vad.frame_samples == FRAME
    assert vad.sample_rate == RATE


def test_silero_reset_zeroes_the_state(cfg: JarvisConfig) -> None:
    session = FakeSession([0.9])
    vad = SileroVad(cfg, session=session)
    for _ in range(3):
        vad.probability(zeros_frame())
    assert np.all(session.calls[2]["state"] == 2.0)

    vad.reset()
    vad.probability(zeros_frame())

    assert np.all(session.calls[3]["state"] == 0.0)
    assert vad.frames_processed == 1


def test_silero_rejects_wrong_frame_length_without_touching_the_session(
    cfg: JarvisConfig,
) -> None:
    session = FakeSession([0.9])
    vad = SileroVad(cfg, session=session)
    vad.probability(zeros_frame())
    vad.probability(zeros_frame())

    with pytest.raises(AudioError) as excinfo:
        vad.probability(np.zeros(FRAME - 1, dtype=np.float32))

    assert "512" in str(excinfo.value)
    assert excinfo.value.context["got_samples"] == FRAME - 1
    # The session never ran, so the recurrent state is still the one produced by
    # the last good frame and the stream carries on undisturbed.
    assert len(session.calls) == 2
    vad.probability(zeros_frame())
    assert np.all(session.calls[2]["state"] == 2.0)


def test_silero_accepts_int16_frames(cfg: JarvisConfig, sine_audio: Any) -> None:
    session = FakeSession([0.6])
    vad = SileroVad(cfg, session=session)
    tone = sine_audio(FRAME / RATE, amplitude=0.5)

    vad.probability((tone * 32767).astype(np.int16))

    fed = session.calls[0]["input"]
    assert fed.dtype == np.float32
    assert float(np.abs(fed).max()) == pytest.approx(0.5, abs=1e-3)


def test_silero_clamps_probabilities_into_range(cfg: JarvisConfig) -> None:
    vad = SileroVad(cfg, session=FakeSession([1.4, -0.3, 0.5]))

    assert vad.probability(zeros_frame()) == 1.0
    assert vad.probability(zeros_frame()) == 0.0
    assert vad.probability(zeros_frame()) == pytest.approx(0.5)


def test_silero_rejects_a_non_finite_probability(cfg: JarvisConfig) -> None:
    vad = SileroVad(cfg, session=FakeSession([float("nan")]))

    with pytest.raises(AudioError, match="non-finite"):
        vad.probability(zeros_frame())


def test_silero_wraps_a_session_failure(cfg: JarvisConfig) -> None:
    vad = SileroVad(cfg, session=FakeSession(fail=True))

    with pytest.raises(AudioError) as excinfo:
        vad.probability(zeros_frame())

    assert "inference failed" in str(excinfo.value)
    assert excinfo.value.speakable  # never a stack trace out loud (§5)


def test_silero_rejects_an_empty_output(cfg: JarvisConfig) -> None:
    vad = SileroVad(cfg, session=FakeSession(empty_output=True))

    with pytest.raises(AudioError, match="no usable probability"):
        vad.probability(zeros_frame())


def test_silero_supports_the_v4_h_and_c_layout(cfg: JarvisConfig) -> None:
    session = FakeSession(
        [0.8],
        inputs=("input", "h", "c", "sr"),
        outputs=("output", "hn", "cn"),
    )
    vad = SileroVad(cfg, session=session)

    vad.probability(zeros_frame())
    vad.probability(zeros_frame())

    assert "state" not in session.calls[0]
    assert session.calls[0]["h"].shape == (1, 1, 128)
    assert np.all(session.calls[0]["c"] == 0.0)
    assert np.all(session.calls[1]["h"] == 1.0)
    assert np.all(session.calls[1]["c"] == 1.5)


def test_silero_falls_back_to_v5_names_when_the_session_will_not_introspect(
    cfg: JarvisConfig,
) -> None:
    session = BareSession()
    vad = SileroVad(cfg, session=session)

    assert vad.probability(zeros_frame()) == pytest.approx(0.75)
    assert set(session.calls[0]) == {"input", "state", "sr"}


def test_silero_rejects_an_unsupported_sample_rate(tmp_path: Path) -> None:
    config = make_config(tmp_path, audio={"sample_rate": 44_100})

    with pytest.raises(AudioError, match="8000 Hz and 16000 Hz"):
        SileroVad(config)


def test_silero_rejects_a_mismatched_rate_and_frame_size(tmp_path: Path) -> None:
    config = make_config(tmp_path, audio={"sample_rate": 8_000}, frame_samples=512)

    with pytest.raises(AudioError, match="256 samples per frame"):
        SileroVad(config)


def test_silero_accepts_the_8k_pairing(tmp_path: Path) -> None:
    config = make_config(tmp_path, audio={"sample_rate": 8_000}, frame_samples=256)
    session = FakeSession([0.5])
    vad = SileroVad(config, session=session)

    vad.probability(np.zeros(256, dtype=np.float32))

    assert int(session.calls[0]["sr"]) == 8_000
    assert session.calls[0]["input"].shape == (1, 256 + _CONTEXT_SAMPLES[8_000])


@pytest.mark.skipif(has_module("onnxruntime"), reason="onnxruntime is installed here")
def test_silero_lazy_load_reports_the_missing_runtime(cfg: JarvisConfig) -> None:
    vad = SileroVad(cfg)

    with pytest.raises(DependencyMissingError) as excinfo:
        vad.probability(zeros_frame())

    assert excinfo.value.package == "onnxruntime"
    assert "uv sync --extra audio" in str(excinfo.value)


@pytest.mark.skipif(not has_module("onnxruntime"), reason="needs onnxruntime")
def test_silero_lazy_load_reports_the_missing_checkpoint(cfg: JarvisConfig) -> None:
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    vad = SileroVad(cfg)

    with pytest.raises(AudioError, match=r"silero_vad\.onnx"):
        vad.probability(zeros_frame())


def test_silero_checkpoint_lookup_prefers_the_exact_name(cfg: JarvisConfig) -> None:
    # Pinned directly because the lookup order is a deployment contract with
    # scripts/pull_models.ps1 and cannot be reached through probability() on a
    # host with no ONNX Runtime.
    models = cfg.models_dir
    (models / "silero_vad").mkdir(parents=True, exist_ok=True)
    nested = models / "silero_vad" / "silero_vad.onnx"
    nested.write_bytes(b"not a real model")
    vad = SileroVad(cfg, session=FakeSession())
    assert vad.find_model_file() == nested

    top = models / "silero_vad.onnx"
    top.write_bytes(b"not a real model either")
    assert vad.find_model_file() == top


def test_silero_checkpoint_lookup_accepts_a_versioned_name(cfg: JarvisConfig) -> None:
    models = cfg.models_dir
    models.mkdir(parents=True, exist_ok=True)
    (models / "silero_vad_v4.onnx").write_bytes(b"old")
    (models / "silero_vad_v5.onnx").write_bytes(b"new")
    vad = SileroVad(cfg, session=FakeSession())

    assert vad.find_model_file() == models / "silero_vad_v5.onnx"


def test_silero_checkpoint_lookup_returns_none_when_absent(cfg: JarvisConfig) -> None:
    vad = SileroVad(cfg, session=FakeSession())

    assert vad.find_model_file() is None


# ---------------------------------------------------------------------------
# Endpointer: frame arithmetic
# ---------------------------------------------------------------------------


def test_endpointer_derives_frame_counts_from_the_config(cfg: JarvisConfig) -> None:
    endpointer = Endpointer(cfg, vad=lambda _frame: 0.0, sample_rate=RATE)

    assert endpointer.frame_samples == FRAME
    assert endpointer.frame_ms == pytest.approx(FRAME_MS)
    # Every threshold rounds up, so it is met rather than undershot.
    assert endpointer.min_speech_frames == math.ceil(250 / FRAME_MS) == 8
    assert endpointer.trailing_silence_frames == math.ceil(500 / FRAME_MS) == 16
    assert endpointer.pad_frames == math.ceil(100 / FRAME_MS) == 4
    assert endpointer.max_frames == math.ceil(30_000 / FRAME_MS) == 938
    assert endpointer.state is SpeechState.SILENCE
    assert endpointer.is_speaking is False


def test_endpointer_accepts_a_bare_callable_source(cfg: JarvisConfig) -> None:
    endpointer = Endpointer(cfg, vad=lambda _frame: 0.99, sample_rate=RATE)

    for _ in range(8):
        assert endpointer.process(zeros_frame()) is None

    assert endpointer.state is SpeechState.SPEECH


def test_endpointer_rejects_a_source_it_cannot_call(cfg: JarvisConfig) -> None:
    with pytest.raises(AudioError, match="neither probability"):
        Endpointer(cfg, vad=42, sample_rate=RATE)


# ---------------------------------------------------------------------------
# Endpointer: the policy
# ---------------------------------------------------------------------------


def test_blip_shorter_than_min_speech_never_opens_a_turn(cfg: JarvisConfig) -> None:
    # Three frames of speech is 96 ms, well under the 250 ms floor. A cough.
    script = [0.9] * 3 + [0.0] * 40
    endpointer, _source, results = run_script(cfg, script)

    assert results == []
    assert endpointer.state is SpeechState.SILENCE
    assert endpointer.is_speaking is False
    assert endpointer.buffered_seconds == 0.0


def test_speech_exactly_at_min_speech_opens_a_turn(cfg: JarvisConfig) -> None:
    seven = [0.9] * 7 + [0.0] * 20
    eight = [0.9] * 8 + [0.0] * 20

    _ep_short, _s1, short_results = run_script(cfg, seven)
    _ep_long, _s2, long_results = run_script(cfg, eight)

    assert short_results == []  # 224 ms, one frame short of the floor
    assert len(long_results) == 1  # 256 ms, over the floor
    assert long_results[0][1].reason == "trailing_silence"


def test_normal_utterance_has_correct_padding_and_duration(
    cfg: JarvisConfig, sine_audio: Any, silence: Any
) -> None:
    speech = sine_audio(FRAME / RATE)
    quiet = silence(FRAME / RATE)
    script = [0.0] * 5 + [0.9] * 20 + [0.0] * 20
    frames = voiced_frames(script, speech, quiet)

    endpointer, _source, results = run_script(cfg, script, frames)

    assert len(results) == 1
    index, endpoint = results[0]
    # Speech runs 5..24, so the 16th silent frame after it is frame 40.
    assert index == 40
    assert endpoint.reason == "trailing_silence"
    # 4 frames of lead pad, 20 of speech, 4 of trailing pad. The trailing pad is
    # trimmed out of the 16 silent frames that were buffered while deciding.
    expected = np.concatenate([quiet] * 4 + [speech] * 20 + [quiet] * 4)
    np.testing.assert_allclose(endpoint.audio, expected)
    assert endpoint.audio.dtype == np.float32
    assert endpoint.samples == 28 * FRAME
    assert endpoint.duration_s == pytest.approx(28 * FRAME / RATE)
    assert endpoint.duration_s == pytest.approx(0.896)
    assert endpointer.state is SpeechState.SILENCE


def test_endpoint_is_produced_exactly_once(cfg: JarvisConfig) -> None:
    script = [0.0] * 4 + [0.9] * 10 + [0.0] * 30
    _endpointer, _source, results = run_script(cfg, script)

    assert len(results) == 1


def test_lead_pad_is_truncated_when_the_stream_starts_mid_speech(
    cfg: JarvisConfig, sine_audio: Any, silence: Any
) -> None:
    speech = sine_audio(FRAME / RATE)
    quiet = silence(FRAME / RATE)
    script = [0.9] * 10 + [0.0] * 20
    frames = voiced_frames(script, speech, quiet)

    _endpointer, _source, results = run_script(cfg, script, frames)

    assert len(results) == 1
    endpoint = results[0][1]
    # No audio existed before the speech, so the pad is short rather than
    # zero filled: the utterance simply starts at its first speech frame.
    expected = np.concatenate([speech] * 10 + [quiet] * 4)
    np.testing.assert_allclose(endpoint.audio, expected)


def test_trailing_silence_closes_exactly_on_the_boundary(tmp_path: Path) -> None:
    # 384 ms is exactly 12 frames, so the boundary is unambiguous.
    config = make_config(tmp_path, trailing_silence_ms=384)
    endpointer = Endpointer(config, vad=lambda _f: 0.0, sample_rate=RATE)
    assert endpointer.trailing_silence_frames == 12

    just_short = [0.9] * 10 + [0.0] * 11
    exact = [0.9] * 10 + [0.0] * 12

    _ep_a, _s_a, short_results = run_script(config, just_short)
    _ep_b, _s_b, exact_results = run_script(config, exact)

    assert short_results == []
    assert len(exact_results) == 1
    index, endpoint = exact_results[0]
    assert index == 21  # the 12th silent frame, 0-indexed
    assert endpoint.decision_ms == pytest.approx(384.0)


def test_a_pause_shorter_than_the_threshold_does_not_split_the_utterance(
    cfg: JarvisConfig, sine_audio: Any, silence: Any
) -> None:
    speech = sine_audio(FRAME / RATE)
    quiet = silence(FRAME / RATE)
    # 10 silent frames is 320 ms, inside the 512 ms threshold.
    script = [0.9] * 8 + [0.0] * 10 + [0.9] * 8 + [0.0] * 20
    frames = voiced_frames(script, speech, quiet)

    endpointer, _source, results = run_script(cfg, script, frames)

    assert len(results) == 1
    endpoint = results[0][1]
    assert endpoint.samples == 30 * FRAME  # 8 + 10 + 8 speech-side, plus 4 pad
    assert endpoint.duration_s == pytest.approx(0.96)
    assert endpointer.state is SpeechState.SILENCE


def test_a_blip_before_real_speech_is_left_out_of_the_utterance(
    cfg: JarvisConfig, sine_audio: Any, silence: Any
) -> None:
    speech = sine_audio(FRAME / RATE)
    quiet = silence(FRAME / RATE)
    script = [0.9] * 3 + [0.0] * 20 + [0.9] * 10 + [0.0] * 20
    frames = voiced_frames(script, speech, quiet)

    _endpointer, _source, results = run_script(cfg, script, frames)

    assert len(results) == 1
    endpoint = results[0][1]
    # 4 pad + 10 speech + 4 pad. The discarded blip's audio is gone, which is
    # only true if abandoning a candidate really does drop its buffer.
    expected = np.concatenate([quiet] * 4 + [speech] * 10 + [quiet] * 4)
    np.testing.assert_allclose(endpoint.audio, expected)


def test_two_utterances_in_a_row_each_get_their_pads(
    cfg: JarvisConfig, sine_audio: Any, silence: Any
) -> None:
    speech = sine_audio(FRAME / RATE)
    quiet = silence(FRAME / RATE)
    one_turn = [0.0] * 4 + [0.9] * 10 + [0.0] * 16
    script = one_turn * 2
    frames = voiced_frames(script, speech, quiet)

    _endpointer, _source, results = run_script(cfg, script, frames)

    assert len(results) == 2
    expected = np.concatenate([quiet] * 4 + [speech] * 10 + [quiet] * 4)
    for _index, endpoint in results:
        np.testing.assert_allclose(endpoint.audio, expected)
        assert endpoint.reason == "trailing_silence"
    assert results[1][0] - results[0][0] == 30


def test_zero_padding_keeps_only_the_speech(
    tmp_path: Path, sine_audio: Any, silence: Any
) -> None:
    config = make_config(tmp_path, speech_pad_ms=0)
    speech = sine_audio(FRAME / RATE)
    quiet = silence(FRAME / RATE)
    script = [0.0] * 4 + [0.9] * 10 + [0.0] * 20
    frames = voiced_frames(script, speech, quiet)

    endpointer, _source, results = run_script(config, script, frames)

    assert endpointer.pad_frames == 0
    assert len(results) == 1
    np.testing.assert_allclose(results[0][1].audio, np.concatenate([speech] * 10))


def test_max_duration_force_closes_the_turn(tmp_path: Path) -> None:
    config = make_config(tmp_path, max_utterance_s=1.0)
    endpointer = Endpointer(config, vad=lambda _f: 0.0, sample_rate=RATE)
    assert endpointer.max_frames == 32  # 1000 ms / 32 ms, rounded up

    _ep, _source, results = run_script(config, [0.99] * 60)

    index, endpoint = results[0]
    assert index == 31
    assert endpoint.reason == "max_duration"
    assert endpoint.samples == 32 * FRAME
    assert endpoint.duration_s == pytest.approx(1.024)
    # The user was still talking when the cap hit, so there was no waiting.
    assert endpoint.decision_ms == pytest.approx(0.0)
    # Speech that runs forever keeps being chunked rather than swallowed. The
    # second chunk carries the previous chunk's trailing pad as its lead pad,
    # so the two overlap slightly and nothing at the seam is lost.
    assert len(results) == 2
    assert results[1][0] == 59
    assert results[1][1].reason == "max_duration"
    assert results[1][1].duration_s == pytest.approx(1.024)


def test_max_duration_discards_a_candidate_that_never_confirmed(tmp_path: Path) -> None:
    # A probability that never accumulates min_speech_ms: one qualifying frame,
    # one silent frame, forever. The floor is raised well past the cap so the
    # candidate can churn for the full second without ever confirming.
    config = make_config(tmp_path, max_utterance_s=1.0, min_speech_ms=2_000)
    script = [0.9, 0.0] * 40
    source = ScriptedVad(script)
    endpointer = Endpointer(config, vad=source, sample_rate=RATE)

    peak = 0.0
    for _ in script:
        assert endpointer.process(zeros_frame()) is None
        peak = max(peak, endpointer.buffered_seconds)

    # Nothing is ever emitted, because emitting would break the min_speech
    # guarantee, and the buffer is never allowed past the cap either, because
    # that is what stops a stuck detector eating memory for the whole session.
    assert endpointer.state is SpeechState.SILENCE
    assert peak == pytest.approx(config.vad.max_utterance_s, abs=FRAME / RATE)


def test_states_walk_silence_speech_trailing_and_back(cfg: JarvisConfig) -> None:
    source = ScriptedVad([0.9] * 3 + [0.9] * 5 + [0.0] + [0.9] + [0.0] * 16)
    endpointer = Endpointer(cfg, vad=source, sample_rate=RATE)

    for _ in range(3):
        endpointer.process(zeros_frame())
    # An unconfirmed candidate still reads as silence: it may be a cough.
    assert endpointer.state is SpeechState.SILENCE
    assert endpointer.is_speaking is False
    assert endpointer.speech_ms == pytest.approx(96.0)

    for _ in range(5):
        endpointer.process(zeros_frame())
    assert endpointer.state is SpeechState.SPEECH
    assert endpointer.is_speaking is True

    endpointer.process(zeros_frame())
    assert endpointer.state is SpeechState.TRAILING
    assert endpointer.is_speaking is True  # the turn is still open
    assert endpointer.silence_ms == pytest.approx(32.0)

    endpointer.process(zeros_frame())
    assert endpointer.state is SpeechState.SPEECH
    assert endpointer.silence_ms == 0.0

    endpoint = None
    for _ in range(16):
        endpoint = endpointer.process(zeros_frame()) or endpoint
    assert endpoint is not None
    assert endpointer.state is SpeechState.SILENCE
    assert endpointer.is_speaking is False


def test_buffered_seconds_tracks_the_open_utterance(cfg: JarvisConfig) -> None:
    endpointer = Endpointer(cfg, vad=lambda _f: 0.9, sample_rate=RATE)

    for _ in range(10):
        endpointer.process(zeros_frame())

    assert endpointer.buffered_seconds == pytest.approx(10 * FRAME / RATE)
    assert endpointer.frames_processed == 10


def test_int16_frames_are_converted_to_float32(
    cfg: JarvisConfig, sine_audio: Any
) -> None:
    tone = sine_audio(FRAME / RATE, amplitude=0.5)
    speech = (tone * 32767).astype(np.int16)
    quiet = np.zeros(FRAME, dtype=np.int16)
    script = [0.9] * 10 + [0.0] * 20
    frames = voiced_frames(script, speech, quiet)

    _endpointer, _source, results = run_script(cfg, script, frames)

    assert len(results) == 1
    audio = results[0][1].audio
    assert audio.dtype == np.float32
    assert float(np.abs(audio).max()) == pytest.approx(0.5, abs=1e-3)


# ---------------------------------------------------------------------------
# Endpointer: reset and bad input
# ---------------------------------------------------------------------------


def test_reset_clears_the_buffer_and_the_source(cfg: JarvisConfig) -> None:
    source = ScriptedVad([0.9] * 10 + [0.9] * 10 + [0.0] * 20)
    endpointer = Endpointer(cfg, vad=source, sample_rate=RATE)
    for _ in range(10):
        endpointer.process(zeros_frame())
    assert endpointer.state is SpeechState.SPEECH

    endpointer.reset()

    assert endpointer.state is SpeechState.SILENCE
    assert endpointer.is_speaking is False
    assert endpointer.buffered_seconds == 0.0
    assert endpointer.speech_ms == 0.0
    assert endpointer.frames_processed == 0
    assert source.resets == 1

    # The abandoned audio must not leak into the next utterance.
    endpoint = None
    for _ in range(30):
        endpoint = endpointer.process(zeros_frame()) or endpoint
    assert endpoint is not None
    # 10 speech frames plus the 4 frame trailing pad, and no lead pad at all:
    # reset cleared the pre-roll, so none of the pre-reset audio survived.
    assert endpoint.samples == 14 * FRAME


def test_wrong_frame_length_raises_without_corrupting_the_state(
    cfg: JarvisConfig, sine_audio: Any, silence: Any
) -> None:
    speech = sine_audio(FRAME / RATE)
    quiet = silence(FRAME / RATE)
    source = ScriptedVad([0.9] * 10 + [0.0] * 20)
    endpointer = Endpointer(cfg, vad=source, sample_rate=RATE)
    for _ in range(10):
        endpointer.process(speech)
    assert endpointer.state is SpeechState.SPEECH

    with pytest.raises(AudioError) as excinfo:
        endpointer.process(np.zeros(FRAME * 2, dtype=np.float32))

    assert excinfo.value.context["expected_samples"] == FRAME
    assert excinfo.value.context["got_samples"] == FRAME * 2
    # The bad frame was never scored and never buffered.
    assert source.index == 10
    assert len(source.frames) == 10
    assert endpointer.state is SpeechState.SPEECH
    assert endpointer.buffered_seconds == pytest.approx(10 * FRAME / RATE)

    endpoint = None
    for _ in range(20):
        endpoint = endpointer.process(quiet) or endpoint
    assert endpoint is not None
    # Exactly what it would have been had the bad frame never arrived.
    np.testing.assert_allclose(
        endpoint.audio, np.concatenate([speech] * 10 + [quiet] * 4)
    )


# ---------------------------------------------------------------------------
# Endpointer: decision latency against the CLAUDE.md §3 budget
# ---------------------------------------------------------------------------


def test_decision_ms_matches_the_trailing_silence_arithmetic(cfg: JarvisConfig) -> None:
    endpointer = Endpointer(cfg, vad=lambda _f: 0.0, sample_rate=RATE)
    assert endpointer.expected_decision_ms == pytest.approx(512.0)

    script = [0.9] * 10 + [0.0] * 20
    _ep, _source, results = run_script(cfg, script)

    assert len(results) == 1
    measured = results[0][1].decision_ms
    # 16 silent frames at 32 ms each, measured from the last speech frame.
    assert measured == pytest.approx(16 * FRAME_MS)
    assert measured == pytest.approx(endpointer.expected_decision_ms)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The shipped vad.trailing_silence_ms of 500 costs 16 frames of 32 ms, so the "
        "endpoint decision takes 512 ms against the 250 ms budget in CLAUDE.md section 3. "
        "Fixing it means lowering trailing_silence_ms to 224 or less, which is a tuning "
        "decision for the Phase 1 gate, not something to change quietly here."
    ),
)
def test_default_endpoint_decision_fits_the_budget(cfg: JarvisConfig) -> None:
    endpointer = Endpointer(cfg, vad=lambda _f: 0.0, sample_rate=RATE)

    assert endpointer.expected_decision_ms <= BUDGET_MS


def test_the_budget_is_reachable_by_tightening_trailing_silence(tmp_path: Path) -> None:
    # 224 ms is 7 frames exactly, the most that fits inside 250 ms.
    inside = Endpointer(
        make_config(tmp_path, trailing_silence_ms=224), vad=lambda _f: 0.0, sample_rate=RATE
    )
    outside = Endpointer(
        make_config(tmp_path, trailing_silence_ms=225), vad=lambda _f: 0.0, sample_rate=RATE
    )

    assert inside.trailing_silence_frames == 7
    assert inside.expected_decision_ms == pytest.approx(224.0)
    assert inside.expected_decision_ms <= BUDGET_MS
    # One millisecond more needs an eighth frame, which lands at 256 ms.
    assert outside.expected_decision_ms == pytest.approx(256.0)
    assert outside.expected_decision_ms > BUDGET_MS


def test_a_tightened_config_measures_inside_the_budget(tmp_path: Path) -> None:
    config = make_config(tmp_path, trailing_silence_ms=224)
    script = [0.9] * 10 + [0.0] * 10

    _ep, _source, results = run_script(config, script)

    assert len(results) == 1
    assert results[0][1].decision_ms <= BUDGET_MS


# ---------------------------------------------------------------------------
# BargeInDetector
# ---------------------------------------------------------------------------


def test_barge_in_uses_the_higher_threshold(cfg: JarvisConfig) -> None:
    detector = BargeInDetector(cfg, vad=lambda _f: 0.6)

    assert cfg.vad.threshold == 0.5  # the endpointer would call 0.6 speech
    assert detector.threshold == 0.7
    for _ in range(50):
        assert detector.process(zeros_frame()) is False
    assert detector.triggered is False
    assert detector.speech_ms == 0.0


def test_barge_in_fires_once_the_run_is_long_enough(cfg: JarvisConfig) -> None:
    detector = BargeInDetector(cfg, vad=lambda _f: 0.85)
    # 200 ms of 32 ms frames rounds up to 7.
    assert detector.min_speech_frames == math.ceil(200 / FRAME_MS) == 7

    results = [detector.process(zeros_frame()) for _ in range(8)]

    assert results == [False] * 6 + [True] * 2
    assert detector.speech_ms == pytest.approx(8 * FRAME_MS)


def test_barge_in_needs_consecutive_frames(cfg: JarvisConfig) -> None:
    script = [0.9, 0.9, 0.9, 0.1] * 20  # never seven in a row
    source = ScriptedVad(script)
    detector = BargeInDetector(cfg, vad=source)

    assert not any(detector.process(zeros_frame()) for _ in script)
    assert detector.triggered is False


def test_barge_in_latches_until_reset(cfg: JarvisConfig) -> None:
    script = [0.9] * 7 + [0.0] * 5
    source = ScriptedVad(script)
    detector = BargeInDetector(cfg, vad=source)

    fired = [detector.process(zeros_frame()) for _ in script]

    # Silence after the trigger must not un-fire it: the caller may only poll
    # every few frames, and a missed edge means playback never stops.
    assert fired == [False] * 6 + [True] * 6
    assert detector.speech_ms == 0.0  # the run itself did reset
    assert source.index == len(script)  # frames keep being scored after firing
    assert detector.frames_processed == len(script)

    detector.reset()

    assert detector.triggered is False
    assert detector.frames_processed == 0
    assert source.resets == 1


def test_barge_in_rejects_the_wrong_frame_length(cfg: JarvisConfig) -> None:
    source = ScriptedVad([0.9] * 3)
    detector = BargeInDetector(cfg, vad=source)
    detector.process(zeros_frame())

    with pytest.raises(AudioError) as excinfo:
        detector.process(np.zeros(FRAME + 1, dtype=np.float32))

    assert excinfo.value.context["got_samples"] == FRAME + 1
    assert source.index == 1
    assert detector.speech_ms == pytest.approx(FRAME_MS)


def test_barge_in_honours_a_custom_minimum(tmp_path: Path) -> None:
    config = make_config(tmp_path, barge_in_min_speech_ms=64, barge_in_threshold=0.9)
    detector = BargeInDetector(config, vad=lambda _f: 0.95)

    assert detector.min_speech_frames == 2
    assert detector.process(zeros_frame()) is False
    assert detector.process(zeros_frame()) is True


def test_barge_in_accepts_int16_frames(cfg: JarvisConfig, sine_audio: Any) -> None:
    source = ScriptedVad([0.99] * 7)
    detector = BargeInDetector(cfg, vad=source)
    tone = (sine_audio(FRAME / RATE, amplitude=0.5) * 32767).astype(np.int16)

    fired = [detector.process(tone) for _ in range(7)]

    assert fired[-1] is True
    assert source.frames[0].dtype == np.float32
    assert float(np.abs(source.frames[0]).max()) == pytest.approx(0.5, abs=1e-3)


# ---------------------------------------------------------------------------
# Manual, Windows host with the checkpoint downloaded
# ---------------------------------------------------------------------------


@pytest.mark.manual
def test_real_silero_model_scores_silence_low(silence: Any) -> None:
    """Run on the Windows host after ``scripts/pull_models.ps1``.

    Check: probabilities stay in range, the session actually loads from
    ``models/silero_vad.onnx``, and a second of digital silence never reads as
    speech. A failure here means the checkpoint or the frame size is wrong.
    """
    if not has_module("onnxruntime"):
        pytest.skip("onnxruntime is not installed, run this on the Windows target")

    config = load_config(None)
    vad = SileroVad(config)
    frame = silence(FRAME / RATE)

    scores = [vad.probability(frame) for _ in range(31)]

    assert vad.is_loaded
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert max(scores) < config.vad.threshold
    assert vad.frames_processed == 31
    vad.reset()
    assert vad.frames_processed == 0


@pytest.mark.manual
def test_real_endpointer_stays_quiet_through_silence(silence: Any) -> None:
    """Run on the Windows host. Two seconds of silence must open no turn."""
    if not has_module("onnxruntime"):
        pytest.skip("onnxruntime is not installed, run this on the Windows target")

    config = load_config(None)
    endpointer = Endpointer(config, sample_rate=config.audio.sample_rate)
    frame = silence(config.vad.frame_samples / config.audio.sample_rate)

    produced = [endpointer.process(frame) for _ in range(62)]

    assert all(result is None for result in produced)
    assert endpointer.state is SpeechState.SILENCE


class TestTheModelGetsItsContextWindow:
    """Silero expects the previous frame's tail prepended to the current one.

    Fed a bare frame the ONNX graph runs without complaint and returns a
    near-zero probability for everything, so speech and silence become
    indistinguishable. Measured against the shipped checkpoint, one sentence of
    clear speech scored 0.003 without the context and 1.000 with it. Nothing
    failed, nothing was logged, and the endpointer simply never fired: the
    assistant heard the wake word and then sat silent.
    """

    @staticmethod
    def _detector(cfg: JarvisConfig, session: FakeSession) -> SileroVad:
        return SileroVad(cfg, session)

    def test_the_fed_window_is_the_frame_plus_the_context(
        self, cfg: JarvisConfig
    ) -> None:
        from jarvis.audio.vad import _CONTEXT_SAMPLES

        session = FakeSession([0.9])
        detector = self._detector(cfg, session)
        context = _CONTEXT_SAMPLES[detector.sample_rate]

        detector.probability(np.zeros(detector.frame_samples, dtype=np.float32))

        fed = session.calls[0]["input"]
        assert fed.shape == (1, detector.frame_samples + context), (
            f"the model was fed {fed.shape[1]} samples, expected "
            f"{detector.frame_samples} plus {context} of context"
        )

    def test_the_first_frame_is_padded_with_silence(self, cfg: JarvisConfig) -> None:
        """There is no previous frame, so the context starts as zeros."""
        from jarvis.audio.vad import _CONTEXT_SAMPLES

        session = FakeSession([0.9])
        detector = self._detector(cfg, session)
        context = _CONTEXT_SAMPLES[detector.sample_rate]

        detector.probability(np.full(detector.frame_samples, 0.5, dtype=np.float32))

        fed = session.calls[0]["input"][0]
        assert np.all(fed[:context] == 0.0), "the leading context is not silence"
        assert np.all(fed[context:] == pytest.approx(0.5)), "the frame itself was altered"

    def test_the_context_is_the_previous_frames_tail(self, cfg: JarvisConfig) -> None:
        from jarvis.audio.vad import _CONTEXT_SAMPLES

        session = FakeSession([0.9, 0.9])
        detector = self._detector(cfg, session)
        context = _CONTEXT_SAMPLES[detector.sample_rate]

        first = np.linspace(-1.0, 1.0, detector.frame_samples, dtype=np.float32)
        second = np.full(detector.frame_samples, -0.25, dtype=np.float32)
        detector.probability(first)
        detector.probability(second)

        fed = session.calls[1]["input"][0]
        assert fed[:context] == pytest.approx(first[-context:]), (
            "the second call was not given the tail of the first frame"
        )
        assert fed[context:] == pytest.approx(second)

    def test_reset_clears_the_context(self, cfg: JarvisConfig) -> None:
        """A new utterance must not inherit audio from the previous one."""
        from jarvis.audio.vad import _CONTEXT_SAMPLES

        session = FakeSession([0.9, 0.9])
        detector = self._detector(cfg, session)
        context = _CONTEXT_SAMPLES[detector.sample_rate]

        detector.probability(np.full(detector.frame_samples, 0.7, dtype=np.float32))
        detector.reset()
        detector.probability(np.full(detector.frame_samples, 0.3, dtype=np.float32))

        fed = session.calls[1]["input"][0]
        assert np.all(fed[:context] == 0.0), "reset left the previous utterance's audio behind"

    def test_a_failed_frame_does_not_poison_the_context(self, cfg: JarvisConfig) -> None:
        """The tail is carried only after the session actually ran."""
        from jarvis.audio.vad import _CONTEXT_SAMPLES

        session = FakeSession([0.9, 0.9])
        detector = self._detector(cfg, session)
        context = _CONTEXT_SAMPLES[detector.sample_rate]

        good = np.full(detector.frame_samples, 0.6, dtype=np.float32)
        detector.probability(good)

        with pytest.raises(AudioError):
            detector.probability(np.zeros(detector.frame_samples + 1, dtype=np.float32))

        detector.probability(np.zeros(detector.frame_samples, dtype=np.float32))
        fed = session.calls[1]["input"][0]
        assert fed[:context] == pytest.approx(good[-context:]), (
            "a rejected frame disturbed the context of the next good one"
        )

    def test_eight_kilohertz_uses_a_smaller_context(self, tmp_path: Path) -> None:
        from jarvis.audio.vad import _CONTEXT_SAMPLES
        from jarvis.config import load_config

        config = load_config(
            tmp_path / "absent.yaml",
            audio={"sample_rate": 8_000},
            vad={"frame_samples": 256},
        )
        session = FakeSession([0.9])
        detector = SileroVad(config, session)
        detector.probability(np.zeros(detector.frame_samples, dtype=np.float32))

        assert session.calls[0]["input"].shape[1] == (
            detector.frame_samples + _CONTEXT_SAMPLES[8_000]
        )


# ---------------------------------------------------------------------------
# BargeInDetector echo gate
# ---------------------------------------------------------------------------


def _tone(amplitude: float, samples: int = FRAME, seed: int = 0) -> np.ndarray[Any, Any]:
    """A frame at a known RMS. Content is irrelevant: the VAD is scripted."""
    rng = np.random.default_rng(seed)
    block = rng.standard_normal(samples)
    return (block / np.sqrt(np.mean(block**2)) * amplitude).astype(np.float32)


class TestTheEchoGate:
    """Barge-in must react to the user, not to the assistant's own voice.

    The microphone hears the speakers. Silero has no opinion about who is
    talking, only whether someone is, and it scores clean synthesised speech at
    essentially 1.0 however quietly it arrives, so barge_in_threshold separates
    nothing. Measured against the real models, the assistant's own reply
    satisfied it within 544 ms at every bleed level down to -40 dB, a microphone
    RMS of 0.0007. On speakers rather than headphones that cut the reply in half
    on every single turn, which is the §7 Phase 1 gate failing.

    What separates them is loudness relative to what is being played, which is
    why the player reports its own output level and this gate compares the two.
    """

    def test_the_assistants_own_voice_never_triggers(self, cfg: JarvisConfig) -> None:
        """The reported bug: constant coupling, the model says speech throughout."""
        detector = BargeInDetector(cfg, vad=lambda _f: 1.0)
        for _ in range(200):
            assert detector.process(_tone(0.02), speaker_level=0.2) is False
        assert detector.triggered is False
        assert detector.suppressed_frames > 100

    @pytest.mark.parametrize("coupling", [1.0, 0.5, 0.1, 0.01, 0.001])
    def test_it_holds_at_every_bleed_level(self, cfg: JarvisConfig, coupling: float) -> None:
        detector = BargeInDetector(cfg, vad=lambda _f: 1.0)
        for _ in range(100):
            assert detector.process(_tone(0.3 * coupling), speaker_level=0.3) is False

    def test_a_voice_above_the_echo_still_interrupts(self, cfg: JarvisConfig) -> None:
        detector = BargeInDetector(cfg, vad=lambda _f: 1.0)
        for _ in range(40):
            detector.process(_tone(0.002), speaker_level=0.2)
        assert detector.triggered is False

        # The user, a good 20 dB over the echo.
        fired = any(detector.process(_tone(0.08), speaker_level=0.2) for _ in range(20))
        assert fired, "a voice well above the echo floor must still cut the assistant off"

    def test_the_coupling_it_learns_is_the_real_one(self, cfg: JarvisConfig) -> None:
        detector = BargeInDetector(cfg, vad=lambda _f: 1.0)
        for _ in range(30):
            detector.process(_tone(0.03), speaker_level=0.3)
        assert detector.coupling == pytest.approx(0.1, rel=0.05)

    def test_it_learns_from_frames_the_model_calls_silence(self, cfg: JarvisConfig) -> None:
        """Learning and gating have to be separate.

        A quiet echo often scores below barge_in_threshold, so an estimate that
        only updated on speech frames stayed unset through the whole quiet
        passage and then initialised from the first frame of the user's voice.
        Measured, that put it at 0.83 instead of 0.01, after which every real
        interruption looked like an echo and barge-in stopped working at all.
        """
        detector = BargeInDetector(cfg, vad=lambda _f: 0.0)
        for _ in range(20):
            detector.process(_tone(0.02), speaker_level=0.2)
        assert detector.coupling == pytest.approx(0.1, rel=0.05)

    def test_a_quiet_echo_then_a_loud_user_still_interrupts(self, cfg: JarvisConfig) -> None:
        """The regression the test above describes, end to end."""
        detector = BargeInDetector(cfg, vad=lambda _f: 0.0)
        for _ in range(30):
            detector.process(_tone(0.002), speaker_level=0.2)

        detector_speaks = BargeInDetector(cfg, vad=lambda _f: 1.0)
        detector_speaks._coupling = detector.coupling
        fired = any(detector_speaks.process(_tone(0.1), speaker_level=0.2) for _ in range(20))
        assert fired

    def test_nothing_playing_means_nothing_to_suppress(self, cfg: JarvisConfig) -> None:
        """Between chunks the speaker is silent, so any speech is the user."""
        detector = BargeInDetector(cfg, vad=lambda _f: 1.0)
        fired = any(detector.process(_tone(0.01), speaker_level=0.0) for _ in range(20))
        assert fired

    def test_without_a_reference_the_gate_does_not_engage(self, cfg: JarvisConfig) -> None:
        """Callers that pass no level get the old behaviour rather than silence."""
        detector = BargeInDetector(cfg, vad=lambda _f: 1.0)
        assert any(detector.process(zeros_frame()) for _ in range(20))
        assert detector.suppressed_frames == 0

    def test_the_first_frames_cannot_trigger(self, cfg: JarvisConfig) -> None:
        """Nothing is known about the room yet, so there is no basis to decide."""
        detector = BargeInDetector(cfg, vad=lambda _f: 1.0)
        assert detector.coupling is None
        assert detector.process(_tone(0.5), speaker_level=0.2) is False

    def test_reset_forgets_the_room(self, cfg: JarvisConfig) -> None:
        """The volume knob may have moved between replies."""
        detector = BargeInDetector(cfg, vad=lambda _f: 1.0)
        for _ in range(10):
            detector.process(_tone(0.02), speaker_level=0.2)
        assert detector.coupling is not None

        detector.reset()
        assert detector.coupling is None
        assert detector.suppressed_frames == 0

    def test_the_estimate_recovers_when_the_room_gets_louder(self, cfg: JarvisConfig) -> None:
        """A min tracker that only ever fell would deafen itself permanently."""
        detector = BargeInDetector(cfg, vad=lambda _f: 0.0)
        detector.process(_tone(0.0002), speaker_level=0.2)
        floor = detector.coupling
        assert floor is not None

        for _ in range(400):
            detector.process(_tone(0.02), speaker_level=0.2)
        assert detector.coupling is not None
        assert detector.coupling > floor * 2

    def test_the_margin_is_configurable(self, tmp_path: Path) -> None:
        wide = make_config(tmp_path, barge_in_echo_margin_db=40.0)
        detector = BargeInDetector(wide, vad=lambda _f: 1.0)
        for _ in range(20):
            detector.process(_tone(0.02), speaker_level=0.2)
        # 20 dB over the echo, which a 40 dB margin still calls an echo.
        assert not any(detector.process(_tone(0.2), speaker_level=0.2) for _ in range(20))

    def test_a_pause_in_the_reply_does_not_look_like_an_interruption(
        self, cfg: JarvisConfig
    ) -> None:
        """Both levels have to be measured over the same span.

        Speech has an envelope, and its echo arrives late: the output device
        buffers, then the room adds its own delay. Compared instant against
        instant, the microphone during the rise of a syllable is still carrying
        the quiet from before it, which drags the learned floor down by 30 dB;
        the rest of that syllable then reads as somebody interrupting. Compared
        as a peak over a matching window the ratio is flat and nothing moves.

        The shape below is an ordinary spoken phrase: about half a second of
        speech, a quarter second of pause, and 128 ms for the sound to get back
        to the microphone.
        """
        detector = BargeInDetector(cfg, vad=lambda _f: 1.0)
        burst_frames, gap_frames, delay_frames = 15, 8, 4
        loud, quiet, coupling = 0.3, 0.01, 0.1

        speaker: list[float] = []
        while len(speaker) < 120:
            speaker.extend([loud] * burst_frames + [quiet] * gap_frames)

        window = max(round(400.0 / FRAME_MS), 1)
        for index in range(120):
            # The microphone hears the speaker, scaled and late.
            echoed = speaker[max(index - delay_frames, 0)] * coupling
            recent = speaker[max(index - window + 1, 0) : index + 1]
            assert detector.process(_tone(echoed), speaker_level=max(recent)) is False

        assert detector.triggered is False
