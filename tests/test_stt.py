"""T-1.4: transcription, RTF, tier selection, and the off-thread STT worker.

Everything here runs on Linux with no GPU, no microphone, and neither
faster-whisper nor pywhispercpp installed (§0b). Both backends take an injected
model, so segment assembly, the real time factor, the short-circuit on a too
short utterance, the sample rate guard, and the CUDA library failure path are all
exercised against fakes that mimic the exact return shapes the real libraries
produce.

The one check that genuinely needs the real model and a recorded WAV is marked
``manual`` and skips itself when the model or the fixture is absent.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import types
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from jarvis.audio import stt as stt_module
from jarvis.audio.stt import (
    WHISPER_SAMPLE_RATE,
    FasterWhisperTranscriber,
    SttResult,
    SttWorker,
    Transcriber,
    Transcript,
    TranscriptSegment,
    WhisperCppTranscriber,
    build_transcriber,
    cuda_runtime_usable,
    to_float32_mono,
)
from jarvis.config import JarvisConfig, SttEngine, load_config
from jarvis.state import EventType
from jarvis.util.errors import DependencyMissingError, JarvisError, SttError
from jarvis.util.platform import has_module

CUDNN_MESSAGE = (
    "Could not locate cudnn_ops64_9.dll. Please make sure it is in your library path!"
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeSegment:
    """Mimics ``faster_whisper.transcribe.Segment``.

    Only the four attributes the transcriber reads are provided, deliberately:
    if the production code ever starts depending on another field, this fake
    stops satisfying it and the test fails rather than silently passing against
    a richer object than the real one.
    """

    def __init__(
        self,
        text: str,
        start: float = 0.0,
        end: float = 0.0,
        no_speech_prob: float = 0.0,
    ) -> None:
        self.text = text
        self.start = start
        self.end = end
        self.no_speech_prob = no_speech_prob


class FakeInfo:
    """Mimics ``faster_whisper.transcribe.TranscriptionInfo``."""

    def __init__(self, language: str = "en", duration: float = 0.0) -> None:
        self.language = language
        self.language_probability = 0.99
        self.duration = duration


class FakeWhisperModel:
    """Stand-in for ``faster_whisper.WhisperModel``.

    ``transcribe`` returns the ``(generator, info)`` pair the real library
    returns, so the production code has to drain the generator inside its timed
    section exactly as it must in reality. Every call is recorded so the tests
    can assert on the audio and the keyword arguments that reached the model.
    """

    def __init__(
        self,
        segments: list[FakeSegment] | None = None,
        *,
        info: FakeInfo | None = None,
        raises: BaseException | None = None,
    ) -> None:
        self.segments = list(segments or [])
        self.info = info or FakeInfo()
        self.raises = raises
        self.calls: list[dict[str, Any]] = []
        self.audio: list[np.ndarray[Any, Any]] = []

    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Any, FakeInfo]:
        self.audio.append(np.asarray(audio))
        self.calls.append(dict(kwargs))
        if self.raises is not None:
            raise self.raises
        return iter(self.segments), self.info


class LazyRaisingModel:
    """A model whose failure only happens once the generator is drained.

    faster-whisper decodes lazily, so a CUDA library that is missing surfaces on
    the first ``next()`` rather than from the ``transcribe`` call itself. The
    error wrapping has to survive that.
    """

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Any, FakeInfo]:
        def generate() -> Any:
            raise self.exc
            yield  # pragma: no cover - unreachable, makes this a generator

        return generate(), FakeInfo()


class FakeCppSegment:
    """Mimics a ``pywhispercpp`` segment, whose timestamps are centiseconds."""

    def __init__(self, text: str, t0: int = 0, t1: int = 0) -> None:
        self.text = text
        self.t0 = t0
        self.t1 = t1


class FakeCppModel:
    """Stand-in for ``pywhispercpp.model.Model``, which returns a plain list."""

    def __init__(
        self,
        segments: list[FakeCppSegment] | None = None,
        *,
        raises: BaseException | None = None,
    ) -> None:
        self.segments = list(segments or [])
        self.raises = raises
        self.audio: list[np.ndarray[Any, Any]] = []

    def transcribe(self, audio: Any, **kwargs: Any) -> list[FakeCppSegment]:
        self.audio.append(np.asarray(audio))
        if self.raises is not None:
            raise self.raises
        return self.segments


class FakeClock:
    """Scripted monotonic clock. The last value repeats once the script runs out."""

    def __init__(self, values: list[float]) -> None:
        self.values = list(values)
        self.index = 0

    def __call__(self) -> float:
        if self.index < len(self.values):
            value = self.values[self.index]
            self.index += 1
            return value
        return self.values[-1] if self.values else 0.0


class ScriptedTranscriber:
    """A :class:`Transcriber` for worker tests. Optionally blocks or raises."""

    def __init__(
        self,
        *,
        text: str = "hello",
        raises: BaseException | None = None,
        gate: threading.Event | None = None,
    ) -> None:
        self.text = text
        self.raises = raises
        self.gate = gate
        self.calls: list[tuple[int, int]] = []

    def transcribe(
        self,
        audio: Any,
        sample_rate: int = WHISPER_SAMPLE_RATE,
    ) -> Transcript:
        block = np.asarray(audio)
        self.calls.append((int(block.size), int(sample_rate)))
        if self.gate is not None:
            self.gate.wait(timeout=5.0)
        if self.raises is not None:
            raise self.raises
        duration = block.size / float(sample_rate)
        return Transcript(
            text=self.text,
            language="en",
            duration_s=duration,
            rtf=0.1,
            segments=(TranscriptSegment(text=self.text, start_s=0.0, end_s=duration),),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(tmp_path: Path, **overrides: Any) -> JarvisConfig:
    """A config with runtime paths in ``tmp_path`` and an explicit tier."""
    sections: dict[str, Any] = {
        "paths": {
            "data_dir": str(tmp_path / "data"),
            "log_dir": str(tmp_path / "logs"),
            "models_dir": str(tmp_path / "models"),
            "vendor_dir": str(tmp_path / "vendor"),
        },
        "hardware": {"tier": "cpu"},
    }
    for key, value in overrides.items():
        if key in sections and isinstance(value, dict):
            sections[key] = {**sections[key], **value}
        else:
            sections[key] = value
    return load_config(tmp_path / "absent.yaml", **sections)


def gpu_config(tmp_path: Path, **overrides: Any) -> JarvisConfig:
    """A config pinned to the gpu-12 tier, which selects faster-whisper."""
    return make_config(tmp_path, hardware={"tier": "gpu-12"}, **overrides)


def speech(seconds: float, sample_rate: int = WHISPER_SAMPLE_RATE) -> np.ndarray[Any, Any]:
    """Float32 mono audio of the requested length. Content does not matter here."""
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


def wait_for(predicate: Any, timeout: float = 5.0) -> bool:
    """Poll ``predicate`` until it is true or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


# ---------------------------------------------------------------------------
# Input conditioning
# ---------------------------------------------------------------------------


def test_int16_input_is_scaled_into_the_unit_range() -> None:
    block = np.array([-32768, -16384, 0, 16384, 32767], dtype=np.int16)
    out = to_float32_mono(block)
    assert out.dtype == np.float32
    assert out[0] == pytest.approx(-1.0)
    assert out[1] == pytest.approx(-0.5)
    assert out[2] == pytest.approx(0.0)
    assert out[3] == pytest.approx(0.5)
    assert out[4] == pytest.approx(1.0, abs=1e-4)


def test_int32_input_is_scaled_by_its_own_range() -> None:
    info = np.iinfo(np.int32)
    block = np.array([info.min, 0, info.max], dtype=np.int32)
    out = to_float32_mono(block)
    assert out[0] == pytest.approx(-1.0)
    assert out[2] == pytest.approx(1.0, abs=1e-6)


def test_float64_input_is_narrowed_without_rescaling() -> None:
    block = np.array([0.25, -0.5], dtype=np.float64)
    out = to_float32_mono(block)
    assert out.dtype == np.float32
    assert out.tolist() == pytest.approx([0.25, -0.5])


def test_stereo_input_is_downmixed_not_truncated() -> None:
    block = np.array([[0.0, 1.0], [0.0, -1.0]], dtype=np.float32)
    out = to_float32_mono(block)
    # A right-channel-only mic must not transcribe as silence.
    assert out.shape == (2,)
    assert out.tolist() == pytest.approx([0.5, -0.5])


def test_single_channel_2d_block_is_flattened() -> None:
    block = np.zeros((512, 1), dtype=np.float32)
    assert to_float32_mono(block).shape == (512,)


def test_three_dimensional_audio_is_refused() -> None:
    with pytest.raises(SttError, match="1-D or 2-D"):
        to_float32_mono(np.zeros((2, 2, 2), dtype=np.float32))


def test_unsigned_pcm_is_refused_rather_than_misread() -> None:
    with pytest.raises(SttError, match="unsigned"):
        to_float32_mono(np.array([0, 128, 255], dtype=np.uint8))


def test_non_numeric_audio_is_refused() -> None:
    with pytest.raises(SttError, match="not numeric"):
        to_float32_mono(np.array(["a", "b"]))


def test_non_finite_samples_become_silence() -> None:
    block = np.array([0.1, np.nan, np.inf, -np.inf], dtype=np.float32)
    out = to_float32_mono(block)
    assert np.isfinite(out).all()
    assert out.tolist() == pytest.approx([0.1, 0.0, 1.0, -1.0])


def test_hot_float_input_is_clipped_into_range() -> None:
    out = to_float32_mono(np.array([4.0, -9.0], dtype=np.float32))
    assert out.tolist() == pytest.approx([1.0, -1.0])


def test_output_is_contiguous_for_the_backend() -> None:
    block = np.zeros((100, 2), dtype=np.float32)
    assert to_float32_mono(block).flags["C_CONTIGUOUS"]


# ---------------------------------------------------------------------------
# faster-whisper: assembly, timing, guards
# ---------------------------------------------------------------------------


def test_text_is_assembled_from_every_segment(tmp_path: Path) -> None:
    model = FakeWhisperModel(
        [
            # faster-whisper emits a leading space on each segment.
            FakeSegment(" what is", 0.0, 0.8, no_speech_prob=0.02),
            FakeSegment(" the CPU", 0.8, 1.6, no_speech_prob=0.04),
            FakeSegment(" doing", 1.6, 2.0, no_speech_prob=0.06),
        ],
        info=FakeInfo(language="en"),
    )
    transcriber = FasterWhisperTranscriber(gpu_config(tmp_path), model)

    result = transcriber.transcribe(speech(2.0), WHISPER_SAMPLE_RATE)

    assert result.text == "what is the CPU doing"
    assert len(result.segments) == 3
    assert result.segments[0] == TranscriptSegment(text="what is", start_s=0.0, end_s=0.8)
    assert result.segments[-1].end_s == pytest.approx(2.0)
    assert result.language == "en"
    assert result.no_speech_prob == pytest.approx(0.04)
    assert not result.is_empty


def test_blank_segments_are_dropped_from_the_text(tmp_path: Path) -> None:
    model = FakeWhisperModel([FakeSegment(" hello", 0.0, 0.5), FakeSegment("   ", 0.5, 0.6)])
    transcriber = FasterWhisperTranscriber(gpu_config(tmp_path), model)

    result = transcriber.transcribe(speech(1.0))

    assert result.text == "hello"
    assert len(result.segments) == 1


def test_rtf_is_processing_time_over_audio_time(tmp_path: Path) -> None:
    # Two seconds of audio transcribed in half a second is an RTF of 0.25.
    clock = FakeClock([10.0, 10.5])
    model = FakeWhisperModel([FakeSegment(" done", 0.0, 2.0)])
    transcriber = FasterWhisperTranscriber(gpu_config(tmp_path), model, clock=clock)

    result = transcriber.transcribe(speech(2.0))

    assert result.duration_s == pytest.approx(2.0)
    assert result.rtf == pytest.approx(0.25)
    assert result.processing_s == pytest.approx(0.5)


def test_rtf_above_one_means_slower_than_real_time(tmp_path: Path) -> None:
    clock = FakeClock([0.0, 3.0])
    transcriber = FasterWhisperTranscriber(
        gpu_config(tmp_path), FakeWhisperModel([FakeSegment(" slow", 0.0, 1.0)]), clock=clock
    )

    result = transcriber.transcribe(speech(1.0))

    assert result.rtf == pytest.approx(3.0)


def test_rtf_is_logged_for_every_utterance(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """T-1.4 requires the real time factor in the log, once per utterance."""
    transcriber = FasterWhisperTranscriber(
        gpu_config(tmp_path),
        FakeWhisperModel([FakeSegment(" logged", 0.0, 1.0)]),
        clock=FakeClock([0.0, 0.4]),
    )
    with caplog.at_level("INFO", logger="jarvis.audio.stt"):
        transcriber.transcribe(speech(1.0))

    records = [r for r in caplog.records if r.getMessage() == "transcribed utterance"]
    assert len(records) == 1
    context = records[0].context  # type: ignore[attr-defined]
    assert context["rtf"] == pytest.approx(0.4)
    assert context["duration_s"] == pytest.approx(1.0)
    assert context["engine"] == "faster-whisper"
    assert context["model"] == "distil-large-v3"


def test_audio_shorter_than_min_audio_ms_never_reaches_the_model(tmp_path: Path) -> None:
    """An accidental wake trigger must cost nothing (§7 T-1.4)."""
    model = FakeWhisperModel([FakeSegment(" thank you", 0.0, 0.1)])
    config = gpu_config(tmp_path, stt={"min_audio_ms": 200})
    transcriber = FasterWhisperTranscriber(config, model)

    result = transcriber.transcribe(speech(0.1))

    assert model.calls == []
    assert result.text == ""
    assert result.is_empty
    assert result.rtf == 0.0
    assert result.duration_s == pytest.approx(0.1)
    assert result.no_speech_prob == 1.0
    assert transcriber.utterances == 0
    assert transcriber.empty_results == 1


def test_the_short_circuit_does_not_load_a_model_at_all(tmp_path: Path) -> None:
    """No model injected and none installed, yet a short buffer still returns."""
    config = gpu_config(tmp_path, stt={"min_audio_ms": 500})
    transcriber = FasterWhisperTranscriber(config)

    result = transcriber.transcribe(speech(0.2))

    assert result.is_empty
    assert not transcriber.is_loaded


def test_audio_at_exactly_the_minimum_is_transcribed(tmp_path: Path) -> None:
    model = FakeWhisperModel([FakeSegment(" yes", 0.0, 0.2)])
    config = gpu_config(tmp_path, stt={"min_audio_ms": 200})
    transcriber = FasterWhisperTranscriber(config, model)

    result = transcriber.transcribe(speech(0.2))

    assert len(model.calls) == 1
    assert result.text == "yes"


def test_int16_capture_reaches_the_model_as_unit_range_float32(tmp_path: Path) -> None:
    model = FakeWhisperModel([FakeSegment(" converted", 0.0, 1.0)])
    transcriber = FasterWhisperTranscriber(gpu_config(tmp_path), model)
    block = np.full(WHISPER_SAMPLE_RATE, 16384, dtype=np.int16)

    result = transcriber.transcribe(block, WHISPER_SAMPLE_RATE)

    assert result.text == "converted"
    seen = model.audio[0]
    assert seen.dtype == np.float32
    assert seen.ndim == 1
    assert float(seen.max()) == pytest.approx(0.5)
    assert float(seen.min()) == pytest.approx(0.5)
    # Duration comes from the sample count, not the dtype.
    assert result.duration_s == pytest.approx(1.0)


def test_a_sample_rate_the_model_does_not_expect_raises(tmp_path: Path) -> None:
    model = FakeWhisperModel([FakeSegment(" nope", 0.0, 1.0)])
    transcriber = FasterWhisperTranscriber(gpu_config(tmp_path), model)

    with pytest.raises(SttError) as excinfo:
        transcriber.transcribe(np.zeros(8_000, dtype=np.float32), 8_000)

    assert excinfo.value.context["given_sample_rate"] == 8_000
    assert excinfo.value.context["expected_sample_rate"] == WHISPER_SAMPLE_RATE
    assert "resampling is not performed" in excinfo.value.message
    assert excinfo.value.speakable
    assert model.calls == [], "the model must not run on wrongly rated audio"


def test_a_higher_sample_rate_also_raises(tmp_path: Path) -> None:
    transcriber = FasterWhisperTranscriber(gpu_config(tmp_path), FakeWhisperModel())
    with pytest.raises(SttError):
        transcriber.transcribe(np.zeros(48_000, dtype=np.float32), 48_000)


def test_backend_exception_becomes_an_stt_error(tmp_path: Path) -> None:
    model = FakeWhisperModel(raises=ValueError("ct2 exploded"))
    transcriber = FasterWhisperTranscriber(gpu_config(tmp_path), model)

    with pytest.raises(SttError) as excinfo:
        transcriber.transcribe(speech(1.0))

    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "ct2 exploded" in excinfo.value.message
    assert excinfo.value.speakable == "I did not catch that."
    assert excinfo.value.context["model"] == "distil-large-v3"


def test_a_failure_while_draining_the_generator_is_also_wrapped(tmp_path: Path) -> None:
    transcriber = FasterWhisperTranscriber(
        gpu_config(tmp_path), LazyRaisingModel(RuntimeError("decode failed"))
    )
    with pytest.raises(SttError, match="decode failed"):
        transcriber.transcribe(speech(1.0))


def test_a_keyboard_interrupt_is_not_swallowed(tmp_path: Path) -> None:
    """Only real failures are wrapped. Ctrl-C must still stop the process."""
    transcriber = FasterWhisperTranscriber(
        gpu_config(tmp_path), FakeWhisperModel(raises=KeyboardInterrupt())
    )
    with pytest.raises(KeyboardInterrupt):
        transcriber.transcribe(speech(1.0))


def test_config_options_are_handed_to_the_backend(tmp_path: Path) -> None:
    config = gpu_config(
        tmp_path,
        stt={
            "language": "en",
            "beam_size": 5,
            "vad_filter": True,
            "condition_on_previous_text": True,
        },
    )
    model = FakeWhisperModel([FakeSegment(" ok", 0.0, 1.0)])
    FasterWhisperTranscriber(config, model).transcribe(speech(1.0))

    call = model.calls[0]
    assert call == {
        "language": "en",
        "beam_size": 5,
        "vad_filter": True,
        "condition_on_previous_text": True,
    }


def test_language_auto_is_passed_as_none_so_the_model_detects(tmp_path: Path) -> None:
    config = gpu_config(tmp_path, stt={"language": "auto"})
    model = FakeWhisperModel([FakeSegment(" bonjour", 0.0, 1.0)], info=FakeInfo(language="fr"))
    transcriber = FasterWhisperTranscriber(config, model)

    result = transcriber.transcribe(speech(1.0))

    assert transcriber.language is None
    assert model.calls[0]["language"] is None
    assert result.language == "fr"


def test_model_device_and_compute_type_come_from_the_tier(tmp_path: Path) -> None:
    transcriber = FasterWhisperTranscriber(gpu_config(tmp_path))
    assert transcriber.model_name == "distil-large-v3"
    assert transcriber.device == "cuda"
    assert transcriber.compute_type == "int8"
    assert transcriber.expected_sample_rate == WHISPER_SAMPLE_RATE


def test_explicit_config_overrides_beat_the_tier_profile(tmp_path: Path) -> None:
    config = gpu_config(
        tmp_path, stt={"model": "small.en", "device": "cpu", "compute_type": "float32"}
    )
    transcriber = FasterWhisperTranscriber(config)
    assert (transcriber.model_name, transcriber.device, transcriber.compute_type) == (
        "small.en",
        "cpu",
        "float32",
    )


def test_unload_drops_the_model_so_vram_comes_back(tmp_path: Path) -> None:
    transcriber = FasterWhisperTranscriber(gpu_config(tmp_path), FakeWhisperModel())
    assert transcriber.is_loaded
    transcriber.unload()
    assert not transcriber.is_loaded


def test_warmup_runs_a_buffer_through_the_model(tmp_path: Path) -> None:
    model = FakeWhisperModel([FakeSegment(" ignored", 0.0, 0.5)])
    transcriber = FasterWhisperTranscriber(gpu_config(tmp_path), model)

    assert transcriber.warmup() is True
    assert len(model.calls) == 1
    # Warmup must not be counted as a user utterance.
    assert transcriber.utterances == 0


def test_warmup_failure_is_swallowed_not_raised(tmp_path: Path) -> None:
    transcriber = FasterWhisperTranscriber(
        gpu_config(tmp_path), FakeWhisperModel(raises=OSError(CUDNN_MESSAGE))
    )
    assert transcriber.warmup() is False


# ---------------------------------------------------------------------------
# The CUDA library failure path
# ---------------------------------------------------------------------------


def test_missing_cudnn_dll_during_inference_is_explained(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    transcriber = FasterWhisperTranscriber(
        gpu_config(tmp_path), FakeWhisperModel(raises=OSError(CUDNN_MESSAGE))
    )
    with caplog.at_level("ERROR", logger="jarvis.audio.stt"), pytest.raises(SttError) as excinfo:
        transcriber.transcribe(speech(1.0))

    error = excinfo.value
    assert "CUDA support library is missing" in error.message
    assert error.speakable == "My speech recogniser cannot reach the graphics card."
    assert "cudnn_ops64_9.dll" not in error.speakable, "a DLL name must never be spoken"
    remediation = error.context["remediation"]
    assert "nvidia-cudnn-cu12" in remediation
    assert "stt.device to cpu" in remediation
    assert any("cudnn" in record.getMessage() for record in caplog.records)


def test_missing_cublas_at_model_load_is_explained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same failure can happen while constructing the model, not just calling it."""

    def exploding_model(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("Error loading cublas64_12.dll: The specified module could not be found.")

    fake_module = types.SimpleNamespace(WhisperModel=exploding_model)
    monkeypatch.setattr(stt_module, "require_module", lambda _name, **_kw: fake_module)
    transcriber = FasterWhisperTranscriber(gpu_config(tmp_path))

    with pytest.raises(SttError) as excinfo:
        transcriber.transcribe(speech(1.0))

    assert excinfo.value.context["remediation"]
    assert excinfo.value.speakable == "My speech recogniser cannot reach the graphics card."
    assert not transcriber.is_loaded, "a failed load must not be cached"


def test_an_ordinary_os_error_is_not_reported_as_a_cuda_problem(tmp_path: Path) -> None:
    transcriber = FasterWhisperTranscriber(
        gpu_config(tmp_path), FakeWhisperModel(raises=OSError("disk is full"))
    )
    with pytest.raises(SttError) as excinfo:
        transcriber.transcribe(speech(1.0))

    assert "remediation" not in excinfo.value.context
    assert excinfo.value.speakable == "I did not catch that."


# ---------------------------------------------------------------------------
# Missing dependencies
# ---------------------------------------------------------------------------


def test_missing_faster_whisper_raises_with_an_install_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(name: str, **_kwargs: Any) -> Any:
        raise DependencyMissingError(name, extra="stt")

    monkeypatch.setattr(stt_module, "require_module", refuse)
    transcriber = FasterWhisperTranscriber(gpu_config(tmp_path))

    with pytest.raises(DependencyMissingError) as excinfo:
        transcriber.transcribe(speech(1.0))

    # The install hint must survive; wrapping it in SttError would lose it.
    assert "uv sync --extra stt" in excinfo.value.message


def test_a_backend_without_the_expected_class_is_an_stt_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        stt_module, "require_module", lambda name, **_kw: types.SimpleNamespace()
    )
    transcriber = FasterWhisperTranscriber(gpu_config(tmp_path))

    with pytest.raises(SttError, match="no WhisperModel class"):
        transcriber.transcribe(speech(1.0))


def test_the_module_imports_without_any_stt_backend_installed() -> None:
    """§0b: nothing heavy may be imported at module scope."""
    assert "faster_whisper" not in sys.modules
    assert "pywhispercpp" not in sys.modules


# ---------------------------------------------------------------------------
# whisper.cpp
# ---------------------------------------------------------------------------


def test_whispercpp_scales_centisecond_timestamps(tmp_path: Path) -> None:
    model = FakeCppModel([FakeCppSegment(" first", 0, 120), FakeCppSegment(" second", 120, 250)])
    transcriber = WhisperCppTranscriber(make_config(tmp_path), model)

    result = transcriber.transcribe(speech(2.5))

    assert result.text == "first second"
    assert result.segments[0].start_s == pytest.approx(0.0)
    assert result.segments[0].end_s == pytest.approx(1.2)
    assert result.segments[1].start_s == pytest.approx(1.2)
    assert result.segments[1].end_s == pytest.approx(2.5)


def test_whispercpp_uses_the_cpu_tier_model_and_reports_cpu(tmp_path: Path) -> None:
    transcriber = WhisperCppTranscriber(make_config(tmp_path))
    assert transcriber.model_name == "base.en"
    assert transcriber.device == "cpu"


def test_whispercpp_maps_a_checkpoint_it_cannot_load(tmp_path: Path) -> None:
    """distil-large-v3 exists only as CTranslate2 weights, never as ggml."""
    config = make_config(tmp_path, stt={"model": "distil-large-v3"})
    assert WhisperCppTranscriber(config).model_name == "small.en"


def test_whispercpp_falls_back_to_base_en_for_an_unknown_model(tmp_path: Path) -> None:
    config = make_config(tmp_path, stt={"model": "not-a-real-checkpoint"})
    assert WhisperCppTranscriber(config).model_name == "base.en"


def test_whispercpp_keeps_an_explicit_ggml_path(tmp_path: Path) -> None:
    config = make_config(tmp_path, stt={"model": "models/ggml-base.en.bin"})
    assert WhisperCppTranscriber(config).model_name == "models/ggml-base.en.bin"


def test_whispercpp_short_circuits_and_wraps_like_the_other_backend(tmp_path: Path) -> None:
    config = make_config(tmp_path, stt={"min_audio_ms": 300})
    model = FakeCppModel([FakeCppSegment(" hi", 0, 10)])
    transcriber = WhisperCppTranscriber(config, model)

    assert transcriber.transcribe(speech(0.1)).is_empty
    assert model.audio == []

    failing = WhisperCppTranscriber(config, FakeCppModel(raises=RuntimeError("ggml failed")))
    with pytest.raises(SttError, match="ggml failed"):
        failing.transcribe(speech(1.0))


def test_whispercpp_reports_the_configured_language(tmp_path: Path) -> None:
    config = make_config(tmp_path, stt={"language": "en"})
    transcriber = WhisperCppTranscriber(config, FakeCppModel([FakeCppSegment(" yes", 0, 50)]))
    assert transcriber.transcribe(speech(1.0)).language == "en"


def test_missing_pywhispercpp_raises_with_an_install_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(name: str, **_kwargs: Any) -> Any:
        raise DependencyMissingError(name, extra="stt")

    monkeypatch.setattr(stt_module, "require_module", refuse)
    with pytest.raises(DependencyMissingError, match="pywhispercpp"):
        WhisperCppTranscriber(make_config(tmp_path)).transcribe(speech(1.0))


# ---------------------------------------------------------------------------
# build_transcriber
# ---------------------------------------------------------------------------


def test_cpu_tier_selects_whispercpp(tmp_path: Path) -> None:
    transcriber = build_transcriber(make_config(tmp_path))
    assert isinstance(transcriber, WhisperCppTranscriber)
    assert transcriber.model_name == "base.en"


def test_gpu_12_tier_selects_faster_whisper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(stt_module, "has_module", lambda name: name == "faster_whisper")
    transcriber = build_transcriber(gpu_config(tmp_path))
    assert isinstance(transcriber, FasterWhisperTranscriber)
    assert transcriber.model_name == "distil-large-v3"
    assert transcriber.device == "cuda"


def test_every_gpu_tier_selects_faster_whisper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(stt_module, "has_module", lambda name: name == "faster_whisper")
    for tier in ("gpu-6", "gpu-8", "gpu-12", "gpu-16", "gpu-24"):
        config = make_config(tmp_path, hardware={"tier": tier})
        assert isinstance(build_transcriber(config), FasterWhisperTranscriber), tier


def test_an_explicit_engine_overrides_the_tier(tmp_path: Path) -> None:
    config = gpu_config(tmp_path, stt={"engine": "whispercpp"})
    assert config.stt_settings()[0] is SttEngine.WHISPERCPP
    assert isinstance(build_transcriber(config), WhisperCppTranscriber)


def test_missing_faster_whisper_falls_back_to_whispercpp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(stt_module, "has_module", lambda _name: False)

    with caplog.at_level("WARNING", logger="jarvis.audio.stt"):
        transcriber = build_transcriber(gpu_config(tmp_path))

    assert isinstance(transcriber, WhisperCppTranscriber)
    # The gpu-12 checkpoint has no ggml build, so the fallback must remap it.
    assert transcriber.model_name == "small.en"
    messages = [record.getMessage() for record in caplog.records]
    assert any("falling back to whisper.cpp" in message for message in messages)


def test_an_injected_model_disables_the_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller holding a working model clearly has the engine it asked for."""
    monkeypatch.setattr(stt_module, "has_module", lambda _name: False)
    transcriber = build_transcriber(gpu_config(tmp_path), model=FakeWhisperModel())
    assert isinstance(transcriber, FasterWhisperTranscriber)


def test_a_built_transcriber_satisfies_the_protocol(tmp_path: Path) -> None:
    assert isinstance(build_transcriber(make_config(tmp_path)), Transcriber)


# ---------------------------------------------------------------------------
# SttWorker
# ---------------------------------------------------------------------------


def test_worker_transcribes_off_the_calling_thread() -> None:
    transcriber = ScriptedTranscriber(text="what is the cpu doing")
    with SttWorker(transcriber) as worker:
        assert worker.submit(speech(1.0), turn_id="turn-1") is True
        result = worker.next_result(timeout=5.0)

    assert result is not None
    assert result.ok
    assert result.text == "what is the cpu doing"
    assert result.turn_id == "turn-1"
    assert result.audio_seconds == pytest.approx(1.0)
    assert result.elapsed_ms >= 0.0
    assert result.queued_ms >= 0.0
    assert transcriber.calls == [(WHISPER_SAMPLE_RATE, WHISPER_SAMPLE_RATE)]


def test_worker_submit_never_blocks_when_the_queue_is_full() -> None:
    """Capture must not stall behind a slow model (§5)."""
    worker = SttWorker(ScriptedTranscriber(), maxsize=1)  # not started, so nothing drains
    assert worker.submit(speech(0.5)) is True
    assert worker.submit(speech(0.5)) is False
    assert worker.dropped == 1
    assert worker.submitted == 1
    assert worker.pending == 1


def test_worker_keeps_running_after_a_dropped_submission() -> None:
    gate = threading.Event()
    transcriber = ScriptedTranscriber(gate=gate)
    with SttWorker(transcriber, maxsize=1) as worker:
        assert worker.submit(speech(0.5), turn_id="a") is True
        assert wait_for(lambda: len(transcriber.calls) == 1)
        # One job is in flight and one more fits the queue; a third is refused.
        worker.submit(speech(0.5), turn_id="b")
        dropped = not worker.submit(speech(0.5), turn_id="c")
        gate.set()
        assert worker.next_result(timeout=5.0) is not None

    assert dropped
    assert worker.dropped == 1


def test_worker_wraps_a_transcriber_failure_and_survives() -> None:
    transcriber = ScriptedTranscriber(raises=SttError("model died"))
    with SttWorker(transcriber) as worker:
        worker.submit(speech(1.0), turn_id="bad")
        first = worker.next_result(timeout=5.0)
        transcriber.raises = None
        worker.submit(speech(1.0), turn_id="good")
        second = worker.next_result(timeout=5.0)

    assert first is not None and not first.ok
    assert isinstance(first.error, SttError)
    assert first.error.speakable
    assert first.text == ""
    assert second is not None and second.ok, "the worker must survive a failed job"
    assert worker.failed == 1
    assert worker.completed == 2


def test_worker_contains_an_unwrapped_exception() -> None:
    """A transcriber that leaks a raw exception must not kill the thread."""
    with SttWorker(ScriptedTranscriber(raises=ValueError("raw"))) as worker:
        worker.submit(speech(1.0))
        result = worker.next_result(timeout=5.0)
        assert worker.is_running

    assert result is not None
    assert isinstance(result.error, SttError)
    assert isinstance(result.error, JarvisError)
    assert "unwrapped ValueError" in result.error.message


def test_worker_publishes_transcripts_on_the_bus(bus: Any) -> None:
    received, unsubscribe = bus.subscribe_queue([EventType.TRANSCRIPT])
    try:
        with SttWorker(ScriptedTranscriber(text="hello sir"), bus) as worker:
            worker.submit(speech(1.0), turn_id="t7")
            assert worker.next_result(timeout=5.0) is not None
        event = received.get(timeout=5.0)
    finally:
        unsubscribe()

    assert event.type is EventType.TRANSCRIPT
    assert event.payload["text"] == "hello sir"
    assert event.payload["turn_id"] == "t7"
    assert event.payload["rtf"] == pytest.approx(0.1)


def test_worker_publishes_errors_on_the_bus(bus: Any) -> None:
    received, unsubscribe = bus.subscribe_queue([EventType.ERROR])
    try:
        with SttWorker(ScriptedTranscriber(raises=SttError("nope")), bus) as worker:
            worker.submit(speech(1.0))
            assert worker.next_result(timeout=5.0) is not None
        event = received.get(timeout=5.0)
    finally:
        unsubscribe()

    assert event.payload["source"] == "stt"
    assert event.payload["error"] == "SttError"
    assert event.payload["speakable"]


def test_worker_invokes_result_callbacks() -> None:
    seen: list[SttResult] = []
    done = threading.Event()

    def record(result: SttResult) -> None:
        seen.append(result)
        done.set()

    with SttWorker(ScriptedTranscriber(), on_result=record) as worker:
        worker.submit(speech(1.0))
        assert done.wait(timeout=5.0)

    assert len(seen) == 1
    assert seen[0].text == "hello"


def test_a_raising_callback_does_not_stop_the_worker() -> None:
    good: list[SttResult] = []

    def explode(_result: SttResult) -> None:
        raise RuntimeError("bad consumer")

    with SttWorker(ScriptedTranscriber(), on_result=explode) as worker:
        worker.add_listener(good.append)
        worker.submit(speech(1.0))
        assert worker.next_result(timeout=5.0) is not None
        assert wait_for(lambda: len(good) == 1)
        assert worker.is_running


def test_add_listener_returns_a_working_remover() -> None:
    seen: list[SttResult] = []
    worker = SttWorker(ScriptedTranscriber())
    remove = worker.add_listener(seen.append)
    remove()
    with worker:
        worker.submit(speech(1.0))
        assert worker.next_result(timeout=5.0) is not None
    assert seen == []


def test_results_queue_is_exposed_for_the_orchestrator() -> None:
    with SttWorker(ScriptedTranscriber()) as worker:
        results = worker.results()
        assert isinstance(results, queue.Queue)
        worker.submit(speech(1.0))
        result = results.get(timeout=5.0)
    assert result.ok


def test_next_result_times_out_without_work() -> None:
    with SttWorker(ScriptedTranscriber()) as worker:
        assert worker.next_result(timeout=0.05) is None


def test_worker_start_and_stop_are_idempotent() -> None:
    worker = SttWorker(ScriptedTranscriber())
    worker.stop()  # never started
    worker.start()
    worker.start()
    assert worker.is_running
    worker.stop()
    worker.stop()
    assert not worker.is_running


def test_worker_stops_promptly_when_idle() -> None:
    worker = SttWorker(ScriptedTranscriber())
    worker.start()
    started = time.monotonic()
    worker.stop(timeout=5.0)
    assert time.monotonic() - started < 2.0
    assert not worker.is_running


def test_worker_honours_a_per_submission_sample_rate() -> None:
    transcriber = ScriptedTranscriber()
    with SttWorker(transcriber, sample_rate=WHISPER_SAMPLE_RATE) as worker:
        worker.submit(speech(1.0), 8_000)
        assert worker.next_result(timeout=5.0) is not None
    assert transcriber.calls[0][1] == 8_000


def test_transcribe_now_runs_on_the_calling_thread() -> None:
    transcriber = ScriptedTranscriber(text="inline")
    worker = SttWorker(transcriber)
    assert worker.transcribe_now(speech(1.0)).text == "inline"
    assert not worker.is_running


def test_worker_drives_a_real_transcriber_end_to_end(tmp_path: Path) -> None:
    """The worker and the faster-whisper path, wired together as in the loop."""
    model = FakeWhisperModel([FakeSegment(" system is idle", 0.0, 1.0)])
    transcriber = FasterWhisperTranscriber(
        gpu_config(tmp_path), model, clock=FakeClock([0.0, 0.2])
    )
    with SttWorker(transcriber) as worker:
        worker.submit(np.full(WHISPER_SAMPLE_RATE, 1000, dtype=np.int16), turn_id="t1")
        result = worker.next_result(timeout=5.0)

    assert result is not None and result.transcript is not None
    assert result.transcript.text == "system is idle"
    assert result.transcript.rtf == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Transcript value object
# ---------------------------------------------------------------------------


def test_transcript_is_frozen_and_serialisable() -> None:
    transcript = Transcript(
        text="hello",
        language="en",
        duration_s=1.25,
        rtf=0.123456,
        segments=(TranscriptSegment(text="hello", start_s=0.0, end_s=1.25),),
        no_speech_prob=0.5,
    )
    with pytest.raises(AttributeError):
        transcript.text = "changed"  # type: ignore[misc]
    payload = transcript.to_dict()
    assert payload == {
        "text": "hello",
        "language": "en",
        "duration_s": 1.25,
        "rtf": 0.123,
        "segments": 1,
        "no_speech_prob": 0.5,
    }
    assert transcript.segments[0].duration_s == pytest.approx(1.25)


def test_segment_duration_never_goes_negative() -> None:
    assert TranscriptSegment(text="x", start_s=2.0, end_s=1.0).duration_s == 0.0


# ---------------------------------------------------------------------------
# Manual: real model, real audio (NEEDS_MANUAL_VERIFY)
# ---------------------------------------------------------------------------

FIXTURE_WAV = Path(__file__).parent / "fixtures" / "stt_sample.wav"
FIXTURE_TRANSCRIPT = "the quick brown fox jumps over the lazy dog"


def say(message: str) -> None:
    """Write to the terminal without tripping the no-print rule."""
    sys.stdout.write(f"\n{message}\n")
    sys.stdout.flush()


def _word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein distance over words, divided by the reference word count."""
    ref = reference.lower().replace(".", "").replace(",", "").split()
    hyp = hypothesis.lower().replace(".", "").replace(",", "").split()
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1] / len(ref)


def test_the_word_error_rate_helper_is_correct() -> None:
    assert _word_error_rate("a b c", "a b c") == 0.0
    assert _word_error_rate("a b c", "a x c") == pytest.approx(1 / 3)
    assert _word_error_rate("a b c", "") == 1.0


@pytest.mark.manual
def test_real_model_transcribes_the_fixture_wav_within_budget() -> None:
    """WER and RTF sanity check against the real model (§7 T-1.4).

    NEEDS_MANUAL_VERIFY on the Windows 11 target host, because it needs the
    faster-whisper package, a downloaded checkpoint, and a recorded WAV, none of
    which exist on the Linux build host.

    Procedure:

    1. ``uv sync --extra stt`` and ``powershell scripts/pull_models.ps1`` so the
       tier checkpoint is cached under ``models/``.
    2. Record yourself saying, in a normal speaking voice::

           the quick brown fox jumps over the lazy dog

       Save it as 16 kHz mono 16-bit PCM at ``tests/fixtures/stt_sample.wav``.
       With ffmpeg on PATH::

           ffmpeg -i raw.wav -ac 1 -ar 16000 -sample_fmt s16 tests/fixtures/stt_sample.wav

    3. Run exactly::

           uv run pytest tests/test_stt.py -m manual -k word_error -s

    The test passes at a word error rate at or below 0.2 and a real time factor
    below 1.0. A WER above that on clean speech means the wrong checkpoint was
    downloaded. An RTF at or above 1.0 means this tier cannot keep up with
    speech, so the 200 ms STT budget in §3 is unreachable and the tier needs
    dropping a step in ``config/config.yaml``.
    """
    if not has_module("faster_whisper"):
        pytest.skip("faster-whisper is not installed; run 'uv sync --extra stt'")
    if not FIXTURE_WAV.is_file():
        pytest.skip(f"record {FIXTURE_WAV} first, see this test's docstring")

    with wave.open(str(FIXTURE_WAV), "rb") as handle:
        assert handle.getnchannels() == 1, "the fixture must be mono"
        assert handle.getframerate() == WHISPER_SAMPLE_RATE, "the fixture must be 16 kHz"
        assert handle.getsampwidth() == 2, "the fixture must be 16-bit PCM"
        frames = handle.readframes(handle.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16)

    config = load_config(None)
    transcriber = build_transcriber(config)
    transcriber.transcribe(np.zeros(WHISPER_SAMPLE_RATE, dtype=np.float32))  # pay the load cost
    result = transcriber.transcribe(audio, WHISPER_SAMPLE_RATE)

    wer = _word_error_rate(FIXTURE_TRANSCRIPT, result.text)
    say(f"transcript: {result.text!r}\nWER: {wer:.3f}  RTF: {result.rtf:.3f}")
    assert wer <= 0.2, f"word error rate {wer:.3f} is too high for clean speech"
    assert result.rtf < 1.0, f"RTF {result.rtf:.3f} means transcription trails real time"


# ---------------------------------------------------------------------------
# Falling back to the CPU when CUDA turns out to be unusable
# ---------------------------------------------------------------------------


class _CudaOnlyFailure:
    """Fails on cuda, works on cpu, like a machine missing cuBLAS.

    nvidia-smi seeing a card is not the same as ctranslate2 being able to use
    it: cuBLAS and cuDNN 9 also have to be on the library path, and their
    absence only surfaces at the first transcription.
    """

    def __init__(self, device: str) -> None:
        self.device = device

    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Any, FakeInfo]:
        if self.device == "cuda":
            raise OSError(CUDNN_MESSAGE)
        return iter([FakeSegment("the processor is fine", 0.0, 1.0)]), FakeInfo()


def _cuda_transcriber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[FasterWhisperTranscriber, list[str]]:
    """A transcriber that owns its model and loads per the current device."""
    transcriber = FasterWhisperTranscriber(gpu_config(tmp_path))
    loaded: list[str] = []

    def _load(self: FasterWhisperTranscriber = transcriber) -> Any:
        loaded.append(self._device)
        return _CudaOnlyFailure(self._device)

    monkeypatch.setattr(transcriber, "_load_model", _load)
    return transcriber, loaded


def test_a_missing_cuda_library_falls_back_to_the_processor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Failing identically on every turn would make the assistant deaf."""
    transcriber, loaded = _cuda_transcriber(tmp_path, monkeypatch)

    with caplog.at_level("WARNING", logger="jarvis.audio.stt"):
        transcript = transcriber.transcribe(speech(1.0))

    assert transcript.text == "the processor is fine"
    assert loaded == ["cuda", "cpu"], f"expected a reload on the cpu, got {loaded}"
    assert any("falling back to the CPU" in r.getMessage() for r in caplog.records)


def test_the_fallback_says_how_to_get_the_gpu_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Running Whisper on the CPU is a real cost, so it is not done silently."""
    transcriber, _ = _cuda_transcriber(tmp_path, monkeypatch)

    with caplog.at_level("WARNING", logger="jarvis.audio.stt"):
        transcriber.transcribe(speech(1.0))

    record = next(r for r in caplog.records if "falling back" in r.getMessage())
    assert "nvidia-cudnn-cu12" in record.context["remediation"]
    assert record.context["was_device"] == "cuda"
    assert record.context["now_device"] == "cpu"


def test_the_fallback_happens_once_not_every_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcriber, loaded = _cuda_transcriber(tmp_path, monkeypatch)

    for _ in range(3):
        transcriber.transcribe(speech(1.0))

    assert loaded == ["cuda", "cpu"], "the gpu was retried after falling back"


def test_an_injected_model_is_not_replaced(tmp_path: Path) -> None:
    """A caller-supplied model is theirs. There is nothing to fall back to."""
    transcriber = FasterWhisperTranscriber(
        gpu_config(tmp_path), FakeWhisperModel(raises=OSError(CUDNN_MESSAGE))
    )

    with pytest.raises(SttError) as info:
        transcriber.transcribe(speech(1.0))

    assert "CUDA support library is missing" in info.value.message


def test_an_unrelated_failure_does_not_move_to_the_processor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a missing CUDA library is a reason to change device."""
    transcriber = FasterWhisperTranscriber(gpu_config(tmp_path))
    loaded: list[str] = []

    class _AlwaysBroken:
        def transcribe(self, audio: Any, **kwargs: Any) -> Any:
            raise RuntimeError("the audio was malformed")

    def _load(self: FasterWhisperTranscriber = transcriber) -> Any:
        loaded.append(self._device)
        return _AlwaysBroken()

    monkeypatch.setattr(transcriber, "_load_model", _load)

    with pytest.raises(SttError):
        transcriber.transcribe(speech(1.0))

    assert loaded == ["cuda"], "an unrelated error should not have moved the device"


class TestWarmupMovesToTheProcessor:
    """Discovering a broken CUDA runtime is what warming up is for.

    Reported from the Windows host: warmup failed on cublas64_12.dll, logged a
    warning, and left the model pinned to CUDA. The first real utterance then
    hit the identical failure, fell back, and reloaded the whole model, all
    while the user waited for a reply that never came in time. transcribe()
    already had the fallback; warmup did not use it, which made warming up
    worse than useless on exactly the machine it was meant to help.
    """

    class _CudalessWhisper(FasterWhisperTranscriber):
        """Loads only on the CPU. Anything else fails the way CTranslate2 does."""

        def __init__(self, config: JarvisConfig) -> None:
            super().__init__(config)
            self.loads: list[str] = []

        def _load_model(self) -> Any:
            self.loads.append(self._device)
            if self._device != "cpu":
                msg = "Library cublas64_12.dll is not found or cannot be loaded"
                raise OSError(msg)
            return FakeWhisperModel()

    def _config(self, tmp_path: Path) -> JarvisConfig:
        return load_config(
            tmp_path / "absent.yaml",
            stt={
                "engine": "faster-whisper",
                "model": "small.en",
                "device": "cuda",
                "compute_type": "int8",
            },
        )

    def test_it_falls_back_rather_than_giving_up(self, tmp_path: Path) -> None:
        transcriber = self._CudalessWhisper(self._config(tmp_path))

        assert transcriber.warmup() is True
        assert transcriber.device == "cpu", "warmup left the model on a device it cannot use"
        assert transcriber.loads == ["cuda", "cpu"]

    def test_the_first_utterance_no_longer_pays_for_it(self, tmp_path: Path) -> None:
        """The reload has to happen before anyone speaks, not during."""
        transcriber = self._CudalessWhisper(self._config(tmp_path))
        transcriber.warmup()
        during_warmup = list(transcriber.loads)

        transcriber.transcribe(speech(1.0), WHISPER_SAMPLE_RATE)
        assert transcriber.loads == during_warmup, (
            "the model was rebuilt during the first utterance"
        )

    def test_a_failure_that_is_not_cuda_still_just_warns(self, tmp_path: Path) -> None:
        """Only a missing CUDA library is worth changing device over."""

        class Broken(FasterWhisperTranscriber):
            def _load_model(self) -> Any:
                msg = "the checkpoint is corrupt"
                raise RuntimeError(msg)

        transcriber = Broken(self._config(tmp_path))
        assert transcriber.warmup() is False
        assert transcriber.device == "cuda"


class TestTheCudaProbeTellsTheTruth:
    """A visible card is not a usable one.

    --check reported "cuda yes" seven seconds before cublas64_12.dll failed to
    load on the same machine, which sent the user looking in the wrong place.
    has_cuda() asks the driver; this asks the library that actually has to work.
    """

    def test_it_returns_a_reason_when_unusable(self) -> None:
        usable, detail = cuda_runtime_usable()
        assert isinstance(usable, bool)
        assert detail, "an unusable runtime must say why"

    def test_it_reports_unusable_without_ctranslate2(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("jarvis.audio.stt.has_module", lambda _name: False)
        usable, detail = cuda_runtime_usable()
        assert usable is False
        assert "ctranslate2" in detail

    def test_it_never_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """--check must not crash on a machine with a half installed CUDA."""

        def explode(_name: str) -> bool:
            msg = "boom"
            raise RuntimeError(msg)

        monkeypatch.setattr("jarvis.audio.stt.has_module", explode)
        with pytest.raises(RuntimeError):
            cuda_runtime_usable()
