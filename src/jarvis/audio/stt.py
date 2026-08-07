"""Speech to text: faster-whisper on the GPU tiers, whisper.cpp on the cpu tier.

The shape of this module follows the same split as the wake word detector. The
transcribers are pure and synchronous: hand one a block of audio, get a
:class:`Transcript` back, and nothing else happens. All of the threading lives in
:class:`SttWorker`, which runs a transcriber on its own thread behind a bounded
queue so a slow model can never stall the capture callback (§5).

Four details are load bearing.

**Real time factor.** T-1.4 requires the RTF to be logged per utterance. It is
processing seconds divided by audio seconds, so 0.2 means the model chewed
through the audio five times faster than real time. It is the single number that
tells you whether the configured tier is actually viable: anything at or above
1.0 means transcription cannot keep up with speech, and the 200 ms STT budget
from §3 is already lost.

**The short circuit.** An accidental wake trigger produces a fraction of a second
of room noise. Running a Whisper model over that costs hundreds of milliseconds
and reliably hallucinates a "Thank you." or a "Bye.", which then gets sent to the
LLM as if the user had said it. Audio shorter than ``stt.min_audio_ms`` therefore
returns an empty transcript without the model ever being loaded or called.

**Sample rate.** Every Whisper variant, including whisper.cpp, is trained at
16 kHz and neither library resamples for you. Resampling is out of scope here, so
a mismatch is a loud :class:`~jarvis.util.errors.SttError` rather than a silent
pitch shift that quietly halves accuracy.

**The CUDA library path.** faster-whisper on CUDA loads cuBLAS and cuDNN 9 out of
the process library path, and on a stock Windows box those are simply not there.
The failure surfaces as a bare ``OSError`` naming ``cudnn_ops64_9.dll`` from deep
inside CTranslate2. It is caught here and turned into an error carrying both a
speakable sentence and a remediation hint in the log, because a raw DLL name read
aloud helps nobody.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from jarvis.config import TIER_PROFILES, SttEngine, Tier
from jarvis.state import EventType
from jarvis.util.errors import JarvisError, SttError
from jarvis.util.platform import has_module, require_module

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jarvis.config import JarvisConfig
    from jarvis.state import EventBus

__all__ = [
    "WHISPER_SAMPLE_RATE",
    "FasterWhisperTranscriber",
    "SttResult",
    "SttWorker",
    "Transcriber",
    "Transcript",
    "TranscriptSegment",
    "WhisperCppTranscriber",
    "build_transcriber",
    "faster_whisper_cache",
    "to_float32_mono",
    "whispercpp_cache",
]

_log = logging.getLogger(__name__)


def faster_whisper_cache(config: JarvisConfig) -> Path:
    """Where faster-whisper checkpoints are cached.

    ``stt.download_root`` wins when set. Otherwise everything lands under
    ``models/`` alongside the rest of the downloaded artefacts (§4), which keeps
    a machine that has been through ``pull_models.ps1`` fully offline.

    Module level rather than a method because ``scripts/fetch_stt_model.py``
    needs the same answer. It used to compute its own, which is to say it did
    not compute one at all: it called ``download_model(model)`` with no cache
    argument, so setup filled the default HuggingFace cache and the runtime
    looked in ``models/faster-whisper`` and downloaded the whole checkpoint
    again during the user's first sentence.
    """
    configured = config.stt.download_root
    if configured is not None:
        return config.resolve_path(configured)
    return config.models_dir / "faster-whisper"


def whispercpp_cache(config: JarvisConfig) -> Path:
    """Where whisper.cpp ggml weights are cached. Same story as above."""
    configured = config.stt.download_root
    if configured is not None:
        return config.resolve_path(configured)
    return config.models_dir / "whispercpp"

Samples = npt.NDArray[Any]

#: Every Whisper checkpoint is trained at 16 kHz mono. Neither faster-whisper nor
#: whisper.cpp resamples on your behalf.
WHISPER_SAMPLE_RATE = 16_000

#: Divisor that brings int16 capture back to -1..1. 2**15 maps -32768 to -1.0.
_INT16_SCALE = 32_768.0

#: Peak above which float input is assumed to be miscaled rather than merely hot.
#: Genuine capture clips at 1.0, so anything past this is a scaling bug upstream.
_CLIP_WARN_PEAK = 1.001

#: Substrings that identify the CUDA support libraries CTranslate2 loads lazily.
#: Their names only ever appear in an exception when the load failed.
_CUDA_LIBRARY_NAMES = ("cudnn", "cublas", "cudart", "nvrtc", "cufft")

#: Phrases that mark a dynamic library load failure on Windows or Linux.
_LOAD_FAILURE_MARKERS = (
    "could not locate",
    "cannot open shared object",
    "no such file",
    "library path",
    "not found",
    "failed to load",
    "unable to load",
    ".dll",
    ".so",
)

#: Printed into the log, never spoken. This is the fix for the single most common
#: faster-whisper failure on a fresh Windows install.
_CUDA_REMEDIATION = (
    "faster-whisper on CUDA needs the cuBLAS and cuDNN 9 runtime libraries on the "
    "process library path. Fix it one of three ways: run "
    "'uv pip install nvidia-cublas-cu12 nvidia-cudnn-cu12', or copy "
    "cudnn_ops64_9.dll and cublas64_12.dll into a directory already on PATH, or set "
    "stt.device to cpu and stt.compute_type to int8 in config.yaml to run on the CPU."
)

#: Checkpoints whisper.cpp actually ships as ggml weights. Anything outside this
#: set has to be mapped before pywhispercpp is asked to download it.
_WHISPERCPP_MODELS = frozenset(
    {
        "tiny", "tiny.en", "base", "base.en", "small", "small.en",
        "medium", "medium.en", "large-v1", "large-v2", "large-v3",
        "large-v3-turbo",
    }
)

#: faster-whisper checkpoints with no ggml equivalent, mapped to the nearest one.
#: The distil family is English only, so the .en variants are the honest match.
_WHISPERCPP_EQUIVALENT: dict[str, str] = {
    "distil-large-v3": "small.en",
    "distil-large-v2": "small.en",
    "distil-medium.en": "small.en",
    "distil-small.en": "small.en",
}


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """One timestamped span of recognised speech.

    Attributes:
        text: The span's text, already stripped of the leading space that both
            backends emit.
        start_s: Offset of the span from the start of the submitted audio, in
            seconds.
        end_s: Offset of the end of the span, in seconds. Never before
            :attr:`start_s`.
    """

    text: str
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        """Length of the span in seconds. Never negative."""
        return max(0.0, self.end_s - self.start_s)


@dataclass(frozen=True, slots=True)
class Transcript:
    """Everything one call to a transcriber produced.

    Attributes:
        text: The full utterance, segments joined by single spaces and stripped.
            Empty when the model heard nothing, which is a normal outcome and not
            an error.
        language: Language code the backend reported, or the configured one when
            the backend does not detect. Empty string when neither is known.
        duration_s: Length of the submitted audio in seconds, computed from the
            sample count rather than trusted from the backend.
        rtf: Real time factor, processing seconds divided by :attr:`duration_s`.
            Below 1.0 is faster than real time. 0.0 when nothing ran.
        segments: The timestamped spans, in order.
        no_speech_prob: Mean probability across segments that the audio contains
            no speech, 0.0 to 1.0. 1.0 for an utterance rejected as too short,
            0.0 when the backend does not report it.
    """

    text: str
    language: str
    duration_s: float
    rtf: float
    segments: tuple[TranscriptSegment, ...] = ()
    no_speech_prob: float = 0.0

    @property
    def is_empty(self) -> bool:
        """True when no speech was recognised."""
        return not self.text

    @property
    def processing_s(self) -> float:
        """Seconds the backend spent, reconstructed from :attr:`rtf`."""
        return self.rtf * self.duration_s

    def to_dict(self) -> dict[str, Any]:
        """Structured form for the log and the HUD."""
        return {
            "text": self.text,
            "language": self.language,
            "duration_s": round(self.duration_s, 3),
            "rtf": round(self.rtf, 3),
            "segments": len(self.segments),
            "no_speech_prob": round(self.no_speech_prob, 3),
        }


@dataclass(slots=True)
class _RawResult:
    """What a backend produced, before timing and text assembly."""

    segments: tuple[TranscriptSegment, ...]
    language: str
    no_speech_prob: float


def _empty_transcript(duration_s: float, language: str) -> Transcript:
    """An empty result for audio that was never handed to a model."""
    return Transcript(
        text="",
        language=language,
        duration_s=duration_s,
        rtf=0.0,
        segments=(),
        no_speech_prob=1.0,
    )


# ---------------------------------------------------------------------------
# Input conditioning
# ---------------------------------------------------------------------------


def to_float32_mono(audio: Samples) -> npt.NDArray[np.float32]:
    """Coerce captured audio into the float32 mono in -1..1 that Whisper wants.

    Capture may hand us int16 when ``audio.dtype`` is set that way, float64 from a
    test fixture, or a 2-D block straight off PortAudio. All three are normal, so
    they are converted rather than refused.

    Args:
        audio: Samples of any numeric dtype. A 2-D array is treated as
            (frames, channels) and downmixed by averaging.

    Returns:
        A contiguous 1-D float32 array in -1..1.

    Raises:
        SttError: The array is not numeric, or has more than two dimensions.
    """
    block = np.asarray(audio)
    if block.ndim > 2:
        raise SttError(
            f"audio must be 1-D or 2-D, got shape {block.shape}",
            context={"shape": tuple(block.shape)},
        )
    # Downmix rather than take channel 0: a mic wired to the right channel only
    # would otherwise transcribe as pure silence.
    multichannel = block.ndim == 2 and block.shape[1] > 1
    block = block.mean(axis=1) if multichannel else block.reshape(-1)

    if np.issubdtype(block.dtype, np.integer):
        info = np.iinfo(block.dtype)
        if info.min == 0:
            raise SttError(
                f"unsigned audio dtype {block.dtype} is not supported, "
                "capture must be signed PCM or float",
                context={"dtype": str(block.dtype)},
            )
        samples = block.astype(np.float32) / float(-info.min)
    elif np.issubdtype(block.dtype, np.floating):
        samples = block.astype(np.float32, copy=False)
    else:
        raise SttError(
            f"audio dtype {block.dtype} is not numeric",
            context={"dtype": str(block.dtype)},
        )

    if samples.size and not np.isfinite(samples).all():
        _log.warning(
            "audio contained non-finite samples, replacing them with silence",
            extra={"context": {"samples": int(samples.size)}},
        )
        samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)

    if samples.size:
        peak = float(np.max(np.abs(samples)))
        if peak > _CLIP_WARN_PEAK:
            # Almost always means float samples still at int16 scale upstream.
            _log.warning(
                "audio peaks above full scale, clipping before transcription",
                extra={"context": {"peak": round(peak, 3)}},
            )
            samples = np.clip(samples, -1.0, 1.0)

    return np.ascontiguousarray(samples, dtype=np.float32)


def _is_cuda_library_failure(exc: BaseException) -> bool:
    """True when ``exc`` looks like a missing CUDA support library.

    CTranslate2 surfaces the classic ``cudnn_ops64_9.dll`` problem as a plain
    ``OSError`` with no type of its own, so the message is all there is to go on.
    """
    if not isinstance(exc, OSError | RuntimeError):
        return False
    text = str(exc).lower()
    if any(name in text for name in _CUDA_LIBRARY_NAMES):
        return True
    return "cuda" in text and any(marker in text for marker in _LOAD_FAILURE_MARKERS)


# ---------------------------------------------------------------------------
# The transcriber interface
# ---------------------------------------------------------------------------


@runtime_checkable
class Transcriber(Protocol):
    """What the orchestrator needs from a speech recogniser.

    Implementations are synchronous and must be safe to call from any single
    thread. Nothing here starts a thread of its own; that is
    :class:`SttWorker`'s job.
    """

    def transcribe(self, audio: Samples, sample_rate: int = WHISPER_SAMPLE_RATE) -> Transcript:
        """Recognise ``audio`` and return the result.

        Args:
            audio: Mono samples, float32 in -1..1 or signed integer PCM.
            sample_rate: Rate of ``audio``. Must match what the model expects.

        Returns:
            The transcript. An empty one when nothing was heard.

        Raises:
            SttError: Anything went wrong. Never a bare backend exception.
        """
        ...


class _BaseTranscriber:
    """Validation, timing, logging, and error wrapping shared by both backends.

    Subclasses supply only :meth:`_load_model` and :meth:`_run`. Everything the
    contract in T-1.4 promises, the short circuit, the sample rate check, the RTF
    log line, and the guarantee that only :class:`SttError` escapes, is enforced
    here so neither backend can quietly skip it.
    """

    #: Short name used in log records and error context.
    engine_name = "stt"

    def __init__(
        self,
        config: JarvisConfig,
        model: Any | None = None,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Configure the transcriber without loading anything.

        Args:
            config: Source of the model choice, language, and thresholds. Model,
                device, and compute type come from
                :meth:`~jarvis.config.JarvisConfig.stt_settings`, which has
                already applied the tier profile.
            model: A preloaded backend model. Injected by tests so the suite
                never needs the real engine. None loads it on first use.
            clock: Monotonic time source used for the RTF measurement.
                Injectable so tests can produce a deterministic RTF.
        """
        engine, model_name, device, compute_type = config.stt_settings()
        self._config = config
        self._model = model
        #: A caller-supplied model is theirs, not ours to discard and rebuild.
        #: Falling back to the CPU means reloading, so it applies only when this
        #: object owns the model's lifecycle.
        self._owns_model = model is None
        self._clock: Callable[[], float] = clock or time.perf_counter
        self._engine = engine
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._min_audio_ms = float(config.stt.min_audio_ms)
        language = config.stt.language.strip()
        self._language: str | None = None if language.lower() in ("", "auto") else language
        # Whisper models are not re-entrant. One lock covers loading and calling.
        self._lock = threading.RLock()
        self._utterances = 0
        self._empty_results = 0

    # -- introspection -----------------------------------------------------

    @property
    def model_name(self) -> str:
        """Checkpoint this transcriber will load or has loaded."""
        return self._model_name

    @property
    def device(self) -> str:
        """Device string handed to the backend, ``cuda`` or ``cpu``."""
        return self._device

    @property
    def compute_type(self) -> str:
        """Quantisation handed to the backend, for example ``int8``."""
        return self._compute_type

    @property
    def language(self) -> str | None:
        """Forced language code, or None to let the backend detect it."""
        return self._language

    @property
    def expected_sample_rate(self) -> int:
        """Sample rate the model requires. Always 16 kHz for Whisper."""
        return WHISPER_SAMPLE_RATE

    @property
    def min_audio_ms(self) -> float:
        """Audio shorter than this is rejected without invoking the model."""
        return self._min_audio_ms

    @property
    def is_loaded(self) -> bool:
        """True once a model is present, injected or lazily loaded."""
        return self._model is not None

    @property
    def utterances(self) -> int:
        """Utterances handed to the model since construction."""
        return self._utterances

    @property
    def empty_results(self) -> int:
        """Utterances that produced no text, including short circuits."""
        return self._empty_results

    # -- the contract ------------------------------------------------------

    def _fall_back_to_cpu(self, exc: BaseException) -> bool:
        """Move to the CPU after a missing CUDA library, once.

        A card that nvidia-smi can see is not the same as a working CUDA
        runtime: ctranslate2 also needs cuBLAS and cuDNN 9 on the library path,
        and their absence only shows up here, at the first transcription. Left
        alone this fails identically on every turn, which makes the assistant
        permanently deaf to a problem the CPU could simply absorb.

        The remediation is still logged in full, because running Whisper on the
        processor is a real cost and the user should know it is happening.

        Returns:
            True when the caller should retry.
        """
        if not self._owns_model or self._device == "cpu":
            return False
        if not _is_cuda_library_failure(exc):
            return False
        with self._lock:
            if self._device == "cpu":  # another thread already moved us
                return True
            _log.warning(
                "falling back to the CPU for transcription: %s. %s",
                exc,
                _CUDA_REMEDIATION,
                extra={
                    "context": {
                        "engine": self.engine_name,
                        "model": self._model_name,
                        "was_device": self._device,
                        "now_device": "cpu",
                        "remediation": _CUDA_REMEDIATION,
                    }
                },
            )
            self._device = "cpu"
            self._compute_type = "int8"
            self._model = None
        return True

    def transcribe(self, audio: Samples, sample_rate: int = WHISPER_SAMPLE_RATE) -> Transcript:
        """Recognise ``audio``.

        Args:
            audio: Mono samples, float32 in -1..1 or signed integer PCM. A 2-D
                block is downmixed.
            sample_rate: Rate of ``audio`` in hertz.

        Returns:
            The transcript. Audio shorter than ``stt.min_audio_ms`` returns an
            empty transcript without loading or calling the model.

        Raises:
            SttError: The audio is unusable, the sample rate does not match, the
                model could not be loaded, or the backend failed. Backend
                exceptions never escape unwrapped.
            DependencyMissingError: The backend package is not installed.
        """
        samples = to_float32_mono(audio)
        expected = self.expected_sample_rate
        if int(sample_rate) != expected:
            raise SttError(
                f"audio is {sample_rate} Hz but {self.engine_name} requires {expected} Hz; "
                "resampling is not performed here, capture at the expected rate instead",
                speakable="My microphone is running at the wrong sample rate.",
                context={
                    "engine": self.engine_name,
                    "given_sample_rate": int(sample_rate),
                    "expected_sample_rate": expected,
                },
            )

        duration_s = samples.size / float(expected)
        if duration_s * 1000.0 < self._min_audio_ms:
            self._empty_results += 1
            _log.debug(
                "utterance too short for transcription, skipping the model",
                extra={
                    "context": {
                        "engine": self.engine_name,
                        "duration_ms": round(duration_s * 1000.0, 1),
                        "min_audio_ms": self._min_audio_ms,
                    }
                },
            )
            return _empty_transcript(duration_s, self._language or "")

        started = self._clock()
        try:
            with self._lock:
                model = self._ensure_model()
                raw = self._run(model, samples)
        except JarvisError:
            # Already structured and speakable. Re-raising keeps the install hint
            # from DependencyMissingError intact.
            raise
        except Exception as exc:
            if self._fall_back_to_cpu(exc):
                try:
                    with self._lock:
                        model = self._ensure_model()
                        raw = self._run(model, samples)
                except JarvisError:
                    raise
                except Exception as retry_exc:
                    raise self._wrap_failure(retry_exc, action="transcription") from retry_exc
            else:
                raise self._wrap_failure(exc, action="transcription") from exc
        elapsed_s = max(0.0, self._clock() - started)

        text = " ".join(segment.text for segment in raw.segments).strip()
        transcript = Transcript(
            text=text,
            language=raw.language,
            duration_s=duration_s,
            rtf=elapsed_s / duration_s if duration_s > 0 else 0.0,
            segments=raw.segments,
            no_speech_prob=raw.no_speech_prob,
        )
        self._utterances += 1
        if transcript.is_empty:
            self._empty_results += 1
        # T-1.4 requires the RTF on every utterance. It is the earliest signal
        # that the configured tier cannot keep up with speech.
        _log.info(
            "transcribed utterance",
            extra={
                "context": {
                    "engine": self.engine_name,
                    "model": self._model_name,
                    "device": self._device,
                    "compute_type": self._compute_type,
                    "processing_ms": round(elapsed_s * 1000.0, 1),
                    **transcript.to_dict(),
                }
            },
        )
        return transcript

    def warmup(self) -> bool:
        """Load the model and push a short buffer through it.

        The first inference pays for model load, CUDA context creation, and
        kernel autotuning, which together run into seconds. Doing that while the
        user is mid-sentence blows the 200 ms STT budget in §3, so the
        orchestrator calls this at startup instead.

        Returns:
            True when the model is ready, False when warmup failed. Failure is
            logged and swallowed: a warmup problem should surface on the first
            real utterance with proper context, not crash startup.
        """
        padding = np.zeros(int(self.expected_sample_rate * 0.5), dtype=np.float32)
        try:
            with self._lock:
                model = self._ensure_model()
                self._run(model, padding)
        except Exception as exc:  # noqa: BLE001 - warmup is best effort by design
            _log.warning(
                "speech recogniser warmup failed, continuing without it",
                extra={"context": {"engine": self.engine_name, "error": str(exc)}},
            )
            return False
        _log.info(
            "speech recogniser ready",
            extra={
                "context": {
                    "engine": self.engine_name,
                    "model": self._model_name,
                    "device": self._device,
                    "compute_type": self._compute_type,
                }
            },
        )
        return True

    def unload(self) -> None:
        """Drop the model so its VRAM comes back. The next call reloads it."""
        with self._lock:
            self._model = None

    # -- internals ---------------------------------------------------------

    def _ensure_model(self) -> Any:
        """Return the model, loading it on first use. Lock held."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> Any:
        """Build the backend model. Subclasses implement this."""
        raise NotImplementedError

    def _run(self, model: Any, samples: npt.NDArray[np.float32]) -> _RawResult:
        """Run one buffer through ``model``. Subclasses implement this."""
        raise NotImplementedError

    def _wrap_failure(self, exc: BaseException, *, action: str) -> SttError:
        """Turn any backend exception into a logged, speakable :class:`SttError`."""
        context = {
            "engine": self.engine_name,
            "model": self._model_name,
            "device": self._device,
            "compute_type": self._compute_type,
            "error": type(exc).__name__,
        }
        if _is_cuda_library_failure(exc):
            _log.error(
                "%s failed because a CUDA support library could not be loaded: %s. %s",
                action,
                exc,
                _CUDA_REMEDIATION,
                extra={"context": {**context, "remediation": _CUDA_REMEDIATION}},
            )
            return SttError(
                f"{action} failed, a CUDA support library is missing: {exc}",
                speakable="My speech recogniser cannot reach the graphics card.",
                context={**context, "remediation": _CUDA_REMEDIATION},
            )
        _log.error(
            "%s failed: %s",
            action,
            exc,
            exc_info=True,
            extra={"context": context},
        )
        return SttError(f"{action} failed: {exc}", context=context)


# ---------------------------------------------------------------------------
# faster-whisper
# ---------------------------------------------------------------------------


class FasterWhisperTranscriber(_BaseTranscriber):
    """CTranslate2-backed Whisper, the GPU tier engine (§1).

    faster-whisper is imported lazily inside :meth:`_load_model` so this module
    still imports on a host without the ``stt`` extra (§0b), and the model itself
    is injectable so the test suite never needs a checkpoint or a GPU.
    """

    engine_name = "faster-whisper"

    def _load_model(self) -> Any:
        """Construct ``faster_whisper.WhisperModel`` from the resolved settings.

        Raises:
            DependencyMissingError: faster-whisper is not installed.
            SttError: The package is present but the model would not load, most
                often because the CUDA support libraries are missing.
        """
        module = require_module("faster_whisper", feature="speech recognition")
        model_cls = getattr(module, "WhisperModel", None)
        if model_cls is None:
            raise SttError(
                "faster_whisper is installed but exposes no WhisperModel class",
                speakable="My speech recogniser is not installed correctly.",
                context={"engine": self.engine_name},
            )
        download_root = self._download_root()
        started = self._clock()
        try:
            model = model_cls(
                self._model_name,
                device=self._device,
                compute_type=self._compute_type,
                download_root=str(download_root),
            )
        except Exception as exc:
            raise self._wrap_failure(exc, action="loading the speech model") from exc
        _log.info(
            "faster-whisper model loaded",
            extra={
                "context": {
                    "model": self._model_name,
                    "device": self._device,
                    "compute_type": self._compute_type,
                    "download_root": str(download_root),
                    "load_ms": round((self._clock() - started) * 1000.0, 1),
                }
            },
        )
        return model

    def _download_root(self) -> Path:
        """Where checkpoints are cached."""
        return faster_whisper_cache(self._config)

    def _run(self, model: Any, samples: npt.NDArray[np.float32]) -> _RawResult:
        """Transcribe and drain the generator faster-whisper returns.

        ``WhisperModel.transcribe`` returns lazily: the segment generator does
        the actual decoding, so it has to be consumed inside the timed section or
        the measured RTF would be a fiction.
        """
        segments_iter, info = model.transcribe(
            samples,
            language=self._language,
            beam_size=self._config.stt.beam_size,
            vad_filter=self._config.stt.vad_filter,
            condition_on_previous_text=self._config.stt.condition_on_previous_text,
        )
        segments: list[TranscriptSegment] = []
        probabilities: list[float] = []
        for raw in segments_iter:
            text = str(getattr(raw, "text", "") or "").strip()
            start = _as_float(getattr(raw, "start", 0.0))
            end = _as_float(getattr(raw, "end", start))
            probability = getattr(raw, "no_speech_prob", None)
            if probability is not None:
                probabilities.append(_as_float(probability))
            if text:
                segments.append(TranscriptSegment(text=text, start_s=start, end_s=max(start, end)))
        language = str(getattr(info, "language", "") or "") or (self._language or "")
        mean_no_speech = sum(probabilities) / len(probabilities) if probabilities else 0.0
        return _RawResult(
            segments=tuple(segments),
            language=language,
            no_speech_prob=mean_no_speech,
        )


# ---------------------------------------------------------------------------
# whisper.cpp
# ---------------------------------------------------------------------------


def _whispercpp_model_name(name: str) -> str:
    """Map a configured checkpoint onto one whisper.cpp can actually load.

    The distil checkpoints only exist as CTranslate2 weights, so a fallback from
    faster-whisper to whisper.cpp on a GPU tier would otherwise ask pywhispercpp
    to download something that does not exist.
    """
    if name in _WHISPERCPP_MODELS or "/" in name or "\\" in name or name.endswith(".bin"):
        return name
    mapped = _WHISPERCPP_EQUIVALENT.get(name)
    if mapped is not None:
        _log.warning(
            "whisper.cpp has no %s checkpoint, using %s instead",
            name,
            mapped,
            extra={"context": {"requested": name, "using": mapped}},
        )
        return mapped
    fallback = TIER_PROFILES[Tier.CPU].stt_model
    _log.warning(
        "whisper.cpp does not recognise the model %s, falling back to %s",
        name,
        fallback,
        extra={"context": {"requested": name, "using": fallback}},
    )
    return fallback


class WhisperCppTranscriber(_BaseTranscriber):
    """ggml-backed Whisper through pywhispercpp, the cpu tier engine (§2).

    whisper.cpp reports segment boundaries in centiseconds, which is the only
    real difference from the faster-whisper path once the timings are scaled.
    """

    engine_name = "whispercpp"

    #: whisper.cpp segment timestamps are in units of 10 ms.
    _TIME_SCALE = 0.01

    def __init__(
        self,
        config: JarvisConfig,
        model: Any | None = None,
        *,
        clock: Callable[[], float] | None = None,
        model_name: str | None = None,
    ) -> None:
        """Configure the transcriber.

        Args:
            config: Source of the resolved STT settings.
            model: A preloaded ``pywhispercpp.model.Model``, or any object with a
                compatible ``transcribe``. None loads it on first use.
            clock: Monotonic time source for the RTF measurement.
            model_name: Overrides the checkpoint from the tier profile. Used by
                :func:`build_transcriber` when falling back from a GPU tier,
                whose checkpoint has no ggml equivalent.
        """
        super().__init__(config, model, clock=clock)
        self._model_name = _whispercpp_model_name(model_name or self._model_name)
        # whisper.cpp is a CPU engine. Reporting the tier's cuda device here
        # would put a lie in every log line this transcriber writes.
        self._device = "cpu"

    def _load_model(self) -> Any:
        """Construct ``pywhispercpp.model.Model``.

        Raises:
            DependencyMissingError: pywhispercpp is not installed.
            SttError: The package is present but the model would not load.
        """
        require_module("pywhispercpp", feature="speech recognition")
        try:
            model_cls = importlib.import_module("pywhispercpp.model").Model
        except (ImportError, AttributeError) as exc:
            raise SttError(
                f"pywhispercpp is installed but exposes no Model class: {exc}",
                speakable="My speech recogniser is not installed correctly.",
                context={"engine": self.engine_name},
            ) from exc
        models_dir = self._models_dir()
        models_dir.mkdir(parents=True, exist_ok=True)
        started = self._clock()
        try:
            model = model_cls(
                self._model_name,
                models_dir=str(models_dir),
                print_progress=False,
                print_realtime=False,
                translate=False,
                **({"language": self._language} if self._language else {}),
            )
        except Exception as exc:
            raise self._wrap_failure(exc, action="loading the speech model") from exc
        _log.info(
            "whisper.cpp model loaded",
            extra={
                "context": {
                    "model": self._model_name,
                    "models_dir": str(models_dir),
                    "load_ms": round((self._clock() - started) * 1000.0, 1),
                }
            },
        )
        return model

    def _models_dir(self) -> Path:
        """Directory ggml weights are cached in."""
        return whispercpp_cache(self._config)

    def _run(self, model: Any, samples: npt.NDArray[np.float32]) -> _RawResult:
        """Transcribe one buffer.

        pywhispercpp returns a concrete list of segments, so unlike
        faster-whisper there is no generator to drain.
        """
        segments: list[TranscriptSegment] = []
        for raw in model.transcribe(samples):
            text = str(getattr(raw, "text", "") or "").strip()
            if not text:
                continue
            start = _as_float(getattr(raw, "t0", getattr(raw, "start", 0.0))) * self._TIME_SCALE
            end = _as_float(getattr(raw, "t1", getattr(raw, "end", 0.0))) * self._TIME_SCALE
            segments.append(TranscriptSegment(text=text, start_s=start, end_s=max(start, end)))
        # whisper.cpp exposes no no_speech probability through pywhispercpp.
        return _RawResult(
            segments=tuple(segments),
            language=self._language or "",
            no_speech_prob=0.0,
        )


def _as_float(value: Any) -> float:
    """Best effort float conversion. Anything unusable becomes 0.0."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def build_transcriber(config: JarvisConfig, *, model: Any | None = None) -> Transcriber:
    """Build the transcriber the resolved tier calls for.

    Selection comes from :meth:`~jarvis.config.JarvisConfig.stt_settings`, which
    has already folded ``stt.engine`` together with the tier profile from §2. The
    only decision made here is the fallback: a GPU tier that asks for
    faster-whisper on a machine where the package is not installed gets
    whisper.cpp instead, loudly, rather than a dead voice loop.

    Args:
        config: Resolved configuration.
        model: A preloaded backend model, injected by tests. Supplying one
            disables the fallback check, since the caller clearly has a working
            model for the configured engine.

    Returns:
        A ready :class:`Transcriber`. No model is loaded until the first call.
    """
    engine, model_name, device, compute_type = config.stt_settings()
    if engine is SttEngine.WHISPERCPP:
        return WhisperCppTranscriber(config, model)

    if model is None and not has_module("faster_whisper"):
        _log.warning(
            "faster-whisper is not installed, falling back to whisper.cpp; "
            "run 'uv sync --extra stt' to restore the %s tier engine",
            device,
            extra={
                "context": {
                    "requested_engine": str(engine),
                    "requested_model": model_name,
                    "device": device,
                    "compute_type": compute_type,
                    "fallback_engine": str(SttEngine.WHISPERCPP),
                    "pywhispercpp_installed": has_module("pywhispercpp"),
                }
            },
        )
        return WhisperCppTranscriber(config)

    return FasterWhisperTranscriber(config, model)


# ---------------------------------------------------------------------------
# Off-thread execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class SttResult:
    """One completed job from :class:`SttWorker`.

    Equality is disabled because a result carries a transcript whose segments
    tuple is cheap to compare but whose identity is what callers actually key on.

    Attributes:
        transcript: The transcript, or None when the job failed.
        error: The failure, or None when the job succeeded. Always a
            :class:`~jarvis.util.errors.JarvisError`, so ``error.speakable`` is
            safe to hand to TTS.
        turn_id: Whatever the caller passed to :meth:`SttWorker.submit`, so a
            result can be matched to the turn that asked for it.
        audio_seconds: Length of the submitted audio.
        elapsed_ms: Wall time from picking the job up to finishing it. Feed this
            to :meth:`~jarvis.util.latency.TurnLatency.record` for
            :attr:`~jarvis.util.latency.Stage.STT`.
        queued_ms: Wall time the job spent waiting in the queue, which is how a
            backlog shows up in the latency breakdown.
    """

    transcript: Transcript | None
    error: JarvisError | None
    turn_id: str | None
    audio_seconds: float
    elapsed_ms: float
    queued_ms: float

    @property
    def ok(self) -> bool:
        """True when transcription succeeded, whether or not it heard anything."""
        return self.error is None

    @property
    def text(self) -> str:
        """Recognised text, empty on failure or silence."""
        return self.transcript.text if self.transcript is not None else ""


@dataclass(frozen=True, slots=True, eq=False)
class _Job:
    """One queued transcription request."""

    audio: Samples
    sample_rate: int
    turn_id: str | None
    submitted_at: float


ResultCallback = Callable[[SttResult], None]


class SttWorker:
    """Runs a :class:`Transcriber` on its own thread behind a bounded queue.

    §5 forbids blocking the audio path, and STT is the first stage slow enough to
    matter: a large model on a busy CPU can take longer than the utterance it is
    transcribing. Capture calls :meth:`submit`, which never blocks, and collects
    the answer later from :meth:`results` or a callback.

    The queue is bounded on purpose. If transcription falls behind, the oldest
    thing to do is drop the newest request and say so, because a backlog of stale
    utterances is worse than a missed one: the assistant would answer questions
    that are already several turns out of date.
    """

    def __init__(
        self,
        transcriber: Transcriber,
        bus: EventBus | None = None,
        *,
        on_result: ResultCallback | None = None,
        maxsize: int = 2,
        sample_rate: int = WHISPER_SAMPLE_RATE,
        name: str = "jarvis-stt",
    ) -> None:
        """Wire the worker. No thread starts until :meth:`start`.

        Args:
            transcriber: The engine to drive.
            bus: Event bus. Transcripts publish
                :attr:`~jarvis.state.EventType.TRANSCRIPT` and failures publish
                :attr:`~jarvis.state.EventType.ERROR`. None publishes nothing.
            on_result: Callback invoked on the worker thread for every result,
                successful or not. Keep it quick.
            maxsize: Depth of the submission queue. Beyond this, submissions are
                refused rather than queued.
            sample_rate: Rate assumed when :meth:`submit` is not told one.
            name: Thread name, which shows up in logs and profilers.
        """
        self._transcriber = transcriber
        self._bus = bus
        self._callbacks: list[ResultCallback] = [] if on_result is None else [on_result]
        self._sample_rate = int(sample_rate)
        self._name = name
        self._jobs: queue.Queue[_Job | None] = queue.Queue(maxsize=max(1, int(maxsize)))
        self._results: queue.Queue[SttResult] = queue.Queue()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._submitted = 0
        self._completed = 0
        self._dropped = 0
        self._failed = 0

    # -- introspection -----------------------------------------------------

    @property
    def transcriber(self) -> Transcriber:
        """The engine being driven."""
        return self._transcriber

    @property
    def is_running(self) -> bool:
        """True while the worker thread is alive."""
        with self._lock:
            thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def pending(self) -> int:
        """Jobs accepted but not yet finished."""
        return self._submitted - self._completed

    @property
    def submitted(self) -> int:
        """Jobs accepted since construction."""
        return self._submitted

    @property
    def completed(self) -> int:
        """Jobs finished, successfully or not."""
        return self._completed

    @property
    def dropped(self) -> int:
        """Submissions refused because the queue was full. Non-zero is a smell."""
        return self._dropped

    @property
    def failed(self) -> int:
        """Jobs that ended in an error."""
        return self._failed

    # -- callbacks ---------------------------------------------------------

    def add_listener(self, callback: ResultCallback) -> Callable[[], None]:
        """Register a result callback. Returns a function that removes it."""
        with self._lock:
            self._callbacks.append(callback)

        def remove() -> None:
            with self._lock:
                if callback in self._callbacks:
                    self._callbacks.remove(callback)

        return remove

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start the worker thread. Idempotent."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            thread = threading.Thread(target=self._run, name=self._name, daemon=True)
            self._thread = thread
            thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Ask the thread to finish the current job and exit. Never raises.

        Args:
            timeout: Seconds to wait. A model mid-inference cannot be
                interrupted, so this is generous by default. Exceeding it is
                logged rather than raised, because shutdown must always complete.
        """
        with self._lock:
            thread = self._thread
            self._thread = None
        self._stop.set()
        if thread is None or thread is threading.current_thread():
            return
        # Unblock a thread parked on an empty queue without waiting for a poll.
        with contextlib.suppress(queue.Full):
            self._jobs.put_nowait(None)
        thread.join(timeout=timeout)
        if thread.is_alive():
            _log.warning("stt worker thread did not stop within %.1f s", timeout)

    def __enter__(self) -> SttWorker:
        """Start the worker and return self."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop the worker."""
        self.stop()

    # -- submission and collection -----------------------------------------

    def submit(
        self,
        audio: Samples,
        sample_rate: int | None = None,
        *,
        turn_id: str | None = None,
    ) -> bool:
        """Queue audio for transcription. Never blocks.

        Args:
            audio: Mono samples for one utterance.
            sample_rate: Rate of ``audio``. None uses the worker's default.
            turn_id: Opaque tag echoed back on the result so the orchestrator can
                match a transcript to the turn that asked for it.

        Returns:
            True when the job was queued, False when the queue was full and the
            submission was dropped. Callers must not treat False as fatal; it
            means the machine is behind, not broken.
        """
        job = _Job(
            audio=np.asarray(audio),
            sample_rate=self._sample_rate if sample_rate is None else int(sample_rate),
            turn_id=turn_id,
            submitted_at=time.perf_counter(),
        )
        try:
            self._jobs.put_nowait(job)
        except queue.Full:
            self._dropped += 1
            _log.warning(
                "stt queue is full, dropping an utterance",
                extra={"context": {"dropped": self._dropped, "turn_id": turn_id}},
            )
            return False
        self._submitted += 1
        return True

    def results(self) -> queue.Queue[SttResult]:
        """The queue completed results land on.

        Returned rather than wrapped so a consumer can park on it with its own
        timeout, or drain it without blocking, whichever suits.
        """
        return self._results

    def next_result(self, timeout: float | None = None) -> SttResult | None:
        """Wait for the next result.

        Args:
            timeout: Seconds to wait. None waits forever.

        Returns:
            The result, or None when ``timeout`` elapsed first.
        """
        try:
            return self._results.get(timeout=timeout)
        except queue.Empty:
            return None

    def transcribe_now(
        self,
        audio: Samples,
        sample_rate: int | None = None,
    ) -> Transcript:
        """Transcribe on the calling thread, bypassing the queue.

        Useful for a script or a test that has no reason to spin up a thread.

        Raises:
            SttError: Transcription failed.
        """
        rate = self._sample_rate if sample_rate is None else int(sample_rate)
        return self._transcriber.transcribe(audio, rate)

    # -- the worker thread -------------------------------------------------

    def _run(self) -> None:
        """Drain the queue until stopped. Nothing here may propagate."""
        _log.info("stt worker started", extra={"context": {"thread": self._name}})
        while not self._stop.is_set():
            try:
                job = self._jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            if job is None:  # shutdown sentinel
                break
            self._process(job)
        _log.info(
            "stt worker stopped",
            extra={
                "context": {
                    "completed": self._completed,
                    "failed": self._failed,
                    "dropped": self._dropped,
                }
            },
        )

    def _process(self, job: _Job) -> None:
        """Transcribe one job and publish the outcome."""
        picked_up = time.perf_counter()
        queued_ms = (picked_up - job.submitted_at) * 1000.0
        transcript: Transcript | None = None
        error: JarvisError | None = None
        try:
            transcript = self._transcriber.transcribe(job.audio, job.sample_rate)
        except JarvisError as exc:
            error = exc
        except Exception as exc:  # noqa: BLE001 - a bad job must not kill the thread
            # A transcriber that leaks a raw exception is a contract violation,
            # but a dead STT thread is a mute assistant, so it is contained here.
            error = SttError(
                f"transcriber raised an unwrapped {type(exc).__name__}: {exc}",
                context={"turn_id": job.turn_id, "error": type(exc).__name__},
            )
            _log.error("stt job failed with an unwrapped exception", exc_info=True)
        elapsed_ms = (time.perf_counter() - picked_up) * 1000.0
        self._completed += 1
        if error is not None:
            self._failed += 1
        result = SttResult(
            transcript=transcript,
            error=error,
            turn_id=job.turn_id,
            audio_seconds=job.audio.size / float(max(1, job.sample_rate)),
            elapsed_ms=elapsed_ms,
            queued_ms=queued_ms,
        )
        self._results.put(result)
        self._publish(result)
        self._notify(result)

    def _publish(self, result: SttResult) -> None:
        """Put the outcome on the event bus. A bus failure is not fatal."""
        if self._bus is None:
            return
        try:
            if result.error is not None:
                self._bus.emit(
                    EventType.ERROR,
                    source="stt",
                    error=type(result.error).__name__,
                    message=result.error.message,
                    speakable=result.error.speakable,
                    turn_id=result.turn_id,
                )
            elif result.transcript is not None:
                self._bus.emit(
                    EventType.TRANSCRIPT,
                    turn_id=result.turn_id,
                    elapsed_ms=round(result.elapsed_ms, 1),
                    queued_ms=round(result.queued_ms, 1),
                    **result.transcript.to_dict(),
                )
        except Exception:  # noqa: BLE001 - a subscriber must not stop the worker
            _log.exception("publishing the stt result failed")

    def _notify(self, result: SttResult) -> None:
        """Invoke result callbacks. One bad consumer must not stop the worker."""
        with self._lock:
            callbacks = list(self._callbacks)
        for callback in callbacks:
            try:
                callback(result)
            except Exception:  # noqa: BLE001 - callbacks are foreign code
                _log.exception("stt result callback failed")
