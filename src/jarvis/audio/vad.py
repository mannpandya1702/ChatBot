"""Silero VAD and utterance endpointing (T-1.3).

Three pieces, deliberately separated so that only the first one needs a model
file and the other two are testable with a scripted probability sequence (§0b):

* :class:`SileroVad` owns the ONNX session and the recurrent state Silero v5
  carries between frames. It converts one fixed size frame into one probability
  and nothing else.
* :class:`Endpointer` owns the turn-taking policy: how much speech opens an
  utterance, how much silence closes it, how much audio is kept either side, and
  the hard cap that stops a stuck detector from hanging the turn. It never
  touches a thread, a device, or the event bus, so the orchestrator decides what
  a closed utterance means.
* :class:`BargeInDetector` answers a much narrower question during playback:
  has the user started talking. It uses the stricter ``barge_in_*`` thresholds
  because the microphone hears the assistant's own voice while it speaks, and a
  false trigger there cuts a sentence in half.

Two details are load bearing.

**Recurrent state.** Silero v5 is an LSTM. Its per-frame output only means
anything if the state tensor from frame N is fed back in at frame N+1, so the
session wrapper threads it through every call and only zeroes it on an explicit
:meth:`SileroVad.reset`. A frame of the wrong length is rejected before the
session is touched, because a partially applied inference would poison every
probability that follows.

**Decision latency.** :attr:`Endpoint.decision_ms` is measured in audio time,
from the last frame that scored as speech to the frame that closed the
utterance. In a live stream, where frames arrive in real time, that is also the
wall clock delay the user feels between finishing a sentence and the assistant
reacting. Measuring it from the audio rather than from ``time.monotonic`` keeps
it deterministic under test and identical in production, where the two agree.
CLAUDE.md §3 budgets this stage at 250 ms; see ``tests/test_vad.py`` for what
the shipped defaults actually produce.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt

from jarvis.util.errors import AudioError, JarvisError
from jarvis.util.platform import require_module

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jarvis.config import JarvisConfig

__all__ = [
    "BargeInDetector",
    "Endpoint",
    "Endpointer",
    "ProbabilitySource",
    "SileroVad",
    "SpeechState",
]

_log = logging.getLogger(__name__)

Samples = npt.NDArray[Any]

#: Divisor that brings int16 capture back to -1..1. 2**15 maps -32768 to -1.0.
_INT16_SCALE = 32_768.0

#: File ``scripts/pull_models.ps1`` drops into ``models/``.
_MODEL_FILENAME = "silero_vad.onnx"

#: Silero only accepts these rates, and only these frame sizes for them.
_FRAMES_FOR_RATE = {16_000: 512, 8_000: 256}

#: Hidden size of the v5 LSTM state tensor, shaped (2, batch, 128).
_STATE_DIM = 128

#: Graph names for Silero v5, which is what ``pull_models.ps1`` fetches. v4 is
#: also accepted: it kept the LSTM hidden and cell tensors apart as ``h`` and
#: ``c`` instead of fusing them into one ``state``, and is detected from the
#: session's own input names rather than assumed.
_V5_INPUTS: tuple[str, ...] = ("input", "state", "sr")
_V5_OUTPUTS: tuple[str, ...] = ("output", "stateN")
_V4_STATE_INPUTS = frozenset({"h", "c"})

#: Reasons an :class:`Endpoint` can be produced.
REASON_TRAILING_SILENCE = "trailing_silence"
REASON_MAX_DURATION = "max_duration"


def _as_float32(samples: Samples) -> Samples:
    """Flatten audio to 1-D float32 in -1..1.

    Args:
        samples: Mono audio of any shape. Integer input is assumed to be int16
            scaled and is divided by 2**15.

    Returns:
        A 1-D float32 view or copy. The caller must not assume it is a copy.
    """
    block = np.asarray(samples).ravel()
    if np.issubdtype(block.dtype, np.integer):
        return block.astype(np.float32) / _INT16_SCALE
    if block.dtype != np.float32:
        return block.astype(np.float32)
    return block


def _frames_for_ms(ms: float, frame_ms: float, *, minimum: int = 0) -> int:
    """Whole frames needed to cover ``ms`` milliseconds.

    Rounds up, so a threshold is always met or exceeded and never undershot by
    the frame quantisation. That matters for ``min_speech_ms``: undershooting it
    would let a cough shorter than the configured floor open a turn.

    Args:
        ms: Duration to cover.
        frame_ms: Duration of one frame.
        minimum: Floor on the result, used where zero frames is meaningless.

    Returns:
        A frame count, never negative.
    """
    if frame_ms <= 0.0:
        return minimum
    return max(minimum, math.ceil(float(ms) / frame_ms))


class ProbabilitySource(Protocol):
    """What :class:`Endpointer` and :class:`BargeInDetector` need from a VAD.

    Stating it as a protocol is what lets the tests drive both detectors from a
    scripted list of numbers with no ONNX Runtime and no model file present.
    """

    def probability(self, frame: Samples) -> float:
        """Return the speech probability of one frame, 0.0 to 1.0."""

    def reset(self) -> None:
        """Forget any state carried between frames."""


def _noop() -> None:
    """Do nothing. Stands in for a probability source with no ``reset``."""


def _bind_source(vad: Any) -> tuple[Callable[[Samples], float], Callable[[], None]]:
    """Resolve a probability source into a scoring and a reset callable.

    Accepts a :class:`ProbabilitySource`, or a bare callable taking a frame and
    returning a probability, which keeps test doubles down to one lambda.

    Args:
        vad: The injected source.

    Returns:
        ``(score, reset)``. ``reset`` is a no-op when the source has none.

    Raises:
        AudioError: The object is neither callable nor exposes ``probability``.
    """
    score = getattr(vad, "probability", None)
    if not callable(score):
        if not callable(vad):
            raise AudioError(
                f"vad source {type(vad).__name__} exposes neither probability() nor __call__",
                context={"source": type(vad).__name__},
            )
        score = vad
    reset = getattr(vad, "reset", None)
    if not callable(reset):
        reset = _noop
    return score, reset


class SileroVad:
    """Per-frame speech probability from the Silero VAD ONNX model.

    The session is injectable and, when it is not injected, loaded lazily on the
    first frame. Both matter: the tests never need ONNX Runtime or a checkpoint,
    and a host without the ``audio`` extra can still import this module (§0b).

    Silero v5 is recurrent. The state tensor returned by frame N is fed back in
    at frame N+1, so probabilities depend on the whole stream so far and not
    just the current 32 ms. :meth:`reset` zeroes it, which is what you want at a
    turn boundary or after dropping audio, and not what you want mid-utterance.

    A lock serialises :meth:`probability` against :meth:`reset`. That keeps the
    state tensor internally consistent, but the recurrence is still per stream:
    give each independent audio stream its own instance.
    """

    def __init__(self, config: JarvisConfig, session: Any | None = None) -> None:
        """Configure the wrapper without loading anything.

        Args:
            config: Source of ``vad.frame_samples``, ``audio.sample_rate``, and
                the models directory the checkpoint is looked up in.
            session: A preloaded ``onnxruntime.InferenceSession``, or any object
                with a matching ``run(output_names, feed)``. None loads the real
                model on the first call to :meth:`probability`.

        Raises:
            AudioError: The configured sample rate and frame size are not a pair
                Silero supports. Caught here rather than inside the ONNX graph,
                where it surfaces as an opaque shape mismatch.
        """
        sample_rate = int(config.audio.sample_rate)
        frame_samples = int(config.vad.frame_samples)
        expected = _FRAMES_FOR_RATE.get(sample_rate)
        if expected is None:
            raise AudioError(
                "Silero VAD supports 8000 Hz and 16000 Hz only, audio.sample_rate is "
                f"{sample_rate}",
                speakable="My voice detector cannot run at that microphone sample rate.",
                context={"sample_rate": sample_rate},
            )
        if frame_samples != expected:
            raise AudioError(
                f"Silero VAD needs {expected} samples per frame at {sample_rate} Hz, "
                f"vad.frame_samples is {frame_samples}",
                speakable="My voice detector is configured with the wrong frame size.",
                context={"sample_rate": sample_rate, "frame_samples": frame_samples},
            )
        self._config = config
        self._session = session
        self._sample_rate = sample_rate
        self._frame_samples = frame_samples
        self._lock = threading.RLock()
        self._input_names: tuple[str, ...] = _V5_INPUTS
        self._output_names: list[str] = list(_V5_OUTPUTS)
        self._legacy = False
        self._state = self._zero_state()
        self._hidden = self._zero_state(pairs=1)
        self._cell = self._zero_state(pairs=1)
        self._frames_processed = 0
        if session is not None:
            self._bind(session)

    # -- introspection -----------------------------------------------------

    @property
    def frame_samples(self) -> int:
        """Samples the model expects per call. 512 is 32 ms at 16 kHz."""
        return self._frame_samples

    @property
    def sample_rate(self) -> int:
        """Sample rate handed to the model with every frame."""
        return self._sample_rate

    @property
    def is_loaded(self) -> bool:
        """True once a session is present, injected or lazily loaded."""
        return self._session is not None

    @property
    def frames_processed(self) -> int:
        """Frames scored since construction or the last :meth:`reset`."""
        with self._lock:
            return self._frames_processed

    # -- inference ---------------------------------------------------------

    def probability(self, frame: Samples) -> float:
        """Score one frame.

        Args:
            frame: Exactly :attr:`frame_samples` mono samples, float32 in -1..1
                or int16. 2-D input of the right total size is flattened.

        Returns:
            Speech probability, clamped to 0.0 to 1.0.

        Raises:
            AudioError: The frame is the wrong length, the model file is
                missing, or the session failed. A wrong length is rejected
                before the session runs, so the recurrent state is untouched and
                the caller can carry on with the next frame.
            DependencyMissingError: ONNX Runtime is not installed and no session
                was injected.
        """
        block = _as_float32(frame)
        if block.size != self._frame_samples:
            raise AudioError(
                f"Silero VAD needs frames of exactly {self._frame_samples} samples, "
                f"got {block.size}",
                speakable="My voice detector was handed the wrong amount of audio.",
                context={"expected_samples": self._frame_samples, "got_samples": int(block.size)},
            )
        with self._lock:
            session = self._session
            if session is None:
                session = self._load_session()
                self._session = session
                self._bind(session)
            feed = self._build_feed(block)
            try:
                outputs = session.run(self._output_names, feed)
            except JarvisError:
                raise
            except Exception as exc:  # any backend failure is one error to us
                raise AudioError(
                    f"Silero VAD inference failed: {exc}",
                    speakable="My voice detector stopped responding.",
                    context={"error": str(exc), "frame_samples": self._frame_samples},
                ) from exc
            self._absorb_state(outputs)
            self._frames_processed += 1
            return self._extract_probability(outputs)

    def reset(self) -> None:
        """Zero the recurrent state so the next frame starts a fresh stream.

        Call this at a turn boundary or after a gap in capture. Calling it
        mid-utterance throws away the context that makes the model accurate.
        """
        with self._lock:
            self._state = self._zero_state()
            self._hidden = self._zero_state(pairs=1)
            self._cell = self._zero_state(pairs=1)
            self._frames_processed = 0

    # -- internals ---------------------------------------------------------

    def _zero_state(self, *, pairs: int = 2) -> Samples:
        """A zeroed state tensor shaped ``(pairs, 1, 128)``."""
        return np.zeros((pairs, 1, _STATE_DIM), dtype=np.float32)

    def _bind(self, session: Any) -> None:
        """Learn the session's input and output names and pick the model era.

        A session that cannot describe itself is assumed to be v5, which is what
        ``pull_models.ps1`` fetches.
        """
        self._input_names = self._names(session, "get_inputs", _V5_INPUTS)
        self._output_names = list(self._names(session, "get_outputs", _V5_OUTPUTS))
        self._legacy = _V4_STATE_INPUTS.issubset(self._input_names)
        if self._legacy:
            _log.debug("silero session looks like v4, feeding separate h and c tensors")

    @staticmethod
    def _names(session: Any, attribute: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
        """Names reported by ``session.get_inputs()`` or ``get_outputs()``."""
        getter = getattr(session, attribute, None)
        if not callable(getter):
            return fallback
        try:
            entries = list(getter())
        except Exception:  # noqa: BLE001 - a session that will not introspect still runs
            return fallback
        names = tuple(str(getattr(entry, "name", entry)) for entry in entries)
        return names or fallback

    def _build_feed(self, block: Samples) -> dict[str, Any]:
        """Assemble the ONNX feed dict for one frame. Lock held."""
        feed: dict[str, Any] = {}
        for name in self._input_names:
            if name == "input":
                feed[name] = block.reshape(1, -1).astype(np.float32, copy=False)
            elif name == "sr":
                feed[name] = np.array(self._sample_rate, dtype=np.int64)
            elif name == "state":
                feed[name] = self._state
            elif name == "h":
                feed[name] = self._hidden
            elif name == "c":
                feed[name] = self._cell
        return feed

    def _absorb_state(self, outputs: Any) -> None:
        """Carry the returned recurrent tensors into the next call. Lock held."""
        values = list(outputs)
        if self._legacy:
            if len(values) >= 3:
                self._hidden = np.asarray(values[1], dtype=np.float32)
                self._cell = np.asarray(values[2], dtype=np.float32)
            return
        if len(values) >= 2:
            self._state = np.asarray(values[1], dtype=np.float32)

    def _extract_probability(self, outputs: Any) -> float:
        """Pull the scalar probability out of the model's first output.

        Raises:
            AudioError: The output is empty or not a number, which means the
                session is not the model we think it is.
        """
        values = list(outputs)
        try:
            flat = np.asarray(values[0], dtype=np.float64).reshape(-1)
            value = float(flat[0])
        except (TypeError, ValueError, IndexError) as exc:
            raise AudioError(
                "Silero VAD returned no usable probability",
                speakable="My voice detector returned something I could not read.",
                context={"error": str(exc)},
            ) from exc
        if not math.isfinite(value):
            raise AudioError(
                f"Silero VAD returned a non-finite probability: {value}",
                speakable="My voice detector returned an invalid reading.",
                context={"value": value},
            )
        return min(1.0, max(0.0, value))

    def _load_session(self) -> Any:
        """Import ONNX Runtime and open the checkpoint. Lock held.

        Imported here rather than at module scope so this file stays importable
        on a host without the ``audio`` extra (§0b).

        Raises:
            DependencyMissingError: ONNX Runtime is not installed.
            AudioError: The checkpoint is missing or will not open.
        """
        onnxruntime = require_module("onnxruntime", feature="voice activity detection")
        path = self.find_model_file()
        if path is None:
            raise AudioError(
                f"{_MODEL_FILENAME} was not found under {self._config.models_dir}; "
                "run scripts/pull_models.ps1 to download it",
                speakable="My voice detection model is not downloaded yet.",
                context={"models_dir": str(self._config.models_dir), "file": _MODEL_FILENAME},
            )
        try:
            options = onnxruntime.SessionOptions()
            # The graph is tiny. Threading it costs more in scheduling than it
            # saves, and adds jitter to a stage budgeted at 250 ms (§3).
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = 1
            session = onnxruntime.InferenceSession(
                str(path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:  # any loader failure is one error to us
            raise AudioError(
                f"could not open {path}: {exc}",
                speakable="I could not load my voice detection model.",
                context={"path": str(path), "error": str(exc)},
            ) from exc
        _log.info(
            "silero vad model loaded",
            extra={
                "context": {
                    "path": str(path),
                    "sample_rate": self._sample_rate,
                    "frame_samples": self._frame_samples,
                }
            },
        )
        return session

    def find_model_file(self) -> Path | None:
        """Locate ``silero_vad.onnx`` under the models directory.

        Checked in order: the directory itself, then the ``silero_vad`` and
        ``vad`` subdirectories a downloader might create, then any versioned
        variant such as ``silero_vad_v5.onnx``, newest sorting name winning.

        Returns:
            The checkpoint, or None when nothing matches.
        """
        models_dir = self._config.models_dir
        roots = (models_dir, models_dir / "silero_vad", models_dir / "vad")
        for root in roots:
            exact = root / _MODEL_FILENAME
            if exact.is_file():
                return exact
        for root in roots:
            try:
                matches = sorted(p for p in root.glob("silero_vad*.onnx") if p.is_file())
            except OSError:  # pragma: no cover - unreadable directory
                matches = []
            if matches:
                return matches[-1]
        return None


class SpeechState(StrEnum):
    """Where the endpointer thinks the utterance is.

    ``SILENCE`` also covers an unconfirmed candidate: audio that has scored as
    speech but has not yet accumulated ``vad.min_speech_ms``, and so may still
    turn out to be a cough.
    """

    SILENCE = "silence"
    SPEECH = "speech"
    TRAILING = "trailing"


@dataclass(frozen=True, slots=True, eq=False)
class Endpoint:
    """One complete utterance, ready for STT.

    Equality is disabled deliberately: a generated ``__eq__`` would compare
    :attr:`audio` elementwise and return an array, which then raises when Python
    asks it for a truth value.

    Attributes:
        audio: float32 mono audio at the endpointer's sample rate, running from
            ``vad.speech_pad_ms`` before the first speech frame to
            ``vad.speech_pad_ms`` after the last one. Both pads are truncated,
            not zero filled, when the stream did not contain that much audio.
        duration_s: Length of :attr:`audio` in seconds, pads included.
        decision_ms: Audio time from the last speech frame to the frame that
            closed the utterance. This is the delay the user feels after they
            stop speaking, and is the figure CLAUDE.md §3 budgets at 250 ms. It
            is close to zero for a ``max_duration`` close, since the user was
            still talking when the cap hit.
        reason: ``"trailing_silence"`` when silence closed the utterance,
            ``"max_duration"`` when the hard cap did.
    """

    audio: Samples
    duration_s: float
    decision_ms: float
    reason: str

    @property
    def samples(self) -> int:
        """Length of :attr:`audio` in samples."""
        return int(self.audio.size)


class Endpointer:
    """Frame by frame utterance endpointing.

    Feed it every frame of a live stream and it hands back exactly one
    :class:`Endpoint` at the moment an utterance completes, then goes back to
    waiting. The policy it applies, all from ``config.vad``:

    * A run of speech shorter than ``min_speech_ms`` never opens an utterance,
      so a cough, a click, or a door does not start a turn.
    * Once open, ``trailing_silence_ms`` of continuous silence closes it.
    * ``speech_pad_ms`` of audio is kept either side of the detected speech, so
      a soft first consonant or a trailing plosive still reaches STT.
    * ``max_utterance_s`` of buffered audio force-closes with reason
      ``"max_duration"``, so a detector stuck on a noisy room cannot hang the
      turn forever. The cap is checked per frame, so the result can overshoot by
      up to one frame.

    The probability source is injectable, so the whole policy is testable from a
    scripted list of numbers with no model present. Instances are not thread
    safe; drive one from the audio thread.
    """

    def __init__(
        self,
        config: JarvisConfig,
        vad: Any | None = None,
        sample_rate: int = 16_000,
    ) -> None:
        """Build an endpointer.

        Args:
            config: Source of every threshold in ``config.vad``.
            vad: Probability source. A :class:`ProbabilitySource`, a bare
                callable taking a frame, or None to build a :class:`SileroVad`
                which loads its model on the first frame.
            sample_rate: Rate of the frames that will be fed in. Only used to
                convert frames to milliseconds and seconds.
        """
        settings = config.vad
        self._config = config
        self._vad = vad if vad is not None else SileroVad(config)
        self._probability, self._reset_source = _bind_source(self._vad)
        self._sample_rate = max(1, int(sample_rate))
        self._frame_samples = int(settings.frame_samples)
        self._frame_ms = 1_000.0 * self._frame_samples / self._sample_rate
        self._threshold = float(settings.threshold)
        self._min_speech_frames = _frames_for_ms(
            settings.min_speech_ms, self._frame_ms, minimum=1
        )
        self._trailing_frames = _frames_for_ms(
            settings.trailing_silence_ms, self._frame_ms, minimum=1
        )
        self._pad_frames = _frames_for_ms(settings.speech_pad_ms, self._frame_ms)
        self._max_frames = _frames_for_ms(
            settings.max_utterance_s * 1_000.0, self._frame_ms, minimum=1
        )
        self._pre: deque[Samples] = deque(maxlen=self._pad_frames)
        self._buffer: list[Samples] = []
        self._state = SpeechState.SILENCE
        self._active = False
        self._speech_frames = 0
        self._silence_run = 0
        self._last_speech_index = -1
        self._frames_processed = 0

    # -- introspection -----------------------------------------------------

    @property
    def state(self) -> SpeechState:
        """Current position in the utterance."""
        return self._state

    @property
    def is_speaking(self) -> bool:
        """True while a confirmed utterance is open.

        That covers :attr:`SpeechState.TRAILING` as well as
        :attr:`SpeechState.SPEECH`: during trailing silence the user has paused
        but the utterance has not closed, and treating that as "not speaking"
        would let the orchestrator start a second turn over the top of the
        first. An unconfirmed candidate reads as False, because it may still
        turn out to be a cough.
        """
        return self._state in (SpeechState.SPEECH, SpeechState.TRAILING)

    @property
    def frame_samples(self) -> int:
        """Samples every frame handed to :meth:`process` must contain."""
        return self._frame_samples

    @property
    def frame_ms(self) -> float:
        """Duration of one frame in milliseconds."""
        return self._frame_ms

    @property
    def min_speech_frames(self) -> int:
        """Speech frames that must accumulate before an utterance counts."""
        return self._min_speech_frames

    @property
    def trailing_silence_frames(self) -> int:
        """Consecutive silent frames that close an open utterance."""
        return self._trailing_frames

    @property
    def pad_frames(self) -> int:
        """Frames of audio kept either side of the detected speech."""
        return self._pad_frames

    @property
    def max_frames(self) -> int:
        """Buffered frames at which the utterance is force-closed."""
        return self._max_frames

    @property
    def expected_decision_ms(self) -> float:
        """Decision latency a ``trailing_silence`` close will report.

        Pure arithmetic on the configuration, available before any audio has
        been seen, so startup can compare it against the 250 ms budget in
        CLAUDE.md §3 and say something useful if the two disagree.
        """
        return self._trailing_frames * self._frame_ms

    @property
    def speech_ms(self) -> float:
        """Speech accumulated in the current utterance or candidate."""
        return self._speech_frames * self._frame_ms

    @property
    def silence_ms(self) -> float:
        """Length of the current run of silence."""
        return self._silence_run * self._frame_ms

    @property
    def buffered_seconds(self) -> float:
        """Audio currently held for the open utterance, pads included."""
        return len(self._buffer) * self._frame_samples / self._sample_rate

    @property
    def frames_processed(self) -> int:
        """Frames accepted since construction or the last :meth:`reset`."""
        return self._frames_processed

    # -- the state machine -------------------------------------------------

    def process(self, frame: Samples) -> Endpoint | None:
        """Feed one frame and report a completed utterance.

        Args:
            frame: Exactly :attr:`frame_samples` mono samples at the configured
                sample rate, float32 in -1..1 or int16.

        Returns:
            The :class:`Endpoint` on the frame that closes the utterance, and
            None on every other frame. Exactly one endpoint is produced per
            utterance; the endpointer resets itself before returning it.

        Raises:
            AudioError: The frame is the wrong length. Nothing is buffered and
                no probability is taken, so the state machine and the model's
                recurrent state both survive the bad frame intact.
        """
        block = _as_float32(frame)
        if block.size != self._frame_samples:
            raise AudioError(
                f"endpointer needs frames of exactly {self._frame_samples} samples, "
                f"got {block.size}",
                speakable="My voice detector was handed the wrong amount of audio.",
                context={
                    "expected_samples": self._frame_samples,
                    "got_samples": int(block.size),
                    "state": str(self._state),
                },
            )
        probability = float(self._probability(block))
        self._frames_processed += 1
        return self._advance(block, probability >= self._threshold)

    def reset(self) -> None:
        """Drop all buffered audio and reset the probability source.

        Use this between turns. Unlike the automatic reset that follows an
        endpoint, this also clears the pre-roll and tells the model to forget
        the stream, which is correct when audio was dropped or the microphone
        was closed and wrong in the middle of a sentence.
        """
        self._pre.clear()
        self._reset_utterance()
        self._reset_source()
        self._frames_processed = 0

    def _advance(self, block: Samples, speech: bool) -> Endpoint | None:
        """Run one frame through the state machine."""
        if not self._active:
            if not speech:
                self._pre.append(block)
                return None
            # First speech frame of a candidate. The pre-roll becomes its lead
            # pad, so the pad is real audio from before the speech started.
            self._buffer = list(self._pre)
            self._pre.clear()
            self._active = True
            self._speech_frames = 0
            self._silence_run = 0
            self._last_speech_index = -1

        self._buffer.append(block)
        index = len(self._buffer) - 1

        if speech:
            self._speech_frames += 1
            self._silence_run = 0
            self._last_speech_index = index
            if self._state is SpeechState.TRAILING:
                self._state = SpeechState.SPEECH
            elif (
                self._state is SpeechState.SILENCE
                and self._speech_frames >= self._min_speech_frames
            ):
                self._state = SpeechState.SPEECH
                _log.debug(
                    "utterance confirmed",
                    extra={"context": {"speech_ms": round(self.speech_ms, 1)}},
                )
        else:
            self._silence_run += 1
            if self._state is SpeechState.SPEECH:
                self._state = SpeechState.TRAILING
            if self._silence_run >= self._trailing_frames:
                if self._state is SpeechState.TRAILING:
                    return self._close(REASON_TRAILING_SILENCE)
                # The candidate never reached min_speech_ms. A blip, not a turn.
                self._discard("below min_speech_ms")
                return None

        if len(self._buffer) >= self._max_frames:
            if self.is_speaking:
                return self._close(REASON_MAX_DURATION)
            # A candidate that has churned for the whole cap without ever
            # confirming is noise, and emitting it would break the min_speech
            # guarantee, so it goes in the bin like any other blip.
            self._discard("max duration reached while unconfirmed")
            return None
        return None

    def _close(self, reason: str) -> Endpoint:
        """Build the endpoint, rewind, and hand it back."""
        last = self._last_speech_index
        frames = len(self._buffer)
        if last >= 0:
            keep = min(frames, last + self._pad_frames + 1)
            decision_frames = (frames - 1) - last
        else:  # pragma: no cover - an active utterance always has a speech frame
            keep = frames
            decision_frames = 0
        retained = self._buffer[:keep]
        audio = (
            np.concatenate(retained).astype(np.float32, copy=False)
            if retained
            else np.zeros(0, dtype=np.float32)
        )
        endpoint = Endpoint(
            audio=audio,
            duration_s=float(audio.size) / self._sample_rate,
            decision_ms=decision_frames * self._frame_ms,
            reason=reason,
        )
        self._rewind()
        _log.info(
            "utterance endpointed",
            extra={
                "context": {
                    "reason": reason,
                    "duration_s": round(endpoint.duration_s, 3),
                    "decision_ms": round(endpoint.decision_ms, 1),
                    "samples": endpoint.samples,
                }
            },
        )
        return endpoint

    def _discard(self, why: str) -> None:
        """Throw away a candidate that never became an utterance."""
        _log.debug(
            "discarding speech candidate",
            extra={"context": {"reason": why, "speech_ms": round(self.speech_ms, 1)}},
        )
        self._rewind()

    def _rewind(self) -> None:
        """Clear the utterance, seeding the pre-roll from its tail.

        The tail is the most recent audio in the stream, so it is exactly what
        the next utterance's lead pad should be. The probability source is left
        alone: capture has not stopped, and Silero's recurrence is only accurate
        while it sees the stream unbroken.
        """
        tail = self._buffer[-self._pad_frames :] if self._pad_frames else []
        self._reset_utterance()
        self._pre.extend(tail)

    def _reset_utterance(self) -> None:
        """Return to the idle state, keeping nothing."""
        self._buffer = []
        self._state = SpeechState.SILENCE
        self._active = False
        self._speech_frames = 0
        self._silence_run = 0
        self._last_speech_index = -1


class BargeInDetector:
    """Detects the user talking over the assistant.

    Same frames, different question and different thresholds. During playback
    the microphone hears the assistant's own voice, so this uses
    ``vad.barge_in_threshold``, which sits above ``vad.threshold``, and requires
    ``vad.barge_in_min_speech_ms`` of *consecutive* speech frames rather than
    accumulated ones. Both make it harder to trigger, on purpose: a false
    positive cuts the assistant off mid-sentence for no reason.

    It latches. Once triggered, :meth:`process` keeps returning True until
    :meth:`reset` is called, so a caller that polls cannot miss the edge.
    """

    def __init__(self, config: JarvisConfig, vad: Any | None = None) -> None:
        """Build a barge-in detector.

        Args:
            config: Source of ``vad.barge_in_threshold``,
                ``vad.barge_in_min_speech_ms``, and the frame size.
            vad: Probability source, as for :class:`Endpointer`. None builds a
                :class:`SileroVad` that loads on first use.
        """
        settings = config.vad
        self._config = config
        self._vad = vad if vad is not None else SileroVad(config)
        self._probability, self._reset_source = _bind_source(self._vad)
        self._sample_rate = max(1, int(config.audio.sample_rate))
        self._frame_samples = int(settings.frame_samples)
        self._frame_ms = 1_000.0 * self._frame_samples / self._sample_rate
        self._threshold = float(settings.barge_in_threshold)
        self._min_frames = _frames_for_ms(
            settings.barge_in_min_speech_ms, self._frame_ms, minimum=1
        )
        self._run = 0
        self._triggered = False
        self._frames_processed = 0

    # -- introspection -----------------------------------------------------

    @property
    def threshold(self) -> float:
        """Probability a frame must reach to count towards a barge-in."""
        return self._threshold

    @property
    def min_speech_frames(self) -> int:
        """Consecutive qualifying frames needed to trigger."""
        return self._min_frames

    @property
    def frame_samples(self) -> int:
        """Samples every frame handed to :meth:`process` must contain."""
        return self._frame_samples

    @property
    def frame_ms(self) -> float:
        """Duration of one frame in milliseconds."""
        return self._frame_ms

    @property
    def triggered(self) -> bool:
        """True once the user has been heard talking, until :meth:`reset`."""
        return self._triggered

    @property
    def speech_ms(self) -> float:
        """Length of the current run of qualifying frames."""
        return self._run * self._frame_ms

    @property
    def frames_processed(self) -> int:
        """Frames accepted since construction or the last :meth:`reset`."""
        return self._frames_processed

    # -- detection ---------------------------------------------------------

    def process(self, frame: Samples) -> bool:
        """Feed one frame and report whether the user is talking.

        Frames continue to be scored after a trigger so the model's recurrent
        state stays aligned with the stream for whatever consumes it next.

        Args:
            frame: Exactly :attr:`frame_samples` mono samples, float32 in -1..1
                or int16.

        Returns:
            True from the frame that completes the required run onwards, until
            :meth:`reset`.

        Raises:
            AudioError: The frame is the wrong length. Nothing is scored, so the
                run counter and the model state survive intact.
        """
        block = _as_float32(frame)
        if block.size != self._frame_samples:
            raise AudioError(
                f"barge-in detector needs frames of exactly {self._frame_samples} samples, "
                f"got {block.size}",
                speakable="My voice detector was handed the wrong amount of audio.",
                context={
                    "expected_samples": self._frame_samples,
                    "got_samples": int(block.size),
                },
            )
        probability = float(self._probability(block))
        self._frames_processed += 1
        if probability >= self._threshold:
            self._run += 1
        else:
            self._run = 0
        if not self._triggered and self._run >= self._min_frames:
            self._triggered = True
            _log.info(
                "barge-in detected",
                extra={
                    "context": {
                        "threshold": self._threshold,
                        "speech_ms": round(self.speech_ms, 1),
                    }
                },
            )
        return self._triggered

    def reset(self) -> None:
        """Clear the trigger and the run counter, and reset the source."""
        self._run = 0
        self._triggered = False
        self._frames_processed = 0
        self._reset_source()
