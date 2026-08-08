"""Wake word detection with openWakeWord, driven off the capture ring buffer.

The detector is deliberately split in two:

* :class:`WakeWordDetector` is pure and synchronous. Hand it audio, get a
  :class:`WakeDetection` or ``None`` back. It owns frame accumulation, the int16
  conversion openWakeWord needs, the score threshold, the cooldown, and the
  pre-roll history. It never touches a thread, a bus, or a device, which is what
  makes it testable with a scripted fake model instead of a microphone (§0b).
* :class:`WakeListener` is the plumbing. It pulls fixed size frames off a
  :class:`~jarvis.audio.ring.RingReader` on its own thread, feeds the detector,
  and fans a detection out to the event bus and to direct callbacks.

Two details are load bearing.

**Pre-roll.** openWakeWord only scores a frame once the whole wake phrase is
inside its feature window, so by the time it fires the user has already started
speaking the actual request. The detector therefore keeps the last
``wake.preroll_s`` seconds of audio it has scored and attaches it to the
detection. That slice ends exactly at the trigger point, which is also where the
listener's reader cursor sits, so the orchestrator can concatenate the pre-roll
with everything the reader returns next and get one unbroken timeline with no
clipped syllables.

**Survivability.** The listener thread is the only thing standing between a
flaky ONNX session and a permanently deaf assistant, so a model that raises is
logged and skipped rather than allowed to unwind the thread (§5).
"""

from __future__ import annotations

import difflib
import importlib
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt

from jarvis.state import EventType
from jarvis.util.errors import AudioError, DependencyMissingError, JarvisError
from jarvis.util.platform import require_module

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jarvis.config import JarvisConfig
    from jarvis.state import EventBus

__all__ = [
    "DetectionCallback",
    "FrameReader",
    "WakeDetection",
    "WakeListener",
    "WakeModelLoadError",
    "WakeWordDetector",
    "float_to_int16",
    "strip_wake_word",
]

_log = logging.getLogger(__name__)

Samples = npt.NDArray[Any]

#: int16 range. Scaling uses the positive limit so that a float sample of
#: exactly 1.0 stays representable, see :func:`float_to_int16`.
_INT16_MAX = 32_767
_INT16_MIN = -32_768

#: Divisor used to bring int16 capture back to -1..1. This is the conventional
#: 2**15, which maps -32768 to exactly -1.0.
_INT16_SCALE = 32_768.0

#: Model file extensions openWakeWord understands, keyed by inference framework.
_MODEL_SUFFIX = {"onnx": ".onnx", "tflite": ".tflite"}

#: Repeated failures log in full this many times, then drop to debug level so a
#: broken model cannot flood the log at frame rate.
_MAX_LOGGED_ERRORS = 5

#: openWakeWord's shared feature extractors, mapped to the keyword arguments it
#: takes for each. It resolves these from its own package directory, which ships
#: empty because upstream expects a runtime call to ``download_models()``. This
#: project downloads them during setup instead, so the paths must be handed over
#: explicitly (§0.1 permits model downloads at setup, not fetches while running).
_FEATURE_MODELS = {
    "melspectrogram": "melspec_model_path",
    "embedding_model": "embedding_model_path",
}


class WakeModelLoadError(AudioError):
    """The wake model itself would not load.

    Distinct from a failure to score one frame. A missing or corrupt model file
    does not fix itself between frames, so the listener treats this as fatal
    rather than retrying it at frame rate.
    """


def float_to_int16(samples: Samples) -> npt.NDArray[np.int16]:
    """Convert float audio in the range -1..1 to the int16 openWakeWord expects.

    openWakeWord's melspectrogram front end was trained on 16 bit PCM and its
    ONNX graph takes an int16-scaled tensor, so float32 blocks coming from
    PortAudio have to be scaled before inference. Feeding it raw float32 in
    -1..1 is not an error the library reports; it simply scores near zero
    forever, which looks exactly like a wake word that never fires.

    The scale factor is 32767 rather than 32768 because 32768 is not
    representable as an int16: scaling by it turns a sample of exactly 1.0 into
    a large negative number once the cast wraps. Values are rounded to nearest
    and clipped into the int16 range instead of being allowed to wrap, so a
    clipped microphone produces a loud frame rather than a noise burst.

    Args:
        samples: Audio of any shape. Integer input is already at int16 scale and
            is only clipped and cast.

    Returns:
        int16 samples with the same shape as the input.
    """
    block = np.asarray(samples)
    if np.issubdtype(block.dtype, np.integer):
        return np.clip(block, _INT16_MIN, _INT16_MAX).astype(np.int16)
    scaled = np.rint(np.asarray(block, dtype=np.float64) * _INT16_MAX)
    return np.clip(scaled, _INT16_MIN, _INT16_MAX).astype(np.int16)


def _as_float32(samples: Samples) -> Samples:
    """Flatten ``samples`` to 1-D float32 in -1..1.

    int16 input is divided by 2**15. Round-tripping int16 capture through this
    and back out through :func:`float_to_int16` loses well under one LSB, which
    is far below anything the wake model can resolve.
    """
    block = np.asarray(samples).ravel()
    if np.issubdtype(block.dtype, np.integer):
        return block.astype(np.float32) / _INT16_SCALE
    if block.dtype != np.float32:
        return block.astype(np.float32)
    return block


#: Everything a transcriber might put between words, stripped before matching.
#: Apostrophes survive, because "jarvis's" should still read as the name.
_NOT_WORD = re.compile(r"[^a-z0-9']+")

#: How closely a leading token must match a fixed word of the wake phrase, for
#: example "hey". These are short and common, so the bar has to sit above the
#: ordinary words that are one edit away from them. "they" scores 0.857 against
#: "hey" and "he" scores 0.800, and at 0.8 with a ``>=`` both matched: "They
#: keep crashing" lost its first word and became "keep crashing". There is
#: nothing useful between 0.9 and a real "hey", so this is close to an exact
#: match by design, and that is the right shape for a word this short.
_PREFIX_RATIO = 0.9

#: How closely a token must match the assistant's name when the phrase's fixed
#: prefix already matched. Deliberately loose. openWakeWord has just told us the
#: phrase was spoken, so whatever follows "hey" is almost certainly the mangled
#: name, and the mangling is severe: this machine's transcriber produced
#: "Jardubyse" (0.53), "Jadwis" (0.67) and "Jarvo" (0.73) for "Jarvis". A real
#: question after a bare "hey" is nowhere near: "there" 0.18, "google" 0.00,
#: "what" 0.20.
_NAME_RATIO_AFTER_PREFIX = 0.5

#: How closely a token must match the name when the prefix is absent, as in
#: "Jarvis, how are you". There is no corroborating evidence here, so the bar is
#: much higher: "java" (0.60) and "jarred" (0.50) are ordinary words that must
#: survive, while "javis" (0.91) and "jarvix" (0.83) are the name.
_NAME_RATIO_ALONE = 0.8

#: Consecutive wake phrases removed from one transcript. "Hey Jarvis. Hey
#: Jarvis. What time is it" happens when the user repeats themselves because the
#: first try seemed to do nothing.
_MAX_WAKE_REPEATS = 3


def _words(text: str) -> list[tuple[str, int, int]]:
    """Split ``text`` into lowercase words with their spans in the original.

    The spans are what let the remainder be sliced back out of the caller's
    string with its own casing and punctuation intact, rather than rebuilt from
    the normalised tokens. Scanning the original rather than a case folded copy
    keeps those spans exact: lowercasing is not length preserving for every
    codepoint, and a fold that shifts by one character would slice the answer a
    letter short.

    Deliberately ASCII. Only the leading word or three are ever matched against
    the wake phrase, and everything after them is sliced out untouched, so a
    non-ASCII letter in the body of the question costs nothing.
    """
    return [
        (match.group(0).lower(), match.start(), match.end())
        for match in re.finditer(r"[A-Za-z0-9']+", text)
    ]


def _ratio(a: str, b: str) -> float:
    """Similarity of two words in 0..1."""
    return difflib.SequenceMatcher(None, a, b).ratio()


def wake_phrase_words(wake_word: str) -> list[str]:
    """The wake phrase as lowercase words, for example ``["hey", "jarvis"]``.

    Accepts either an openWakeWord model name (``hey_jarvis``) or a spoken
    phrase (``"hey jarvis"``), since the same setting is read as both.
    """
    return [w for w in _NOT_WORD.split(wake_word.lower().replace("_", " ")) if w]


def _match_name(
    words: list[tuple[str, int, int]], start: int, name: str, *, anchored: bool
) -> int:
    """Words consumed by the name at ``start``, or 0 for no match.

    Args:
        anchored: Whether the phrase's fixed prefix already matched, which is
            what buys the looser threshold and the two-token join.

    When anchored, one token is tried and then two joined, because a
    transcriber that does not know the word sometimes splits it: "jarv is"
    scores 0.92 against "jarvis" while neither half alone would. Unanchored,
    the join is off entirely: "java is" joins to "javais", which scores 0.83
    and would swallow the front of "Java is a language".
    """
    if start >= len(words):
        return 0
    threshold = _NAME_RATIO_AFTER_PREFIX if anchored else _NAME_RATIO_ALONE
    single = _ratio(words[start][0], name)
    best, taken = single, 1
    if anchored and start + 1 < len(words):
        pair = _ratio(words[start][0] + words[start + 1][0], name)
        if pair > single:
            best, taken = pair, 2
    return taken if best >= threshold else 0


def _strip_once(text: str, phrase: list[str]) -> str | None:
    """Remove one leading wake phrase. None when there is nothing to remove."""
    words = _words(text)
    if not words:
        return None
    name = phrase[-1]
    prefix = phrase[:-1]

    index = 0
    # A phrase with no fixed prefix, such as a bare "jarvis", is never anchored:
    # there is nothing corroborating the name, so it has to carry the match on
    # its own at the stricter threshold.
    anchored = bool(prefix)
    for word in prefix:
        if index < len(words) and _ratio(words[index][0], word) >= _PREFIX_RATIO:
            index += 1
        else:
            anchored = False
            break

    taken = _match_name(words, index, name, anchored=anchored) if anchored or index == 0 else 0
    if taken:
        index += taken
    elif not (anchored and index > 0):
        # Neither the name nor a prefix leading into it. Not a wake phrase.
        return None

    # A bare "hey" with the name mangled beyond recognition still leaves filler
    # at the front of the question, and it is filler the user did not say to be
    # transcribed. Dropping it is safe: we are only here because the wake model
    # fired.
    return text[words[index - 1][2] :].lstrip(" \t\n,.!?;:-")


def strip_wake_word(text: str, wake_word: str) -> str:
    """Remove the wake phrase from the front of a transcript.

    The wake word is not part of the question. The ring buffer hands the
    endpointer a second of pre-roll so the front of the request is not clipped,
    which means the phrase that triggered the turn is inside the audio sent to
    STT, and every transcript arrives with it attached. Passed through, the
    assistant answers "Hey, Jardubyse." as though it were the thing asked, which
    is what it sounds like when it reads the wake word back instead of holding a
    conversation.

    Matching is fuzzy because the transcriber has no reason to know the word:
    this machine produced "Hey, Jardubyse.", "Hey, Jadwis!" and "Hey Jarvo" for
    the same two syllables. See the ratio constants above for where each
    threshold sits and what it is holding off.

    Args:
        text: Raw transcript.
        wake_word: The configured wake phrase, ``hey_jarvis`` or ``"hey jarvis"``.

    Returns:
        The request with the wake phrase removed, or an empty string when the
        transcript was nothing but the wake phrase. Text that does not start
        with the phrase is returned unchanged.
    """
    phrase = wake_phrase_words(wake_word)
    if not phrase:
        return text.strip()
    current = text.strip()
    for _ in range(_MAX_WAKE_REPEATS):
        stripped = _strip_once(current, phrase)
        if stripped is None:
            break
        current = stripped.strip()
        if not current:
            return ""
    # Punctuation alone is not a question. "Hey Jarvis." leaves "." behind when
    # the transcriber puts the stop after the phrase.
    if not any(c.isalnum() for c in current):
        return ""
    return current


@dataclass(frozen=True, slots=True, eq=False)
class WakeDetection:
    """One wake word trigger.

    Equality is disabled deliberately: the generated ``__eq__`` would compare
    :attr:`preroll` elementwise and return an array, which then raises when
    Python asks it for a truth value.

    Attributes:
        model: Name of the openWakeWord model that fired, for example
            ``hey_jarvis_v0.1``.
        score: Its confidence at the triggering frame, 0.0 to 1.0. Always at or
            above ``wake.threshold``.
        timestamp: Reading of the detector's clock, :func:`time.monotonic` by
            default, taken as the triggering frame was scored. Monotonic rather
            than wall clock so the orchestrator can subtract it from later
            readings to get a latency (§3).
        preroll: ``wake.preroll_s`` seconds of float32 mono audio ending at the
            trigger point, zero padded at the front when the detector has not
            seen that much audio yet. Prepend it to what the reader returns next
            and the utterance starts unclipped.
    """

    model: str
    score: float
    timestamp: float
    preroll: Samples

    @property
    def preroll_samples(self) -> int:
        """Length of :attr:`preroll` in samples."""
        return int(self.preroll.size)


class WakeWordDetector:
    """Scores fixed size frames with openWakeWord and applies the trigger policy.

    The model is injectable so the test suite never needs openWakeWord, ONNX
    Runtime, or a downloaded checkpoint. When none is supplied it is loaded
    lazily on the first frame, which keeps import time and startup cheap and
    means a machine without the ``audio`` extra still imports this module fine
    (§0b).

    Instances are safe to share between the listener thread and a HUD thread
    reading :attr:`scores`; a single lock covers all mutable state.
    """

    def __init__(
        self,
        config: JarvisConfig,
        model: Any | None = None,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Configure the detector without loading anything.

        Args:
            config: Source of the model name, threshold, cooldown, frame size,
                and pre-roll depth.
            model: A preloaded openWakeWord ``Model``, or anything exposing
                ``predict(int16_frame)``. None loads the real model on first use.
            clock: Monotonic time source, injectable so cooldown tests do not
                have to sleep. Defaults to :func:`time.monotonic`.
        """
        wake = config.wake
        self._config = config
        self._model = model
        self._clock: Callable[[], float] = clock or time.monotonic
        self._model_name = str(wake.model)
        self._frame_samples = int(wake.frame_samples)
        self._threshold = float(wake.threshold)
        self._cooldown_s = float(wake.cooldown_s)
        self._preroll_samples = max(0, round(float(wake.preroll_s) * config.audio.sample_rate))
        self._lock = threading.RLock()
        self._pending: Samples = np.zeros(0, dtype=np.float32)
        self._history: Samples = np.zeros(0, dtype=np.float32)
        self._scores: dict[str, float] = {}
        self._last_detection: float | None = None
        self._frames_processed = 0

    # -- introspection -----------------------------------------------------

    @property
    def frame_samples(self) -> int:
        """Samples per inference frame. 1280 is 80 ms at 16 kHz."""
        return self._frame_samples

    @property
    def threshold(self) -> float:
        """Score at or above which a frame triggers."""
        return self._threshold

    @property
    def cooldown_s(self) -> float:
        """Seconds of suppression after a detection."""
        return self._cooldown_s

    @property
    def preroll_samples(self) -> int:
        """Length of the pre-roll window in samples."""
        return self._preroll_samples

    @property
    def model_name(self) -> str:
        """Configured wake model name, used when a model reports a bare score."""
        return self._model_name

    @property
    def scores(self) -> dict[str, float]:
        """Last score seen per model, for the HUD. A copy, safe to keep."""
        with self._lock:
            return dict(self._scores)

    @property
    def frames_processed(self) -> int:
        """Whole frames handed to the model since construction or reset."""
        with self._lock:
            return self._frames_processed

    @property
    def pending_samples(self) -> int:
        """Buffered samples not yet forming a whole frame."""
        with self._lock:
            return int(self._pending.size)

    @property
    def is_loaded(self) -> bool:
        """True once a model is present, injected or lazily loaded."""
        return self._model is not None

    @property
    def cooldown_remaining(self) -> float:
        """Seconds until detections are allowed again. 0.0 when not suppressed."""
        with self._lock:
            last = self._last_detection
            if last is None:
                return 0.0
            return max(0.0, self._cooldown_s - (self._clock() - last))

    # -- inference ---------------------------------------------------------

    def process(self, frame: Samples) -> WakeDetection | None:
        """Feed audio to the model and report a trigger.

        Callers may pass any number of samples. Whole frames of
        ``wake.frame_samples`` are scored in order and anything left over is
        buffered until the next call, so a caller whose block size does not
        divide the frame size still gets every sample scored exactly once and in
        order.

        At most one detection is returned per call. When a trigger fires part
        way through a long block, the rest of the block stays buffered and is
        scored on the next call, which keeps the pre-roll aligned with the exact
        trigger point.

        Args:
            frame: Mono audio, float32 in -1..1 or int16, at the configured
                sample rate. 2-D input is flattened.

        Returns:
            The detection, or None when nothing crossed the threshold or the
            cooldown is still running.

        Raises:
            DependencyMissingError: openWakeWord is not installed and no model
                was injected.
            AudioError: The model could not be built or returned a score that is
                not a number.
        """
        block = _as_float32(frame)
        with self._lock:
            if block.size:
                self._pending = np.concatenate((self._pending, block))
            size = self._frame_samples
            offset = 0
            detection: WakeDetection | None = None
            while self._pending.size - offset >= size:
                chunk = self._pending[offset : offset + size]
                offset += size
                self._remember_locked(chunk)
                detection = self._score_locked(chunk)
                if detection is not None:
                    break
            if offset:
                self._pending = self._pending[offset:].copy()
            return detection

    def preroll(self) -> Samples:
        """The pre-roll window as it stands right now.

        Returns:
            Exactly :attr:`preroll_samples` float32 samples, oldest first, zero
            padded at the front when less audio has been scored than the window
            holds. A fixed length keeps the padding at the front, so the trigger
            point is always the last sample and the timeline never shifts.
        """
        with self._lock:
            return self._preroll_locked()

    def reset(self) -> None:
        """Clear buffered audio, scores, the pre-roll, and the cooldown.

        Call this when returning to wake-word mode after a turn: the audio from
        the turn just finished must not contribute to the next trigger, and the
        cooldown from that trigger has served its purpose.
        """
        with self._lock:
            self._pending = np.zeros(0, dtype=np.float32)
            self._history = np.zeros(0, dtype=np.float32)
            self._scores = {}
            self._last_detection = None
            self._frames_processed = 0
            model = self._model
        if model is None:
            return
        reset_fn = getattr(model, "reset", None)
        if callable(reset_fn):
            try:
                reset_fn()
            except Exception:  # noqa: BLE001 - a model that cannot reset is not fatal
                _log.warning("wake model reset failed", exc_info=True)

    # -- internals, lock held ----------------------------------------------

    def _remember_locked(self, chunk: Samples) -> None:
        """Append ``chunk`` to the pre-roll history, keeping only the tail."""
        limit = self._preroll_samples
        if limit <= 0:
            return
        combined = np.concatenate((self._history, chunk))
        self._history = combined if combined.size <= limit else combined[-limit:].copy()

    def _preroll_locked(self) -> Samples:
        """Front padded copy of the history. Lock held."""
        limit = self._preroll_samples
        out: Samples = np.zeros(limit, dtype=np.float32)
        take = min(limit, int(self._history.size))
        if take > 0:
            out[limit - take :] = self._history[-take:]
        return out

    def _score_locked(self, chunk: Samples) -> WakeDetection | None:
        """Run one frame through the model and apply threshold and cooldown."""
        model = self._model
        if model is None:
            model = self._load_model()
            self._model = model
        raw = model.predict(float_to_int16(chunk))
        scores = self._normalise(raw)
        self._scores.update(scores)
        self._frames_processed += 1
        if not scores:
            return None
        name, score = max(scores.items(), key=lambda item: item[1])
        if score < self._threshold:
            return None
        now = self._clock()
        last = self._last_detection
        # The cooldown runs from the last emitted detection, never from a
        # suppressed one, so continuous loud speech cannot extend it forever.
        if last is not None and now - last < self._cooldown_s:
            _log.debug(
                "wake word suppressed by cooldown",
                extra={
                    "context": {
                        "model": name,
                        "score": round(score, 4),
                        "remaining_s": round(self._cooldown_s - (now - last), 3),
                    }
                },
            )
            return None
        self._last_detection = now
        return WakeDetection(
            model=name,
            score=score,
            timestamp=now,
            preroll=self._preroll_locked(),
        )

    def _normalise(self, raw: Any) -> dict[str, float]:
        """Coerce a model's return value into ``{model_name: score}``.

        openWakeWord returns a dict keyed by model name. A bare number is
        accepted too so a single-model wrapper can be dropped in.

        Raises:
            AudioError: The value is not a number and not a mapping of numbers.
        """
        if isinstance(raw, dict):
            out: dict[str, float] = {}
            for key, value in raw.items():
                try:
                    out[str(key)] = float(value)
                except (TypeError, ValueError):
                    _log.debug("ignoring non-numeric wake score for %s", key)
            return out
        try:
            return {self._model_name: float(raw)}
        except (TypeError, ValueError) as exc:
            raise AudioError(
                f"wake model returned {type(raw).__name__}, expected a score or a dict of scores",
                context={"model": self._model_name},
            ) from exc

    # -- model loading -----------------------------------------------------

    def _load_model(self) -> Any:
        """Import openWakeWord and build the configured model.

        Imported here rather than at module scope so this file stays importable
        on a host without the ``audio`` extra (§0b).

        Raises:
            DependencyMissingError: openWakeWord is not installed.
            AudioError: The package is installed but the model would not load.
        """
        wake = self._config.wake
        module = require_module("openwakeword", feature="wake word detection")
        model_cls = getattr(module, "Model", None)
        if model_cls is None:
            try:
                model_cls = importlib.import_module("openwakeword.model").Model
            except (ImportError, AttributeError) as exc:
                raise AudioError(
                    f"openwakeword is installed but exposes no Model class: {exc}",
                    speakable="My wake word engine is not installed correctly.",
                ) from exc
        found = self._find_model_file()
        target = str(found) if found is not None else wake.model
        features = self._find_feature_models()
        try:
            model = model_cls(
                wakeword_models=[target],
                inference_framework=wake.inference_framework,
                vad_threshold=float(wake.vad_threshold),
                **features,
            )
        except JarvisError:
            raise
        except Exception as exc:  # any loader failure is one error to us
            missing = sorted(set(_FEATURE_MODELS.values()) - set(features))
            raise WakeModelLoadError(
                f"could not load wake word model {target!r}: {exc}",
                speakable="I could not load my wake word model.",
                context={
                    "model": target,
                    "framework": wake.inference_framework,
                    "feature_models": features or "none found under models_dir",
                    "feature_models_missing": missing,
                    "models_dir": str(self._config.models_dir),
                    "error": str(exc),
                },
            ) from exc
        _log.info(
            "wake word model loaded",
            extra={
                "context": {
                    "model": target,
                    "bundled": found is None,
                    "framework": wake.inference_framework,
                    "threshold": self._threshold,
                }
            },
        )
        return model

    def _find_feature_models(self) -> dict[str, str]:
        """Locate the shared feature extractors, as keyword arguments.

        openWakeWord looks for these inside its own package directory, which is
        empty on a fresh install: upstream expects ``download_models()`` to be
        called at runtime. ``scripts/pull_models.ps1`` fetches them into
        ``models/`` during setup instead, so the paths have to be passed in.
        Without them the loader fails naming its own packaged path, which sends
        the reader looking in the wrong place entirely.

        Returns:
            Keyword arguments for openWakeWord. A file that is not found is
            simply left out, so the packaged resolution still gets its chance.
        """
        suffix = _MODEL_SUFFIX.get(self._config.wake.inference_framework, ".onnx")
        models_dir = self._config.models_dir
        found: dict[str, str] = {}
        for stem, keyword in _FEATURE_MODELS.items():
            for root in (models_dir / "openwakeword", models_dir):
                candidate = root / f"{stem}{suffix}"
                if candidate.is_file():
                    found[keyword] = str(candidate)
                    break
        return found

    def _find_model_file(self) -> Path | None:
        """Locate the wake model under ``models_dir`` before the packaged copy.

        ``scripts/pull_models.ps1`` downloads checkpoints into ``models/``, and a
        local file must win over whatever openWakeWord ships so an offline host
        stays offline. Checkpoints are versioned in the file name
        (``hey_jarvis_v0.1.onnx``), so when several match, the highest sorting
        name is taken as the newest.

        Returns:
            The file, or None to let openWakeWord resolve the name itself.
        """
        wake = self._config.wake
        suffix = _MODEL_SUFFIX.get(wake.inference_framework, ".onnx")
        name = wake.model
        explicit = Path(name)
        if explicit.suffix in _MODEL_SUFFIX.values() and explicit.is_file():
            return explicit
        models_dir = self._config.models_dir
        for root in (models_dir, models_dir / "openwakeword"):
            exact = root / f"{name}{suffix}"
            if exact.is_file():
                return exact
            try:
                matches = sorted(p for p in root.glob(f"{name}*{suffix}") if p.is_file())
            except OSError:  # pragma: no cover - unreadable directory
                matches = []
            if matches:
                return matches[-1]
        return None


class FrameReader(Protocol):
    """The part of :class:`~jarvis.audio.ring.RingReader` the listener uses.

    Stating it as a protocol keeps the listener testable with a hand driven fake
    and free of any dependency on how capture is wired up.
    """

    def read(self, n: int) -> Samples | None:
        """Return exactly ``n`` samples, or None when fewer are available."""


#: Readers may also expose a ``dropped`` count of samples lost to the writer
#: lapping them. Read through :func:`_dropped_count` rather than declared on the
#: protocol above, so a hand driven fake stays a two line class: a reader with
#: no counter simply never reports a gap, which is the truth for a fake that
#: hands over one frame at a time.


def _dropped_count(reader: Any) -> int:
    """Samples this reader has lost, or 0 when it does not keep count."""
    value = getattr(reader, "dropped", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


DetectionCallback = Callable[[WakeDetection], None]


class WakeListener:
    """Runs a :class:`WakeWordDetector` over a reader on a background thread.

    On a trigger it publishes :attr:`~jarvis.state.EventType.WAKE_DETECTED` and
    invokes any registered callbacks. The orchestrator can use either: the bus
    for the HUD and logging, a callback when it wants the detection object
    itself, pre-roll included, without going through a queue.

    Nothing raised by the model, the reader, or a callback is allowed to unwind
    this thread. A dead wake thread means an assistant that never answers again,
    so failures are counted and logged and the loop carries on (§5).
    """

    def __init__(
        self,
        config: JarvisConfig,
        reader: FrameReader,
        bus: EventBus | None = None,
        detector: WakeWordDetector | None = None,
        *,
        on_detected: DetectionCallback | None = None,
        poll_interval_s: float = 0.005,
    ) -> None:
        """Wire the listener. No thread starts until :meth:`start`.

        Args:
            config: Used for the detector and for logging context.
            reader: Cursor over the capture ring buffer.
            bus: Event bus. None publishes nothing and uses callbacks only.
            detector: Detector to drive. None builds one from ``config``, which
                loads the real openWakeWord model on the first frame.
            on_detected: Convenience callback, equivalent to calling
                :meth:`add_listener` straight after construction.
            poll_interval_s: How long to wait when the reader has less than a
                whole frame. Small enough that it adds no meaningful latency,
                large enough that the loop is not a spin.
        """
        self._config = config
        self._reader = reader
        self._bus = bus
        self._detector = detector if detector is not None else WakeWordDetector(config)
        self._poll_interval_s = max(0.0, float(poll_interval_s))
        self._callbacks: list[DetectionCallback] = []
        if on_detected is not None:
            self._callbacks.append(on_detected)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._detections = 0
        self._errors = 0
        # Last value seen from the reader, so a change means the writer
        # lapped this consumer and the next frame is not contiguous.
        self._dropped = 0
        self._sample_rate = max(int(config.audio.sample_rate), 1)

    # -- introspection -----------------------------------------------------

    @property
    def detector(self) -> WakeWordDetector:
        """The detector being driven."""
        return self._detector

    @property
    def is_running(self) -> bool:
        """True while the listener thread is alive."""
        with self._lock:
            thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def detections(self) -> int:
        """Detections dispatched since construction."""
        return self._detections

    @property
    def errors(self) -> int:
        """Exceptions swallowed by the loop. Non-zero deserves a look."""
        return self._errors

    @property
    def scores(self) -> dict[str, float]:
        """Latest model scores, for the HUD."""
        return self._detector.scores

    # -- callbacks ---------------------------------------------------------

    def add_listener(self, callback: DetectionCallback) -> Callable[[], None]:
        """Register a callback invoked on the listener thread for each detection.

        Args:
            callback: Called with the :class:`WakeDetection`. Keep it quick, it
                runs on the wake thread. An exception is logged and ignored.

        Returns:
            A function that removes the callback.
        """
        with self._lock:
            self._callbacks.append(callback)

        def remove() -> None:
            with self._lock:
                if callback in self._callbacks:
                    self._callbacks.remove(callback)

        return remove

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start the listener thread. Idempotent."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            thread = threading.Thread(target=self._run, name="jarvis-wake", daemon=True)
            self._thread = thread
            thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Ask the thread to finish and wait for it. Idempotent, never raises.

        Args:
            timeout: Seconds to wait for the thread. Exceeding it is logged, not
                raised, because shutdown must always complete.
        """
        with self._lock:
            thread = self._thread
            self._thread = None
        self._stop.set()
        if thread is None or thread is threading.current_thread():
            return
        thread.join(timeout=timeout)
        if thread.is_alive():
            _log.warning("wake listener thread did not stop within %.1f s", timeout)

    def __enter__(self) -> WakeListener:
        """Start listening and return self."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop listening."""
        self.stop()

    # -- the listener thread -----------------------------------------------

    def _run(self) -> None:
        """Pull frames, score them, dispatch detections. Never propagates."""
        frame_samples = self._detector.frame_samples
        _log.info(
            "wake listener started",
            extra={
                "context": {
                    "model": self._config.wake.model,
                    "threshold": self._detector.threshold,
                    "frame_samples": frame_samples,
                    "preroll_samples": self._detector.preroll_samples,
                }
            },
        )
        while not self._stop.is_set():
            frame = self._next_frame(frame_samples)
            if frame is None:
                continue
            try:
                detection = self._detector.process(frame)
            except (DependencyMissingError, WakeModelLoadError) as exc:
                # Neither recovers between frames: the package will not install
                # itself and the model file will not appear. Retrying only spins
                # at frame rate and buries the one message worth reading.
                self._fail_fatally(exc)
                break
            except Exception:  # noqa: BLE001 - a bad frame must not kill the thread
                self._note_error("wake model inference failed")
                continue
            if detection is not None:
                self._dispatch(detection)
        _log.info(
            "wake listener stopped",
            extra={"context": {"detections": self._detections, "errors": self._errors}},
        )

    def _next_frame(self, frame_samples: int) -> Samples | None:
        """One frame from the reader, or None when the loop should go round again."""
        try:
            frame = self._reader.read(frame_samples)
        except Exception:  # noqa: BLE001 - the buffer is shared, treat a failure as a gap
            self._note_error("reading the capture buffer failed")
            self._stop.wait(self._poll_interval_s)
            return None
        if frame is None or frame.size == 0:
            self._stop.wait(self._poll_interval_s)
            return None

        # The ring counts what a lapped reader lost and, until this, nobody
        # asked. This loop is the one most likely to be lapped: the detector
        # loads its models lazily inside process(), so the very first frame can
        # block for seconds while the writer keeps going. Scoring across the
        # splice looks for a wake word in audio that never existed, and the
        # pre-roll handed to the turn would be the same splice.
        dropped = _dropped_count(self._reader)
        if dropped != self._dropped:
            lost = dropped - self._dropped
            self._dropped = dropped
            _log.warning(
                "capture fell behind the wake listener, resyncing",
                extra={
                    "context": {
                        "samples": lost,
                        "seconds": round(lost / max(self._sample_rate, 1), 3),
                    }
                },
            )
            self._detector.reset()
            return None
        return frame

    def _dispatch(self, detection: WakeDetection) -> None:
        """Publish and fan out one detection."""
        self._detections += 1
        _log.info(
            "wake word detected",
            extra={
                "context": {
                    "model": detection.model,
                    "score": round(detection.score, 4),
                    "preroll_samples": detection.preroll_samples,
                }
            },
        )
        if self._bus is not None:
            try:
                self._bus.emit(
                    EventType.WAKE_DETECTED,
                    model=detection.model,
                    score=round(detection.score, 4),
                    timestamp=detection.timestamp,
                    preroll_samples=detection.preroll_samples,
                    preroll_seconds=round(
                        detection.preroll_samples / max(1, self._config.audio.sample_rate), 3
                    ),
                    preroll=detection.preroll,
                )
            except Exception:  # noqa: BLE001 - a subscriber must not stop the wake loop
                self._note_error("publishing the wake event failed")
        with self._lock:
            callbacks = list(self._callbacks)
        for callback in callbacks:
            try:
                callback(detection)
            except Exception:  # noqa: BLE001 - one bad consumer must not deafen us
                self._note_error("wake detection callback failed")

    def _note_error(self, message: str) -> None:
        """Count and log a swallowed failure, with a cap on full tracebacks."""
        self._errors += 1
        if self._errors <= _MAX_LOGGED_ERRORS:
            _log.exception(message, extra={"context": {"errors": self._errors}})
        else:
            _log.debug("%s (suppressed, %d total)", message, self._errors)

    def _fail_fatally(self, exc: JarvisError) -> None:
        """Report an unrecoverable failure and let the loop exit."""
        self._errors += 1
        _log.error(
            "wake listener stopping: %s",
            exc.message,
            extra={"context": {"error": type(exc).__name__, **exc.context}},
        )
        if self._bus is not None:
            self._bus.emit(
                EventType.ERROR,
                source="wake",
                error=type(exc).__name__,
                message=exc.message,
                speakable=exc.speakable,
            )
