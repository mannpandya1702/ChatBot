"""Kokoro speech synthesis and streaming sentence chunking.

T-1.5. The §3 budget allows 300 ms to first audio, which is far less than a full
reply takes to generate. The way that budget is met is chunking: the LLM's
stream is split into speakable pieces as it arrives, and the first complete
sentence is synthesised and played while the rest is still being written.

The splitter is deliberately careful about abbreviations and decimals. Breaking
"forty three point two percent" after "point" would make the assistant pause
mid-number, which sounds broken in a way a missed sentence boundary does not.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import numpy.typing as npt

from jarvis.config import JarvisConfig
from jarvis.util.errors import TtsError
from jarvis.util.platform import require_module

__all__ = [
    "KokoroSynthesizer",
    "SentenceChunker",
    "build_synthesizer",
    "split_sentences",
]

_log = logging.getLogger(__name__)

Samples = npt.NDArray[np.float32]

#: Abbreviations whose trailing full stop never ends a sentence.
_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt",
        "e.g", "i.e", "etc", "vs", "approx", "no", "fig", "vol",
        "inc", "ltd", "co", "corp", "dept", "est", "min", "max",
        "gb", "mb", "kb", "tb", "ghz", "mhz", "rpm",
    }
)

#: A terminator is a full stop, question mark, or exclamation mark, optionally
#: followed by a closing quote or bracket, then whitespace.
_TERMINATOR = re.compile(r'([.!?]+)(["\')\]]*)(\s+|$)')


def _ends_with_abbreviation(text: str) -> bool:
    """Whether ``text`` ends in a known abbreviation rather than a sentence."""
    tail = re.split(r"[\s(]", text.strip())[-1].lower().rstrip(".")
    if not tail:
        return False
    return tail in _ABBREVIATIONS


def _is_decimal_boundary(text: str, index: int) -> bool:
    """Whether the stop at ``index`` sits inside a number such as 43.2."""
    if index == 0 or index + 1 >= len(text):
        return False
    return text[index - 1].isdigit() and text[index + 1].isdigit()


def _is_initial(text: str, index: int) -> bool:
    """Whether the stop at ``index`` follows a single capital, as in J. Smith."""
    if index < 1:
        return False
    before = text[:index].rstrip()
    if len(before) < 1 or not before[-1].isupper():
        return False
    return len(before) == 1 or not before[-2].isalpha()


def split_sentences(text: str) -> list[str]:
    """Split text into speakable sentences.

    Does not break on abbreviations, decimal numbers, version strings, or single
    letter initials.

    Args:
        text: Arbitrary prose.

    Returns:
        Sentences with their terminating punctuation, whitespace stripped.
    """
    if not text or not text.strip():
        return []

    sentences: list[str] = []
    start = 0
    for match in _TERMINATOR.finditer(text):
        stop_index = match.start()
        candidate = text[start : match.end(2)]

        if _is_decimal_boundary(text, stop_index):
            continue
        if _is_initial(text, stop_index):
            continue
        if _ends_with_abbreviation(text[start:stop_index]):
            continue

        trimmed = candidate.strip()
        if trimmed:
            sentences.append(trimmed)
        start = match.end()

    remainder = text[start:].strip()
    if remainder:
        sentences.append(remainder)
    return sentences


class SentenceChunker:
    """Accumulates streaming deltas and emits speakable chunks early.

    The trade-off is stated in the config: below ``min_chunk_chars`` a fragment
    is held back so the assistant does not utter two words and stop, and above
    ``max_chunk_chars`` a chunk is flushed even without punctuation so a run-on
    sentence still starts playing.
    """

    def __init__(self, config: JarvisConfig) -> None:
        self._min = config.tts.min_chunk_chars
        self._max = config.tts.max_chunk_chars
        self._buffer = ""

    @property
    def pending(self) -> str:
        """Text held back so far."""
        return self._buffer

    def feed(self, delta: str) -> list[str]:
        """Add streamed text and return whatever is ready to synthesise now."""
        if not delta:
            return []
        self._buffer += delta
        ready: list[str] = []

        while True:
            sentences = split_sentences(self._buffer)
            # The last piece may be incomplete, so it is never emitted here;
            # only a sentence with something after it is certainly finished.
            if len(sentences) < 2:
                break
            complete = sentences[0]
            # Too short to speak alone: merge it with the next sentence by
            # leaving it in the buffer, unless the buffer is already long.
            if len(complete) < self._min and len(self._buffer) < self._max:
                break
            ready.append(complete)
            index = self._buffer.find(complete)
            if index < 0:
                # Cannot happen: split_sentences returns substrings of the
                # buffer. Guarded anyway because failing to shrink the buffer
                # here would spin forever inside the turn loop.
                _log.error("chunker lost track of its buffer, flushing it")
                self._buffer = ""
                break
            self._buffer = self._buffer[index + len(complete) :].lstrip()

        if len(self._buffer) >= self._max:
            # A run-on with no terminator. Break at the last space so a word is
            # not cut in half.
            cut = self._buffer.rfind(" ", 0, self._max)
            if cut <= 0:
                cut = self._max
            ready.append(self._buffer[:cut].strip())
            self._buffer = self._buffer[cut:].lstrip()

        return [chunk for chunk in ready if chunk]

    def flush(self) -> list[str]:
        """Emit whatever remains at the end of the stream."""
        remaining = self._buffer.strip()
        self._buffer = ""
        return [remaining] if remaining else []

    def reset(self) -> None:
        """Discard buffered text, for example after a barge-in."""
        self._buffer = ""


class KokoroSynthesizer:
    """Kokoro-82M text to speech.

    Args:
        config: Supplies the voice, language code, speed, and sample rate.
        pipeline: A preloaded Kokoro pipeline. Tests inject a fake, so nothing
            here needs the real 82M model or its weights.
    """

    def __init__(self, config: JarvisConfig, pipeline: Any | None = None) -> None:
        self._config = config
        self._pipeline = pipeline
        self._voice = config.tts.voice
        self._lang_code = config.tts.lang_code
        self._speed = config.tts.speed
        self._sample_rate = config.tts.sample_rate

    @property
    def sample_rate(self) -> int:
        """Output sample rate in hertz."""
        return self._sample_rate

    @property
    def voice(self) -> str:
        """The configured voice."""
        return self._voice

    @property
    def is_loaded(self) -> bool:
        """Whether the pipeline has been built."""
        return self._pipeline is not None

    def _ensure_pipeline(self) -> Any:
        """Load Kokoro on first use."""
        if self._pipeline is not None:
            return self._pipeline
        kokoro = require_module("kokoro", feature="speech synthesis")
        try:
            self._pipeline = kokoro.KPipeline(lang_code=self._lang_code)
        except Exception as exc:
            raise TtsError(
                f"could not load Kokoro: {exc}",
                speakable="I could not load my voice.",
            ) from exc
        _log.info(
            "kokoro loaded",
            extra={"context": {"voice": self._voice, "lang_code": self._lang_code}},
        )
        return self._pipeline

    def synthesize(self, text: str) -> Samples:
        """Render one chunk of text to audio.

        Args:
            text: The text to speak.

        Returns:
            float32 mono audio at :attr:`sample_rate`. Empty for empty input.

        Raises:
            TtsError: Synthesis failed.
        """
        if not text or not text.strip():
            return np.zeros(0, dtype=np.float32)

        pipeline = self._ensure_pipeline()
        try:
            pieces: list[Samples] = []
            for result in pipeline(text, voice=self._voice, speed=self._speed):
                audio = _extract_audio(result)
                if audio is not None and audio.size:
                    pieces.append(audio)
        except Exception as exc:
            _log.exception("synthesis failed")
            raise TtsError(
                f"Kokoro failed to synthesise: {exc}",
                speakable="I could not speak that.",
            ) from exc

        if not pieces:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(pieces).astype(np.float32, copy=False)

    def stream(self, chunks: Iterable[str]) -> Iterator[Samples]:
        """Synthesise chunks lazily, yielding each as soon as it is ready.

        Laziness is the point: the caller can start playing the first chunk
        while this generator is still working on the second, which is what keeps
        time to first audio inside the §3 budget.
        """
        for chunk in chunks:
            audio = self.synthesize(chunk)
            if audio.size:
                yield audio


def _extract_audio(result: Any) -> Samples | None:
    """Pull the waveform out of whatever shape Kokoro returned.

    The pipeline yields a named tuple in current versions and a bare array in
    older ones, so both are handled rather than pinning a version.
    """
    candidate = result
    if isinstance(result, tuple):
        # Current Kokoro yields (graphemes, phonemes, audio).
        candidate = result[-1]
    else:
        for attribute in ("audio", "output", "wav"):
            if hasattr(result, attribute):
                candidate = getattr(result, attribute)
                break

    if candidate is None:
        return None
    if hasattr(candidate, "detach"):  # a torch tensor
        candidate = candidate.detach().cpu().numpy()
    try:
        array = np.asarray(candidate, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    return array.reshape(-1) if array.ndim > 1 else array


def build_synthesizer(config: JarvisConfig, *, pipeline: Any | None = None) -> KokoroSynthesizer:
    """Build the configured synthesiser."""
    return KokoroSynthesizer(config, pipeline=pipeline)
