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

from jarvis.audio.effects import VoiceEffect, build_voice_effect
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
    """Kokoro-82M text to speech, followed by the JARVIS voice rack.

    Kokoro supplies the accent and the register. What makes the output sound
    like the assistant rather than like a person reading is
    :mod:`jarvis.audio.effects`, applied to every chunk on the way out. It is
    part of the voice, not a decoration, so it lives here rather than in the
    player: anything that asks this class to speak gets the same voice.

    Args:
        config: Supplies the voice, language code, speed, sample rate, and
            which effect profile to apply.
        pipeline: A preloaded Kokoro pipeline. Tests inject a fake, so nothing
            here needs the real 82M model or its weights.
        effect: An override for the voice rack. Defaults to whatever the config
            asks for; pass a bypassed one to hear raw synthesis.
    """

    def __init__(
        self,
        config: JarvisConfig,
        pipeline: Any | None = None,
        effect: VoiceEffect | None = None,
    ) -> None:
        self._config = config
        self._pipeline = pipeline
        self._voice = config.tts.voice
        self._lang_code = config.tts.lang_code
        self._speed = config.tts.speed
        self._sample_rate = config.tts.sample_rate
        self._effect = effect if effect is not None else build_voice_effect(config)

    @property
    def sample_rate(self) -> int:
        """Output sample rate in hertz."""
        return self._sample_rate

    @property
    def voice(self) -> str:
        """The configured voice."""
        return self._voice

    @property
    def effect(self) -> VoiceEffect:
        """The voice rack applied after synthesis."""
        return self._effect

    def begin_utterance(self) -> None:
        """Start a fresh reply.

        Clears the rack's delay lines so the previous reply's reverb tail does
        not lead into this one. Within a reply the state is deliberately kept:
        sentences are synthesised separately but played back to back, and
        resetting between them would click on every full stop.
        """
        self._effect.reset()

    @property
    def is_loaded(self) -> bool:
        """Whether the pipeline has been built."""
        return self._pipeline is not None

    def _local_weights(self) -> tuple[Any, Any] | None:
        """The weights ``pull_models.ps1`` downloaded, when they are present.

        Returns ``(config.json, kokoro-v1_0.pth)`` or None.

        Setup fetches about 330 MB into ``models/kokoro`` and nothing read it:
        ``KPipeline(lang_code=...)`` with no repo_id sends KModel to
        ``hf_hub_download``, so the weights came down a second time into the
        HuggingFace cache, and a machine that had been through setup and then
        gone offline could not speak at all.
        """
        root = self._config.models_dir / "kokoro"
        config_file = root / "config.json"
        weights = root / "kokoro-v1_0.pth"
        if config_file.is_file() and weights.is_file():
            return config_file, weights
        return None

    def _preload_voice(self, pipeline: Any) -> None:
        """Put the vendored voice pack into the pipeline before it asks the hub.

        Loading the weights locally is only half of going offline. The voice is
        a separate file, and ``KPipeline.load_single_voice`` reaches for
        ``hf_hub_download`` unless the voice is already in its ``voices`` dict.
        ``pull_models.ps1`` downloads ``kokoro/voices/bm_george.pt`` for exactly
        this, so it goes in the dict.

        Never raises: a voice that will not preload is one hub request away, and
        that is a far better outcome than refusing to speak.
        """
        pack = self._config.models_dir / "kokoro" / "voices" / f"{self._voice}.pt"
        if not pack.is_file():
            return
        try:
            import torch

            pipeline.voices[self._voice] = torch.load(str(pack), weights_only=True)
        except Exception:  # noqa: BLE001 - the hub path still works
            _log.debug("could not preload the vendored voice pack", exc_info=True)

    def _ensure_pipeline(self) -> Any:
        """Load Kokoro on first use, preferring the weights setup downloaded."""
        if self._pipeline is not None:
            return self._pipeline
        kokoro = require_module("kokoro", feature="speech synthesis")

        local = self._local_weights()
        if local is not None:
            config_file, weights = local
            try:
                model = kokoro.KModel(config=str(config_file), model=str(weights))
                self._pipeline = kokoro.KPipeline(lang_code=self._lang_code, model=model)
            except Exception:  # noqa: BLE001 - fall back rather than refuse to speak
                _log.warning(
                    "could not load the vendored Kokoro weights, falling back to the hub",
                    extra={"context": {"weights": str(weights)}},
                    exc_info=True,
                )
            else:
                self._preload_voice(self._pipeline)
                _log.info(
                    "kokoro loaded from models/",
                    extra={
                        "context": {
                            "voice": self._voice,
                            "lang_code": self._lang_code,
                            "weights": str(weights),
                        }
                    },
                )
                return self._pipeline

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
        raw = np.concatenate(pieces).astype(np.float32, copy=False)
        return self._effect.process(raw)

    def stream(self, chunks: Iterable[str]) -> Iterator[Samples]:
        """Synthesise chunks lazily, yielding each as soon as it is ready.

        Laziness is the point: the caller can start playing the first chunk
        while this generator is still working on the second, which is what keeps
        time to first audio inside the §3 budget.

        Treated as one utterance, so the voice rack is reset once at the start
        and then carries its state across the chunks.
        """
        self.begin_utterance()
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


def build_synthesizer(
    config: JarvisConfig,
    *,
    pipeline: Any | None = None,
    effect: VoiceEffect | None = None,
) -> KokoroSynthesizer:
    """Build the configured synthesiser, voice rack included."""
    return KokoroSynthesizer(config, pipeline=pipeline, effect=effect)
