"""Round trip through the real Kokoro and faster-whisper engines.

Every other test in the suite injects a fake engine, which is what lets the
suite run on a build host with no GPU and no microphone. The cost is that the
call sites themselves, ``kokoro.KPipeline(...)`` and
``faster_whisper.WhisperModel(...)``, are the one part of the audio path that
nothing executes: a renamed keyword or a changed return shape would pass the
whole suite and fail on the first spoken word.

This module closes that gap. It needs no microphone and no human, only the two
optional extras and enough network to fetch the models once, so it can run on
the Linux build host as well as on the Windows target.

Marked ``manual`` because it downloads model weights on first run and takes
tens of seconds. Run it deliberately::

    uv sync --extra stt --extra tts
    uv run pytest tests/test_engines_live.py -m manual -v
"""

from __future__ import annotations

import numpy as np
import pytest

from jarvis.audio.stt import FasterWhisperTranscriber, build_transcriber
from jarvis.audio.tts import build_synthesizer
from jarvis.config import JarvisConfig, load_config
from jarvis.util.platform import has_module

pytestmark = pytest.mark.manual

needs_tts = pytest.mark.skipif(
    not has_module("kokoro"), reason="kokoro is not installed; uv sync --extra tts"
)
needs_stt = pytest.mark.skipif(
    not has_module("faster_whisper"),
    reason="faster-whisper is not installed; uv sync --extra stt",
)

SPOKEN = "What is the temperature of the processor right now."

#: What the transcriber demands. It refuses to resample rather than doing it
#: silently, so the test has to arrive at the rate the microphone would.
WHISPER_RATE = 16_000


@pytest.fixture
def live_config(tmp_path: object) -> JarvisConfig:
    """Force the faster-whisper path on the smallest model there is.

    The cpu tier would select whisper.cpp, whose package is published for
    Windows only, so the engine is pinned here instead of left to the tier.
    """
    return load_config(
        "/nonexistent.yaml",
        stt={
            "engine": "faster-whisper",
            "model": "tiny.en",
            "device": "cpu",
            "compute_type": "int8",
        },
    )


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Linear resample, standing in for what the capture stage delivers."""
    count = int(audio.size * target_rate / source_rate)
    return np.interp(
        np.linspace(0, audio.size - 1, count), np.arange(audio.size), audio
    ).astype(np.float32)


@needs_tts
class TestKokoroForReal:
    def test_synthesis_returns_audible_float32_mono(self, live_config: JarvisConfig) -> None:
        synth = build_synthesizer(live_config)
        assert synth.is_loaded is False, "the model must not load until first use"

        audio = synth.synthesize("Good evening, sir.")

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert audio.ndim == 1, "the player expects mono"
        assert audio.size > 0
        assert np.isfinite(audio).all(), "non-finite samples would click"
        assert float(np.abs(audio).max()) > 0.01, "silence: the voice failed to load"

    def test_the_reported_rate_is_the_rate_of_the_samples(
        self, live_config: JarvisConfig
    ) -> None:
        """A mismatch here plays every reply at the wrong pitch.

        Nothing else catches it: the player just reads ``config.tts.sample_rate``
        and trusts it.
        """
        synth = build_synthesizer(live_config)
        audio = synth.synthesize("One two three four five.")
        seconds = audio.size / synth.sample_rate
        assert 1.0 < seconds < 6.0, (
            f"{audio.size} samples at {synth.sample_rate} Hz is {seconds:.2f}s, "
            "which is not how long that sentence takes to say"
        )

    def test_empty_input_yields_empty_output(self, live_config: JarvisConfig) -> None:
        assert build_synthesizer(live_config).synthesize("").size == 0


@needs_stt
class TestFasterWhisperForReal:
    def test_the_configured_engine_is_the_one_built(
        self, live_config: JarvisConfig
    ) -> None:
        """A silent fallback to whisper.cpp would be a different engine entirely."""
        assert isinstance(build_transcriber(live_config), FasterWhisperTranscriber)

    def test_the_wrong_sample_rate_is_refused_not_resampled(
        self, live_config: JarvisConfig
    ) -> None:
        from jarvis.util.errors import SttError

        with pytest.raises(SttError, match="16000 Hz"):
            build_transcriber(live_config).transcribe(
                np.zeros(8000, dtype=np.float32), sample_rate=24_000
            )


@needs_tts
@needs_stt
class TestTheTwoEnginesAgree:
    """Speak a sentence and transcribe it back, with no fake in between."""

    def test_a_synthesised_sentence_transcribes_back(
        self, live_config: JarvisConfig
    ) -> None:
        synth = build_synthesizer(live_config)
        speech = _resample(synth.synthesize(SPOKEN), synth.sample_rate, WHISPER_RATE)

        transcript = build_transcriber(live_config).transcribe(
            speech, sample_rate=WHISPER_RATE
        )

        assert transcript.text.strip(), "nothing came back"
        heard = set(transcript.text.lower().replace(".", "").replace("?", "").split())
        said = set(SPOKEN.lower().replace(".", "").split())
        overlap = len(heard & said) / len(said)
        assert overlap >= 0.6, (
            f"heard {transcript.text!r} for {SPOKEN!r}, only {overlap:.0%} of words match"
        )

    def test_the_transcript_carries_usable_metadata(
        self, live_config: JarvisConfig
    ) -> None:
        synth = build_synthesizer(live_config)
        speech = _resample(synth.synthesize(SPOKEN), synth.sample_rate, WHISPER_RATE)

        transcript = build_transcriber(live_config).transcribe(
            speech, sample_rate=WHISPER_RATE
        )

        assert transcript.language == "en"
        assert transcript.segments, "no segments, so no timing information"
        assert transcript.duration_s == pytest.approx(speech.size / WHISPER_RATE, abs=0.5)


needs_vad = pytest.mark.skipif(
    not has_module("onnxruntime"),
    reason="onnxruntime is not installed; uv sync --extra audio",
)


@needs_vad
@needs_tts
class TestSileroForReal:
    """Score real speech through the real checkpoint.

    The unit tests use a fake session, so they can prove the wrapper feeds the
    shape it intends to and nothing more. They cannot notice that the shape
    itself is wrong. This one can: fed without its context window the model
    returns near zero for speech, which read as an endpointer that simply never
    fired, with nothing logged and nothing failing.
    """

    def test_speech_scores_far_above_silence(self, live_config: JarvisConfig) -> None:
        from jarvis.audio.vad import SileroVad

        synth = build_synthesizer(live_config)
        speech = _resample(synth.synthesize(SPOKEN), synth.sample_rate, WHISPER_RATE)

        detector = SileroVad(live_config)
        frame = detector.frame_samples
        if detector.find_model_file() is None:
            pytest.skip("silero_vad.onnx is not downloaded; run scripts/pull_models.ps1")

        spoken = [
            detector.probability(speech[i : i + frame])
            for i in range(0, speech.size - frame + 1, frame)
        ]
        detector.reset()
        quiet = np.zeros(speech.size, dtype=np.float32)
        silent = [
            detector.probability(quiet[i : i + frame])
            for i in range(0, quiet.size - frame + 1, frame)
        ]

        assert max(spoken) > 0.8, (
            f"clear speech peaked at {max(spoken):.3f}. Near zero here means the "
            "model is not getting the context window it expects."
        )
        assert max(silent) < 0.2, f"silence peaked at {max(silent):.3f}"

    def test_the_endpointer_completes_an_utterance(self, live_config: JarvisConfig) -> None:
        """The stage the assistant actually depends on, end to end."""
        from jarvis.audio.vad import Endpointer, SileroVad

        detector = SileroVad(live_config)
        if detector.find_model_file() is None:
            pytest.skip("silero_vad.onnx is not downloaded; run scripts/pull_models.ps1")

        synth = build_synthesizer(live_config)
        speech = _resample(synth.synthesize(SPOKEN), synth.sample_rate, WHISPER_RATE)
        # Trailing silence, so the endpointer has an end to find.
        padded = np.concatenate([speech, np.zeros(WHISPER_RATE, dtype=np.float32)])

        endpointer = Endpointer(live_config, detector)
        frame = endpointer.frame_samples
        endpoint = None
        for i in range(0, padded.size - frame + 1, frame):
            endpoint = endpointer.process(padded[i : i + frame])
            if endpoint is not None:
                break

        assert endpoint is not None, (
            "the endpointer never completed an utterance from clear speech, "
            "which is the silent failure the assistant showed"
        )
        assert endpoint.audio.size > 0
