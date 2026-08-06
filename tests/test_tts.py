"""T-1.5 verification: sentence chunking, synthesis, and interruptible playback."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from jarvis.audio.player import NullSink, StreamingPlayer
from jarvis.audio.tts import KokoroSynthesizer, SentenceChunker, split_sentences
from jarvis.config import JarvisConfig, load_config
from jarvis.util.errors import TtsError


class TestSplitSentences:
    def test_simple_split(self) -> None:
        assert split_sentences("One. Two. Three.") == ["One.", "Two.", "Three."]

    def test_question_and_exclamation(self) -> None:
        assert split_sentences("Really? Yes! Fine.") == ["Really?", "Yes!", "Fine."]

    def test_empty_input(self) -> None:
        assert split_sentences("") == []
        assert split_sentences("   ") == []

    def test_no_terminator_is_one_sentence(self) -> None:
        assert split_sentences("no full stop here") == ["no full stop here"]

    @pytest.mark.parametrize(
        "text",
        [
            "The CPU is at 43.2 percent right now.",
            "You have 12.5 GB free.",
            "It is 99.99 percent done.",
        ],
    )
    def test_decimals_do_not_split(self, text: str) -> None:
        """Breaking mid-number makes the assistant pause in the middle of it."""
        assert split_sentences(text) == [text]

    @pytest.mark.parametrize(
        "text",
        [
            "Dr. Smith called about it.",
            "Mr. Jones is here now.",
            "That is e.g. a good idea.",
            "Use approx. ten of them.",
            "See vol. three for that.",
        ],
    )
    def test_abbreviations_do_not_split(self, text: str) -> None:
        assert split_sentences(text) == [text]

    def test_version_strings_do_not_split(self) -> None:
        assert split_sentences("Version 1.2.3 is installed.") == [
            "Version 1.2.3 is installed."
        ]

    def test_initials_do_not_split(self) -> None:
        assert split_sentences("J. Smith wrote it.") == ["J. Smith wrote it."]

    def test_abbreviation_then_a_real_boundary(self) -> None:
        assert split_sentences("Dr. Smith called. He left a message.") == [
            "Dr. Smith called.",
            "He left a message.",
        ]

    def test_ellipsis_is_one_terminator(self) -> None:
        assert len(split_sentences("Well... that is odd.")) <= 2

    def test_closing_quote_stays_attached(self) -> None:
        assert split_sentences('He said "yes." Then he left.') == [
            'He said "yes."',
            "Then he left.",
        ]


class TestSentenceChunker:
    def _chunker(self, tmp_path: Path, **tts: object) -> SentenceChunker:
        base: dict[str, object] = {"min_chunk_chars": 10, "max_chunk_chars": 100}
        base.update(tts)
        return SentenceChunker(load_config(tmp_path / "absent.yaml", tts=base))

    def test_emits_a_complete_sentence_early(self, tmp_path: Path) -> None:
        """§3: playback starts on the first sentence, not the last."""
        chunker = self._chunker(tmp_path)
        assert chunker.feed("The processor is light. ") == []
        ready = chunker.feed("Memory is comfortable. ")
        assert ready == ["The processor is light."]

    def test_holds_an_incomplete_sentence(self, tmp_path: Path) -> None:
        chunker = self._chunker(tmp_path)
        assert chunker.feed("The processor is") == []
        assert chunker.pending == "The processor is"

    def test_flush_returns_the_remainder(self, tmp_path: Path) -> None:
        chunker = self._chunker(tmp_path)
        chunker.feed("A trailing thought")
        assert chunker.flush() == ["A trailing thought"]

    def test_flush_is_empty_when_drained(self, tmp_path: Path) -> None:
        assert self._chunker(tmp_path).flush() == []

    def test_short_fragments_are_held_back(self, tmp_path: Path) -> None:
        """Speaking "Yes." alone then stopping sounds broken."""
        chunker = self._chunker(tmp_path, min_chunk_chars=40)
        ready = chunker.feed("Yes. No. ")
        assert ready == []

    def test_a_run_on_is_flushed_at_the_max(self, tmp_path: Path) -> None:
        chunker = self._chunker(tmp_path, max_chunk_chars=50)
        ready = chunker.feed("word " * 30)
        assert ready
        assert all(len(chunk) <= 50 for chunk in ready)

    def test_the_run_on_break_falls_on_a_space(self, tmp_path: Path) -> None:
        """Cutting mid-word would be audible."""
        chunker = self._chunker(tmp_path, max_chunk_chars=40)
        ready = chunker.feed("alpha bravo charlie delta echo foxtrot golf hotel india")
        assert ready
        assert not ready[0].endswith("-")
        for chunk in ready:
            assert chunk == chunk.strip()

    def test_empty_delta_is_a_no_op(self, tmp_path: Path) -> None:
        assert self._chunker(tmp_path).feed("") == []

    def test_reset_clears_the_buffer(self, tmp_path: Path) -> None:
        chunker = self._chunker(tmp_path)
        chunker.feed("half a thought")
        chunker.reset()
        assert chunker.pending == ""
        assert chunker.flush() == []

    def test_streaming_token_by_token(self, tmp_path: Path) -> None:
        """The real case: one or two characters at a time from the model."""
        chunker = self._chunker(tmp_path)
        emitted: list[str] = []
        for char in "The disk is healthy. Memory is fine. All good.":
            emitted.extend(chunker.feed(char))
        emitted.extend(chunker.flush())
        assert " ".join(emitted) == "The disk is healthy. Memory is fine. All good."


class TestChunkerFuzz:
    def test_never_loses_text_and_never_hangs(self, cfg: JarvisConfig) -> None:
        """Every character fed in must come back out, across arbitrary input.

        The chunker sits in the middle of the speech path, so losing text means
        the assistant silently drops part of its answer, and failing to shrink
        the buffer means the turn loop spins forever.
        """
        import random
        import string

        random.seed(7)
        alphabet = string.ascii_letters + " .!?,'\"()0123456789" + "..." + "\n\t"

        for _ in range(500):
            text = "".join(random.choice(alphabet) for _ in range(random.randint(0, 120)))
            chunker = SentenceChunker(cfg)
            out: list[str] = []
            index = 0
            while index < len(text):
                step = random.randint(1, 5)
                out.extend(chunker.feed(text[index : index + step]))
                index += step
            out.extend(chunker.flush())

            assert "".join("".join(out).split()) == "".join(text.split()), repr(text)

    @pytest.mark.parametrize(
        "pathological", [". . . . .", "!!!!!", "?" * 50, "a." * 100, ".\n.\n.\n", "..."]
    )
    def test_pathological_punctuation_terminates(
        self, cfg: JarvisConfig, pathological: str
    ) -> None:
        chunker = SentenceChunker(cfg)
        assert chunker.feed(pathological) + chunker.flush() is not None


class FakePipeline:
    """Stands in for Kokoro's KPipeline."""

    def __init__(self, samples_per_char: int = 100, fail: bool = False) -> None:
        self.samples_per_char = samples_per_char
        self.fail = fail
        self.calls: list[str] = []

    def __call__(self, text: str, voice: str = "", speed: float = 1.0) -> Any:
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("the model exploded")
        count = max(1, len(text) * self.samples_per_char)
        yield ("graphemes", "phonemes", np.linspace(-0.5, 0.5, count, dtype=np.float32))


class TestSynthesizer:
    def test_produces_audio(self, cfg: JarvisConfig) -> None:
        synth = KokoroSynthesizer(cfg, pipeline=FakePipeline())
        audio = synth.synthesize("Hello there.")
        assert audio.dtype == np.float32
        assert audio.size > 0

    def test_empty_text_produces_no_audio(self, cfg: JarvisConfig) -> None:
        synth = KokoroSynthesizer(cfg, pipeline=FakePipeline())
        assert synth.synthesize("   ").size == 0

    def test_the_voice_is_passed_through(self, cfg: JarvisConfig) -> None:
        assert KokoroSynthesizer(cfg, pipeline=FakePipeline()).voice == "bm_george"

    def test_stream_preserves_order(self, cfg: JarvisConfig) -> None:
        pipeline = FakePipeline()
        synth = KokoroSynthesizer(cfg, pipeline=pipeline)
        list(synth.stream(["First one.", "Second one.", "Third one."]))
        assert pipeline.calls == ["First one.", "Second one.", "Third one."]

    def test_stream_is_lazy(self, cfg: JarvisConfig) -> None:
        """Laziness is what lets chunk N play while N+1 is synthesised."""
        pipeline = FakePipeline()
        synth = KokoroSynthesizer(cfg, pipeline=pipeline)
        stream = synth.stream(["One.", "Two.", "Three."])

        next(stream)
        assert pipeline.calls == ["One."], "the whole list was synthesised up front"
        next(stream)
        assert pipeline.calls == ["One.", "Two."]

    def test_stream_skips_empty_chunks(self, cfg: JarvisConfig) -> None:
        synth = KokoroSynthesizer(cfg, pipeline=FakePipeline())
        assert len(list(synth.stream(["Real text.", "   ", "More text."]))) == 2

    def test_a_pipeline_failure_becomes_a_tts_error(self, cfg: JarvisConfig) -> None:
        synth = KokoroSynthesizer(cfg, pipeline=FakePipeline(fail=True))
        with pytest.raises(TtsError) as excinfo:
            synth.synthesize("anything")
        assert excinfo.value.speakable == "I could not speak that."

    def test_bare_array_results_are_accepted(self, cfg: JarvisConfig) -> None:
        """Older Kokoro yields a bare array rather than a tuple."""

        class BareArray:
            def __call__(self, text: str, voice: str = "", speed: float = 1.0) -> Any:
                yield np.zeros(1000, dtype=np.float32)

        assert KokoroSynthesizer(cfg, pipeline=BareArray()).synthesize("hi").size == 1000


class TestStreamingPlayer:
    def _player(self, cfg: JarvisConfig) -> tuple[StreamingPlayer, NullSink]:
        sink = NullSink()
        return StreamingPlayer(cfg, sink=sink, blocksize=256), sink

    def test_queue_and_drain(self, cfg: JarvisConfig) -> None:
        player, _sink = self._player(cfg)
        player.start()
        try:
            player.play(np.ones(512, dtype=np.float32))
            assert player.is_playing is True
            # Pull the audio through the way the callback would.
            player._next_block(512)
            assert player.is_playing is False
        finally:
            player.close()

    def test_queued_seconds_reflects_the_backlog(self, cfg: JarvisConfig) -> None:
        player, _sink = self._player(cfg)
        player.play(np.ones(cfg.tts.sample_rate, dtype=np.float32))
        assert player.queued_seconds == pytest.approx(1.0, rel=0.01)

    def test_empty_chunks_are_ignored(self, cfg: JarvisConfig) -> None:
        player, _sink = self._player(cfg)
        player.play(np.zeros(0, dtype=np.float32))
        assert player.is_playing is False

    def test_blocks_come_out_in_order(self, cfg: JarvisConfig) -> None:
        player, _sink = self._player(cfg)
        player.play(np.arange(4, dtype=np.float32))
        player.play(np.arange(4, 8, dtype=np.float32))
        block = player._next_block(8)
        assert list(block) == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_underrun_produces_silence_not_a_stall(self, cfg: JarvisConfig) -> None:
        player, _sink = self._player(cfg)
        player.play(np.ones(2, dtype=np.float32))
        block = player._next_block(6)
        assert list(block) == [1, 1, 0, 0, 0, 0]

    def test_stop_drops_queued_audio(self, cfg: JarvisConfig) -> None:
        """The barge-in primitive: everything queued must go, at once."""
        player, _sink = self._player(cfg)
        player.play(np.ones(cfg.tts.sample_rate * 10, dtype=np.float32))
        assert player.is_playing is True
        player.stop()
        assert player.is_playing is False
        assert player.queued_seconds == 0.0

    def test_the_next_block_after_stop_is_silence(self, cfg: JarvisConfig) -> None:
        player, _sink = self._player(cfg)
        player.play(np.ones(10_000, dtype=np.float32))
        player._next_block(256)
        player.stop()
        assert not player._next_block(256).any()

    def test_stop_is_well_under_the_deadline(self, cfg: JarvisConfig) -> None:
        """T-1.5: playback must cut within tts.stop_latency_ms, default 100."""
        player, _sink = self._player(cfg)
        player.start()
        try:
            player.play(np.ones(cfg.tts.sample_rate * 30, dtype=np.float32))
            elapsed_ms = player.measure_stop_latency()
            assert elapsed_ms < cfg.tts.stop_latency_ms, f"stop took {elapsed_ms:.1f} ms"
        finally:
            player.close()

    def test_stop_when_idle_is_safe(self, cfg: JarvisConfig) -> None:
        player, _sink = self._player(cfg)
        player.stop()

    def test_double_stop_is_safe(self, cfg: JarvisConfig) -> None:
        player, _sink = self._player(cfg)
        player.play(np.ones(100, dtype=np.float32))
        player.stop()
        player.stop()

    def test_stop_aborts_the_sink(self, cfg: JarvisConfig) -> None:
        """The device buffer holds audio the queue no longer knows about."""
        player, sink = self._player(cfg)
        player.start()
        try:
            player.play(np.ones(1000, dtype=np.float32))
            player.stop()
            assert sink.aborted is True
        finally:
            player.close()

    def test_wait_returns_once_drained(self, cfg: JarvisConfig) -> None:
        player, _sink = self._player(cfg)
        player.play(np.ones(256, dtype=np.float32))
        player._next_block(256)
        assert player.wait(1.0) is True

    def test_wait_times_out_while_audio_remains(self, cfg: JarvisConfig) -> None:
        player, _sink = self._player(cfg)
        player.play(np.ones(100_000, dtype=np.float32))
        assert player.wait(0.05) is False

    def test_context_manager(self, cfg: JarvisConfig) -> None:
        sink = NullSink()
        with StreamingPlayer(cfg, sink=sink):
            assert sink.started is True

    def test_audio_from_a_cancelled_turn_is_discarded(self, cfg: JarvisConfig) -> None:
        """The barge-in race.

        A chunk still being synthesised when the user interrupts would otherwise
        be queued a moment after stop() cleared everything, so the assistant
        resumes talking over the person who just cut it off.
        """
        player, _sink = self._player(cfg)
        generation = player.generation
        player.play(np.ones(1000, dtype=np.float32))

        player.stop()  # the user interrupts

        # The synthesiser finishes and hands over audio from the dead turn.
        player.play(np.ones(1000, dtype=np.float32), generation=generation)
        assert player.is_playing is False, "audio from a cancelled turn started playing"
        assert not player._next_block(256).any()

    def test_audio_from_the_current_generation_still_plays(self, cfg: JarvisConfig) -> None:
        player, _sink = self._player(cfg)
        player.stop()
        player.play(np.ones(500, dtype=np.float32), generation=player.generation)
        assert player.is_playing is True

    def test_generation_advances_on_every_stop(self, cfg: JarvisConfig) -> None:
        player, _sink = self._player(cfg)
        first = player.generation
        player.stop()
        assert player.generation != first

    def test_playback_can_resume_after_a_stop(self, cfg: JarvisConfig) -> None:
        """A barge-in must not permanently disable the speaker."""
        player, _sink = self._player(cfg)
        player.start()
        try:
            player.play(np.ones(1000, dtype=np.float32))
            player.stop()
            player.play(np.ones(500, dtype=np.float32))
            assert player.is_playing is True
            assert player._next_block(500).any()
        finally:
            player.close()


@pytest.mark.manual
class TestRealAudio:
    """Checks that need a real speaker.

    Run on the Windows host with:
        uv run pytest tests/test_tts.py -m manual -v

    Expect: a British male voice reads the sentence, and the second test cuts
    the audio dead within about a tenth of a second of the stop call.
    """

    def test_speaks_aloud(self, cfg: JarvisConfig) -> None:
        from jarvis.audio.tts import build_synthesizer

        synth = build_synthesizer(cfg)
        audio = synth.synthesize("The processor is running at forty three percent, sir.")
        with StreamingPlayer(cfg) as player:
            player.play(audio)
            player.wait(20)

    def test_stop_cuts_real_playback(self, cfg: JarvisConfig) -> None:
        from jarvis.audio.tts import build_synthesizer

        synth = build_synthesizer(cfg)
        audio = synth.synthesize(
            "This is a deliberately long sentence that should be cut off part way through, "
            "well before it reaches the end."
        )
        with StreamingPlayer(cfg) as player:
            player.play(audio)
            time.sleep(1.0)
            elapsed = player.measure_stop_latency()
            assert elapsed < cfg.tts.stop_latency_ms
