"""The voice rack.

Two kinds of test live here. Most assert the invariants that make it safe to
put an effects chain in the audio path at all: it must not clip, must not go
non-finite, must not change the length of the audio, must not run away over a
long reply, and must not change how loud the assistant is. Those are the ones
that would produce a broken or unusable voice if they regressed.

The rest assert that chunking is invisible. A reply is synthesised sentence by
sentence and played back to back, so anything in the rack whose output depends
on where the sentence chunker cut would make the same words come out
differently depending on their punctuation.
"""

from __future__ import annotations

import numpy as np
import pytest

from jarvis.audio.effects import (
    PROFILES,
    EffectSpec,
    VoiceEffect,
    _Band,
    _feedback_comb,
    _RingModulator,
    build_voice_effect,
    describe_profile,
    design_eq_kernel,
    soft_limit,
    speech_shaped_noise,
)
from jarvis.config import JarvisConfig, VoiceProfile

RATE = 24_000
CHARACTER = (VoiceProfile.CLEAN, VoiceProfile.JARVIS, VoiceProfile.ROBOT)


@pytest.fixture
def speech() -> np.ndarray:
    """Two seconds of speech shaped noise at a realistic level."""
    return speech_shaped_noise(RATE, 2.0, -20.0)


def rms_dbfs(block: np.ndarray) -> float:
    """Level of a block in dBFS."""
    value = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
    return 20.0 * np.log10(max(value, 1e-12))


class TestBypass:
    """Intensity 0 and profile none must be genuinely nothing, not near nothing."""

    def test_profile_none_returns_the_same_array(self, speech: np.ndarray) -> None:
        effect = VoiceEffect(RATE, VoiceProfile.NONE)
        assert effect.is_bypass
        assert effect.process(speech) is speech

    @pytest.mark.parametrize("profile", CHARACTER)
    def test_zero_intensity_returns_the_same_array(
        self, profile: VoiceProfile, speech: np.ndarray
    ) -> None:
        effect = VoiceEffect(RATE, profile, intensity=0.0)
        assert effect.is_bypass
        assert effect.process(speech) is speech

    def test_bypass_reset_is_harmless(self) -> None:
        VoiceEffect(RATE, VoiceProfile.NONE).reset()

    def test_negative_intensity_is_clamped_to_bypass(self, speech: np.ndarray) -> None:
        assert VoiceEffect(RATE, VoiceProfile.JARVIS, intensity=-1.0).is_bypass

    def test_intensity_above_one_is_clamped(self) -> None:
        assert VoiceEffect(RATE, VoiceProfile.JARVIS, intensity=5.0).intensity == 1.0


class TestSafety:
    """What must hold before this goes anywhere near a speaker."""

    @pytest.mark.parametrize("profile", CHARACTER)
    @pytest.mark.parametrize("intensity", [0.25, 0.75, 1.0])
    def test_output_is_finite_and_in_range(
        self, profile: VoiceProfile, intensity: float, speech: np.ndarray
    ) -> None:
        out = VoiceEffect(RATE, profile, intensity).process(speech)
        assert np.isfinite(out).all()
        assert float(np.abs(out).max()) <= 1.0

    @pytest.mark.parametrize("profile", CHARACTER)
    def test_length_and_dtype_are_preserved(
        self, profile: VoiceProfile, speech: np.ndarray
    ) -> None:
        out = VoiceEffect(RATE, profile).process(speech)
        assert out.shape == speech.shape
        assert out.dtype == np.float32

    @pytest.mark.parametrize("profile", CHARACTER)
    def test_a_loud_input_does_not_clip(self, profile: VoiceProfile) -> None:
        """Full scale in must not come out as a square wave."""
        loud = np.clip(speech_shaped_noise(RATE, 1.0, -6.0), -1.0, 1.0)
        out = VoiceEffect(RATE, profile).process(loud)
        assert float(np.abs(out).max()) <= 1.0
        # Hard clipping shows up as a pile of samples sitting exactly at the rail.
        assert int((np.abs(out) >= 0.999).sum()) < out.size // 100

    @pytest.mark.parametrize("profile", CHARACTER)
    def test_it_does_not_run_away_over_a_long_reply(
        self, profile: VoiceProfile, speech: np.ndarray
    ) -> None:
        """Feedback loops that creep would take a minute of speech to show it."""
        effect = VoiceEffect(RATE, profile)
        peaks = [float(np.abs(effect.process(speech)).max()) for _ in range(30)]
        assert np.isfinite(peaks).all()
        assert max(peaks) <= 1.0
        assert max(peaks[-5:]) < max(peaks[:5]) * 1.5

    @pytest.mark.parametrize("profile", CHARACTER)
    def test_silence_stays_silent(self, profile: VoiceProfile) -> None:
        silence = np.zeros(RATE, dtype=np.float32)
        assert float(np.abs(VoiceEffect(RATE, profile).process(silence)).max()) == 0.0

    def test_empty_input_is_returned_unchanged(self) -> None:
        empty = np.zeros(0, dtype=np.float32)
        assert VoiceEffect(RATE, VoiceProfile.JARVIS).process(empty).size == 0

    @pytest.mark.parametrize("size", [1, 2, 7, 63, 256, 257])
    def test_very_short_blocks_are_handled(self, size: int) -> None:
        """The sentence chunker can emit a two word chunk, and combs and
        envelope blocks are both longer than that."""
        tiny = speech_shaped_noise(RATE, 1.0, -20.0)[:size]
        out = VoiceEffect(RATE, VoiceProfile.JARVIS).process(tiny)
        assert out.shape == tiny.shape
        assert np.isfinite(out).all()

    def test_a_stage_failure_falls_back_to_the_dry_voice(
        self, speech: np.ndarray, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cosmetic stage must never be able to silence the assistant."""
        effect = VoiceEffect(RATE, VoiceProfile.JARVIS)

        def explode(*_: object, **__: object) -> np.ndarray:
            raise RuntimeError("boom")

        monkeypatch.setattr(effect, "_run", explode)
        assert effect.process(speech) is speech


class TestLoudness:
    """A profile is a character, not a volume control."""

    @pytest.mark.parametrize("profile", CHARACTER)
    @pytest.mark.parametrize("intensity", [0.25, 0.5, 0.75, 1.0])
    def test_level_is_preserved_within_a_decibel(
        self, profile: VoiceProfile, intensity: float, speech: np.ndarray
    ) -> None:
        out = VoiceEffect(RATE, profile, intensity).process(speech)
        assert abs(rms_dbfs(out) - rms_dbfs(speech)) < 1.0

    @pytest.mark.parametrize("profile", CHARACTER)
    def test_calibration_had_to_do_something(self, profile: VoiceProfile) -> None:
        """If this ever reads zero the calibration silently stopped running."""
        assert VoiceEffect(RATE, profile).calibration_db > 0.0

    def test_calibration_is_deterministic(self) -> None:
        """Two racks built the same way must be the same voice."""
        first = VoiceEffect(RATE, VoiceProfile.JARVIS, 0.75)
        second = VoiceEffect(RATE, VoiceProfile.JARVIS, 0.75)
        assert first.calibration_db == pytest.approx(second.calibration_db)

    def test_calibration_leaves_no_state_behind(self, speech: np.ndarray) -> None:
        """The probe must not leave its own reverb tail in the first sentence."""
        effect = VoiceEffect(RATE, VoiceProfile.JARVIS)
        after_build = effect.process(np.zeros(RATE // 2, dtype=np.float32))
        assert float(np.abs(after_build).max()) == 0.0


class TestChunkingIsInvisible:
    """The rack must not care where the sentence chunker cut."""

    @pytest.mark.parametrize("profile", CHARACTER)
    @pytest.mark.parametrize("size", [1_000, 4_000, 7_331])
    def test_chunked_matches_whole(
        self, profile: VoiceProfile, size: int, speech: np.ndarray
    ) -> None:
        whole = VoiceEffect(RATE, profile, 0.75).process(speech)
        chunked = VoiceEffect(RATE, profile, 0.75)
        pieces = [chunked.process(speech[i : i + size]) for i in range(0, speech.size, size)]
        joined = np.concatenate(pieces)
        # Every stage but the compressor is exact here. The compressor tracks
        # its envelope per block, so a cut landing inside a block shifts one
        # gain knot. Measured as a residual rather than as a sample distance,
        # because a slow gain difference and a click are worlds apart to listen
        # to and identical to subtract. A regression means a stage stopped
        # carrying its state, and that shows up as tens of dB, not tenths.
        assert rms_dbfs(whole - joined) < rms_dbfs(whole) - 40.0
        assert abs(rms_dbfs(whole) - rms_dbfs(joined)) < 0.1

    def test_reset_makes_the_rack_repeat_itself(self, speech: np.ndarray) -> None:
        effect = VoiceEffect(RATE, VoiceProfile.JARVIS)
        first = effect.process(speech)
        effect.reset()
        assert np.allclose(effect.process(speech), first, atol=1e-6)

    def test_state_carries_without_a_reset(self, speech: np.ndarray) -> None:
        """The opposite of the above: the tail of one sentence must reach the next."""
        effect = VoiceEffect(RATE, VoiceProfile.JARVIS)
        first = effect.process(speech)
        assert not np.allclose(effect.process(speech), first, atol=1e-6)


class TestEqDesign:
    """The filter designer, which every other stage depends on."""

    def test_no_bands_is_a_unit_impulse(self) -> None:
        kernel = design_eq_kernel((), RATE, taps=65)
        assert kernel.argmax() == 32
        assert kernel[32] == pytest.approx(1.0)
        assert float(np.abs(np.delete(kernel, 32)).max()) == pytest.approx(0.0)

    def test_taps_are_forced_odd_so_the_delay_is_whole(self) -> None:
        assert len(design_eq_kernel((), RATE, taps=64)) == 65

    def test_it_is_linear_phase(self) -> None:
        kernel = design_eq_kernel((_Band("peak", 3_000.0, gain_db=6.0),), RATE, taps=129)
        assert np.allclose(kernel, kernel[::-1], atol=1e-9)

    def test_a_highpass_removes_low_frequencies(self) -> None:
        kernel = design_eq_kernel((_Band("highpass", 300.0, order=3),), RATE, taps=257)
        response = np.abs(np.fft.rfft(kernel, n=8_192))
        freqs = np.fft.rfftfreq(8_192, d=1.0 / RATE)
        assert response[np.argmin(np.abs(freqs - 50.0))] < 0.1
        assert response[np.argmin(np.abs(freqs - 2_000.0))] > 0.9

    def test_a_lowpass_removes_high_frequencies(self) -> None:
        kernel = design_eq_kernel((_Band("lowpass", 4_000.0, order=3),), RATE, taps=257)
        response = np.abs(np.fft.rfft(kernel, n=8_192))
        freqs = np.fft.rfftfreq(8_192, d=1.0 / RATE)
        assert response[np.argmin(np.abs(freqs - 1_000.0))] > 0.9
        assert response[np.argmin(np.abs(freqs - 10_000.0))] < 0.1

    def test_a_peak_lifts_its_own_band_and_leaves_the_rest(self) -> None:
        kernel = design_eq_kernel((_Band("peak", 3_000.0, gain_db=6.0, width=0.8),), RATE, 257)
        response = np.abs(np.fft.rfft(kernel, n=8_192))
        freqs = np.fft.rfftfreq(8_192, d=1.0 / RATE)
        at_peak = response[np.argmin(np.abs(freqs - 3_000.0))]
        far_below = response[np.argmin(np.abs(freqs - 250.0))]
        assert at_peak == pytest.approx(2.0, rel=0.15)
        assert far_below == pytest.approx(1.0, rel=0.15)

    def test_an_unknown_band_kind_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown band kind"):
            design_eq_kernel((_Band("notch", 1_000.0),), RATE)


class TestFeedbackComb:
    """The vectorised recursion, checked against the definition it replaces."""

    def test_it_matches_the_sample_loop(self) -> None:
        rng = np.random.default_rng(7)
        block = rng.standard_normal(2_000).astype(np.float32)
        delay, feedback = 137, 0.6

        expected = np.zeros_like(block)
        for n in range(len(block)):
            expected[n] = block[n] + (feedback * expected[n - delay] if n >= delay else 0.0)

        out, state = _feedback_comb(block, np.zeros(delay, dtype=np.float32), feedback)
        assert np.allclose(out, expected, atol=1e-4)
        assert np.allclose(state, expected[-delay:], atol=1e-4)

    def test_state_carries_across_calls(self) -> None:
        rng = np.random.default_rng(11)
        block = rng.standard_normal(1_500).astype(np.float32)
        delay, feedback = 91, 0.5

        whole, _ = _feedback_comb(block, np.zeros(delay, dtype=np.float32), feedback)
        state = np.zeros(delay, dtype=np.float32)
        pieces = []
        for start in range(0, len(block), 200):
            piece, state = _feedback_comb(block[start : start + 200], state, feedback)
            pieces.append(piece)
        assert np.allclose(np.concatenate(pieces), whole, atol=1e-4)

    def test_a_block_shorter_than_the_delay_still_tracks(self) -> None:
        delay = 500
        state = np.zeros(delay, dtype=np.float32)
        for _ in range(20):
            _, state = _feedback_comb(np.ones(50, dtype=np.float32), state, 0.5)
        assert np.isfinite(state).all()

    def test_an_empty_state_is_a_pass_through(self) -> None:
        block = np.ones(10, dtype=np.float32)
        out, state = _feedback_comb(block, np.zeros(0, dtype=np.float32), 0.5)
        assert out is block
        assert state.size == 0


class TestRingModulatorBandSplit:
    """The stage that decides whether the words survive.

    Ring modulation smears every component it touches by plus and minus the
    carrier. Applied across the whole band it wrecks the formants that carry the
    words, which is measurable: full band, a transcriber hears "but even so" for
    "good evening sir". Restricting it to the low band keeps the buzz and costs
    nothing, and these tests pin that down without needing a speech model.
    """

    def _modulator(self, depth: float = 0.5, split_hz: float = 1_100.0) -> _RingModulator:
        return _RingModulator(RATE, carrier_hz=56.0, depth=depth, split_hz=split_hz)

    def _tone(self, hz: float, seconds: float = 0.5) -> np.ndarray:
        t = np.arange(int(RATE * seconds)) / RATE
        return (0.3 * np.sin(2.0 * np.pi * hz * t)).astype(np.float32)

    def _energy_at(self, block: np.ndarray, hz: float, width: float = 25.0) -> float:
        spectrum = np.abs(np.fft.rfft(block.astype(np.float64))) ** 2
        freqs = np.fft.rfftfreq(len(block), d=1.0 / RATE)
        return float(spectrum[np.abs(freqs - hz) <= width].sum())

    def test_a_formant_passes_through_untouched(self) -> None:
        """3 kHz is where the words live. It must come out as it went in.

        The bound is set between the two regimes rather than at the measured
        value. Modulating the full band at this depth puts half the carrier's
        energy into sidebands; the band split leaves 0.0015 of it. A hundredth
        is far enough below the first to catch a regression and far enough above
        the second not to flake on the filter's stopband.
        """
        tone = self._tone(3_000.0)
        out = self._modulator().process(tone)
        carried = self._energy_at(out, 3_000.0)
        sidebands = self._energy_at(out, 3_056.0) + self._energy_at(out, 2_944.0)
        assert sidebands < carried * 1e-2

    def test_the_low_band_does_get_modulated(self) -> None:
        """Otherwise the split has simply switched the effect off."""
        tone = self._tone(300.0)
        out = self._modulator().process(tone)
        sidebands = self._energy_at(out, 356.0) + self._energy_at(out, 244.0)
        assert sidebands > self._energy_at(out, 300.0) * 1e-3

    def test_the_split_reconstructs_exactly_at_zero_depth(self) -> None:
        """The bands are complementary, so nothing is lost by splitting."""
        tone = self._tone(2_000.0) + self._tone(400.0)
        out = _RingModulator(RATE, 56.0, depth=0.0).process(tone)
        assert np.allclose(out, tone, atol=1e-6)

    def test_state_carries_so_the_split_does_not_click(self) -> None:
        tone = self._tone(1_500.0, seconds=1.0)
        whole = self._modulator().process(tone)
        chunked = self._modulator()
        joined = np.concatenate(
            [chunked.process(tone[i : i + 700]) for i in range(0, tone.size, 700)]
        )
        assert np.allclose(whole, joined, atol=1e-5)

    def test_reset_clears_the_band_split(self) -> None:
        modulator = self._modulator()
        tone = self._tone(800.0)
        first = modulator.process(tone)
        modulator.reset()
        assert np.allclose(modulator.process(tone), first, atol=1e-6)


class TestSoftLimit:
    """The last thing between the rack and the speaker."""

    def test_quiet_audio_is_untouched(self) -> None:
        block = np.linspace(-0.5, 0.5, 100, dtype=np.float32)
        assert soft_limit(block) is block

    def test_loud_audio_is_bounded(self) -> None:
        block = np.linspace(-4.0, 4.0, 1_000, dtype=np.float32)
        out = soft_limit(block)
        assert float(np.abs(out).max()) <= 1.0

    def test_it_stays_monotonic_so_the_waveform_is_not_folded(self) -> None:
        block = np.linspace(0.0, 3.0, 1_000, dtype=np.float32)
        out = soft_limit(block)
        assert np.all(np.diff(out) >= -1e-6)

    def test_it_is_odd_symmetric(self) -> None:
        block = np.linspace(-3.0, 3.0, 999, dtype=np.float32)
        out = soft_limit(block)
        assert np.allclose(out, -out[::-1], atol=1e-6)

    def test_empty_input(self) -> None:
        assert soft_limit(np.zeros(0, dtype=np.float32)).size == 0


class TestSpec:
    """Profile definitions and how intensity scales them."""

    @pytest.mark.parametrize("profile", CHARACTER)
    def test_every_profile_has_a_description(self, profile: VoiceProfile) -> None:
        assert describe_profile(profile).strip()
        assert describe_profile(profile) == PROFILES[profile].summary

    def test_none_is_described_too(self) -> None:
        assert "no processing" in describe_profile(VoiceProfile.NONE)

    def test_scaling_to_zero_is_neutral(self) -> None:
        scaled = PROFILES[VoiceProfile.JARVIS].scaled(0.0)
        assert scaled.comb_mix == 0.0
        assert scaled.chorus_mix == 0.0
        assert scaled.ring_depth == 0.0
        assert scaled.reverb_mix == 0.0
        assert scaled.comp_ratio == 1.0
        assert all(band.gain_db == 0.0 for band in scaled.pre_eq + scaled.post_eq)

    def test_scaling_to_one_changes_nothing(self) -> None:
        assert PROFILES[VoiceProfile.JARVIS].scaled(1.0) == PROFILES[VoiceProfile.JARVIS]

    def test_frequencies_are_not_scaled(self) -> None:
        """Half intensity means quieter metal, not retuned metal."""
        full = PROFILES[VoiceProfile.ROBOT]
        half = full.scaled(0.5)
        assert half.comb_ms == full.comb_ms
        assert half.ring_hz == full.ring_hz
        assert half.chorus_rate_hz == full.chorus_rate_hz
        assert [b.hz for b in half.post_eq] == [b.hz for b in full.post_eq]

    def test_robot_is_more_processed_than_jarvis(self) -> None:
        jarvis, robot = PROFILES[VoiceProfile.JARVIS], PROFILES[VoiceProfile.ROBOT]
        assert robot.ring_depth > jarvis.ring_depth
        assert robot.comb_mix > jarvis.comb_mix

    def test_clean_has_no_synthetic_character(self) -> None:
        clean = PROFILES[VoiceProfile.CLEAN]
        assert clean.comb_mix == 0.0
        assert clean.chorus_mix == 0.0
        assert clean.ring_depth == 0.0
        assert clean.reverb_mix == 0.0

    def test_an_empty_spec_builds_and_passes_audio(self, speech: np.ndarray) -> None:
        """Every optional stage skipped, which is the branch profiles never take."""
        effect = VoiceEffect(RATE, VoiceProfile.JARVIS, 1.0)
        effect._stages = effect._build(EffectSpec())
        out = effect.process(speech)
        assert out.shape == speech.shape
        assert np.isfinite(out).all()


class TestCalibrationSignal:
    """The probe the rack measures itself with."""

    def test_it_hits_the_requested_level(self) -> None:
        assert rms_dbfs(speech_shaped_noise(RATE, 1.0, -20.0)) == pytest.approx(-20.0, abs=0.1)

    def test_it_is_deterministic(self) -> None:
        assert np.array_equal(
            speech_shaped_noise(RATE, 0.5, -20.0), speech_shaped_noise(RATE, 0.5, -20.0)
        )

    def test_its_energy_sits_where_speech_does(self) -> None:
        probe = speech_shaped_noise(RATE, 2.0, -20.0)
        spectrum = np.abs(np.fft.rfft(probe)) ** 2
        freqs = np.fft.rfftfreq(len(probe), d=1.0 / RATE)
        in_band = spectrum[(freqs > 200.0) & (freqs < 4_000.0)].sum()
        assert in_band / spectrum.sum() > 0.5

    def test_zero_seconds_still_returns_a_sample(self) -> None:
        assert speech_shaped_noise(RATE, 0.0, -20.0).size >= 1


class TestFromConfig:
    """What the rest of the assistant actually calls."""

    def test_the_default_config_gives_the_jarvis_voice(self) -> None:
        config = JarvisConfig()
        assert config.tts.effect_profile is VoiceProfile.JARVIS
        assert not build_voice_effect(config).is_bypass

    def test_it_follows_the_configured_sample_rate(self) -> None:
        """Kokoro emits 24 kHz and the field is pinned to it, so this is 24 kHz.

        The rack scales its delays by the rate, which is why the field stopped
        being a range: a config claiming 16 kHz built a comb tuned for audio
        that was actually 24 kHz, on top of playing the whole voice a fifth flat.
        """
        assert build_voice_effect(JarvisConfig())._sample_rate == 24_000

    def test_turning_it_off_in_config_bypasses(self) -> None:
        assert build_voice_effect(JarvisConfig(tts={"effect_profile": "none"})).is_bypass

    def test_zero_intensity_in_config_bypasses(self) -> None:
        assert build_voice_effect(JarvisConfig(tts={"effect_intensity": 0.0})).is_bypass

    def test_the_profile_name_round_trips_through_yaml(self) -> None:
        config = JarvisConfig(tts={"effect_profile": "robot", "effect_intensity": 0.5})
        effect = build_voice_effect(config)
        assert effect.profile is VoiceProfile.ROBOT
        assert effect.intensity == 0.5

    def test_an_unknown_profile_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            JarvisConfig(tts={"effect_profile": "vocoder"})

    @pytest.mark.parametrize("rate", [16_000, 22_050, 24_000, 44_100, 48_000])
    def test_every_plausible_sample_rate_works(self, rate: int) -> None:
        """Delays are quoted in milliseconds and scaled, so none may round to zero."""
        probe = speech_shaped_noise(rate, 0.5, -20.0)
        out = VoiceEffect(rate, VoiceProfile.ROBOT).process(probe)
        assert np.isfinite(out).all()
        assert abs(rms_dbfs(out) - rms_dbfs(probe)) < 1.5


class TestCost:
    """This sits inside the §3 time to first audio budget of 300 ms."""

    @pytest.mark.parametrize("profile", CHARACTER)
    def test_a_sentence_costs_far_less_than_the_budget(self, profile: VoiceProfile) -> None:
        import time

        effect = VoiceEffect(RATE, profile)
        sentence = speech_shaped_noise(RATE, 2.0, -20.0)
        effect.process(sentence)  # warm numpy's FFT plan cache

        start = time.perf_counter()
        for _ in range(3):
            effect.process(sentence)
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / 3.0
        # Two seconds of audio. Generous enough not to flake on a loaded CI box,
        # tight enough to catch a stage that stopped vectorising.
        assert elapsed_ms < 120.0
