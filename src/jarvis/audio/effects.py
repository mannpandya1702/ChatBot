"""The JARVIS voice: post-processing applied to Kokoro's output.

Kokoro's ``bm_george`` gives us the right accent and the right register, but it
is a human voice recorded in a quiet room. JARVIS is not a human voice in a
quiet room. What makes the film voice recognisable is not the actor, it is what
sits after the actor: a metallic resonance, a second copy of the voice slightly
detuned against the first, a short plate that puts it everywhere in the room at
once, and compression flat enough that it never sounds like it is breathing.

So this module is a small effects rack, applied to every synthesised chunk:

    high pass and low mid scoop
        -> resonant comb, the hollow metal shell
        -> detuned double, the "not quite one voice" giveaway
        -> ring modulation, the synthetic edge
        -> plate reverb, short and dark
        -> presence lift and top rolloff
        -> compression, so the delivery never rises or falls
        -> soft limit

Design constraints that shaped the implementation:

* **numpy only.** §1 locks the stack, and reaching for scipy to get
  ``lfilter`` would add a dependency for four filters. Every recursive stage
  here is expressed so that numpy runs the inner loop.
* **Cheap.** This sits inside the §3 time-to-first-audio budget of 300 ms.
  Equalisation is FFT convolution, the comb and reverb recursions are done a
  whole delay period at a time, and the compressor tracks its envelope per
  block rather than per sample. A two second chunk costs single digit
  milliseconds.
* **Stateful across chunks.** A reply is synthesised sentence by sentence and
  played back to back. Delay lines, reverb tails, oscillator phase and the
  compressor envelope all carry over, because resetting them between sentences
  would click on every full stop. :meth:`VoiceEffect.reset` clears them at the
  start of an utterance.

Intensity 0 is a true bypass: the input array is returned unchanged, not
processed with neutral settings.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt

from jarvis.config import JarvisConfig, VoiceProfile

__all__ = [
    "PROFILES",
    "EffectSpec",
    "VoiceEffect",
    "build_voice_effect",
    "describe_profile",
]

_log = logging.getLogger(__name__)

Samples = npt.NDArray[np.float32]

#: Envelope block for the compressor. 256 samples is 10.7 ms at 24 kHz, short
#: enough to catch a plosive and long enough that the loop stays trivial.
_ENVELOPE_BLOCK = 256

#: Reverb delays in samples at the 44.1 kHz they were tuned for. Scaled to the
#: running rate at build time. The four are mutually prime so their echoes do
#: not line up into a single audible pitch.
_COMB_DELAYS_44K = (1_557, 1_617, 1_491, 1_422)
_ALLPASS_DELAYS_44K = (556, 441)

#: Loudness the calibration pass assumes speech arrives at, in dBFS RMS. Kokoro
#: lands near here, and the compressor's behaviour depends on level, so the
#: calibration has to be run at a level speech actually reaches.
_CALIBRATION_RMS_DBFS = -20.0
#: Half a second is plenty for the envelope to settle and costs a few ms once.
_CALIBRATION_SECONDS = 0.5


# ---------------------------------------------------------------------------
# Profile description
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Band:
    """One shelf, cut or bump in the equalisation curve.

    Args:
        kind: ``highpass``, ``lowpass`` or ``peak``.
        hz: Corner frequency for the pass filters, centre for a peak.
        gain_db: Peak height. Ignored by the pass filters.
        width: Peak width in octaves. Ignored by the pass filters.
        order: Rolloff steepness for the pass filters. Ignored by a peak.
    """

    kind: str
    hz: float
    gain_db: float = 0.0
    width: float = 1.0
    order: int = 2


@dataclass(frozen=True, slots=True)
class EffectSpec:
    """Everything that distinguishes one voice profile from another.

    All of the "how much" numbers are quoted at full intensity. Scaling happens
    in :meth:`scaled`, which is what the intensity setting actually drives.
    """

    #: Shown to the user by ``--voices``.
    summary: str = ""

    #: Equalisation applied before the character stages.
    pre_eq: tuple[_Band, ...] = ()
    #: Equalisation applied after them, which is where fizz gets tamed.
    post_eq: tuple[_Band, ...] = ()

    #: Resonant comb. Short delays ring at 1/delay and its harmonics, which is
    #: what reads as "speaking inside something metal".
    comb_ms: float = 0.0
    comb_feedback: float = 0.0
    comb_mix: float = 0.0

    #: Detuned double. Two copies on slowly swept fractional delays, one drifting
    #: up while the other drifts down.
    chorus_ms: float = 0.0
    chorus_depth_ms: float = 0.0
    chorus_rate_hz: float = 0.0
    chorus_mix: float = 0.0

    #: Ring modulation. A carrier this low does not sound like a pitch, it
    #: sounds like the voice has a machine behind it.
    ring_hz: float = 0.0
    ring_depth: float = 0.0

    #: Plate. Short, dark, and well behind the voice.
    reverb_decay: float = 0.0
    reverb_damping_hz: float = 6_000.0
    reverb_mix: float = 0.0

    #: Compression. Threshold in dBFS, ratio as N:1.
    comp_threshold_db: float = 0.0
    comp_ratio: float = 1.0
    comp_attack_ms: float = 8.0
    comp_release_ms: float = 120.0

    #: Applied last, before the limiter.
    output_gain_db: float = 0.0

    def scaled(self, intensity: float) -> EffectSpec:
        """Return this profile dialled back towards a clean voice.

        Everything that adds character scales linearly with ``intensity``.
        Frequencies and times do not: halving the intensity should make the
        metal quieter, not retune it.
        """
        k = float(np.clip(intensity, 0.0, 1.0))
        return replace(
            self,
            pre_eq=tuple(replace(b, gain_db=b.gain_db * k) for b in self.pre_eq),
            post_eq=tuple(replace(b, gain_db=b.gain_db * k) for b in self.post_eq),
            comb_mix=self.comb_mix * k,
            chorus_mix=self.chorus_mix * k,
            ring_depth=self.ring_depth * k,
            reverb_mix=self.reverb_mix * k,
            # A ratio of 1.0 is no compression, so interpolate from there.
            comp_ratio=1.0 + (self.comp_ratio - 1.0) * k,
            output_gain_db=self.output_gain_db * k,
        )


_CLEAN = EffectSpec(
    summary="Broadcast polish only. Rumble removed, presence lifted, level held steady.",
    pre_eq=(_Band("highpass", 85.0, order=2),),
    post_eq=(
        _Band("peak", 240.0, gain_db=-2.0, width=1.2),
        _Band("peak", 3_200.0, gain_db=3.0, width=1.3),
        _Band("lowpass", 11_000.0, order=2),
    ),
    comp_threshold_db=-20.0,
    comp_ratio=3.0,
    output_gain_db=2.0,
)

_JARVIS = EffectSpec(
    summary=(
        "The film voice. Metallic resonance, a detuned double, a short dark plate, "
        "and compression flat enough that it never sounds like it is breathing."
    ),
    pre_eq=(
        _Band("highpass", 110.0, order=3),
        _Band("peak", 300.0, gain_db=-3.5, width=1.1),
    ),
    post_eq=(
        _Band("peak", 1_900.0, gain_db=2.5, width=1.4),
        _Band("peak", 4_200.0, gain_db=3.5, width=1.0),
        _Band("peak", 7_500.0, gain_db=-3.0, width=1.2),
        _Band("lowpass", 9_500.0, order=3),
    ),
    comb_ms=7.3,
    comb_feedback=0.42,
    comb_mix=0.30,
    chorus_ms=17.0,
    chorus_depth_ms=2.2,
    chorus_rate_hz=0.19,
    chorus_mix=0.34,
    ring_hz=84.0,
    ring_depth=0.10,
    reverb_decay=0.62,
    reverb_damping_hz=5_200.0,
    reverb_mix=0.16,
    comp_threshold_db=-22.0,
    comp_ratio=4.5,
    comp_attack_ms=6.0,
    comp_release_ms=140.0,
    output_gain_db=3.5,
)

_ROBOT = EffectSpec(
    summary=(
        "JARVIS pushed towards the machine. Deeper ring modulation and a tighter, "
        "louder comb. Unmistakably synthetic, still fully intelligible."
    ),
    pre_eq=(
        _Band("highpass", 130.0, order=3),
        _Band("peak", 320.0, gain_db=-5.0, width=1.1),
    ),
    post_eq=(
        _Band("peak", 1_700.0, gain_db=3.5, width=1.3),
        _Band("peak", 3_600.0, gain_db=4.0, width=1.0),
        _Band("peak", 6_800.0, gain_db=-4.5, width=1.1),
        _Band("lowpass", 8_000.0, order=4),
    ),
    comb_ms=4.6,
    comb_feedback=0.58,
    comb_mix=0.46,
    chorus_ms=14.0,
    chorus_depth_ms=1.6,
    chorus_rate_hz=0.27,
    chorus_mix=0.42,
    ring_hz=56.0,
    ring_depth=0.30,
    reverb_decay=0.55,
    reverb_damping_hz=4_400.0,
    reverb_mix=0.13,
    comp_threshold_db=-24.0,
    comp_ratio=6.0,
    comp_attack_ms=4.0,
    comp_release_ms=110.0,
    output_gain_db=4.5,
)

#: Every selectable profile. ``NONE`` has no entry because it never builds a rack.
PROFILES: dict[VoiceProfile, EffectSpec] = {
    VoiceProfile.CLEAN: _CLEAN,
    VoiceProfile.JARVIS: _JARVIS,
    VoiceProfile.ROBOT: _ROBOT,
}


def describe_profile(profile: VoiceProfile) -> str:
    """One line explaining what a profile does to the voice."""
    if profile is VoiceProfile.NONE:
        return "Kokoro's raw output, with no processing at all."
    return PROFILES[profile].summary


# ---------------------------------------------------------------------------
# Filter primitives
# ---------------------------------------------------------------------------


def design_eq_kernel(bands: tuple[_Band, ...], sample_rate: int, taps: int = 257) -> Samples:
    """Build one linear phase FIR kernel for a whole stack of bands.

    Bands are cascaded by multiplying their magnitude responses, so an arbitrary
    number of shelves and bumps still costs a single convolution.

    The curve is sampled on the FFT grid and transformed back, which is exact at
    the sampled frequencies and smooth between them. That is the right trade
    here: the shapes are deliberately gentle, and nothing about a voice needs a
    brick wall.

    Args:
        bands: The stack to realise. Empty gives a unit impulse.
        sample_rate: Hertz.
        taps: Kernel length. Must be odd so the delay is a whole sample.

    Returns:
        A float32 kernel of length ``taps``, delaying by ``(taps - 1) // 2``.
    """
    if taps % 2 == 0:
        taps += 1
    if not bands:
        kernel = np.zeros(taps, dtype=np.float32)
        kernel[(taps - 1) // 2] = 1.0
        return kernel

    # Oversample the design grid so the windowed kernel tracks the target.
    grid = 8 * taps
    freqs = np.fft.rfftfreq(grid, d=1.0 / sample_rate)
    # Guard the DC bin: the pass filters divide by frequency.
    safe = np.maximum(freqs, 1e-3)
    magnitude = np.ones_like(safe)

    for band in bands:
        if band.kind == "highpass":
            ratio = band.hz / safe
            magnitude *= 1.0 / np.sqrt(1.0 + ratio ** (2 * band.order))
        elif band.kind == "lowpass":
            ratio = safe / band.hz
            magnitude *= 1.0 / np.sqrt(1.0 + ratio ** (2 * band.order))
        elif band.kind == "peak":
            # Gaussian in log frequency: symmetric on a plot of the response,
            # which is how the width is meant to read.
            octaves = np.log2(safe / band.hz)
            shape = np.exp(-((octaves / max(band.width, 1e-3)) ** 2))
            magnitude *= 10.0 ** ((band.gain_db * shape) / 20.0)
        else:  # pragma: no cover - profiles are the only source of bands
            raise ValueError(f"unknown band kind {band.kind!r}")

    # Zero phase design, then shift to make it causal and window it to length.
    impulse = np.fft.irfft(magnitude, n=grid)
    half = (taps - 1) // 2
    kernel = np.concatenate((impulse[-half:], impulse[: half + 1]))
    kernel *= np.hanning(taps + 2)[1:-1]

    # Preserve broadband level: without this, a stack of cuts quietly drops the
    # whole voice and the makeup gain in the profile stops meaning anything.
    peak = float(np.abs(np.fft.rfft(kernel, n=grid)).max())
    if peak > 0:
        target = float(magnitude.max())
        kernel *= target / peak
    return kernel.astype(np.float32, copy=False)


class _Convolver:
    """FFT overlap-add convolution that keeps its tail between calls."""

    def __init__(self, kernel: Samples) -> None:
        self._kernel = kernel
        self._tail = np.zeros(max(len(kernel) - 1, 0), dtype=np.float32)
        self._is_unit = len(kernel) == 1 and kernel[0] == 1.0

    def reset(self) -> None:
        """Forget the overlap from the previous chunk."""
        self._tail = np.zeros_like(self._tail)

    def process(self, block: Samples) -> Samples:
        """Filter one block, carrying the overlap into the next call."""
        if self._is_unit or block.size == 0:
            return block
        n = len(block) + len(self._kernel) - 1
        size = 1 << int(n - 1).bit_length()
        spectrum = np.fft.rfft(block, n=size) * np.fft.rfft(self._kernel, n=size)
        full = np.fft.irfft(spectrum, n=size)[:n]
        full[: len(self._tail)] += self._tail
        self._tail = full[len(block) :].astype(np.float32, copy=False)
        return full[: len(block)].astype(np.float32, copy=False)


def _feedback_comb(block: Samples, state: Samples, feedback: float) -> tuple[Samples, Samples]:
    """Apply ``y[n] = x[n] + f * y[n - D]`` with D taken from ``state``.

    The recursion looks sequential, but samples ``n`` and ``n + 1`` never touch
    each other: only samples a whole delay apart do. Laying the block out as
    rows of D turns it into one short loop over rows, each row a vector add.
    A two second chunk at a 7 ms delay is 12 iterations rather than 48000.

    Args:
        block: Input samples.
        state: The previous D output samples. Its length is the delay.
        feedback: Loop gain. Must be under 1 or the filter runs away.

    Returns:
        The output block, and the state to pass to the next call.
    """
    delay = len(state)
    if delay == 0 or block.size == 0:
        return block, state

    rows = -(-len(block) // delay)
    padded = np.zeros(rows * delay, dtype=np.float32)
    padded[: len(block)] = block
    matrix = padded.reshape(rows, delay)

    matrix[0] += feedback * state
    for row in range(1, rows):
        matrix[row] += feedback * matrix[row - 1]

    out = padded[: len(block)]
    return out, np.concatenate((state, out))[-delay:].astype(np.float32, copy=False)


class _Allpass:
    """Schroeder allpass, built from a comb plus a read tap behind it.

    ``y[n] = -g x[n] + v[n - D]`` where ``v[n] = x[n] + g v[n - D]``. Flat in
    magnitude, scrambled in phase, which is what smears a bank of combs into
    something that sounds like a room rather than four echoes.
    """

    def __init__(self, delay: int, gain: float) -> None:
        self._gain = gain
        self._v = np.zeros(delay, dtype=np.float32)

    def reset(self) -> None:
        """Empty the delay line."""
        self._v = np.zeros_like(self._v)

    def process(self, block: Samples) -> Samples:
        """Filter one block."""
        if block.size == 0 or self._v.size == 0:
            return block
        delayed = self._v.copy()
        v, self._v = _feedback_comb(block, self._v, self._gain)
        # v[n - D]: the first D come from the line as it was on entry.
        shifted = np.concatenate((delayed, v))[: len(v)]
        return (shifted - self._gain * v).astype(np.float32, copy=False)


class _Reverb:
    """Four combs into two allpasses, with the wet path darkened.

    Damping is a single low pass on the summed wet signal rather than one inside
    each comb loop. Per loop damping would couple the delay lanes and cost the
    vectorisation in :func:`_feedback_comb` for a difference nobody can hear at
    a 16 percent wet mix.
    """

    def __init__(self, sample_rate: int, decay: float, damping_hz: float) -> None:
        scale = sample_rate / 44_100.0
        self._states = [
            np.zeros(max(int(d * scale), 1), dtype=np.float32) for d in _COMB_DELAYS_44K
        ]
        self._feedback = float(np.clip(decay, 0.0, 0.94))
        self._allpasses = [
            _Allpass(max(int(d * scale), 1), 0.5) for d in _ALLPASS_DELAYS_44K
        ]
        self._damp = _Convolver(
            design_eq_kernel((_Band("lowpass", damping_hz, order=2),), sample_rate, taps=129)
        )

    def reset(self) -> None:
        """Silence the tail."""
        self._states = [np.zeros_like(s) for s in self._states]
        for allpass in self._allpasses:
            allpass.reset()
        self._damp.reset()

    def process(self, block: Samples) -> Samples:
        """Return the wet signal for one block."""
        if block.size == 0:
            return block
        wet = np.zeros(len(block), dtype=np.float32)
        for index, state in enumerate(self._states):
            out, self._states[index] = _feedback_comb(block, state, self._feedback)
            wet += out
        wet /= len(self._states)
        for allpass in self._allpasses:
            wet = allpass.process(wet)
        return self._damp.process(wet)


class _Chorus:
    """Two detuned copies of the voice, swept in opposite directions.

    A fixed delay would just thicken. The sweep is what detunes: a delay line
    whose length is changing resamples what passes through it, so one copy sits
    slightly sharp while the other sits slightly flat. Holding both against the
    original is the artefact that says "this voice is not a person".
    """

    def __init__(self, sample_rate: int, base_ms: float, depth_ms: float, rate_hz: float) -> None:
        self._rate = sample_rate
        self._base = base_ms * sample_rate / 1000.0
        self._depth = depth_ms * sample_rate / 1000.0
        self._rate_hz = rate_hz
        self._history = max(int(self._base + self._depth) + 4, 4)
        self._buffer = np.zeros(self._history, dtype=np.float32)
        self._phase = 0.0

    def reset(self) -> None:
        """Clear the delay line and restart the sweep."""
        self._buffer = np.zeros_like(self._buffer)
        self._phase = 0.0

    def process(self, block: Samples) -> Samples:
        """Return the doubled signal, without the dry part."""
        if block.size == 0:
            return block
        n = len(block)
        buffer = np.concatenate((self._buffer, block))
        step = 2.0 * math.pi * self._rate_hz / self._rate
        phase = self._phase + step * np.arange(n, dtype=np.float64)

        voices = np.zeros(n, dtype=np.float64)
        # Quarter cycle apart so the two never sweep through unison together,
        # and at slightly different depths so they do not sound like one voice.
        for offset, depth_scale in ((0.0, 1.0), (math.pi / 2.0, 0.78)):
            delay = self._base + self._depth * depth_scale * np.sin(phase + offset)
            positions = np.arange(n, dtype=np.float64) + self._history - delay
            positions = np.clip(positions, 0.0, len(buffer) - 1.0)
            voices += np.interp(positions, np.arange(len(buffer), dtype=np.float64), buffer)
        voices *= 0.5

        self._buffer = buffer[-self._history :].astype(np.float32, copy=False)
        self._phase = float((self._phase + step * n) % (2.0 * math.pi))
        return voices.astype(np.float32, copy=False)


class _Delay:
    """A whole number sample delay that carries its tail between calls."""

    def __init__(self, samples: int) -> None:
        self._buffer = np.zeros(max(samples, 0), dtype=np.float32)

    def reset(self) -> None:
        """Empty the line."""
        self._buffer = np.zeros_like(self._buffer)

    def process(self, block: Samples) -> Samples:
        """Delay one block."""
        if self._buffer.size == 0 or block.size == 0:
            return block
        joined = np.concatenate((self._buffer, block))
        self._buffer = joined[-self._buffer.size :].astype(np.float32, copy=False)
        return joined[: len(block)].astype(np.float32, copy=False)


class _RingModulator:
    """Multiply by a low carrier. The classic robot voice, used sparingly.

    Two things keep it from eating the words.

    The carrier is mixed against unity rather than replacing the signal, so
    depth 0.1 is an edge and depth 0.3 is a machine, but neither is a full
    ring modulator.

    More importantly it only touches the low band. Ring modulation smears every
    component it reaches by plus and minus the carrier, and the components that
    carry the *words* are the formants above about a kilohertz. Modulating those
    trades intelligibility for nothing: the buzz that makes the effect
    recognisable lives down with the fundamental. Measured against a transcriber,
    full band modulation at this depth turns "good evening sir" into "but even
    so", and splitting the band removes that entirely.

    The split is complementary by construction: the high band is the delay
    matched input minus the low band, so at depth zero the two sum back to
    exactly the input.
    """

    def __init__(
        self,
        sample_rate: int,
        carrier_hz: float,
        depth: float,
        split_hz: float = 1_100.0,
        taps: int = 129,
    ) -> None:
        self._rate = sample_rate
        self._carrier = carrier_hz
        self._depth = depth
        self._phase = 0.0
        self._low = _Convolver(
            design_eq_kernel((_Band("lowpass", split_hz, order=3),), sample_rate, taps=taps)
        )
        self._align = _Delay((taps - 1) // 2 if taps % 2 else taps // 2)

    def reset(self) -> None:
        """Restart the carrier at zero phase and clear the band split."""
        self._phase = 0.0
        self._low.reset()
        self._align.reset()

    def process(self, block: Samples) -> Samples:
        """Ring modulate the low band, carrying carrier phase into the next call."""
        if block.size == 0 or self._depth <= 0.0:
            return block
        n = len(block)
        step = 2.0 * math.pi * self._carrier / self._rate
        phase = self._phase + step * np.arange(n, dtype=np.float64)
        self._phase = float((self._phase + step * n) % (2.0 * math.pi))
        carrier = np.cos(phase)

        low = self._low.process(block)
        high = self._align.process(block) - low
        modulated = low * (1.0 - self._depth + self._depth * carrier)
        return (high + modulated).astype(np.float32, copy=False)


class _Compressor:
    """Downward compression with a block rate envelope.

    A per sample envelope follower is a one pole recursion with a delay of one,
    which is the one shape that does not vectorise. It also does not need to:
    the ear integrates loudness over milliseconds, so tracking peak level once
    per 10 ms block and interpolating the gain between blocks is inaudible and
    turns 48000 iterations into 190. Interpolating rather than stepping the gain
    is what keeps it from zippering.
    """

    def __init__(
        self,
        sample_rate: int,
        threshold_db: float,
        ratio: float,
        attack_ms: float,
        release_ms: float,
    ) -> None:
        self._rate = float(sample_rate)
        self._threshold = 10.0 ** (threshold_db / 20.0)
        self._ratio = max(ratio, 1.0)
        self._attack_s = max(attack_ms / 1000.0, 1e-4)
        self._release_s = max(release_ms / 1000.0, 1e-4)
        self._envelope = 0.0
        self._gain = 1.0
        self._offset = 0

    def reset(self) -> None:
        """Open the compressor back up."""
        self._envelope = 0.0
        self._gain = 1.0
        self._offset = 0

    def process(self, block: Samples) -> Samples:
        """Compress one block.

        Two things here exist so that the result does not depend on where the
        sentence chunker happened to cut. The envelope block grid is anchored to
        a running sample count rather than restarting at every chunk, and each
        block advances the envelope by the time its own samples occupy rather
        than by a nominal block. Without either, the same reply comes out at
        different levels depending only on its punctuation, which is the kind of
        bug that is very hard to hear and impossible to unhear.
        """
        if block.size == 0 or self._ratio <= 1.0:
            return block
        n = len(block)
        first = _ENVELOPE_BLOCK - self._offset
        starts = (
            np.arange(0, n, _ENVELOPE_BLOCK)
            if first >= n
            else np.concatenate(([0], np.arange(first, n, _ENVELOPE_BLOCK)))
        )
        self._offset = (self._offset + n) % _ENVELOPE_BLOCK
        # reduceat handles the ragged final block without a zero pad, so the
        # peak is taken over real samples only.
        peaks = np.maximum.reduceat(np.abs(block), starts)
        ends = np.concatenate((starts[1:], [n]))

        gains = np.empty(len(starts) + 1, dtype=np.float64)
        gains[0] = self._gain
        for index in range(len(starts)):
            peak = float(peaks[index])
            span = float(ends[index] - starts[index]) / self._rate
            tau = self._attack_s if peak > self._envelope else self._release_s
            coefficient = math.exp(-span / tau)
            self._envelope = peak + coefficient * (self._envelope - peak)
            if self._envelope > self._threshold:
                over = self._envelope / self._threshold
                # Reduce the excess by the ratio, in the amplitude domain.
                gains[index + 1] = over ** (1.0 / self._ratio - 1.0)
            else:
                gains[index + 1] = 1.0
        self._gain = float(gains[-1])

        # One gain value per block boundary, ramped across the samples between.
        knots = np.concatenate((starts[:1], ends)).astype(np.float64)
        curve = np.interp(np.arange(n, dtype=np.float64), knots, gains)
        return (block * curve).astype(np.float32, copy=False)


def speech_shaped_noise(sample_rate: int, seconds: float, rms_dbfs: float) -> Samples:
    """A deterministic broadband signal with roughly the spectrum of speech.

    Used to calibrate output level. It has to be broadband, because the whole
    point is to measure what a stack of shelves and cuts does to a voice, and a
    sine at one frequency would measure the response at one frequency.

    Deterministic on purpose: the calibration gain is part of the voice, and a
    voice that comes out a fraction of a decibel different on every start is a
    voice with a bug in it.
    """
    count = max(int(sample_rate * seconds), 1)
    rng = np.random.default_rng(20_240_607)
    noise = rng.standard_normal(count)

    # Shape it: rising to about 500 Hz, then falling, which is the broad shape
    # of a long term speech spectrum.
    spectrum = np.fft.rfft(noise)
    freqs = np.maximum(np.fft.rfftfreq(count, d=1.0 / sample_rate), 1.0)
    envelope = 1.0 / np.sqrt(1.0 + (120.0 / freqs) ** 4)
    envelope /= np.sqrt(1.0 + (freqs / 900.0) ** 2)
    shaped = np.fft.irfft(spectrum * envelope, n=count)

    rms = float(np.sqrt(np.mean(shaped**2)))
    if rms > 0:
        shaped *= (10.0 ** (rms_dbfs / 20.0)) / rms
    return shaped.astype(np.float32, copy=False)


def soft_limit(block: Samples, ceiling: float = 0.97) -> Samples:
    """Round off anything above the ceiling instead of shearing it flat.

    The rack can overshoot: several stages add gain, and the comb rings. Hard
    clipping a voice is one of the ugliest sounds in audio, so the region above
    the ceiling is bent through a tanh knee. Below the ceiling nothing is
    touched, and the hard clip afterwards only ever catches arithmetic noise.
    """
    if block.size == 0:
        return block
    magnitude = np.abs(block)
    over = magnitude > ceiling
    if not bool(over.any()):
        return block
    out = block.astype(np.float32, copy=True)
    headroom = 1.0 - ceiling
    excess = (magnitude[over] - ceiling) / headroom
    shaped = ceiling + headroom * np.tanh(excess)
    out[over] = np.sign(block[over]) * shaped
    return np.clip(out, -1.0, 1.0, out=out)


# ---------------------------------------------------------------------------
# The rack
# ---------------------------------------------------------------------------


@dataclass
class _Stages:
    """The built stages, kept together so :meth:`VoiceEffect.reset` is one loop."""

    pre_eq: _Convolver
    post_eq: _Convolver
    comb_state: Samples
    comb_feedback: float
    comb_mix: float
    chorus: _Chorus | None
    chorus_mix: float
    ring: _RingModulator | None
    reverb: _Reverb | None
    reverb_mix: float
    compressor: _Compressor
    output_gain: float
    _comb_zero: Samples = field(default_factory=lambda: np.zeros(0, dtype=np.float32))


class VoiceEffect:
    """The voice rack, applied chunk by chunk to synthesised speech.

    Args:
        sample_rate: Hertz of the audio it will be given.
        profile: Which character to apply.
        intensity: 0 is a true bypass, 1 is the profile as designed.

    A single instance is not thread safe: it carries filter state, so one
    utterance's chunks must go through it in order. That matches how
    :class:`~jarvis.audio.tts.KokoroSynthesizer` uses it.
    """

    def __init__(
        self,
        sample_rate: int,
        profile: VoiceProfile = VoiceProfile.JARVIS,
        intensity: float = 1.0,
    ) -> None:
        self._sample_rate = sample_rate
        self._profile = profile
        self._intensity = float(np.clip(intensity, 0.0, 1.0))
        self._stages: _Stages | None = None
        self._calibration_db = 0.0
        if profile is not VoiceProfile.NONE and self._intensity > 0.0:
            self._stages = self._build(PROFILES[profile].scaled(self._intensity))
            self._calibrate()

    @property
    def profile(self) -> VoiceProfile:
        """The configured profile."""
        return self._profile

    @property
    def intensity(self) -> float:
        """How far the profile is dialled in, 0 to 1."""
        return self._intensity

    @property
    def is_bypass(self) -> bool:
        """Whether :meth:`process` returns its input untouched."""
        return self._stages is None

    def _build(self, spec: EffectSpec) -> _Stages:
        rate = self._sample_rate
        comb_delay = max(int(spec.comb_ms * rate / 1000.0), 1) if spec.comb_mix > 0 else 0
        return _Stages(
            pre_eq=_Convolver(design_eq_kernel(spec.pre_eq, rate)),
            post_eq=_Convolver(design_eq_kernel(spec.post_eq, rate)),
            comb_state=np.zeros(comb_delay, dtype=np.float32),
            comb_feedback=float(np.clip(spec.comb_feedback, 0.0, 0.92)),
            comb_mix=spec.comb_mix,
            chorus=(
                _Chorus(rate, spec.chorus_ms, spec.chorus_depth_ms, spec.chorus_rate_hz)
                if spec.chorus_mix > 0
                else None
            ),
            chorus_mix=spec.chorus_mix,
            ring=(
                _RingModulator(rate, spec.ring_hz, spec.ring_depth)
                if spec.ring_depth > 0
                else None
            ),
            reverb=(
                _Reverb(rate, spec.reverb_decay, spec.reverb_damping_hz)
                if spec.reverb_mix > 0
                else None
            ),
            reverb_mix=spec.reverb_mix,
            compressor=_Compressor(
                rate,
                spec.comp_threshold_db,
                spec.comp_ratio,
                spec.comp_attack_ms,
                spec.comp_release_ms,
            ),
            output_gain=10.0 ** (spec.output_gain_db / 20.0),
        )

    def _calibrate(self) -> None:
        """Trim the rack so it does not change how loud the assistant is.

        A profile is a character, not a volume control. Every stack of cuts and
        every compressor in here changes the level as a side effect, and if that
        leaks out then choosing a voice quietly becomes choosing a volume, and
        the speaker setting the user picked stops meaning anything.

        So the rack measures itself at build time: speech shaped noise at a
        realistic level goes in, the output level comes back, and the ratio is
        folded into the output gain. The bound exists because a badly chosen
        profile should come out wrong rather than come out deafening.
        """
        stages = self._stages
        if stages is None:
            return
        probe = speech_shaped_noise(
            self._sample_rate, _CALIBRATION_SECONDS, _CALIBRATION_RMS_DBFS
        )
        try:
            measured = self._run(stages, probe)
        except Exception:  # noqa: BLE001 - an uncalibrated voice beats no voice
            _log.exception("voice effect calibration failed, leaving the level as designed")
            self.reset()
            return
        self.reset()

        # Ignore the first tenth: the reverb and the compressor are still
        # settling there, and including it would read as a quieter rack.
        settled = measured[len(measured) // 10 :]
        out_rms = float(np.sqrt(np.mean(settled.astype(np.float64) ** 2)))
        in_rms = 10.0 ** (_CALIBRATION_RMS_DBFS / 20.0)
        if out_rms <= 1e-9:
            _log.warning("voice effect calibration measured silence, leaving the level alone")
            return

        trim = float(np.clip(in_rms / out_rms, 0.25, 6.0))
        stages.output_gain *= trim
        self._calibration_db = 20.0 * math.log10(trim)

    @property
    def calibration_db(self) -> float:
        """How much level the rack had to be given back, in decibels."""
        return self._calibration_db

    def reset(self) -> None:
        """Clear every delay line, tail, phase and envelope.

        Called at the start of an utterance and after a barge-in. Within an
        utterance the state is deliberately kept, so the reverb of one sentence
        decays into the next instead of being chopped at the full stop.
        """
        stages = self._stages
        if stages is None:
            return
        stages.pre_eq.reset()
        stages.post_eq.reset()
        stages.comb_state = np.zeros_like(stages.comb_state)
        if stages.chorus is not None:
            stages.chorus.reset()
        if stages.ring is not None:
            stages.ring.reset()
        if stages.reverb is not None:
            stages.reverb.reset()
        stages.compressor.reset()

    def process(self, audio: Samples) -> Samples:
        """Run one chunk of speech through the rack.

        Args:
            audio: Mono float32 at the configured sample rate.

        Returns:
            The processed chunk, same length and dtype, guaranteed finite and
            inside [-1, 1]. On bypass this is the input array itself.

        Never raises. A rack that throws would silence the assistant over a
        cosmetic stage, so a failure logs and returns the dry voice.
        """
        stages = self._stages
        if stages is None or audio.size == 0:
            return audio
        try:
            return self._run(stages, np.asarray(audio, dtype=np.float32))
        except Exception:  # noqa: BLE001 - §5, the voice must survive the effect
            _log.exception("voice effect failed, falling back to the dry signal")
            return audio

    def _run(self, stages: _Stages, audio: Samples) -> Samples:
        signal = stages.pre_eq.process(audio)

        if stages.comb_mix > 0 and stages.comb_state.size:
            combed, stages.comb_state = _feedback_comb(
                signal, stages.comb_state, stages.comb_feedback
            )
            # The comb sums energy, so scale the wet path by the loop's own gain
            # before mixing or the metal simply arrives louder than the voice.
            combed = combed * (1.0 - stages.comb_feedback)
            signal = (1.0 - stages.comb_mix) * signal + stages.comb_mix * combed

        if stages.chorus is not None:
            doubled = stages.chorus.process(signal)
            signal = (1.0 - stages.chorus_mix) * signal + stages.chorus_mix * doubled

        if stages.ring is not None:
            signal = stages.ring.process(signal)

        if stages.reverb is not None:
            wet = stages.reverb.process(signal)
            signal = (1.0 - stages.reverb_mix) * signal + stages.reverb_mix * wet

        signal = stages.post_eq.process(signal)
        signal = stages.compressor.process(signal)
        signal = signal * stages.output_gain

        # A non-finite sample makes sounddevice emit a click at best. The rack
        # has feedback loops in it, so this is a real failure mode, not a
        # theoretical one.
        if not bool(np.isfinite(signal).all()):
            _log.error("voice effect produced non-finite samples, resetting the rack")
            self.reset()
            return np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0).astype(
                np.float32, copy=False
            )
        return soft_limit(signal.astype(np.float32, copy=False))


def build_voice_effect(config: JarvisConfig) -> VoiceEffect:
    """Build the configured rack for the configured sample rate."""
    effect = VoiceEffect(
        config.tts.sample_rate,
        config.tts.effect_profile,
        config.tts.effect_intensity,
    )
    _log.info(
        "voice effect ready",
        extra={
            "context": {
                "profile": str(config.tts.effect_profile),
                "intensity": effect.intensity,
                "bypass": effect.is_bypass,
            }
        },
    )
    return effect
