"""Render the same line in every voice profile so it can be judged by ear.

A voice cannot be chosen from a description. This synthesises one sentence with
Kokoro, runs it through each profile in :mod:`jarvis.audio.effects`, and writes
the results as WAV files, so the choice is made by listening rather than by
reading adjectives.

    uv run --extra tts python scripts/preview_voice.py
    uv run --extra tts python scripts/preview_voice.py --text "Good evening, sir."
    uv run --extra tts python scripts/preview_voice.py --intensity 0.5 1.0

Writes into ``build/voice-preview`` by default, which is gitignored. Nothing
here touches ``config/config.yaml``: once a favourite is found, set
``tts.effect_profile`` and ``tts.effect_intensity`` there.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from jarvis.audio.effects import VoiceEffect, describe_profile
from jarvis.config import JarvisConfig, VoiceProfile, load_config

DEFAULT_TEXT = (
    "Good evening, sir. The processor is running at forty three percent, "
    "and the graphics card is holding at sixty one degrees. Everything is nominal."
)


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write mono float audio as a 16 bit WAV.

    soundfile ships with the tts extra, but writing a WAV header is twelve lines
    and this script should still work for someone who only installed numpy.
    """
    import wave

    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32_767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _synthesize_dry(config: JarvisConfig, text: str) -> np.ndarray:
    """Synthesise once, with the rack bypassed.

    Kokoro is the slow part, so it runs a single time and every profile is
    applied to the same waveform. That also makes the comparison honest: the
    profiles differ only by their processing, not by a different take.
    """
    from jarvis.audio.tts import KokoroSynthesizer

    bypass = VoiceEffect(config.tts.sample_rate, VoiceProfile.NONE)
    synth = KokoroSynthesizer(config, effect=bypass)
    return synth.synthesize(text)


def main(argv: list[str] | None = None) -> int:
    """Render one line in every profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml.")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="What to say.")
    parser.add_argument("--voice", default=None, help="Override the Kokoro voice.")
    parser.add_argument(
        "--intensity",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 1.0],
        help="Intensities to render for each profile.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Defaults to build/voice-preview.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.voice:
        tts = config.tts.model_copy(update={"voice": args.voice})
        config = config.model_copy(update={"tts": tts})

    out_dir = args.out or (Path(__file__).resolve().parents[1] / "build" / "voice-preview")
    out_dir.mkdir(parents=True, exist_ok=True)
    rate = config.tts.sample_rate

    print(f"voice {config.tts.voice}, {rate} Hz")
    print(f"synthesising: {args.text}")
    try:
        dry = _synthesize_dry(config, args.text)
    except Exception as exc:  # noqa: BLE001 - a missing extra lands here too
        print(f"could not synthesise: {exc}")
        print("  install the speech extra with: uv sync --extra tts")
        return 1
    if dry.size == 0:
        print("Kokoro returned no audio")
        return 1

    _write_wav(out_dir / "00-none.wav", dry, rate)
    print(f"\n  00-none.wav          {describe_profile(VoiceProfile.NONE)}")

    index = 1
    for profile in (VoiceProfile.CLEAN, VoiceProfile.JARVIS, VoiceProfile.ROBOT):
        print(f"\n  {profile}: {describe_profile(profile)}")
        for intensity in args.intensity:
            effect = VoiceEffect(rate, profile, intensity)
            wet = effect.process(dry)
            name = f"{index:02d}-{profile}-{intensity:g}.wav"
            _write_wav(out_dir / name, wet, rate)
            print(
                f"    {name:28} level trim {effect.calibration_db:+5.1f} dB, "
                f"peak {float(np.abs(wet).max()):.2f}"
            )
            index += 1

    print(f"\nwritten to {out_dir}")
    print("Set the winner in config/config.yaml under tts.effect_profile.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
