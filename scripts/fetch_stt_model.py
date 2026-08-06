"""Pre-fetch the transcription model the resolved hardware tier will use.

Whisper weights download on first use otherwise, which puts several hundred
megabytes in the middle of the first sentence anyone speaks to JARVIS: the
wake word fires, the utterance endpoints, and then nothing happens for
minutes with no progress shown. Setup is the right place to pay that, so
``scripts/pull_models.ps1`` calls this.

Which model to fetch is not decided here. It comes from
:meth:`JarvisConfig.stt_settings`, so the tier logic lives in one place and a
machine whose small GPU can host Whisper gets the same model at setup that it
will use at runtime.

Run directly to fetch for the current config::

    uv run python scripts/fetch_stt_model.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jarvis.config import SttEngine, load_config
from jarvis.util.platform import has_module


def fetch_faster_whisper(model: str) -> int:
    """Download a faster-whisper checkpoint into the shared cache."""
    if not has_module("faster_whisper"):
        print(
            f"faster-whisper is not installed, so {model} cannot be pre-fetched.\n"
            "  install it with: uv sync --extra stt"
        )
        return 1

    from faster_whisper.utils import download_model

    print(f"fetching faster-whisper {model} ...")
    try:
        path = download_model(model)
    except Exception as exc:  # noqa: BLE001 - network, disk, and auth all land here
        print(f"could not fetch {model}: {exc}")
        return 1
    print(f"  ready: {path}")
    return 0


def fetch_whispercpp(model: str) -> int:
    """Download a whisper.cpp checkpoint.

    pywhispercpp fetches on first construction and offers no separate download
    entry point, so the model is built once and thrown away.
    """
    if not has_module("pywhispercpp"):
        print(
            f"pywhispercpp is not installed, so {model} cannot be pre-fetched.\n"
            "  it is published for Windows only; on this host the model will\n"
            "  download on first use instead."
        )
        return 1

    from pywhispercpp.model import Model

    print(f"fetching whisper.cpp {model} ...")
    try:
        Model(model)
    except Exception as exc:  # noqa: BLE001
        print(f"could not fetch {model}: {exc}")
        return 1
    print(f"  ready: {model}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Fetch the STT model for the resolved tier."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml.")
    parser.add_argument(
        "--model", default=None, help="Override the model the tier would choose."
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    engine, model, device, compute_type = config.stt_settings()
    model = args.model or model

    print(
        f"tier {config.effective_tier()}: {engine} {model} on {device} ({compute_type})"
    )
    if engine is SttEngine.WHISPERCPP:
        return fetch_whispercpp(model)
    return fetch_faster_whisper(model)


if __name__ == "__main__":
    sys.exit(main())
