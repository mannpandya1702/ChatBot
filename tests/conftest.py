"""Shared fixtures.

Two things matter here. First, no test may read the developer's real
``config/config.yaml``; every test gets an isolated config. Second, the suite has
to pass on a Linux CI box with no GPU, no microphone, and no Windows, so anything
platform-bound is marked ``manual`` and deselected by default.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from jarvis import config as config_module
from jarvis.config import JarvisConfig, load_config, reset_config
from jarvis.state import EventBus


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point config loading at an empty temp file and strip JARVIS_ env vars."""
    for key in list(os.environ):
        if key.startswith("JARVIS_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("JARVIS_CONFIG_FILE", str(tmp_path / "absent.yaml"))
    reset_config()
    yield
    reset_config()


@pytest.fixture
def cfg(tmp_path: Path) -> JarvisConfig:
    """A default config with runtime paths redirected into ``tmp_path``."""
    return load_config(
        tmp_path / "absent.yaml",
        paths={
            "data_dir": str(tmp_path / "data"),
            "log_dir": str(tmp_path / "logs"),
            "models_dir": str(tmp_path / "models"),
            "vendor_dir": str(tmp_path / "vendor"),
        },
    )


@pytest.fixture
def write_config(tmp_path: Path) -> Any:
    """Return a helper that writes a YAML config file and loads it."""

    def _write(body: str, **overrides: Any) -> JarvisConfig:
        path = tmp_path / "config.yaml"
        path.write_text(body, encoding="utf-8")
        return load_config(path, **overrides)

    return _write


@pytest.fixture
def bus() -> Iterator[EventBus]:
    """A fresh event bus, closed on teardown."""
    instance = EventBus()
    yield instance
    instance.close()


@pytest.fixture
def clean_registry() -> Iterator[Any]:
    """An empty tool registry that does not touch the process-wide one."""
    from jarvis.tools.registry import ToolRegistry

    yield ToolRegistry()


@pytest.fixture
def sine_audio() -> Any:
    """Return a generator for float32 mono test audio at a given sample rate."""

    def _make(
        seconds: float,
        *,
        sample_rate: int = 16_000,
        freq: float = 440.0,
        amplitude: float = 0.2,
    ) -> np.ndarray[Any, Any]:
        t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
        return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    return _make


@pytest.fixture
def silence() -> Any:
    """Return a generator for float32 mono silence."""

    def _make(seconds: float, *, sample_rate: int = 16_000) -> np.ndarray[Any, Any]:
        return np.zeros(int(seconds * sample_rate), dtype=np.float32)

    return _make


@pytest.fixture
def config_module_ref() -> Any:
    """Direct handle on ``jarvis.config`` for tests that poke module state."""
    return config_module
