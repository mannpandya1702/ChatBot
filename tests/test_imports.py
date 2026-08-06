"""Every module must import on a bare development host.

CLAUDE.md §0b: no Windows-only or GPU-only dependency may be imported at module
scope. This test walks the whole package and imports each module with only the
core dependency set installed, which is what CI has. A regression here means
somebody moved an optional import back to the top of a file, and the assistant
would fail at startup on a machine missing that extra.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import jarvis

#: Modules that legitimately cannot import without an optional dependency.
#: Keep this empty. An entry here is a bug, not a feature.
ALLOWED_IMPORT_FAILURES: frozenset[str] = frozenset()


def _module_names() -> list[str]:
    found = [
        info.name
        for info in pkgutil.walk_packages(jarvis.__path__, prefix="jarvis.")
        if not info.ispkg
    ]
    return sorted(found)


@pytest.mark.parametrize("module_name", _module_names())
def test_module_imports_cleanly(module_name: str) -> None:
    if module_name in ALLOWED_IMPORT_FAILURES:
        pytest.skip(f"{module_name} is a known optional-dependency import")
    importlib.import_module(module_name)


def test_the_walk_found_modules() -> None:
    """Guard against the parametrisation silently collapsing to nothing."""
    assert len(_module_names()) >= 5


def test_no_optional_dependency_is_imported_at_module_scope() -> None:
    """Importing the package must not pull in an optional engine."""
    import sys

    for name in ("sounddevice", "openwakeword", "faster_whisper", "kokoro", "mss", "pynvml"):
        assert name not in sys.modules, (
            f"{name} was imported at module scope; move it inside the function that needs it"
        )
