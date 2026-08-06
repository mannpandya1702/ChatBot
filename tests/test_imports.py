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


#: §0b: none of these may be pulled in by importing the package.
OPTIONAL_ENGINES = (
    "sounddevice",
    "openwakeword",
    "faster_whisper",
    "kokoro",
    "mss",
    "pynvml",
    "torch",
    "win32api",
    "wmi",
    "clr",
)


def test_no_optional_dependency_is_imported_at_module_scope() -> None:
    """Importing the package must not pull in an optional engine.

    Checked in a fresh interpreter rather than by reading this one's
    ``sys.modules``. Any other test that touches an engine, directly or through
    a live-engine check, leaves it loaded process-wide, and the assertion then
    reports whatever ran earlier instead of what importing jarvis does. That
    made the test pass or fail on collection order and on whether the optional
    extras happened to be installed.
    """
    import json
    import subprocess
    import sys
    import textwrap

    program = textwrap.dedent(
        f"""
        import json, sys, importlib
        for name in {list(OPTIONAL_ENGINES)!r}:
            assert name not in sys.modules, name
        for module in {_module_names()!r}:
            importlib.import_module(module)
        leaked = [n for n in {list(OPTIONAL_ENGINES)!r} if n in sys.modules]
        print(json.dumps(leaked))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, f"importing the package failed:\n{result.stderr}"

    leaked = json.loads(result.stdout.strip().splitlines()[-1])
    assert leaked == [], (
        f"{leaked} imported at module scope; move each inside the function that needs it"
    )
