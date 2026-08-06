"""CLAUDE.md §5: no em dashes in any generated prose, docs, comments, or output.

Ruff cannot express this, so it is enforced here across the whole tracked tree.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Written as escapes so this checker does not trip over its own source.
EM_DASH = "\u2014"
EN_DASH = "\u2013"

CHECKED_SUFFIXES = {
    ".py", ".md", ".yaml", ".yml", ".toml", ".ps1", ".js", ".ts",
    ".jsx", ".tsx", ".html", ".css", ".json", ".rs",
}

# The build contract itself is the user's document, and the uploaded original
# used em dashes. Everything JARVIS generates must not.
EXEMPT = {
    "PROGRESS.md",
}


def _tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("not a git checkout")
    return [
        ROOT / line
        for line in result.stdout.splitlines()
        if line and Path(line).suffix in CHECKED_SUFFIXES and line not in EXEMPT
    ]


def test_no_em_dashes_in_tracked_sources() -> None:
    offenders: list[str] = []
    for path in _tracked_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if EM_DASH in line or EN_DASH in line:
                offenders.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()[:80]}")
    assert not offenders, "em dashes found (CLAUDE.md §5):\n" + "\n".join(offenders[:20])


def test_the_scan_actually_covers_files() -> None:
    assert len(_tracked_files()) >= 5


def test_config_yaml_is_not_tracked() -> None:
    """§0.6: machine-specific config must never be committed."""
    result = subprocess.run(
        ["git", "ls-files", "config/config.yaml"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout.strip() == "", "config/config.yaml must stay gitignored"


def test_no_model_artifacts_are_tracked() -> None:
    result = subprocess.run(
        ["git", "ls-files", "models/"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout.strip() == "", "models/ must stay gitignored"
