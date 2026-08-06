"""Local file search.

T-4.4: fast search via the Everything CLI when installed, ``os.scandir``
fallback otherwise. Read-only, results capped at 5.

Nothing here opens, reads, moves, or deletes a file. It reports names, paths,
sizes, and modification times only, which is what "where is my tax return"
actually needs.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field

from jarvis.config import JarvisConfig, get_config
from jarvis.tools.registry import ToolCategory, ToolInput, ToolOutput, tool
from jarvis.util.errors import ToolExecutionError

__all__ = ["FileHit", "FileSearchInput", "FileSearchOutput", "find_everything_cli", "search_files"]

_log = logging.getLogger(__name__)

_MAX_RESULTS = 5
_EVERYTHING_TIMEOUT_S = 10.0
#: Ceiling on the scandir walk so a search of a huge tree still answers promptly.
_SCAN_BUDGET_S = 8.0
_SCAN_MAX_ENTRIES = 200_000

#: Directories that are never worth walking and are slow or noisy.
_SKIP_DIRS = frozenset(
    {
        "node_modules", ".git", ".venv", "venv", "__pycache__", ".mypy_cache",
        ".ruff_cache", "AppData", "Windows", "$Recycle.Bin", "System Volume Information",
        ".cache", "site-packages",
    }
)


class FileHit(ToolOutput):
    """One matching file."""

    name: str = Field(description="File name including extension.")
    path: str = Field(description="Full path to the file.")
    size_mb: float = Field(description="File size in megabytes.")
    modified: str = Field(description="Last modification time, ISO 8601.")
    is_directory: bool = Field(default=False, description="Whether this is a folder.")


class FileSearchInput(ToolInput):
    """Arguments for the file search tool."""

    query: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "Part of the file name to look for, for example invoice or tax return. "
            "Matching is case insensitive and matches anywhere in the name."
        ),
    )
    search_root: str | None = Field(
        default=None,
        description=(
            "Folder to search under. Leave unset to search the user's home folder. Ignored "
            "when the Everything index is available, which searches everywhere."
        ),
    )
    extension: str | None = Field(
        default=None,
        description="Restrict to one file extension, for example pdf or docx. No leading dot.",
    )


class FileSearchOutput(ToolOutput):
    """Search results, capped for speech."""

    query: str = Field(description="What was searched for.")
    backend: str = Field(description="Which search backend answered, everything or scandir.")
    total_found: int = Field(description="How many matches were found before capping.")
    truncated: bool = Field(description="Whether more matches exist than are listed.")
    results: list[FileHit] = Field(
        default_factory=list, description="Up to 5 matches, most recently modified first."
    )


def find_everything_cli(config: JarvisConfig | None = None) -> Path | None:
    """Locate ``es.exe``, the Everything command line interface.

    Checks the configured path first, then PATH, then the usual install
    locations. Returns None when it is not installed, which is the common case
    and not an error.
    """
    cfg = config or get_config()
    configured = cfg.tools.everything_cli_path
    if configured is not None:
        candidate = Path(configured)
        if candidate.is_file():
            return candidate
        _log.warning(
            "the configured Everything CLI path does not exist",
            extra={"context": {"path": str(candidate)}},
        )

    found = shutil.which("es") or shutil.which("es.exe")
    if found:
        return Path(found)

    for base in (
        Path("C:/Program Files/Everything"),
        Path("C:/Program Files (x86)/Everything"),
    ):
        candidate = base / "es.exe"
        if candidate.is_file():
            return candidate
    return None


def _hit_from_path(path: Path) -> FileHit | None:
    """Build a result row, or None when the entry vanished or is unreadable."""
    try:
        stat = path.stat()
        return FileHit(
            name=path.name,
            path=str(path),
            size_mb=round(stat.st_size / (1024**2), 3),
            modified=datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
            is_directory=path.is_dir(),
        )
    except (OSError, ValueError):
        return None


def _search_everything(exe: Path, query: str, extension: str | None) -> list[FileHit]:
    """Query the Everything index. Much faster than walking the filesystem."""
    argv = [str(exe), "-n", str(_MAX_RESULTS * 4)]
    if extension:
        argv += ["-ext", extension.lstrip(".")]
    argv.append(query)

    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            argv,
            capture_output=True,
            text=True,
            timeout=_EVERYTHING_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _log.debug("Everything CLI failed: %s", exc)
        return []
    if completed.returncode != 0:
        return []

    hits: list[FileHit] = []
    for line in completed.stdout.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        hit = _hit_from_path(Path(candidate))
        if hit is not None:
            hits.append(hit)
    return hits


def _walk(root: Path, budget_s: float) -> Iterator[Path]:
    """Iterative scandir walk with a time and entry budget.

    Iterative rather than recursive so a deep or cyclic tree cannot blow the
    stack, and budgeted so a search of an entire drive still answers.
    """
    deadline = time.monotonic() + budget_s
    seen = 0
    stack = [root]
    while stack:
        if time.monotonic() > deadline or seen > _SCAN_MAX_ENTRIES:
            _log.debug("file walk hit its budget", extra={"context": {"entries": seen}})
            return
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    seen += 1
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name not in _SKIP_DIRS and not entry.name.startswith("."):
                                stack.append(Path(entry.path))
                            continue
                    except OSError:
                        continue
                    yield Path(entry.path)
        except (PermissionError, OSError):
            continue


def _search_scandir(root: Path, query: str, extension: str | None) -> list[FileHit]:
    """Fallback search by walking the filesystem."""
    needle = query.lower()
    suffix = f".{extension.lstrip('.').lower()}" if extension else None
    hits: list[FileHit] = []

    for path in _walk(root, _SCAN_BUDGET_S):
        name = path.name.lower()
        if needle not in name:
            continue
        if suffix and not name.endswith(suffix):
            continue
        hit = _hit_from_path(path)
        if hit is not None:
            hits.append(hit)
            # Collect more than we return so the newest-first sort is meaningful.
            if len(hits) >= _MAX_RESULTS * 10:
                break
    return hits


@tool(
    name="files.search",
    description=(
        "Find files by name on this computer. Use this for questions like where is my tax "
        "return, find the invoice pdf, or do I have a file called notes. Uses the Everything "
        "index when installed and falls back to walking the folder tree. Returns at most 5 "
        "matches, most recently modified first, with size in megabytes. Read-only, it never "
        "opens, moves, or deletes anything."
    ),
    category=ToolCategory.FILES,
    read_only=True,
)
def search_files(params: FileSearchInput) -> FileSearchOutput:
    """Search for files by name.

    Args:
        params: Query, optional root, and optional extension filter.

    Returns:
        Up to five matches, newest first.

    Raises:
        ToolExecutionError: The search could not be performed at all.
    """
    config = get_config()
    try:
        exe = find_everything_cli(config)
        if exe is not None:
            hits = _search_everything(exe, params.query, params.extension)
            backend = "everything"
        else:
            root = Path(params.search_root).expanduser() if params.search_root else Path.home()
            if not root.is_dir():
                return FileSearchOutput(
                    query=params.query,
                    backend="scandir",
                    total_found=0,
                    truncated=False,
                )
            hits = _search_scandir(root, params.query, params.extension)
            backend = "scandir"

        hits.sort(key=lambda hit: hit.modified, reverse=True)
        return FileSearchOutput(
            query=params.query,
            backend=backend,
            total_found=len(hits),
            truncated=len(hits) > _MAX_RESULTS,
            results=hits[:_MAX_RESULTS],
        )
    except Exception as exc:
        _log.exception("files.search failed")
        raise ToolExecutionError(
            "files.search",
            f"the file search failed: {exc}",
            speakable="I could not search the files.",
        ) from exc
