"""Application launch, focus, and window listing.

T-4.2, gated: ``apps.launch`` and ``apps.focus`` are on the §6 mutating
allowlist. ``apps.list_windows`` is read-only and is not.

Launching is deliberately restrictive. The model supplies a friendly name such
as "spotify", which is resolved against a known-applications map and then the
Start Menu shortcuts. An arbitrary path or command line from the model is
refused, because ``apps.launch`` would otherwise be a second, unhardened
``shell.run``.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from pydantic import Field

from jarvis.tools.registry import ToolCategory, ToolInput, ToolOutput, tool
from jarvis.util.errors import SafetyViolationError, ToolExecutionError
from jarvis.util.platform import is_windows, require_module

__all__ = [
    "AppLaunchInput",
    "AppLaunchOutput",
    "WindowInfo",
    "focus_window",
    "launch_app",
    "list_windows",
    "resolve_application",
]

_log = logging.getLogger(__name__)

_MAX_WINDOWS = 5

#: Friendly name to executable. Resolution is by name only; the model never
#: supplies a path, so this map plus the Start Menu is the entire launch surface.
_KNOWN_APPS: dict[str, str] = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "paint": "mspaint.exe",
    "explorer": "explorer.exe",
    "file explorer": "explorer.exe",
    "task manager": "taskmgr.exe",
    "control panel": "control.exe",
    "settings": "ms-settings:",
    "snipping tool": "snippingtool.exe",
    "character map": "charmap.exe",
    "spotify": "spotify.exe",
    "chrome": "chrome.exe",
    "google chrome": "chrome.exe",
    "firefox": "firefox.exe",
    "edge": "msedge.exe",
    "microsoft edge": "msedge.exe",
    "discord": "discord.exe",
    "steam": "steam.exe",
    "vs code": "code.exe",
    "visual studio code": "code.exe",
    "code": "code.exe",
    "obs": "obs64.exe",
}

#: Anything resolving to one of these is refused outright. Launching a shell
#: would hand the model an unhardened execution path around §6.
_FORBIDDEN_TARGETS = frozenset(
    {
        "cmd.exe", "powershell.exe", "pwsh.exe", "wscript.exe", "cscript.exe",
        "mshta.exe", "rundll32.exe", "regsvr32.exe", "regedit.exe", "wmic.exe",
        "bash.exe", "wsl.exe", "sh.exe", "python.exe", "certutil.exe",
    }
)


class WindowInfo(ToolOutput):
    """One top-level window."""

    title: str = Field(description="Window title text.")
    process: str = Field(description="Owning process name.")
    is_active: bool = Field(default=False, description="Whether it currently has focus.")


class AppLaunchInput(ToolInput):
    """Arguments for launching an application."""

    name: str = Field(
        min_length=1,
        max_length=100,
        description=(
            "Friendly application name, for example spotify, chrome, or notepad. "
            "A file path or command line is not accepted."
        ),
    )


class AppLaunchOutput(ToolOutput):
    """Outcome of a launch attempt."""

    launched: bool = Field(description="Whether the application was started.")
    name: str = Field(description="The name that was requested.")
    resolved_to: str | None = Field(default=None, description="What it resolved to.")
    detail: str = Field(description="A short spoken description of the outcome.")


class AppFocusInput(ToolInput):
    """Arguments for focusing a window."""

    name: str = Field(
        min_length=1,
        max_length=100,
        description="Part of the window title or the application name, case insensitive.",
    )


class AppFocusOutput(ToolOutput):
    """Outcome of a focus attempt."""

    focused: bool = Field(description="Whether a window was brought to the front.")
    title: str | None = Field(default=None, description="Title of the window that was focused.")
    detail: str = Field(description="A short spoken description of the outcome.")


class WindowListInput(ToolInput):
    """Arguments for listing windows."""


class WindowListOutput(ToolOutput):
    """Currently open windows."""

    count: int = Field(description="How many visible top-level windows are open.")
    windows: list[WindowInfo] = Field(
        default_factory=list, description="Up to 5 windows, active one first."
    )


def resolve_application(name: str) -> str | None:
    """Resolve a friendly name to something launchable, or None.

    Resolution order: the known-applications map, then PATH, then Start Menu
    shortcuts. A resolution landing on a shell or script host is refused.

    Raises:
        SafetyViolationError: The name resolved to a forbidden target.
    """
    needle = name.strip().lower()
    if not needle:
        return None

    # A path or a command line is not a name. Refuse rather than interpret.
    if any(ch in needle for ch in ("\\", "/", "&", "|", ";", '"', "'")) or needle.endswith(
        (".bat", ".cmd", ".ps1", ".vbs", ".js")
    ):
        raise SafetyViolationError(
            f"apps.launch takes an application name, not a path or command: {name!r}",
            speakable="I can only open applications by name.",
        )

    target = _KNOWN_APPS.get(needle)
    if target is None and needle.endswith(".exe"):
        target = needle
    if target is None:
        found = shutil.which(needle) or shutil.which(f"{needle}.exe")
        if found:
            target = Path(found).name
    if target is None:
        target = _find_start_menu_shortcut(needle)

    if target is None:
        return None

    if Path(target).name.lower() in _FORBIDDEN_TARGETS:
        raise SafetyViolationError(
            f"apps.launch will not start {target}",
            speakable="I am not permitted to open that.",
        )
    return target


def _find_start_menu_shortcut(needle: str) -> str | None:
    """Look for a matching .lnk in the Start Menu folders."""
    if not is_windows():
        return None
    roots = [
        Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
        Path(os.environ.get("PROGRAMDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
    ]
    for root in roots:
        if not root.is_dir():
            continue
        try:
            for shortcut in root.rglob("*.lnk"):
                if needle in shortcut.stem.lower():
                    return str(shortcut)
        except (OSError, PermissionError):
            continue
    return None


@tool(
    name="apps.launch",
    description=(
        "Open an application by name, for example spotify, chrome, notepad, or calculator. "
        "Only accepts an application name; file paths and command lines are refused, and it "
        "will not open a shell or script host. This changes system state, so it requires the "
        "user to confirm out loud first."
    ),
    category=ToolCategory.APPS,
    read_only=False,
    requires_windows=True,
)
def launch_app(params: AppLaunchInput) -> AppLaunchOutput:
    """Start an application.

    Args:
        params: The friendly application name.

    Returns:
        Whether it started, phrased for speech.

    Raises:
        ToolExecutionError: The launch failed for a reason other than a refusal.
    """
    if not is_windows():
        return AppLaunchOutput(
            launched=False,
            name=params.name,
            detail="I can only open applications on Windows.",
        )

    target = resolve_application(params.name)
    if target is None:
        return AppLaunchOutput(
            launched=False,
            name=params.name,
            detail=f"I could not find an application called {params.name}.",
        )

    try:
        if target.endswith(".lnk") or target.startswith("ms-settings:"):
            # os.startfile exists only on Windows, so it is fetched dynamically
            # rather than referenced directly, which would fail type checking on
            # the Linux build host (§0b).
            start_file = getattr(os, "startfile", None)
            if start_file is None:  # pragma: no cover - guarded by is_windows above
                raise ToolExecutionError(
                    "apps.launch",
                    "shortcuts can only be opened on Windows",
                    speakable="I can only open that on Windows.",
                )
            start_file(target)
        else:
            subprocess.Popen(  # noqa: S603 - fixed argv, no shell, resolved target
                [target],
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        _log.exception("apps.launch failed", extra={"context": {"target": target}})
        raise ToolExecutionError(
            "apps.launch",
            f"could not start {target}: {exc}",
            speakable=f"I could not open {params.name}.",
        ) from exc

    return AppLaunchOutput(
        launched=True,
        name=params.name,
        resolved_to=target,
        detail=f"Opened {params.name}.",
    )


def _windows() -> list[WindowInfo]:
    """Enumerate visible top-level windows."""
    gw = require_module("pygetwindow", feature="window listing")
    rows: list[WindowInfo] = []
    try:
        active = gw.getActiveWindow()
        active_title = getattr(active, "title", None)
    except Exception:  # noqa: BLE001 - no active window is a normal state
        active_title = None

    for window in gw.getAllWindows():
        title = (getattr(window, "title", "") or "").strip()
        if not title:
            continue
        if not getattr(window, "visible", True):
            continue
        rows.append(
            WindowInfo(
                title=title[:120],
                process=title.split(" - ")[-1][:60],
                is_active=title == active_title,
            )
        )
    rows.sort(key=lambda row: (not row.is_active, row.title.lower()))
    return rows


@tool(
    name="apps.list_windows",
    description=(
        "List the application windows that are currently open. Use this for questions like "
        "what do I have open, what am I running, or which window is in front. Returns at "
        "most 5 windows with the active one first. Read-only."
    ),
    category=ToolCategory.APPS,
    read_only=True,
    requires_windows=True,
)
def list_windows(params: WindowListInput) -> WindowListOutput:
    """List open windows.

    Args:
        params: No arguments.

    Returns:
        Up to five windows, active one first.

    Raises:
        ToolExecutionError: Window enumeration failed.
    """
    if not is_windows():
        return WindowListOutput(count=0)
    try:
        rows = _windows()
    except Exception as exc:
        _log.exception("apps.list_windows failed")
        raise ToolExecutionError(
            "apps.list_windows",
            f"could not enumerate windows: {exc}",
            speakable="I could not see which windows are open.",
        ) from exc
    return WindowListOutput(count=len(rows), windows=rows[:_MAX_WINDOWS])


@tool(
    name="apps.focus",
    description=(
        "Bring an already-open window to the front by name, for example bring Chrome to the "
        "front or switch to Spotify. Matches part of the window title, case insensitive. "
        "This changes system state, so it requires the user to confirm out loud first."
    ),
    category=ToolCategory.APPS,
    read_only=False,
    requires_windows=True,
)
def focus_window(params: AppFocusInput) -> AppFocusOutput:
    """Bring a window to the foreground.

    Args:
        params: Part of the window title.

    Returns:
        Whether a window was focused, phrased for speech.

    Raises:
        ToolExecutionError: The focus attempt failed.
    """
    if not is_windows():
        return AppFocusOutput(focused=False, detail="I can only switch windows on Windows.")

    gw = require_module("pygetwindow", feature="window focus")
    needle = params.name.strip().lower()
    try:
        matches = [
            window
            for window in gw.getAllWindows()
            if needle in (getattr(window, "title", "") or "").lower()
        ]
        if not matches:
            return AppFocusOutput(
                focused=False, detail=f"I could not find a window matching {params.name}."
            )
        window = matches[0]
        # A minimised window has to be restored before it can take focus.
        if getattr(window, "isMinimized", False):
            window.restore()
        window.activate()
    except Exception as exc:
        _log.exception("apps.focus failed")
        raise ToolExecutionError(
            "apps.focus",
            f"could not focus a window matching {params.name}: {exc}",
            speakable=f"I could not switch to {params.name}.",
        ) from exc

    title = (getattr(window, "title", "") or "")[:120]
    return AppFocusOutput(focused=True, title=title, detail=f"Switched to {title}.")
