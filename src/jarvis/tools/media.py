"""Media transport control via Windows media keys.

T-4.3, and gated: play/pause, next, previous, and volume are on the §6 mutating
allowlist, so every call routes through ``tools/gate.py`` and needs a spoken
confirmation before the registry will run it.

Media keys are used rather than per-application APIs because they work with
whatever is currently playing, which is what a spoken request means.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import Field

from jarvis.tools.registry import ToolCategory, ToolInput, ToolOutput, tool
from jarvis.util.errors import ToolExecutionError
from jarvis.util.platform import is_windows, require_module

__all__ = ["MediaAction", "MediaInput", "MediaOutput", "media_control"]

_log = logging.getLogger(__name__)

MediaAction = Literal[
    "play_pause", "next", "previous", "stop", "volume_up", "volume_down", "mute"
]

#: Windows virtual key codes for the media and volume keys.
_VK_CODES: dict[str, int] = {
    "play_pause": 0xB3,
    "next": 0xB0,
    "previous": 0xB1,
    "stop": 0xB2,
    "volume_up": 0xAF,
    "volume_down": 0xAE,
    "mute": 0xAD,
}

#: How many times a volume key is tapped per step. Each tap is about 2 percent,
#: so a single tap is imperceptible and a spoken request means more than that.
_VOLUME_TAPS = 5

_KEYEVENTF_KEYUP = 0x0002


class MediaInput(ToolInput):
    """Arguments for the media control tool."""

    action: MediaAction = Field(
        description=(
            "What to do: play_pause toggles playback, next skips forward, previous goes back, "
            "stop halts playback, volume_up and volume_down change the system volume by "
            "roughly ten percent, mute toggles mute."
        )
    )
    repeat: int = Field(
        default=1,
        ge=1,
        le=10,
        description="How many times to apply the action, for example skipping three tracks.",
    )


class MediaOutput(ToolOutput):
    """Outcome of a media command."""

    performed: bool = Field(description="Whether the key was actually sent.")
    action: str = Field(description="The action that was requested.")
    repeat: int = Field(description="How many times it was applied.")
    detail: str = Field(description="A short spoken description of what happened.")


def _send_key(code: int) -> None:
    """Press and release one virtual key.

    Uses keybd_event through pywin32, which reaches the shell's media handler.
    """
    win32api = require_module("win32api", feature="media keys")
    win32api.keybd_event(code, 0, 0, 0)
    win32api.keybd_event(code, 0, _KEYEVENTF_KEYUP, 0)


@tool(
    name="media.control",
    description=(
        "Control whatever is currently playing: play or pause, skip to the next or previous "
        "track, stop, change the system volume, or toggle mute. Use this for requests like "
        "pause the music, skip this song, turn it down, or mute. This changes system state, "
        "so it requires the user to confirm out loud first."
    ),
    category=ToolCategory.MEDIA,
    read_only=False,
    requires_windows=True,
)
def media_control(params: MediaInput) -> MediaOutput:
    """Send a media key.

    Args:
        params: Which action, and how many times.

    Returns:
        What was done, phrased for speech.

    Raises:
        ToolExecutionError: The key could not be sent.
    """
    if not is_windows():
        return MediaOutput(
            performed=False,
            action=params.action,
            repeat=0,
            detail="I can only control media playback on Windows.",
        )

    code = _VK_CODES.get(params.action)
    if code is None:  # pragma: no cover - the Literal makes this unreachable
        raise ToolExecutionError("media.control", f"unknown action {params.action}")

    # Volume steps are tiny, so a spoken "turn it down" taps several times.
    taps = _VOLUME_TAPS if params.action in ("volume_up", "volume_down") else 1

    try:
        for _ in range(params.repeat):
            for _ in range(taps):
                _send_key(code)
    except Exception as exc:
        _log.exception("media.control failed")
        raise ToolExecutionError(
            "media.control",
            f"could not send the {params.action} key: {exc}",
            speakable="I could not reach the media controls.",
        ) from exc

    spoken = {
        "play_pause": "Toggled playback",
        "next": "Skipped forward",
        "previous": "Went back",
        "stop": "Stopped playback",
        "volume_up": "Turned the volume up",
        "volume_down": "Turned the volume down",
        "mute": "Toggled mute",
    }[params.action]
    if params.repeat > 1:
        spoken = f"{spoken} {params.repeat} times"

    return MediaOutput(
        performed=True,
        action=params.action,
        repeat=params.repeat,
        detail=f"{spoken}.",
    )
