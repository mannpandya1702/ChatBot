"""Error hierarchy for JARVIS.

The rule from CLAUDE.md §5 is "fail loud in logs, fail soft in voice". Every error
therefore carries a ``speakable`` string: a short, plain sentence the LLM can read
aloud without exposing a stack trace or raw JSON.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "AudioError",
    "ConfigError",
    "DependencyMissingError",
    "HelperUnavailableError",
    "JarvisError",
    "LlmError",
    "PlatformUnsupportedError",
    "SafetyViolationError",
    "SttError",
    "ToolExecutionError",
    "TtsError",
    "as_speakable",
    "redact",
]


class JarvisError(Exception):
    """Base class for every error raised inside JARVIS.

    Args:
        message: Developer-facing detail. Goes to the log.
        speakable: Short user-facing sentence. Goes to the LLM, then to TTS.
        context: Structured detail attached to the log record.
    """

    default_speakable = "Something went wrong on my end."

    def __init__(
        self,
        message: str,
        *,
        speakable: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.speakable = speakable or self.default_speakable
        self.context: dict[str, Any] = context or {}

    def to_dict(self) -> dict[str, Any]:
        """Render as the structured error payload handed back to the LLM."""
        return {
            "error": type(self).__name__,
            "message": self.message,
            "speakable": self.speakable,
            "context": self.context,
        }


class ConfigError(JarvisError):
    """Configuration is missing, malformed, or internally inconsistent."""

    default_speakable = "My configuration is not valid, so I cannot start that."


class DependencyMissingError(JarvisError):
    """An optional dependency needed for this code path is not installed."""

    default_speakable = "A component I need for that is not installed."

    def __init__(
        self,
        package: str,
        *,
        extra: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        hint = f"uv sync --extra {extra}" if extra else f"uv add {package}"
        super().__init__(
            f"optional dependency '{package}' is not installed; install it with: {hint}",
            speakable=f"The {package} component is not installed on this machine.",
            context={"package": package, "extra": extra, **(context or {})},
        )
        self.package = package
        self.extra = extra


class PlatformUnsupportedError(JarvisError):
    """This code path only exists on a platform we are not currently running on."""

    default_speakable = "That is only available on Windows."

    def __init__(
        self,
        feature: str,
        *,
        required: str = "Windows",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            f"{feature} requires {required}",
            speakable=f"I can only do that on {required}.",
            context={"feature": feature, "required": required, **(context or {})},
        )
        self.feature = feature


class AudioError(JarvisError):
    """Capture or playback failed."""

    default_speakable = "I had trouble with the audio device."


class SttError(JarvisError):
    """Speech recognition failed."""

    default_speakable = "I did not catch that."


class TtsError(JarvisError):
    """Speech synthesis failed."""

    default_speakable = "I could not speak that."


class LlmError(JarvisError):
    """The Ollama backend failed, timed out, or returned something unusable."""

    default_speakable = "My language model is not responding."


class ToolExecutionError(JarvisError):
    """A tool raised while running. Never fatal to the turn loop."""

    default_speakable = "I could not read that sensor."

    def __init__(
        self,
        tool_name: str,
        message: str,
        *,
        speakable: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            f"tool '{tool_name}' failed: {message}",
            speakable=speakable,
            context={"tool": tool_name, **(context or {})},
        )
        self.tool_name = tool_name


class HelperUnavailableError(JarvisError):
    """The elevated sidecar is not running or refused the connection."""

    default_speakable = "The elevated helper is not running, so I cannot read those sensors."


class SafetyViolationError(JarvisError):
    """A call tried to cross a §6 safety boundary. Always refuse, never soften."""

    default_speakable = "I am not permitted to do that."


def as_speakable(exc: BaseException) -> str:
    """Return a sentence safe to hand to TTS for any exception.

    Non-JARVIS exceptions never leak their message, since it may contain paths,
    argument values, or other detail that has no business being spoken aloud.
    """
    if isinstance(exc, JarvisError):
        return exc.speakable
    return "Something went wrong on my end."


#: Home directory paths, which name the user, and URL userinfo, which carries
#: credentials. Both turn up verbatim in ordinary OSError and httpx messages.
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"[A-Za-z]:\\+Users\\+[^\\\"'\s]+(?:\\+[^\"'\s]*)?"), "<path>"),
    (re.compile(r"/(?:home|Users)/[^/\"'\s]+(?:/[^\"'\s]*)?"), "<path>"),
    (re.compile(r"(?<=://)[^/@\s]+:[^/@\s]+@"), "<credentials>@"),
)


def redact(text: str) -> str:
    """Strip user-identifying paths and URL credentials from error text.

    The full text still reaches the log file, which is local. This is for the
    copies that travel: the tool error handed to the model, which is persisted
    into the conversation memory and re-sent on every later turn.
    """
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text
