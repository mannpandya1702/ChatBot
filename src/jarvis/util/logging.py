"""Structured logging: JSON lines to file, readable text to console.

CLAUDE.md §5: "fail loud in logs, fail soft in voice". The file handler is the
loud half. It captures full tracebacks and structured context so a failure can be
diagnosed after the fact, while the spoken channel stays a single calm sentence.

This module deliberately takes plain arguments rather than a config object so
that ``jarvis.config`` and ``jarvis.util.logging`` never import each other.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "ConsoleFormatter",
    "JsonFormatter",
    "get_logger",
    "setup_logging",
]

# LogRecord attributes that are not user-supplied context.
_RESERVED = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
        "levelname", "levelno", "lineno", "message", "module", "msecs", "msg", "name",
        "pathname", "process", "processName", "relativeCreated", "stack_info",
        "taskName", "thread", "threadName",
    }
)

_LEVEL_COLORS = {
    "DEBUG": "\033[38;5;244m",
    "INFO": "\033[38;5;39m",
    "WARNING": "\033[38;5;214m",
    "ERROR": "\033[38;5;203m",
    "CRITICAL": "\033[1;38;5;196m",
}
_RESET = "\033[0m"
_DIM = "\033[38;5;244m"

_configured = False
_configure_lock = threading.Lock()


class JsonFormatter(logging.Formatter):
    """One JSON object per line, safe to tail and safe to parse."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "thread": record.threadName,
        }
        context = getattr(record, "context", None)
        if isinstance(context, dict):
            payload["context"] = context
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED and key != "context" and not key.startswith("_")
        }
        if extras:
            payload.setdefault("context", {}).update(extras)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, default=_json_default, ensure_ascii=False)


def _json_default(value: Any) -> str:
    """Fallback encoder so a stray object never kills a log line."""
    return repr(value)


class ConsoleFormatter(logging.Formatter):
    """Compact, optionally coloured single-line output for a terminal."""

    def __init__(self, *, color: bool = True) -> None:
        super().__init__()
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        stamp = datetime.fromtimestamp(record.created, tz=UTC).strftime("%H:%M:%S.%f")[:-3]
        level = record.levelname
        name = record.name.removeprefix("jarvis.")
        message = record.getMessage()

        context = getattr(record, "context", None)
        suffix = ""
        if isinstance(context, dict) and context:
            suffix = " " + " ".join(f"{k}={_short(v)}" for k, v in context.items())

        if self.color:
            tint = _LEVEL_COLORS.get(level, "")
            line = (
                f"{_DIM}{stamp}{_RESET} {tint}{level:<8}{_RESET} "
                f"{_DIM}{name}{_RESET} {message}{_DIM}{suffix}{_RESET}"
            )
        else:
            line = f"{stamp} {level:<8} {name} {message}{suffix}"

        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def _short(value: Any, limit: int = 80) -> str:
    """Render a context value compactly for the console."""
    if isinstance(value, dict | list):
        text = json.dumps(value, default=_json_default)
    else:
        text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def setup_logging(
    *,
    level: str = "INFO",
    log_dir: Path | str = "logs",
    json_file: str = "jarvis.jsonl",
    console: bool = True,
    console_level: str | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    color: bool | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the ``jarvis`` logger tree. Idempotent unless ``force`` is set.

    Args:
        level: Threshold for the file handler and the logger itself.
        log_dir: Directory for the JSON log. Created if absent.
        json_file: File name inside ``log_dir``. Empty disables file logging.
        console: Whether to attach a console handler.
        console_level: Console threshold. Defaults to ``level``.
        max_bytes: Rotation size for the JSON log.
        backup_count: Number of rotated files to keep.
        color: Force colour on or off. Defaults to auto-detecting a TTY.
        force: Reconfigure even if setup already ran.

    Returns:
        The configured ``jarvis`` logger.
    """
    global _configured  # noqa: PLW0603 - module-level singleton guard

    root = logging.getLogger("jarvis")
    with _configure_lock:
        if _configured and not force:
            return root

        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()

        root.setLevel(_coerce_level(level))
        root.propagate = False

        if json_file:
            directory = Path(log_dir)
            try:
                directory.mkdir(parents=True, exist_ok=True)
                file_handler: logging.Handler = logging.handlers.RotatingFileHandler(
                    directory / json_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8",
                )
                file_handler.setFormatter(JsonFormatter())
                file_handler.setLevel(_coerce_level(level))
                root.addHandler(file_handler)
            except OSError as exc:  # read-only volume, permissions, full disk
                print(f"jarvis: file logging disabled ({exc})", file=sys.stderr)  # noqa: T201

        if console:
            use_color = sys.stderr.isatty() if color is None else color
            stream = logging.StreamHandler(sys.stderr)
            stream.setFormatter(ConsoleFormatter(color=use_color))
            stream.setLevel(_coerce_level(console_level or level))
            root.addHandler(stream)

        if not root.handlers:
            root.addHandler(logging.NullHandler())

        _configured = True
        return root


def _coerce_level(level: str | int) -> int:
    """Turn a level name or number into a logging level int."""
    if isinstance(level, int):
        return level
    resolved = logging.getLevelNamesMapping().get(level.upper())
    if resolved is None:
        msg = f"unknown log level: {level!r}"
        raise ValueError(msg)
    return resolved


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``jarvis`` tree.

    ``get_logger(__name__)`` from inside the package returns that module's logger
    unchanged; anything else is nested under ``jarvis.`` so it inherits handlers.
    """
    if name == "jarvis" or name.startswith("jarvis."):
        return logging.getLogger(name)
    return logging.getLogger(f"jarvis.{name}")
