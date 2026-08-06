"""JSON-RPC over a Windows named pipe, between the main process and the sidecar.

CLAUDE.md §6, privilege separation. The sidecar runs elevated, so its input
surface is the most security-sensitive boundary in the system. The rules encoded
here:

* Exactly three methods exist: ``get_thermals``, ``get_fans``,
  ``get_pending_updates``. Anything else is rejected before dispatch.
* **No method takes parameters.** A request carrying params is rejected
  outright. The helper therefore cannot be steered by anything an LLM writes,
  because there is no field for an LLM-authored value to travel in.
* The message parser is a pure function, :meth:`HelperServer.handle_message`,
  so the security properties can be tested without Windows, a pipe, or
  elevation.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from jarvis.util.errors import HelperUnavailableError

__all__ = [
    "ALLOWED_METHODS",
    "ErrorCode",
    "HelperClient",
    "HelperServer",
    "RpcError",
    "pipe_path",
]

_log = logging.getLogger(__name__)

JSONRPC_VERSION = "2.0"

#: The complete method surface of the elevated helper. §6 permits exactly these.
ALLOWED_METHODS = frozenset({"get_thermals", "get_fans", "get_pending_updates"})

#: Cap on an inbound message. A named pipe peer should never send more than a
#: few hundred bytes; anything larger is malformed or hostile.
MAX_MESSAGE_BYTES = 4096


class ErrorCode:
    """JSON-RPC error codes, plus the ones this helper adds."""

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    SENSOR_UNAVAILABLE = -32000


class RpcError(Exception):
    """An error to send back to the caller."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def pipe_path(pipe_name: str) -> str:
    """Full Windows named pipe path for a bare pipe name."""
    return rf"\\.\pipe\{pipe_name}"


class HelperServer:
    """Serves the three read-only sensor methods over a named pipe.

    Args:
        handlers: Mapping of method name to a zero-argument callable. Only names
            in :data:`ALLOWED_METHODS` are ever dispatched, regardless of what
            this mapping contains.
    """

    def __init__(self, handlers: dict[str, Callable[[], dict[str, Any]]]) -> None:
        unknown = set(handlers) - ALLOWED_METHODS
        if unknown:
            msg = (
                f"helper may not expose {sorted(unknown)}; "
                f"§6 permits only {sorted(ALLOWED_METHODS)}"
            )
            raise ValueError(msg)
        self._handlers = handlers
        self._running = False

    # -- the security-critical core, deliberately pure ---------------------

    def handle_message(self, raw: str | bytes) -> str:
        """Parse, validate, dispatch, and serialise one message.

        Never raises. Every failure becomes a JSON-RPC error response, because
        an elevated process that dies on malformed input is a denial of service
        against the user's own machine.
        """
        if isinstance(raw, bytes):
            if len(raw) > MAX_MESSAGE_BYTES:
                return self._error(None, ErrorCode.INVALID_REQUEST, "message too large")
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError:
                return self._error(None, ErrorCode.PARSE_ERROR, "message is not valid UTF-8")
        elif len(raw.encode("utf-8", errors="ignore")) > MAX_MESSAGE_BYTES:
            return self._error(None, ErrorCode.INVALID_REQUEST, "message too large")

        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return self._error(None, ErrorCode.PARSE_ERROR, "message is not valid JSON")

        if not isinstance(payload, dict):
            return self._error(None, ErrorCode.INVALID_REQUEST, "request must be a JSON object")

        request_id = payload.get("id")

        if payload.get("jsonrpc") != JSONRPC_VERSION:
            return self._error(request_id, ErrorCode.INVALID_REQUEST, "unsupported jsonrpc version")

        method = payload.get("method")
        if not isinstance(method, str):
            return self._error(request_id, ErrorCode.INVALID_REQUEST, "method must be a string")

        # §6: the helper accepts no arbitrary commands and never takes an
        # LLM-authored string. No method has parameters, so any params field at
        # all is a red flag and is refused rather than ignored.
        if "params" in payload and payload["params"] not in (None, {}, []):
            _log.warning(
                "rejected helper request carrying parameters",
                extra={"context": {"method": method}},
            )
            return self._error(
                request_id, ErrorCode.INVALID_PARAMS, "helper methods take no parameters"
            )

        if method not in ALLOWED_METHODS:
            _log.warning("rejected unknown helper method", extra={"context": {"method": method}})
            return self._error(request_id, ErrorCode.METHOD_NOT_FOUND, f"unknown method: {method}")

        handler = self._handlers.get(method)
        if handler is None:
            return self._error(
                request_id, ErrorCode.METHOD_NOT_FOUND, f"method not available: {method}"
            )

        try:
            result = handler()
        except RpcError as exc:
            return self._error(request_id, exc.code, exc.message)
        except Exception as exc:  # noqa: BLE001 - the elevated process must not die
            _log.exception("helper method failed", extra={"context": {"method": method}})
            return self._error(
                request_id, ErrorCode.SENSOR_UNAVAILABLE, f"{type(exc).__name__}: {exc}"
            )

        return json.dumps({"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result})

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> str:
        """Serialise a JSON-RPC error response."""
        return json.dumps(
            {
                "jsonrpc": JSONRPC_VERSION,
                "id": request_id,
                "error": {"code": code, "message": message},
            }
        )

    # -- transport ---------------------------------------------------------

    def serve_forever(self, pipe_name: str) -> None:  # pragma: no cover - needs Windows
        """Accept and answer requests on a Windows named pipe until stopped.

        One request per connection, which keeps the state machine trivial and
        means a wedged client cannot hold the pipe open indefinitely.
        """
        from jarvis.util.platform import require_module, require_windows

        require_windows("the elevated helper")
        win32pipe = require_module("win32pipe")
        win32file = require_module("win32file")

        path = pipe_path(pipe_name)
        self._running = True
        _log.info("helper listening", extra={"context": {"pipe": path}})

        while self._running:
            handle = win32pipe.CreateNamedPipe(
                path,
                win32pipe.PIPE_ACCESS_DUPLEX,
                win32pipe.PIPE_TYPE_MESSAGE
                | win32pipe.PIPE_READMODE_MESSAGE
                | win32pipe.PIPE_WAIT,
                win32pipe.PIPE_UNLIMITED_INSTANCES,
                MAX_MESSAGE_BYTES,
                MAX_MESSAGE_BYTES,
                0,
                None,
            )
            try:
                win32pipe.ConnectNamedPipe(handle, None)
                _code, data = win32file.ReadFile(handle, MAX_MESSAGE_BYTES)
                response = self.handle_message(data)
                win32file.WriteFile(handle, response.encode("utf-8"))
            except Exception:  # noqa: BLE001 - one bad client must not stop the server
                _log.exception("helper connection failed")
            finally:
                try:
                    win32pipe.DisconnectNamedPipe(handle)
                    win32file.CloseHandle(handle)
                except Exception:  # noqa: BLE001
                    _log.debug("helper handle cleanup failed", exc_info=True)

    def stop(self) -> None:
        """Ask the serve loop to exit after the current connection."""
        self._running = False


class HelperClient:
    """Main-process client for the elevated helper.

    Caches results for ``config.helper.cache_ttl_s`` so repeated questions do not
    hammer the sensors, and degrades to :class:`HelperUnavailableError` when the
    helper is not running or the user declined elevation.
    """

    def __init__(
        self,
        config: Any,
        *,
        transport: Callable[[str], str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._transport = transport
        self._clock = clock
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._next_id = 0
        self._autostart_attempted = False

    def call(self, method: str, *, use_cache: bool = True) -> dict[str, Any]:
        """Invoke one helper method.

        Args:
            method: Must be in :data:`ALLOWED_METHODS`.
            use_cache: Serve from the short-lived cache when fresh.

        Raises:
            ValueError: The method is not one of the three permitted ones.
            HelperUnavailableError: The helper could not be reached.
        """
        if method not in ALLOWED_METHODS:
            msg = f"{method} is not a helper method"
            raise ValueError(msg)

        if use_cache:
            cached = self._cache.get(method)
            if cached is not None:
                stamped, value = cached
                if self._clock() - stamped < float(self._config.helper.cache_ttl_s):
                    return value

        self._next_id += 1
        request = json.dumps(
            {"jsonrpc": JSONRPC_VERSION, "id": self._next_id, "method": method}
        )
        raw = self._send(request)

        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            raise HelperUnavailableError(
                f"helper returned malformed JSON: {raw[:200]!r}"
            ) from exc

        if "error" in payload:
            error = payload["error"]
            raise HelperUnavailableError(
                f"helper error {error.get('code')}: {error.get('message')}",
                speakable="I could not read those sensors.",
            )

        result = payload.get("result")
        if not isinstance(result, dict):
            raise HelperUnavailableError("helper returned no result object")

        self._cache[method] = (self._clock(), result)
        return result

    def get_thermals(self) -> dict[str, Any]:
        """CPU and board temperatures from LibreHardwareMonitor."""
        return self.call("get_thermals")

    def get_fans(self) -> dict[str, Any]:
        """Fan speeds from LibreHardwareMonitor."""
        return self.call("get_fans")

    def get_pending_updates(self) -> dict[str, Any]:
        """Pending Windows updates. Never installs anything."""
        return self.call("get_pending_updates")

    def is_available(self) -> bool:
        """True when the helper answers. Never raises."""
        try:
            self.call("get_thermals", use_cache=False)
        except (HelperUnavailableError, ValueError):
            return False
        return True

    def clear_cache(self) -> None:
        """Drop cached readings."""
        self._cache.clear()

    def _send(self, request: str) -> str:
        """Write one request and read one response."""
        if self._transport is not None:
            return self._transport(request)
        return self._send_over_pipe(request)

    def _send_over_pipe(self, request: str) -> str:  # pragma: no cover - needs Windows
        """Named pipe round trip, starting the helper once if necessary."""
        from jarvis.util.platform import is_windows, require_module

        if not is_windows():
            raise HelperUnavailableError(
                "the elevated helper only exists on Windows",
                speakable="I can only read those sensors on Windows.",
            )

        win32file = require_module("win32file")
        path = pipe_path(self._config.helper.pipe_name)

        for attempt in (1, 2):
            try:
                handle = win32file.CreateFile(
                    path,
                    win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                    0,
                    None,
                    win32file.OPEN_EXISTING,
                    0,
                    None,
                )
            except Exception as exc:
                if attempt == 1 and self._maybe_autostart():
                    continue
                raise HelperUnavailableError(
                    f"could not connect to the helper pipe {path}: {exc}",
                    speakable=(
                        "The elevated helper is not running, so I cannot read those sensors."
                    ),
                ) from exc

            try:
                win32file.WriteFile(handle, request.encode("utf-8"))
                _code, data = win32file.ReadFile(handle, MAX_MESSAGE_BYTES)
                return bytes(data).decode("utf-8", errors="replace")
            finally:
                try:
                    win32file.CloseHandle(handle)
                except Exception:  # noqa: BLE001
                    _log.debug("client handle cleanup failed", exc_info=True)

        raise HelperUnavailableError("the helper did not respond")

    def _maybe_autostart(self) -> bool:  # pragma: no cover - needs Windows
        """Launch ``jarvis-helper.exe`` once. Returns whether a start was tried.

        The launch raises a UAC prompt. Declining it is a normal user choice, so
        a failure here degrades rather than escalating.
        """
        if self._autostart_attempted or not self._config.helper.autostart:
            return False
        self._autostart_attempted = True

        from pathlib import Path

        from jarvis.util.platform import project_root, require_module

        exe = self._config.helper.exe_path or (project_root() / "jarvis-helper.exe")
        exe = Path(exe)
        if not exe.is_file():
            _log.warning("helper executable not found", extra={"context": {"path": str(exe)}})
            return False

        try:
            shell = require_module("win32com.shell.shell")
            # runas raises the UAC prompt. The main process itself stays
            # non-elevated, which is the whole point of §6.
            shell.ShellExecuteEx(lpVerb="runas", lpFile=str(exe), nShow=0)
        except Exception:  # noqa: BLE001 - a declined UAC prompt lands here
            _log.warning("could not start the elevated helper", exc_info=True)
            return False

        time.sleep(1.0)  # give the pipe a moment to appear
        return True
