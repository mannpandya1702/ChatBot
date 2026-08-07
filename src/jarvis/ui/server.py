"""WebSocket state broadcaster for the HUD.

T-3.1: broadcast ``{state, audio_level, transcript, response, metrics}`` on
localhost at 30 Hz. States are idle, listening, thinking, speaking, error.

Two design points matter:

* The broadcaster runs its own asyncio loop on its own thread. The turn loop is
  threaded and must never block on a WebSocket write (§5), so events cross the
  boundary through the event bus's queue subscription, which drops the oldest
  message rather than blocking the publisher.
* Broadcasting is rate limited and change-aware. At 30 Hz a naive implementation
  would push an identical frame 30 times a second forever; here a frame goes out
  on the tick only when something actually changed, plus a periodic keepalive so
  a freshly connected client always gets current state.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from jarvis.config import JarvisConfig
from jarvis.state import AssistantState, Event, EventBus, EventType

__all__ = ["HudState", "UiServer"]

_log = logging.getLogger(__name__)

#: A client that cannot keep up is dropped rather than allowed to buffer.
_SEND_TIMEOUT_S = 2.0
#: Resend current state at least this often even when nothing changed.
_KEEPALIVE_S = 1.0


@dataclass
class HudState:
    """Everything the HUD renders. Serialised as one JSON frame."""

    state: str = AssistantState.IDLE.value
    audio_level: float = 0.0
    transcript: str = ""
    partial_transcript: str = ""
    response: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    tool: str | None = None
    confirmation: str | None = None
    error: str | None = None
    history: deque[str] = field(default_factory=lambda: deque(maxlen=20))

    def to_frame(self) -> dict[str, Any]:
        """The JSON payload sent to clients."""
        return {
            "type": "state",
            "state": self.state,
            "audio_level": round(self.audio_level, 4),
            "transcript": self.transcript,
            "partial_transcript": self.partial_transcript,
            "response": self.response,
            "metrics": self.metrics,
            "tool": self.tool,
            "confirmation": self.confirmation,
            "error": self.error,
            "history": list(self.history),
            "ts": time.time(),
        }

    def apply(self, event: Event) -> bool:
        """Fold one bus event into the state. Returns whether anything changed."""
        payload = event.payload
        match event.type:
            case EventType.STATE_CHANGED:
                new_state = str(payload.get("state", self.state))
                if new_state == self.state:
                    return False
                self.state = new_state
                if new_state != AssistantState.ERROR.value:
                    self.error = None
                return True

            case EventType.AUDIO_LEVEL:
                level = float(payload.get("level", 0.0))
                # The orb reacts to amplitude, so small changes matter, but
                # floating point jitter should not count as a change.
                if abs(level - self.audio_level) < 0.005:
                    return False
                self.audio_level = level
                return True

            case EventType.PARTIAL_TRANSCRIPT:
                text = str(payload.get("text", ""))
                if text == self.partial_transcript:
                    return False
                self.partial_transcript = text
                return True

            case EventType.TRANSCRIPT:
                self.transcript = str(payload.get("text", ""))
                self.partial_transcript = ""
                self.response = ""
                if self.transcript:
                    self.history.append(f"you: {self.transcript}")
                return True

            case EventType.RESPONSE_CHUNK:
                self.response += str(payload.get("text", ""))
                return True

            case EventType.RESPONSE:
                self.response = str(payload.get("text", self.response))
                if self.response:
                    self.history.append(f"jarvis: {self.response}")
                return True

            case EventType.TOOL_CALL:
                self.tool = str(payload.get("name", ""))
                return True

            case EventType.TOOL_RESULT:
                self.tool = None
                return True

            case EventType.CONFIRMATION_REQUIRED:
                self.confirmation = str(payload.get("prompt", ""))
                return True

            case EventType.CONFIRMATION_RESOLVED:
                self.confirmation = None
                return True

            case EventType.METRICS:
                self.metrics.update(payload)
                return True

            case EventType.ERROR:
                self.error = str(payload.get("speakable") or payload.get("message") or "error")
                return True

            case _:
                return False


class UiServer:
    """Serves HUD state over a WebSocket on its own thread.

    Args:
        config: Supplies host, port, and broadcast rate.
        bus: Event source. The server subscribes with a queue so it can never
            stall the audio path.
    """

    def __init__(self, config: JarvisConfig, bus: EventBus) -> None:
        self._config = config
        self._bus = bus
        self.state = HudState()

        self._clients: set[Any] = set()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: Any = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._queue: queue.Queue[Event] | None = None
        self._unsubscribe: Any = None
        # A one-slot handoff from the server thread to start(). A list rather
        # than an optional attribute because the write happens on another
        # thread, which a type checker cannot see.
        self._error: list[BaseException] = []
        self._port = config.ui.port

    # -- lifecycle ---------------------------------------------------------

    @property
    def port(self) -> int:
        """The bound port. Differs from the configured one when 0 was requested."""
        return self._port

    @property
    def client_count(self) -> int:
        """How many HUD clients are connected."""
        return len(self._clients)

    @property
    def is_running(self) -> bool:
        """True while the server thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self, timeout: float = 5.0) -> None:
        """Start the server thread and block until it is listening."""
        if self.is_running:
            return
        self._stop.clear()
        self._ready.clear()
        self._error.clear()
        # Release any subscription from a previous failed start, otherwise each
        # attempt leaves a queue behind that fills and is never drained.
        if self._unsubscribe is not None:
            self._unsubscribe()
        self._queue, self._unsubscribe = self._bus.subscribe_queue()
        self._thread = threading.Thread(target=self._run, name="jarvis-ui", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            self._cleanup_failed_start()
            msg = "the UI server did not start in time"
            raise TimeoutError(msg)
        if self._error:
            # _run sets _ready on failure so start() cannot hang, which means
            # the flag alone does not mean success. Without this check a HUD
            # that could not bind its port is reported as running.
            error = self._error[0]
            self._cleanup_failed_start()
            raise error

    def _cleanup_failed_start(self) -> None:
        """Release the bus subscription after a start that did not succeed."""
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        self._queue = None
        self._thread = None

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the server and release the port."""
        self._stop.set()
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        # Deliberately no loop.stop() here. Setting the stop flag lets _pump
        # return, which lets the websockets serve() context manager close the
        # listening socket and every client before run_until_complete returns.
        # Stopping the loop from outside would tear it down mid-shutdown and
        # leave Server._close as a never-awaited coroutine.
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def __enter__(self) -> UiServer:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # -- server thread -----------------------------------------------------

    def _run(self) -> None:
        """Thread body: own asyncio loop, serve until stopped."""
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._serve())
        except Exception as exc:  # noqa: BLE001 - report and unblock start(), never hang
            _log.exception("UI server failed")
            self._error.append(exc)
            self._ready.set()  # unblock start(), which then re-raises this
        finally:
            with contextlib.suppress(Exception):
                loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            self._loop = None

    async def _serve(self) -> None:
        """Bind the socket and run the pump until stop is requested."""
        import websockets

        async with websockets.serve(
            self._handle_client,
            self._config.ui.host,
            self._config.ui.port,
            ping_interval=20,
            ping_timeout=20,
            origins=self._allowed_origins(),
        ) as server:
            self._server = server
            for sock in server.sockets or []:
                self._port = sock.getsockname()[1]
                break
            _log.info(
                "HUD websocket listening",
                extra={"context": {"host": self._config.ui.host, "port": self._port}},
            )
            self._ready.set()
            await self._pump()

    def _allowed_origins(self) -> list[Any] | None:
        """Which browser origins may open this socket.

        Binding to 127.0.0.1 is not access control here. A WebSocket handshake
        is exempt from the same origin policy: there is no preflight and no
        CORS, so any page the user has open, down to an ad in an iframe, can
        connect to ws://127.0.0.1:8765 and read whatever is broadcast. What
        this socket broadcasts is the live microphone transcript, the reply, the
        rolling history, and the confirmation prompts for mutating actions. An
        unrestricted one is a transcript exfiltration channel that is on by
        default, which is a worse breach of §0.1 than any cloud call.

        ``websockets`` matches a missing Origin against ``None``, which is what
        every non browser client sends. That entry is configurable but on by
        default, because the Tauri shell and every debugging script rely on it
        and a browser always sends an Origin.

        Returns:
            The allowlist, or None to accept anything, which only happens when
            the config explicitly asks for it.
        """
        ui = self._config.ui
        # websockets types an origin as a NewType over str. Casting rather than
        # importing it keeps this module free of a websockets import at module
        # scope, which §0b requires.
        origins: list[Any] = [origin for origin in ui.allowed_origins if origin != "*"]
        if "*" in ui.allowed_origins:
            _log.warning(
                "ui.allowed_origins contains *, so any website the user visits can read "
                "the transcript from this socket"
            )
            return None
        if ui.allow_originless:
            origins.append(None)
        return origins

    async def _handle_client(self, websocket: Any) -> None:
        """Serve one HUD client until it disconnects."""
        self._clients.add(websocket)
        _log.info("HUD client connected", extra={"context": {"clients": len(self._clients)}})
        try:
            # A client that connects mid-session must see current state at once
            # rather than waiting for the next change.
            await websocket.send(json.dumps(self.state.to_frame()))
            async for _message in websocket:
                # The HUD is a display. It sends nothing that changes behaviour,
                # so inbound messages are drained and ignored on purpose.
                pass
        except Exception:  # noqa: BLE001 - a dropped client is entirely routine
            _log.debug("HUD client error", exc_info=True)
        finally:
            self._clients.discard(websocket)
            _log.info("HUD client disconnected", extra={"context": {"clients": len(self._clients)}})

    async def _pump(self) -> None:
        """Drain the event queue and broadcast at the configured rate."""
        interval = 1.0 / float(self._config.ui.broadcast_hz)
        last_sent = 0.0
        dirty = False

        while not self._stop.is_set():
            dirty = self._drain() or dirty

            now = time.monotonic()
            if self._clients and (dirty or now - last_sent >= _KEEPALIVE_S):
                await self._broadcast(self.state.to_frame())
                last_sent = now
                dirty = False

            await asyncio.sleep(interval)

    def _drain(self) -> bool:
        """Apply every queued event. Returns whether state changed."""
        if self._queue is None:
            return False
        changed = False
        while True:
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                return changed
            try:
                changed = self.state.apply(event) or changed
            except Exception:  # noqa: BLE001 - a bad event must not kill the HUD
                _log.exception(
                    "could not apply event to HUD state",
                    extra={"context": {"event_type": str(event.type)}},
                )

    async def _broadcast(self, frame: dict[str, Any]) -> None:
        """Send a frame to every client, dropping any that cannot keep up."""
        if not self._clients:
            return
        message = json.dumps(frame)
        stale: list[Any] = []
        for client in list(self._clients):
            try:
                await asyncio.wait_for(client.send(message), timeout=_SEND_TIMEOUT_S)
            except Exception:  # noqa: BLE001 - slow or gone, either way drop it
                stale.append(client)
        for client in stale:
            self._clients.discard(client)
            with contextlib.suppress(Exception):
                await client.close()

    # -- direct updates ----------------------------------------------------

    def publish_metrics(self, metrics: dict[str, Any]) -> None:
        """Push a metrics sample without going through the bus."""
        self._bus.emit(EventType.METRICS, **metrics)
