"""Assistant state machine and the event bus that fans state out to subscribers.

The turn loop runs across several threads (§5: no blocking calls in the audio
path), and the HUD runs in an asyncio loop of its own. Both need to observe the
same stream of events, so everything goes through one thread-safe bus.

Subscribers come in two flavours:

* Callbacks, invoked synchronously on the publishing thread. Fast handlers only.
* Queues, drained by the subscriber at its own pace. Slow consumers such as the
  WebSocket broadcaster use these so they cannot stall the audio path.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "AssistantState",
    "Event",
    "EventBus",
    "EventType",
    "StateMachine",
]

_log = logging.getLogger(__name__)


class AssistantState(StrEnum):
    """States the HUD renders and the orchestrator drives (§7 T-3.1)."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


class EventType(StrEnum):
    """Everything that can be published on the bus."""

    STATE_CHANGED = "state_changed"
    WAKE_DETECTED = "wake_detected"
    AUDIO_LEVEL = "audio_level"
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"
    PARTIAL_TRANSCRIPT = "partial_transcript"
    TRANSCRIPT = "transcript"
    RESPONSE_CHUNK = "response_chunk"
    RESPONSE = "response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    CONFIRMATION_REQUIRED = "confirmation_required"
    CONFIRMATION_RESOLVED = "confirmation_resolved"
    BARGE_IN = "barge_in"
    METRICS = "metrics"
    LATENCY = "latency"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass(slots=True, frozen=True)
class Event:
    """A single bus message."""

    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, as sent to the HUD."""
        return {
            "type": str(self.type),
            "payload": self.payload,
            "timestamp": self.timestamp,
            "event_id": self.event_id,
        }


Handler = Callable[[Event], None]


class EventBus:
    """Thread-safe publish and subscribe.

    A handler that raises is logged and unsubscribed is not attempted; the bus
    keeps going. Nothing a subscriber does may break the turn loop (§5).
    """

    def __init__(self, *, queue_maxsize: int = 512) -> None:
        self._lock = threading.RLock()
        self._handlers: list[tuple[Handler, frozenset[EventType] | None]] = []
        self._queues: list[tuple[queue.Queue[Event], frozenset[EventType] | None]] = []
        self._queue_maxsize = queue_maxsize
        self._closed = False

    # -- subscription ------------------------------------------------------

    def subscribe(
        self,
        handler: Handler,
        types: Iterable[EventType] | None = None,
    ) -> Callable[[], None]:
        """Register a synchronous callback. Returns an unsubscribe function."""
        wanted = frozenset(types) if types is not None else None
        entry = (handler, wanted)
        with self._lock:
            self._handlers.append(entry)

        def unsubscribe() -> None:
            with self._lock, contextlib.suppress(ValueError):
                self._handlers.remove(entry)

        return unsubscribe

    def subscribe_queue(
        self,
        types: Iterable[EventType] | None = None,
        *,
        maxsize: int | None = None,
    ) -> tuple[queue.Queue[Event], Callable[[], None]]:
        """Register a queue subscription for a slow consumer.

        The queue drops the oldest event when full rather than blocking the
        publisher, because the publisher may be the audio thread.
        """
        wanted = frozenset(types) if types is not None else None
        q: queue.Queue[Event] = queue.Queue(maxsize=maxsize or self._queue_maxsize)
        entry = (q, wanted)
        with self._lock:
            self._queues.append(entry)

        def unsubscribe() -> None:
            with self._lock, contextlib.suppress(ValueError):
                self._queues.remove(entry)

        return q, unsubscribe

    # -- publishing --------------------------------------------------------

    def publish(self, event: Event) -> None:
        """Deliver ``event`` to every matching subscriber."""
        if self._closed:
            return
        with self._lock:
            handlers = [h for h, want in self._handlers if want is None or event.type in want]
            queues = [q for q, want in self._queues if want is None or event.type in want]

        for handler in handlers:
            try:
                handler(event)
            except Exception:  # noqa: BLE001 - a subscriber must never break the turn loop
                _log.exception(
                    "event handler failed",
                    extra={"context": {"event_type": str(event.type)}},
                )

        for q in queues:
            _offer(q, event)

    def emit(self, event_type: EventType, **payload: Any) -> Event:
        """Convenience wrapper: build an :class:`Event` and publish it."""
        event = Event(type=event_type, payload=payload)
        self.publish(event)
        return event

    def close(self) -> None:
        """Stop delivery and drop every subscriber."""
        with self._lock:
            self._closed = True
            self._handlers.clear()
            self._queues.clear()

    @property
    def subscriber_count(self) -> int:
        """Number of live subscriptions, callbacks plus queues."""
        with self._lock:
            return len(self._handlers) + len(self._queues)


def _offer(q: queue.Queue[Event], event: Event) -> None:
    """Put ``event`` on ``q``, discarding the oldest entry when full."""
    try:
        q.put_nowait(event)
    except queue.Full:
        with contextlib.suppress(queue.Empty):
            q.get_nowait()
        with contextlib.suppress(queue.Full):
            q.put_nowait(event)


class StateMachine:
    """Holds the current :class:`AssistantState` and announces every change.

    Transitions are not restricted. The orchestrator owns the legal sequence, and
    an error path must be able to jump to ``ERROR`` from anywhere.
    """

    def __init__(self, bus: EventBus, *, initial: AssistantState = AssistantState.IDLE) -> None:
        self._bus = bus
        self._state = initial
        self._lock = threading.RLock()
        self._changed_at = time.time()

    @property
    def state(self) -> AssistantState:
        """Current state."""
        with self._lock:
            return self._state

    @property
    def seconds_in_state(self) -> float:
        """How long the machine has been in the current state."""
        with self._lock:
            return time.time() - self._changed_at

    def set(self, state: AssistantState, **payload: Any) -> bool:
        """Move to ``state``. Returns False when already there.

        Re-entering the same state is a no-op so the HUD does not flicker.
        """
        with self._lock:
            if state is self._state:
                return False
            previous = self._state
            self._state = state
            self._changed_at = time.time()
        _log.debug(
            "state change",
            extra={"context": {"from": str(previous), "to": str(state), **payload}},
        )
        self._bus.emit(
            EventType.STATE_CHANGED,
            state=str(state),
            previous=str(previous),
            **payload,
        )
        return True

    @contextlib.contextmanager
    def during(self, state: AssistantState, revert_to: AssistantState) -> Iterator[None]:
        """Hold ``state`` for the duration of the block, then move to ``revert_to``.

        On an exception the machine goes to ``ERROR`` and the exception propagates.
        """
        self.set(state)
        try:
            yield
        except BaseException:
            self.set(AssistantState.ERROR)
            raise
        else:
            self.set(revert_to)
