"""Crash resilience: thread supervision, health, and graceful shutdown.

T-5.2. The turn loop runs across several threads, and a voice assistant that
silently loses its microphone thread is worse than one that crashes, because
the user has no way to tell. The supervisor restarts a dead worker with
backoff, and gives up loudly rather than restarting forever.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from types import FrameType
from typing import Any

from jarvis.state import EventBus, EventType

__all__ = [
    "HealthReport",
    "ShutdownCoordinator",
    "Supervisor",
    "WorkerState",
]

_log = logging.getLogger(__name__)

#: Backoff between restarts, in seconds, indexed by consecutive failure count.
_BACKOFF_S = (0.5, 1.0, 2.0, 5.0, 10.0)


class WorkerState(StrEnum):
    """Lifecycle of a supervised worker."""

    STOPPED = "stopped"
    RUNNING = "running"
    RESTARTING = "restarting"
    #: Exhausted its restart budget. Never restarted again without intervention.
    FAILED = "failed"
    #: Asked to stop but still running when the shutdown deadline passed. The
    #: thread is a daemon so it cannot hold the interpreter open, but it is
    #: still touching whatever it owned, so shutdown must not pretend otherwise.
    STALLED = "stalled"


@dataclass
class Worker:
    """One supervised thread."""

    name: str
    target: Callable[[], None]
    #: Restarting a worker that exited cleanly would spin on a finished job.
    restart: bool = True
    max_restarts: int = 5
    critical: bool = False
    thread: threading.Thread | None = None
    state: WorkerState = WorkerState.STOPPED
    restarts: int = 0
    last_error: str | None = None
    started_at: float | None = None
    _stop: threading.Event = field(default_factory=threading.Event)

    @property
    def alive(self) -> bool:
        """True while the worker thread is running."""
        return self.thread is not None and self.thread.is_alive()

    @property
    def uptime_s(self) -> float:
        """Seconds since this worker last started."""
        return 0.0 if self.started_at is None else time.monotonic() - self.started_at


@dataclass(frozen=True, slots=True)
class HealthReport:
    """A snapshot of supervised worker health."""

    healthy: bool
    workers: dict[str, dict[str, Any]]
    failed: list[str]
    #: Workers that ignored a stop request. Separate from ``failed`` because the
    #: remedy differs: a failed worker needs a fix, a stalled one needs a kill.
    stalled: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, suitable for a health endpoint."""
        return {
            "healthy": self.healthy,
            "workers": self.workers,
            "failed": self.failed,
            "stalled": self.stalled,
        }


class Supervisor:
    """Runs worker threads and restarts them when they die unexpectedly.

    Args:
        bus: Optional event bus. Failures are published so the HUD can show them.
        poll_interval_s: How often the monitor thread checks liveness.
    """

    def __init__(self, bus: EventBus | None = None, *, poll_interval_s: float = 0.5) -> None:
        self._bus = bus
        self._poll = poll_interval_s
        self._workers: dict[str, Worker] = {}
        self._lock = threading.RLock()
        self._monitor: threading.Thread | None = None
        self._stop = threading.Event()

    # -- registration ------------------------------------------------------

    def add(
        self,
        name: str,
        target: Callable[[], None],
        *,
        restart: bool = True,
        max_restarts: int = 5,
        critical: bool = False,
    ) -> Worker:
        """Register a worker.

        Args:
            name: Identifier used in logs and the health report.
            target: The thread body. Should return when asked to stop.
            restart: Whether to restart it when it dies.
            max_restarts: Restart budget before the worker is marked FAILED.
            critical: When True, exhausting the budget shuts the whole app down.
        """
        with self._lock:
            if name in self._workers:
                msg = f"worker {name!r} is already registered"
                raise ValueError(msg)
            worker = Worker(
                name=name,
                target=target,
                restart=restart,
                max_restarts=max_restarts,
                critical=critical,
            )
            self._workers[name] = worker
            return worker

    def remove(self, name: str) -> None:
        """Unregister a worker. Does not stop a running thread."""
        with self._lock:
            self._workers.pop(name, None)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start every registered worker plus the monitor thread."""
        self._stop.clear()
        with self._lock:
            for worker in self._workers.values():
                self._spawn(worker)
        if self._monitor is None or not self._monitor.is_alive():
            self._monitor = threading.Thread(
                target=self._watch, name="jarvis-supervisor", daemon=True
            )
            self._monitor.start()

    def _spawn(self, worker: Worker) -> None:
        """Start one worker's thread. Caller holds the lock."""
        worker._stop.clear()
        worker.started_at = time.monotonic()
        worker.state = WorkerState.RUNNING
        worker.thread = threading.Thread(
            target=self._run_worker, args=(worker,), name=f"jarvis-{worker.name}", daemon=True
        )
        worker.thread.start()
        _log.info("worker started", extra={"context": {"worker": worker.name}})

    def _run_worker(self, worker: Worker) -> None:
        """Thread body wrapper that records why a worker exited."""
        try:
            worker.target()
        except Exception as exc:  # noqa: BLE001 - the whole point is to catch this
            worker.last_error = f"{type(exc).__name__}: {exc}"
            _log.exception("worker crashed", extra={"context": {"worker": worker.name}})
            if self._bus is not None:
                self._bus.emit(
                    EventType.ERROR,
                    message=f"worker {worker.name} crashed: {worker.last_error}",
                    speakable="One of my background tasks failed.",
                    worker=worker.name,
                )
        else:
            _log.info("worker exited", extra={"context": {"worker": worker.name}})

    def _watch(self) -> None:
        """Monitor loop: restart dead workers with backoff."""
        while not self._stop.is_set():
            with self._lock:
                workers = list(self._workers.values())

            for worker in workers:
                if self._stop.is_set():
                    return
                if worker.state is not WorkerState.RUNNING or worker.alive:
                    continue

                # The thread is gone. Decide whether to bring it back.
                if not worker.restart or worker._stop.is_set():
                    worker.state = WorkerState.STOPPED
                    continue

                if worker.restarts >= worker.max_restarts:
                    worker.state = WorkerState.FAILED
                    _log.error(
                        "worker exhausted its restart budget",
                        extra={
                            "context": {
                                "worker": worker.name,
                                "restarts": worker.restarts,
                                "last_error": worker.last_error,
                            }
                        },
                    )
                    if self._bus is not None:
                        self._bus.emit(
                            EventType.ERROR,
                            message=f"worker {worker.name} failed permanently",
                            speakable="Part of me has stopped working and will not recover.",
                            worker=worker.name,
                        )
                    if worker.critical:
                        _log.critical(
                            "a critical worker failed, requesting shutdown",
                            extra={"context": {"worker": worker.name}},
                        )
                        self._stop.set()
                    continue

                delay = _BACKOFF_S[min(worker.restarts, len(_BACKOFF_S) - 1)]
                worker.state = WorkerState.RESTARTING
                worker.restarts += 1
                _log.warning(
                    "restarting worker",
                    extra={
                        "context": {
                            "worker": worker.name,
                            "attempt": worker.restarts,
                            "delay_s": delay,
                        }
                    },
                )
                # Backoff happens on the monitor thread, which is fine: it has
                # nothing else to do and the delay is bounded.
                if self._stop.wait(delay):
                    return
                with self._lock:
                    self._spawn(worker)

            if self._stop.wait(self._poll):
                return

    def stop(self, timeout: float = 5.0) -> None:
        """Ask every worker to stop and wait for them, then stop the monitor.

        Two states survive shutdown rather than being overwritten with STOPPED.
        A worker that exhausted its restart budget stays FAILED, because a
        permanent fault is still a permanent fault after a restart. A worker
        that is still running when the deadline passes becomes STALLED rather
        than being recorded as a clean stop it never reached: reporting a live
        thread as stopped is the one lie that hides exactly the fault the
        supervisor exists to surface.
        """
        self._stop.set()
        with self._lock:
            workers = list(self._workers.values())
        for worker in workers:
            worker._stop.set()
        deadline = time.monotonic() + timeout
        stalled: list[str] = []
        for worker in workers:
            if worker.thread is not None:
                worker.thread.join(max(0.0, deadline - time.monotonic()))
            if worker.alive:
                worker.state = WorkerState.STALLED
                stalled.append(worker.name)
            elif worker.state is not WorkerState.FAILED:
                worker.state = WorkerState.STOPPED

        if stalled:
            _log.error(
                "workers ignored the stop request",
                extra={"context": {"workers": stalled, "timeout_s": timeout}},
            )
            if self._bus is not None:
                self._bus.emit(
                    EventType.ERROR,
                    message=f"workers still running after shutdown: {', '.join(stalled)}",
                    speakable="Some of my background tasks would not shut down.",
                    workers=stalled,
                )

        if self._monitor is not None:
            self._monitor.join(max(0.0, deadline - time.monotonic()))
            self._monitor = None

    def should_stop(self, name: str) -> bool:
        """Whether the named worker has been asked to stop.

        Worker bodies poll this so they can exit their loop cleanly.
        """
        with self._lock:
            worker = self._workers.get(name)
        return worker is None or worker._stop.is_set() or self._stop.is_set()

    def __enter__(self) -> Supervisor:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # -- health ------------------------------------------------------------

    def health(self) -> HealthReport:
        """Current worker health, for the health check endpoint (T-5.2)."""
        with self._lock:
            workers = {
                worker.name: {
                    "state": str(worker.state),
                    "alive": worker.alive,
                    "restarts": worker.restarts,
                    "uptime_s": round(worker.uptime_s, 1),
                    "last_error": worker.last_error,
                    "critical": worker.critical,
                }
                for worker in self._workers.values()
            }
        failed = [name for name, row in workers.items() if row["state"] == WorkerState.FAILED]
        stalled = [name for name, row in workers.items() if row["state"] == WorkerState.STALLED]
        return HealthReport(
            healthy=not failed and not stalled,
            workers=workers,
            failed=failed,
            stalled=stalled,
        )


class ShutdownCoordinator:
    """Runs shutdown callbacks once, in reverse registration order.

    Reverse order matters: the audio stream is opened before the orchestrator
    and must be closed after it, or the orchestrator's last write goes to a
    closed device.
    """

    def __init__(self) -> None:
        self._callbacks: list[tuple[str, Callable[[], None]]] = []
        self._lock = threading.RLock()
        self._done = threading.Event()
        self._previous: dict[int, Any] = {}

    def register(self, name: str, callback: Callable[[], None]) -> None:
        """Add a shutdown step."""
        with self._lock:
            self._callbacks.append((name, callback))

    def install_signal_handlers(self) -> None:
        """Handle SIGINT and SIGTERM by running shutdown once.

        A second signal is left to the default handler, so an operator who is
        really in a hurry can still force the process down.
        """

        def handler(signum: int, _frame: FrameType | None) -> None:
            _log.info("shutdown signal received", extra={"context": {"signal": signum}})
            previous = self._previous.get(signum)
            if previous is not None:
                signal.signal(signum, previous)
            self.shutdown()

        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous[signum] = signal.getsignal(signum)
                signal.signal(signum, handler)
            except (ValueError, OSError):
                # Signal handlers can only be installed on the main thread.
                _log.debug("could not install a handler for signal %s", signum)

    def shutdown(self) -> None:
        """Run every callback once, in reverse order. Never raises."""
        if self._done.is_set():
            return
        self._done.set()
        with self._lock:
            callbacks = list(reversed(self._callbacks))

        for name, callback in callbacks:
            try:
                callback()
                _log.info("shutdown step complete", extra={"context": {"step": name}})
            except Exception:  # noqa: BLE001 - one bad step must not block the rest
                _log.exception("shutdown step failed", extra={"context": {"step": name}})

    @property
    def is_shut_down(self) -> bool:
        """Whether shutdown has already run."""
        return self._done.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until shutdown runs. Returns whether it did."""
        return self._done.wait(timeout)
