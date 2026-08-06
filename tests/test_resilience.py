"""T-5.2 verification: supervision, health, and graceful shutdown."""

from __future__ import annotations

import threading
import time

import pytest

from jarvis.state import Event, EventBus, EventType
from jarvis.util.resilience import ShutdownCoordinator, Supervisor, WorkerState


def _wait_for(predicate: object, timeout: float = 5.0, interval: float = 0.02) -> bool:
    """Poll until ``predicate`` is true or the timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():  # type: ignore[operator]
            return True
        time.sleep(interval)
    return False


class TestSupervisorBasics:
    def test_runs_a_worker(self) -> None:
        ran = threading.Event()
        supervisor = Supervisor(poll_interval_s=0.05)
        supervisor.add("job", ran.set, restart=False)
        try:
            supervisor.start()
            assert ran.wait(3)
        finally:
            supervisor.stop()

    def test_duplicate_name_is_rejected(self) -> None:
        supervisor = Supervisor()
        supervisor.add("job", lambda: None)
        with pytest.raises(ValueError, match="already registered"):
            supervisor.add("job", lambda: None)

    def test_context_manager_starts_and_stops(self) -> None:
        ran = threading.Event()
        with Supervisor(poll_interval_s=0.05) as supervisor:
            supervisor.add("job", ran.set, restart=False)
            supervisor.start()
            assert ran.wait(3)

    def test_stop_is_idempotent(self) -> None:
        supervisor = Supervisor(poll_interval_s=0.05)
        supervisor.add("job", lambda: time.sleep(0.01), restart=False)
        supervisor.start()
        supervisor.stop()
        supervisor.stop()

    def test_worker_can_poll_for_a_stop_request(self) -> None:
        """A long-running worker needs a way to exit its loop cleanly."""
        supervisor = Supervisor(poll_interval_s=0.05)
        exited = threading.Event()

        def loop() -> None:
            while not supervisor.should_stop("looper"):
                time.sleep(0.01)
            exited.set()

        supervisor.add("looper", loop)
        supervisor.start()
        time.sleep(0.1)
        supervisor.stop()
        assert exited.is_set()


class TestRestart:
    def test_a_crashed_worker_is_restarted(self) -> None:
        """T-5.2: restart a dead audio or TTS thread."""
        attempts: list[int] = []
        started = threading.Event()

        def flaky() -> None:
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("boom")
            started.set()
            time.sleep(5)

        supervisor = Supervisor(poll_interval_s=0.02)
        supervisor.add("flaky", flaky, max_restarts=5)
        try:
            supervisor.start()
            assert started.wait(10), "the worker was never restarted successfully"
            assert len(attempts) >= 3
        finally:
            supervisor.stop(timeout=1)

    def test_restart_budget_is_finite(self) -> None:
        """Restarting forever would hide a permanent fault."""
        attempts: list[int] = []

        def always_fails() -> None:
            attempts.append(1)
            raise RuntimeError("permanent")

        supervisor = Supervisor(poll_interval_s=0.01)
        worker = supervisor.add("doomed", always_fails, max_restarts=2)
        try:
            supervisor.start()
            assert _wait_for(lambda: worker.state is WorkerState.FAILED, timeout=15)
            # One original run plus exactly max_restarts retries.
            assert len(attempts) <= 3
        finally:
            supervisor.stop(timeout=1)

    def test_a_worker_that_returns_cleanly_is_not_restarted(self) -> None:
        """Restarting a finished job would spin forever."""
        attempts: list[int] = []
        supervisor = Supervisor(poll_interval_s=0.02)
        supervisor.add("once", lambda: attempts.append(1), restart=False)
        try:
            supervisor.start()
            time.sleep(0.4)
            assert len(attempts) == 1
        finally:
            supervisor.stop(timeout=1)

    def test_failure_is_recorded(self) -> None:
        supervisor = Supervisor(poll_interval_s=0.01)

        def boom() -> None:
            raise ValueError("the sensor bus is wedged")

        worker = supervisor.add("boom", boom, max_restarts=0)
        try:
            supervisor.start()
            assert _wait_for(lambda: worker.last_error is not None, timeout=5)
            assert "the sensor bus is wedged" in (worker.last_error or "")
        finally:
            supervisor.stop(timeout=1)

    def test_crash_is_published_to_the_bus(self, bus: EventBus) -> None:
        seen: list[Event] = []
        bus.subscribe(seen.append, [EventType.ERROR])

        supervisor = Supervisor(bus, poll_interval_s=0.01)

        def boom() -> None:
            raise RuntimeError("gone")

        supervisor.add("boom", boom, max_restarts=0)
        try:
            supervisor.start()
            assert _wait_for(lambda: len(seen) > 0, timeout=5)
            assert "boom" in seen[0].payload.get("worker", "")
        finally:
            supervisor.stop(timeout=1)

    def test_a_speakable_is_attached_to_the_failure(self, bus: EventBus) -> None:
        """§5: fail loud in logs, fail soft in voice."""
        seen: list[Event] = []
        bus.subscribe(seen.append, [EventType.ERROR])
        supervisor = Supervisor(bus, poll_interval_s=0.01)

        def boom() -> None:
            raise RuntimeError("C:/Users/mann/secret path")

        supervisor.add("boom", boom, max_restarts=0)
        try:
            supervisor.start()
            assert _wait_for(lambda: len(seen) > 0, timeout=5)
            spoken = seen[0].payload.get("speakable", "")
            assert spoken
            assert "secret path" not in spoken
        finally:
            supervisor.stop(timeout=1)


class TestHealth:
    def test_healthy_when_everything_runs(self) -> None:
        supervisor = Supervisor(poll_interval_s=0.05)
        supervisor.add("looper", lambda: time.sleep(2))
        try:
            supervisor.start()
            time.sleep(0.1)
            report = supervisor.health()
            assert report.healthy is True
            assert report.failed == []
            assert report.workers["looper"]["alive"] is True
        finally:
            supervisor.stop(timeout=1)

    def test_unhealthy_when_a_worker_gives_up(self) -> None:
        supervisor = Supervisor(poll_interval_s=0.01)

        def boom() -> None:
            raise RuntimeError("dead")

        supervisor.add("dead", boom, max_restarts=0)
        try:
            supervisor.start()
            assert _wait_for(lambda: not supervisor.health().healthy, timeout=5)
            assert "dead" in supervisor.health().failed
        finally:
            supervisor.stop(timeout=1)

    def test_report_is_json_serialisable(self) -> None:
        import json

        supervisor = Supervisor()
        supervisor.add("job", lambda: None)
        json.dumps(supervisor.health().to_dict())

    def test_report_includes_restart_count(self) -> None:
        supervisor = Supervisor(poll_interval_s=0.01)
        attempts: list[int] = []

        def flaky() -> None:
            attempts.append(1)
            raise RuntimeError("again")

        supervisor.add("flaky", flaky, max_restarts=2)
        try:
            supervisor.start()
            assert _wait_for(lambda: supervisor.health().workers["flaky"]["restarts"] > 0, 10)
        finally:
            supervisor.stop(timeout=1)


class TestShutdownCoordinator:
    def test_runs_callbacks_in_reverse_order(self) -> None:
        """Resources must be released in the opposite order they were acquired."""
        order: list[str] = []
        coordinator = ShutdownCoordinator()
        coordinator.register("audio", lambda: order.append("audio"))
        coordinator.register("orchestrator", lambda: order.append("orchestrator"))
        coordinator.register("ui", lambda: order.append("ui"))
        coordinator.shutdown()
        assert order == ["ui", "orchestrator", "audio"]

    def test_runs_only_once(self) -> None:
        calls: list[int] = []
        coordinator = ShutdownCoordinator()
        coordinator.register("x", lambda: calls.append(1))
        coordinator.shutdown()
        coordinator.shutdown()
        assert len(calls) == 1

    def test_a_failing_step_does_not_block_the_rest(self) -> None:
        order: list[str] = []
        coordinator = ShutdownCoordinator()
        coordinator.register("first", lambda: order.append("first"))

        def explode() -> None:
            raise RuntimeError("close failed")

        coordinator.register("bad", explode)
        coordinator.register("last", lambda: order.append("last"))
        coordinator.shutdown()
        assert order == ["last", "first"]

    def test_is_shut_down_flag(self) -> None:
        coordinator = ShutdownCoordinator()
        assert coordinator.is_shut_down is False
        coordinator.shutdown()
        assert coordinator.is_shut_down is True

    def test_wait_returns_after_shutdown(self) -> None:
        coordinator = ShutdownCoordinator()
        threading.Timer(0.05, coordinator.shutdown).start()
        assert coordinator.wait(3) is True

    def test_wait_times_out_when_nothing_happens(self) -> None:
        assert ShutdownCoordinator().wait(0.05) is False

    def test_signal_handler_installation_is_safe(self) -> None:
        """Installing handlers off the main thread must not raise."""
        coordinator = ShutdownCoordinator()
        errors: list[BaseException] = []

        def install() -> None:
            try:
                coordinator.install_signal_handlers()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        thread = threading.Thread(target=install)
        thread.start()
        thread.join(2)
        assert errors == []

    def test_sigint_triggers_shutdown(self) -> None:
        """T-5.2: graceful shutdown on SIGINT."""
        import os
        import signal as signal_module

        coordinator = ShutdownCoordinator()
        ran: list[str] = []
        coordinator.register("cleanup", lambda: ran.append("cleanup"))
        coordinator.install_signal_handlers()
        try:
            os.kill(os.getpid(), signal_module.SIGINT)
            assert coordinator.wait(3) is True
            assert ran == ["cleanup"]
        finally:
            signal_module.signal(signal_module.SIGINT, signal_module.default_int_handler)
