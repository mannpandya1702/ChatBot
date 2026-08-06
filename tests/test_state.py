"""Event bus and state machine."""

from __future__ import annotations

import queue
import threading

import pytest

from jarvis.state import AssistantState, Event, EventBus, EventType, StateMachine


class TestEventBus:
    def test_delivers_to_a_subscriber(self, bus: EventBus) -> None:
        seen: list[Event] = []
        bus.subscribe(seen.append)
        bus.emit(EventType.TRANSCRIPT, text="hello")
        assert len(seen) == 1
        assert seen[0].payload == {"text": "hello"}

    def test_type_filter(self, bus: EventBus) -> None:
        seen: list[Event] = []
        bus.subscribe(seen.append, [EventType.ERROR])
        bus.emit(EventType.TRANSCRIPT, text="ignored")
        bus.emit(EventType.ERROR, message="caught")
        assert [e.type for e in seen] == [EventType.ERROR]

    def test_unsubscribe(self, bus: EventBus) -> None:
        seen: list[Event] = []
        cancel = bus.subscribe(seen.append)
        bus.emit(EventType.TRANSCRIPT)
        cancel()
        bus.emit(EventType.TRANSCRIPT)
        assert len(seen) == 1

    def test_unsubscribe_twice_is_safe(self, bus: EventBus) -> None:
        cancel = bus.subscribe(lambda _e: None)
        cancel()
        cancel()

    def test_a_raising_handler_does_not_stop_delivery(self, bus: EventBus) -> None:
        """§5: nothing a subscriber does may break the turn loop."""
        seen: list[Event] = []

        def explode(_event: Event) -> None:
            raise RuntimeError("subscriber bug")

        bus.subscribe(explode)
        bus.subscribe(seen.append)
        bus.emit(EventType.TRANSCRIPT, text="still delivered")
        assert len(seen) == 1

    def test_queue_subscription(self, bus: EventBus) -> None:
        q, _cancel = bus.subscribe_queue([EventType.METRICS])
        bus.emit(EventType.METRICS, cpu=10.0)
        bus.emit(EventType.TRANSCRIPT, text="filtered out")
        assert q.get_nowait().payload == {"cpu": 10.0}
        with pytest.raises(queue.Empty):
            q.get_nowait()

    def test_full_queue_drops_oldest_rather_than_blocking(self, bus: EventBus) -> None:
        """The publisher may be the audio thread, so it must never block."""
        q, _cancel = bus.subscribe_queue(maxsize=2)
        for index in range(5):
            bus.emit(EventType.AUDIO_LEVEL, index=index)
        drained = [q.get_nowait().payload["index"] for _ in range(q.qsize())]
        assert drained == [3, 4]

    def test_close_stops_delivery(self, bus: EventBus) -> None:
        seen: list[Event] = []
        bus.subscribe(seen.append)
        bus.close()
        bus.emit(EventType.TRANSCRIPT)
        assert seen == []

    def test_subscriber_count(self, bus: EventBus) -> None:
        bus.subscribe(lambda _e: None)
        bus.subscribe_queue()
        assert bus.subscriber_count == 2

    def test_event_is_serialisable(self, bus: EventBus) -> None:
        event = bus.emit(EventType.STATE_CHANGED, state="idle")
        payload = event.to_dict()
        assert payload["type"] == "state_changed"
        assert payload["payload"]["state"] == "idle"
        assert payload["event_id"]

    def test_concurrent_publishing_is_safe(self, bus: EventBus) -> None:
        received: list[Event] = []
        lock = threading.Lock()

        def collect(event: Event) -> None:
            with lock:
                received.append(event)

        bus.subscribe(collect)
        threads = [
            threading.Thread(target=lambda: [bus.emit(EventType.AUDIO_LEVEL) for _ in range(50)])
            for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(received) == 200


class TestStateMachine:
    def test_starts_idle(self, bus: EventBus) -> None:
        assert StateMachine(bus).state is AssistantState.IDLE

    def test_transition_publishes(self, bus: EventBus) -> None:
        seen: list[Event] = []
        bus.subscribe(seen.append, [EventType.STATE_CHANGED])
        machine = StateMachine(bus)
        assert machine.set(AssistantState.LISTENING) is True
        assert machine.state is AssistantState.LISTENING
        assert seen[0].payload["state"] == "listening"
        assert seen[0].payload["previous"] == "idle"

    def test_repeat_transition_is_a_noop(self, bus: EventBus) -> None:
        seen: list[Event] = []
        bus.subscribe(seen.append, [EventType.STATE_CHANGED])
        machine = StateMachine(bus)
        machine.set(AssistantState.THINKING)
        assert machine.set(AssistantState.THINKING) is False
        assert len(seen) == 1

    def test_extra_payload_is_forwarded(self, bus: EventBus) -> None:
        seen: list[Event] = []
        bus.subscribe(seen.append, [EventType.STATE_CHANGED])
        StateMachine(bus).set(AssistantState.ERROR, reason="stt failed")
        assert seen[0].payload["reason"] == "stt failed"

    def test_during_reverts(self, bus: EventBus) -> None:
        machine = StateMachine(bus)
        with machine.during(AssistantState.THINKING, AssistantState.IDLE):
            assert machine.state is AssistantState.THINKING
        assert machine.state is AssistantState.IDLE

    def test_during_goes_to_error_on_exception(self, bus: EventBus) -> None:
        machine = StateMachine(bus)
        with (
            pytest.raises(RuntimeError),
            machine.during(AssistantState.THINKING, AssistantState.IDLE),
        ):
            raise RuntimeError("llm died")
        assert machine.state is AssistantState.ERROR

    def test_seconds_in_state_is_non_negative(self, bus: EventBus) -> None:
        assert StateMachine(bus).seconds_in_state >= 0

    def test_every_hud_state_exists(self) -> None:
        """§7 T-3.1 names exactly these five states."""
        assert {s.value for s in AssistantState} == {
            "idle",
            "listening",
            "thinking",
            "speaking",
            "error",
        }
