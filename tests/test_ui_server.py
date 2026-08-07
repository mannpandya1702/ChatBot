"""T-3.1 verification: the HUD WebSocket broadcaster."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import websockets

from jarvis.config import JarvisConfig, load_config
from jarvis.state import AssistantState, Event, EventBus, EventType
from jarvis.ui.server import HudState, UiServer


class TestHudState:
    def test_defaults_are_idle(self) -> None:
        assert HudState().state == AssistantState.IDLE.value

    def test_frame_has_every_required_field(self) -> None:
        """T-3.1 names state, audio_level, transcript, response, metrics."""
        frame = HudState().to_frame()
        for key in ("state", "audio_level", "transcript", "response", "metrics"):
            assert key in frame

    def test_frame_is_json_serialisable(self) -> None:
        state = HudState()
        state.history.append("you: hello")
        json.dumps(state.to_frame())

    def test_state_change_is_applied(self) -> None:
        state = HudState()
        assert state.apply(Event(EventType.STATE_CHANGED, {"state": "listening"})) is True
        assert state.state == "listening"

    def test_repeat_state_is_not_a_change(self) -> None:
        """A redundant frame would make the HUD flicker for no reason."""
        state = HudState()
        state.apply(Event(EventType.STATE_CHANGED, {"state": "listening"}))
        assert state.apply(Event(EventType.STATE_CHANGED, {"state": "listening"})) is False

    def test_audio_level_jitter_is_ignored(self) -> None:
        state = HudState()
        state.apply(Event(EventType.AUDIO_LEVEL, {"level": 0.5}))
        assert state.apply(Event(EventType.AUDIO_LEVEL, {"level": 0.5001})) is False
        assert state.apply(Event(EventType.AUDIO_LEVEL, {"level": 0.9})) is True

    def test_transcript_clears_partial_and_response(self) -> None:
        state = HudState()
        state.apply(Event(EventType.PARTIAL_TRANSCRIPT, {"text": "what is the"}))
        state.apply(Event(EventType.RESPONSE_CHUNK, {"text": "old answer"}))
        state.apply(Event(EventType.TRANSCRIPT, {"text": "what is the time"}))
        assert state.partial_transcript == ""
        assert state.response == ""
        assert state.transcript == "what is the time"

    def test_response_chunks_accumulate(self) -> None:
        state = HudState()
        for chunk in ("It is ", "half past ", "four."):
            state.apply(Event(EventType.RESPONSE_CHUNK, {"text": chunk}))
        assert state.response == "It is half past four."

    def test_history_records_both_sides(self) -> None:
        state = HudState()
        state.apply(Event(EventType.TRANSCRIPT, {"text": "hello"}))
        state.apply(Event(EventType.RESPONSE, {"text": "good evening"}))
        assert list(state.history) == ["you: hello", "jarvis: good evening"]

    def test_history_is_bounded(self) -> None:
        """An unbounded transcript log would grow without limit."""
        state = HudState()
        for index in range(100):
            state.apply(Event(EventType.TRANSCRIPT, {"text": f"line {index}"}))
        assert len(state.history) <= 20

    def test_tool_call_and_result(self) -> None:
        state = HudState()
        state.apply(Event(EventType.TOOL_CALL, {"name": "sys.cpu"}))
        assert state.tool == "sys.cpu"
        state.apply(Event(EventType.TOOL_RESULT, {}))
        assert state.tool is None

    def test_confirmation_round_trip(self) -> None:
        state = HudState()
        state.apply(Event(EventType.CONFIRMATION_REQUIRED, {"prompt": "Shall I go ahead?"}))
        assert state.confirmation == "Shall I go ahead?"
        state.apply(Event(EventType.CONFIRMATION_RESOLVED, {}))
        assert state.confirmation is None

    def test_metrics_merge_rather_than_replace(self) -> None:
        state = HudState()
        state.apply(Event(EventType.METRICS, {"cpu_percent": 40.0}))
        state.apply(Event(EventType.METRICS, {"memory_percent": 60.0}))
        assert state.metrics == {"cpu_percent": 40.0, "memory_percent": 60.0}

    def test_error_is_recorded_and_cleared_on_recovery(self) -> None:
        state = HudState()
        state.apply(Event(EventType.ERROR, {"speakable": "I did not catch that."}))
        assert state.error == "I did not catch that."
        state.apply(Event(EventType.STATE_CHANGED, {"state": "error"}))
        assert state.error == "I did not catch that."
        state.apply(Event(EventType.STATE_CHANGED, {"state": "idle"}))
        assert state.error is None

    def test_unknown_event_is_not_a_change(self) -> None:
        assert HudState().apply(Event(EventType.SHUTDOWN, {})) is False


@pytest.fixture
def ui_config(tmp_path: Any) -> JarvisConfig:
    """Config bound to an ephemeral port so tests never collide."""
    return load_config(
        tmp_path / "absent.yaml", ui={"port": 0, "host": "127.0.0.1", "broadcast_hz": 60.0}
    )


class TestUiServer:
    async def _recv(self, ws: Any, wait_s: float = 3.0) -> dict[str, Any]:
        raw = await asyncio.wait_for(ws.recv(), timeout=wait_s)
        return dict(json.loads(raw))

    async def test_starts_and_stops(self, ui_config: JarvisConfig, bus: EventBus) -> None:
        server = UiServer(ui_config, bus)
        server.start()
        assert server.is_running is True
        assert server.port > 0
        server.stop()
        assert server.is_running is False

    async def test_client_gets_current_state_on_connect(
        self, ui_config: JarvisConfig, bus: EventBus
    ) -> None:
        """A HUD launched mid-session must not wait for the next change."""
        with UiServer(ui_config, bus) as server:
            server.state.state = "thinking"
            async with websockets.connect(f"ws://127.0.0.1:{server.port}") as ws:
                frame = await self._recv(ws)
                assert frame["state"] == "thinking"

    async def test_events_reach_the_client(
        self, ui_config: JarvisConfig, bus: EventBus
    ) -> None:
        with UiServer(ui_config, bus) as server:
            async with websockets.connect(f"ws://127.0.0.1:{server.port}") as ws:
                await self._recv(ws)  # initial frame
                bus.emit(EventType.STATE_CHANGED, state="listening")

                for _ in range(30):
                    frame = await self._recv(ws)
                    if frame["state"] == "listening":
                        return
                pytest.fail("the state change never arrived")

    async def test_transcript_and_response_reach_the_client(
        self, ui_config: JarvisConfig, bus: EventBus
    ) -> None:
        with UiServer(ui_config, bus) as server:
            async with websockets.connect(f"ws://127.0.0.1:{server.port}") as ws:
                await self._recv(ws)
                bus.emit(EventType.TRANSCRIPT, text="what time is it")
                bus.emit(EventType.RESPONSE, text="Half past four.")

                for _ in range(30):
                    frame = await self._recv(ws)
                    if frame["response"] == "Half past four.":
                        assert frame["transcript"] == "what time is it"
                        return
                pytest.fail("the transcript never arrived")

    async def test_metrics_reach_the_client(
        self, ui_config: JarvisConfig, bus: EventBus
    ) -> None:
        with UiServer(ui_config, bus) as server:
            async with websockets.connect(f"ws://127.0.0.1:{server.port}") as ws:
                await self._recv(ws)
                bus.emit(EventType.METRICS, cpu_percent=42.0)

                for _ in range(30):
                    frame = await self._recv(ws)
                    if frame["metrics"].get("cpu_percent") == 42.0:
                        return
                pytest.fail("metrics never arrived")

    async def test_two_clients_both_receive(
        self, ui_config: JarvisConfig, bus: EventBus
    ) -> None:
        with UiServer(ui_config, bus) as server:
            url = f"ws://127.0.0.1:{server.port}"
            async with websockets.connect(url) as first, websockets.connect(url) as second:
                await self._recv(first)
                await self._recv(second)
                bus.emit(EventType.STATE_CHANGED, state="speaking")

                async def wait_for(ws: Any) -> bool:
                    for _ in range(30):
                        if (await self._recv(ws))["state"] == "speaking":
                            return True
                    return False

                assert await wait_for(first)
                assert await wait_for(second)

    async def test_client_count_tracks_connections(
        self, ui_config: JarvisConfig, bus: EventBus
    ) -> None:
        with UiServer(ui_config, bus) as server:
            assert server.client_count == 0
            async with websockets.connect(f"ws://127.0.0.1:{server.port}"):
                for _ in range(50):
                    if server.client_count == 1:
                        break
                    await asyncio.sleep(0.02)
                assert server.client_count == 1

    async def test_inbound_messages_are_ignored(
        self, ui_config: JarvisConfig, bus: EventBus
    ) -> None:
        """The HUD is a display. Nothing it sends may change behaviour."""
        with UiServer(ui_config, bus) as server:
            async with websockets.connect(f"ws://127.0.0.1:{server.port}") as ws:
                await self._recv(ws)
                await ws.send(json.dumps({"type": "shell", "command": "shutdown /s"}))
                await asyncio.sleep(0.1)
                assert server.is_running is True

    async def test_publisher_never_blocks_without_a_client(
        self, ui_config: JarvisConfig, bus: EventBus
    ) -> None:
        """§5: the audio thread must never stall on the HUD."""
        with UiServer(ui_config, bus):
            for index in range(2000):
                bus.emit(EventType.AUDIO_LEVEL, level=index % 100 / 100)

    async def test_server_survives_an_abrupt_disconnect(
        self, ui_config: JarvisConfig, bus: EventBus
    ) -> None:
        with UiServer(ui_config, bus) as server:
            ws = await websockets.connect(f"ws://127.0.0.1:{server.port}")
            await self._recv(ws)
            await ws.close()
            await asyncio.sleep(0.15)
            bus.emit(EventType.STATE_CHANGED, state="idle")
            await asyncio.sleep(0.1)
            assert server.is_running is True

    async def test_reconnect_after_client_drop(
        self, ui_config: JarvisConfig, bus: EventBus
    ) -> None:
        """T-3.5 requires reconnect-on-drop to work against a live server."""
        with UiServer(ui_config, bus) as server:
            url = f"ws://127.0.0.1:{server.port}"
            async with websockets.connect(url) as first:
                await self._recv(first)
            async with websockets.connect(url) as second:
                assert (await self._recv(second))["type"] == "state"

    async def test_a_bind_failure_is_raised_not_swallowed(
        self, ui_config: JarvisConfig, bus: EventBus
    ) -> None:
        """_run sets the ready flag on failure so start() cannot hang, which
        means the flag alone does not mean success. A HUD that could not bind
        must not be reported as running."""
        occupied = UiServer(ui_config, bus)
        occupied.start()
        try:
            from jarvis.config import load_config

            clashing = load_config(
                Path(ui_config.paths.log_dir).parent / "absent.yaml",
                ui={"port": occupied.port, "host": "127.0.0.1"},
            )
            second = UiServer(clashing, bus)
            with pytest.raises(OSError):
                second.start()
            assert second.is_running is False
        finally:
            occupied.stop()

    async def test_a_failed_start_does_not_leak_a_subscription(
        self, ui_config: JarvisConfig, bus: EventBus
    ) -> None:
        """Each failed attempt used to leave a queue behind that fills forever."""
        from jarvis.config import load_config

        occupied = UiServer(ui_config, bus)
        occupied.start()
        try:
            clashing = load_config(
                Path(ui_config.paths.log_dir).parent / "absent.yaml",
                ui={"port": occupied.port, "host": "127.0.0.1"},
            )
            before = bus.subscriber_count
            for _ in range(3):
                with pytest.raises(OSError):
                    UiServer(clashing, bus).start()
            assert bus.subscriber_count <= before + 1
        finally:
            occupied.stop()

    async def test_double_start_is_safe(self, ui_config: JarvisConfig, bus: EventBus) -> None:
        server = UiServer(ui_config, bus)
        server.start()
        server.start()
        server.stop()

    async def test_stop_without_start_is_safe(
        self, ui_config: JarvisConfig, bus: EventBus
    ) -> None:
        UiServer(ui_config, bus).stop()

    async def test_a_bad_event_does_not_kill_the_pump(
        self, ui_config: JarvisConfig, bus: EventBus
    ) -> None:
        with UiServer(ui_config, bus) as server:
            bus.emit(EventType.METRICS, **{"weird": object()})
            bus.emit(EventType.STATE_CHANGED, state="listening")
            await asyncio.sleep(0.2)
            assert server.is_running is True


class TestOnlyTheHudMayConnect:
    """A WebSocket handshake is exempt from the same origin policy.

    There is no preflight and no CORS, so binding to 127.0.0.1 keeps nothing out
    of a browser: any page the user has open, down to an ad in an iframe, can
    open ws://127.0.0.1:8765. What this socket sends is the live microphone
    transcript, the reply, the rolling history, and the confirmation prompts for
    mutating actions. Unrestricted, it is a transcript exfiltration channel that
    is on by default, which breaks §0.1 more thoroughly than a cloud call would.

    The whole suite passed with it open, because every test connected from the
    same process and none of them sent an Origin header.
    """

    async def _connect(self, port: int, origin: str | None) -> dict[str, Any]:
        extra = {"additional_headers": {"Origin": origin}} if origin is not None else {}
        async with websockets.connect(f"ws://127.0.0.1:{port}", **extra) as ws:
            return dict(json.loads(await asyncio.wait_for(ws.recv(), timeout=3.0)))

    @pytest.mark.parametrize(
        "origin",
        [
            "https://evil.example.com",
            "http://evil.example.com",
            "http://localhost:3000",
            "null",
            "file://",
        ],
    )
    async def test_a_web_page_cannot_read_the_transcript(
        self, ui_config: JarvisConfig, bus: EventBus, origin: str
    ) -> None:
        with UiServer(ui_config, bus) as server:
            server.state.transcript = "my passphrase is hunter2"
            with pytest.raises(websockets.exceptions.InvalidStatus):
                await self._connect(server.port, origin)

    @pytest.mark.parametrize(
        "origin", ["tauri://localhost", "http://tauri.localhost", "https://tauri.localhost"]
    )
    async def test_the_tauri_shell_still_connects(
        self, ui_config: JarvisConfig, bus: EventBus, origin: str
    ) -> None:
        with UiServer(ui_config, bus) as server:
            server.state.state = "listening"
            assert (await self._connect(server.port, origin))["state"] == "listening"

    async def test_a_client_with_no_origin_still_connects(
        self, ui_config: JarvisConfig, bus: EventBus
    ) -> None:
        """Every non browser client sends none, and a browser always sends one."""
        with UiServer(ui_config, bus) as server:
            server.state.state = "idle"
            assert (await self._connect(server.port, None))["state"] == "idle"

    async def test_originless_clients_can_be_refused_too(
        self, tmp_path: Any, bus: EventBus
    ) -> None:
        config = load_config(
            tmp_path / "absent.yaml",
            ui={"port": 0, "host": "127.0.0.1", "allow_originless": False},
        )
        with UiServer(config, bus) as server, pytest.raises(websockets.exceptions.InvalidStatus):
            await self._connect(server.port, None)

    async def test_a_star_disables_the_check_loudly(
        self, tmp_path: Any, bus: EventBus, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An escape hatch has to exist, and has to say what it costs."""
        import logging

        config = load_config(
            tmp_path / "absent.yaml",
            ui={"port": 0, "host": "127.0.0.1", "allowed_origins": ["*"]},
        )
        with caplog.at_level(logging.WARNING), UiServer(config, bus) as server:
            frame = await self._connect(server.port, "https://evil.example.com")
        assert frame["ts"] > 0
        assert any("allowed_origins" in record.message for record in caplog.records)

    def test_the_default_allowlist_has_no_wildcard(self) -> None:
        assert "*" not in JarvisConfig().ui.allowed_origins

    def test_the_allowlist_accepts_a_comma_separated_string(self, tmp_path: Any) -> None:
        """YAML and environment variables both hand it over as one string."""
        config = load_config(
            tmp_path / "absent.yaml", ui={"allowed_origins": "tauri://localhost, http://a.b"}
        )
        assert config.ui.allowed_origins == ("tauri://localhost", "http://a.b")
