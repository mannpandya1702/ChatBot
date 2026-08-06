"""T-2.6 verification: sys.network."""

from __future__ import annotations

import json
import socket
from typing import Any

import pytest

from jarvis.tools import sys_network
from jarvis.tools.registry import registry
from jarvis.tools.sys_network import NetworkInput, network_status
from jarvis.util.errors import ToolExecutionError


class _Counters:
    def __init__(self, recv: int, sent: int) -> None:
        self.bytes_recv = recv
        self.bytes_sent = sent


class TestActivity:
    @pytest.mark.parametrize(
        ("mbps", "expected"),
        [(0.0, "idle"), (0.05, "idle"), (1.0, "light"), (20.0, "moderate"), (200.0, "heavy")],
    )
    def test_thresholds(self, mbps: float, expected: str) -> None:
        assert sys_network._describe_activity(mbps) == expected


class TestThroughputMath:
    def test_bytes_convert_to_megabits_correctly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """125000 bytes over 1 second is exactly 1 megabit per second."""
        samples = iter([_Counters(0, 0), _Counters(125_000, 250_000)])
        monkeypatch.setattr(sys_network.psutil, "net_io_counters", lambda: next(samples))
        monkeypatch.setattr(sys_network.time, "sleep", lambda _s: None)

        result = network_status(NetworkInput(sample_seconds=1.0, include_adapters=False))
        assert result.download_mbps == pytest.approx(1.0)
        assert result.upload_mbps == pytest.approx(2.0)

    def test_counter_reset_does_not_go_negative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An adapter reset can make the second sample smaller than the first."""
        samples = iter([_Counters(1_000_000, 1_000_000), _Counters(0, 0)])
        monkeypatch.setattr(sys_network.psutil, "net_io_counters", lambda: next(samples))
        monkeypatch.setattr(sys_network.time, "sleep", lambda _s: None)

        result = network_status(NetworkInput(sample_seconds=1.0, include_adapters=False))
        assert result.download_mbps == 0.0
        assert result.upload_mbps == 0.0

    def test_shorter_window_scales_correctly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        samples = iter([_Counters(0, 0), _Counters(125_000, 0)])
        monkeypatch.setattr(sys_network.psutil, "net_io_counters", lambda: next(samples))
        monkeypatch.setattr(sys_network.time, "sleep", lambda _s: None)

        result = network_status(NetworkInput(sample_seconds=0.5, include_adapters=False))
        assert result.download_mbps == pytest.approx(2.0)


class TestNetworkStatus:
    def test_returns_plausible_values(self) -> None:
        result = network_status(NetworkInput(sample_seconds=0.1))
        assert result.download_mbps >= 0
        assert result.upload_mbps >= 0
        assert result.total_received_gb >= 0
        assert result.active_connections >= 0

    def test_adapters_capped_at_five(self) -> None:
        assert len(network_status(NetworkInput(sample_seconds=0.1)).adapters) <= 5

    def test_adapters_are_opt_out(self) -> None:
        result = network_status(NetworkInput(sample_seconds=0.1, include_adapters=False))
        assert result.adapters == []
        assert result.primary_adapter is None

    def test_loopback_is_excluded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Loopback is up with an address but is never the active connection."""

        class Stat:
            def __init__(self, speed: int) -> None:
                self.isup = True
                self.speed = speed

        class Addr:
            def __init__(self, address: str) -> None:
                self.family = socket.AF_INET
                self.address = address

        monkeypatch.setattr(
            sys_network.psutil, "net_if_stats", lambda: {"lo": Stat(0), "eth0": Stat(1000)}
        )
        monkeypatch.setattr(
            sys_network.psutil,
            "net_if_addrs",
            lambda: {"lo": [Addr("127.0.0.1")], "eth0": [Addr("192.168.1.5")]},
        )
        adapters, primary = sys_network._adapters()
        assert [a.name for a in adapters] == ["eth0"]
        assert primary == "eth0"

    def test_down_adapters_are_excluded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Stat:
            def __init__(self, isup: bool) -> None:
                self.isup = isup
                self.speed = 100

        monkeypatch.setattr(sys_network.psutil, "net_if_stats", lambda: {"eth1": Stat(False)})
        monkeypatch.setattr(sys_network.psutil, "net_if_addrs", lambda: {"eth1": []})
        adapters, _primary = sys_network._adapters()
        assert adapters == []

    def test_denied_connection_enumeration_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Listing every socket needs privileges the main process must not have."""

        def denied(kind: str = "inet") -> Any:
            raise sys_network.psutil.AccessDenied()

        monkeypatch.setattr(sys_network.psutil, "net_connections", denied)
        assert sys_network._connection_count() == 0

    def test_failure_becomes_tool_execution_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom() -> Any:
            raise RuntimeError("no counters")

        monkeypatch.setattr(sys_network.psutil, "net_io_counters", boom)
        with pytest.raises(ToolExecutionError) as excinfo:
            network_status(NetworkInput(sample_seconds=0.1))
        assert excinfo.value.speakable == "I could not read the network counters."


class TestRegistration:
    def test_registered_and_read_only(self) -> None:
        spec = registry.get("sys.network")
        assert spec is not None
        assert spec.read_only is True

    def test_description_names_the_units(self) -> None:
        spec = registry.get("sys.network")
        assert spec is not None
        assert "megabits per second" in spec.description.lower()

    def test_dispatch_and_serialise(self) -> None:
        result = registry.dispatch("sys.network", {"sample_seconds": 0.1})
        assert result.ok is True
        json.dumps(result.data)


class TestSnapshot:
    def test_first_call_has_zero_rate(self) -> None:
        payload = sys_network.snapshot()
        assert payload["net_down_mbps"] == 0.0

    def test_second_call_computes_a_rate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        samples = iter([_Counters(0, 0), _Counters(125_000, 0)])
        clock = iter([100.0, 101.0])
        monkeypatch.setattr(sys_network.psutil, "net_io_counters", lambda: next(samples))
        monkeypatch.setattr(sys_network.time, "monotonic", lambda: next(clock))

        first = sys_network.snapshot()
        second = sys_network.snapshot(first)
        assert second["net_down_mbps"] == pytest.approx(1.0)

    def test_never_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom() -> Any:
            raise RuntimeError("gone")

        monkeypatch.setattr(sys_network.psutil, "net_io_counters", boom)
        assert sys_network.snapshot()["net_down_mbps"] == 0.0
