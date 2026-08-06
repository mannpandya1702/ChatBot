"""Network throughput, adapters, and connection counts.

Read-only. Nothing here changes an adapter, a route, or a firewall rule; §6
forbids network configuration outright and lists no network tool on the
mutating allowlist.
"""

from __future__ import annotations

import logging
import socket
import time
from typing import Any

import psutil
from pydantic import Field

from jarvis.tools.registry import ToolCategory, ToolInput, ToolOutput, tool
from jarvis.util.errors import ToolExecutionError

__all__ = ["AdapterInfo", "NetworkInput", "NetworkOutput", "network_status"]

_log = logging.getLogger(__name__)

_MAX_ADAPTERS = 5


class AdapterInfo(ToolOutput):
    """One network interface."""

    name: str = Field(description="Adapter name, for example Wi-Fi or Ethernet.")
    is_up: bool = Field(description="Whether the adapter is currently up.")
    ipv4: str | None = Field(default=None, description="IPv4 address, when assigned.")
    speed_mbps: int | None = Field(
        default=None, description="Link speed in megabits per second, when reported."
    )


class NetworkInput(ToolInput):
    """Arguments for the network status tool."""

    sample_seconds: float = Field(
        default=0.5,
        ge=0.1,
        le=3.0,
        description=(
            "How long to sample throughput. Throughput is a rate, so it needs a window; "
            "longer is steadier but slower to answer."
        ),
    )
    include_adapters: bool = Field(
        default=True, description="Include the active adapters and their addresses."
    )


class NetworkOutput(ToolOutput):
    """Current network state. Throughput in megabits per second."""

    download_mbps: float = Field(
        description="Current inbound throughput in megabits per second, sampled over the window."
    )
    upload_mbps: float = Field(
        description="Current outbound throughput in megabits per second, sampled over the window."
    )
    activity: str = Field(description="One of idle, light, moderate, heavy.")
    total_received_gb: float = Field(description="Total received since boot, in gigabytes.")
    total_sent_gb: float = Field(description="Total sent since boot, in gigabytes.")
    active_connections: int = Field(description="Number of established network connections.")
    primary_adapter: str | None = Field(
        default=None, description="The adapter carrying the default route, when identifiable."
    )
    adapters: list[AdapterInfo] = Field(
        default_factory=list, description="Up to 5 active adapters."
    )


def _describe_activity(mbps: float) -> str:
    """Turn a throughput figure into a word the assistant can say."""
    if mbps < 0.1:
        return "idle"
    if mbps < 5:
        return "light"
    if mbps < 50:
        return "moderate"
    return "heavy"


def _adapters() -> tuple[list[AdapterInfo], str | None]:
    """Enumerate up adapters and guess which one carries the default route."""
    rows: list[AdapterInfo] = []
    primary: str | None = None
    try:
        stats = psutil.net_if_stats()
        addresses = psutil.net_if_addrs()
    except (OSError, AttributeError):
        return rows, primary

    for name, stat in stats.items():
        if not stat.isup:
            continue
        ipv4: str | None = None
        for addr in addresses.get(name, []):
            if addr.family == socket.AF_INET:
                ipv4 = addr.address
                break
        # Loopback is up and has an address but is never the answer to
        # "what am I connected through".
        if ipv4 in (None, "127.0.0.1"):
            continue
        rows.append(
            AdapterInfo(
                name=name,
                is_up=True,
                ipv4=ipv4,
                speed_mbps=stat.speed or None,
            )
        )
        if primary is None:
            primary = name

    rows.sort(key=lambda row: (row.speed_mbps or 0), reverse=True)
    if rows:
        primary = rows[0].name
    return rows[:_MAX_ADAPTERS], primary


def _connection_count() -> int:
    """Count established connections, tolerating permission failures."""
    try:
        connections = psutil.net_connections(kind="inet")
        return sum(1 for conn in connections if conn.status == "ESTABLISHED")
    except (psutil.AccessDenied, PermissionError, OSError):
        # Enumerating every socket needs privileges we deliberately do not have.
        _log.debug("connection enumeration was denied")
        return 0


@tool(
    name="sys.network",
    description=(
        "Current network throughput, active adapter, and connection count. "
        "Use this for questions like how fast is my internet right now, is anything "
        "downloading, what network am I on, or how much data have I used. "
        "Throughput is in megabits per second sampled over a short window, totals are in "
        "gigabytes since boot. Read-only, it never changes network settings."
    ),
    category=ToolCategory.SYSTEM,
    read_only=True,
)
def network_status(params: NetworkInput) -> NetworkOutput:
    """Sample network throughput and describe the active connection.

    Args:
        params: Sampling window and whether to enumerate adapters.

    Returns:
        A speakable summary of network state.

    Raises:
        ToolExecutionError: Network counters could not be read.
    """
    try:
        first = psutil.net_io_counters()
        time.sleep(params.sample_seconds)
        second = psutil.net_io_counters()

        # Bytes to megabits: multiply by 8, divide by a million, divide by the window.
        scale = 1_000_000 * params.sample_seconds / 8
        download = round(max(0.0, (second.bytes_recv - first.bytes_recv) / scale), 2)
        upload = round(max(0.0, (second.bytes_sent - first.bytes_sent) / scale), 2)

        adapters: list[AdapterInfo] = []
        primary: str | None = None
        if params.include_adapters:
            adapters, primary = _adapters()

        return NetworkOutput(
            download_mbps=download,
            upload_mbps=upload,
            activity=_describe_activity(max(download, upload)),
            total_received_gb=round(second.bytes_recv / (1024**3), 2),
            total_sent_gb=round(second.bytes_sent / (1024**3), 2),
            active_connections=_connection_count(),
            primary_adapter=primary,
            adapters=adapters,
        )
    except Exception as exc:
        _log.exception("sys.network failed")
        raise ToolExecutionError(
            "sys.network",
            f"could not read network counters: {exc}",
            speakable="I could not read the network counters.",
        ) from exc


def snapshot(previous: dict[str, Any] | None = None) -> dict[str, Any]:
    """Cheap rate sample for the HUD, differencing against the previous call.

    The HUD polls continuously, so it computes rates by differencing rather than
    by sleeping. Pass the previous return value back in.
    """
    try:
        counters = psutil.net_io_counters()
        now = time.monotonic()
        result: dict[str, Any] = {
            "_recv": counters.bytes_recv,
            "_sent": counters.bytes_sent,
            "_at": now,
            "net_down_mbps": 0.0,
            "net_up_mbps": 0.0,
        }
        if previous and "_at" in previous:
            elapsed = now - float(previous["_at"])
            if elapsed > 0:
                scale = 1_000_000 * elapsed / 8
                result["net_down_mbps"] = round(
                    max(0.0, (counters.bytes_recv - float(previous["_recv"])) / scale), 2
                )
                result["net_up_mbps"] = round(
                    max(0.0, (counters.bytes_sent - float(previous["_sent"])) / scale), 2
                )
        return result
    except Exception:  # noqa: BLE001 - the HUD must never take down the loop
        _log.debug("network snapshot failed", exc_info=True)
        return {"net_down_mbps": 0.0, "net_up_mbps": 0.0}
