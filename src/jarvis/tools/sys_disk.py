"""Disk usage, throughput, and SMART health.

CLAUDE.md T-2.4 carries an explicit prohibition: never call ``Win32_Product``.
Querying that WMI class triggers a consistency check on every installed MSI,
which can take minutes and can silently start repair operations. Nothing here
touches it. Volume enumeration goes through psutil, and SMART data through
``smartctl`` only when it is already installed.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from typing import Any

import psutil
from pydantic import Field

from jarvis.tools.registry import ToolCategory, ToolInput, ToolOutput, tool
from jarvis.util.errors import ToolExecutionError

__all__ = ["DiskInput", "DiskOutput", "VolumeUsage", "disk_status", "smart_health"]

_log = logging.getLogger(__name__)

_MAX_VOLUMES = 5
_SMARTCTL_TIMEOUT_S = 10.0


class VolumeUsage(ToolOutput):
    """One mounted volume."""

    mount: str = Field(description="Drive letter on Windows, for example C:, or mount point.")
    filesystem: str = Field(description="Filesystem type, for example NTFS.")
    total_gb: float = Field(description="Capacity in gigabytes.")
    used_gb: float = Field(description="Used space in gigabytes.")
    free_gb: float = Field(description="Free space in gigabytes.")
    percent_used: float = Field(description="Share of capacity used, 0 to 100.")
    status: str = Field(description="One of healthy, filling, low, critical.")


class DiskInput(ToolInput):
    """Arguments for the disk status tool."""

    include_throughput: bool = Field(
        default=False,
        description=(
            "Sample read and write throughput over a short interval. Only ask for this when "
            "the user wants current disk activity, since it adds a sampling delay."
        ),
    )
    include_smart: bool = Field(
        default=False,
        description=(
            "Include SMART drive health. Requires smartctl to be installed; the result says "
            "so when it is not available."
        ),
    )


class DiskOutput(ToolOutput):
    """Storage state. Sizes in gigabytes, throughput in megabytes per second."""

    volumes: list[VolumeUsage] = Field(
        default_factory=list,
        description="Up to 5 volumes, fullest first, so the one at risk is mentioned first.",
    )
    total_free_gb: float = Field(description="Combined free space across all volumes.")
    read_mb_per_s: float | None = Field(
        default=None, description="Read throughput in megabytes per second, when sampled."
    )
    write_mb_per_s: float | None = Field(
        default=None, description="Write throughput in megabytes per second, when sampled."
    )
    smart_status: str | None = Field(
        default=None,
        description=(
            "Overall SMART health when available: healthy, failing, or unavailable with a "
            "reason."
        ),
    )


def _gb(value: float) -> float:
    """Bytes to gigabytes."""
    return round(value / (1024**3), 2)


def _describe_volume(percent: float, free_gb: float) -> str:
    """Classify a volume by how close it is to full.

    Uses both a percentage and an absolute floor: 5 percent free on a 4 TB disk
    is plenty, while 5 percent on a 128 GB disk is not.
    """
    if percent >= 95 or free_gb < 5:
        return "critical"
    if percent >= 90 or free_gb < 15:
        return "low"
    if percent >= 75:
        return "filling"
    return "healthy"


def _volumes() -> list[VolumeUsage]:
    """Enumerate mounted volumes, skipping ones that cannot be read."""
    rows: list[VolumeUsage] = []
    for part in psutil.disk_partitions(all=False):
        # Optical drives and unmounted card readers raise on Windows.
        if "cdrom" in part.opts or part.fstype == "":
            continue
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except (PermissionError, OSError):
            continue
        rows.append(
            VolumeUsage(
                mount=part.device if part.device else part.mountpoint,
                filesystem=part.fstype or "unknown",
                total_gb=_gb(usage.total),
                used_gb=_gb(usage.used),
                free_gb=_gb(usage.free),
                percent_used=round(float(usage.percent), 1),
                status=_describe_volume(float(usage.percent), _gb(usage.free)),
            )
        )
    rows.sort(key=lambda row: row.percent_used, reverse=True)
    return rows[:_MAX_VOLUMES]


def _throughput(interval_s: float) -> tuple[float | None, float | None]:
    """Sample disk read and write rates in megabytes per second."""
    try:
        first = psutil.disk_io_counters()
        if first is None:
            return None, None
        time.sleep(interval_s)
        second = psutil.disk_io_counters()
        if second is None:
            return None, None
        scale = (1024**2) * interval_s
        return (
            round((second.read_bytes - first.read_bytes) / scale, 2),
            round((second.write_bytes - first.write_bytes) / scale, 2),
        )
    except (OSError, AttributeError):
        _log.debug("disk io counters are unavailable", exc_info=True)
        return None, None


def smart_health() -> str:
    """Overall SMART status via smartctl, degrading gracefully when absent.

    Returns a short phrase rather than raising, because a missing smartctl is
    an ordinary condition and not a tool failure.
    """
    exe = shutil.which("smartctl")
    if exe is None:
        return "unavailable, smartctl is not installed"
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell, no LLM-authored input
            [exe, "--scan"],
            capture_output=True,
            text=True,
            timeout=_SMARTCTL_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _log.debug("smartctl scan failed: %s", exc)
        return "unavailable, smartctl could not enumerate drives"

    devices = [line.split()[0] for line in result.stdout.splitlines() if line.strip()]
    if not devices:
        return "unavailable, no drives reported SMART data"

    failing: list[str] = []
    for device in devices[:_MAX_VOLUMES]:
        try:
            health = subprocess.run(  # noqa: S603 - fixed argv, device from smartctl itself
                [exe, "-H", device],
                capture_output=True,
                text=True,
                timeout=_SMARTCTL_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        text = health.stdout.upper()
        if "FAILED" in text or "FAILING" in text:
            failing.append(device)

    if failing:
        return f"failing on {', '.join(failing)}"
    return "healthy"


@tool(
    name="sys.disk",
    description=(
        "Free space and usage for every drive, optionally with current read and write "
        "throughput and SMART drive health. Use this for questions like how much disk space "
        "is left, is my drive full, how much free space on C, or is my disk healthy. "
        "Sizes are in gigabytes, throughput in megabytes per second. Volumes are returned "
        "fullest first. Read-only."
    ),
    category=ToolCategory.SYSTEM,
    read_only=True,
)
def disk_status(params: DiskInput) -> DiskOutput:
    """Read storage usage and health.

    Args:
        params: Whether to sample throughput and query SMART.

    Returns:
        A speakable summary of storage state.

    Raises:
        ToolExecutionError: Volume enumeration failed entirely.
    """
    try:
        volumes = _volumes()
        read_rate: float | None = None
        write_rate: float | None = None
        if params.include_throughput:
            read_rate, write_rate = _throughput(0.5)

        return DiskOutput(
            volumes=volumes,
            total_free_gb=round(sum(v.free_gb for v in volumes), 2),
            read_mb_per_s=read_rate,
            write_mb_per_s=write_rate,
            smart_status=smart_health() if params.include_smart else None,
        )
    except Exception as exc:
        _log.exception("sys.disk failed")
        raise ToolExecutionError(
            "sys.disk",
            f"could not enumerate volumes: {exc}",
            speakable="I could not read the drive information.",
        ) from exc


def snapshot() -> dict[str, Any]:
    """Cheap sample for the HUD metrics stream, primary volume only."""
    try:
        volumes = _volumes()
        if not volumes:
            return {"disk_percent": 0.0, "disk_free_gb": 0.0}
        primary = volumes[0]
        return {"disk_percent": primary.percent_used, "disk_free_gb": primary.free_gb}
    except Exception:  # noqa: BLE001 - the HUD must never take down the loop
        _log.debug("disk snapshot failed", exc_info=True)
        return {"disk_percent": 0.0, "disk_free_gb": 0.0}
