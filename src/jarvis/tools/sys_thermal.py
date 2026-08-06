"""CPU temperatures and fan speeds, read through the elevated helper.

These are the two readings a non-elevated process genuinely cannot get on
Windows, so both go over the sidecar RPC (§6). When the helper is not running,
or the user declined the UAC prompt, the tools return a calm explanation rather
than failing, because declining elevation is a legitimate choice.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import Field

from jarvis.config import JarvisConfig, get_config
from jarvis.helper.rpc import HelperClient
from jarvis.tools.registry import ToolCategory, ToolInput, ToolOutput, tool
from jarvis.util.errors import HelperUnavailableError

__all__ = [
    "FanReading",
    "ThermalInput",
    "ThermalOutput",
    "fan_status",
    "reset_helper_client",
    "thermal_status",
]

_log = logging.getLogger(__name__)

_client: HelperClient | None = None


def _helper(config: JarvisConfig | None = None) -> HelperClient:
    """Process-wide helper client, created on first use."""
    global _client  # noqa: PLW0603 - one client owns the cache and the autostart flag
    if _client is None:
        _client = HelperClient(config or get_config())
    return _client


def set_helper_client(client: HelperClient | None) -> None:
    """Replace the shared client. Used by the orchestrator and by tests."""
    global _client  # noqa: PLW0603
    _client = client


def reset_helper_client() -> None:
    """Drop the shared client so the next call rebuilds it."""
    set_helper_client(None)


class ThermalInput(ToolInput):
    """Arguments for the temperature tool."""

    include_all_sensors: bool = Field(
        default=False,
        description="Include every temperature sensor, not just the CPU summary.",
    )


class SensorReading(ToolOutput):
    """One temperature sensor."""

    hardware: str = Field(description="Which component the sensor belongs to.")
    sensor: str = Field(description="Sensor name.")
    celsius: float = Field(description="Temperature in degrees Celsius.")


class ThermalOutput(ToolOutput):
    """CPU temperature, or an explanation of why it is unavailable."""

    available: bool = Field(
        description="False when the elevated helper is not running. The reason field says why."
    )
    reason: str | None = Field(default=None, description="Why the reading is unavailable.")
    cpu_celsius: float | None = Field(
        default=None, description="CPU package temperature in degrees Celsius."
    )
    cpu_core_max_celsius: float | None = Field(
        default=None, description="Hottest individual core in degrees Celsius."
    )
    thermal_state: str | None = Field(
        default=None, description="One of cool, warm, hot, critical."
    )
    sensor_count: int = Field(default=0, description="How many temperature sensors were found.")
    sensors: list[SensorReading] = Field(
        default_factory=list,
        description="Up to 5 individual sensors, only when all sensors were requested.",
    )


class FanInput(ToolInput):
    """Arguments for the fan tool."""


class FanReading(ToolOutput):
    """One fan."""

    hardware: str = Field(description="Which component the fan belongs to.")
    sensor: str = Field(description="Fan name, for example CPU Fan.")
    rpm: float = Field(description="Speed in revolutions per minute.")


class FanOutput(ToolOutput):
    """Fan speeds, or an explanation of why they are unavailable."""

    available: bool = Field(description="False when the elevated helper is not running.")
    reason: str | None = Field(default=None, description="Why the reading is unavailable.")
    fan_count: int = Field(default=0, description="How many fans were detected.")
    spinning_count: int = Field(default=0, description="How many are currently turning.")
    max_rpm: float | None = Field(
        default=None, description="Fastest fan in revolutions per minute."
    )
    fans: list[FanReading] = Field(default_factory=list, description="Up to 5 fans.")


def _describe_cpu_temperature(celsius: float) -> str:
    """Classify a CPU temperature for speech.

    Thresholds reflect what is normal for a desktop CPU: sustained load in the
    seventies is unremarkable, above ninety is worth mentioning.
    """
    if celsius < 45:
        return "cool"
    if celsius < 70:
        return "warm"
    if celsius < 90:
        return "hot"
    return "critical"


@tool(
    name="sys.thermal",
    description=(
        "CPU temperature and the hottest core, in degrees Celsius, read through the elevated "
        "helper. Use this for questions like how hot is my CPU, is my computer overheating, "
        "or what temperature is the processor. Requires the elevated helper to be running; "
        "if it is not, the result says so and you must say so too rather than guessing a "
        "number. Read-only."
    ),
    category=ToolCategory.SYSTEM,
    read_only=True,
)
def thermal_status(params: ThermalInput) -> ThermalOutput:
    """Read CPU temperatures via the helper.

    Args:
        params: Whether to include every sensor.

    Returns:
        Temperatures, or ``available=False`` with a reason.
    """
    try:
        payload = _helper().get_thermals()
    except HelperUnavailableError as exc:
        _log.info("thermal reading unavailable: %s", exc.message)
        return ThermalOutput(available=False, reason=exc.speakable)

    cpu = payload.get("cpu_celsius")
    sensors: list[SensorReading] = []
    if params.include_all_sensors:
        for entry in payload.get("sensors", [])[:5]:
            try:
                sensors.append(SensorReading(**entry))
            except (TypeError, ValueError):
                continue

    return ThermalOutput(
        available=True,
        cpu_celsius=cpu,
        cpu_core_max_celsius=payload.get("cpu_core_max_celsius"),
        thermal_state=_describe_cpu_temperature(float(cpu)) if cpu is not None else None,
        sensor_count=int(payload.get("sensor_count", 0)),
        sensors=sensors,
    )


@tool(
    name="sys.fans",
    description=(
        "Fan speeds in revolutions per minute, read through the elevated helper. "
        "Use this for questions like how fast are my fans spinning, are the fans running, "
        "or why is my computer loud. Requires the elevated helper; if it is not running the "
        "result says so and you must say so rather than guessing. Read-only."
    ),
    category=ToolCategory.SYSTEM,
    read_only=True,
)
def fan_status(params: FanInput) -> FanOutput:
    """Read fan speeds via the helper.

    Args:
        params: No arguments.

    Returns:
        Fan speeds, or ``available=False`` with a reason.
    """
    try:
        payload = _helper().get_fans()
    except HelperUnavailableError as exc:
        _log.info("fan reading unavailable: %s", exc.message)
        return FanOutput(available=False, reason=exc.speakable)

    fans: list[FanReading] = []
    for entry in payload.get("fans", [])[:5]:
        try:
            fans.append(FanReading(**entry))
        except (TypeError, ValueError):
            continue

    return FanOutput(
        available=True,
        fan_count=int(payload.get("fan_count", 0)),
        spinning_count=int(payload.get("spinning_count", 0)),
        max_rpm=payload.get("max_rpm"),
        fans=fans,
    )


def snapshot() -> dict[str, Any]:
    """Cheap sample for the HUD metrics stream. Never raises."""
    try:
        payload = _helper().get_thermals()
        return {"cpu_temp_c": payload.get("cpu_celsius")}
    except Exception:  # noqa: BLE001 - the HUD must never take down the loop
        return {"cpu_temp_c": None}
