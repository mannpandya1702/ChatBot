"""LibreHardwareMonitor sensor access, and the pending Windows Update query.

This module runs inside the elevated sidecar only. CPU package temperatures and
fan tachometers are not readable from a non-elevated process on Windows, which
is the entire reason the sidecar exists (§6).

LibreHardwareMonitorLib.dll is MPL-2.0, which §1 permits. It is loaded through
pythonnet at runtime rather than linked, and it lives in ``vendor/``.
"""

from __future__ import annotations

import logging
import re
import subprocess
from typing import Any

from jarvis.util.errors import JarvisError

__all__ = [
    "LibreHardwareMonitor",
    "get_pending_updates",
    "verify_assembly",
]

_log = logging.getLogger(__name__)

_UPDATE_TIMEOUT_S = 90.0
_MAX_UPDATE_NAMES = 5

#: SHA-256 of the LibreHardwareMonitorLib.dll this repo vendors, matching
#: ``vendor/README.md`` and what ``scripts/fetch_vendor.ps1`` verifies on
#: download. Checked again at load time, because download time is the wrong
#: time: this process is elevated, ``clr.AddReference`` executes whatever it is
#: pointed at, and the file has been sitting on disk since setup. Verifying the
#: bytes is what makes the path a convenience rather than the whole trust
#: boundary.
_VENDORED_DLL_SHA256 = "a0f2728f1734c236a9d02d9e25a88bc4f8cb7bd1faff1770726beb7af06bf8dc"

#: Set to skip the hash check when deliberately running a different build of the
#: library. Off by default, and refusing to load is the safe failure.
_ALLOW_UNVERIFIED = "JARVIS_ALLOW_UNVERIFIED_LHM"

#: Sensor names LibreHardwareMonitor uses for the package temperature, in
#: descending order of preference.
_PACKAGE_HINTS = ("package", "cpu package", "tctl", "tdie", "core (tctl/tdie)")


def verify_assembly(dll: Any, *, explicit: bool = False) -> None:
    """Refuse to load a native assembly whose bytes are not the vendored ones.

    ``clr.AddReference`` runs whatever it is given, and this process is
    elevated, so the question is not whether the file exists but whether it is
    the library that was shipped. ``scripts/fetch_vendor.ps1`` checks the hash
    at download time, which is months and one writable directory away from the
    moment it matters.

    An explicitly configured path is allowed through with a warning rather than
    refused. Someone who set ``helper.dll_path`` by hand has chosen a different
    build on purpose, and refusing them would only teach them to set the escape
    hatch permanently.

    Args:
        dll: Path to the assembly.
        explicit: Whether the path came from configuration rather than the
            bundled default.

    Raises:
        JarvisError: The bundled assembly does not match the pinned hash.
    """
    import hashlib
    import os
    from pathlib import Path

    path = Path(dll)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest == _VENDORED_DLL_SHA256:
        return

    if explicit or os.environ.get(_ALLOW_UNVERIFIED):
        _log.warning(
            "loading an unverified sensor library into the elevated helper",
            extra={"context": {"path": str(path), "sha256": digest}},
        )
        return

    raise JarvisError(
        f"{path} does not match the vendored LibreHardwareMonitorLib "
        f"(expected {_VENDORED_DLL_SHA256}, found {digest})",
        speakable="The sensor library has been altered, so I have not loaded it.",
        context={"path": str(path), "sha256": digest},
    )


class LibreHardwareMonitor:
    """Thin wrapper over LibreHardwareMonitorLib via pythonnet.

    The computer object is opened once and reused. Each read calls ``Update()``
    on the relevant hardware node, which is what actually refreshes the values;
    without it the sensors return whatever they held at open time.
    """

    def __init__(self, dll_path: Any = None) -> None:
        self._dll_path = dll_path
        self._computer: Any = None

    def open(self) -> None:
        """Load the DLL and open the sensor tree.

        Raises:
            JarvisError: pythonnet or the DLL is unavailable, or the library
                refused to open, which normally means the process is not
                elevated.
        """
        if self._computer is not None:
            return

        from pathlib import Path

        from jarvis.util.platform import bundle_root, require_module, require_windows

        require_windows("LibreHardwareMonitor")

        # bundle_root, not project_root: in the frozen helper the DLL is inside
        # the PyInstaller unpack directory, and project_root deliberately points
        # at the install directory instead.
        dll = Path(self._dll_path) if self._dll_path else (
            bundle_root() / "vendor" / "LibreHardwareMonitorLib.dll"
        )
        if not dll.is_file():
            raise JarvisError(
                f"LibreHardwareMonitorLib.dll not found at {dll}",
                speakable="The sensor library is not installed.",
                context={"path": str(dll)},
            )
        verify_assembly(dll, explicit=self._dll_path is not None)

        clr = require_module("clr")
        try:
            clr.AddReference(str(dll.with_suffix("")))
            from LibreHardwareMonitor.Hardware import Computer  # type: ignore[import-not-found]
        except Exception as exc:
            raise JarvisError(
                f"could not load LibreHardwareMonitorLib: {exc}",
                speakable="I could not load the sensor library.",
            ) from exc

        computer = Computer()
        computer.IsCpuEnabled = True
        computer.IsMotherboardEnabled = True
        computer.IsGpuEnabled = True
        computer.IsStorageEnabled = True
        try:
            computer.Open()
        except Exception as exc:
            raise JarvisError(
                f"LibreHardwareMonitor refused to open, is this process elevated? {exc}",
                speakable="I need administrator rights to read those sensors.",
            ) from exc
        self._computer = computer

    def close(self) -> None:
        """Release the sensor tree."""
        if self._computer is not None:
            try:
                self._computer.Close()
            except Exception:  # noqa: BLE001 - shutdown failure is not interesting
                _log.debug("LibreHardwareMonitor close failed", exc_info=True)
            self._computer = None

    def _iter_sensors(self) -> Any:
        """Yield ``(hardware, sensor)`` pairs across hardware and sub-hardware."""
        self.open()
        for hardware in self._computer.Hardware:
            hardware.Update()
            for sensor in hardware.Sensors:
                yield hardware, sensor
            for sub in hardware.SubHardware:
                sub.Update()
                for sensor in sub.Sensors:
                    yield sub, sensor

    def get_thermals(self) -> dict[str, Any]:
        """CPU, motherboard, and storage temperatures in degrees Celsius."""
        readings: list[dict[str, Any]] = []
        cpu_package: float | None = None
        core_values: list[float] = []

        for hardware, sensor in self._iter_sensors():
            if str(sensor.SensorType) != "Temperature" or sensor.Value is None:
                continue
            name = str(sensor.Name)
            value = round(float(sensor.Value), 1)
            hardware_name = str(hardware.Name)
            readings.append({"hardware": hardware_name, "sensor": name, "celsius": value})

            lowered = name.lower()
            if any(hint in lowered for hint in _PACKAGE_HINTS):
                cpu_package = value if cpu_package is None else max(cpu_package, value)
            elif lowered.startswith("cpu core"):
                core_values.append(value)

        # Not every CPU exposes a package sensor. The hottest core is the
        # honest fallback, and it is what a user means by "how hot is my CPU".
        if cpu_package is None and core_values:
            cpu_package = max(core_values)

        return {
            "cpu_celsius": cpu_package,
            "cpu_core_max_celsius": max(core_values) if core_values else None,
            "sensor_count": len(readings),
            "sensors": readings[:20],
        }

    def get_fans(self) -> dict[str, Any]:
        """Fan speeds in RPM, plus control percentages where exposed."""
        fans: list[dict[str, Any]] = []
        controls: list[dict[str, Any]] = []

        for hardware, sensor in self._iter_sensors():
            if sensor.Value is None:
                continue
            kind = str(sensor.SensorType)
            entry = {
                "hardware": str(hardware.Name),
                "sensor": str(sensor.Name),
                "value": round(float(sensor.Value), 1),
            }
            if kind == "Fan":
                entry["rpm"] = entry.pop("value")
                fans.append(entry)
            elif kind == "Control":
                entry["percent"] = entry.pop("value")
                controls.append(entry)

        spinning = [f for f in fans if float(f["rpm"]) > 0]
        return {
            "fan_count": len(fans),
            "spinning_count": len(spinning),
            "max_rpm": max((float(f["rpm"]) for f in fans), default=None),
            "fans": fans[:10],
            "controls": controls[:10],
        }


def get_pending_updates() -> dict[str, Any]:
    """Query pending Windows updates. Never installs anything.

    Prefers the ``PSWindowsUpdate`` module's ``Get-WindowsUpdate`` because it is
    faster and gives sizes, then falls back to the ``Microsoft.Update.Session``
    COM API, which is always present on Windows.

    Returns:
        Count, names, and total download size in megabytes.
    """
    result = _query_pswindowsupdate()
    if result is not None:
        return result
    result = _query_com_api()
    if result is not None:
        return result
    return {
        "available": False,
        "reason": "neither PSWindowsUpdate nor the Windows Update COM API responded",
        "count": 0,
        "names": [],
        "total_size_mb": None,
    }


def _run_powershell(script: str) -> str | None:
    """Run a fixed, code-authored PowerShell script and return stdout.

    The script is a literal in this file. Nothing from an LLM, a tool argument,
    or the pipe ever reaches this function, which is why building a command
    string here does not violate §6.
    """
    import shutil

    exe = shutil.which("powershell.exe") or shutil.which("pwsh")
    if exe is None:
        return None
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, literal script
            [exe, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=_UPDATE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _log.debug("powershell query failed: %s", exc)
        return None
    if completed.returncode != 0:
        _log.debug("powershell query returned %s: %s", completed.returncode, completed.stderr[:200])
        return None
    return completed.stdout


_PSWU_SCRIPT = (
    "if (Get-Module -ListAvailable -Name PSWindowsUpdate) { "
    "Import-Module PSWindowsUpdate; "
    "Get-WindowsUpdate | Select-Object Title,Size | ConvertTo-Json -Compress "
    "} else { 'NOMODULE' }"
)

_COM_SCRIPT = (
    "$s = New-Object -ComObject Microsoft.Update.Session; "
    "$r = $s.CreateUpdateSearcher().Search('IsInstalled=0 and IsHidden=0'); "
    "$r.Updates | ForEach-Object { "
    "[pscustomobject]@{ Title = $_.Title; Size = $_.MaxDownloadSize } } "
    "| ConvertTo-Json -Compress"
)


def _parse_size_mb(value: Any) -> float | None:
    """Parse a size that may be bytes, or a string such as "1.2 GB"."""
    if value is None:
        return None
    if isinstance(value, int | float):
        return round(float(value) / (1024**2), 1)
    match = re.match(r"([\d.]+)\s*([KMGT]?B)", str(value).strip(), re.IGNORECASE)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2).upper()
    factor = {"B": 1 / (1024**2), "KB": 1 / 1024, "MB": 1.0, "GB": 1024.0, "TB": 1024.0**2}
    return round(amount * factor.get(unit, 1.0), 1)


def _shape(entries: list[dict[str, Any]], source: str) -> dict[str, Any]:
    """Build the response payload from parsed update entries."""
    sizes = [_parse_size_mb(entry.get("Size")) for entry in entries]
    known = [size for size in sizes if size is not None]
    return {
        "available": True,
        "source": source,
        "count": len(entries),
        # §5 caps spoken lists at five items.
        "names": [
            str(entry.get("Title", "unnamed update")) for entry in entries[:_MAX_UPDATE_NAMES]
        ],
        "total_size_mb": round(sum(known), 1) if known else None,
    }


def _decode_json_list(raw: str) -> list[dict[str, Any]] | None:
    """ConvertTo-Json emits an object for one item and an array for many."""
    import json

    text = raw.strip()
    if not text or text == "NOMODULE":
        return None
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return None


def _query_pswindowsupdate() -> dict[str, Any] | None:
    """Try the PSWindowsUpdate module."""
    raw = _run_powershell(_PSWU_SCRIPT)
    if raw is None or raw.strip() == "NOMODULE":
        return None
    entries = _decode_json_list(raw)
    if entries is None:
        # An empty result means no updates pending, which is a real answer.
        if raw.strip() == "":
            return _shape([], "PSWindowsUpdate")
        return None
    return _shape(entries, "PSWindowsUpdate")


def _query_com_api() -> dict[str, Any] | None:
    """Fall back to the Microsoft.Update.Session COM API."""
    raw = _run_powershell(_COM_SCRIPT)
    if raw is None:
        return None
    entries = _decode_json_list(raw)
    if entries is None:
        if raw.strip() == "":
            return _shape([], "Microsoft.Update.Session")
        return None
    return _shape(entries, "Microsoft.Update.Session")
