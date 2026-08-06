"""Audio device enumeration and selection.

PortAudio device names on Windows are messy: WASAPI, MME, and DirectSound each
expose the same physical microphone under a slightly different name, and the MME
host API truncates names to 31 characters. Config therefore accepts a loose spec
(``None``, ``"auto"``, an index, or a partial name) and this module resolves it
with a documented precedence, failing with the closest candidates listed rather
than silently grabbing the wrong microphone.

``sounddevice`` is imported lazily inside the functions that need it (§0b), so
this module loads on a host with no PortAudio at all.
"""

from __future__ import annotations

import difflib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from jarvis.util.errors import AudioError
from jarvis.util.platform import require_module

__all__ = [
    "AUTO_SPECS",
    "DeviceInfo",
    "Direction",
    "default_input_device",
    "default_output_device",
    "describe_devices",
    "find_input_device",
    "find_output_device",
    "is_auto_spec",
    "list_devices",
    "supports_input_rate",
    "why_the_rate_failed",
]

_log = logging.getLogger(__name__)

#: String specs meaning "whatever the operating system considers the default".
#: A physical device genuinely named "default" (some ALSA setups) can still be
#: selected by index.
AUTO_SPECS: frozenset[str] = frozenset({"", "auto", "default", "system", "none"})

#: Which side of a device we care about.
Direction = Literal["input", "output"]

#: How many alternative names an error message offers.
_MAX_SUGGESTIONS = 4

#: difflib similarity below which a name is not worth suggesting.
_SUGGESTION_CUTOFF = 0.4


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """One audio endpoint as PortAudio reports it.

    Attributes:
        index: PortAudio device index, the value handed back to sounddevice.
        name: Human readable device name.
        max_input_channels: Capture channels the device offers. Zero means it is
            output only.
        max_output_channels: Playback channels the device offers. Zero means it
            is input only.
        default_samplerate: Rate PortAudio reports as native, in hertz.
        hostapi: Index of the host API that owns this device.
        is_default_input: True when this is the system default microphone.
        is_default_output: True when this is the system default speaker.
    """

    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    hostapi: int
    is_default_input: bool = False
    is_default_output: bool = False

    @property
    def is_input(self) -> bool:
        """True when the device can capture audio."""
        return self.max_input_channels > 0

    @property
    def is_output(self) -> bool:
        """True when the device can play audio."""
        return self.max_output_channels > 0

    def supports(self, direction: Direction) -> bool:
        """True when the device works in ``direction``."""
        return self.is_input if direction == "input" else self.is_output

    def is_default_for(self, direction: Direction) -> bool:
        """True when the device is the system default for ``direction``."""
        return self.is_default_input if direction == "input" else self.is_default_output

    def label(self) -> str:
        """Short identifier for logs, for example ``3: Microphone (Realtek)``."""
        return f"{self.index}: {self.name}"


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


def _default_indices(raw_default: Any) -> tuple[int, int]:
    """Pull ``(input_index, output_index)`` out of ``sounddevice.default.device``.

    PortAudio reports -1 when a direction has no default. The value is a pair in
    practice but is defensive here because sounddevice exposes it as a custom
    sequence type.
    """
    try:
        pair = list(raw_default)
    except TypeError:
        return (-1, -1)
    if len(pair) < 2:
        return (-1, -1)

    def _as_index(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return -1

    return (_as_index(pair[0]), _as_index(pair[1]))


def _parse_devices(
    raw_devices: Sequence[Mapping[str, Any]],
    raw_default: Any = (-1, -1),
) -> list[DeviceInfo]:
    """Convert sounddevice's dict rows into :class:`DeviceInfo`.

    Kept separate from :func:`list_devices` so the parsing rules can be tested
    without PortAudio present.

    Args:
        raw_devices: Rows as returned by ``sounddevice.query_devices()``.
        raw_default: Value of ``sounddevice.default.device``.

    Returns:
        One entry per device, in PortAudio index order.
    """
    default_in, default_out = _default_indices(raw_default)
    devices: list[DeviceInfo] = []
    for position, row in enumerate(raw_devices):
        index = int(row.get("index", position))
        devices.append(
            DeviceInfo(
                index=index,
                name=str(row.get("name", f"device {index}")).strip(),
                max_input_channels=int(row.get("max_input_channels", 0) or 0),
                max_output_channels=int(row.get("max_output_channels", 0) or 0),
                default_samplerate=float(row.get("default_samplerate", 0.0) or 0.0),
                hostapi=int(row.get("hostapi", 0) or 0),
                is_default_input=index == default_in,
                is_default_output=index == default_out,
            )
        )
    return devices


def list_devices() -> list[DeviceInfo]:
    """Enumerate every audio device PortAudio can see.

    Returns:
        Devices in PortAudio index order, both inputs and outputs.

    Raises:
        DependencyMissingError: ``sounddevice`` is not installed.
        AudioError: PortAudio is installed but enumeration failed, which usually
            means no audio backend is running.
    """
    sd = require_module("sounddevice", feature="audio device enumeration")
    try:
        raw_devices = list(sd.query_devices())
        raw_default = sd.default.device
    except Exception as exc:  # PortAudio raises several unrelated types
        raise AudioError(
            f"could not enumerate audio devices: {exc}",
            speakable="I could not see any audio devices on this machine.",
            context={"error": str(exc)},
        ) from exc
    return _parse_devices(raw_devices, raw_default)


def _hostapi_names() -> dict[int, str]:
    """Host API index to name. Empty when PortAudio is unavailable."""
    try:
        sd = require_module("sounddevice")
        return {i: str(api.get("name", i)) for i, api in enumerate(sd.query_hostapis())}
    except Exception:  # noqa: BLE001 - decoration only, never worth failing over
        _log.debug("host API names unavailable", exc_info=True)
        return {}


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def is_auto_spec(spec: str | int | None) -> bool:
    """True when ``spec`` asks for the system default rather than a named device.

    Args:
        spec: A config device spec: ``None``, an index, or a name.

    Returns:
        True for ``None`` and for the placeholder names in :data:`AUTO_SPECS`.
        Callers use this to skip device enumeration entirely, which matters
        because enumeration needs PortAudio and the default path does not.
    """
    if spec is None:
        return True
    return isinstance(spec, str) and spec.strip().lower() in AUTO_SPECS


def _pick(matches: Sequence[DeviceInfo], direction: Direction) -> DeviceInfo:
    """Choose one device from several equally good matches.

    The system default wins, otherwise the lowest index does. On Windows the
    same microphone appears once per host API, and the default is the one the
    user actually chose in the sound control panel.
    """
    for device in matches:
        if device.is_default_for(direction):
            return device
    return min(matches, key=lambda device: device.index)


def _default_device(candidates: Sequence[DeviceInfo], direction: Direction) -> DeviceInfo | None:
    """System default for ``direction``, or None when there is none."""
    for device in candidates:
        if device.is_default_for(direction):
            return device
    if not candidates:
        _log.warning("no %s devices are available", direction)
        return None
    # PortAudio did not flag a default. Falling back to the first usable device
    # is better than refusing to start.
    fallback = min(candidates, key=lambda device: device.index)
    _log.debug(
        "no default %s device reported, falling back to %s", direction, fallback.label()
    )
    return fallback


def _suggestions(spec: str, candidates: Sequence[DeviceInfo]) -> list[str]:
    """Names closest to ``spec``, for an actionable error message."""
    names = [device.name for device in candidates]
    close = difflib.get_close_matches(
        spec, names, n=_MAX_SUGGESTIONS, cutoff=_SUGGESTION_CUTOFF
    )
    return close or names[:_MAX_SUGGESTIONS]


def _by_name(
    spec: str,
    candidates: Sequence[DeviceInfo],
    direction: Direction,
) -> DeviceInfo:
    """Resolve a device by name.

    Precedence: exact match, then case-insensitive exact, then
    case-insensitive substring. Ties are broken by :func:`_pick`.

    Raises:
        AudioError: Nothing matched. The message names the closest candidates.
    """
    wanted = spec.strip()
    folded = wanted.casefold()

    exact = [device for device in candidates if device.name == wanted]
    if exact:
        return _pick(exact, direction)

    insensitive = [device for device in candidates if device.name.casefold() == folded]
    if insensitive:
        return _pick(insensitive, direction)

    partial = [device for device in candidates if folded in device.name.casefold()]
    if partial:
        chosen = _pick(partial, direction)
        _log.debug("device spec %r matched %s by substring", spec, chosen.label())
        return chosen

    hints = _suggestions(wanted, candidates)
    raise AudioError(
        f"no {direction} device matches {spec!r}. "
        + (
            f"Closest names: {', '.join(hints)}"
            if hints
            else f"This machine reports no {direction} devices at all."
        ),
        speakable=(
            "I could not find that microphone."
            if direction == "input"
            else "I could not find that speaker."
        ),
        context={"spec": spec, "direction": direction, "candidates": hints},
    )


def _by_index(
    index: int,
    devices: Sequence[DeviceInfo],
    direction: Direction,
) -> DeviceInfo:
    """Resolve a device by PortAudio index, validated against the device list.

    Raises:
        AudioError: No such index, or the device has no channels in ``direction``.
    """
    for device in devices:
        if device.index == index:
            if not device.supports(direction):
                raise AudioError(
                    f"device {index} ({device.name}) has no {direction} channels",
                    speakable=(
                        "The audio device I was told to use cannot record."
                        if direction == "input"
                        else "The audio device I was told to use cannot play sound."
                    ),
                    context={"index": index, "direction": direction, "name": device.name},
                )
            return device
    usable = sorted(d.index for d in devices if d.supports(direction))
    raise AudioError(
        f"there is no audio device with index {index}. "
        f"Usable {direction} indices: {usable if usable else 'none'}",
        speakable="The audio device I was told to use does not exist.",
        context={"index": index, "direction": direction, "usable": usable},
    )


def _find(
    spec: str | int | None,
    direction: Direction,
    devices: Sequence[DeviceInfo] | None,
) -> DeviceInfo | None:
    """Shared resolution for :func:`find_input_device` and :func:`find_output_device`."""
    pool = list(devices) if devices is not None else list_devices()
    candidates = [device for device in pool if device.supports(direction)]

    if spec is None or is_auto_spec(spec):
        return _default_device(candidates, direction)
    if isinstance(spec, bool):
        # bool is an int subclass, and True would silently mean "device 1".
        raise AudioError(
            f"audio device spec must be a name, an index, or null, got {spec!r}",
            context={"spec": spec, "direction": direction},
        )
    if isinstance(spec, int):
        return _by_index(spec, pool, direction)
    return _by_name(spec, candidates, direction)


def find_input_device(
    spec: str | int | None,
    *,
    devices: Sequence[DeviceInfo] | None = None,
) -> DeviceInfo | None:
    """Resolve a capture device from a config spec.

    Args:
        spec: ``None`` or ``"auto"`` for the system default, an integer
            PortAudio index, or a device name. Names are matched exactly first,
            then case-insensitively, then by case-insensitive substring.
        devices: Device list to search. Defaults to :func:`list_devices`.
            Injected by tests and by callers that already enumerated.

    Returns:
        The matching device, or None when the machine has no input device at all.

    Raises:
        AudioError: The spec names an index or a device that does not exist, or
            names a device with no input channels.
    """
    return _find(spec, "input", devices)


def find_output_device(
    spec: str | int | None,
    *,
    devices: Sequence[DeviceInfo] | None = None,
) -> DeviceInfo | None:
    """Resolve a playback device from a config spec.

    Args:
        spec: ``None`` or ``"auto"`` for the system default, an integer
            PortAudio index, or a device name, matched as in
            :func:`find_input_device`.
        devices: Device list to search. Defaults to :func:`list_devices`.

    Returns:
        The matching device, or None when the machine has no output device.

    Raises:
        AudioError: The spec names an index or a device that does not exist, or
            names a device with no output channels.
    """
    return _find(spec, "output", devices)


def default_input_device(
    *, devices: Sequence[DeviceInfo] | None = None
) -> DeviceInfo | None:
    """System default microphone, or None when there is no input device."""
    return _find(None, "input", devices)


def default_output_device(
    *, devices: Sequence[DeviceInfo] | None = None
) -> DeviceInfo | None:
    """System default speaker, or None when there is no output device."""
    return _find(None, "output", devices)


# ---------------------------------------------------------------------------
# Troubleshooting
# ---------------------------------------------------------------------------


def describe_devices(
    devices: Sequence[DeviceInfo] | None = None,
    *,
    hostapi_names: Mapping[int, str] | None = None,
) -> str:
    """Render the device list as a table for logs and troubleshooting.

    Args:
        devices: Devices to render. Defaults to :func:`list_devices`.
        hostapi_names: Host API index to name. Looked up from PortAudio when
            omitted, and silently skipped if PortAudio cannot answer.

    Returns:
        A multi-line table, or a single explanatory line when nothing is
        available. Never raises: this is the function a user is told to run when
        audio is already broken.
    """
    if devices is None:
        try:
            devices = list_devices()
        except Exception as exc:  # noqa: BLE001 - diagnostics must never add a failure
            return f"audio devices could not be listed: {exc}"
    if not devices:
        return "no audio devices were found"

    apis = dict(hostapi_names) if hostapi_names is not None else _hostapi_names()
    rows: list[tuple[str, str, str, str, str, str]] = []
    for device in devices:
        flags = []
        if device.is_default_input:
            flags.append("default in")
        if device.is_default_output:
            flags.append("default out")
        rows.append(
            (
                str(device.index),
                str(device.max_input_channels),
                str(device.max_output_channels),
                f"{device.default_samplerate:.0f}",
                apis.get(device.hostapi, str(device.hostapi)),
                device.name + (f"  [{', '.join(flags)}]" if flags else ""),
            )
        )

    headers = ("idx", "in", "out", "rate", "host api", "name")
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]
    lines = [
        "  ".join(headers[column].ljust(widths[column]) for column in range(len(headers))).rstrip()
    ]
    lines.append("  ".join("-" * widths[column] for column in range(len(headers))))
    lines.extend(
        "  ".join(row[column].ljust(widths[column]) for column in range(len(headers))).rstrip()
        for row in rows
    )
    return "\n".join(lines)


def supports_input_rate(index: int, sample_rate: int, channels: int = 1) -> bool:
    """Whether PortAudio will open ``index`` for capture at ``sample_rate``.

    Args:
        index: PortAudio device index.
        sample_rate: Rate in hertz.
        channels: Capture channels.

    Returns:
        True when the combination opens. Never raises: an unanswerable question
        is reported as unsupported.
    """
    try:
        sounddevice = require_module("sounddevice", feature="audio devices")
        sounddevice.check_input_settings(
            device=index, samplerate=sample_rate, channels=channels
        )
    except Exception:  # noqa: BLE001 - any refusal means the same thing here
        return False
    return True


def why_the_rate_failed(spec: str | int | None, sample_rate: int, channels: int = 1) -> str:
    """Explain a sample rate refusal, and name devices that would work.

    PortAudio reports only "Invalid sample rate", which is a dead end: it does
    not say what the device wanted or which other device would do. WASAPI in
    particular refuses to resample in shared mode, so a device whose native
    rate is 48 kHz simply will not open at the 16 kHz the wake word and STT
    both require.

    Returns:
        A multi-line explanation. Never raises.
    """
    lines: list[str] = []
    try:
        devices = list_devices()
    except Exception:  # noqa: BLE001 - diagnostics must not add a second failure
        return f"the device would not open at {sample_rate} Hz"

    chosen = None
    if isinstance(spec, int):
        chosen = next((d for d in devices if d.index == spec), None)
    if chosen is not None:
        lines.append(
            f"device {chosen.index} ({chosen.name}) reports a native rate of "
            f"{chosen.default_samplerate:.0f} Hz and refused {sample_rate} Hz."
        )
        lines.append(
            "WASAPI and WDM-KS will not resample, so a device must accept the rate directly."
        )
    else:
        lines.append(f"no input device accepted {sample_rate} Hz.")

    usable = [
        device
        for device in devices
        if device.is_input and supports_input_rate(device.index, sample_rate, channels)
    ]
    if usable:
        lines.append(f"These inputs do accept {sample_rate} Hz:")
        lines.extend(f"  {device.index}: {device.name}" for device in usable[:5])
        lines.append("Set audio.input_device to one of those, or to null for the system default.")
    else:
        lines.append(
            "No input device accepts it directly. Leave audio.input_device unset: "
            "the MME default resamples, which is why it works when a specific device does not."
        )
    return "\n".join(lines)
