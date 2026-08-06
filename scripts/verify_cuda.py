"""Detect the machine, resolve the hardware tier, and persist it (CLAUDE.md T-0.4).

Run it from the repository root:

    uv run python scripts/verify_cuda.py            # detect and write config.yaml
    uv run python scripts/verify_cuda.py --dry-run  # detect and print, write nothing
    uv run python scripts/verify_cuda.py --json     # machine readable output

Every detection function here is importable and injectable so the whole flow can
be tested on a Linux box with no GPU. Nothing GPU-specific is imported at module
scope, per CLAUDE.md section 0b.

Having no NVIDIA GPU is a normal outcome. It resolves to the ``cpu`` tier and the
process still exits 0. Only a genuine failure, such as a config path that cannot
be written, exits non-zero.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import logging
import platform as _platform
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

# This file lives outside the package, so mypy resolves jarvis through the
# installed distribution, which ships no py.typed marker. The paired unused-ignore
# code keeps the same line clean when mypy is pointed at src/ instead.
from jarvis.config import (  # type: ignore[import-untyped, unused-ignore]
    TIER_PROFILES,
    Tier,
    tier_for_vram,
    write_hardware,
)
from jarvis.util.errors import (  # type: ignore[import-untyped, unused-ignore]
    ConfigError,
    DependencyMissingError,
)
from jarvis.util.platform import (  # type: ignore[import-untyped, unused-ignore]
    cpu_model,
    has_module,
    has_nvml,
    nvidia_smi_query,
    total_ram_gb,
    windows_version,
)

__all__ = [
    "CT2_MISSING_NOTE",
    "CUDNN_REMEDIATION",
    "CudaStatus",
    "GpuInfo",
    "MachineInfo",
    "Options",
    "build_hardware_values",
    "build_payload",
    "cudnn_remediation",
    "detect_cuda",
    "detect_gpu",
    "detect_machine",
    "format_report",
    "main",
    "parse_args",
    "resolve_tier",
]

_log = logging.getLogger("jarvis.verify_cuda")

#: The one failure every faster-whisper user hits on Windows. ctranslate2 loads
#: cuDNN and cuBLAS by name at runtime, so a missing DLL surfaces as an OSError
#: or a RuntimeError long after "CUDA is installed" looked true.
CUDNN_REMEDIATION = (
    "cuDNN or cuBLAS could not be loaded. This is the classic "
    "'Could not locate cudnn_ops64_9.dll' failure: faster-whisper (ctranslate2) needs the "
    "NVIDIA cuDNN 9 runtime built for CUDA 12, which the CUDA toolkit does not install.\n"
    "  Fix it either way:\n"
    "    1. Install the NVIDIA cuDNN 9 runtime for CUDA 12 from the NVIDIA developer site.\n"
    "    2. Or install the Python wheels: pip install nvidia-cudnn-cu12 nvidia-cublas-cu12\n"
    "  Then put the DLL directory on PATH, for example\n"
    "    <venv>\\Lib\\site-packages\\nvidia\\cudnn\\bin\n"
    "    <venv>\\Lib\\site-packages\\nvidia\\cublas\\bin\n"
    "  and open a new terminal so the updated PATH is picked up."
)

#: Emitted when the CUDA probe cannot run at all. Not a failure, just unverified.
CT2_MISSING_NOTE = (
    "ctranslate2 is not installed, so cuDNN could not be verified. Install the speech "
    "extra with 'uv sync --extra stt' and run this script again before trusting the "
    "GPU speech-to-text path."
)

#: Substrings that mark a shared-library load failure rather than a logic error.
_DLL_MARKERS = (
    "cudnn",
    "cublas",
    "dll load failed",
    "is not found",
    "could not locate",
    "cannot open shared object",
    "no such file or directory",
    "specified module could not be found",
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GpuInfo:
    """A detected NVIDIA GPU.

    Attributes:
        name: Marketing name, for example "NVIDIA GeForce RTX 4070".
        vram_gb: Total VRAM in GB, rounded to one decimal.
        driver_version: Display driver version, or None when unknown.
        cuda_version: CUDA version the driver supports, or None when unknown.
    """

    name: str
    vram_gb: float
    driver_version: str | None = None
    cuda_version: str | None = None


@dataclass(frozen=True, slots=True)
class CudaStatus:
    """Whether CUDA is usable, and what is broken when it is not.

    Attributes:
        available: A CUDA-capable GPU is present and the driver answered.
        version: CUDA version string, or None when it could not be read.
        cudnn_ok: True when the ctranslate2 probe loaded CUDA cleanly, False when
            it failed, None when the probe could not run at all.
        problems: Human-readable remediation lines. Empty means nothing to fix.
    """

    available: bool
    version: str | None = None
    cudnn_ok: bool | None = None
    problems: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class MachineInfo:
    """Host facts that do not depend on the GPU.

    Attributes:
        cpu_model: Best-effort CPU model string.
        ram_gb: Installed RAM in GB, rounded to one decimal.
        os_version: Windows release string, or the platform name elsewhere.
        python_version: Running interpreter version, for example "3.12.11".
    """

    cpu_model: str
    ram_gb: float
    os_version: str
    python_version: str


@dataclass(frozen=True, slots=True)
class Options:
    """Parsed command line options.

    Attributes:
        dry_run: Detect and print, but write nothing.
        as_json: Emit a JSON document on stdout instead of the text report.
        config_path: Config file to update. None means the default location.
    """

    dry_run: bool = False
    as_json: bool = False
    config_path: Path | None = None


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------


def _as_text(value: object) -> str:
    """Decode an NVML string, which is bytes on older pynvml builds."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _nvml_cuda_version(pynvml: ModuleType) -> str | None:
    """Format the driver's CUDA version, for example 12040 becomes "12.4"."""
    getter = getattr(pynvml, "nvmlSystemGetCudaDriverVersion_v2", None) or getattr(
        pynvml, "nvmlSystemGetCudaDriverVersion", None
    )
    if getter is None:
        return None
    try:
        raw = int(getter())
    except Exception:  # noqa: BLE001 - an unreadable version is not fatal
        return None
    if raw <= 0:
        return None
    return f"{raw // 1000}.{(raw % 1000) // 10}"


def _detect_gpu_nvml() -> GpuInfo | None:
    """Read GPU 0 through NVML, or None when NVML is unavailable."""
    if not has_nvml():
        return None
    try:
        pynvml = importlib.import_module("pynvml")
    except ImportError as exc:
        _log.debug("pynvml import failed", extra={"context": {"error": str(exc)}})
        return None
    try:
        pynvml.nvmlInit()
    except Exception as exc:  # noqa: BLE001 - no driver means no GPU telemetry
        _log.debug("nvmlInit failed", extra={"context": {"error": str(exc)}})
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = _as_text(pynvml.nvmlDeviceGetName(handle))
        total_bytes = float(pynvml.nvmlDeviceGetMemoryInfo(handle).total)
        driver = _as_text(pynvml.nvmlSystemGetDriverVersion())
        cuda = _nvml_cuda_version(pynvml)
    except Exception as exc:  # noqa: BLE001 - fall back to nvidia-smi below
        _log.debug("NVML query failed", extra={"context": {"error": str(exc)}})
        return None
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()
    return GpuInfo(
        name=name or "NVIDIA GPU",
        vram_gb=round(total_bytes / (1024**3), 1),
        driver_version=driver or None,
        cuda_version=cuda,
    )


def _detect_gpu_smi() -> GpuInfo | None:
    """Read GPU 0 through nvidia-smi, or None when the tool is absent."""
    query = nvidia_smi_query()
    if not query:
        return None
    try:
        vram_gb = round(float(query.get("memory_total_mib", "0")) / 1024.0, 1)
    except ValueError:
        vram_gb = 0.0
    return GpuInfo(
        name=query.get("name") or "NVIDIA GPU",
        vram_gb=vram_gb,
        driver_version=query.get("driver_version") or None,
        cuda_version=query.get("cuda_version") or None,
    )


def detect_gpu() -> GpuInfo | None:
    """Detect the primary NVIDIA GPU.

    NVML is preferred because it reports exact VRAM in bytes. nvidia-smi is the
    fallback for machines with a driver but no pynvml.

    Returns:
        The GPU, or None when no NVIDIA GPU could be found. None is a normal
        result, not an error.
    """
    return _detect_gpu_nvml() or _detect_gpu_smi()


# ---------------------------------------------------------------------------
# CUDA and cuDNN
# ---------------------------------------------------------------------------


def cudnn_remediation(exc: BaseException) -> str | None:
    """Return the cuDNN remediation text when ``exc`` looks like a DLL load failure.

    Args:
        exc: The exception raised by the CUDA probe.

    Returns:
        :data:`CUDNN_REMEDIATION` when the message names cuDNN, cuBLAS, or a
        failed library load, otherwise None.
    """
    message = f"{type(exc).__name__}: {exc}".lower()
    if any(marker in message for marker in _DLL_MARKERS):
        return CUDNN_REMEDIATION
    return None


def _ctranslate2_device_count() -> int:
    """Count CUDA devices the way faster-whisper does, through ctranslate2.

    Returns:
        The number of CUDA devices ctranslate2 can see.

    Raises:
        DependencyMissingError: ctranslate2 is not installed, so nothing can be
            verified.
        Exception: Any load failure. On Windows a missing cuDNN or cuBLAS DLL
            surfaces here as an OSError or a RuntimeError.
    """
    if not has_module("ctranslate2"):
        raise DependencyMissingError("ctranslate2", extra="stt")
    module = importlib.import_module("ctranslate2")
    return int(module.get_cuda_device_count())


def detect_cuda(
    gpu: GpuInfo | None = None,
    *,
    probe: Callable[[], int] | None = None,
) -> CudaStatus:
    """Probe whether CUDA, cuBLAS, and cuDNN are actually usable.

    Args:
        gpu: Result of :func:`detect_gpu`, passed in to avoid a second NVML probe.
            When omitted, :func:`detect_gpu` is called.
        probe: Zero-argument callable returning the CUDA device count. Defaults to
            the ctranslate2 probe. Injected by tests.

    Returns:
        A :class:`CudaStatus`. A missing GPU yields ``available=False`` with no
        problems, since running on CPU is a supported configuration.
    """
    if gpu is None:
        gpu = detect_gpu()
    if gpu is None:
        return CudaStatus(available=False, version=None, cudnn_ok=None, problems=[])

    version = gpu.cuda_version
    run_probe = probe if probe is not None else _ctranslate2_device_count
    problems: list[str] = []
    try:
        count = int(run_probe())
    except DependencyMissingError:
        problems.append(CT2_MISSING_NOTE)
        return CudaStatus(available=True, version=version, cudnn_ok=None, problems=problems)
    except Exception as exc:  # noqa: BLE001 - OSError, RuntimeError, and ImportError all mean
        # the same thing here: the CUDA stack did not load. The message says which library.
        _log.debug("CUDA probe failed", extra={"context": {"error": str(exc)}})
        problems.append(cudnn_remediation(exc) or f"ctranslate2 could not initialise CUDA: {exc}")
        # The driver and the card are fine, only the user-space libraries are not,
        # so the tier stays as the VRAM dictates and the fix is printed instead.
        return CudaStatus(available=True, version=version, cudnn_ok=False, problems=problems)

    if count <= 0:
        problems.append(
            "An NVIDIA GPU is present but ctranslate2 reports no usable CUDA device. "
            "Install the CUDA 12 runtime and a matching NVIDIA driver, then re-run this script."
        )
        return CudaStatus(available=False, version=version, cudnn_ok=None, problems=problems)
    return CudaStatus(available=True, version=version, cudnn_ok=True, problems=problems)


# ---------------------------------------------------------------------------
# Machine and tier
# ---------------------------------------------------------------------------


def detect_machine() -> MachineInfo:
    """Collect CPU, RAM, OS, and interpreter facts. Never raises."""
    return MachineInfo(
        cpu_model=cpu_model(),
        ram_gb=total_ram_gb(),
        os_version=windows_version(),
        python_version=_platform.python_version(),
    )


def resolve_tier(gpu: GpuInfo | None, cuda: CudaStatus) -> Tier:
    """Resolve the hardware tier from the detected GPU and CUDA status.

    Delegates the thresholds to :func:`jarvis.config.tier_for_vram` so the table in
    CLAUDE.md section 2 lives in exactly one place. A broken cuDNN install does not
    lower the tier: the card is still the card, and the remediation is printed.

    Args:
        gpu: Detected GPU, or None.
        cuda: Result of :func:`detect_cuda`.

    Returns:
        The resolved tier. ``Tier.CPU`` whenever there is no usable CUDA GPU.
    """
    vram = gpu.vram_gb if gpu is not None else None
    return tier_for_vram(vram, cuda_available=cuda.available and gpu is not None)


def build_hardware_values(
    gpu: GpuInfo | None,
    cuda: CudaStatus,
    machine: MachineInfo,
    tier: Tier,
) -> dict[str, Any]:
    """Build the ``hardware`` mapping for :func:`jarvis.config.write_hardware`.

    Every key matches a field of :class:`jarvis.config.HardwareConfig`. ``None``
    values are dropped by ``write_hardware`` so a failed detection never
    overwrites a value the user set by hand.
    """
    return {
        "tier": tier.value,
        "gpu_model": gpu.name if gpu is not None else None,
        "gpu_vram_gb": gpu.vram_gb if gpu is not None else None,
        "cpu_model": machine.cpu_model,
        "ram_gb": machine.ram_gb,
        "windows_version": machine.os_version,
        "cuda_available": cuda.available,
        "cuda_version": cuda.version,
        "driver_version": gpu.driver_version if gpu is not None else None,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _cudnn_label(cudnn_ok: bool | None) -> str:
    """Render the tri-state cuDNN result."""
    if cudnn_ok is None:
        return "not verified"
    return "ok" if cudnn_ok else "FAILED"


def _row(label: str, value: str) -> str:
    """One aligned report line."""
    return f"  {label:<9} {value}"


def format_report(
    gpu: GpuInfo | None,
    cuda: CudaStatus,
    machine: MachineInfo,
    tier: Tier,
    *,
    written_to: Path | None = None,
    dry_run: bool = False,
) -> str:
    """Render the human-readable report, including any remediation banner."""
    profile = TIER_PROFILES[tier]
    lines = ["JARVIS hardware check", "=====================", ""]
    if gpu is None:
        lines.append(_row("GPU", "none detected (valid cpu tier, not an error)"))
    else:
        lines.append(_row("GPU", gpu.name))
        lines.append(_row("VRAM", f"{gpu.vram_gb:.1f} GB"))
        lines.append(_row("Driver", gpu.driver_version or "unknown"))
    lines.append(_row("CUDA", cuda.version or ("unavailable" if not cuda.available else "unknown")))
    lines.append(_row("cuDNN", _cudnn_label(cuda.cudnn_ok)))
    lines.append(_row("CPU", machine.cpu_model))
    lines.append(_row("RAM", f"{machine.ram_gb:.1f} GB"))
    lines.append(_row("OS", machine.os_version))
    lines.append(_row("Python", machine.python_version))
    lines.append("")
    lines.append(_row("Tier", tier.value))
    lines.append(
        _row(
            "STT",
            f"{profile.stt_engine.value} {profile.stt_model} "
            f"{profile.stt_compute_type} on {profile.stt_device}",
        )
    )
    lines.append(_row("LLM", profile.llm_model))
    vision = "available" if profile.supports_vision else "disabled on this tier"
    lines.append(_row("Vision", vision))
    if dry_run:
        lines.append(_row("Config", "dry run, nothing written"))
    elif written_to is not None:
        lines.append(_row("Config", f"wrote {written_to}"))
    if cuda.problems:
        lines.extend(["", "ACTION REQUIRED", "==============="])
        for problem in cuda.problems:
            lines.extend(["", problem])
    return "\n".join(lines)


def build_payload(
    gpu: GpuInfo | None,
    cuda: CudaStatus,
    machine: MachineInfo,
    tier: Tier,
    hardware: dict[str, Any],
    *,
    written_to: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build the ``--json`` document. Every value is JSON serialisable."""
    profile = TIER_PROFILES[tier]
    return {
        "gpu": asdict(gpu) if gpu is not None else None,
        "cuda": asdict(cuda),
        "machine": asdict(machine),
        "tier": tier.value,
        "stt": {
            "engine": profile.stt_engine.value,
            "model": profile.stt_model,
            "device": profile.stt_device,
            "compute_type": profile.stt_compute_type,
        },
        "llm_model": profile.llm_model,
        "vision_supported": profile.supports_vision,
        "hardware": hardware,
        "dry_run": dry_run,
        "written_to": str(written_to) if written_to is not None else None,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> Options:
    """Parse the command line into an :class:`Options`."""
    parser = argparse.ArgumentParser(
        prog="verify_cuda",
        description=(
            "Detect GPU, VRAM, CUDA and cuDNN availability, resolve the JARVIS "
            "hardware tier, and write it into config.yaml."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="detect and report, but do not write config.yaml",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a JSON document on stdout instead of the text report",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="config file to update, defaults to config/config.yaml",
    )
    namespace = parser.parse_args(argv)
    raw_path = namespace.config
    return Options(
        dry_run=bool(namespace.dry_run),
        as_json=bool(namespace.json),
        config_path=Path(str(raw_path)) if raw_path else None,
    )


def main(argv: list[str] | None = None) -> int:
    """Detect the machine, print the report, and persist the resolved tier.

    Args:
        argv: Command line arguments. None reads ``sys.argv``.

    Returns:
        0 on success, including the no-GPU case. 2 when the config file could not
        be written.
    """
    options = parse_args(argv)
    gpu = detect_gpu()
    cuda = detect_cuda(gpu)
    machine = detect_machine()
    tier = resolve_tier(gpu, cuda)
    hardware = build_hardware_values(gpu, cuda, machine, tier)

    written: Path | None = None
    if not options.dry_run:
        try:
            written = write_hardware(hardware, options.config_path)
        except (ConfigError, OSError) as exc:
            print(f"could not write hardware settings: {exc}", file=sys.stderr)
            return 2

    if options.as_json:
        payload = build_payload(
            gpu, cuda, machine, tier, hardware, written_to=written, dry_run=options.dry_run
        )
        print(json.dumps(payload, indent=2))
        # Keep stdout parseable, so remediation goes to stderr in JSON mode.
        for problem in cuda.problems:
            print(problem, file=sys.stderr)
    else:
        print(
            format_report(
                gpu, cuda, machine, tier, written_to=written, dry_run=options.dry_run
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
