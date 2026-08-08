"""Capability detection.

CLAUDE.md §0b: the target host is Windows 11 but the development host may not be.
Nothing in ``src/jarvis`` may import a Windows-only or GPU-only module at module
scope. Code asks this module what is available and degrades to a structured,
speakable error instead of raising ImportError deep in the audio path.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import os
import platform as _platform
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from jarvis.util.errors import DependencyMissingError, PlatformUnsupportedError

__all__ = [
    "bundle_root",
    "cpu_model",
    "cuda_runtime_usable",
    "has_cuda",
    "has_module",
    "has_nvml",
    "is_frozen",
    "is_linux",
    "is_macos",
    "is_windows",
    "nvidia_smi_query",
    "project_root",
    "require_module",
    "require_windows",
    "total_ram_gb",
    "windows_version",
]

# Optional dependency name -> the pyproject extra that provides it. Used to build
# an actionable install hint when something is missing.
_EXTRA_FOR_MODULE: dict[str, str] = {
    "sounddevice": "audio",
    "openwakeword": "audio",
    "onnxruntime": "audio",
    "faster_whisper": "stt",
    "pywhispercpp": "stt",
    "kokoro": "tts",
    "soundfile": "tts",
    "pynvml": "gpu",
    "mss": "vision",
    "win32api": "windows",
    "win32con": "windows",
    "win32evtlog": "windows",
    "win32gui": "windows",
    "win32pipe": "windows",
    "win32file": "windows",
    "wmi": "windows",
    "clr": "windows",
    "pywinauto": "windows",
    "pygetwindow": "windows",
    "windows_toasts": "windows",
}


def is_windows() -> bool:
    """True when running on a Windows host."""
    return sys.platform == "win32"


def is_linux() -> bool:
    """True when running on a Linux host."""
    return sys.platform.startswith("linux")


def is_macos() -> bool:
    """True when running on a macOS host."""
    return sys.platform == "darwin"


def has_module(name: str) -> bool:
    """True when ``name`` can be imported without actually importing it.

    Never raises. A package whose finder blows up is reported as absent.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError, ModuleNotFoundError):
        return False


def require_module(name: str, *, feature: str | None = None) -> ModuleType:
    """Import ``name`` or raise :class:`DependencyMissingError` with an install hint.

    Args:
        name: Importable module name, for example ``"sounddevice"``.
        feature: Human-readable feature name used in the log message.

    Raises:
        DependencyMissingError: The module is not installed.
    """
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise DependencyMissingError(
            name,
            extra=_EXTRA_FOR_MODULE.get(name),
            context={"feature": feature or name, "import_error": str(exc)},
        ) from exc


def require_windows(feature: str) -> None:
    """Raise :class:`PlatformUnsupportedError` unless running on Windows."""
    if not is_windows():
        raise PlatformUnsupportedError(feature)


@lru_cache(maxsize=1)
def has_nvml() -> bool:
    """True when pynvml is installed and an NVIDIA driver responds."""
    if not has_module("pynvml"):
        return False
    try:
        pynvml = importlib.import_module("pynvml")
        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001 - any NVML failure means "no GPU telemetry"
        return False
    try:
        return bool(pynvml.nvmlDeviceGetCount() > 0)
    except Exception:  # noqa: BLE001
        return False
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()


@lru_cache(maxsize=1)
def nvidia_smi_query() -> dict[str, str] | None:
    """Query ``nvidia-smi`` for the first GPU. Returns None when unavailable.

    Kept separate from NVML because ``nvidia-smi`` reports the driver's CUDA
    version, which NVML does not expose in the same form.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
            [
                exe,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    first = out.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 3:
        return None
    result = {"name": parts[0], "memory_total_mib": parts[1], "driver_version": parts[2]}
    try:
        ver = subprocess.run(  # noqa: S603
            [exe, "--query", "--display=COMPUTE"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        for line in ver.stdout.splitlines():
            if "CUDA Version" in line:
                result["cuda_version"] = line.split(":", 1)[1].strip()
                break
    except (OSError, subprocess.SubprocessError):
        pass
    return result


@lru_cache(maxsize=1)
def has_cuda() -> bool:
    """True when a CUDA-capable NVIDIA GPU appears usable."""
    return has_nvml() or nvidia_smi_query() is not None


@lru_cache(maxsize=1)
def cpu_model() -> str:
    """Best-effort CPU model string. Never raises."""
    if is_windows():
        name = os.environ.get("PROCESSOR_IDENTIFIER")
        if name:
            return name.strip()
    if is_linux():
        cpuinfo = Path("/proc/cpuinfo")
        try:
            for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return _platform.processor() or _platform.machine() or "unknown"


def total_ram_gb() -> float:
    """Installed RAM in GB, rounded to one decimal. Never raises."""
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:  # noqa: BLE001
        return 0.0


@lru_cache(maxsize=1)
def windows_version() -> str:
    """Windows release string, or the platform name when not on Windows."""
    if not is_windows():
        return f"{_platform.system()} {_platform.release()}"
    release, version, _csd, _ptype = _platform.win32_ver()
    edition = ""
    try:
        edition = _platform.win32_edition() or ""
    except (AttributeError, OSError):
        edition = ""
    # Windows 11 still reports release "10"; build 22000+ distinguishes it.
    build = 0
    try:
        build = int(version.split(".")[-1])
    except (ValueError, IndexError):
        build = 0
    if release == "10" and build >= 22000:
        release = "11"
    return " ".join(p for p in (f"Windows {release}", edition, f"build {build or version}") if p)


def cuda_runtime_usable() -> tuple[bool, str]:
    """Whether CTranslate2 can actually place a model on the GPU.

    A card nvidia-smi can see is not a CUDA runtime. CTranslate2 loads cuBLAS
    and cuDNN 9 lazily, so a machine with a healthy driver and no runtime
    libraries reports a GPU everywhere and then fails at the first
    transcription. ``--check`` said "cuda yes" seven seconds before
    ``cublas64_12.dll is not found`` on exactly such a machine, which is worse
    than saying nothing.

    Returns:
        ``(usable, detail)``. The detail names what is missing when it is not.
    """
    if not has_module("ctranslate2"):
        return False, "ctranslate2 is not installed"
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() < 1:
            return False, "no CUDA device visible to ctranslate2"
    except Exception as exc:  # noqa: BLE001 - any probe failure means unusable
        return False, str(exc)

    # Device count alone does not load the support libraries. Only building
    # something on the device does, which is what fails in practice.
    try:
        ctranslate2.models.Whisper  # noqa: B018 - attribute presence check
    except AttributeError:  # pragma: no cover - very old ctranslate2
        return True, "usable"
    try:
        from ctypes import CDLL

        for name in ("cublas64_12.dll", "cublas64_11.dll", "libcublas.so.12"):
            try:
                CDLL(name)
            except OSError:
                continue
            return True, "usable"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    return False, "cuBLAS is not on the library path"


def is_frozen() -> bool:
    """Whether this is running from a PyInstaller bundle rather than source.

    Deliberately uncached: it is one attribute read, and caching it would make
    the frozen and source paths untestable in the same process.
    """
    return bool(getattr(sys, "frozen", False))


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Absolute path to the root that user-editable files live under.

    Resolved from the installed package location, never hardcoded, per §0.6.
    Honours ``JARVIS_PROJECT_ROOT`` so a packaged build can point elsewhere.

    Frozen builds are the interesting case. ``scripts/build_helper.ps1``
    produces a PyInstaller onefile exe, which unpacks itself into a fresh
    ``%TEMP%\\_MEIxxxx`` on every run. Walking up from ``__file__`` there lands
    on the *parent* of that directory, which is ``%TEMP%`` itself, so the root
    became a directory every process running as the user can write to. Since
    ``jarvis-helper.exe`` runs elevated by design, that turned a path lookup
    into a way to hand it attacker-controlled bytes. Frozen builds resolve to
    the directory the executable sits in instead.
    """
    override = os.environ.get("JARVIS_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    if is_frozen():
        return Path(sys.executable).resolve().parent
    # src/jarvis/util/platform.py -> src/jarvis/util -> src/jarvis -> src -> root
    return Path(__file__).resolve().parents[3]


def bundle_root() -> Path:
    """Absolute path to read-only assets shipped inside the build.

    Distinct from :func:`project_root` because the two diverge in a frozen
    build: files added with PyInstaller's ``--add-binary`` land in the unpack
    directory, while anything the user edits or writes belongs beside the
    executable. Use this for the vendored DLL and other bundled data, and
    :func:`project_root` for config, logs, and models.
    """
    if is_frozen():
        unpacked = getattr(sys, "_MEIPASS", None)
        if unpacked:
            return Path(str(unpacked)).resolve()
    return project_root()
