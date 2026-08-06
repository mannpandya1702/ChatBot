"""Tests for ``scripts/verify_cuda.py`` (T-0.4).

The script is the only thing standing between a fresh Windows box and a silently
wrong hardware tier, so the whole detection path is exercised here with fakes.
Nothing in this file needs a GPU, a driver, or Windows: every probe is injected.

The one failure mode that matters most is the missing ``cudnn_ops64_9.dll``,
which is what faster-whisper actually dies on. There are dedicated tests for the
remediation text and for the fact that a broken cuDNN does not silently demote
the machine to the cpu tier.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
import types
from pathlib import Path

import pytest
import yaml

from jarvis.config import HardwareConfig, Tier, load_config
from jarvis.util.errors import ConfigError, DependencyMissingError

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_cuda.py"


def _load_script() -> types.ModuleType:
    """Import ``scripts/verify_cuda.py`` by path. ``scripts`` is not a package."""
    spec = importlib.util.spec_from_file_location("jarvis_verify_cuda", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


vc = _load_script()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_gpu(vram_gb: float = 12.0, **kwargs: object) -> object:
    """A GpuInfo with sensible defaults."""
    fields: dict[str, object] = {
        "name": "NVIDIA GeForce RTX 4070",
        "vram_gb": vram_gb,
        "driver_version": "551.86",
        "cuda_version": "12.4",
    }
    fields.update(kwargs)
    return vc.GpuInfo(**fields)


def raising(exc: BaseException):
    """A zero-argument probe that raises ``exc``."""

    def _probe() -> int:
        raise exc

    return _probe


def fake_pynvml(
    *,
    total_bytes: int = 25_757_220_864,
    name: bytes | str = b"NVIDIA GeForce RTX 4090",
    driver: bytes | str = b"551.86",
    cuda_raw: int = 12_040,
    fail_on_query: bool = False,
) -> types.ModuleType:
    """A stand-in pynvml module good enough for detect_gpu."""
    module = types.ModuleType("pynvml")
    calls: list[str] = []

    def _init() -> None:
        calls.append("init")

    def _handle(index: int) -> str:
        calls.append(f"handle:{index}")
        if fail_on_query:
            raise RuntimeError("NVML_ERROR_NOT_SUPPORTED")
        return "handle"

    def _memory(handle: str) -> types.SimpleNamespace:
        return types.SimpleNamespace(total=total_bytes, free=total_bytes, used=0)

    def _shutdown() -> None:
        calls.append("shutdown")

    module.nvmlInit = _init  # type: ignore[attr-defined]
    module.nvmlDeviceGetHandleByIndex = _handle  # type: ignore[attr-defined]
    module.nvmlDeviceGetName = lambda handle: name  # type: ignore[attr-defined]
    module.nvmlDeviceGetMemoryInfo = _memory  # type: ignore[attr-defined]
    module.nvmlSystemGetDriverVersion = lambda: driver  # type: ignore[attr-defined]
    module.nvmlSystemGetCudaDriverVersion_v2 = lambda: cuda_raw  # type: ignore[attr-defined]
    module.nvmlShutdown = _shutdown  # type: ignore[attr-defined]
    module.calls = calls  # type: ignore[attr-defined]
    return module


def patch_detection(monkeypatch: pytest.MonkeyPatch, gpu: object, cuda: object) -> None:
    """Freeze detect_gpu and detect_cuda so main() runs on known hardware."""
    monkeypatch.setattr(vc, "detect_gpu", lambda: gpu)
    monkeypatch.setattr(vc, "detect_cuda", lambda _gpu=None, **_kw: cuda)


CUDNN_FAILURES = [
    OSError("[WinError 126] Could not locate cudnn_ops64_9.dll"),
    OSError("Error loading cublas64_12.dll"),
    RuntimeError("Library cudnn_ops_infer64_8.dll is not found or cannot be loaded"),
    ImportError("DLL load failed while importing _ext: The specified module could not be found."),
    OSError("libcudnn_ops.so.9: cannot open shared object file"),
]


# ---------------------------------------------------------------------------
# Module hygiene (CLAUDE.md section 0b)
# ---------------------------------------------------------------------------


def test_script_imports_without_gpu_modules() -> None:
    """Importing the script must not drag in pynvml or ctranslate2."""
    assert "pynvml" not in sys.modules
    assert "ctranslate2" not in sys.modules
    assert callable(vc.main)


def test_public_api_is_exported() -> None:
    for name in (
        "detect_gpu",
        "detect_cuda",
        "detect_machine",
        "resolve_tier",
        "build_hardware_values",
        "main",
    ):
        assert name in vc.__all__
        assert callable(getattr(vc, name))


# ---------------------------------------------------------------------------
# detect_gpu
# ---------------------------------------------------------------------------


def test_detect_gpu_prefers_nvml(monkeypatch: pytest.MonkeyPatch) -> None:
    module = fake_pynvml()
    monkeypatch.setattr(vc, "has_nvml", lambda: True)
    monkeypatch.setitem(sys.modules, "pynvml", module)
    monkeypatch.setattr(vc, "nvidia_smi_query", lambda: pytest.fail("nvidia-smi should not run"))

    gpu = vc.detect_gpu()

    assert gpu is not None
    assert gpu.name == "NVIDIA GeForce RTX 4090"
    # 25757220864 bytes is 23.988 GiB, which must round up to the marketed 24.0.
    assert gpu.vram_gb == 24.0
    assert gpu.driver_version == "551.86"
    assert gpu.cuda_version == "12.4"
    assert "shutdown" in module.calls  # NVML must not be left initialised


def test_detect_gpu_nvml_vram_maps_to_the_expected_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    """An 8 GB card reports 8589934592 bytes and must land on gpu-8, not gpu-6."""
    monkeypatch.setattr(vc, "has_nvml", lambda: True)
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml(total_bytes=8_589_934_592))

    gpu = vc.detect_gpu()

    assert gpu is not None
    assert gpu.vram_gb == 8.0
    assert vc.resolve_tier(gpu, vc.CudaStatus(available=True)) is Tier.GPU_8


def test_detect_gpu_falls_back_to_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vc, "has_nvml", lambda: False)
    monkeypatch.setattr(
        vc,
        "nvidia_smi_query",
        lambda: {
            "name": "NVIDIA GeForce RTX 3070",
            "memory_total_mib": "8192",
            "driver_version": "546.33",
            "cuda_version": "12.3",
        },
    )

    gpu = vc.detect_gpu()

    assert gpu is not None
    assert gpu.name == "NVIDIA GeForce RTX 3070"
    assert gpu.vram_gb == 8.0
    assert gpu.driver_version == "546.33"
    assert gpu.cuda_version == "12.3"


def test_detect_gpu_falls_back_when_nvml_query_explodes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A driver that answers nvmlInit but nothing else must not kill the script."""
    module = fake_pynvml(fail_on_query=True)
    monkeypatch.setattr(vc, "has_nvml", lambda: True)
    monkeypatch.setitem(sys.modules, "pynvml", module)
    monkeypatch.setattr(
        vc,
        "nvidia_smi_query",
        lambda: {
            "name": "NVIDIA T1000",
            "memory_total_mib": "4096",
            "driver_version": "537.13",
        },
    )

    gpu = vc.detect_gpu()

    assert gpu is not None
    assert gpu.name == "NVIDIA T1000"
    assert gpu.vram_gb == 4.0
    assert gpu.cuda_version is None
    assert "shutdown" in module.calls


def test_detect_gpu_returns_none_on_a_machine_without_nvidia(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vc, "has_nvml", lambda: False)
    monkeypatch.setattr(vc, "nvidia_smi_query", lambda: None)

    assert vc.detect_gpu() is None


def test_detect_gpu_survives_unparseable_smi_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vc, "has_nvml", lambda: False)
    monkeypatch.setattr(
        vc,
        "nvidia_smi_query",
        lambda: {"name": "NVIDIA GPU", "memory_total_mib": "N/A", "driver_version": ""},
    )

    gpu = vc.detect_gpu()

    assert gpu is not None
    assert gpu.vram_gb == 0.0
    assert gpu.driver_version is None


# ---------------------------------------------------------------------------
# detect_cuda and the cuDNN remediation
# ---------------------------------------------------------------------------


def test_detect_cuda_healthy_stack() -> None:
    status = vc.detect_cuda(make_gpu(), probe=lambda: 1)

    assert status.available is True
    assert status.cudnn_ok is True
    assert status.version == "12.4"
    assert status.problems == []


@pytest.mark.parametrize("exc", CUDNN_FAILURES, ids=lambda e: type(e).__name__ + ":" + str(e)[:24])
def test_detect_cuda_reports_cudnn_dll_failure(exc: BaseException) -> None:
    status = vc.detect_cuda(make_gpu(), probe=raising(exc))

    assert status.cudnn_ok is False
    assert len(status.problems) == 1
    assert status.problems[0] == vc.CUDNN_REMEDIATION
    # The card and the driver are fine, only the user-space libraries are not.
    assert status.available is True


def test_cudnn_remediation_text_names_every_fix() -> None:
    text = vc.CUDNN_REMEDIATION
    for fragment in (
        "cudnn_ops64_9.dll",
        "cuDNN 9",
        "CUDA 12",
        "nvidia-cudnn-cu12",
        "nvidia-cublas-cu12",
        "PATH",
    ):
        assert fragment in text, f"remediation must mention {fragment!r}"


def test_cudnn_remediation_ignores_unrelated_errors() -> None:
    assert vc.cudnn_remediation(ValueError("beam_size must be positive")) is None
    assert vc.cudnn_remediation(OSError("disk quota exceeded")) is None


def test_detect_cuda_reports_unrecognised_probe_failure() -> None:
    status = vc.detect_cuda(make_gpu(), probe=raising(ValueError("unexpected device index")))

    assert status.cudnn_ok is False
    assert status.problems
    assert vc.CUDNN_REMEDIATION not in status.problems
    assert "unexpected device index" in status.problems[0]


def test_detect_cuda_without_ctranslate2_cannot_verify() -> None:
    status = vc.detect_cuda(
        make_gpu(), probe=raising(DependencyMissingError("ctranslate2", extra="stt"))
    )

    assert status.available is True
    assert status.cudnn_ok is None, "unverified is not the same as broken"
    assert status.problems == [vc.CT2_MISSING_NOTE]
    assert "ctranslate2" in vc.CT2_MISSING_NOTE
    assert "uv sync --extra stt" in vc.CT2_MISSING_NOTE


def test_detect_cuda_with_zero_devices_is_unavailable() -> None:
    status = vc.detect_cuda(make_gpu(24.0), probe=lambda: 0)

    assert status.available is False
    assert status.problems
    assert "no usable CUDA device" in status.problems[0]
    assert vc.resolve_tier(make_gpu(24.0), status) is Tier.CPU


def test_detect_cuda_without_a_gpu_is_not_a_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    """No GPU is a supported configuration, so it produces no remediation noise."""
    monkeypatch.setattr(vc, "detect_gpu", lambda: None)
    monkeypatch.setattr(vc, "_ctranslate2_device_count", lambda: pytest.fail("must not probe"))

    status = vc.detect_cuda()

    assert status.available is False
    assert status.cudnn_ok is None
    assert status.problems == []


def test_detect_cuda_falls_back_to_detect_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vc, "detect_gpu", lambda: make_gpu(cuda_version="12.6"))

    status = vc.detect_cuda(probe=lambda: 2)

    assert status.available is True
    assert status.version == "12.6"


def test_default_probe_reports_missing_ctranslate2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vc, "has_module", lambda name: False)

    with pytest.raises(DependencyMissingError) as excinfo:
        vc._ctranslate2_device_count()

    assert excinfo.value.package == "ctranslate2"


# ---------------------------------------------------------------------------
# Tier resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("vram_gb", "expected"),
    [
        (2.0, Tier.CPU),
        (5.9, Tier.CPU),
        (6.0, Tier.GPU_6),
        (7.9, Tier.GPU_6),
        (8.0, Tier.GPU_8),
        (11.9, Tier.GPU_8),
        (12.0, Tier.GPU_12),
        (15.9, Tier.GPU_12),
        (16.0, Tier.GPU_16),
        (23.9, Tier.GPU_16),
        (24.0, Tier.GPU_24),
        (48.0, Tier.GPU_24),
    ],
)
def test_resolve_tier_follows_the_vram_table(vram_gb: float, expected: Tier) -> None:
    status = vc.CudaStatus(available=True, version="12.4", cudnn_ok=True)

    assert vc.resolve_tier(make_gpu(vram_gb), status) is expected


def test_resolve_tier_without_a_gpu_is_cpu() -> None:
    assert vc.resolve_tier(None, vc.CudaStatus(available=False)) is Tier.CPU


def test_resolve_tier_ignores_vram_when_cuda_is_unavailable() -> None:
    status = vc.CudaStatus(available=False, problems=["broken"])

    assert vc.resolve_tier(make_gpu(24.0), status) is Tier.CPU


def test_broken_cudnn_does_not_downgrade_the_tier() -> None:
    """The card is still the card. The fix is printed, the tier stays honest."""
    status = vc.detect_cuda(make_gpu(8.0), probe=raising(OSError("cudnn_ops64_9.dll not found")))

    assert vc.resolve_tier(make_gpu(8.0), status) is Tier.GPU_8


# ---------------------------------------------------------------------------
# Machine facts and the config payload
# ---------------------------------------------------------------------------


def test_detect_machine_reports_this_host() -> None:
    machine = vc.detect_machine()

    assert machine.python_version == platform.python_version()
    assert machine.cpu_model
    assert machine.ram_gb >= 0.0
    assert machine.os_version


def test_build_hardware_values_matches_the_config_schema() -> None:
    gpu = make_gpu(12.0)
    status = vc.CudaStatus(available=True, version="12.4", cudnn_ok=True)
    values = vc.build_hardware_values(gpu, status, vc.detect_machine(), Tier.GPU_12)

    assert set(values) <= set(HardwareConfig.model_fields), "unknown key would be rejected"
    section = HardwareConfig(**values)
    assert section.tier is Tier.GPU_12
    assert section.gpu_vram_gb == 12.0
    assert section.gpu_model == "NVIDIA GeForce RTX 4070"
    assert section.cuda_available is True
    assert section.driver_version == "551.86"


def test_build_hardware_values_without_a_gpu() -> None:
    status = vc.CudaStatus(available=False)
    values = vc.build_hardware_values(None, status, vc.detect_machine(), Tier.CPU)

    assert values["tier"] == "cpu"
    assert values["gpu_model"] is None
    assert values["gpu_vram_gb"] is None
    assert values["cuda_available"] is False
    assert HardwareConfig(**values).tier is Tier.CPU


def test_written_hardware_drives_the_model_selection(tmp_path: Path) -> None:
    """The whole point of the script: the tier it writes picks the models."""
    from jarvis.config import write_hardware

    target = tmp_path / "config.yaml"
    status = vc.CudaStatus(available=True, version="12.4", cudnn_ok=True)
    values = vc.build_hardware_values(make_gpu(12.0), status, vc.detect_machine(), Tier.GPU_12)
    write_hardware(values, target)

    cfg = load_config(target)

    assert cfg.hardware.tier is Tier.GPU_12
    assert cfg.effective_tier() is Tier.GPU_12
    assert cfg.llm_model() == "qwen3:8b"
    engine, model, device, compute = cfg.stt_settings()
    assert model == "distil-large-v3"
    assert device == "cuda"
    assert engine.value == "faster-whisper"
    assert compute == "int8"


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def test_format_report_names_the_tier_and_its_models() -> None:
    gpu = make_gpu(24.0, name="NVIDIA GeForce RTX 4090")
    status = vc.CudaStatus(available=True, version="12.4", cudnn_ok=True)
    report = vc.format_report(gpu, status, vc.detect_machine(), Tier.GPU_24, dry_run=True)

    assert "NVIDIA GeForce RTX 4090" in report
    assert "24.0 GB" in report
    assert "551.86" in report
    assert "12.4" in report
    assert "gpu-24" in report
    assert "distil-large-v3" in report
    assert "float16" in report
    assert "qwen3:32b" in report
    assert "ACTION REQUIRED" not in report


def test_format_report_without_a_gpu_reads_as_normal() -> None:
    report = vc.format_report(None, vc.CudaStatus(available=False), vc.detect_machine(), Tier.CPU)

    assert "none detected" in report
    assert "not an error" in report
    assert "cpu" in report
    assert "base.en" in report
    assert "qwen3:4b" in report
    assert "FAILED" not in report


def test_format_report_shows_the_remediation_banner() -> None:
    status = vc.detect_cuda(make_gpu(8.0), probe=raising(OSError("cudnn_ops64_9.dll is not found")))
    report = vc.format_report(make_gpu(8.0), status, vc.detect_machine(), Tier.GPU_8)

    assert "ACTION REQUIRED" in report
    assert "cuDNN     FAILED" in report
    assert "nvidia-cudnn-cu12" in report


def test_format_report_states_where_it_wrote(tmp_path: Path) -> None:
    target = tmp_path / "config.yaml"
    report = vc.format_report(
        None, vc.CudaStatus(available=False), vc.detect_machine(), Tier.CPU, written_to=target
    )

    assert str(target) in report
    assert "dry run" not in report


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def test_parse_args_defaults() -> None:
    options = vc.parse_args([])

    assert options.dry_run is False
    assert options.as_json is False
    assert options.config_path is None


def test_parse_args_flags(tmp_path: Path) -> None:
    options = vc.parse_args(["--dry-run", "--json", "--config", str(tmp_path / "c.yaml")])

    assert options.dry_run is True
    assert options.as_json is True
    assert options.config_path == tmp_path / "c.yaml"


def test_main_dry_run_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "config.yaml"
    patch_detection(monkeypatch, make_gpu(8.0), vc.CudaStatus(available=True, cudnn_ok=True))

    assert vc.main(["--dry-run", "--config", str(target)]) == 0

    assert not target.exists()
    out = capsys.readouterr().out
    assert "dry run, nothing written" in out
    assert "gpu-8" in out


def test_main_writes_the_resolved_tier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "config.yaml"
    patch_detection(
        monkeypatch,
        make_gpu(8.0, name="NVIDIA GeForce RTX 3070"),
        vc.CudaStatus(available=True, version="12.3", cudnn_ok=True),
    )

    assert vc.main(["--config", str(target)]) == 0

    written = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert written["hardware"]["tier"] == "gpu-8"
    assert written["hardware"]["gpu_vram_gb"] == 8.0
    assert written["hardware"]["gpu_model"] == "NVIDIA GeForce RTX 3070"
    assert written["hardware"]["cuda_available"] is True
    assert load_config(target).llm_model() == "qwen3:8b"


def test_main_preserves_other_config_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "config.yaml"
    target.write_text("persona:\n  assistant_name: Jarvis\n", encoding="utf-8")
    patch_detection(monkeypatch, None, vc.CudaStatus(available=False))

    assert vc.main(["--config", str(target)]) == 0

    written = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert written["persona"]["assistant_name"] == "Jarvis"
    assert written["hardware"]["tier"] == "cpu"


def test_main_uses_the_default_config_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no --config the script writes wherever config_file_path points."""
    from jarvis.config import config_file_path

    patch_detection(monkeypatch, None, vc.CudaStatus(available=False))
    target = config_file_path()  # redirected into tmp_path by the conftest fixture
    assert not target.exists()

    assert vc.main([]) == 0

    assert target.is_file()
    assert yaml.safe_load(target.read_text(encoding="utf-8"))["hardware"]["tier"] == "cpu"


def test_main_json_output_is_parseable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    patch_detection(
        monkeypatch,
        make_gpu(24.0),
        vc.CudaStatus(available=True, version="12.4", cudnn_ok=True),
    )

    assert vc.main(["--json", "--dry-run", "--config", str(tmp_path / "c.yaml")]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["tier"] == "gpu-24"
    assert payload["gpu"]["vram_gb"] == 24.0
    assert payload["cuda"]["cudnn_ok"] is True
    assert payload["stt"]["model"] == "distil-large-v3"
    assert payload["llm_model"] == "qwen3:32b"
    assert payload["vision_supported"] is True
    assert payload["hardware"]["tier"] == "gpu-24"
    assert payload["dry_run"] is True
    assert payload["written_to"] is None
    assert payload["machine"]["python_version"] == platform.python_version()


def test_main_json_stays_parseable_when_cudnn_is_broken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    status = vc.detect_cuda(make_gpu(12.0), probe=raising(OSError("cudnn_ops64_9.dll not found")))
    patch_detection(monkeypatch, make_gpu(12.0), status)

    assert vc.main(["--json", "--dry-run", "--config", str(tmp_path / "c.yaml")]) == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["cuda"]["cudnn_ok"] is False
    assert payload["cuda"]["problems"] == [vc.CUDNN_REMEDIATION]
    assert payload["tier"] == "gpu-12", "a DLL problem must not change the tier"
    # Remediation still has to reach the user, just not on stdout.
    assert "nvidia-cudnn-cu12" in captured.err


def test_main_prints_the_cudnn_remediation_prominently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    status = vc.detect_cuda(
        make_gpu(12.0), probe=raising(OSError("[WinError 126] Could not locate cudnn_ops64_9.dll"))
    )
    patch_detection(monkeypatch, make_gpu(12.0), status)

    assert vc.main(["--config", str(tmp_path / "c.yaml")]) == 0

    out = capsys.readouterr().out
    assert "ACTION REQUIRED" in out
    assert "cudnn_ops64_9.dll" in out
    assert "nvidia-cudnn-cu12 nvidia-cublas-cu12" in out
    assert "PATH" in out


def test_main_returns_zero_with_no_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No GPU is a valid cpu tier, not a failure."""
    target = tmp_path / "config.yaml"
    monkeypatch.setattr(vc, "detect_gpu", lambda: None)
    monkeypatch.setattr(vc, "_ctranslate2_device_count", lambda: pytest.fail("must not probe"))

    assert vc.main(["--config", str(target)]) == 0

    out = capsys.readouterr().out
    assert "none detected" in out
    assert "cpu" in out
    assert yaml.safe_load(target.read_text(encoding="utf-8"))["hardware"]["tier"] == "cpu"


def test_main_fails_when_the_config_path_is_unwritable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("i am a file\n", encoding="utf-8")
    patch_detection(monkeypatch, None, vc.CudaStatus(available=False))

    assert vc.main(["--config", str(blocker / "config.yaml")]) != 0

    assert "could not write" in capsys.readouterr().err


def test_main_reports_a_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    patch_detection(monkeypatch, None, vc.CudaStatus(available=False))

    def _boom(values: dict[str, object], path: Path | None = None) -> Path:
        raise ConfigError("config.yaml is not valid YAML")

    monkeypatch.setattr(vc, "write_hardware", _boom)

    assert vc.main(["--config", str(tmp_path / "c.yaml")]) == 2
    assert "not valid YAML" in capsys.readouterr().err


def test_main_end_to_end_on_this_host(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No monkeypatching at all. Whatever this box is, the script must exit 0."""
    target = tmp_path / "config.yaml"

    assert vc.main(["--json", "--config", str(target)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["tier"] in {t.value for t in Tier if t is not Tier.AUTO}
    assert target.is_file()
    assert load_config(target).hardware.tier.value == payload["tier"]


@pytest.mark.manual
def test_gpu_host_resolves_a_gpu_tier(capsys: pytest.CaptureFixture[str]) -> None:
    """Run on the Windows target host: it must see the real card and a clean cuDNN.

    Check: ``uv run pytest tests/test_verify_cuda.py -m manual`` on the Windows box,
    with the stt extra installed so the ctranslate2 probe can actually run. Skips on
    a machine that has no NVIDIA hardware at all, since there is nothing to verify.
    """
    from jarvis.util.platform import has_cuda

    if not has_cuda():
        pytest.skip("no NVIDIA GPU on this host, run this on the Windows target")

    assert vc.main(["--json", "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["gpu"] is not None, "no NVIDIA GPU visible on the target host"
    assert payload["cuda"]["available"] is True
    assert payload["tier"] != Tier.CPU.value
    assert payload["cuda"]["cudnn_ok"] is True, payload["cuda"]["problems"]
