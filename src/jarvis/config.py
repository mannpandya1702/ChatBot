"""Configuration model and loader.

One pydantic tree describes every setting referenced in CLAUDE.md. Sources are
layered, later winning over earlier:

1. Field defaults, which are chosen so that JARVIS starts with no config file.
2. ``config/config.yaml`` (gitignored, machine specific).
3. Environment variables prefixed ``JARVIS_``, nested with ``__``, for example
   ``JARVIS_LLM__MODEL=qwen3:8b``.
4. Explicit keyword arguments, used by tests.

Per §0.6 nothing user specific is hardcoded here. Hardware fields default to
``None`` meaning "detect at runtime", and every path is resolved relative to the
project root discovered at import time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from jarvis.util.errors import ConfigError
from jarvis.util.platform import project_root

__all__ = [
    "TIER_PROFILES",
    "AudioConfig",
    "GateConfig",
    "HardwareConfig",
    "HelperConfig",
    "JarvisConfig",
    "LlmConfig",
    "LoggingConfig",
    "MemoryConfig",
    "PathsConfig",
    "PersonaConfig",
    "ShellConfig",
    "SttConfig",
    "SttEngine",
    "Tier",
    "TierProfile",
    "ToolsConfig",
    "TtsConfig",
    "UiConfig",
    "VadConfig",
    "WakeConfig",
    "config_file_path",
    "get_config",
    "load_config",
    "reset_config",
    "set_config",
    "tier_for_vram",
    "write_hardware",
]


# ---------------------------------------------------------------------------
# Hardware tiering (CLAUDE.md §2)
# ---------------------------------------------------------------------------


class Tier(StrEnum):
    """Hardware tier. ``AUTO`` means detect at first run."""

    AUTO = "auto"
    CPU = "cpu"
    GPU_6 = "gpu-6"
    GPU_8 = "gpu-8"
    GPU_12 = "gpu-12"
    GPU_16 = "gpu-16"
    GPU_24 = "gpu-24"


class SttEngine(StrEnum):
    """Speech to text backend."""

    AUTO = "auto"
    FASTER_WHISPER = "faster-whisper"
    WHISPERCPP = "whispercpp"


@dataclass(frozen=True, slots=True)
class TierProfile:
    """The model choices a tier implies."""

    tier: Tier
    stt_engine: SttEngine
    stt_model: str
    stt_compute_type: str
    stt_device: str
    llm_model: str
    #: The vision model needs roughly 6 GB of free VRAM (§2).
    supports_vision: bool


TIER_PROFILES: dict[Tier, TierProfile] = {
    Tier.CPU: TierProfile(
        tier=Tier.CPU,
        stt_engine=SttEngine.WHISPERCPP,
        stt_model="base.en",
        stt_compute_type="int8",
        stt_device="cpu",
        llm_model="qwen3:4b",
        supports_vision=False,
    ),
    Tier.GPU_6: TierProfile(
        tier=Tier.GPU_6,
        stt_engine=SttEngine.FASTER_WHISPER,
        stt_model="small.en",
        stt_compute_type="int8",
        stt_device="cuda",
        llm_model="qwen3:4b",
        supports_vision=False,
    ),
    Tier.GPU_8: TierProfile(
        tier=Tier.GPU_8,
        stt_engine=SttEngine.FASTER_WHISPER,
        stt_model="small.en",
        stt_compute_type="int8",
        stt_device="cuda",
        llm_model="qwen3:8b",
        supports_vision=True,
    ),
    Tier.GPU_12: TierProfile(
        tier=Tier.GPU_12,
        stt_engine=SttEngine.FASTER_WHISPER,
        stt_model="distil-large-v3",
        stt_compute_type="int8",
        stt_device="cuda",
        llm_model="qwen3:8b",
        supports_vision=True,
    ),
    Tier.GPU_16: TierProfile(
        tier=Tier.GPU_16,
        stt_engine=SttEngine.FASTER_WHISPER,
        stt_model="distil-large-v3",
        stt_compute_type="int8",
        stt_device="cuda",
        llm_model="qwen3:14b",
        supports_vision=True,
    ),
    Tier.GPU_24: TierProfile(
        tier=Tier.GPU_24,
        stt_engine=SttEngine.FASTER_WHISPER,
        stt_model="distil-large-v3",
        stt_compute_type="float16",
        stt_device="cuda",
        llm_model="qwen3:32b",
        supports_vision=True,
    ),
}


def tier_for_vram(vram_gb: float | None, *, cuda_available: bool) -> Tier:
    """Resolve a hardware tier from detected VRAM.

    A card under 6 GB cannot hold a useful LLM alongside the STT model, so it maps
    to ``cpu`` even when CUDA is present.
    """
    if not cuda_available or vram_gb is None or vram_gb < 6.0:
        return Tier.CPU
    if vram_gb < 8.0:
        return Tier.GPU_6
    if vram_gb < 12.0:
        return Tier.GPU_8
    if vram_gb < 16.0:
        return Tier.GPU_12
    if vram_gb < 24.0:
        return Tier.GPU_16
    return Tier.GPU_24


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


class _Section(BaseModel):
    """Base for config sections. Unknown keys are an error, not a silent no-op."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PathsConfig(_Section):
    """Directories. Relative values resolve against the project root."""

    data_dir: Path = Path("data")
    log_dir: Path = Path("logs")
    models_dir: Path = Path("models")
    vendor_dir: Path = Path("vendor")


class HardwareConfig(_Section):
    """Detected machine facts. ``None`` means "not detected yet"."""

    tier: Tier = Tier.AUTO
    gpu_model: str | None = None
    gpu_vram_gb: float | None = Field(default=None, ge=0)
    cpu_model: str | None = None
    ram_gb: float | None = Field(default=None, ge=0)
    windows_version: str | None = None
    cuda_available: bool | None = None
    cuda_version: str | None = None
    driver_version: str | None = None


class AudioConfig(_Section):
    """Capture and playback. ``None`` device means the system default."""

    input_device: str | int | None = None
    output_device: str | int | None = None
    sample_rate: int = Field(default=16_000, ge=8_000, le=48_000)
    channels: Literal[1] = 1
    #: Capture block. 512 samples at 16 kHz is 32 ms, one Silero VAD frame.
    block_samples: int = Field(default=512, ge=128, le=4_096)
    #: Circular buffer depth. Two seconds of pre-roll (§7 T-1.1).
    ring_seconds: float = Field(default=2.0, ge=0.5, le=30.0)
    #: Playback rate for Kokoro output.
    output_sample_rate: int = Field(default=24_000, ge=8_000, le=48_000)
    dtype: Literal["float32", "int16"] = "float32"


class WakeConfig(_Section):
    """openWakeWord settings (§1)."""

    enabled: bool = True
    model: str = "hey_jarvis"
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    #: Seconds to ignore further detections after a hit.
    cooldown_s: float = Field(default=2.0, ge=0.0, le=30.0)
    #: Audio handed to STT ahead of the trigger so the trailing word survives.
    preroll_s: float = Field(default=1.0, ge=0.0, le=5.0)
    #: openWakeWord consumes 80 ms frames at 16 kHz.
    frame_samples: int = Field(default=1_280, ge=160, le=8_000)
    #: openWakeWord's own speech gate, reduces false accepts from noise.
    vad_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    inference_framework: Literal["onnx", "tflite"] = "onnx"


class VadConfig(_Section):
    """Silero VAD endpointing (§7 T-1.3)."""

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    #: Speech shorter than this is treated as a noise blip.
    min_speech_ms: int = Field(default=250, ge=0, le=5_000)
    #: Silence after speech that closes the utterance.
    trailing_silence_ms: int = Field(default=500, ge=100, le=5_000)
    #: Hard cap so a stuck endpoint cannot hang the turn.
    max_utterance_s: float = Field(default=30.0, ge=1.0, le=300.0)
    #: Audio kept either side of detected speech.
    speech_pad_ms: int = Field(default=100, ge=0, le=1_000)
    #: Silero v5 requires exactly 512 samples at 16 kHz.
    frame_samples: Literal[512, 256] = 512
    #: Threshold for the barge-in detector during playback. Higher than the main
    #: threshold so the assistant's own leaked audio does not interrupt itself.
    barge_in_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    barge_in_min_speech_ms: int = Field(default=200, ge=0, le=5_000)


class SttConfig(_Section):
    """Transcription. ``None`` fields fall back to the tier profile."""

    engine: SttEngine = SttEngine.AUTO
    model: str | None = None
    device: str | None = None
    compute_type: str | None = None
    language: str = "en"
    beam_size: int = Field(default=1, ge=1, le=10)
    #: faster-whisper's own VAD is redundant, we endpoint with Silero already.
    vad_filter: bool = False
    condition_on_previous_text: bool = False
    #: Utterances shorter than this are discarded as accidental triggers.
    min_audio_ms: int = Field(default=200, ge=0, le=5_000)
    download_root: Path | None = None


class TtsConfig(_Section):
    """Kokoro synthesis (§1)."""

    enabled: bool = True
    voice: str = "bm_george"
    #: Kokoro language code. "b" is British English, which matches bm_* voices.
    lang_code: str = "b"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    sample_rate: int = Field(default=24_000, ge=8_000, le=48_000)
    #: Smallest chunk worth synthesising. Below this we wait for more text so the
    #: first sentence does not come out clipped.
    min_chunk_chars: int = Field(default=24, ge=1, le=500)
    #: Force a flush once a chunk gets this long even without terminal punctuation.
    max_chunk_chars: int = Field(default=240, ge=20, le=2_000)
    #: Playback must die within this many milliseconds of stop() (§7 T-1.5).
    stop_latency_ms: int = Field(default=100, ge=10, le=1_000)


class LlmConfig(_Section):
    """Ollama client (§1)."""

    base_url: str = "http://localhost:11434"
    model: str | None = None
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    num_ctx: int = Field(default=8_192, ge=512, le=131_072)
    num_predict: int = Field(default=512, ge=16, le=8_192)
    keep_alive: str = "10m"
    request_timeout_s: float = Field(default=120.0, ge=1.0)
    connect_timeout_s: float = Field(default=5.0, ge=0.1)
    #: Ceiling on tool call rounds inside one turn, prevents runaway loops.
    max_tool_iterations: int = Field(default=4, ge=1, le=10)
    #: Retries when the model emits malformed tool JSON (§7 T-1.6).
    tool_json_retries: int = Field(default=2, ge=0, le=5)
    #: Qwen3 thinking mode costs time to first token, off by default.
    think: bool = False

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")


class MemoryConfig(_Section):
    """Rolling conversation memory (§7 T-1.7)."""

    db_path: Path = Path("data/memory.db")
    #: Compact into a summary once the window exceeds this many tokens.
    max_tokens: int = Field(default=3_000, ge=256, le=100_000)
    #: Hard cap on retained messages regardless of token count.
    max_messages: int = Field(default=40, ge=2, le=500)
    #: How many recent messages survive a compaction verbatim.
    keep_recent: int = Field(default=8, ge=2, le=100)
    #: Model used to write the running summary. None means the main model.
    summary_model: str | None = None
    enabled: bool = True
    session_id: str = "default"


class PersonaConfig(_Section):
    """System prompt inputs (§7 T-1.8)."""

    assistant_name: str = "Jarvis"
    wake_word: str = "hey_jarvis"
    #: How the assistant addresses the user. Empty string means no form of address.
    user_address_form: str = "sir"
    response_style: str = "terse, one or two spoken sentences unless asked to elaborate"
    #: Appended verbatim to the system prompt.
    extra_instructions: str | None = None


class UiConfig(_Section):
    """WebSocket broadcaster and HUD (§7 Phase 3)."""

    enabled: bool = True
    host: str = "127.0.0.1"
    #: 0 asks the OS for any free port, which tests use to avoid collisions.
    port: int = Field(default=8765, ge=0, le=65_535)
    broadcast_hz: float = Field(default=30.0, ge=1.0, le=120.0)
    hud_position: Literal[
        "top-left", "top-center", "top-right",
        "bottom-left", "bottom-center", "bottom-right",
    ] = "bottom-right"
    accent_color: str = "#22d3ee"
    #: Sparkline history depth in the HUD.
    metrics_history: int = Field(default=60, ge=10, le=600)

    @field_validator("accent_color")
    @classmethod
    def _validate_hex(cls, value: str) -> str:
        if not value.startswith("#") or len(value) not in (4, 7):
            msg = f"accent_color must be a hex colour like #22d3ee, got {value!r}"
            raise ValueError(msg)
        int(value[1:], 16)  # raises ValueError when not hex
        return value


class HelperConfig(_Section):
    """Elevated sidecar (§6 privilege separation)."""

    enabled: bool = True
    pipe_name: str = "jarvis-helper"
    exe_path: Path | None = None
    #: Results are cached this long so we do not hammer the sensors (§7 T-2.8).
    cache_ttl_s: float = Field(default=5.0, ge=0.0, le=300.0)
    autostart: bool = True
    connect_timeout_s: float = Field(default=3.0, ge=0.1, le=60.0)
    request_timeout_s: float = Field(default=10.0, ge=0.1, le=120.0)


class GateConfig(_Section):
    """Confirmation gate for mutating tools (§6)."""

    #: Timeout equals denial. No exceptions.
    confirmation_timeout_s: float = Field(default=15.0, ge=1.0, le=120.0)
    affirmatives: list[str] = Field(
        default_factory=lambda: [
            "yes", "yeah", "yep", "yup", "confirm", "confirmed", "do it",
            "go ahead", "affirmative", "please do", "proceed", "ok do it",
        ]
    )
    negatives: list[str] = Field(
        default_factory=lambda: [
            "no", "nope", "cancel", "stop", "abort", "negative", "never mind", "nevermind",
        ]
    )
    #: The only tools permitted to change system state (§6).
    mutating_allowlist: list[str] = Field(
        default_factory=lambda: [
            "apps.launch",
            "apps.focus",
            "media.control",
            "reminders.create",
            "reminders.delete",
            "shell.run",
        ]
    )
    audit_log: Path = Path("logs/gate_audit.jsonl")


class ShellConfig(_Section):
    """``shell.run`` hardening (§6). Every field here is a safety control."""

    enabled: bool = False
    #: Allowlist first. A command not on this list is refused outright.
    allowlist: list[str] = Field(
        default_factory=lambda: [
            "Get-Date", "Get-Uptime", "Get-ComputerInfo", "Get-Process",
            "Get-Service", "Get-Volume", "Get-PSDrive", "Get-NetAdapter",
            "Get-HotFix", "Get-WindowsUpdateLog", "systeminfo", "ipconfig",
            "hostname", "whoami", "tasklist", "ver",
        ]
    )
    #: Secondary layer only (§6). The allowlist is the primary control.
    blocked_commands: list[str] = Field(
        default_factory=lambda: [
            "rm", "del", "rmdir", "format", "diskpart", "shutdown", "restart",
            "reg", "regedit", "net", "netsh", "takeown", "icacls", "sc",
            "bcdedit", "vssadmin", "cipher",
        ]
    )
    blocked_operators: list[str] = Field(
        default_factory=lambda: ["&", "&&", "|", "||", ";", "`", "$(", ">", ">>", "<", "\n", "\r"]
    )
    blocked_powershell_args: list[str] = Field(
        default_factory=lambda: [
            "-enc", "-encodedcommand", "-command", "-e", "-ec", "-nop", "-noprofile",
            "-windowstyle", "-executionpolicy", "-file",
        ]
    )
    max_command_length: int = Field(default=512, ge=1, le=8_192)
    timeout_s: float = Field(default=30.0, ge=1.0, le=300.0)
    max_output_bytes: int = Field(default=4_096, ge=256, le=1_048_576)
    #: Empty means no working directory may be selected at all.
    working_dir_allowlist: list[Path] = Field(default_factory=list)
    audit_log: Path = Path("logs/shell_audit.jsonl")


class ToolsConfig(_Section):
    """Feature switches for the tool surface (§9)."""

    enable_shell: bool = False
    enable_vision: bool = True
    enable_websearch: bool = False
    enable_home_assistant: bool = False
    searxng_url: str | None = None
    home_assistant_url: str | None = None
    home_assistant_token: SecretStr | None = None
    everything_cli_path: Path | None = None
    vision_model: str = "qwen3-vl:8b"
    vision_fallback_model: str = "moondream"
    #: Spoken answers stay short, so list results are capped (§5).
    max_results: int = Field(default=5, ge=1, le=20)
    #: Sampling window for rate-style metrics such as network throughput.
    sample_interval_s: float = Field(default=0.5, ge=0.05, le=5.0)
    reminders_db_path: Path = Path("data/reminders.db")

    @model_validator(mode="after")
    def _check_dependent_settings(self) -> ToolsConfig:
        if self.enable_websearch and not self.searxng_url:
            msg = "tools.enable_websearch is true but tools.searxng_url is not set"
            raise ValueError(msg)
        if self.enable_home_assistant and not self.home_assistant_url:
            msg = "tools.enable_home_assistant is true but tools.home_assistant_url is not set"
            raise ValueError(msg)
        return self


class LoggingConfig(_Section):
    """Logging (§7 T-0.3)."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    console: bool = True
    console_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None = None
    json_file: str = "jarvis.jsonl"
    max_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    backup_count: int = Field(default=5, ge=0, le=50)


class LatencyConfig(_Section):
    """Latency instrumentation (§3)."""

    enabled: bool = True
    #: Budgets in milliseconds, keyed by ``jarvis.util.latency.Stage`` values.
    budgets_ms: dict[str, float] = Field(
        default_factory=lambda: {
            "vad_endpoint": 250.0,
            "stt": 200.0,
            "llm_first_token": 400.0,
            "tts_first_audio": 300.0,
            "turn_total": 1200.0,
        }
    )
    #: Fail the benchmark when p95 regresses by more than this fraction (§7 T-5.1).
    regression_tolerance: float = Field(default=0.15, ge=0.0, le=1.0)
    results_path: Path = Path("bench/results.json")


class OrchestratorConfig(_Section):
    """Turn loop behaviour (§7 T-1.9)."""

    #: Seconds of silence after a turn before returning to wake-word mode.
    idle_timeout_s: float = Field(default=30.0, ge=1.0, le=600.0)
    #: Allow the user to interrupt playback by speaking.
    barge_in: bool = True
    #: Skip the wake word and listen continuously. Useful for desk testing.
    always_listening: bool = False
    #: Greeting spoken on start. Empty disables it.
    greeting: str = ""
    max_turn_seconds: float = Field(default=120.0, ge=5.0, le=600.0)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


def config_file_path() -> Path:
    """Path to ``config.yaml``. ``JARVIS_CONFIG_FILE`` overrides the default."""
    override = os.environ.get("JARVIS_CONFIG_FILE")
    if override:
        return Path(override).expanduser()
    return project_root() / "config" / "config.yaml"


class _YamlSettingsSource(PydanticBaseSettingsSource):
    """Feeds ``config.yaml`` into pydantic-settings."""

    def __init__(self, settings_cls: type[BaseSettings], path: Path | None) -> None:
        super().__init__(settings_cls)
        self._path = path
        self._data: dict[str, Any] = {}
        if path is not None and path.is_file():
            try:
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                raise ConfigError(
                    f"{path} is not valid YAML: {exc}",
                    context={"path": str(path)},
                ) from exc
            except OSError as exc:
                raise ConfigError(
                    f"could not read {path}: {exc}",
                    context={"path": str(path)},
                ) from exc
            if raw is None:
                raw = {}
            if not isinstance(raw, dict):
                raise ConfigError(
                    f"{path} must contain a YAML mapping at the top level, "
                    f"got {type(raw).__name__}",
                    context={"path": str(path)},
                )
            self._data = raw

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return self._data


class JarvisConfig(BaseSettings):
    """Root configuration object."""

    model_config = SettingsConfigDict(
        env_prefix="JARVIS_",
        env_nested_delimiter="__",
        extra="forbid",
        validate_assignment=True,
        nested_model_default_partial_update=True,
    )

    paths: PathsConfig = Field(default_factory=PathsConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    wake: WakeConfig = Field(default_factory=WakeConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    ui: UiConfig = Field(default_factory=UiConfig)
    helper: HelperConfig = Field(default_factory=HelperConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    shell: ShellConfig = Field(default_factory=ShellConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    latency: LatencyConfig = Field(default_factory=LatencyConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)

    # -- source layering ---------------------------------------------------

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        path = _yaml_path_override if _yaml_path_override is not _UNSET else config_file_path()
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _YamlSettingsSource(settings_cls, path),
            file_secret_settings,
        )

    # -- derived values ----------------------------------------------------

    @model_validator(mode="after")
    def _cross_section_checks(self) -> JarvisConfig:
        if self.shell.enabled and not self.tools.enable_shell:
            msg = "shell.enabled is true but tools.enable_shell is false; enable both or neither"
            raise ValueError(msg)
        if self.audio.block_samples > self.audio.sample_rate * self.audio.ring_seconds:
            msg = "audio.ring_seconds is too small to hold a single capture block"
            raise ValueError(msg)
        return self

    def resolve_path(self, value: Path) -> Path:
        """Absolute path for ``value``, relative paths resolving to the project root."""
        return value if value.is_absolute() else (project_root() / value)

    @property
    def log_dir(self) -> Path:
        """Absolute log directory."""
        return self.resolve_path(self.paths.log_dir)

    @property
    def data_dir(self) -> Path:
        """Absolute data directory."""
        return self.resolve_path(self.paths.data_dir)

    @property
    def models_dir(self) -> Path:
        """Absolute models directory."""
        return self.resolve_path(self.paths.models_dir)

    @property
    def vendor_dir(self) -> Path:
        """Absolute vendor directory."""
        return self.resolve_path(self.paths.vendor_dir)

    def effective_tier(self) -> Tier:
        """The tier to build model choices from. ``AUTO`` resolves via detection."""
        if self.hardware.tier is not Tier.AUTO:
            return self.hardware.tier
        from jarvis.util.platform import has_cuda  # local import, keeps startup cheap

        return tier_for_vram(self.hardware.gpu_vram_gb, cuda_available=has_cuda())

    def tier_profile(self) -> TierProfile:
        """Model profile for the effective tier."""
        return TIER_PROFILES[self.effective_tier()]

    def stt_settings(self) -> tuple[SttEngine, str, str, str]:
        """Resolved ``(engine, model, device, compute_type)`` for STT."""
        profile = self.tier_profile()
        engine = self.stt.engine if self.stt.engine is not SttEngine.AUTO else profile.stt_engine
        return (
            engine,
            self.stt.model or profile.stt_model,
            self.stt.device or profile.stt_device,
            self.stt.compute_type or profile.stt_compute_type,
        )

    def llm_model(self) -> str:
        """Resolved LLM model name."""
        return self.llm.model or self.tier_profile().llm_model

    def vision_available(self) -> bool:
        """True when the tier can host the VLM and the tool is enabled."""
        return self.tools.enable_vision and self.tier_profile().supports_vision

    def ensure_directories(self) -> None:
        """Create the runtime directories. Safe to call repeatedly."""
        for directory in (self.log_dir, self.data_dir, self.models_dir):
            directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_UNSET = object()
#: Set by :func:`load_config` so the settings source knows which file to read.
_yaml_path_override: Any = _UNSET
_active: JarvisConfig | None = None


def load_config(path: Path | str | None = None, **overrides: Any) -> JarvisConfig:
    """Build a config from defaults, ``config.yaml``, the environment, and overrides.

    Args:
        path: Explicit YAML file. ``None`` uses :func:`config_file_path`. Pass
            ``False``-like ``Path("")`` is not supported; use ``skip_yaml``.
        **overrides: Highest-priority values, keyed by section name.

    Raises:
        ConfigError: The file is unreadable, malformed, or fails validation.
    """
    global _yaml_path_override  # noqa: PLW0603 - module-level source selection
    previous = _yaml_path_override
    _yaml_path_override = Path(path) if path is not None else config_file_path()
    try:
        return JarvisConfig(**overrides)
    except ConfigError:
        raise
    except Exception as exc:  # pydantic ValidationError and anything it wraps
        raise ConfigError(
            f"configuration is invalid: {exc}",
            speakable="My configuration file has an error, so I cannot start.",
            context={"path": str(_yaml_path_override)},
        ) from exc
    finally:
        _yaml_path_override = previous


def load_defaults(**overrides: Any) -> JarvisConfig:
    """Build a config ignoring ``config.yaml`` entirely. Used by tests."""
    global _yaml_path_override  # noqa: PLW0603
    previous = _yaml_path_override
    _yaml_path_override = None
    try:
        return JarvisConfig(**overrides)
    except Exception as exc:
        raise ConfigError(f"configuration is invalid: {exc}") from exc
    finally:
        _yaml_path_override = previous


def get_config() -> JarvisConfig:
    """Return the process-wide config, loading it on first use."""
    global _active  # noqa: PLW0603 - deliberate singleton
    if _active is None:
        _active = load_config()
    return _active


def set_config(config: JarvisConfig) -> None:
    """Replace the process-wide config. Used by tests and by ``main``."""
    global _active  # noqa: PLW0603
    _active = config


def reset_config() -> None:
    """Drop the cached config so the next :func:`get_config` reloads."""
    global _active  # noqa: PLW0603
    _active = None


def write_hardware(values: dict[str, Any], path: Path | None = None) -> Path:
    """Merge detected hardware facts into ``config.yaml``.

    Used by ``scripts/verify_cuda.py`` to persist the resolved tier. Creates the
    file when absent. Only the ``hardware`` section is touched; other sections
    are round-tripped unchanged.

    Note: YAML comments are not preserved by this rewrite. The committed template
    at ``config/config.example.yaml`` remains the documented reference.
    """
    target = path or config_file_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, Any] = {}
    if target.is_file():
        try:
            loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
        except (yaml.YAMLError, OSError) as exc:
            raise ConfigError(f"could not read {target}: {exc}") from exc
        if isinstance(loaded, dict):
            existing = loaded

    hardware = dict(existing.get("hardware") or {})
    hardware.update({k: v for k, v in values.items() if v is not None})
    existing["hardware"] = hardware

    target.write_text(
        yaml.safe_dump(existing, sort_keys=False, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target
