"""T-0.2 verification: the config loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from jarvis.config import (
    TIER_PROFILES,
    JarvisConfig,
    SttEngine,
    Tier,
    config_file_path,
    get_config,
    load_config,
    load_defaults,
    set_config,
    tier_for_vram,
    write_hardware,
)
from jarvis.util.errors import ConfigError


class TestDefaults:
    def test_loads_with_no_file_present(self, tmp_path: Path) -> None:
        cfg = load_config(tmp_path / "does-not-exist.yaml")
        assert cfg.persona.assistant_name == "Jarvis"
        assert cfg.tts.voice == "am_onyx"
        assert cfg.llm.base_url == "http://localhost:11434"

    def test_safety_defaults_are_closed(self, tmp_path: Path) -> None:
        """§6: the shell tool and web access are off until explicitly enabled."""
        cfg = load_config(tmp_path / "absent.yaml")
        assert cfg.tools.enable_shell is False
        assert cfg.shell.enabled is False
        assert cfg.tools.enable_websearch is False
        assert cfg.tools.enable_home_assistant is False
        assert cfg.gate.confirmation_timeout_s == 15.0

    def test_mutating_allowlist_matches_the_contract(self, cfg: JarvisConfig) -> None:
        """§6 names exactly six mutating tools."""
        assert sorted(cfg.gate.mutating_allowlist) == [
            "apps.focus",
            "apps.launch",
            "media.control",
            "reminders.create",
            "reminders.delete",
            "shell.run",
        ]

    def test_shell_blocklist_covers_every_named_command(self, cfg: JarvisConfig) -> None:
        required = {
            "rm", "del", "rmdir", "format", "diskpart", "shutdown", "restart",
            "reg", "regedit", "net", "netsh", "takeown", "icacls", "sc",
            "bcdedit", "vssadmin", "cipher",
        }
        assert required <= set(cfg.shell.blocked_commands)

    def test_shell_operator_blocklist_covers_every_named_operator(self, cfg: JarvisConfig) -> None:
        for operator in ("&", "&&", "|", ";", "`", "$("):
            assert operator in cfg.shell.blocked_operators


class TestYamlLoading:
    def test_reads_values_from_file(self, write_config: object) -> None:
        cfg = write_config("llm:\n  model: qwen3:14b\npersona:\n  assistant_name: Friday\n")  # type: ignore[operator]
        assert cfg.llm.model == "qwen3:14b"
        assert cfg.persona.assistant_name == "Friday"

    def test_partial_section_keeps_other_defaults(self, write_config: object) -> None:
        cfg = write_config("llm:\n  temperature: 0.1\n")  # type: ignore[operator]
        assert cfg.llm.temperature == 0.1
        assert cfg.llm.num_ctx == 8192

    def test_empty_file_is_valid(self, write_config: object) -> None:
        cfg = write_config("")  # type: ignore[operator]
        assert cfg.persona.assistant_name == "Jarvis"

    def test_malformed_yaml_raises_config_error(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("llm:\n  model: [unclosed\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="not valid YAML"):
            load_config(path)

    def test_non_mapping_top_level_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="mapping"):
            load_config(path)

    def test_unknown_key_is_rejected(self, tmp_path: Path) -> None:
        """A typo must fail loudly rather than be silently ignored."""
        path = tmp_path / "config.yaml"
        path.write_text("llm:\n  temperatur: 0.5\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(path)

    def test_example_file_is_valid(self) -> None:
        """config.example.yaml must load cleanly; it is the documented template."""
        example = Path(__file__).resolve().parents[1] / "config" / "config.example.yaml"
        assert example.is_file()
        cfg = load_config(example)
        assert cfg.persona.assistant_name == "Jarvis"
        assert cfg.tts.voice == "am_onyx"

    def test_example_file_covers_every_section(self) -> None:
        """Every top-level section in the model must appear in the template."""
        example = Path(__file__).resolve().parents[1] / "config" / "config.example.yaml"
        data = yaml.safe_load(example.read_text(encoding="utf-8"))
        assert set(JarvisConfig.model_fields) == set(data)


class TestEnvironmentOverrides:
    def test_env_beats_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("llm:\n  model: from-file\n", encoding="utf-8")
        monkeypatch.setenv("JARVIS_LLM__MODEL", "from-env")
        assert load_config(path).llm.model == "from-env"

    def test_kwargs_beat_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("JARVIS_LLM__MODEL", "from-env")
        cfg = load_config(tmp_path / "absent.yaml", llm={"model": "from-kwargs"})
        assert cfg.llm.model == "from-kwargs"

    def test_config_file_env_var_is_honoured(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "elsewhere.yaml"
        path.write_text("persona:\n  assistant_name: Elsewhere\n", encoding="utf-8")
        monkeypatch.setenv("JARVIS_CONFIG_FILE", str(path))
        assert config_file_path() == path
        assert load_config().persona.assistant_name == "Elsewhere"


class TestValidation:
    def test_rejects_out_of_range_temperature(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError):
            load_config(tmp_path / "absent.yaml", llm={"temperature": 9.0})

    def test_rejects_bad_accent_colour(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError):
            load_config(tmp_path / "absent.yaml", ui={"accent_color": "cyan"})

    def test_accepts_short_hex_colour(self, tmp_path: Path) -> None:
        cfg = load_config(tmp_path / "absent.yaml", ui={"accent_color": "#0ff"})
        assert cfg.ui.accent_color == "#0ff"

    def test_websearch_without_url_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="searxng_url"):
            load_config(tmp_path / "absent.yaml", tools={"enable_websearch": True})

    def test_home_assistant_without_url_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="home_assistant_url"):
            load_config(tmp_path / "absent.yaml", tools={"enable_home_assistant": True})

    def test_shell_enabled_without_tool_switch_is_rejected(self, tmp_path: Path) -> None:
        """Both switches must agree, so shell cannot be half-enabled by accident."""
        with pytest.raises(ConfigError, match="enable both or neither"):
            load_config(tmp_path / "absent.yaml", shell={"enabled": True})

    def test_shell_enabled_with_both_switches_is_accepted(self, tmp_path: Path) -> None:
        cfg = load_config(
            tmp_path / "absent.yaml",
            shell={"enabled": True},
            tools={"enable_shell": True},
        )
        assert cfg.shell.enabled is True

    def test_base_url_trailing_slash_is_stripped(self, tmp_path: Path) -> None:
        cfg = load_config(tmp_path / "absent.yaml", llm={"base_url": "http://host:11434/"})
        assert cfg.llm.base_url == "http://host:11434"

    def test_ring_must_hold_a_capture_block(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="ring_seconds"):
            load_config(
                tmp_path / "absent.yaml",
                audio={"block_samples": 4096, "ring_seconds": 0.5, "sample_rate": 8000},
            )

    def test_secret_is_not_printed(self, tmp_path: Path) -> None:
        cfg = load_config(
            tmp_path / "absent.yaml",
            tools={
                "enable_home_assistant": True,
                "home_assistant_url": "http://ha.local:8123",
                "home_assistant_token": "super-secret-value",
            },
        )
        assert "super-secret-value" not in repr(cfg)
        assert cfg.tools.home_assistant_token is not None
        assert cfg.tools.home_assistant_token.get_secret_value() == "super-secret-value"


class TestTiering:
    @pytest.mark.parametrize(
        ("vram", "expected"),
        [
            (None, Tier.CPU),
            (4.0, Tier.CPU),
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
            (80.0, Tier.GPU_24),
        ],
    )
    def test_tier_boundaries(self, vram: float | None, expected: Tier) -> None:
        assert tier_for_vram(vram, cuda_available=True) is expected

    def test_no_cuda_is_always_cpu_tier(self) -> None:
        assert tier_for_vram(24.0, cuda_available=False) is Tier.CPU

    def test_every_tier_has_a_profile(self) -> None:
        for tier in Tier:
            if tier is Tier.AUTO:
                continue
            assert tier in TIER_PROFILES

    def test_explicit_tier_selects_its_models(self, tmp_path: Path) -> None:
        cfg = load_config(tmp_path / "absent.yaml", hardware={"tier": "gpu-12"})
        engine, model, device, compute = cfg.stt_settings()
        assert engine is SttEngine.FASTER_WHISPER
        assert model == "distil-large-v3"
        assert device == "cuda"
        assert compute == "int8"
        assert cfg.llm_model() == "qwen3:8b"

    def test_cpu_tier_uses_whispercpp(self, tmp_path: Path) -> None:
        cfg = load_config(tmp_path / "absent.yaml", hardware={"tier": "cpu"})
        engine, model, _device, _compute = cfg.stt_settings()
        assert engine is SttEngine.WHISPERCPP
        assert model == "base.en"
        assert cfg.llm_model() == "qwen3:4b"

    def test_explicit_stt_overrides_the_tier(self, tmp_path: Path) -> None:
        cfg = load_config(
            tmp_path / "absent.yaml",
            hardware={"tier": "cpu"},
            stt={"model": "large-v3", "engine": "faster-whisper"},
        )
        engine, model, _device, _compute = cfg.stt_settings()
        assert engine is SttEngine.FASTER_WHISPER
        assert model == "large-v3"

    def test_vision_is_unavailable_on_small_tiers(self, tmp_path: Path) -> None:
        """§2: the VLM needs roughly 6 GB of free VRAM."""
        for tier in ("cpu", "gpu-6"):
            cfg = load_config(tmp_path / "absent.yaml", hardware={"tier": tier})
            assert cfg.vision_available() is False
        cfg = load_config(tmp_path / "absent.yaml", hardware={"tier": "gpu-12"})
        assert cfg.vision_available() is True

    def test_vision_switch_off_disables_it_on_any_tier(self, tmp_path: Path) -> None:
        cfg = load_config(
            tmp_path / "absent.yaml",
            hardware={"tier": "gpu-24"},
            tools={"enable_vision": False},
        )
        assert cfg.vision_available() is False


class TestPaths:
    def test_relative_paths_resolve_against_project_root(self, cfg: JarvisConfig) -> None:
        assert cfg.log_dir.is_absolute()
        assert cfg.data_dir.is_absolute()

    def test_absolute_paths_are_left_alone(self, tmp_path: Path) -> None:
        cfg = load_config(tmp_path / "absent.yaml", paths={"log_dir": str(tmp_path / "elsewhere")})
        assert cfg.log_dir == tmp_path / "elsewhere"

    def test_ensure_directories_creates_them(self, cfg: JarvisConfig) -> None:
        cfg.ensure_directories()
        assert cfg.log_dir.is_dir()
        assert cfg.data_dir.is_dir()


class TestSingleton:
    def test_get_config_is_cached(self) -> None:
        assert get_config() is get_config()

    def test_set_config_replaces_it(self, tmp_path: Path) -> None:
        replacement = load_config(tmp_path / "absent.yaml", persona={"assistant_name": "Friday"})
        set_config(replacement)
        assert get_config().persona.assistant_name == "Friday"

    def test_load_defaults_ignores_the_yaml_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("persona:\n  assistant_name: FromFile\n", encoding="utf-8")
        monkeypatch.setenv("JARVIS_CONFIG_FILE", str(path))
        assert load_config().persona.assistant_name == "FromFile"
        assert load_defaults().persona.assistant_name == "Jarvis"


class TestWriteHardware:
    def test_creates_the_file_when_absent(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "config.yaml"
        write_hardware({"tier": "gpu-12", "gpu_vram_gb": 12.0}, target)
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert data["hardware"]["tier"] == "gpu-12"
        assert data["hardware"]["gpu_vram_gb"] == 12.0

    def test_preserves_other_sections(self, tmp_path: Path) -> None:
        target = tmp_path / "config.yaml"
        target.write_text("llm:\n  model: qwen3:8b\n", encoding="utf-8")
        write_hardware({"tier": "gpu-8"}, target)
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert data["llm"]["model"] == "qwen3:8b"
        assert data["hardware"]["tier"] == "gpu-8"

    def test_skips_none_values(self, tmp_path: Path) -> None:
        target = tmp_path / "config.yaml"
        write_hardware({"tier": "cpu", "gpu_model": None}, target)
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert "gpu_model" not in data["hardware"]

    def test_result_reloads_cleanly(self, tmp_path: Path) -> None:
        target = tmp_path / "config.yaml"
        write_hardware(
            {"tier": "gpu-16", "gpu_model": "RTX 4080", "gpu_vram_gb": 16.0, "ram_gb": 32.0},
            target,
        )
        cfg = load_config(target)
        assert cfg.hardware.tier is Tier.GPU_16
        assert cfg.llm_model() == "qwen3:14b"


class TestASmallCardStillTranscribes:
    """§2: a card below 6 GB is cpu for the LLM but may still run faster-whisper.

    The cpu tier covers two different machines: one with no GPU at all, and one
    whose GPU is too small for the LLM yet perfectly able to run a small
    Whisper model. Treating them identically left a working card idle while
    whisper.cpp ground away on the processor.
    """

    @staticmethod
    def _config(**hardware: Any) -> JarvisConfig:
        base = {"tier": "cpu", "cuda_available": True, "gpu_vram_gb": 4.0}
        return load_config("/nonexistent.yaml", hardware={**base, **hardware})

    def test_a_four_gigabyte_card_transcribes_on_the_gpu(self) -> None:
        engine, model, device, compute = self._config().stt_settings()

        assert engine is SttEngine.FASTER_WHISPER
        assert model == "small.en"
        assert device == "cuda"
        assert compute == "int8"

    def test_the_llm_still_runs_on_the_processor(self) -> None:
        """The tier is about the LLM, and 4 GB cannot host it."""
        config = self._config()
        assert config.effective_tier() is Tier.CPU
        assert config.llm_model() == TIER_PROFILES[Tier.CPU].llm_model

    def test_no_gpu_still_means_whisper_cpp(self) -> None:
        engine, model, device, _ = self._config(cuda_available=False).stt_settings()

        assert engine is SttEngine.WHISPERCPP
        assert model == "base.en"
        assert device == "cpu"

    def test_a_card_too_small_even_for_whisper_stays_on_the_processor(self) -> None:
        engine, _, device, _ = self._config(gpu_vram_gb=1.0).stt_settings()

        assert engine is SttEngine.WHISPERCPP
        assert device == "cpu"

    def test_unknown_vram_is_not_assumed_to_be_enough(self) -> None:
        engine, _, _, _ = self._config(gpu_vram_gb=None).stt_settings()
        assert engine is SttEngine.WHISPERCPP

    @pytest.mark.parametrize(
        ("override", "field"),
        [
            ({"engine": "whispercpp"}, 0),
            ({"model": "tiny.en"}, 1),
            ({"device": "cpu"}, 2),
            ({"compute_type": "float32"}, 3),
        ],
    )
    def test_an_explicit_setting_always_wins(self, override: dict, field: int) -> None:
        """This upgrades a default. It must never override a deliberate choice."""
        config = load_config(
            "/nonexistent.yaml",
            hardware={"tier": "cpu", "cuda_available": True, "gpu_vram_gb": 4.0},
            stt=override,
        )
        chosen = config.stt_settings()[field]
        expected = next(iter(override.values()))
        assert str(chosen) == str(expected)

    def test_the_gpu_tiers_are_untouched(self) -> None:
        """The upgrade applies to the cpu tier only."""
        for tier in (Tier.GPU_6, Tier.GPU_8, Tier.GPU_12):
            config = load_config(
                "/nonexistent.yaml",
                hardware={"tier": str(tier), "cuda_available": True, "gpu_vram_gb": 8.0},
            )
            profile = TIER_PROFILES[tier]
            engine, model, device, compute = config.stt_settings()
            assert (engine, model, device, compute) == (
                profile.stt_engine,
                profile.stt_model,
                profile.stt_device,
                profile.stt_compute_type,
            )


class TestTheLanguageCodeFollowsTheVoice:
    """Kokoro picks its phonemiser from the language code, not the voice name.

    Both have to agree and only one of them is worth thinking about. A British
    voice given the American phonemiser mispronounces enough words to be
    obvious, and nothing anywhere reports it: no error, no warning, just a voice
    that says things slightly wrong. Deriving the code removes the mismatch
    rather than documenting it.
    """

    @pytest.mark.parametrize(
        ("voice", "code"),
        [
            ("am_onyx", "a"),
            ("am_michael", "a"),
            ("bm_george", "b"),
            ("bf_emma", "b"),
            ("jf_alpha", "j"),
            ("zm_yunxi", "z"),
            ("if_sara", "i"),
        ],
    )
    def test_it_follows_the_prefix(self, voice: str, code: str) -> None:
        assert JarvisConfig(tts={"voice": voice}).tts.resolved_lang_code() == code

    def test_an_explicit_code_still_wins(self) -> None:
        """Reading one language in another's accent is a real, if odd, choice."""
        config = JarvisConfig(tts={"voice": "am_onyx", "lang_code": "b"})
        assert config.tts.resolved_lang_code() == "b"

    def test_an_unrecognised_voice_falls_back(self) -> None:
        """A custom voice pack should synthesise, not raise."""
        assert JarvisConfig(tts={"voice": "custom"}).tts.resolved_lang_code() == "a"

    def test_the_default_voice_resolves(self) -> None:
        config = JarvisConfig()
        assert config.tts.voice == "am_onyx"
        assert config.tts.resolved_lang_code() == "a"

    def test_the_synthesiser_uses_the_resolved_code(self) -> None:
        """Reading tts.lang_code directly would now get None."""
        from jarvis.audio.tts import KokoroSynthesizer

        synth = KokoroSynthesizer(JarvisConfig(tts={"voice": "bm_george"}))
        assert synth._lang_code == "b"
