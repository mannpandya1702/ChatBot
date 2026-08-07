"""T-0.3 verification: logging, latency, errors, and platform detection."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from jarvis.tools.registry import ToolCategory as _ToolCategory
from jarvis.tools.registry import ToolInput as _ToolInput
from jarvis.tools.registry import ToolOutput as _ToolOutput
from jarvis.util import platform as plat
from jarvis.util.errors import (
    ConfigError,
    DependencyMissingError,
    JarvisError,
    PlatformUnsupportedError,
    ToolExecutionError,
    as_speakable,
)
from jarvis.util.latency import (
    DEFAULT_BUDGETS_MS,
    LatencyStats,
    Stage,
    StageTiming,
    TurnLatency,
    percentile,
)
from jarvis.util.logging import ConsoleFormatter, JsonFormatter, get_logger, setup_logging


class TestErrors:
    def test_base_error_carries_a_speakable(self) -> None:
        err = JarvisError("internal detail")
        assert "internal detail" in str(err)
        assert err.speakable == "Something went wrong on my end."

    def test_speakable_can_be_overridden(self) -> None:
        err = ConfigError("bad key", speakable="My settings are wrong.")
        assert err.speakable == "My settings are wrong."

    def test_dependency_error_names_the_install_command(self) -> None:
        err = DependencyMissingError("kokoro", extra="tts")
        assert "uv sync --extra tts" in err.message
        assert err.package == "kokoro"

    def test_platform_error_names_the_requirement(self) -> None:
        err = PlatformUnsupportedError("fan speeds")
        assert "Windows" in err.speakable
        assert err.context["feature"] == "fan speeds"

    def test_tool_error_records_the_tool(self) -> None:
        err = ToolExecutionError("sys.gpu", "NVML init failed")
        assert err.tool_name == "sys.gpu"
        assert err.context["tool"] == "sys.gpu"
        assert err.speakable == "I could not read that sensor."

    def test_to_dict_is_json_serialisable(self) -> None:
        payload = ToolExecutionError("sys.cpu", "boom").to_dict()
        assert json.loads(json.dumps(payload))["error"] == "ToolExecutionError"

    def test_as_speakable_never_leaks_a_foreign_message(self) -> None:
        """§5: an unexpected exception must not have its text spoken aloud."""
        leaky = RuntimeError("C:/Users/mann/secret-path/token=abc123")
        spoken = as_speakable(leaky)
        assert "secret-path" not in spoken
        assert "abc123" not in spoken
        assert spoken == "Something went wrong on my end."

    def test_as_speakable_uses_jarvis_errors(self) -> None:
        assert as_speakable(ToolExecutionError("t", "x")) == "I could not read that sensor."


class TestJsonLogging:
    def test_writes_one_json_object_per_line(self, tmp_path: Path) -> None:
        setup_logging(level="DEBUG", log_dir=tmp_path, console=False, force=True)
        log = get_logger("jarvis.test")
        log.info("hello", extra={"context": {"cpu": 43.2}})
        log.warning("careful")

        lines = (tmp_path / "jarvis.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["message"] == "hello"
        assert first["level"] == "INFO"
        assert first["context"]["cpu"] == 43.2
        assert json.loads(lines[1])["level"] == "WARNING"

    def test_captures_tracebacks(self, tmp_path: Path) -> None:
        setup_logging(level="DEBUG", log_dir=tmp_path, console=False, force=True)
        log = get_logger("jarvis.test")
        try:
            raise ValueError("kaboom")
        except ValueError:
            log.exception("tool raised")
        record = json.loads((tmp_path / "jarvis.jsonl").read_text(encoding="utf-8").strip())
        assert "ValueError: kaboom" in record["exception"]

    def test_unserialisable_context_does_not_break_the_line(self, tmp_path: Path) -> None:
        setup_logging(level="DEBUG", log_dir=tmp_path, console=False, force=True)
        get_logger("jarvis.test").info("odd", extra={"context": {"obj": object()}})
        line = (tmp_path / "jarvis.jsonl").read_text(encoding="utf-8").strip()
        assert json.loads(line)["context"]["obj"].startswith("<object")

    def test_respects_the_level(self, tmp_path: Path) -> None:
        setup_logging(level="WARNING", log_dir=tmp_path, console=False, force=True)
        log = get_logger("jarvis.test")
        log.debug("invisible")
        log.error("visible")
        lines = (tmp_path / "jarvis.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

    def test_formatter_directly(self) -> None:
        record = logging.LogRecord("jarvis.x", logging.INFO, __file__, 1, "msg", None, None)
        payload = json.loads(JsonFormatter().format(record))
        assert payload["logger"] == "jarvis.x"
        assert "ts" in payload

    def test_console_formatter_is_plain_without_colour(self) -> None:
        record = logging.LogRecord("jarvis.x", logging.INFO, __file__, 1, "msg", None, None)
        line = ConsoleFormatter(color=False).format(record)
        assert "\033[" not in line
        assert "msg" in line

    def test_get_logger_namespaces_foreign_names(self) -> None:
        assert get_logger("something").name == "jarvis.something"
        assert get_logger("jarvis.audio.ring").name == "jarvis.audio.ring"

    def test_setup_is_idempotent(self, tmp_path: Path) -> None:
        first = setup_logging(log_dir=tmp_path, console=False, force=True)
        before = len(first.handlers)
        setup_logging(log_dir=tmp_path, console=False)
        assert len(first.handlers) == before


class TestPercentile:
    def test_empty_is_zero(self) -> None:
        assert percentile([], 95) == 0.0

    def test_single_value(self) -> None:
        assert percentile([42.0], 95) == 42.0

    def test_known_values(self) -> None:
        data = [float(n) for n in range(1, 101)]
        assert percentile(data, 50) == pytest.approx(50.5)
        assert percentile(data, 95) == pytest.approx(95.05)
        assert percentile(data, 0) == 1.0
        assert percentile(data, 100) == 100.0

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 100"):
            percentile([1.0], 101)


class TestTurnLatency:
    def test_stage_records_duration(self) -> None:
        clock = _FakeClock()
        turn = TurnLatency(clock=clock)
        with turn.stage(Stage.STT):
            clock.advance(0.15)
        timing = turn.get(Stage.STT)
        assert timing is not None
        assert timing.duration_ms == pytest.approx(150.0)

    def test_stage_applies_the_budget(self) -> None:
        clock = _FakeClock()
        turn = TurnLatency(clock=clock)
        with turn.stage(Stage.STT):
            clock.advance(0.5)
        timing = turn.get(Stage.STT)
        assert timing is not None
        assert timing.budget_ms == DEFAULT_BUDGETS_MS[Stage.STT]
        assert timing.over_budget is True

    def test_stage_records_even_when_the_body_raises(self) -> None:
        turn = TurnLatency()
        with pytest.raises(RuntimeError), turn.stage(Stage.LLM_TOTAL):
            raise RuntimeError("boom")
        assert turn.get(Stage.LLM_TOTAL) is not None

    def test_metadata_survives(self) -> None:
        turn = TurnLatency()
        with turn.stage(Stage.STT, model="small.en") as timing:
            timing.metadata["chars"] = 12
        recorded = turn.get(Stage.STT)
        assert recorded is not None
        assert recorded.metadata == {"model": "small.en", "chars": 12}

    def test_mark_measures_from_turn_start(self) -> None:
        clock = _FakeClock()
        turn = TurnLatency(clock=clock)
        clock.advance(0.3)
        timing = turn.mark(Stage.LLM_FIRST_TOKEN)
        assert timing.duration_ms == pytest.approx(300.0)

    def test_tool_use_widens_the_turn_budget(self) -> None:
        """§3: a tool call adds an LLM round trip, so the budget goes to 2000 ms."""
        turn = TurnLatency()
        assert turn.total_budget_ms() == 1200.0
        turn.record(Stage.TOOL, 5.0)
        assert turn.total_budget_ms() == 2000.0

    def test_finish_records_the_total(self) -> None:
        clock = _FakeClock()
        turn = TurnLatency(clock=clock)
        clock.advance(1.0)
        total = turn.finish()
        assert total.stage is Stage.TURN_TOTAL
        assert total.duration_ms == pytest.approx(1000.0)
        assert total.over_budget is False

    def test_breakdown_lists_over_budget_stages(self) -> None:
        turn = TurnLatency()
        turn.record(Stage.STT, 900.0)
        turn.record(Stage.TTS_FIRST_AUDIO, 10.0)
        assert turn.breakdown()["over_budget"] == ["stt"]

    def test_log_warns_when_over_budget(self, caplog: pytest.LogCaptureFixture) -> None:
        turn = TurnLatency()
        turn.record(Stage.STT, 900.0)
        with caplog.at_level(logging.INFO):
            turn.log(logging.getLogger("jarvis.test.latency"))
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_timing_to_dict_is_serialisable(self) -> None:
        payload = StageTiming(Stage.STT, 10.0, 200.0, {"x": 1}).to_dict()
        assert json.loads(json.dumps(payload))["stage"] == "stt"


class TestLatencyStats:
    def test_aggregates_percentiles(self) -> None:
        stats = LatencyStats()
        for value in range(1, 101):
            stats.add(Stage.STT, float(value))
        row = stats.summary()["stt"]
        assert row["count"] == 100
        assert row["p50_ms"] == pytest.approx(50.5)
        assert row["min_ms"] == 1.0
        assert row["max_ms"] == 100.0

    def test_flags_budget_failures(self) -> None:
        stats = LatencyStats()
        for _ in range(20):
            stats.add(Stage.TTS_FIRST_AUDIO, 900.0)
        assert stats.summary()["tts_first_audio"]["within_budget"] is False
        assert stats.failing_stages() == ["tts_first_audio"]

    def test_passing_stage_is_not_flagged(self) -> None:
        stats = LatencyStats()
        for _ in range(20):
            stats.add(Stage.TTS_FIRST_AUDIO, 50.0)
        assert stats.failing_stages() == []

    def test_add_turn_folds_every_stage(self) -> None:
        turn = TurnLatency()
        turn.record(Stage.STT, 10.0)
        turn.record(Stage.LLM_TOTAL, 20.0)
        stats = LatencyStats()
        stats.add_turn(turn)
        assert stats.count(Stage.STT) == 1
        assert stats.count(Stage.LLM_TOTAL) == 1

    def test_summary_is_serialisable(self) -> None:
        stats = LatencyStats()
        stats.add(Stage.STT, 1.0)
        json.dumps(stats.summary())


class TestPlatform:
    def test_exactly_one_platform_matches(self) -> None:
        assert sum([plat.is_windows(), plat.is_linux(), plat.is_macos()]) <= 1

    def test_has_module_finds_a_real_module(self) -> None:
        assert plat.has_module("json") is True

    def test_has_module_is_false_for_nonsense(self) -> None:
        assert plat.has_module("definitely_not_a_real_module_xyz") is False

    def test_require_module_returns_the_module(self) -> None:
        assert plat.require_module("json").dumps({}) == "{}"

    def test_require_module_raises_with_a_hint(self) -> None:
        with pytest.raises(DependencyMissingError) as excinfo:
            plat.require_module("definitely_not_a_real_module_xyz")
        assert "definitely_not_a_real_module_xyz" in str(excinfo.value)

    def test_require_module_maps_known_packages_to_extras(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mapping is what matters, not whether the extra happens to be installed.

        Asserting that importing kokoro fails only held on a machine without the
        tts extra, so the test broke the moment anyone installed the engines it
        exists to describe. The import is forced to fail instead.
        """

        def _absent(name: str) -> object:
            raise ImportError(f"No module named {name!r}")

        monkeypatch.setattr(plat.importlib, "import_module", _absent)

        with pytest.raises(DependencyMissingError) as excinfo:
            plat.require_module("kokoro")
        assert excinfo.value.extra == "tts"

    @pytest.mark.parametrize(
        ("module", "extra"),
        [
            ("kokoro", "tts"),
            ("faster_whisper", "stt"),
            ("sounddevice", "audio"),
            ("openwakeword", "audio"),
            ("pynvml", "gpu"),
        ],
    )
    def test_every_optional_engine_names_the_extra_that_installs_it(
        self, module: str, extra: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A wrong hint sends the user to an extra that does not contain the package."""

        def _absent(name: str) -> object:
            raise ImportError(f"No module named {name!r}")

        monkeypatch.setattr(plat.importlib, "import_module", _absent)

        with pytest.raises(DependencyMissingError) as excinfo:
            plat.require_module(module)
        assert excinfo.value.extra == extra

    def test_the_named_extras_exist_in_the_manifest(self) -> None:
        """The hint tells the user to run `uv sync --extra X`, so X must be real."""
        import tomllib
        from pathlib import Path

        manifest = Path(__file__).resolve().parents[1] / "pyproject.toml"
        with manifest.open("rb") as handle:
            declared = set(tomllib.load(handle)["project"]["optional-dependencies"])

        named = {extra for extra in plat._EXTRA_FOR_MODULE.values() if extra}
        assert named <= declared, f"hints name extras that do not exist: {named - declared}"

    @pytest.mark.skipif(plat.is_windows(), reason="checks the non-Windows branch")
    def test_require_windows_raises_off_windows(self) -> None:
        with pytest.raises(PlatformUnsupportedError):
            plat.require_windows("fan speeds")

    def test_capability_probes_never_raise(self) -> None:
        assert isinstance(plat.has_nvml(), bool)
        assert isinstance(plat.has_cuda(), bool)
        assert plat.nvidia_smi_query() is None or isinstance(plat.nvidia_smi_query(), dict)

    def test_cpu_model_returns_something(self) -> None:
        assert isinstance(plat.cpu_model(), str)
        assert plat.cpu_model()

    def test_total_ram_is_positive(self) -> None:
        assert plat.total_ram_gb() > 0

    def test_project_root_holds_the_contract(self) -> None:
        assert (plat.project_root() / "CLAUDE.md").is_file()
        assert (plat.project_root() / "pyproject.toml").is_file()

    def test_project_root_honours_the_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        plat.project_root.cache_clear()
        monkeypatch.setenv("JARVIS_PROJECT_ROOT", "/tmp/elsewhere")
        assert plat.project_root() == Path("/tmp/elsewhere")
        plat.project_root.cache_clear()

    def test_a_frozen_build_does_not_root_itself_in_temp(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """PyInstaller onefile unpacks into %TEMP%\\_MEIxxxx on every run.

        Walking up from __file__ inside that bundle lands on %TEMP% itself, a
        directory every process running as the user can write to. The helper
        runs elevated and loads a native assembly from a path derived from this,
        so getting it wrong turned a path lookup into a way to hand an
        Administrator process attacker-supplied bytes.
        """
        unpacked = tmp_path / "_MEIabc123"
        (unpacked / "vendor").mkdir(parents=True)
        installed = tmp_path / "Program Files" / "JARVIS"
        installed.mkdir(parents=True)

        plat.project_root.cache_clear()
        monkeypatch.delenv("JARVIS_PROJECT_ROOT", raising=False)
        monkeypatch.setattr(plat.sys, "frozen", True, raising=False)
        monkeypatch.setattr(plat.sys, "_MEIPASS", str(unpacked), raising=False)
        monkeypatch.setattr(plat.sys, "executable", str(installed / "jarvis-helper.exe"))
        try:
            assert plat.is_frozen() is True
            assert plat.project_root() == installed, "user files must live beside the exe"
            assert plat.bundle_root() == unpacked, "bundled assets live in the unpack dir"
            assert plat.project_root() != tmp_path, "rooted in the writable parent of the bundle"
        finally:
            plat.project_root.cache_clear()

    def test_the_two_roots_agree_when_not_frozen(self) -> None:
        plat.project_root.cache_clear()
        assert plat.bundle_root() == plat.project_root()
        assert plat.is_frozen() is False


class _FakeClock:
    """Deterministic monotonic clock for latency tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def __call__(self) -> float:
        return self.now


class _RedactIn(_ToolInput):
    """Input for the redaction dispatch test. Module level so get_type_hints resolves it."""


class _RedactOut(_ToolOutput):
    """Output for the redaction dispatch test."""

    ok: bool = True


class TestRedaction:
    """Error text that travels to the model must not carry the user with it."""

    def test_windows_home_path_is_removed(self) -> None:
        from jarvis.util.errors import redact

        text = r"FileNotFoundError: 'C:\Users\mann\AppData\Roaming\jarvis\config.yaml'"
        result = redact(text)
        assert "mann" not in result
        assert "AppData" not in result
        assert "<path>" in result
        assert "FileNotFoundError" in result, "the model still needs the failure kind"

    def test_posix_home_paths_are_removed(self) -> None:
        from jarvis.util.errors import redact

        for text in ("/home/mann/notes/private.txt", "/Users/mann/Desktop/tax.pdf"):
            result = redact(f"OSError: could not open {text}")
            assert "mann" not in result
            assert "<path>" in result

    def test_url_credentials_are_removed(self) -> None:
        from jarvis.util.errors import redact

        result = redact("ConnectError: http://admin:hunter2@localhost:8080/search failed")
        assert "hunter2" not in result
        assert "admin" not in result
        assert "localhost:8080" in result, "the host itself is not a secret"

    def test_system_paths_are_left_alone(self) -> None:
        """Over-redacting would strip the detail that makes an error useful."""
        from jarvis.util.errors import redact

        text = r"OSError: C:\Windows\System32\drivers\etc\hosts is not readable"
        assert redact(text) == text

    def test_ordinary_text_is_unchanged(self) -> None:
        from jarvis.util.errors import redact

        text = "ToolExecutionError: the sensor bus did not respond within 5 seconds"
        assert redact(text) == text

    def test_a_failing_tool_does_not_hand_the_model_a_user_path(self) -> None:
        """The end-to-end path: dispatch, then what for_llm actually sends."""
        from jarvis.tools.registry import ToolRegistry, tool

        isolated = ToolRegistry()

        @tool(
            name="demo.explode",
            description="A tool that fails, used to check what failure text travels.",
            category=_ToolCategory.SYSTEM,
            read_only=True,
            target=isolated,
        )
        def _explode(params: _RedactIn) -> _RedactOut:
            raise FileNotFoundError(
                2, "No such file or directory", r"C:\Users\mann\secrets\token.txt"
            )

        payload = str(isolated.dispatch("demo.explode", {}).for_llm())
        assert "mann" not in payload
        assert "secrets" not in payload
        assert "FileNotFoundError" in payload
