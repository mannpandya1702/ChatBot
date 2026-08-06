"""T-5.1 verification: the latency benchmark harness.

The harness's own logic is what these tests cover: percentile aggregation,
budget verdicts, regression detection against a previous run, and the promise
that an unavailable engine is reported as skipped rather than faked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jarvis.config import JarvisConfig
from jarvis.util.latency import Stage


class TestSyntheticMeasurement:
    def test_produces_every_budgeted_stage(self) -> None:
        from tests.bench_latency import measure_synthetic

        results = measure_synthetic(10)
        for stage in (
            Stage.VAD_ENDPOINT,
            Stage.STT,
            Stage.LLM_FIRST_TOKEN,
            Stage.TTS_FIRST_AUDIO,
            Stage.TURN_TOTAL,
        ):
            assert stage in results
            assert len(results[stage].samples) == 10

    def test_synthetic_values_sit_inside_the_budgets(self, cfg: JarvisConfig) -> None:
        """A CI run must pass, otherwise the signal is worthless."""
        from tests.bench_latency import _budget, measure_synthetic

        for stage, result in measure_synthetic(30).items():
            summary = result.summary(_budget(cfg, stage))
            assert summary["within_budget"] is True, stage

    def test_percentiles_differ_from_the_mean(self) -> None:
        """Flat samples would let a broken percentile implementation pass."""
        from jarvis.config import load_defaults
        from tests.bench_latency import _budget, measure_synthetic

        summary = measure_synthetic(30)[Stage.STT].summary(
            _budget(load_defaults(), Stage.STT)
        )
        assert summary["p95_ms"] > summary["p50_ms"]


class TestStageResult:
    def test_skipped_stage_reports_a_reason(self) -> None:
        from tests.bench_latency import StageResult

        result = StageResult(stage="stt", skipped=True, skip_reason="no model")
        summary = result.summary(200.0)
        assert summary["skipped"] is True
        assert summary["reason"] == "no model"
        assert "p95_ms" not in summary

    def test_empty_samples_count_as_skipped(self) -> None:
        """An empty stage must never be reported as a passing zero."""
        from tests.bench_latency import StageResult

        assert StageResult(stage="stt").summary(200.0)["skipped"] is True

    def test_over_budget_is_flagged(self) -> None:
        from tests.bench_latency import StageResult

        result = StageResult(stage="stt", samples=[500.0] * 20)
        assert result.summary(200.0)["within_budget"] is False

    def test_within_budget_is_flagged(self) -> None:
        from tests.bench_latency import StageResult

        result = StageResult(stage="stt", samples=[50.0] * 20)
        assert result.summary(200.0)["within_budget"] is True

    def test_no_budget_gives_no_verdict(self) -> None:
        from tests.bench_latency import StageResult

        assert StageResult(stage="tool", samples=[1.0]).summary(None)["within_budget"] is None


class TestRegressionDetection:
    def _row(self, p95: float) -> dict[str, Any]:
        return {"skipped": False, "p95_ms": p95, "budget_ms": 1000.0, "within_budget": True}

    def test_no_previous_run_is_not_a_regression(self) -> None:
        from tests.bench_latency import check_regression

        assert check_regression({"stt": self._row(100)}, None, 0.15) == []

    def test_small_increase_is_tolerated(self) -> None:
        from tests.bench_latency import check_regression

        current = {"stt": self._row(110)}
        previous = {"stt": self._row(100)}
        assert check_regression(current, previous, 0.15) == []

    def test_increase_beyond_tolerance_fails(self) -> None:
        """T-5.1: fail when p95 regresses more than 15 percent."""
        from tests.bench_latency import check_regression

        current = {"stt": self._row(120)}
        previous = {"stt": self._row(100)}
        failures = check_regression(current, previous, 0.15)
        assert len(failures) == 1
        assert "20.0 percent" in failures[0]

    def test_exactly_at_tolerance_passes(self) -> None:
        from tests.bench_latency import check_regression

        assert check_regression({"stt": self._row(115)}, {"stt": self._row(100)}, 0.15) == []

    def test_improvement_is_not_a_regression(self) -> None:
        from tests.bench_latency import check_regression

        assert check_regression({"stt": self._row(50)}, {"stt": self._row(100)}, 0.15) == []

    def test_skipped_stages_are_ignored(self) -> None:
        from tests.bench_latency import check_regression

        current = {"stt": {"skipped": True}}
        previous = {"stt": self._row(1)}
        assert check_regression(current, previous, 0.15) == []

    def test_new_stage_is_not_a_regression(self) -> None:
        from tests.bench_latency import check_regression

        assert check_regression({"tts": self._row(500)}, {"stt": self._row(1)}, 0.15) == []

    def test_multiple_regressions_are_all_reported(self) -> None:
        from tests.bench_latency import check_regression

        current = {"stt": self._row(200), "tts": self._row(300)}
        previous = {"stt": self._row(100), "tts": self._row(100)}
        assert len(check_regression(current, previous, 0.15)) == 2


class TestBudgetCheck:
    def test_over_budget_is_reported(self) -> None:
        from tests.bench_latency import check_budgets

        rows = {
            "stt": {"skipped": False, "p95_ms": 900.0, "budget_ms": 200.0, "within_budget": False}
        }
        assert len(check_budgets(rows)) == 1

    def test_within_budget_is_silent(self) -> None:
        from tests.bench_latency import check_budgets

        rows = {
            "stt": {"skipped": False, "p95_ms": 50.0, "budget_ms": 200.0, "within_budget": True}
        }
        assert check_budgets(rows) == []

    def test_skipped_is_silent(self) -> None:
        from tests.bench_latency import check_budgets

        assert check_budgets({"stt": {"skipped": True}}) == []


class TestMain:
    def test_synthetic_run_writes_results(self, tmp_path: Path) -> None:
        from tests.bench_latency import main

        output = tmp_path / "results.json"
        assert main(["--synthetic", "--iterations", "5", "--output", str(output)]) == 0

        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["synthetic"] is True
        assert payload["iterations"] == 5
        assert "stt" in payload["stages"]
        assert "tier" in payload

    def test_second_run_compares_against_the_first(self, tmp_path: Path) -> None:
        from tests.bench_latency import main

        output = tmp_path / "results.json"
        main(["--synthetic", "--iterations", "5", "--output", str(output)])
        # Same synthetic shape, so the second run must not regress.
        assert main(["--synthetic", "--iterations", "5", "--output", str(output)]) == 0

    def test_regression_causes_a_non_zero_exit(self, tmp_path: Path) -> None:
        from tests.bench_latency import main

        output = tmp_path / "results.json"
        output.write_text(
            json.dumps(
                {
                    "stages": {
                        "stt": {
                            "skipped": False,
                            "p95_ms": 1.0,
                            "budget_ms": 200.0,
                            "within_budget": True,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        assert main(["--synthetic", "--iterations", "5", "--output", str(output)]) == 1

    def test_no_fail_flag_suppresses_the_exit_code(self, tmp_path: Path) -> None:
        from tests.bench_latency import main

        output = tmp_path / "results.json"
        output.write_text(
            json.dumps(
                {
                    "stages": {
                        "stt": {
                            "skipped": False,
                            "p95_ms": 1.0,
                            "budget_ms": 200.0,
                            "within_budget": True,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        assert (
            main(["--synthetic", "--iterations", "5", "--output", str(output), "--no-fail"]) == 0
        )

    def test_corrupt_previous_results_do_not_crash_the_run(self, tmp_path: Path) -> None:
        from tests.bench_latency import main

        output = tmp_path / "results.json"
        output.write_text("{not json", encoding="utf-8")
        assert main(["--synthetic", "--iterations", "5", "--output", str(output)]) == 0


class TestLiveMeasurementDegradesGracefully:
    def test_missing_engines_are_skipped_not_faked(self, cfg: JarvisConfig) -> None:
        """A benchmark that invents numbers is worse than no benchmark."""
        from tests.bench_latency import measure_live

        results = measure_live(cfg, 1)
        for stage, result in results.items():
            if result.skipped:
                assert result.skip_reason, f"{stage} was skipped without saying why"
            else:
                assert result.samples, f"{stage} reported as measured but has no samples"

    def test_live_run_never_raises(self, cfg: JarvisConfig) -> None:
        from tests.bench_latency import measure_live

        measure_live(cfg, 1)
