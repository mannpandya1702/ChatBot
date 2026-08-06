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


class TestRegressionNoiseFloor:
    """A percentage on a sub-millisecond stage is noise, not a regression.

    vad_endpoint runs in about a tenth of a millisecond. Ordinary scheduling
    jitter of four hundredths of a millisecond is a 37 percent increase, and a
    CI job that fails on that trains everyone to ignore CI.
    """

    @staticmethod
    def _row(p95: float) -> dict[str, Any]:
        return {"p95_ms": p95, "skipped": False}

    def test_sub_millisecond_jitter_is_not_a_regression(self) -> None:
        from tests.bench_latency import check_regression

        current = {"vad_endpoint": self._row(0.11)}
        previous = {"vad_endpoint": self._row(0.08)}
        assert check_regression(current, previous, tolerance=0.15) == []

    def test_a_real_slowdown_on_a_slow_stage_still_fails(self) -> None:
        from tests.bench_latency import check_regression

        current = {"tts_first_audio": self._row(400.0)}
        previous = {"tts_first_audio": self._row(300.0)}
        failures = check_regression(current, previous, tolerance=0.15)
        assert len(failures) == 1
        assert "33.3 percent" in failures[0]

    def test_the_floor_is_absolute_not_relative(self) -> None:
        """A big percentage on a tiny stage stays quiet; the same on a big one does not."""
        from tests.bench_latency import REGRESSION_FLOOR_MS, check_regression

        tiny = check_regression(
            {"s": self._row(REGRESSION_FLOOR_MS - 0.1)}, {"s": self._row(0.01)}, 0.15
        )
        assert tiny == [], "an increase under the floor must not fail"

        # Clears the floor by 45 ms and the tolerance by 35 percentage points.
        large = check_regression({"s": self._row(150.0)}, {"s": self._row(100.0)}, 0.15)
        assert large, "an increase over the floor must still be checked"

        # Clears the floor but not the tolerance: still quiet.
        assert (
            check_regression(
                {"s": self._row(100.0 + REGRESSION_FLOOR_MS + 1.0)},
                {"s": self._row(100.0)},
                0.15,
            )
            == []
        ), "the percentage tolerance still applies above the floor"

    def test_the_floor_sits_far_below_every_budget(self) -> None:
        """A floor near a budget would mask regressions that matter."""
        from jarvis.config import LatencyConfig
        from tests.bench_latency import REGRESSION_FLOOR_MS

        budgets = [float(v) for v in LatencyConfig().budgets_ms.values()]
        assert budgets, "no budgets configured"
        assert min(budgets) / 10 > REGRESSION_FLOOR_MS


class TestBudgetsApplyOnlyToTheirTier:
    """§3 heads its table "gpu-12 target", so the numbers describe that machine.

    Holding a cpu tier to them reports a failure no amount of correct code can
    fix, which makes the benchmark unusable on the tier most people start on.
    """

    @staticmethod
    def _over_budget() -> dict[str, Any]:
        return {
            "tts_first_audio": {
                "p95_ms": 1100.0,
                "budget_ms": 300.0,
                "within_budget": False,
                "skipped": False,
            }
        }

    def test_a_miss_on_the_cpu_tier_is_advisory(self) -> None:
        from tests.bench_latency import budget_advisories, check_budgets

        stages = self._over_budget()
        assert check_budgets(stages, "cpu") == []
        notes = budget_advisories(stages, "cpu")
        assert len(notes) == 1
        assert "does not apply to the cpu tier" in notes[0]

    def test_a_miss_on_the_target_tier_fails(self) -> None:
        from tests.bench_latency import budget_advisories, check_budgets

        stages = self._over_budget()
        assert len(check_budgets(stages, "gpu-12")) == 1
        assert budget_advisories(stages, "gpu-12") == []

    def test_faster_tiers_are_held_to_the_budget_too(self) -> None:
        from tests.bench_latency import check_budgets

        for tier in ("gpu-12", "gpu-16", "gpu-24"):
            assert check_budgets(self._over_budget(), tier), f"{tier} should be enforced"

    def test_slower_tiers_are_not(self) -> None:
        from tests.bench_latency import check_budgets

        for tier in ("cpu", "gpu-6", "gpu-8"):
            assert check_budgets(self._over_budget(), tier) == [], f"{tier} should be advisory"

    def test_every_enforced_tier_is_a_real_tier(self) -> None:
        from jarvis.config import TIER_PROFILES
        from tests.bench_latency import BUDGET_TIER, BUDGET_TIERS

        known = {str(tier) for tier in TIER_PROFILES}
        assert known >= BUDGET_TIERS, f"unknown tiers: {BUDGET_TIERS - known}"
        assert BUDGET_TIER in known
