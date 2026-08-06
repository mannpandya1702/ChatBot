"""Latency benchmark suite.

T-5.1: p50, p95, and p99 per stage, written to ``bench/results.json``, failing
when p95 regresses by more than the configured tolerance against the previous
run.

Run it two ways:

* ``uv run python tests/bench_latency.py`` measures what is actually available
  on this machine. Stages whose engine is not installed are reported as skipped
  rather than faked, because a benchmark that invents numbers is worse than no
  benchmark.
* ``uv run python tests/bench_latency.py --synthetic`` exercises the harness
  itself with deterministic timings, which is what CI runs.

Budgets come from CLAUDE.md section 3, via ``config.latency.budgets_ms``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jarvis.config import JarvisConfig, load_config
from jarvis.util.latency import DEFAULT_BUDGETS_MS, LatencyStats, Stage, TurnLatency, percentile

DEFAULT_ITERATIONS = 30


@dataclass
class StageResult:
    """Aggregated timings for one stage."""

    stage: str
    samples: list[float] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None

    def summary(self, budget: float | None) -> dict[str, Any]:
        """p50/p95/p99 plus a budget verdict."""
        if self.skipped or not self.samples:
            return {
                "stage": self.stage,
                "skipped": True,
                "reason": self.skip_reason or "no samples",
                "budget_ms": budget,
            }
        p95 = percentile(self.samples, 95)
        return {
            "stage": self.stage,
            "skipped": False,
            "count": len(self.samples),
            "p50_ms": round(percentile(self.samples, 50), 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(percentile(self.samples, 99), 2),
            "mean_ms": round(statistics.fmean(self.samples), 2),
            "min_ms": round(min(self.samples), 2),
            "max_ms": round(max(self.samples), 2),
            "budget_ms": budget,
            "within_budget": None if budget is None else p95 <= budget,
        }


def _budget(config: JarvisConfig, stage: Stage) -> float | None:
    """Configured budget for a stage, falling back to the §3 defaults."""
    configured = config.latency.budgets_ms.get(str(stage))
    if configured is not None:
        return float(configured)
    return DEFAULT_BUDGETS_MS.get(stage)


def measure_synthetic(iterations: int) -> dict[Stage, StageResult]:
    """Deterministic timings that exercise the harness without any engine.

    Values sit comfortably inside the §3 budgets so a CI run is meaningful:
    if the aggregation, percentile, or regression logic breaks, this fails.
    """
    shape = {
        Stage.VAD_ENDPOINT: 120.0,
        Stage.STT: 95.0,
        Stage.LLM_FIRST_TOKEN: 210.0,
        Stage.TTS_FIRST_AUDIO: 140.0,
        Stage.TURN_TOTAL: 620.0,
    }
    results: dict[Stage, StageResult] = {}
    for stage, base in shape.items():
        result = StageResult(stage=str(stage))
        for index in range(iterations):
            # A small deterministic spread so the percentiles differ from
            # the mean and a broken percentile implementation shows up.
            result.samples.append(base + (index % 7) * 3.0)
        results[stage] = result
    return results


def measure_live(config: JarvisConfig, iterations: int) -> dict[Stage, StageResult]:
    """Measure whatever this machine can actually run.

    Every stage is attempted independently, so a missing TTS engine does not
    prevent the LLM stage from being measured.
    """
    results: dict[Stage, StageResult] = {
        stage: StageResult(stage=str(stage))
        for stage in (
            Stage.VAD_ENDPOINT,
            Stage.STT,
            Stage.LLM_FIRST_TOKEN,
            Stage.TTS_FIRST_AUDIO,
            Stage.TURN_TOTAL,
        )
    }

    _measure_llm(config, iterations, results)
    _measure_stt(config, iterations, results)
    _measure_tts(config, iterations, results)
    _measure_vad(config, iterations, results)
    _compose_turn_total(results)
    return results


def _compose_turn_total(results: dict[Stage, StageResult]) -> None:
    """Derive the end-to-end figure from the four stages that make it up.

    TURN_TOTAL is not measured directly here because doing so would need a real
    spoken turn. It is only meaningful when every component stage was measured;
    summing a partial set would understate the total and quietly pass a budget
    the machine cannot actually meet.
    """
    components = [Stage.VAD_ENDPOINT, Stage.STT, Stage.LLM_FIRST_TOKEN, Stage.TTS_FIRST_AUDIO]
    missing = [str(stage) for stage in components if results[stage].skipped]
    total = results[Stage.TURN_TOTAL]

    if missing:
        total.skipped = True
        total.skip_reason = f"not all component stages were measured, missing {', '.join(missing)}"
        return

    counts = [len(results[stage].samples) for stage in components]
    if not all(counts):
        total.skipped = True
        total.skip_reason = "one or more component stages produced no samples"
        return

    # Pair samples index by index so the composed distribution keeps its spread
    # rather than collapsing to a single mean-of-means figure.
    for index in range(min(counts)):
        total.samples.append(sum(results[stage].samples[index] for stage in components))


def _skip(results: dict[Stage, StageResult], stage: Stage, reason: str) -> None:
    """Record that a stage could not be measured on this machine."""
    results[stage].skipped = True
    results[stage].skip_reason = reason


def _measure_llm(
    config: JarvisConfig, iterations: int, results: dict[Stage, StageResult]
) -> None:
    """Time to first token from a live Ollama."""
    try:
        from jarvis.brain.llm import OllamaClient
    except ImportError as exc:
        _skip(results, Stage.LLM_FIRST_TOKEN, f"brain.llm is unavailable: {exc}")
        return

    try:
        client = OllamaClient(config)
        if not client.is_available():
            _skip(results, Stage.LLM_FIRST_TOKEN, "Ollama is not running")
            return
    except Exception as exc:  # noqa: BLE001 - a benchmark must never crash the run
        _skip(results, Stage.LLM_FIRST_TOKEN, f"could not reach Ollama: {exc}")
        return

    messages = [{"role": "user", "content": "Say the single word: ready."}]
    for _ in range(iterations):
        turn = TurnLatency(budgets_ms=_budgets_from(config))
        try:
            for chunk in client.chat_stream(messages):
                if chunk.content:
                    turn.mark(Stage.LLM_FIRST_TOKEN)
                    break
        except Exception as exc:  # noqa: BLE001
            _skip(results, Stage.LLM_FIRST_TOKEN, f"the request failed: {exc}")
            return
        timing = turn.get(Stage.LLM_FIRST_TOKEN)
        if timing is not None:
            results[Stage.LLM_FIRST_TOKEN].samples.append(timing.duration_ms)


def _measure_stt(
    config: JarvisConfig, iterations: int, results: dict[Stage, StageResult]
) -> None:
    """Transcription time for a short fixed utterance."""
    try:
        import numpy as np

        from jarvis.audio.stt import build_transcriber
    except ImportError as exc:
        _skip(results, Stage.STT, f"the STT engine is not installed: {exc}")
        return

    try:
        transcriber = build_transcriber(config)
    except Exception as exc:  # noqa: BLE001
        _skip(results, Stage.STT, f"could not load the STT model: {exc}")
        return

    rate = config.audio.sample_rate
    audio = (
        0.1 * np.sin(2 * np.pi * 220 * np.arange(int(rate * 1.5), dtype=np.float32) / rate)
    ).astype(np.float32)

    for _ in range(iterations):
        started = time.perf_counter()
        try:
            transcriber.transcribe(audio, rate)
        except Exception as exc:  # noqa: BLE001
            _skip(results, Stage.STT, f"transcription failed: {exc}")
            return
        results[Stage.STT].samples.append((time.perf_counter() - started) * 1000.0)


def _measure_tts(
    config: JarvisConfig, iterations: int, results: dict[Stage, StageResult]
) -> None:
    """Time to first audio chunk from the synthesiser."""
    try:
        from jarvis.audio.tts import build_synthesizer
    except ImportError as exc:
        _skip(results, Stage.TTS_FIRST_AUDIO, f"the TTS engine is not installed: {exc}")
        return

    try:
        synth = build_synthesizer(config)
    except Exception as exc:  # noqa: BLE001
        _skip(results, Stage.TTS_FIRST_AUDIO, f"could not load Kokoro: {exc}")
        return

    for _ in range(iterations):
        started = time.perf_counter()
        try:
            synth.synthesize("The processor is running at forty three percent.")
        except Exception as exc:  # noqa: BLE001
            _skip(results, Stage.TTS_FIRST_AUDIO, f"synthesis failed: {exc}")
            return
        results[Stage.TTS_FIRST_AUDIO].samples.append((time.perf_counter() - started) * 1000.0)


def _measure_vad(
    config: JarvisConfig, iterations: int, results: dict[Stage, StageResult]
) -> None:
    """Endpoint decision latency, measured frame by frame."""
    try:
        import numpy as np

        from jarvis.audio.vad import Endpointer
    except ImportError as exc:
        _skip(results, Stage.VAD_ENDPOINT, f"the VAD is unavailable: {exc}")
        return

    frame_samples = config.vad.frame_samples
    speech = (0.2 * np.ones(frame_samples, dtype=np.float32)).astype(np.float32)
    silence = np.zeros(frame_samples, dtype=np.float32)

    # A scripted probability source keeps this measuring the endpointer's own
    # decision cost rather than the neural VAD's inference time.
    class Scripted:
        def __init__(self) -> None:
            self.calls = 0

        def probability(self, _frame: Any) -> float:
            self.calls += 1
            return 0.9 if self.calls <= 20 else 0.0

        def reset(self) -> None:
            self.calls = 0

    for _ in range(iterations):
        try:
            endpointer = Endpointer(config, vad=Scripted())
        except Exception as exc:  # noqa: BLE001
            _skip(results, Stage.VAD_ENDPOINT, f"could not build the endpointer: {exc}")
            return

        started = time.perf_counter()
        endpoint = None
        for index in range(400):
            frame = speech if index < 20 else silence
            try:
                endpoint = endpointer.process(frame)
            except Exception as exc:  # noqa: BLE001
                _skip(results, Stage.VAD_ENDPOINT, f"the endpointer failed: {exc}")
                return
            if endpoint is not None:
                break
        if endpoint is None:
            _skip(results, Stage.VAD_ENDPOINT, "the endpointer never closed the utterance")
            return
        results[Stage.VAD_ENDPOINT].samples.append((time.perf_counter() - started) * 1000.0)


def _budgets_from(config: JarvisConfig) -> dict[Stage, float]:
    """Config budgets keyed by Stage."""
    out: dict[Stage, float] = {}
    for stage in Stage:
        budget = _budget(config, stage)
        if budget is not None:
            out[stage] = budget
    return out


def check_regression(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
    tolerance: float,
) -> list[str]:
    """Report stages whose p95 regressed beyond ``tolerance``.

    Args:
        current: This run's stage summaries, keyed by stage name.
        previous: The previous run's, or None on a first run.
        tolerance: Allowed fractional increase, for example 0.15 for 15 percent.

    Returns:
        Human-readable regression descriptions, empty when nothing regressed.
    """
    if not previous:
        return []
    failures: list[str] = []
    for name, row in current.items():
        if row.get("skipped"):
            continue
        before = previous.get(name)
        if not before or before.get("skipped"):
            continue
        old = float(before.get("p95_ms", 0.0))
        new = float(row.get("p95_ms", 0.0))
        if old <= 0:
            continue
        change = (new - old) / old
        if change > tolerance:
            failures.append(
                f"{name}: p95 went from {old:.1f} ms to {new:.1f} ms, "
                f"a {change * 100:.1f} percent regression, over the "
                f"{tolerance * 100:.0f} percent tolerance"
            )
    return failures


def check_budgets(current: dict[str, Any]) -> list[str]:
    """Report stages whose p95 exceeds its §3 budget."""
    return [
        f"{name}: p95 {row['p95_ms']:.1f} ms exceeds the {row['budget_ms']:.0f} ms budget"
        for name, row in current.items()
        if not row.get("skipped") and row.get("within_budget") is False
    ]


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and write ``bench/results.json``."""
    parser = argparse.ArgumentParser(
        prog="bench_latency",
        description="Measure per-stage latency against the CLAUDE.md section 3 budgets.",
    )
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Exercise the harness with deterministic timings instead of real engines.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Report regressions and budget misses without a non-zero exit.",
    )
    args = parser.parse_args(argv)

    config = load_config()
    output = args.output or config.resolve_path(config.latency.results_path)

    previous: dict[str, Any] | None = None
    if output.is_file():
        try:
            previous = json.loads(output.read_text(encoding="utf-8")).get("stages")
        except (OSError, ValueError):
            previous = None

    print(f"Running {'synthetic' if args.synthetic else 'live'} benchmark, "  # noqa: T201
          f"{args.iterations} iterations per stage")

    measured = (
        measure_synthetic(args.iterations)
        if args.synthetic
        else measure_live(config, args.iterations)
    )

    stats = LatencyStats(_budgets_from(config))
    stages: dict[str, Any] = {}
    for stage, result in measured.items():
        stages[str(stage)] = result.summary(_budget(config, stage))
        for sample in result.samples:
            stats.add(stage, sample)

    payload = {
        "tier": str(config.effective_tier()),
        "llm_model": config.llm_model(),
        "iterations": args.iterations,
        "synthetic": args.synthetic,
        "stages": stages,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print()  # noqa: T201
    header = f"{'stage':<20} {'p50':>9} {'p95':>9} {'p99':>9} {'budget':>9}  verdict"
    print(header)  # noqa: T201
    print("-" * len(header))  # noqa: T201
    for name, row in stages.items():
        if row.get("skipped"):
            print(f"{name:<20} {'skipped':>9}  {row.get('reason', '')}")  # noqa: T201
            continue
        budget = row.get("budget_ms")
        verdict = "ok" if row.get("within_budget") is not False else "OVER BUDGET"
        print(  # noqa: T201
            f"{name:<20} {row['p50_ms']:>9.1f} {row['p95_ms']:>9.1f} "
            f"{row['p99_ms']:>9.1f} {budget if budget else '-':>9}  {verdict}"
        )

    problems = check_budgets(stages)
    problems += check_regression(stages, previous, config.latency.regression_tolerance)

    print()  # noqa: T201
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")  # noqa: T201
        print(f"\nResults written to {output}")  # noqa: T201
        return 0 if args.no_fail else 1

    print("All measured stages are within budget and show no regression.")  # noqa: T201
    print(f"Results written to {output}")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
