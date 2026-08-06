"""Per-stage latency instrumentation.

CLAUDE.md §3 sets a hard budget: p95 from end of user speech to first audio out
must be at or under 1200 ms on the gpu-12 tier. Nothing can be held to that
budget without measuring each stage separately, so every stage in the turn loop
runs inside :meth:`TurnLatency.stage` and the per-turn breakdown is logged.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "DEFAULT_BUDGETS_MS",
    "LatencyStats",
    "Stage",
    "StageTiming",
    "TurnLatency",
    "percentile",
]

_log = logging.getLogger(__name__)


class Stage(StrEnum):
    """A measured segment of one conversational turn."""

    WAKE = "wake"
    VAD_ENDPOINT = "vad_endpoint"
    STT = "stt"
    LLM_FIRST_TOKEN = "llm_first_token"  # noqa: S105 - a stage name, not a secret
    LLM_TOTAL = "llm_total"
    TOOL = "tool"
    TTS_FIRST_AUDIO = "tts_first_audio"
    TTS_TOTAL = "tts_total"
    TURN_TOTAL = "turn_total"


#: Budgets from CLAUDE.md §3, in milliseconds. TURN_TOTAL is the p95 target.
DEFAULT_BUDGETS_MS: dict[Stage, float] = {
    Stage.VAD_ENDPOINT: 250.0,
    Stage.STT: 200.0,
    Stage.LLM_FIRST_TOKEN: 400.0,
    Stage.TTS_FIRST_AUDIO: 300.0,
    Stage.TURN_TOTAL: 1200.0,
}

#: A turn that calls a tool pays one extra LLM round trip (§3).
TOOL_TURN_TOTAL_BUDGET_MS = 2000.0


@dataclass(slots=True)
class StageTiming:
    """One measured stage."""

    stage: Stage
    duration_ms: float
    budget_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def over_budget(self) -> bool:
        """True when this stage exceeded its configured budget."""
        return self.budget_ms is not None and self.duration_ms > self.budget_ms

    def to_dict(self) -> dict[str, Any]:
        """Render for structured logging."""
        payload: dict[str, Any] = {
            "stage": str(self.stage),
            "duration_ms": round(self.duration_ms, 2),
        }
        if self.budget_ms is not None:
            payload["budget_ms"] = self.budget_ms
            payload["over_budget"] = self.over_budget
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


class TurnLatency:
    """Collects stage timings for a single turn.

    Usage::

        turn = TurnLatency()
        with turn.stage(Stage.STT) as t:
            text = transcribe(audio)
            t.metadata["chars"] = len(text)
        turn.finish()
    """

    def __init__(
        self,
        turn_id: str | None = None,
        *,
        budgets_ms: dict[Stage, float] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.turn_id = turn_id or uuid.uuid4().hex[:12]
        self.budgets_ms = dict(DEFAULT_BUDGETS_MS if budgets_ms is None else budgets_ms)
        self._clock = clock
        self._started = self._clock()
        self.timings: list[StageTiming] = []
        self.used_tool = False

    @contextmanager
    def stage(self, stage: Stage, **metadata: Any) -> Iterator[StageTiming]:
        """Time a block and record it. The yielded timing accepts extra metadata."""
        timing = StageTiming(stage=stage, duration_ms=0.0, budget_ms=self.budgets_ms.get(stage))
        timing.metadata.update(metadata)
        start = self._clock()
        try:
            yield timing
        finally:
            timing.duration_ms = (self._clock() - start) * 1000.0
            if stage is Stage.TOOL:
                self.used_tool = True
            self.timings.append(timing)

    def record(self, stage: Stage, duration_ms: float, **metadata: Any) -> StageTiming:
        """Record a stage that was timed elsewhere."""
        timing = StageTiming(
            stage=stage,
            duration_ms=duration_ms,
            budget_ms=self.budgets_ms.get(stage),
            metadata=dict(metadata),
        )
        if stage is Stage.TOOL:
            self.used_tool = True
        self.timings.append(timing)
        return timing

    def mark(self, stage: Stage, **metadata: Any) -> StageTiming:
        """Record elapsed time since the turn started, for time-to-first-X stages."""
        return self.record(stage, (self._clock() - self._started) * 1000.0, **metadata)

    @property
    def elapsed_ms(self) -> float:
        """Milliseconds since this turn was created."""
        return (self._clock() - self._started) * 1000.0

    def get(self, stage: Stage) -> StageTiming | None:
        """Most recent timing for ``stage``, or None."""
        for timing in reversed(self.timings):
            if timing.stage is stage:
                return timing
        return None

    def total_budget_ms(self) -> float:
        """Turn budget, widened to the tool-call budget when a tool ran."""
        if self.used_tool:
            return TOOL_TURN_TOTAL_BUDGET_MS
        return self.budgets_ms.get(Stage.TURN_TOTAL, DEFAULT_BUDGETS_MS[Stage.TURN_TOTAL])

    def finish(self) -> StageTiming:
        """Close the turn, recording TURN_TOTAL against the right budget."""
        total = StageTiming(
            stage=Stage.TURN_TOTAL,
            duration_ms=self.elapsed_ms,
            budget_ms=self.total_budget_ms(),
            metadata={"used_tool": self.used_tool},
        )
        self.timings.append(total)
        return total

    def breakdown(self) -> dict[str, Any]:
        """Structured per-turn summary suitable for the JSON log."""
        return {
            "turn_id": self.turn_id,
            "used_tool": self.used_tool,
            "stages": [t.to_dict() for t in self.timings],
            "over_budget": [str(t.stage) for t in self.timings if t.over_budget],
        }

    def log(self, logger: logging.Logger | None = None) -> None:
        """Emit the breakdown. Warns when any stage blew its budget."""
        target = logger or _log
        payload = self.breakdown()
        if payload["over_budget"]:
            target.warning("turn latency over budget", extra={"context": payload})
        else:
            target.info("turn latency", extra={"context": payload})


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile. ``pct`` is 0-100.

    Returns 0.0 for an empty sequence so callers never have to special-case it.
    """
    if not values:
        return 0.0
    if not 0.0 <= pct <= 100.0:
        msg = f"percentile must be between 0 and 100, got {pct}"
        raise ValueError(msg)
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


class LatencyStats:
    """Aggregates many turns into p50/p95/p99 per stage."""

    def __init__(self, budgets_ms: dict[Stage, float] | None = None) -> None:
        self.budgets_ms = dict(DEFAULT_BUDGETS_MS if budgets_ms is None else budgets_ms)
        self._samples: dict[Stage, list[float]] = {}

    def add_turn(self, turn: TurnLatency) -> None:
        """Fold one turn's stage timings into the aggregate."""
        for timing in turn.timings:
            self._samples.setdefault(timing.stage, []).append(timing.duration_ms)

    def add(self, stage: Stage, duration_ms: float) -> None:
        """Fold a single sample into the aggregate."""
        self._samples.setdefault(stage, []).append(duration_ms)

    def count(self, stage: Stage) -> int:
        """Number of samples recorded for ``stage``."""
        return len(self._samples.get(stage, []))

    def summary(self) -> dict[str, dict[str, float | int | bool | None]]:
        """p50/p95/p99 per stage, with budget comparison on p95."""
        out: dict[str, dict[str, float | int | bool | None]] = {}
        for stage, values in sorted(self._samples.items(), key=lambda kv: str(kv[0])):
            budget = self.budgets_ms.get(stage)
            p95 = percentile(values, 95)
            out[str(stage)] = {
                "count": len(values),
                "p50_ms": round(percentile(values, 50), 2),
                "p95_ms": round(p95, 2),
                "p99_ms": round(percentile(values, 99), 2),
                "min_ms": round(min(values), 2),
                "max_ms": round(max(values), 2),
                "budget_ms": budget,
                "within_budget": None if budget is None else p95 <= budget,
            }
        return out

    def failing_stages(self) -> list[str]:
        """Stages whose p95 exceeds their budget."""
        return [
            name
            for name, row in self.summary().items()
            if row.get("within_budget") is False
        ]
