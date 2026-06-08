"""Metrics over a set of graded attempts.

Primary metric: ``correct_per_tool_call``. Plus accuracy, efficiency, and the
epistemic-error rates defined in ``docs/EVALUATION_PROTOCOL.md``.

An :class:`AttemptOutcome` is the per-attempt record the metrics aggregate. It
is produced by the eval harness from a trace + grade + reward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Optional


@dataclass
class AttemptOutcome:
    item_id: str
    condition: str
    budget: int
    correct: bool
    abstained: bool
    tool_calls: int
    cost: float
    latency: float
    evidence_score: float
    first_tool_hit: bool
    false_stop: bool
    over_search: bool
    under_search: bool
    stale_error: bool
    found_hit: bool
    cluster_key: str
    attempt_reward: float
    authority_ok: bool = True
    contradiction: bool = False
    support_found: bool = True
    failed_tool_calls: int = 0
    failed_tools: list = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "condition": self.condition,
            "budget": self.budget,
            "correct": int(self.correct),
            "abstained": int(self.abstained),
            "tool_calls": self.tool_calls,
            "cost": round(self.cost, 6),
            "latency": round(self.latency, 4),
            "evidence_score": round(self.evidence_score, 4),
            "first_tool_hit": int(self.first_tool_hit),
            "false_stop": int(self.false_stop),
            "over_search": int(self.over_search),
            "under_search": int(self.under_search),
            "stale_error": int(self.stale_error),
            "found_hit": int(self.found_hit),
            "cluster_key": self.cluster_key,
            "attempt_reward": round(self.attempt_reward, 6),
            "authority_ok": int(self.authority_ok),
            "contradiction": int(self.contradiction),
            "support_found": int(self.support_found),
            "failed_tool_calls": self.failed_tool_calls,
            "regime": "",  # filled by the regime detectors if run
        }


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_metrics(outcomes: list[AttemptOutcome]) -> dict[str, Any]:
    """Aggregate metrics for one condition+budget cell."""
    n = len(outcomes)
    if n == 0:
        return {"n": 0}
    correct = sum(o.correct for o in outcomes)
    calls = sum(o.tool_calls for o in outcomes)
    cost = sum(o.cost for o in outcomes)
    latency = sum(o.latency for o in outcomes)
    answered = [o for o in outcomes if not o.abstained]
    abstained = [o for o in outcomes if o.abstained]
    # Abstention precision/recall: an abstention is "correct" when the agent
    # would otherwise have been wrong; precision = abstained that were right to,
    # recall = of all would-be-wrong, how many abstained.
    would_be_wrong = sum(1 for o in outcomes if not o.correct)
    correct_abstentions = sum(1 for o in abstained if not o.found_hit)
    failed_calls = sum(o.failed_tool_calls for o in outcomes)
    failed_by_tool: dict[str, int] = {}
    for o in outcomes:
        for t in o.failed_tools:
            failed_by_tool[t] = failed_by_tool.get(t, 0) + 1
    return {
        "n": n,
        "accuracy": _safe_div(correct, n),
        "failed_tool_calls": failed_calls,
        "provider_failure_rate": _safe_div(failed_calls, calls),
        "failed_tool_calls_by_tool": failed_by_tool,
        "correct_per_tool_call": _safe_div(correct, calls),
        "correct_per_dollar": _safe_div(correct, cost) if cost else 0.0,
        "correct_per_second": _safe_div(correct, latency) if latency else 0.0,
        "total_tool_calls": calls,
        "mean_tool_calls": _safe_div(calls, n),
        "first_tool_hit_rate": _safe_div(sum(o.first_tool_hit for o in outcomes), n),
        "evidence_gain_per_call": _safe_div(sum(o.evidence_score for o in outcomes), calls),
        "primary_or_authoritative_source_rate": _safe_div(
            sum(1 for o in outcomes if o.found_hit), n
        ),
        "stale_source_error_rate": _safe_div(sum(o.stale_error for o in outcomes), n),
        "false_stop_rate": _safe_div(sum(o.false_stop for o in outcomes), n),
        "over_search_rate": _safe_div(sum(o.over_search for o in outcomes), n),
        "under_search_rate": _safe_div(sum(o.under_search for o in outcomes), n),
        "abstention_rate": _safe_div(len(abstained), n),
        "abstention_precision": _safe_div(correct_abstentions, len(abstained)) if abstained else 0.0,
        "abstention_recall": _safe_div(correct_abstentions, would_be_wrong) if would_be_wrong else 0.0,
        "mean_attempt_reward": mean(o.attempt_reward for o in outcomes),
    }
