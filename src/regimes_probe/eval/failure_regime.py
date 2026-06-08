"""First-class failure-regime objects attached to attempts.

A :class:`FailureRegime` is the structured form of the per-attempt *failure seam*
that ``eval/debug.py`` infers, enriched with the answer-free flags that triggered
it and the ids of the supporting tool calls / evidence. It is projected as a
``failure_regime`` object in ``graph_projection.json`` and surfaced (as
``regime_objects``) in ``debug_questions.jsonl``.

Regimes are diagnosed from answer-free outcome flags + the bounded debug record,
so detection never depends on the subject of a question. Provider failures from
``safe_search`` surface here as ``provider_error`` regimes (not bare strings).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

#: Canonical failure-regime names (superset of the debug failure seams plus the
#: generic detector regimes that are answer-free and attempt-attached).
FAILURE_REGIMES: tuple[str, ...] = (
    "provider_error",
    "provider_returned_no_results",
    "weak_query",
    "evidence_absent",
    "evidence_not_selected",
    "answer_extraction",
    "grader_strictness",
    "exact_answer_missing",
    "stale_evidence",
    "contradiction_unresolved",
    "support_answer_mismatch",
    "unknown",
)

#: The debug ``failure_seam`` maps onto a canonical regime 1:1 (same vocabulary,
#: with ``answer_extraction`` already shared). Kept explicit for clarity / drift.
_SEAM_TO_REGIME = {s: s for s in FAILURE_REGIMES}


@dataclass
class FailureRegime:
    """Structured failure-regime object for one failed/abstained attempt."""

    regime: str
    item_id: str
    attempt_id: str
    condition: str
    budget: int
    heuristic_score: float = 1.0
    # answer-free fields that triggered the regime
    triggers: dict[str, Any] = field(default_factory=dict)
    # provenance: ids of supporting tool calls / evidence in the projection
    tool_call_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "item_id": self.item_id,
            "attempt_id": self.attempt_id,
            "condition": self.condition,
            "budget": self.budget,
            "heuristic_score": round(float(self.heuristic_score), 3),
            "triggers": self.triggers,
            "tool_call_ids": list(self.tool_call_ids),
            "evidence_ids": list(self.evidence_ids),
        }


def _extra_regimes(dr: dict[str, Any]) -> list[str]:
    """Detector-style regimes that the seam alone does not capture."""
    extra: list[str] = []
    if dr.get("contradiction") and not dr.get("correct"):
        extra.append("contradiction_unresolved")
    # answered+graded-correct but on weak support: a support/answer mismatch risk.
    if dr.get("support_found") is False and dr.get("found_hit") and not dr.get("correct"):
        extra.append("support_answer_mismatch")
    return extra


def build_failure_regime(debug_row: dict[str, Any], *,
                         attempt_id: str) -> Optional[FailureRegime]:
    """Build a :class:`FailureRegime` from a debug record dict, or ``None`` if the
    attempt was correct (no failure to attribute)."""
    if debug_row.get("correct"):
        return None
    seam = debug_row.get("failure_seam") or "unknown"
    regime = _SEAM_TO_REGIME.get(seam, "unknown")
    triggers = {
        "abstained": bool(debug_row.get("abstained")),
        "n_results": int(debug_row.get("n_results", 0)),
        "failed_tool_calls": int(debug_row.get("failed_tool_calls", 0)),
        "support_found": bool(debug_row.get("support_found")),
        "found_hit": bool(debug_row.get("found_hit")),
        "authority_ok": bool(debug_row.get("authority_ok")),
        "contradiction": bool(debug_row.get("contradiction")),
    }
    return FailureRegime(
        regime=regime, item_id=debug_row.get("item_id", ""), attempt_id=attempt_id,
        condition=debug_row.get("condition", ""), budget=int(debug_row.get("budget", 0)),
        triggers=triggers)


def regime_names(debug_row: dict[str, Any]) -> list[str]:
    """All applicable canonical regime names for an attempt (seam + extras)."""
    if debug_row.get("correct"):
        return []
    seam = debug_row.get("failure_seam") or "unknown"
    names = [_SEAM_TO_REGIME.get(seam, "unknown")]
    for r in _extra_regimes(debug_row):
        if r not in names:
            names.append(r)
    return names
