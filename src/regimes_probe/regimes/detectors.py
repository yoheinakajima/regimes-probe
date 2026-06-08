"""Generic (non-topical) failure-regime detectors.

Regimes are diagnosed from answer-free attempt outcomes/flags, so detection
never depends on the subject of a question. See
``docs/REGIMES_IMPROVEMENT_LOOP.md`` for the catalogue.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from regimes_probe.eval.metrics import AttemptOutcome

REGIMES: tuple[str, ...] = (
    "route_miss",
    "query_miss",
    "evidence_sparse",
    "stale_evidence",
    "over_search",
    "under_search",
    "verification_miss",
    "stop_too_early",
    "stop_too_late",
    "answer_extraction_miss",
    "contradiction_unresolved",
    "support_answer_mismatch",
)


def detect_regimes(o: AttemptOutcome) -> list[str]:
    """Return the failure regime label(s) for one attempt (may be empty)."""
    labels: list[str] = []
    if o.over_search:
        labels.append("over_search")
        labels.append("stop_too_late")
    if o.false_stop:
        labels.append("stop_too_early")
    if o.under_search:
        labels.append("under_search")
    if o.stale_error:
        labels.append("stale_evidence")
    if not o.correct and not o.found_hit:
        # never surfaced the answer-bearing evidence: a retrieval failure. We
        # cannot perfectly separate route vs query from outcome alone; attribute
        # both as candidate causes and let the eval-diff decide.
        labels.append("route_miss")
        labels.append("query_miss")
        if o.tool_calls <= 1:
            labels.append("evidence_sparse")
    if o.correct is False and o.found_hit:
        # had the right evidence but answered wrong.
        labels.append("answer_extraction_miss")
    if o.correct and o.evidence_score < 0.5:
        labels.append("support_answer_mismatch")
    # Wrong answer that the agent nonetheless accepted (stopped/answered) on weak
    # support is a verification failure.
    if (not o.correct) and (not o.abstained) and (not o.authority_ok):
        labels.append("verification_miss")
    if o.contradiction and not o.correct:
        labels.append("contradiction_unresolved")
    return labels


def dominant_regimes(outcomes: list[AttemptOutcome]) -> Counter:
    """Aggregate regime counts across a set of attempts."""
    c: Counter = Counter()
    for o in outcomes:
        for label in detect_regimes(o):
            c[label] += 1
    return c


def label_outcome(o: AttemptOutcome) -> str:
    """Single primary label for per-question reporting (first detected)."""
    labels = detect_regimes(o)
    return labels[0] if labels else ("ok" if o.correct else "unknown")
