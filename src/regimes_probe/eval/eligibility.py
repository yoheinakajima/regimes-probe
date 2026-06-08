"""Headline-eligibility computation for a report.

A run is *headline-eligible* only if every structural guard passes AND the
dataset is a real benchmark (a synthetic/placeholder fixture is never a benchmark
headline). When ineligible, the report carries the reasons. See
``docs/METHODOLOGY_RISKS.md`` and ``docs/FIRST_REAL_RESULT_CRITERIA.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: The structural checks every headline-eligible run must satisfy.
REQUIRED_CHECKS = (
    "optimize_confirm_disjoint",
    "confirm_memory_frozen",
    "no_answer_leakage",
    "same_conditions",
    "replay_passed",
    "baseline_and_policy_completed",
    "budget_enforced",
    "no_live_updates_during_confirm",
)

_REASONS = {
    "optimize_confirm_disjoint": "OPTIMIZE and CONFIRM splits overlap",
    "confirm_memory_frozen": "CONFIRM did not use a frozen memory snapshot",
    "no_answer_leakage": "answer-leakage check failed on policy memory",
    "same_conditions": "baseline and policy runs differ in more than memory access",
    "replay_passed": "replay check failed (graph != projection of log)",
    "baseline_and_policy_completed": "baseline and/or policy run did not complete",
    "budget_enforced": "a run exceeded its tool-call budget",
    "no_live_updates_during_confirm": "policy memory was updated during CONFIRM",
}


@dataclass
class Eligibility:
    mechanism_ok: bool
    headline_eligible: bool
    dataset_is_real: bool
    checks: dict[str, bool]
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "headline_eligible": self.headline_eligible,
            "mechanism_ok": self.mechanism_ok,
            "dataset_is_real": self.dataset_is_real,
            "checks": self.checks,
            "reasons": self.reasons,
        }


def compute_eligibility(checks: dict[str, bool], *, dataset_is_real: bool) -> Eligibility:
    """Combine structural checks + dataset realness into an eligibility verdict."""
    full = {name: bool(checks.get(name, False)) for name in REQUIRED_CHECKS}
    mechanism_ok = all(full.values())
    reasons = [_REASONS[name] for name, ok in full.items() if not ok]
    if not dataset_is_real:
        reasons.append("dataset is a synthetic/placeholder fixture — not a benchmark headline")
    return Eligibility(
        mechanism_ok=mechanism_ok,
        headline_eligible=bool(mechanism_ok and dataset_is_real),
        dataset_is_real=dataset_is_real,
        checks=full,
        reasons=reasons,
    )
