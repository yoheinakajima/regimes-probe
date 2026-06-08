"""Eligibility for a report — TWO distinct verdicts.

1. ``structurally_valid`` — the run's mechanics are sound: OPTIMIZE/CONFIRM
   disjoint, replay passes, no answer leakage, budgets enforced, the requested
   runs completed. A plumbing run (e.g. ``closed_book,no_memory_search``) can be
   structurally valid.

2. ``headline_eligible_memory_claim`` — the run can support the **main
   regimes-probe memory-learning claim**. This is much stricter: it requires a
   real dataset, ALL of the comparison conditions (closed_book, no_memory_search,
   random_memory, policy_memory), the same-conditions check on
   no_memory_search-vs-policy_memory, frozen CONFIRM memory with no live updates,
   and a nontrivial CONFIRM size. A run without ``policy_memory`` can NEVER be
   headline-eligible.

See ``docs/METHODOLOGY_RISKS.md`` and ``docs/FIRST_REAL_RESULT_CRITERIA.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

#: Conditions the main memory-learning comparison requires.
REQUIRED_CONDITIONS = ("closed_book", "no_memory_search", "random_memory", "policy_memory")

#: Structural checks (a plumbing run can satisfy these).
STRUCTURAL_CHECKS = (
    "optimize_confirm_disjoint",
    "replay_passed",
    "no_answer_leakage",
    "budget_enforced",
    "runs_completed",
)
#: Extra checks the memory-learning headline needs (only meaningful when the
#: no_memory_search vs policy_memory comparison actually ran).
MEMORY_CLAIM_CHECKS = (
    "confirm_memory_frozen",
    "no_live_updates_during_confirm",
    "same_conditions",
)
#: Back-compat union (some callers/tests iterate this).
REQUIRED_CHECKS = STRUCTURAL_CHECKS + MEMORY_CLAIM_CHECKS

_REASONS = {
    "optimize_confirm_disjoint": "OPTIMIZE and CONFIRM splits overlap",
    "replay_passed": "replay check failed (graph != projection of log)",
    "no_answer_leakage": "answer-leakage check failed on policy memory",
    "budget_enforced": "a run exceeded its tool-call budget",
    "runs_completed": "the requested runs did not complete",
    "confirm_memory_frozen": "CONFIRM did not use a frozen memory snapshot",
    "no_live_updates_during_confirm": "policy memory was updated during CONFIRM",
    "same_conditions": "no_memory_search and policy_memory differ in more than memory access",
}


@dataclass
class Eligibility:
    structurally_valid: bool
    headline_eligible_memory_claim: bool
    dataset_is_real: bool
    conditions_present: list[str]
    checks: dict[str, bool]
    structural_reasons: list[str]
    headline_eligibility_reasons: list[str]
    confirm_size: int = 0
    min_confirm: int = 20
    provider_failure_rate: float = 0.0
    failed_conditions: list[str] = field(default_factory=list)

    # --- back-compat aliases (older code/reports read these) ---
    @property
    def mechanism_ok(self) -> bool:
        return self.structurally_valid

    @property
    def headline_eligible(self) -> bool:
        return self.headline_eligible_memory_claim

    @property
    def reasons(self) -> list[str]:
        return self.headline_eligibility_reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "structurally_valid": self.structurally_valid,
            "headline_eligible_memory_claim": self.headline_eligible_memory_claim,
            # back-compat keys:
            "headline_eligible": self.headline_eligible_memory_claim,
            "mechanism_ok": self.structurally_valid,
            "reasons": self.headline_eligibility_reasons,
            # detail:
            "dataset_is_real": self.dataset_is_real,
            "conditions_present": self.conditions_present,
            "confirm_size": self.confirm_size,
            "min_confirm": self.min_confirm,
            "provider_failure_rate": round(self.provider_failure_rate, 4),
            "failed_conditions": self.failed_conditions,
            "checks": self.checks,
            "structural_reasons": self.structural_reasons,
            "headline_eligibility_reasons": self.headline_eligibility_reasons,
        }


def compute_eligibility(
    checks: dict[str, bool],
    *,
    dataset_is_real: bool,
    conditions_present: Optional[list[str]] = None,
    confirm_size: int = 0,
    min_confirm: int = 20,
    provider_failure_rate: float = 0.0,
    failed_conditions: Optional[list[str]] = None,
    max_provider_failure_rate: float = 0.2,
) -> Eligibility:
    """Compute structural validity AND headline (memory-claim) eligibility.

    ``conditions_present`` is the list of comparison conditions that actually
    completed in the run. The memory-learning headline requires all of
    :data:`REQUIRED_CONDITIONS`; a run missing ``policy_memory`` (e.g. a plumbing
    run) is structurally valid at best, never headline-eligible.
    """
    c = {k: bool(v) for k, v in (checks or {}).items()}
    # tolerate the legacy key name for "runs completed"
    c.setdefault("runs_completed", bool(checks.get("baseline_and_policy_completed", False)))
    present = list(conditions_present or [])

    structural = {name: c.get(name, False) for name in STRUCTURAL_CHECKS}
    structurally_valid = all(structural.values())
    structural_reasons = [_REASONS[n] for n, ok in structural.items() if not ok]

    missing = [cond for cond in REQUIRED_CONDITIONS if cond not in present]
    confirm_ok = confirm_size >= min_confirm
    has_comparison = ("no_memory_search" in present) and ("policy_memory" in present)
    mem_checks = {name: c.get(name, False) for name in MEMORY_CLAIM_CHECKS}
    mem_checks_ok = all(mem_checks.values())

    reasons: list[str] = list(structural_reasons)
    if not dataset_is_real:
        reasons.append("dataset is a synthetic/placeholder fixture — not a benchmark headline")
    if missing:
        reasons.append(
            "main memory-learning comparison requires all of "
            f"{list(REQUIRED_CONDITIONS)}; missing: {missing}")
    elif not mem_checks_ok:                       # only meaningful when all present
        reasons += [_REASONS[n] for n, ok in mem_checks.items() if not ok]
    if not confirm_ok:
        reasons.append(
            f"CONFIRM size {confirm_size} is below the minimum {min_confirm} for a "
            "headline memory claim")

    # Provider failures: a few recorded failures are fine (still structurally
    # valid), but a high failure rate, or an entirely-failed required condition,
    # makes the run not headline-eligible.
    failed_conds = [c for c in (failed_conditions or []) if c in REQUIRED_CONDITIONS]
    failures_ok = (provider_failure_rate <= max_provider_failure_rate) and not failed_conds
    if provider_failure_rate > max_provider_failure_rate:
        reasons.append(
            f"provider failure rate {provider_failure_rate:.2f} exceeds the maximum "
            f"{max_provider_failure_rate:.2f} for a headline memory claim")
    if failed_conds:
        reasons.append(f"required condition(s) failed entirely (all tool calls failed): {failed_conds}")

    headline = bool(
        structurally_valid and dataset_is_real and has_comparison and not missing
        and confirm_ok and mem_checks_ok and failures_ok)

    all_checks = {**structural, **mem_checks,
                  "provider_failures_within_threshold": failures_ok}
    return Eligibility(
        structurally_valid=structurally_valid,
        headline_eligible_memory_claim=headline,
        dataset_is_real=dataset_is_real,
        conditions_present=present,
        checks=all_checks,
        structural_reasons=structural_reasons,
        headline_eligibility_reasons=reasons,
        confirm_size=confirm_size,
        min_confirm=min_confirm,
        provider_failure_rate=provider_failure_rate,
        failed_conditions=failed_conds,
    )
