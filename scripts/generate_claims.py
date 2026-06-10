#!/usr/bin/env python
"""Generate CONSERVATIVE claim candidates from a report (no network).

    python scripts/generate_claims.py results/demo/report.json

Emits docs/STATUS-style claim candidates (verified / partially verified / not
supported / not headline eligible / limitations). It REFUSES to emit any
performance claim when headline_eligible is false — a synthetic or
structurally-invalid run can never yield a benchmark claim. Output is written to
results/{run_id}/claims_candidates.md and printed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_CHECK_CLAIM = {
    "optimize_confirm_disjoint": "OPTIMIZE and CONFIRM splits are disjoint",
    "confirm_memory_frozen": "CONFIRM used a frozen policy-memory snapshot",
    "no_answer_leakage": "policy memory contains no answer text (leakage check passed)",
    "same_conditions": "no_memory_search and policy_memory differ only in memory access",
    "replay_passed": "the graph is a deterministic projection of the event log (replay)",
    "runs_completed": "the requested runs completed",
    "budget_enforced": "tool-call budgets were enforced",
    "no_live_updates_during_confirm": "no live memory updates during CONFIRM",
    "provider_failures_within_threshold": "provider failure rate within the headline threshold",
}
# Checks that only make sense once a memory comparison (policy_memory) was run.
_MEMORY_ONLY_CHECKS = {"same_conditions", "confirm_memory_frozen", "no_live_updates_during_confirm"}


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python scripts/generate_claims.py results/{run_id}/report.json")
        return 2
    report_path = Path(sys.argv[1])
    report = json.loads(report_path.read_text(encoding="utf-8"))

    elig = report.get("eligibility", {})
    # Prefer the flat first-class verdict object when present (see eval/eligibility.py).
    verdict = report.get("eligibility_verdict", {})
    structurally_valid = bool(verdict.get("structurally_valid",
                              report.get("structurally_valid",
                                         elig.get("structurally_valid", elig.get("mechanism_ok")))))
    headline = bool(verdict.get("headline_eligible_memory_claim",
                    report.get("headline_eligible_memory_claim",
                               elig.get("headline_eligible_memory_claim", report.get("headline_eligible")))))
    # 5i-N: a debug/dev slice is NEVER headline eligible — claims stay conservative even if a
    # tiny slice happened to satisfy other gates. (Existing headline criteria still apply too.)
    from regimes_probe.eval.debug_slice import is_debug_slice_report
    if is_debug_slice_report(report):
        headline = False
    reasons = (verdict.get("reasons") or report.get("headline_eligibility_reasons")
               or elig.get("headline_eligibility_reasons") or elig.get("reasons") or [])
    checks = elig.get("checks", {})
    meta = report.get("meta", {})
    dataset = meta.get("dataset", "unknown")
    dataset_is_real = bool(elig.get("dataset_is_real"))
    sig = report.get("significance", {})
    cells = {(c["condition"], c["budget"]): c["metrics"] for c in report.get("conditions", [])}
    present = report.get("conditions_present") or elig.get("conditions_present") \
        or sorted({c for (c, _b) in cells})
    has_policy = "policy_memory" in present

    L: list[str] = []
    A = L.append
    A(f"# Claim candidates — `{report.get('run_id')}` (dataset: {dataset})\n")
    A(f"> structurally_valid = **{structurally_valid}**; "
      f"headline_eligible_memory_claim = **{headline}**.")
    A(f"> conditions present: {present}.\n")
    if structurally_valid and dataset_is_real and not has_policy:
        A(f"**Real `{dataset}` PLUMBING run completed** (conditions: {present}). This "
          "exercises the pipeline on real data but is NOT a memory-learning result.\n")
    A(("Performance/benchmark claims are permitted (review before publishing)."
       if headline else "Memory-performance claims are REFUSED below.") + "\n")

    A("## Verified (structural)")
    # Don't surface memory-comparison checks as 'verified' when no policy_memory ran.
    verified = [c for c, ok in checks.items()
                if ok and (has_policy or c not in _MEMORY_ONLY_CHECKS)]
    for c in verified:
        A(f"- {_CHECK_CLAIM.get(c, c)}.")
    if not verified:
        A("- (none)")
    A("")

    A("## Partially verified")
    if ("closed_book", 0) in cells:
        acc = cells[("closed_book", 0)].get("accuracy", 0.0)
        A(f"- A closed-book baseline ran (0 tool calls); intrinsic-knowledge "
          f"accuracy estimate = {acc:.3f} on this dataset.")
    A("- Mechanism (routing/query/stop learning) executes end to end and is "
      "recorded/replayable; generalization is not established by this run.")
    A("")

    A("## Memory-performance claims")
    if not has_policy:
        A("- **REFUSED**: `policy_memory` was NOT run, so there is NO memory comparison. "
          "The main regimes-probe memory-learning claim cannot be made from this run.")
        A(f"  - conditions present: {present} (missing `policy_memory`"
          + ("" if "random_memory" in present else " and `random_memory`") + ").")
        A("  - Run all four conditions (closed_book, no_memory_search, random_memory, "
          "policy_memory) to evaluate the memory headline.")
    elif headline:
        for b in sorted({bud for (c, bud) in cells if c == "policy_memory"}):
            base = cells.get(("no_memory_search", b), {}).get("correct_per_tool_call")
            pol = cells.get(("policy_memory", b), {}).get("correct_per_tool_call")
            if base is not None and pol is not None:
                A(f"- CANDIDATE: at budget {b}, frozen policy memory changes "
                  f"correct_per_tool_call {base:.3f} → {pol:.3f} on CONFIRM "
                  f"(verify CI/McNemar before claiming).")
        mc = sig.get("mcnemar", {})
        if mc:
            A(f"- CANDIDATE: McNemar @ budget {sig.get('budget')}: "
              f"{mc.get('c_only_treatment_correct')} wrong→correct vs "
              f"{mc.get('b_only_baseline_correct')} reverse (p={mc.get('p_value')}).")
    else:
        A("- **REFUSED**: not headline-eligible; no memory-performance claim may be made.")
        for r in reasons:
            A(f"  - reason: {r}")

    A("\n## Not supported / not headline eligible")
    if not headline:
        A(f"- No memory-learning claim about `{dataset}` is supported by this run.")
        if not dataset_is_real:
            A("- This is a synthetic/placeholder fixture: it demonstrates the "
              "mechanism only, not benchmark performance.")
        for r in reasons:
            A(f"- reason: {r}")

    A("## Limitations")
    for lim in meta.get("limitations", []):
        A(f"- {lim}")
    for nc in meta.get("not_claimed", []):
        A(f"- not claimed: {nc}")
    A("")

    text = "\n".join(L)
    out = report_path.parent / "claims_candidates.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n(wrote {out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
