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
    "same_conditions": "baseline and policy differ only in memory access",
    "replay_passed": "the graph is a deterministic projection of the event log (replay)",
    "baseline_and_policy_completed": "baseline and policy runs both completed",
    "budget_enforced": "tool-call budgets were enforced",
    "no_live_updates_during_confirm": "no live memory updates during CONFIRM",
}


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python scripts/generate_claims.py results/{run_id}/report.json")
        return 2
    report_path = Path(sys.argv[1])
    report = json.loads(report_path.read_text(encoding="utf-8"))

    he = bool(report.get("headline_eligible"))
    elig = report.get("eligibility", {})
    checks = elig.get("checks", {})
    reasons = elig.get("reasons", [])
    meta = report.get("meta", {})
    dataset = meta.get("dataset", "unknown")
    sig = report.get("significance", {})
    cells = {(c["condition"], c["budget"]): c["metrics"] for c in report.get("conditions", [])}

    L: list[str] = []
    A = L.append
    A(f"# Claim candidates — `{report.get('run_id')}` (dataset: {dataset})\n")
    A(f"> headline_eligible = **{he}**. "
      + ("Performance claims are permitted (review before publishing)."
         if he else "Performance claims are REFUSED below.") + "\n")

    A("## Verified (structural)")
    verified = [c for c, ok in checks.items() if ok]
    if verified:
        for c in verified:
            A(f"- {_CHECK_CLAIM.get(c, c)}.")
    else:
        A("- (none)")
    A("")

    A("## Partially verified")
    if ("closed_book", 0) in cells:
        acc = cells[("closed_book", 0)].get("accuracy", 0.0)
        A(f"- A closed-book baseline ran (0 tool calls); intrinsic-knowledge "
          f"accuracy estimate = {acc:.3f} on this dataset.")
    A("- Mechanism (routing/query/stop learning) executes end to end and is "
      "recorded/replayable; generalization to real data is not established here.")
    A("")

    A("## Performance claims")
    if he:
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
        A("- **REFUSED**: headline_eligible is false; no performance/benchmark claim "
          "may be generated from this run.")
        for r in reasons:
            A(f"  - reason: {r}")

    A("\n## Not supported / not headline eligible")
    if not he:
        A(f"- No claim about `{dataset}` performance is supported by this run.")
        if not elig.get("dataset_is_real", False):
            A("- This is a synthetic/placeholder fixture: it demonstrates the "
              "mechanism only, not benchmark performance.")
    failed = [c for c, ok in checks.items() if not ok]
    for c in failed:
        A(f"- NOT verified: {_CHECK_CLAIM.get(c, c)}.")
    A("")

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
