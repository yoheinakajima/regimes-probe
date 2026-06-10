#!/usr/bin/env python
"""Compare two report.json runs (no keys, no network).

    python scripts/compare_runs.py results/run_a/report.json results/run_b/report.json

Compares per-(condition,budget) metrics, budget curves, headline eligibility,
and — when per_question.csv sits next to each report — McNemar flips for a focus
condition/budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regimes_probe.eval.significance import mcnemar

_METRICS = [
    "accuracy", "correct_per_tool_call", "mean_tool_calls", "first_tool_hit_rate",
    "stale_source_error_rate", "false_stop_rate", "over_search_rate",
    "correct_per_dollar", "correct_per_second",
]


def _load(report_path: str):
    p = Path(report_path)
    report = json.loads(p.read_text(encoding="utf-8"))
    cells = {(c["condition"], c["budget"]): c["metrics"] for c in report.get("conditions", [])}
    pq = p.parent / "per_question.csv"
    return report, cells, (pq if pq.exists() else None)


def _read_pq(path: Path, condition: str, budget: int) -> dict[str, int]:
    out: dict[str, int] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("condition") == condition and int(row.get("budget", -1)) == budget:
                out[row["item_id"]] = int(row.get("correct", 0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare two report.json runs.")
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--condition", default="policy_memory", help="focus condition for McNemar")
    ap.add_argument("--budget", type=int, default=3, help="focus budget for McNemar")
    args = ap.parse_args()

    rep_a, cells_a, pq_a = _load(args.run_a)
    rep_b, cells_b, pq_b = _load(args.run_b)

    print(f"A = {args.run_a}\nB = {args.run_b}\n")
    print(f"headline_eligible: A={rep_a.get('headline_eligible')}  B={rep_b.get('headline_eligible')}")
    print(f"dataset: A={rep_a.get('meta',{}).get('dataset')}  B={rep_b.get('meta',{}).get('dataset')}\n")

    keys = sorted(set(cells_a) & set(cells_b), key=lambda k: (k[0], k[1]))
    print("per-(condition,budget) metric deltas (B - A):")
    for cond, budget in keys:
        ma, mb = cells_a[(cond, budget)], cells_b[(cond, budget)]
        print(f"\n  [{cond} @ budget {budget}]")
        for m in _METRICS:
            if m in ma or m in mb:
                a, b = ma.get(m, 0.0), mb.get(m, 0.0)
                print(f"    {m:<32} A={a:>8.3f}  B={b:>8.3f}  Δ={b - a:+.3f}")

    # 5i-N: mechanism-detector deltas (DEBUG labels only — never a benchmark claim). These
    # surface loop-closure / pollution / target-binding mechanism movement on the same items.
    from regimes_probe.eval.debug_slice import (
        DEBUG_SLICE_DETECTOR_KEYS, detector_deltas, is_debug_slice_report)
    if is_debug_slice_report(rep_a) or is_debug_slice_report(rep_b):
        print("\n[DEBUG/DEV slice — mechanism deltas only, NOT a benchmark claim]")
    print("\nmechanism-detector deltas (B - A) — debug labels only:")
    for cond, budget in keys:
        d = detector_deltas(cells_a[(cond, budget)], cells_b[(cond, budget)])
        if not d:
            continue
        print(f"\n  [{cond} @ budget {budget}]")
        for k in DEBUG_SLICE_DETECTOR_KEYS:
            if k in d:
                print(f"    {k:<52} A={d[k]['a']!s:>6}  B={d[k]['b']!s:>6}  Δ={d[k]['delta']:+}")

    # Budget curve for policy_memory (the headline condition), if present.
    print("\nbudget curve — correct_per_tool_call (condition=policy_memory):")
    budgets = sorted({bud for (c, bud) in set(cells_a) | set(cells_b) if c == "policy_memory"})
    for bud in budgets:
        a = cells_a.get(("policy_memory", bud), {}).get("correct_per_tool_call")
        b = cells_b.get(("policy_memory", bud), {}).get("correct_per_tool_call")
        print(f"    budget {bud:>2}: A={a}  B={b}")

    # McNemar flips between A and B for the focus condition/budget.
    if pq_a and pq_b:
        ca = _read_pq(pq_a, args.condition, args.budget)
        cb = _read_pq(pq_b, args.condition, args.budget)
        ids = sorted(set(ca) & set(cb))
        if ids:
            mc = mcnemar([bool(ca[i]) for i in ids], [bool(cb[i]) for i in ids])
            print(f"\nMcNemar ({args.condition} @ budget {args.budget}, A vs B, n={len(ids)}):")
            print(f"    A-only-correct={mc.b}  B-only-correct={mc.c}  "
                  f"stat={mc.statistic:.3f}  p={mc.p_value:.4f}")
        else:
            print("\nMcNemar: no overlapping item_ids for the focus condition/budget.")
    else:
        print("\nMcNemar: per_question.csv not found next to one/both reports — skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
