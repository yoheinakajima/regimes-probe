#!/usr/bin/env python
"""Human-readable failure view for a run (no network).

    python scripts/debug_run_failures.py results/live/<run_id> --limit 10
    python scripts/debug_run_failures.py results/live/<run_id> --condition no_memory_search --all

Reads results/{run_id}/debug_questions.jsonl (bounded previews) and prints, per
failed question: the question/gold/model-answer previews, tools used, top
evidence, any failed-tool errors, and the inferred failure seam.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _load(run_dir: Path) -> list[dict]:
    p = run_dir / "debug_questions.jsonl"
    if not p.exists():
        raise SystemExit(f"no debug_questions.jsonl in {run_dir} "
                         "(re-run with a build that writes it).")
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Compact failure view for a run.")
    ap.add_argument("run_dir")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--condition", default=None, help="filter to one condition")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--all", action="store_true", help="include correct rows too")
    ap.add_argument("--seam", default=None, help="filter to one failure seam")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    rows = _load(run_dir)

    rows = [r for r in rows
            if (args.all or not r.get("correct"))
            and (args.condition is None or r.get("condition") == args.condition)
            and (args.budget is None or r.get("budget") == args.budget)
            and (args.seam is None or r.get("failure_seam") == args.seam)]

    seam_counts = Counter(r.get("failure_seam", "unknown") for r in rows)
    print(f"=== {run_dir} — {len(rows)} shown ===")
    print("failure seams:", dict(seam_counts.most_common()))
    print()

    for r in rows[: args.limit]:
        print(f"[{r['item_id']}] {r['condition']}@b{r['budget']}  "
              f"correct={r['correct']} abstained={r['abstained']}  seam={r['failure_seam']}  "
              f"regime={r.get('regime','')} regimes={r.get('regime_names', [])}")
        print(f"  Q   : {r['question_preview']}")
        print(f"  gold: {r['gold_preview']}")
        print(f"  pred: {r['prediction_preview'] or '(none / abstained)'}")
        print(f"  tools: {r['tool_sequence']}  providers: {r['provider_names']}  "
              f"n_results={r['n_results']}")
        print(f"  flags: support_found={r['support_found']} found_hit={r['found_hit']} "
              f"authority_ok={r['authority_ok']} contradiction={r['contradiction']} "
              f"failed_tool_calls={r['failed_tool_calls']} "
              f"contaminated_results={r.get('contaminated_results', 0)}")
        for c in r.get("calls", []):
            print(f"  query [{c.get('tool')}/{c.get('query_arm')}]: "
                  f"{c.get('query_preview', '')!r}  "
                  f"(n_results={c.get('n_results', 0)}, "
                  f"contaminated={c.get('contaminated_results', 0)}"
                  + (f", clues={c.get('clue_ids')}" if c.get('clue_ids') else "") + ")")
        for e in r.get("failed_tool_errors", []):
            print(f"  ⚠ tool error [{e['tool']}]: {e.get('error_type')} "
                  f"{e.get('status_code')} — {e.get('message_preview')}")
        for ev in r.get("evidence", [])[:3]:
            mark = "✓" if ev.get("supports") else " "
            tags = []
            if ev.get("contains_gold"):
                tags.append("HAS-GOLD")
            if ev.get("benchmark_contaminated"):
                tags.append(f"CONTAMINATED({ev.get('contamination_reason')})")
            tagstr = ("  [" + ", ".join(tags) + "]") if tags else ""
            print(f"  {mark} {ev.get('url','')}{tagstr}")
            if ev.get("title_preview"):
                print(f"      {ev['title_preview']}")
            if ev.get("snippet_preview"):
                print(f"      {ev['snippet_preview']}")
        print()

    if not rows:
        print("(no matching rows — try --all or different filters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
