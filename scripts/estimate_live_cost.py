#!/usr/bin/env python
"""Estimate the call/cost footprint of a live run (no network, no provider calls).

    python scripts/estimate_live_cost.py --optimize 10 --confirm 20 --budgets 1 3

Dollar cost is "unknown" unless per-call prices are supplied under `pricing:` in
the config (vendor prices are never hard-coded).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import base_argparser, load_config

from regimes_probe.eval.cost import estimate_calls


def main() -> int:
    ap = base_argparser("Estimate live-run call/cost footprint.")
    ap.add_argument("--optimize", type=int, default=10)
    ap.add_argument("--confirm", type=int, default=20)
    ap.add_argument("--budgets", type=int, nargs="+", default=None)
    ap.add_argument("--passes", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    budgets = args.budgets or cfg.get("budgets", [1, 3, 5, 10])
    mem = cfg.get("memory", {})

    est = estimate_calls(
        n_optimize=args.optimize, n_confirm=args.confirm, budgets=budgets,
        passes=args.passes or mem.get("experience_passes", 4),
        experience_budget=mem.get("experience_budget", 5),
        judge=cfg.get("grading", {}).get("judge", "exact"),
        prices=cfg.get("pricing"),
    )
    print("Live-run call/cost estimate (no providers called):\n")
    print(est.render())
    if est.estimated_cost_usd == "unknown":
        print("\n  (set `pricing: {answerer_per_call, tool_per_call, judge_per_call}` "
              "in config to get a dollar estimate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
