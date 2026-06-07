#!/usr/bin/env python
"""Run the regimes-style improvement loop and record the promotion decision.

Detects dominant failure regimes on OPTIMIZE, proposes a bounded policy update,
gates it in-sample, then promotes only if it holds on the held-out CONFIRM
split. Writes results/{run_id}/policy_updates.json and prints the decision.

Use --degrade to start from a deliberately suboptimal stop threshold so the loop
has something to repair (demonstrates the accept path).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import bandit_params, base_argparser, build_synthetic, load_config, reward_weights

from regimes_probe.activegraph_pack import EventLog
from regimes_probe.regimes.action_space import PolicyConfigBundle
from regimes_probe.regimes.runner import run_regimes_loop


def main() -> int:
    ap = base_argparser("Regimes improvement loop with OPTIMIZE/CONFIRM gating.")
    ap.add_argument("--degrade", action="store_true", help="start from a suboptimal stop threshold")
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--gate-budget", type=int, default=3)
    args = ap.parse_args()
    cfg = load_config(args.config)
    items, providers, tools, _ = build_synthetic(cfg)

    base = PolicyConfigBundle(bandit=bandit_params(cfg), reward=reward_weights(cfg))
    if args.degrade:
        base.stop.stop_threshold = 0.95

    log = EventLog(run_id=args.run_id or "regimes-loop")
    res = run_regimes_loop(items, providers, tools, base_bundle=base,
                           passes=args.passes, gate_budget=args.gate_budget, log=log)

    run_id = args.run_id or "regimes-loop"
    out = Path(args.results_root) / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "policy_updates.json").write_text(json.dumps([res.to_dict()], indent=2), encoding="utf-8")

    print(f"dominant regimes : {res.proposal['rationale']}")
    print(f"proposed update  : {[ (m['target'], round(m['old'],3), '->', round(m['new'],3)) for m in res.mutation_records ]}")
    print(f"OPTIMIZE gate    : {res.promotion.optimize_gate.to_dict()}")
    print(f"CONFIRM gate     : {res.promotion.confirm_gate.to_dict()}")
    print(f"DECISION         : {'PROMOTED' if res.promotion.accepted else 'REJECTED'}")
    print(f"-> {out/'policy_updates.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
