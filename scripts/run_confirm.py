#!/usr/bin/env python
"""Run the frozen policy-memory condition on CONFIRM (loads a snapshot).

Compares no_memory vs policy_memory vs random_memory at one budget and prints
metrics + a McNemar test. Loads results/{run_id}/memory_snapshot.json if present,
otherwise runs an experience phase first.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (bandit_params, base_argparser, build_agent, build_synthetic,
                     load_config, reward_weights)

from regimes_probe.datasets.synthetic import SyntheticBrowseAdapter
from regimes_probe.eval.harness import experience_phase, run_condition
from regimes_probe.eval.metrics import compute_metrics
from regimes_probe.eval.significance import mcnemar
from regimes_probe.eval.split import build_split, partition
from regimes_probe.policy.memory import PolicyMemory, PolicyMemorySnapshot


def main() -> int:
    ap = base_argparser("Frozen policy-memory CONFIRM evaluation.")
    ap.add_argument("--budget", type=int, default=3)
    args = ap.parse_args()
    cfg = load_config(args.config)
    items, providers, tools, _ = build_synthetic(cfg)
    agent = build_agent(cfg, tools)
    weights = reward_weights(cfg)
    params = bandit_params(cfg)

    split = build_split(items, confirm_fraction=cfg.get("split", {}).get("confirm_fraction", 0.4),
                        salt=cfg.get("split", {}).get("salt", "regimes-probe-v0"))
    opt, con = partition(items, split)

    snap_path = Path(args.results_root) / (args.run_id or "experience") / "memory_snapshot.json"
    if snap_path.exists():
        snap = PolicyMemorySnapshot.from_dict(json.loads(snap_path.read_text()))
    else:
        mem = PolicyMemory(params.copy(), nearest_k=cfg.get("memory", {}).get("nearest_k", 8))
        experience_phase(opt, agent, providers, mem,
                         budget=cfg.get("memory", {}).get("experience_budget", 5),
                         passes=cfg.get("memory", {}).get("experience_passes", 4),
                         weights=weights, dataset_version=SyntheticBrowseAdapter().version())
        snap = mem.snapshot()

    base = run_condition(con, agent, providers, PolicyMemory(params.copy()),
                         condition="no_memory", budget=args.budget, weights=weights).outcomes
    pol = run_condition(con, agent, providers, PolicyMemory.from_snapshot(snap, frozen=True),
                        condition="policy_memory", budget=args.budget, weights=weights).outcomes

    bmap = {o.item_id: o for o in base}
    pmap = {o.item_id: o for o in pol}
    ids = [i for i in bmap if i in pmap]
    mc = mcnemar([bmap[i].correct for i in ids], [pmap[i].correct for i in ids])
    print(json.dumps({
        "budget": args.budget,
        "no_memory": {k: round(compute_metrics(base)[k], 3) for k in ("accuracy", "correct_per_tool_call", "mean_tool_calls")},
        "policy_memory": {k: round(compute_metrics(pol)[k], 3) for k in ("accuracy", "correct_per_tool_call", "mean_tool_calls")},
        "mcnemar": mc.to_dict(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
