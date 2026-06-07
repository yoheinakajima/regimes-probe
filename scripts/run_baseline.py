#!/usr/bin/env python
"""Run the no-memory baseline on CONFIRM and print metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (bandit_params, base_argparser, build_agent, build_synthetic,
                     load_config, reward_weights)

from regimes_probe.eval.harness import run_condition
from regimes_probe.eval.metrics import compute_metrics
from regimes_probe.eval.split import build_split, partition
from regimes_probe.policy.memory import PolicyMemory


def main() -> int:
    ap = base_argparser("No-memory baseline on CONFIRM.")
    ap.add_argument("--budget", type=int, default=3)
    args = ap.parse_args()
    cfg = load_config(args.config)
    items, providers, tools, _ = build_synthetic(cfg)
    agent = build_agent(cfg, tools)

    split = build_split(items, confirm_fraction=cfg.get("split", {}).get("confirm_fraction", 0.4),
                        salt=cfg.get("split", {}).get("salt", "regimes-probe-v0"))
    _, con = partition(items, split)
    mem = PolicyMemory(bandit_params(cfg))
    res = run_condition(con, agent, providers, mem, condition="no_memory",
                        budget=args.budget, weights=reward_weights(cfg), explore=False)
    m = compute_metrics(res.outcomes)
    print(json.dumps({"condition": "no_memory", "budget": args.budget,
                      "accuracy": round(m["accuracy"], 3),
                      "correct_per_tool_call": round(m["correct_per_tool_call"], 3),
                      "mean_tool_calls": round(m["mean_tool_calls"], 2)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
