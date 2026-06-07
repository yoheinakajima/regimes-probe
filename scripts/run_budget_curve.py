#!/usr/bin/env python
"""Produce a budget curve (no_memory vs policy_memory) across budget caps."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (bandit_params, base_argparser, build_agent, build_synthetic,
                     load_config, reward_weights)

from regimes_probe.datasets.synthetic import SyntheticBrowseAdapter
from regimes_probe.eval.harness import experience_phase, run_condition
from regimes_probe.eval.metrics import compute_metrics
from regimes_probe.eval.split import build_split, partition
from regimes_probe.policy.memory import PolicyMemory


def main() -> int:
    args = base_argparser("Budget curve across budget caps.").parse_args()
    cfg = load_config(args.config)
    items, providers, tools, _ = build_synthetic(cfg)
    agent = build_agent(cfg, tools)
    weights = reward_weights(cfg)
    params = bandit_params(cfg)
    budgets = cfg.get("budgets", [1, 3, 5, 10])

    split = build_split(items, confirm_fraction=cfg.get("split", {}).get("confirm_fraction", 0.4))
    opt, con = partition(items, split)
    mem = PolicyMemory(params.copy(), nearest_k=cfg.get("memory", {}).get("nearest_k", 8))
    experience_phase(opt, agent, providers, mem,
                     budget=cfg.get("memory", {}).get("experience_budget", 5),
                     passes=cfg.get("memory", {}).get("experience_passes", 4),
                     weights=weights, dataset_version=SyntheticBrowseAdapter().version())
    snap = mem.snapshot()

    print(f"{'budget':>6} | {'no_mem cptc':>12} | {'policy cptc':>12} | {'no_mem acc':>10} | {'policy acc':>10}")
    print("-" * 64)
    for b in budgets:
        base = compute_metrics(run_condition(con, agent, providers, PolicyMemory(params.copy()),
                               condition="no_memory", budget=b, weights=weights).outcomes)
        pol = compute_metrics(run_condition(con, agent, providers,
                              PolicyMemory.from_snapshot(snap, frozen=True),
                              condition="policy_memory", budget=b, weights=weights).outcomes)
        print(f"{b:>6} | {base['correct_per_tool_call']:>12.3f} | {pol['correct_per_tool_call']:>12.3f} | "
              f"{base['accuracy']:>10.3f} | {pol['accuracy']:>10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
