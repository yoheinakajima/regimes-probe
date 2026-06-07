#!/usr/bin/env python
"""Run the experience phase on OPTIMIZE and freeze a policy memory snapshot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (bandit_params, base_argparser, build_agent, build_synthetic,
                     load_config, reward_weights)

from regimes_probe.datasets.synthetic import SyntheticBrowseAdapter
from regimes_probe.eval.harness import experience_phase
from regimes_probe.eval.split import build_split, partition
from regimes_probe.policy.consolidation import consolidate
from regimes_probe.policy.memory import PolicyMemory


def main() -> int:
    args = base_argparser("Experience phase on OPTIMIZE; freeze snapshot.").parse_args()
    cfg = load_config(args.config)
    items, providers, tools, _ = build_synthetic(cfg)
    agent = build_agent(cfg, tools)
    weights = reward_weights(cfg)
    mem_cfg = cfg.get("memory", {})

    split = build_split(items, confirm_fraction=cfg.get("split", {}).get("confirm_fraction", 0.4),
                        salt=cfg.get("split", {}).get("salt", "regimes-probe-v0"))
    opt_items, _ = partition(items, split)

    mem = PolicyMemory(bandit_params(cfg), nearest_k=mem_cfg.get("nearest_k", 8))
    experience_phase(opt_items, agent, providers, mem,
                     budget=mem_cfg.get("experience_budget", 5),
                     passes=mem_cfg.get("experience_passes", 4),
                     weights=weights, dataset_version=SyntheticBrowseAdapter().version())
    consolidate(mem)
    snap = mem.snapshot(meta={"n_optimize": len(opt_items)})

    run_id = args.run_id or "experience"
    out = Path(args.results_root) / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "memory_snapshot.json").write_text(json.dumps(snap.to_dict(), indent=2), encoding="utf-8")
    print(f"experience over {len(opt_items)} items, {len(mem.traces)} traces, "
          f"{len(mem.fragments)} fragments -> {out/'memory_snapshot.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
