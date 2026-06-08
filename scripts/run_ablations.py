#!/usr/bin/env python
"""Run ablation variants on the synthetic fixture (no keys, no network).

    python scripts/run_ablations.py                    # run all ablations
    python scripts/run_ablations.py --ablation routing_only
    python scripts/run_ablations.py --list

Ablations are defined in config/default.yaml under `ablations:` and override
parts of the config (query/stop policy, reward preset, frozen vs online CONFIRM).
Each writes a full report under results/ablations/<name>/. The point is to verify
plumbing and reporting — NOT to make a benchmark claim.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    base_argparser,
    build_agent,
    build_synthetic,
    full_pipeline,
    load_config,
)

from regimes_probe.datasets.synthetic import SyntheticBrowseAdapter


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def main() -> int:
    ap = base_argparser("Run ablation variants on the synthetic fixture.")
    ap.add_argument("--ablation", default=None, help="single ablation name (default: all)")
    ap.add_argument("--list", action="store_true", help="list available ablations and exit")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ablations = cfg.get("ablations", {})

    if args.list:
        for name, spec in ablations.items():
            tag = " (NON-HEADLINE)" if spec.get("_non_headline") else ""
            print(f"  {name}{tag}: {[k for k in spec if not k.startswith('_')]}")
        return 0

    names = [args.ablation] if args.ablation else list(ablations)
    results_root = Path(args.results_root) / "ablations"

    print(f"{'ablation':<38} | {'budget':>6} | {'no_mem_search':>13} | {'policy':>7} | headline")
    print("-" * 90)
    for name in names:
        if name not in ablations:
            print(f"!! unknown ablation: {name}")
            continue
        spec = ablations[name]
        ab_cfg = _deep_merge(cfg, spec)
        online = bool(spec.get("_online_confirm"))
        items, providers, tools, search_tools = build_synthetic(ab_cfg)
        agent = build_agent(ab_cfg, tools)
        summary = full_pipeline(ab_cfg, items, providers, search_tools, agent,
                                run_id=name, results_root=str(results_root),
                                dataset_label="synthetic_browse",
                                dataset_version=SyntheticBrowseAdapter().version(),
                                online_confirm=online)
        he = summary["eligibility"]["headline_eligible"]
        flag = "online(NON-HEADLINE)" if online else f"eligible={he}"
        gb = summary["gate_budget"]
        h = summary["headline"][gb]
        print(f"{name:<38} | {gb:>6} | "
              f"{h['no_memory_search']['correct_per_tool_call']:>13.3f} | "
              f"{h['policy_memory']['correct_per_tool_call']:>7.3f} | {flag}")
    print(f"\nReports under {results_root}/<name>/ . Synthetic plumbing check only; "
          "no benchmark claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
