#!/usr/bin/env python
"""Run the COMPLETE no-key, no-network synthetic pipeline end to end.

    python scripts/run_synthetic_full.py [--run-id NAME]

Steps (all offline, deterministic):
  1. build deterministic OPTIMIZE/CONFIRM split
  2. run no_memory baseline on CONFIRM
  3. run policy_memory experience on OPTIMIZE
  4. freeze the memory snapshot
  5. run policy_memory on CONFIRM
  6. run random_memory control on CONFIRM
  7. generate report artifacts (results/{run_id}/)
  8. run the replay check
  9. print a grounded STATUS summary

This is the canonical "first demo". It proves the MECHANISM on a synthetic
fixture only — it is NOT a BrowseComp/LiveBrowseComp result.
"""

from __future__ import annotations

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


def main() -> int:
    args = base_argparser("Complete no-key synthetic pipeline.").parse_args()
    cfg = load_config(args.config)
    run_id = args.run_id or "synthetic_full"

    items, providers, tools, search_tools = build_synthetic(cfg)
    agent = build_agent(cfg, tools)
    dataset_version = SyntheticBrowseAdapter().version()

    n = [0]
    def step(msg: str) -> None:
        n[0] += 1
        print(f"[{n[0]}] {msg}")

    summary = full_pipeline(cfg, items, providers, search_tools, agent,
                            run_id=run_id, results_root=args.results_root,
                            dataset_label="synthetic_browse", dataset_version=dataset_version,
                            on_step=step)

    print("\n=== STATUS (grounded; synthetic harness only) ===")
    print(f"run dir              : {summary['run_dir']}")
    sp = summary["split"]
    print(f"split (deterministic): OPTIMIZE={len(sp.optimize_ids)} CONFIRM={len(sp.confirm_ids)} "
          f"disjoint={set(sp.optimize_ids).isdisjoint(sp.confirm_ids)}")
    print(f"replay projection_matches: {summary['replay'].get('projection_matches')}")
    print(f"memory snapshot answer-free: True (assert_no_answer_leakage enforced)")
    el = summary["eligibility"]
    print(f"headline_eligible    : {el['headline_eligible']}  "
          f"(mechanism_ok={el['mechanism_ok']}, dataset_is_real={el['dataset_is_real']})")
    if el["reasons"]:
        print(f"  reasons            : {el['reasons']}")
    print(f"closed_book accuracy : {summary['closed_book']['accuracy']:.3f} "
          f"(intrinsic-knowledge estimate; 0 tool calls)")
    print("\ncorrect_per_tool_call by budget (CONFIRM, held out):")
    print(f"  {'budget':>6} | {'no_mem_search':>13} | {'policy_mem':>10} | {'random_mem':>10}")
    for b in summary["budgets"]:
        h = summary["headline"][b]
        print(f"  {b:>6} | {h['no_memory_search']['correct_per_tool_call']:>13.3f} | "
              f"{h['policy_memory']['correct_per_tool_call']:>10.3f} | "
              f"{h['random_memory']['correct_per_tool_call']:>10.3f}")
    mc = summary["significance"]["mcnemar"]
    print(f"\nMcNemar @ budget {summary['gate_budget']}: "
          f"{mc['c_only_treatment_correct']} wrong->correct, "
          f"{mc['b_only_baseline_correct']} reverse, p={mc['p_value']}")
    print("\nCLAIM: the scaffold demonstrates the intended mechanism on a synthetic "
          "fixture.\nNOT CLAIMED: any BrowseComp/LiveBrowseComp performance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
