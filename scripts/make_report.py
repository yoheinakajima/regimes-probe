#!/usr/bin/env python
"""End-to-end synthetic demo: baseline vs frozen policy memory, with budget
curve, statistical tests, a replay check, and a full report.

Produces results/{run_id}/ exactly as documented in docs/REPORTING.md. Runs with
NO keys and NO network. Thin wrapper over the shared no-key pipeline; see also
scripts/run_synthetic_full.py for a step-by-step STATUS print.

    python scripts/make_report.py --run-id demo
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import base_argparser, build_agent, build_synthetic, full_pipeline, load_config

from regimes_probe.datasets.synthetic import SyntheticBrowseAdapter


def main() -> int:
    args = base_argparser("Full synthetic report (baseline vs policy memory).").parse_args()
    cfg = load_config(args.config)
    run_id = args.run_id or "demo"
    items, providers, tools, search_tools = build_synthetic(cfg)
    agent = build_agent(cfg, tools)

    summary = full_pipeline(cfg, items, providers, search_tools, agent,
                            run_id=run_id, results_root=args.results_root,
                            dataset_label="synthetic_browse",
                            dataset_version=SyntheticBrowseAdapter().version())
    print(f"wrote {summary['run_dir']}")
    print("Headline (correct_per_tool_call) by budget:")
    for b in summary["budgets"]:
        h = summary["headline"][b]
        print(f"  budget {b:>2}: no_memory={h['no_memory']['correct_per_tool_call']:.3f}  "
              f"policy_memory={h['policy_memory']['correct_per_tool_call']:.3f}")
    print(f"replay projection_matches={summary['replay'].get('projection_matches')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
