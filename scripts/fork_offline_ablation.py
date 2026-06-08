#!/usr/bin/env python
"""Offline forked ablation — rerun policy variants on CACHED outcomes (no spend).

    python scripts/fork_offline_ablation.py results/live/<run_id> \
        --out <new_run_id> --policy-config path/to/overrides.yaml

Reuses the parent run's recording cache in REPLAY mode: every provider/model/tool
outcome is served from cache; any cache miss RAISES (it never calls a provider).
The forked run is written with offline_fork=true + parent_run_id and is never
headline-eligible. Use it to compare reward/router/stop variants without
re-spending. See docs/OFFLINE_FORK_ABLATIONS.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regimes_probe.live.cache import ReplayMiss
from regimes_probe.live.providers import NotArmed


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline forked ablation (no provider calls).")
    ap.add_argument("run_dir", help="parent run dir, e.g. results/live/<run_id>")
    ap.add_argument("--out", required=True, help="new run id for the fork")
    ap.add_argument("--results-root", default=str(ROOT / "results" / "live"))
    ap.add_argument("--policy-config", default=None,
                    help="YAML of policy/reward/router/bandit overrides to apply")
    ap.add_argument("--cache", default=None,
                    help="recording cache path (default: parent manifest's cache.path)")
    ap.add_argument("--conditions", default="no_memory_search,policy_memory")
    ap.add_argument("--budgets", default=None, help="comma-separated; default: parent budgets")
    ap.add_argument("--answerer", default="deterministic", choices=["deterministic", "live"],
                    help="'live' replays the parent's cached model answers (for live-origin runs)")
    args = ap.parse_args()

    overrides = {}
    if args.policy_config:
        import yaml
        overrides = yaml.safe_load(Path(args.policy_config).read_text(encoding="utf-8")) or {}
    budgets = ([int(b) for b in args.budgets.split(",") if b.strip()] if args.budgets else None)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    from regimes_probe.live.fork import fork_offline
    try:
        result = fork_offline(
            args.run_dir, out_run_id=args.out, results_root=args.results_root,
            cache_path=args.cache, policy_overrides=overrides,
            conditions=conditions, budgets=budgets, answerer=args.answerer)
    except FileNotFoundError as exc:
        print(f"=== fork_offline_ablation: REFUSING ===\n{exc}")
        return 2
    except (ReplayMiss, NotArmed) as exc:
        print("=== fork_offline_ablation: REFUSING (cache miss; would call a provider) ===")
        print(f"{exc}")
        print("An offline fork can only change parameters that do NOT alter the realized "
              "query/model requests. No providers were called.")
        return 2

    el = result["eligibility"]
    print(f"=== offline fork -> {result['run_dir']} ===")
    print(f"parent_run_id recorded; offline_fork=true")
    print(f"cache: {result['cache']}")
    print(f"structurally_valid={el.get('structurally_valid')}  "
          f"headline_eligible_memory_claim={el.get('headline_eligible_memory_claim')} "
          f"(offline forks are never headline-eligible)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
