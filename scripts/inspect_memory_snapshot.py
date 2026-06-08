#!/usr/bin/env python
"""Inspect a frozen policy-memory snapshot (no keys, no network).

    python scripts/inspect_memory_snapshot.py results/demo/memory_snapshot.json

Prints fragment/trace counts, top tool & query-template priors, stop/verify arm
priors and bandit thresholds, a leakage scan, and example numeric policy
fragments — confirming the snapshot stores rewards/priors, not answers. Very
useful to run before/after a live run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regimes_probe.policy.policy_fragment import FORBIDDEN_KEYS, assert_no_answer_leakage


def _aggregate_arms(bandit: dict) -> list[tuple[str, float, float]]:
    """Aggregate (arm, weighted_mean, total_count) across clusters."""
    acc: dict[str, list[float]] = {}
    for arms in bandit.get("ctx", {}).values():
        for arm, st in arms.items():
            slot = acc.setdefault(arm, [0.0, 0.0, 0.0])  # sum(mean*w), sum(w), sum(n)
            slot[0] += st.get("mean", 0.0) * st.get("w", 0.0)
            slot[1] += st.get("w", 0.0)
            slot[2] += st.get("n", 0)
    rows = [(arm, (s[0] / s[1] if s[1] else 0.0), s[2]) for arm, s in acc.items()]
    rows.sort(key=lambda r: (-r[1], r[0]))
    return rows


def _print_arms(title: str, bandit: dict, top: int = 8) -> None:
    print(f"\n{title}")
    rows = _aggregate_arms(bandit)
    if not rows:
        print("  (none)")
        return
    for arm, mean, n in rows[:top]:
        print(f"  {arm:<32} mean_reward={mean:+.3f}  pulls={int(n)}")


def _scan_answer_like(snap: dict) -> tuple[bool, list[str]]:
    found: list[str] = []

    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                if str(k).lower() in FORBIDDEN_KEYS:
                    found.append(f"{path}.{k}")
                walk(v, f"{path}.{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")

    walk(snap, "snapshot")
    return (len(found) == 0, found)


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect a policy-memory snapshot.")
    ap.add_argument("snapshot", help="path to memory_snapshot.json")
    args = ap.parse_args()
    snap = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))

    bandits = snap.get("bandits", {})
    traces = snap.get("traces", [])
    fragments = snap.get("fragments", {})
    params = snap.get("params", {})

    print(f"=== policy-memory snapshot: {args.snapshot} ===")
    print(f"policy fragments      : {len(fragments)}")
    print(f"raw traces represented: {len(traces)}")
    print(f"signature clusters    : {len(bandits.get('tool', {}).get('ctx', {}))}")

    _print_arms("top tool priors (aggregated across clusters):", bandits.get("tool", {}))
    _print_arms("top query-template priors:", bandits.get("query", {}))
    _print_arms("stop-arm priors:", bandits.get("stop", {}))
    _print_arms("verify-arm priors:", bandits.get("verify", {}))

    print("\nbandit parameters (thresholds / knobs):")
    for k in ("strategy", "exploration_coeff", "recency_decay", "confidence_penalty",
              "blend_local", "blend_global", "blend_neighbor"):
        print(f"  {k:<20} = {params.get(k)}")
    print(f"  nearest_k            = {snap.get('nearest_k')}")
    sv = snap.get("meta", {})
    print("  (stop_threshold / verification thresholds are agent-config level, not "
          "stored in memory; see config/default.yaml)")

    ok, hits = _scan_answer_like(snap)
    print(f"\nanswer-like fields present : {'NO' if ok else 'YES -> ' + str(hits)}")
    try:
        assert_no_answer_leakage(snap, "snapshot")
        print("leakage scan (assert_no_answer_leakage): PASS")
    except Exception as e:
        print(f"leakage scan: FAIL -> {e}")

    print("\nexample numeric policy fragments (reward stats only, no answers):")
    for ck, frag in list(fragments.items())[:2]:
        tr = frag.get("tool_rewards", {})
        qr = frag.get("query_rewards", {})
        print(f"  cluster '{ck}': support={frag.get('support_count')} "
              f"correct={frag.get('correct_count')} best_tool={frag.get('best_tool')} "
              f"best_query_arm={frag.get('best_query_arm')}")
        print(f"    tool_rewards={ {k: round(v.get('mean',0),3) for k,v in tr.items()} }")
        print(f"    query_rewards={ {k: round(v.get('mean',0),3) for k,v in qr.items()} }")
    if not fragments:
        print("  (no fragments — run consolidation, e.g. scripts/run_experience.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
