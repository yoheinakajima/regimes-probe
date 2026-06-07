#!/usr/bin/env python
"""End-to-end synthetic demo: baseline vs frozen policy memory, with budget
curve, statistical tests, a replay check, and a full report.

Produces results/{run_id}/ exactly as documented in docs/REPORTING.md. Runs with
NO keys and NO network.

    python scripts/make_report.py --run-id demo
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import random

from _common import (
    bandit_params,
    base_argparser,
    build_agent,
    build_synthetic,
    load_config,
    reward_weights,
)

from regimes_probe.activegraph_pack import replay_check
from regimes_probe.datasets.synthetic import SyntheticBrowseAdapter
from regimes_probe.eval.harness import experience_phase, run_condition
from regimes_probe.eval.metrics import compute_metrics
from regimes_probe.eval.report import ConditionRun, write_full_report
from regimes_probe.eval.significance import (
    bootstrap_correct_per_tool_call,
    mcnemar,
)
from regimes_probe.eval.split import build_split, partition
from regimes_probe.policy.memory import PolicyMemory


def _random_memory(items, agent, params, seed=7):
    """Random-memory control: real priors, but uninformative (shuffled rewards)."""
    rng = random.Random(seed)
    mem = PolicyMemory(params.copy())
    families = {"tool": ["generic_web_search", "news_search", "official_domain_search", "brave_search"],
                "query": ["direct_question", "freshness_terms", "source_constrained", "keyword_compressed"],
                "stop": ["stop_now", "search_more", "fetch_page"]}
    for it in items:
        sig = agent.signature(it)
        rewards = {fam: {rng.choice(arms): rng.uniform(-1, 1)} for fam, arms in families.items()}
        mem.observe(sig, f"rand-{it.id}", rewards, correct=bool(rng.random() < 0.5), tool_calls=1)
    return mem


def main() -> int:
    args = base_argparser("Full synthetic report (baseline vs policy memory).").parse_args()
    cfg = load_config(args.config)
    run_id = args.run_id or "demo"
    budgets = cfg.get("budgets", [1, 3, 5, 10])
    weights = reward_weights(cfg)
    params = bandit_params(cfg)

    items, providers, tools, search_tools = build_synthetic(cfg)
    agent = build_agent(cfg, tools)
    dataset_version = SyntheticBrowseAdapter().version()

    split = build_split(items, confirm_fraction=cfg.get("split", {}).get("confirm_fraction", 0.4),
                        salt=cfg.get("split", {}).get("salt", "regimes-probe-v0"))
    opt_items, con_items = partition(items, split)

    # Experience phase on OPTIMIZE -> frozen snapshot.
    mem = PolicyMemory(params.copy(), nearest_k=cfg.get("memory", {}).get("nearest_k", 8))
    experience_phase(opt_items, agent, providers, mem,
                     budget=cfg.get("memory", {}).get("experience_budget", 5),
                     passes=cfg.get("memory", {}).get("experience_passes", 4),
                     weights=weights, dataset_version=dataset_version)
    snapshot = mem.snapshot(meta={"dataset_version": dataset_version,
                                  "n_optimize": len(opt_items)})

    runs: list[ConditionRun] = []
    replay_log = None
    aligned = {}  # (budget) -> {item_id: (base_correct, pol_correct)}
    for b in budgets:
        base_mem = PolicyMemory(params.copy())
        base = run_condition(con_items, agent, providers, base_mem, condition="no_memory",
                             budget=b, weights=weights, explore=False, dataset_version=dataset_version)
        frozen = PolicyMemory.from_snapshot(snapshot, frozen=True)
        pol = run_condition(con_items, agent, providers, frozen, condition="policy_memory",
                            budget=b, weights=weights, explore=False, dataset_version=dataset_version)
        rnd = run_condition(con_items, agent, providers,
                            _random_memory(opt_items, agent, params), condition="random_memory",
                            budget=b, weights=weights, explore=False, dataset_version=dataset_version)
        runs += [ConditionRun("no_memory", b, base.outcomes),
                 ConditionRun("policy_memory", b, pol.outcomes),
                 ConditionRun("random_memory", b, rnd.outcomes)]
        aligned[b] = (base.outcomes, pol.outcomes)
        if b == 3:
            replay_log = pol.log

    # Statistical tests at budget 3 (paired by item).
    sig_b = 3 if 3 in budgets else budgets[0]
    base_out, pol_out = aligned[sig_b]
    bmap = {o.item_id: o for o in base_out}
    pmap = {o.item_id: o for o in pol_out}
    ids = [i for i in bmap if i in pmap]
    mc = mcnemar([bmap[i].correct for i in ids], [pmap[i].correct for i in ids])
    ci = bootstrap_correct_per_tool_call([int(pmap[i].correct) for i in ids],
                                         [pmap[i].tool_calls for i in ids])
    significance = {"budget": sig_b, "mcnemar": mc.to_dict(),
                    "policy_correct_per_tool_call_ci": ci.to_dict()}

    replay = replay_check(replay_log).to_dict() if replay_log is not None else {}

    meta = {
        "answer_model": cfg.get("live", {}).get("answer_model"),
        "search_baseline": cfg.get("live", {}).get("search_baseline"),
        "tools_enabled": search_tools,
        "embedder": "hash_embedder",
        "dataset": "synthetic_browse",
        "dataset_version": dataset_version,
        "split": split.to_dict() | {"optimize_ids": "...", "confirm_ids": "..."},
        "budgets": budgets,
        "memory_condition": "frozen_policy_memory",
        "policy_condition": f"query={cfg['policy']['query_mode']},stop={cfg['policy']['stop_mode']}",
    }
    run_dir = write_full_report(run_id, runs=runs, snapshot=snapshot.to_dict(),
                                replay=replay, significance=significance, meta=meta,
                                results_root=args.results_root)
    print(f"wrote {run_dir}")
    print("Headline (correct_per_tool_call) by budget:")
    for b in budgets:
        bm = compute_metrics(aligned[b][0])["correct_per_tool_call"]
        pm = compute_metrics(aligned[b][1])["correct_per_tool_call"]
        print(f"  budget {b:>2}: no_memory={bm:.3f}  policy_memory={pm:.3f}")
    print(f"replay projection_matches={replay.get('projection_matches')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
