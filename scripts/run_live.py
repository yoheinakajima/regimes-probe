#!/usr/bin/env python
"""Live run executor — SAFE BY DEFAULT (dry-run; refuses to spend without --execute).

    # A. dry-run plan (no providers called):
    python scripts/run_live.py --dataset real-shaped --optimize 5 --confirm 10 --budgets 1

    # B. actually call providers (requires keys; you must pass --execute):
    python scripts/run_live.py --dataset livebrowsecomp --dataset-path data/lbc.jsonl \
        --optimize 10 --confirm 20 --budgets 1,3 --recording-cache results/live/cache.json \
        --execute

Default behaviour is dry-run/no-spend. Provider calls happen ONLY when --execute
is passed (and not overridden by --dry-run). Credentials come from env vars and
are never written to artifacts. Every live call flows through the recording cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from _common import bandit_params, load_config, reward_weights

from regimes_probe.agent.planner import AgentConfig, EpistemicAgent
from regimes_probe.live.providers import (
    build_live_answerer, build_live_providers, missing_keys)
from regimes_probe.live.cache import RecordingCache
from regimes_probe.live.runner import ALL_CONDITIONS, build_plan, run_live_pipeline


def _load_dataset(name, path, cfg):
    """Return (items, label, version, path, is_real). No network unless a real
    local file is provided; real datasets without local data fall back to the
    real-shaped placeholder for dry-run plumbing (is_real=False)."""
    ds = cfg.get("dataset", {})
    if name in ("livebrowsecomp", "LiveBrowseComp"):
        from regimes_probe.datasets.livebrowsecomp import LiveBrowseCompAdapter
        local = path or ds.get("livebrowsecomp_local_jsonl") or ""
        if local and Path(local).exists():
            a = LiveBrowseCompAdapter(local_jsonl=local)
            return a.load(), "LiveBrowseComp", a.version(), local, True
    if name in ("browsecomp", "BrowseComp"):
        from regimes_probe.datasets.browsecomp import BrowseCompAdapter
        csv = path or ds.get("browsecomp_csv") or ""
        if csv and Path(csv).exists():
            a = BrowseCompAdapter(csv)
            return a.load(), "BrowseComp", a.version(), csv, True
    # real-shaped placeholder fallback (NOT a benchmark)
    from regimes_probe.datasets.livebrowsecomp import LiveBrowseCompAdapter
    rp = ROOT / "fixtures" / "real_shaped" / "livebrowsecomp_sample.jsonl"
    a = LiveBrowseCompAdapter(local_jsonl=rp)
    return a.load(), "real_shaped_placeholder", a.version(), str(rp), False


def _follow_ups(run_dir: str) -> str:
    return (f"  python scripts/hash_artifacts.py {run_dir}\n"
            f"  python scripts/generate_claims.py {run_dir}/report.json\n"
            f"  python scripts/inspect_memory_snapshot.py {run_dir}/memory_snapshot.json\n"
            f"  python scripts/compare_runs.py {run_dir}/report.json <other>/report.json")


def main() -> int:
    ap = argparse.ArgumentParser(description="Live run executor (safe by default).")
    ap.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    ap.add_argument("--dataset", default="real-shaped",
                    choices=["livebrowsecomp", "browsecomp", "real-shaped"])
    ap.add_argument("--dataset-path", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--optimize", type=int, default=10)
    ap.add_argument("--confirm", type=int, default=20)
    ap.add_argument("--budgets", default="1,3")
    ap.add_argument("--conditions", default=",".join(ALL_CONDITIONS))
    ap.add_argument("--tools", default=None, help="comma-separated; default from config")
    ap.add_argument("--answer-model", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--split-seed", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="force dry-run even with --execute (default is already dry-run)")
    ap.add_argument("--execute", action="store_true",
                    help="REQUIRED to actually call providers (spends money)")
    ap.add_argument("--recording-cache", default=None)
    ap.add_argument("--cache-mode", default="auto", choices=["auto", "replay", "record", "off"])
    ap.add_argument("--resume-from-snapshot", default=None)
    ap.add_argument("--strict-preflight", action="store_true")
    ap.add_argument("--results-root", default=str(ROOT / "results" / "live"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.answer_model:
        cfg.setdefault("live", {})["answer_model"] = args.answer_model
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    bad = [c for c in conditions if c not in ALL_CONDITIONS]
    if bad:
        print(f"unknown condition(s): {bad}; valid: {list(ALL_CONDITIONS)}")
        return 2
    # Default to the benchmark-matching hosted baseline (+ page_fetch) so the
    # minimal run needs only OPENAI_API_KEY. Override with --tools for diversity.
    default_tools = [cfg.get("live", {}).get("search_baseline", "openai_web_search"), "page_fetch"]
    tools = ([t.strip() for t in args.tools.split(",") if t.strip()] if args.tools
             else default_tools)
    if "page_fetch" not in tools:
        tools = tools + ["page_fetch"]
    search_tools = [t for t in tools if t != "page_fetch"]
    split_seed = args.split_seed or cfg.get("split", {}).get("salt", "regimes-probe-v0")
    answer_model = cfg.get("live", {}).get("answer_model", "gpt-5.5")

    items, label, version, ds_path, is_real = _load_dataset(args.dataset, args.dataset_path, cfg)
    run_id = args.run_id or "live-" + hashlib.sha256(
        f"{label}|{version}|{split_seed}|{args.optimize}|{args.confirm}|{budgets}|{conditions}".encode()
    ).hexdigest()[:8]

    executing = args.execute and not args.dry_run

    # Always write a dry-run plan (manifest + plan.json + config snapshot). No calls.
    plan = build_plan(cfg, items, conditions=conditions, budgets=budgets,
                      optimize=args.optimize, confirm=args.confirm, split_seed=split_seed,
                      run_id=run_id, results_root=args.results_root, dataset_label=label,
                      dataset_version=version, dataset_path=ds_path, is_real=is_real,
                      search_tools=search_tools)

    miss = missing_keys(tools)
    print(f"=== run_live ({'EXECUTE' if executing else 'DRY-RUN'}) — dataset={label}, "
          f"run_id={run_id} ===")
    print(f"conditions={conditions}  budgets={budgets}  tools={tools}")
    print(f"optimize/confirm available: {plan.n_optimize}/{plan.n_confirm}")
    print(f"estimated answerer calls={plan.cost['answerer_calls']}  "
          f"worst-case tool calls={plan.cost['worst_case_tool_calls']}")
    print(f"headline_eligible(preflight)={plan.eligibility_preflight['headline_eligible']} "
          f"(dataset_is_real={is_real})")
    print(f"required env vars present: {'yes' if not miss else 'NO -> missing ' + str(miss)}")
    print(f"plan + dry-run manifest -> {plan.run_dir}/")

    if not executing:
        print("\nDRY-RUN: NO providers were called and NO money was spent.")
        print("To actually run, re-invoke with --execute (and provide keys).")
        print("Follow-up (after a real run):\n" + _follow_ups(plan.run_dir))
        return 0

    # ---- EXECUTE PATH (provider calls) ----
    if miss:
        print(f"\nREFUSING to execute: missing required env var(s): {miss}")
        return 2
    if args.strict_preflight and not plan.eligibility_preflight["checks"].get("same_conditions"):
        print("\nREFUSING (strict preflight): same-conditions check failed.")
        return 2

    cache = RecordingCache(args.recording_cache, mode=args.cache_mode)
    providers = build_live_providers(tools, cache=cache, armed=True)
    agent_cfg = AgentConfig(available_tools=tools,
                            query_mode=cfg["policy"]["query_mode"],
                            stop_mode=cfg["policy"]["stop_mode"],
                            as_of=cfg.get("run", {}).get("as_of", "2026-06-01"))
    search_agent = EpistemicAgent(agent_cfg,
                                  answerer=build_live_answerer("search", model=answer_model,
                                                               cache=cache, armed=True))
    cb_agent = EpistemicAgent(agent_cfg,
                              answerer=build_live_answerer("closed_book", model=answer_model,
                                                           cache=cache, armed=True))
    resume = (json.loads(Path(args.resume_from_snapshot).read_text())
              if args.resume_from_snapshot else None)

    print("\nEXECUTING live run — this will call providers and may spend money.\n")
    result = run_live_pipeline(
        cfg, items, providers=providers, search_agent=search_agent, cb_agent=cb_agent,
        cache=cache, conditions=conditions, budgets=budgets, optimize=args.optimize,
        confirm=args.confirm, split_seed=split_seed, run_id=run_id,
        results_root=args.results_root, dataset_label=label, dataset_version=version,
        dataset_path=ds_path, is_real=is_real, search_tools=search_tools,
        weights=reward_weights(cfg), params=bandit_params(cfg), resume_snapshot=resume)
    print(f"run dir: {result['run_dir']}")
    print(f"cache: {result['cache']}")
    print(f"headline_eligible={result['eligibility']['headline_eligible']} "
          f"reasons={result['eligibility']['reasons']}")
    print("Follow-up:\n" + _follow_ups(result["run_dir"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
