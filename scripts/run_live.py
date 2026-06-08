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
    from regimes_probe.datasets.base import DatasetUnavailable
    ds = cfg.get("dataset", {})
    if name in ("livebrowsecomp", "LiveBrowseComp"):
        from regimes_probe.datasets.livebrowsecomp import LiveBrowseCompAdapter
        real_path = path or ds.get("livebrowsecomp_local_jsonl") or ""
        if real_path:  # a real dataset path was supplied -> load real or FAIL CLOSED
            if not Path(real_path).exists():
                raise DatasetUnavailable(f"LiveBrowseComp path not found: {real_path}")
            a = LiveBrowseCompAdapter(local_jsonl=real_path)
            return a.load(), "LiveBrowseComp", a.version(), real_path, True   # may raise (fail closed)
    if name in ("browsecomp", "BrowseComp"):
        from regimes_probe.datasets.browsecomp import BrowseCompAdapter
        real_path = path or ds.get("browsecomp_csv") or ""
        if real_path:  # supplied -> load real or FAIL CLOSED (no placeholder fallback)
            if not Path(real_path).exists():
                raise DatasetUnavailable(f"BrowseComp path not found: {real_path}")
            a = BrowseCompAdapter(real_path)
            return a.load(), "BrowseComp", a.version(), real_path, True
    # No real dataset path supplied: real-shaped placeholder (NOT a benchmark) for
    # dry-run plumbing only. (When a path IS supplied we never reach here.)
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
    ap.add_argument("--tools", default=None,
                    help="comma-separated; overrides the provider-mode default. "
                         "Default does NOT require openai_web_search.")
    ap.add_argument("--search-provider-mode", default=None,
                    choices=["cheap", "diverse", "openai-hosted"],
                    help="cheap (default): mini + page_fetch + cheap external search; "
                         "diverse: providers as separate arms (the main experiment); "
                         "openai-hosted: explicit expensive hosted baseline.")
    ap.add_argument("--answer-model", default=None, help="default: gpt-5.4-mini (cheap-first)")
    ap.add_argument("--web-search-model", default=None, help="default: gpt-5.4-mini")
    ap.add_argument("--web-search-context-size", default=None,
                    choices=["low", "medium", "high", "unlimited"], help="default: low")
    ap.add_argument("--disable-openai-web-search", action="store_true",
                    help="remove openai_web_search from the tool set entirely")
    ap.add_argument("--judge-model", default=None,
                    help="(reserved) LLM judge; grading currently uses exact/normalized match")
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

    from regimes_probe.live.settings import resolve_live_settings
    cfg = load_config(args.config)
    live_cfg = cfg.get("live", {})
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    bad = [c for c in conditions if c not in ALL_CONDITIONS]
    if bad:
        print(f"unknown condition(s): {bad}; valid: {list(ALL_CONDITIONS)}")
        return 2

    # Resolve models + tools from mode/CLI/config/env (cheap-first; OpenAI hosted
    # web_search is opt-in, never a silent default).
    settings = resolve_live_settings(
        mode=args.search_provider_mode or live_cfg.get("search_provider_mode", "cheap"),
        cli_tools=([t.strip() for t in args.tools.split(",") if t.strip()] if args.tools else None),
        answer_model=args.answer_model or live_cfg.get("answer_model", "gpt-5.4-mini"),
        web_search_model=args.web_search_model or live_cfg.get("web_search_model", "gpt-5.4-mini"),
        web_search_context_size=(args.web_search_context_size
                                 or live_cfg.get("web_search_context_size", "low")),
        disable_openai_web_search=args.disable_openai_web_search,
    )
    tools = settings.tools
    search_tools = settings.search_tools
    answer_model = settings.answer_model
    cfg.setdefault("live", {})["answer_model"] = answer_model   # stamp resolved model
    split_seed = args.split_seed or cfg.get("split", {}).get("salt", "regimes-probe-v0")

    from regimes_probe.datasets.base import DatasetUnavailable
    try:
        items, label, version, ds_path, is_real = _load_dataset(args.dataset, args.dataset_path, cfg)
    except DatasetUnavailable as exc:
        print(f"=== run_live: REFUSING (dataset unavailable / fail-closed) ===\n{exc}")
        print("\nLiveBrowseComp's HF 'problem'/'answer' fields are obfuscated; this run "
              "will NOT spend API calls on encrypted text. Provide a plaintext export "
              "or a validated decode path. See docs/NEXT_LIVE_RUN.md and docs/BENCHMARK_TARGETS.md.")
        return 2

    # Dry-run validation guard: refuse if the loaded REAL questions look obfuscated
    # (belt-and-suspenders beyond the adapter's own fail-closed loading).
    if is_real:
        from regimes_probe.datasets.livebrowsecomp import looks_obfuscated
        bad = [it.id for it in items[:50] if looks_obfuscated(it.question)]
        if bad:
            print("=== run_live: REFUSING — loaded questions look encrypted/obfuscated ===")
            print(f"e.g. item ids {bad[:5]}. LiveBrowseComp rows require a supported decode "
                  "path or a plaintext export; the adapter must not pass encrypted text as a "
                  "question. No providers were called. See docs/BENCHMARK_TARGETS.md.")
            return 2

    run_id = args.run_id or "live-" + hashlib.sha256(
        f"{label}|{version}|{split_seed}|{args.optimize}|{args.confirm}|{budgets}|{conditions}".encode()
    ).hexdigest()[:8]

    executing = args.execute and not args.dry_run

    # Always write a dry-run plan (manifest + plan.json + config snapshot). No calls.
    plan = build_plan(cfg, items, conditions=conditions, budgets=budgets,
                      optimize=args.optimize, confirm=args.confirm, split_seed=split_seed,
                      run_id=run_id, results_root=args.results_root, dataset_label=label,
                      dataset_version=version, dataset_path=ds_path, is_real=is_real,
                      search_tools=search_tools, live_settings=settings.to_dict())

    miss = missing_keys(tools)
    print(f"=== run_live ({'EXECUTE' if executing else 'DRY-RUN'}) — dataset={label}, "
          f"run_id={run_id} ===")
    print(f"provider_mode={settings.provider_mode}  conditions={conditions}  budgets={budgets}")
    print(f"tools (bandit arms)={tools}")
    print(f"answer_model={settings.answer_model}  "
          f"web_search_model={settings.web_search_model} (ctx={settings.web_search_context_size})  "
          f"openai_web_search_enabled={settings.openai_web_search_enabled}")
    print(f"optimize/confirm available: {plan.n_optimize}/{plan.n_confirm}")
    print(f"estimated answerer calls={plan.cost['answerer_calls']}  "
          f"worst-case tool calls={plan.cost['worst_case_tool_calls']}  "
          f"max_calls_by_tool={plan.cost['max_calls_by_tool']}")
    print(f"headline_eligible(preflight)={plan.eligibility_preflight['headline_eligible']} "
          f"(dataset_is_real={is_real})")
    print(f"required env vars present: {'yes' if not miss else 'NO -> missing ' + str(miss)}")
    for note in settings.notes:
        print(f"  note: {note}")
    for w in settings.warnings:
        print(f"  ⚠️  {w}")
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
    providers = build_live_providers(tools, cache=cache, armed=True,
                                     web_search_model=settings.web_search_model,
                                     web_search_context=settings.web_search_context_size)
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
        weights=reward_weights(cfg), params=bandit_params(cfg), resume_snapshot=resume,
        live_settings=settings.to_dict())
    print(f"run dir: {result['run_dir']}")
    print(f"cache: {result['cache']}")
    print(f"headline_eligible={result['eligibility']['headline_eligible']} "
          f"reasons={result['eligibility']['reasons']}")
    print("Follow-up:\n" + _follow_ups(result["run_dir"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
