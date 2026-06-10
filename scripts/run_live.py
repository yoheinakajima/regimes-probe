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

from _common import (bandit_params, load_config, reward_weights,
                     validate_task_frame_flags)

from regimes_probe.agent.planner import AgentConfig, EpistemicAgent
from regimes_probe.live.providers import (
    build_live_answerer, build_live_providers, missing_keys)
from regimes_probe.live.cache import RecordingCache
from regimes_probe.live.runner import ALL_CONDITIONS, build_plan, run_live_pipeline
from regimes_probe.agent.read_judgment import DEFAULT_READ_CONFIG as _READ_CFG

#: bounded raw-payload cap for read-class tools when --read-cache-store-raw is set (5m-7).
_DEFAULT_READ_RAW_CHARS = _READ_CFG.read_cache_raw_chars


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
    ap.add_argument("--enable-agentic-tool-discovery", action="store_true",
                    help="enable Monid discover/inspect arms (auto in diverse mode)")
    ap.add_argument("--enable-scrape-tools", action="store_true",
                    help="enable firecrawl_scrape (full-page markdown evidence)")
    ap.add_argument("--enable-browserish-tools", action="store_true",
                    help="enable browser-like tools (firecrawl_interact). Off by default.")
    ap.add_argument("--allow-stateful-or-paid-tools", action="store_true",
                    help="allow stateful/paid execution (monid_run). Off by default.")
    ap.add_argument("--enable-query-decomposition", action="store_true",
                    help="Level 2: decompose long questions into several targeted clue "
                         "queries (query forms become bandit arms). Off by default.")
    ap.add_argument("--enable-iterative-clue-resolution", action="store_true",
                    help="Staged search: extract candidate entities from results and "
                         "search them with the next clue. Off by default.")
    ap.add_argument("--enable-task-frame", action="store_true",
                    help="Level 4: parse a task frame (slots+constraints) and plan "
                         "actions to resolve it. Off by default.")
    ap.add_argument("--enable-llm-task-frame-parser", action="store_true",
                    help="Use the cached/validated LLM task-frame parser (requires "
                         "--enable-task-frame); falls back to deterministic on "
                         "invalid/low-quality output. Off by default.")
    ap.add_argument("--task-frame-parser-model", default=None,
                    help="Model for the LLM task-frame parser (default: --answer-model). "
                         "Cached/replayable; only used with --enable-llm-task-frame-parser.")
    ap.add_argument("--auto-epistemic-mode", action="store_true",
                    help="Let the escalation controller pick the epistemic mode per "
                         "question (direct/simple/decomposed/iterative/task_frame). "
                         "Off by default; the benchmark uses explicit flags.")
    ap.add_argument("--force-task-frame", action="store_true",
                    help="Always escalate to task-frame mode (overrides the controller).")
    ap.add_argument("--disable-direct-answer", action="store_true",
                    help="Never use the direct-answer mode (always search at least once).")
    ap.add_argument("--enable-frontier-controller", action="store_true",
                    help="Level 5: the candidate-slate FrontierScheduler DRIVES tool "
                         "selection (each call links to a frontier_action id; falls back "
                         "to the old planner per step). Requires --enable-task-frame. "
                         "Off by default (shadow mode: recommendations recorded only).")
    ap.add_argument("--enable-llm-frontier-repair", action="store_true",
                    help="Level 5c: call the LLM to repair a GENERIC deterministic "
                         "frontier query into a constraint-grounded one. Requires "
                         "--enable-task-frame. Cached/replayable; off by default.")
    ap.add_argument("--enable-llm-frontier-planner", action="store_true",
                    help="Level 5c: ask the LLM for top-K frontier-action proposals from "
                         "the ActiveGraph state card (deterministic code validates/scores/"
                         "executes). Requires --enable-task-frame. Off by default.")
    ap.add_argument("--llm-frontier-model", default=None,
                    help="Model for the LLM frontier proposer (default: --answer-model).")
    ap.add_argument("--enable-llm-evidence-interpreter", action="store_true",
                    help="Level 5d: let an LLM re-classify a result's SOURCE ROLE from "
                         "bounded snippets (the deterministic interpreter always runs; this "
                         "only refines source-role typing). Requires --enable-task-frame. "
                         "Cached/replayable; off by default.")
    ap.add_argument("--llm-evidence-interpreter-model", default=None,
                    help="Model for the LLM evidence interpreter (default: --answer-model).")
    ap.add_argument("--enable-llm-evidence-judge", action="store_true",
                    help="Level 5f: a narrow LLM evidence JUDGE that decides whether a source "
                         "excerpt supports/partially-supports/contradicts/requires-read for a "
                         "candidate/slot/constraint triple. Never answers; cached/replayable; "
                         "deterministic recognizer fallback. Requires --enable-task-frame.")
    ap.add_argument("--llm-evidence-judge-model", default=None,
                    help="Model for the LLM evidence judge (default: --answer-model).")
    ap.add_argument("--judge-model", default=None,
                    help="(reserved) LLM judge; grading currently uses exact/normalized match")
    ap.add_argument("--split-seed", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="force dry-run even with --execute (default is already dry-run)")
    ap.add_argument("--execute", action="store_true",
                    help="REQUIRED to actually call providers (spends money)")
    ap.add_argument("--recording-cache", default=None)
    ap.add_argument("--cache-mode", default="auto", choices=["auto", "replay", "record", "off"])
    # 5m-7: read-class persistence for replay validation. Bounded raw storage of read-tool
    # payloads (sanitized by the recording cache; no secrets/gold) + configurable read caps,
    # so future validation never falls back to debug snippets.
    ap.add_argument("--read-cache-store-raw", action="store_true",
                    help="store bounded RAW payloads for read-class tools in the recording "
                         "cache (enables beyond-cap passage validation in offline replay)")
    ap.add_argument("--read-max-chars", type=int, default=0,
                    help="override the read-class adapter caps (page_fetch/firecrawl_scrape); "
                         "0 keeps the conservative 4000 default. Judge input stays bounded by "
                         "passage windows regardless, so a larger retrieval body does not grow "
                         "judge prompts (recommended for read-judgment-backed runs: 20000)")
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

    # Effective Level-2 flags: CLI flag OR config default.
    decompose_enabled = (args.enable_query_decomposition
                         or bool(cfg.get("policy", {}).get("enable_query_decomposition", False)))
    iterative_enabled = (args.enable_iterative_clue_resolution
                         or bool(cfg.get("policy", {}).get("enable_iterative_clue_resolution", False)))
    task_frame_enabled = (args.enable_task_frame
                          or bool(cfg.get("policy", {}).get("enable_task_frame", False)))
    # The LLM parser REQUIRES the task frame. Fail fast (before writing any
    # artifacts) on an explicit/config request without it — never silently
    # downgrade a user-requested flag.
    llm_parser_requested = (args.enable_llm_task_frame_parser
                            or bool(cfg.get("policy", {}).get("enable_llm_task_frame_parser", False)))
    frontier_controller_requested = (args.enable_frontier_controller
                                     or bool(cfg.get("policy", {}).get("enable_frontier_controller", False)))
    llm_frontier_requested = (args.enable_llm_frontier_repair or args.enable_llm_frontier_planner
                              or bool(cfg.get("policy", {}).get("enable_llm_frontier_repair", False))
                              or bool(cfg.get("policy", {}).get("enable_llm_frontier_planner", False)))
    llm_evidence_interpreter_requested = (
        args.enable_llm_evidence_interpreter
        or bool(cfg.get("policy", {}).get("enable_llm_evidence_interpreter", False)))
    llm_evidence_judge_requested = (
        args.enable_llm_evidence_judge
        or bool(cfg.get("policy", {}).get("enable_llm_evidence_judge", False)))
    try:
        validate_task_frame_flags(task_frame=task_frame_enabled, llm_parser=llm_parser_requested,
                                  frontier_controller=frontier_controller_requested,
                                  llm_frontier=llm_frontier_requested,
                                  llm_evidence_interpreter=llm_evidence_interpreter_requested,
                                  llm_evidence_judge=llm_evidence_judge_requested)
    except ValueError as exc:
        print(f"=== run_live: REFUSING (configuration error) ===\n{exc}")
        return 2
    llm_parser_enabled = llm_parser_requested
    frontier_controller_enabled = frontier_controller_requested
    llm_frontier_repair_enabled = bool(args.enable_llm_frontier_repair
                                       or cfg.get("policy", {}).get("enable_llm_frontier_repair", False))
    llm_frontier_planner_enabled = bool(args.enable_llm_frontier_planner
                                        or cfg.get("policy", {}).get("enable_llm_frontier_planner", False))
    llm_evidence_interpreter_enabled = llm_evidence_interpreter_requested
    llm_evidence_judge_enabled = llm_evidence_judge_requested

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
        enable_agentic_tool_discovery=args.enable_agentic_tool_discovery,
        enable_scrape_tools=args.enable_scrape_tools,
        enable_browserish_tools=args.enable_browserish_tools,
        allow_stateful_or_paid_tools=args.allow_stateful_or_paid_tools,
        enable_query_decomposition=decompose_enabled,
        enable_iterative_clue_resolution=iterative_enabled,
        enable_task_frame=task_frame_enabled,
        enable_llm_task_frame_parser=llm_parser_enabled,
        enable_frontier_controller=frontier_controller_enabled,
        enable_llm_frontier=llm_frontier_repair_enabled or llm_frontier_planner_enabled,
    )
    # Stamp the effective flags into cfg.policy so the agent + manifest both see them.
    cfg.setdefault("policy", {})["enable_query_decomposition"] = decompose_enabled
    cfg.setdefault("policy", {})["enable_iterative_clue_resolution"] = iterative_enabled
    cfg.setdefault("policy", {})["enable_task_frame"] = task_frame_enabled
    cfg.setdefault("policy", {})["enable_llm_task_frame_parser"] = llm_parser_enabled
    cfg["policy"]["auto_epistemic_mode"] = bool(
        args.auto_epistemic_mode or cfg.get("policy", {}).get("auto_epistemic_mode", False))
    cfg["policy"]["force_task_frame"] = bool(
        args.force_task_frame or cfg.get("policy", {}).get("force_task_frame", False))
    cfg["policy"]["disable_direct_answer"] = bool(
        args.disable_direct_answer or cfg.get("policy", {}).get("disable_direct_answer", False))
    cfg["policy"]["enable_frontier_controller"] = frontier_controller_enabled
    cfg["policy"]["enable_llm_frontier_repair"] = llm_frontier_repair_enabled
    cfg["policy"]["enable_llm_frontier_planner"] = llm_frontier_planner_enabled
    cfg["policy"]["enable_llm_evidence_interpreter"] = llm_evidence_interpreter_enabled
    cfg["policy"]["enable_llm_evidence_judge"] = llm_evidence_judge_enabled
    # Parser model defaults to the answer model unless explicitly overridden.
    task_frame_parser_model = (args.task_frame_parser_model
                               or cfg.get("policy", {}).get("task_frame_parser_model")
                               or settings.answer_model)
    if llm_parser_enabled:
        cfg["policy"]["task_frame_parser_model"] = task_frame_parser_model
        settings.task_frame_parser_model = task_frame_parser_model
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
    print(f"first-hop bandit arms={settings.first_hop_tools}")
    print(f"follow-up tools (URL-only, not bandit arms)={settings.followup_tools}")
    print(f"all enabled tools={settings.tools}")
    print(f"provider_classes={settings.provider_classes()}")
    print(f"query_decomposition_enabled={settings.query_decomposition_enabled}  "
          f"iterative_clue_resolution_enabled={settings.iterative_clue_resolution_enabled}  "
          f"task_frame_enabled={settings.task_frame_enabled}  "
          f"llm_task_frame_parser_enabled={settings.llm_task_frame_parser_enabled}")
    print(f"flags: agentic_discovery={settings.agentic_tool_discovery_enabled} "
          f"scrape={settings.scrape_tools_enabled} browserish={settings.browserish_tools_enabled} "
          f"stateful_or_paid_allowed={settings.stateful_or_paid_tools_allowed}")
    print(f"answer_model={settings.answer_model}  "
          f"web_search_model={settings.web_search_model} (ctx={settings.web_search_context_size})  "
          f"openai_web_search_enabled={settings.openai_web_search_enabled}")
    print(f"optimize/confirm available: {plan.n_optimize}/{plan.n_confirm}")
    print(f"estimated answerer calls={plan.cost['answerer_calls']}  "
          f"worst-case tool calls={plan.cost['worst_case_tool_calls']}  "
          f"max_calls_by_tool={plan.cost['max_calls_by_tool']}")
    if llm_parser_enabled:
        print(f"task-frame parser: model={task_frame_parser_model}  "
              f"estimated model calls={plan.cost.get('parser_model_calls_estimated')} "
              f"(cached/replayable; dry-run makes none)")
    ep = plan.eligibility_preflight
    print(f"structurally_valid(preflight)={ep.get('structurally_valid')}  "
          f"headline_eligible_memory_claim(preflight)={ep.get('headline_eligible_memory_claim')} "
          f"(dataset_is_real={is_real}, conditions={conditions})")
    if not ep.get("headline_eligible_memory_claim") and ep.get("headline_eligibility_reasons"):
        print(f"  not a memory headline: {ep['headline_eligibility_reasons']}")
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

    cache = RecordingCache(args.recording_cache, mode=args.cache_mode,
                           store_raw=bool(args.read_cache_store_raw))
    providers = build_live_providers(tools, cache=cache, armed=True,
                                     web_search_model=settings.web_search_model,
                                     web_search_context=settings.web_search_context_size,
                                     allow_stateful_or_paid=args.allow_stateful_or_paid_tools,
                                     enable_browserish=args.enable_browserish_tools,
                                     read_max_chars=int(args.read_max_chars or 0),
                                     read_raw_chars=(
                                         _DEFAULT_READ_RAW_CHARS
                                         if args.read_cache_store_raw else 0))
    agent_cfg = AgentConfig(available_tools=tools,
                            query_mode=cfg["policy"]["query_mode"],
                            stop_mode=cfg["policy"]["stop_mode"],
                            enable_query_decomposition=decompose_enabled,
                            enable_iterative_clue_resolution=iterative_enabled,
                            enable_task_frame=task_frame_enabled,
                            enable_llm_task_frame_parser=llm_parser_enabled,
                            auto_epistemic_mode=cfg["policy"]["auto_epistemic_mode"],
                            force_task_frame=cfg["policy"]["force_task_frame"],
                            enable_frontier_controller=cfg["policy"]["enable_frontier_controller"],
                            enable_llm_frontier_repair=llm_frontier_repair_enabled,
                            enable_llm_frontier_planner=llm_frontier_planner_enabled,
                            enable_llm_evidence_interpreter=llm_evidence_interpreter_enabled,
                            enable_llm_evidence_judge=llm_evidence_judge_enabled,
                            disable_direct_answer=cfg["policy"]["disable_direct_answer"],
                            scrape_fallback_to_page_fetch=bool(
                                cfg["policy"].get("scrape_fallback_to_page_fetch", True)),
                            allow_social_scrape=bool(cfg["policy"].get("allow_social_scrape", False)),
                            as_of=cfg.get("run", {}).get("as_of", "2026-06-01"))
    # Cached/replayable task-frame parser (Level 4b). The model_fn routes through
    # the SAME RecordingCache as the answerer (dry-run raises, replay reads cache,
    # success is recorded); the parser's own ParserCache file dedups parses across
    # conditions/passes. Any model failure -> deterministic fallback.
    tf_parser = None
    if llm_parser_enabled:
        from regimes_probe.agent.llm_task_frame import LLMTaskFrameParser, ParserCache
        from regimes_probe.live.providers import build_task_frame_model_fn
        cache_path = str(Path(plan.run_dir) / "task_frame_parser_cache.json")
        tf_model_fn = build_task_frame_model_fn(task_frame_parser_model, cache, armed=True)
        tf_parser = LLMTaskFrameParser(
            model_fn=tf_model_fn, cache=ParserCache(cache_path),
            model=task_frame_parser_model)
    # Cached/replayable LLM frontier proposer (Level 5c): model_fn through the SAME
    # RecordingCache; its own ParserCache file dedups proposals across conditions.
    lf_proposer = None
    if llm_frontier_repair_enabled or llm_frontier_planner_enabled:
        from regimes_probe.agent.llm_frontier import LLMFrontierProposer
        from regimes_probe.agent.llm_task_frame import ParserCache
        from regimes_probe.live.providers import build_frontier_model_fn
        lf_model = args.llm_frontier_model or answer_model
        lf_cache_path = str(Path(plan.run_dir) / "llm_frontier_cache.json")
        lf_proposer = LLMFrontierProposer(
            model_fn=build_frontier_model_fn(lf_model, cache, armed=True),
            cache=ParserCache(lf_cache_path), model=lf_model)
    # Cached/replayable LLM evidence interpreter (Level 5d): a SHARED (model_fn, cache)
    # the per-attempt interpreter is built from; the deterministic interpreter always runs.
    ev_interpreter = None
    if llm_evidence_interpreter_enabled:
        from regimes_probe.agent.evidence_interpreter import EvidenceInterpreter
        from regimes_probe.agent.llm_task_frame import ParserCache
        from regimes_probe.live.providers import build_evidence_model_fn
        ev_model = args.llm_evidence_interpreter_model or answer_model
        ev_cache_path = str(Path(plan.run_dir) / "llm_evidence_interpreter_cache.json")
        ev_interpreter = EvidenceInterpreter(
            model_fn=build_evidence_model_fn(ev_model, cache, armed=True),
            cache=ParserCache(ev_cache_path), model=ev_model, enabled_llm=True)
    # Cached/replayable LLM evidence judge (Level 5f): shared cache/model -> per-attempt judge.
    ev_judge = None
    if llm_evidence_judge_enabled:
        from regimes_probe.agent.evidence_judge import EvidenceJudge
        from regimes_probe.agent.llm_task_frame import ParserCache
        from regimes_probe.live.providers import build_evidence_judge_model_fn
        ej_model = args.llm_evidence_judge_model or answer_model
        ej_cache_path = str(Path(plan.run_dir) / "llm_evidence_judge_cache.json")
        ev_judge = EvidenceJudge(
            model_fn=build_evidence_judge_model_fn(ej_model, cache, armed=True),
            cache=ParserCache(ej_cache_path), model=ej_model, enabled=True)
    search_agent = EpistemicAgent(agent_cfg,
                                  answerer=build_live_answerer("search", model=answer_model,
                                                               cache=cache, armed=True),
                                  task_frame_parser=tf_parser, llm_frontier=lf_proposer,
                                  evidence_interpreter=ev_interpreter, evidence_judge=ev_judge)
    cb_agent = EpistemicAgent(agent_cfg,
                              answerer=build_live_answerer("closed_book", model=answer_model,
                                                           cache=cache, armed=True),
                              task_frame_parser=tf_parser, llm_frontier=lf_proposer,
                              evidence_interpreter=ev_interpreter, evidence_judge=ev_judge)
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
        live_settings=settings.to_dict(), task_frame_parser=tf_parser,
        llm_frontier=lf_proposer)
    el = result["eligibility"]
    print(f"run dir: {result['run_dir']}")
    print(f"cache: {result['cache']}")
    print(f"structurally_valid={el.get('structurally_valid')}  "
          f"headline_eligible_memory_claim={el.get('headline_eligible_memory_claim')}  "
          f"conditions_present={el.get('conditions_present')}")
    if not el.get("headline_eligible_memory_claim"):
        print(f"  not a memory headline: {el.get('headline_eligibility_reasons')}")
    print("Follow-up:\n" + _follow_ups(result["run_dir"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
