"""Shared setup helpers for the CLI scripts (synthetic, no keys/network)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from regimes_probe.agent.planner import AgentConfig, EpistemicAgent
from regimes_probe.datasets.synthetic import SyntheticBrowseAdapter
from regimes_probe.eval.reward import RewardWeights
from regimes_probe.policy.contextual_bandit import BanditParams
from regimes_probe.policy.router import RouterConfig
from regimes_probe.policy.stopping_policy import StopConfig
from regimes_probe.policy.verification_policy import VerificationConfig
from regimes_probe.tools.fake import build_fake_providers, load_corpus

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    return yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))


def reward_weights(cfg: dict[str, Any]) -> RewardWeights:
    preset = cfg.get("reward", {}).get("preset", "full")
    if preset != "full" and hasattr(RewardWeights, preset):
        return getattr(RewardWeights, preset)()
    w = cfg.get("reward", {}).get("weights", {})
    return RewardWeights.from_dict(w) if w else RewardWeights.full()


def bandit_params(cfg: dict[str, Any]) -> BanditParams:
    return BanditParams.from_dict(cfg.get("bandit", {}))


def build_synthetic(cfg: dict[str, Any]):
    """Return (items, providers, tools, search_tools)."""
    ds = cfg.get("dataset", {})
    items = SyntheticBrowseAdapter(ROOT / ds.get("synthetic_path", "fixtures/synthetic_browse.json")).load()
    corpus = load_corpus(ROOT / ds.get("fake_corpus_path", "fixtures/fake_search_corpus.json"))
    search_tools = ["generic_web_search", "news_search", "official_domain_search", "brave_search"]
    providers = build_fake_providers(corpus, search_tools)
    tools = search_tools + ["page_fetch"]
    return items, providers, tools, search_tools


#: Human-readable error for the LLM-parser-without-task-frame misconfiguration.
LLM_PARSER_REQUIRES_TASK_FRAME = (
    "--enable-llm-task-frame-parser requires --enable-task-frame")


def validate_task_frame_flags(*, task_frame: bool, llm_parser: bool) -> None:
    """Fail fast on an LLM-parser-without-task-frame request (never downgrade)."""
    if llm_parser and not task_frame:
        raise ValueError(LLM_PARSER_REQUIRES_TASK_FRAME)


def build_agent(cfg: dict[str, Any], tools: list[str], *, task_frame_parser=None) -> EpistemicAgent:
    pol = cfg.get("policy", {})
    enable_llm_parser = bool(pol.get("enable_llm_task_frame_parser", False))
    validate_task_frame_flags(task_frame=bool(pol.get("enable_task_frame", False)),
                              llm_parser=enable_llm_parser)
    agent_cfg = AgentConfig(
        available_tools=tools,
        query_mode=pol.get("query_mode", "learned"),
        stop_mode=pol.get("stop_mode", "learned"),
        enable_query_decomposition=bool(pol.get("enable_query_decomposition", False)),
        enable_iterative_clue_resolution=bool(pol.get("enable_iterative_clue_resolution", False)),
        enable_task_frame=bool(pol.get("enable_task_frame", False)),
        enable_llm_task_frame_parser=enable_llm_parser,
        auto_epistemic_mode=bool(pol.get("auto_epistemic_mode", False)),
        force_task_frame=bool(pol.get("force_task_frame", False)),
        disable_direct_answer=bool(pol.get("disable_direct_answer", False)),
        scrape_fallback_to_page_fetch=bool(pol.get("scrape_fallback_to_page_fetch", True)),
        allow_social_scrape=bool(pol.get("allow_social_scrape", False)),
        as_of=cfg.get("run", {}).get("as_of", "2026-06-01"),
        router=RouterConfig(**cfg.get("router", {})) if cfg.get("router") else RouterConfig(),
        stop=StopConfig(**cfg.get("stopping", {})) if cfg.get("stopping") else StopConfig(),
        verification=VerificationConfig(**cfg.get("verification", {})) if cfg.get("verification") else VerificationConfig(),
    )
    # When the LLM parser is enabled, attach a cached/replayable parser. No live
    # model client is wired here (offline-safe): with no injected model_fn the
    # parser is cache/replay-only and falls back to the deterministic parser on a
    # cache miss. A live runner injects a model_fn + file-backed cache.
    if task_frame_parser is None and enable_llm_parser:
        from regimes_probe.agent.llm_task_frame import LLMTaskFrameParser, ParserCache
        cache_path = pol.get("task_frame_parser_cache_path")
        task_frame_parser = LLMTaskFrameParser(
            model_fn=None, cache=ParserCache(cache_path),
            model=str(pol.get("task_frame_parser_model", "stub")),
            replay_only=bool(pol.get("task_frame_parser_replay_only", False)))
    return EpistemicAgent(agent_cfg, task_frame_parser=task_frame_parser)


def base_argparser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--run-id", default=None)
    p.add_argument("--results-root", default=str(ROOT / "results"))
    return p


# ---------------------------------------------------------------------------
# Shared no-key, no-network pipeline (used by make_report, run_synthetic_full,
# and the real-data-shaped smoke test).
# ---------------------------------------------------------------------------
import hashlib
import random as _random
from typing import Optional

from regimes_probe.activegraph_pack import replay_check
from regimes_probe.agent.answerer import build_closed_book_knowledge
from regimes_probe.agent.planner import build_closed_book_agent
from regimes_probe.eval.conditions import ConditionSpec, same_conditions
from regimes_probe.eval.cost import estimate_calls
from regimes_probe.eval.eligibility import compute_eligibility
from regimes_probe.eval.manifest import build_manifest, write_manifest
from regimes_probe.eval.harness import experience_phase, run_condition
from regimes_probe.eval.metrics import compute_metrics
from regimes_probe.eval.report import ConditionRun, write_full_report
from regimes_probe.eval.significance import bootstrap_correct_per_tool_call, mcnemar
from regimes_probe.eval.split import build_split, partition
from regimes_probe.policy.memory import PolicyMemory
from regimes_probe.eval.leakage import leakage_check_details
from regimes_probe.policy.policy_fragment import assert_no_answer_leakage

_REAL_DATASETS = {"BrowseComp", "LiveBrowseComp", "browsecomp", "livebrowsecomp"}


def _hash(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12]


def _dataset_checksum(items) -> str:
    h = hashlib.sha256()
    for it in sorted(items, key=lambda i: i.id):
        h.update(it.id.encode()); h.update(b"\x00"); h.update(it.question.encode())
    return h.hexdigest()[:16]


def _condition_spec(cfg, search_tools, budget, split, memory_access) -> ConditionSpec:
    from regimes_probe.agent import prompts
    return ConditionSpec(
        answer_model=cfg.get("live", {}).get("answer_model", "deterministic_answerer"),
        answer_prompt_version=prompts.fingerprint("answerer"),
        enabled_tools=tuple(sorted(search_tools)),
        tool_budget=budget,
        split_id=_hash(split.mode, split.salt, tuple(split.confirm_ids)),
        grader="normalized_match",
        provider_config_id=_hash(tuple(sorted(search_tools))),
        query_policy=cfg["policy"]["query_mode"],
        verify_policy="default",
        stop_policy=cfg["policy"]["stop_mode"],
        memory_access=memory_access,
    )


def random_memory(items, agent, params, *, seed: int = 7) -> PolicyMemory:
    """Random-memory control: real priors, but uninformative (shuffled rewards)."""
    rng = _random.Random(seed)
    mem = PolicyMemory(params.copy())
    families = {
        "tool": ["generic_web_search", "news_search", "official_domain_search", "brave_search"],
        "query": ["direct_question", "freshness_terms", "source_constrained", "keyword_compressed"],
        "stop": ["stop_now", "search_more", "fetch_page"],
    }
    for it in items:
        sig = agent.signature(it)
        rewards = {fam: {rng.choice(arms): rng.uniform(-1, 1)} for fam, arms in families.items()}
        mem.observe(sig, f"rand-{it.id}", rewards, correct=bool(rng.random() < 0.5), tool_calls=1)
    return mem


def full_pipeline(
    cfg: dict[str, Any],
    items,
    providers,
    search_tools: list[str],
    agent,
    *,
    run_id: str,
    results_root: str | Path,
    dataset_label: str,
    dataset_version: str,
    on_step=None,
    online_confirm: bool = False,
) -> dict[str, Any]:
    """Run the complete no-key pipeline and write report artifacts.

    Steps: deterministic split -> no_memory baseline -> experience on OPTIMIZE ->
    freeze snapshot -> policy_memory on CONFIRM -> random_memory control ->
    budget curve -> significance -> replay check -> report. Returns a summary.
    """
    def step(msg: str) -> None:
        if on_step:
            on_step(msg)

    weights = reward_weights(cfg)
    params = bandit_params(cfg)
    budgets = cfg.get("budgets", [1, 3, 5, 10])
    mem_cfg = cfg.get("memory", {})
    sp = cfg.get("split", {})

    step("build deterministic OPTIMIZE/CONFIRM split")
    split = build_split(items, confirm_fraction=sp.get("confirm_fraction", 0.4),
                        salt=sp.get("salt", "regimes-probe-v0"), mode=sp.get("mode", "hash"))
    split.assert_disjoint()
    opt_items, con_items = partition(items, split)

    step(f"experience phase on OPTIMIZE ({len(opt_items)} items) -> frozen snapshot")
    mem = PolicyMemory(params.copy(), nearest_k=mem_cfg.get("nearest_k", 8))
    experience_phase(opt_items, agent, providers, mem,
                     budget=mem_cfg.get("experience_budget", 5),
                     passes=mem_cfg.get("experience_passes", 4),
                     weights=weights, dataset_version=dataset_version)
    # Consolidate raw traces into answer-free policy fragments (with trace lineage)
    # BEFORE freezing, so the CONFIRM snapshot carries fragments — mirrors the live
    # runner and scripts/run_experience.py.
    from regimes_probe.policy.consolidation import consolidate
    consolidate(mem, consolidation_event_id=f"consolidate@{run_id}")
    snapshot = mem.snapshot(meta={"dataset_version": dataset_version,
                                  "n_optimize": len(opt_items)})

    runs: list[ConditionRun] = []
    debug_records: list = []
    aligned: dict[int, tuple] = {}
    replay_log = None
    budget_ok = True
    gate_budget = 3 if 3 in budgets else budgets[0]

    # closed-book baseline (no tools at all): estimates intrinsic knowledge.
    step("closed_book baseline on CONFIRM (no tool calls)")
    cb_agent = build_closed_book_agent(agent.config, build_closed_book_knowledge(items))
    cb = run_condition(con_items, cb_agent, providers, PolicyMemory(params.copy()),
                       condition="closed_book", budget=0, weights=weights,
                       explore=False, dataset_version=dataset_version)
    runs.append(ConditionRun("closed_book", 0, cb.outcomes)); debug_records += cb.debug

    for b in budgets:
        step(f"budget {b}: no_memory_search / policy_memory / random_memory on CONFIRM")
        base = run_condition(con_items, agent, providers, PolicyMemory(params.copy()),
                             condition="no_memory_search", budget=b, weights=weights,
                             explore=False, dataset_version=dataset_version)
        # Headline path: a FROZEN snapshot, no updates during CONFIRM. The
        # online-learning variant (non-headline) instead keeps learning on CONFIRM.
        pol_mem = PolicyMemory.from_snapshot(snapshot, frozen=not online_confirm)
        pol = run_condition(con_items, agent, providers, pol_mem,
                            condition="policy_memory", budget=b, weights=weights,
                            explore=False, update_memory=online_confirm,
                            dataset_version=dataset_version)
        rnd = run_condition(con_items, agent, providers,
                            random_memory(opt_items, agent, params), condition="random_memory",
                            budget=b, weights=weights, explore=False, dataset_version=dataset_version)
        runs += [ConditionRun("no_memory_search", b, base.outcomes),
                 ConditionRun("policy_memory", b, pol.outcomes),
                 ConditionRun("random_memory", b, rnd.outcomes)]
        debug_records += base.debug + pol.debug + rnd.debug
        aligned[b] = (base.outcomes, pol.outcomes, rnd.outcomes)
        budget_ok = budget_ok and all(
            o.tool_calls <= b for o in base.outcomes + pol.outcomes + rnd.outcomes)
        if b == gate_budget:
            replay_log = pol.log

    step("paired statistical tests (McNemar + bootstrap CI) at the gate budget")
    base_out, pol_out, _ = aligned[gate_budget]
    bmap = {o.item_id: o for o in base_out}
    pmap = {o.item_id: o for o in pol_out}
    ids = [i for i in bmap if i in pmap]
    mc = mcnemar([bmap[i].correct for i in ids], [pmap[i].correct for i in ids])
    ci = bootstrap_correct_per_tool_call([int(pmap[i].correct) for i in ids],
                                         [pmap[i].tool_calls for i in ids])
    significance = {"budget": gate_budget, "mcnemar": mc.to_dict(),
                    "policy_correct_per_tool_call_ci": ci.to_dict()}

    step("replay check (graph is a deterministic projection of the log)")
    replay = replay_check(replay_log).to_dict() if replay_log is not None else {}

    step("same-conditions check (baseline vs policy differ only in memory access)")
    base_spec = _condition_spec(cfg, search_tools, gate_budget, split, "none")
    pol_spec = _condition_spec(cfg, search_tools, gate_budget, split, "frozen_snapshot")
    sc = same_conditions(base_spec, pol_spec)

    step("headline-eligibility computation")
    snap_dict = snapshot.to_dict()
    leakage_details = leakage_check_details(snap_dict, items)
    leak_ok = leakage_details["memory_snapshot_leakage_pass"]
    dataset_is_real = dataset_label in _REAL_DATASETS
    # full_pipeline always runs all four comparison conditions.
    conditions_present = ["closed_book", "no_memory_search", "policy_memory", "random_memory"]
    checks = {
        "optimize_confirm_disjoint": set(split.optimize_ids).isdisjoint(split.confirm_ids),
        "no_answer_leakage": leak_ok,
        "replay_passed": bool(replay.get("projection_matches")),
        "runs_completed": bool(aligned[gate_budget][0]) and bool(aligned[gate_budget][1]),
        "budget_enforced": budget_ok,
        "confirm_memory_frozen": not online_confirm,
        "no_live_updates_during_confirm": not online_confirm,
        "same_conditions": sc.ok,
    }
    all_outcomes = [o for r in runs for o in r.outcomes]
    total_calls = sum(o.tool_calls for o in all_outcomes)
    total_failed = sum(o.failed_tool_calls for o in all_outcomes)
    pf_rate = (total_failed / total_calls) if total_calls else 0.0
    failed_conditions = []
    for cond in conditions_present:
        cc = sum(o.tool_calls for o in all_outcomes if o.condition == cond)
        cf = sum(o.failed_tool_calls for o in all_outcomes if o.condition == cond)
        if cc > 0 and cf == cc:
            failed_conditions.append(cond)
    hcfg = cfg.get("headline", {})
    min_confirm = hcfg.get("min_confirm_size", 20)
    eligibility = compute_eligibility(checks, dataset_is_real=dataset_is_real,
                                      conditions_present=conditions_present,
                                      confirm_size=len(con_items), min_confirm=min_confirm,
                                      provider_failure_rate=pf_rate,
                                      failed_conditions=failed_conditions,
                                      max_provider_failure_rate=hcfg.get("max_provider_failure_rate", 0.2))

    meta = {
        "answer_model": cfg.get("live", {}).get("answer_model"),
        "search_baseline": cfg.get("live", {}).get("search_baseline"),
        "tools_enabled": search_tools,
        "embedder": "hash_embedder",
        "query_decomposition_enabled": bool(cfg.get("policy", {}).get("enable_query_decomposition", False)),
        "iterative_clue_resolution_enabled": bool(cfg.get("policy", {}).get("enable_iterative_clue_resolution", False)),
        "task_frame_enabled": bool(cfg.get("policy", {}).get("enable_task_frame", False)),
        "llm_task_frame_parser_enabled": bool(
            cfg.get("policy", {}).get("enable_llm_task_frame_parser", False)),
        "dataset": dataset_label,
        "dataset_version": dataset_version,
        "split": split.to_dict() | {"optimize_ids": "...", "confirm_ids": "..."},
        "budgets": budgets,
        "memory_condition": "frozen_policy_memory",
        "policy_condition": f"query={cfg['policy']['query_mode']},stop={cfg['policy']['stop_mode']}",
        "limitations": [
            f"Dataset is `{dataset_label}` — a synthetic/placeholder harness, NOT a real benchmark score.",
            "The synthetic answerer models a competent extractor; it isolates retrieval/epistemic policy.",
            "No live providers were called; only fixture-backed, deterministic tools.",
        ],
        "not_claimed": [
            "No claim of BrowseComp or LiveBrowseComp performance.",
            "No claim base model weights changed (they do not).",
            "No claim benchmark answers are stored in policy memory (they are not).",
        ],
    }
    step("write report artifacts + run manifest")
    run_dir = write_full_report(run_id, runs=runs, snapshot=snapshot.to_dict(),
                                replay=replay, significance=significance, meta=meta,
                                eligibility=eligibility.to_dict(),
                                same_conditions=sc.to_dict(),
                                condition_specs={"no_memory_search": base_spec.to_dict(),
                                                 "policy_memory": pol_spec.to_dict()},
                                leakage_details=leakage_details, debug_records=debug_records,
                                results_root=results_root)

    # Run manifest (no secrets) — written for every run dir so any run is
    # auditable. The eligibility/cost here describe the run that just executed.
    cost = estimate_calls(
        n_optimize=len(opt_items), n_confirm=len(con_items), budgets=budgets,
        passes=mem_cfg.get("experience_passes", 4),
        experience_budget=mem_cfg.get("experience_budget", 5),
        judge=cfg.get("grading", {}).get("judge", "exact"),
        prices=cfg.get("pricing"),
    ).to_dict()
    manifest = build_manifest(
        run_id=run_id, cfg=cfg, dataset_label=dataset_label,
        dataset_version=dataset_version, dataset_checksum=_dataset_checksum(items),
        dataset_path=None, split=split, search_tools=search_tools, tools_cfg=None,
        memory_cfg=mem_cfg, eligibility_preflight=eligibility.to_dict(),
        cost_estimate=cost, live=False,
    )
    write_manifest(run_dir, manifest)
    # Snapshot the effective config into the run dir so it is part of the
    # hashable artifact set (scripts/hash_artifacts.py).
    (Path(run_dir) / "config_snapshot.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")

    headline = {b: {
        "no_memory_search": compute_metrics(aligned[b][0]),
        "policy_memory": compute_metrics(aligned[b][1]),
        "random_memory": compute_metrics(aligned[b][2]),
    } for b in budgets}
    return {"run_dir": str(run_dir), "split": split, "headline": headline,
            "closed_book": compute_metrics(cb.outcomes),
            "significance": significance, "replay": replay, "snapshot": snapshot,
            "eligibility": eligibility.to_dict(), "same_conditions": sc.to_dict(),
            "budgets": budgets, "gate_budget": gate_budget}
