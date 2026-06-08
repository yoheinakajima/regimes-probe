"""Live run planning (dry-run) + execution pipeline.

``build_plan`` is pure and never calls a provider — it sizes the run, estimates
calls, writes a dry-run manifest, and reports optimistic eligibility.
``run_live_pipeline`` executes the requested conditions through the existing
provider-agnostic harness; it receives already-built providers + agents (so it is
mockable and never constructs live clients itself).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from regimes_probe.activegraph_pack import replay_check
from regimes_probe.datasets.base import Item
from regimes_probe.eval.conditions import ConditionSpec, same_conditions
from regimes_probe.eval.eligibility import compute_eligibility
from regimes_probe.eval.harness import experience_phase, run_condition
from regimes_probe.eval.manifest import build_manifest, write_manifest
from regimes_probe.eval.metrics import compute_metrics
from regimes_probe.eval.report import ConditionRun, write_full_report
from regimes_probe.eval.significance import bootstrap_correct_per_tool_call, mcnemar
from regimes_probe.eval.split import build_split, partition
from regimes_probe.policy.consolidation import consolidate
from regimes_probe.policy.contextual_bandit import BanditParams
from regimes_probe.policy.memory import PolicyMemory
from regimes_probe.policy.policy_fragment import assert_no_answer_leakage

ALL_CONDITIONS = ("closed_book", "no_memory_search", "random_memory", "policy_memory")
_SEARCH_CONDS = ("no_memory_search", "random_memory", "policy_memory")
_REAL_DATASETS = {"BrowseComp", "LiveBrowseComp", "browsecomp", "livebrowsecomp"}


def _dataset_checksum(items) -> str:
    import hashlib
    h = hashlib.sha256()
    for it in sorted(items, key=lambda i: i.id):
        h.update(it.id.encode()); h.update(b"\x00"); h.update(it.question.encode())
    return h.hexdigest()[:16]


def subsample_and_split(items: list[Item], *, optimize: int, confirm: int,
                        seed: str, mode: str = "hash"):
    want = optimize + confirm
    subset = sorted(items, key=lambda i: i.id)[:want]
    frac = confirm / max(1, (optimize + confirm))
    split = build_split(subset, confirm_fraction=frac, salt=seed, mode=mode)
    split.assert_disjoint()
    opt, con = partition(subset, split)
    return subset, split, opt, con


def estimate_live(conditions, budgets, *, n_opt, n_con, passes, exp_budget, judge,
                  settings: Optional[dict] = None):
    s = settings or {}
    search_conds = [c for c in conditions if c in _SEARCH_CONDS]
    has_exp = "policy_memory" in conditions
    exp_attempts = n_opt * passes if has_exp else 0
    cb_attempts = n_con if "closed_book" in conditions else 0
    search_attempts = n_con * len(search_conds) * len(budgets)
    total = exp_attempts + cb_attempts + search_attempts
    worst_exp_tools = exp_attempts * exp_budget
    worst_search_tools = n_con * len(search_conds) * sum(budgets)
    worst_tools = worst_exp_tools + worst_search_tools
    # Each enabled search tool is one bandit arm and could receive up to the full
    # worst-case tool budget (the router may pick it every time).
    tools = s.get("tools", [])
    max_calls_by_tool = {t: (0 if t == "page_fetch" else worst_tools) for t in tools}
    max_calls_by_tool["page_fetch"] = worst_tools if "page_fetch" in tools else 0
    return {
        "conditions": list(conditions),
        "budgets": list(budgets),
        "n_optimize": n_opt, "n_confirm": n_con, "passes": passes,
        # models / tools in play (so cost is legible at a glance)
        "answer_model": s.get("answer_model"),
        "web_search_model": s.get("web_search_model"),
        "web_search_context_size": s.get("web_search_context_size"),
        "provider_mode": s.get("provider_mode"),
        "enabled_tools": tools,
        "provider_classes": s.get("provider_classes", []),
        "tools_meta": s.get("tools_meta", {}),
        "openai_web_search_enabled": s.get("openai_web_search_enabled"),
        "agentic_tool_discovery_enabled": s.get("agentic_tool_discovery_enabled"),
        "scrape_tools_enabled": s.get("scrape_tools_enabled"),
        "browserish_tools_enabled": s.get("browserish_tools_enabled"),
        "stateful_or_paid_tools_allowed": s.get("stateful_or_paid_tools_allowed"),
        "answerer_calls": total,
        "grader_calls": total,
        "judge_calls": total if judge == "llm" else 0,
        "worst_case_tool_calls": worst_tools,
        "worst_case_search_calls": worst_tools,
        "worst_case_page_fetch_calls": worst_tools,
        "max_calls_by_tool": max_calls_by_tool,
        "experience_attempts": exp_attempts,
        "closed_book_attempts": cb_attempts,
        "search_attempts": search_attempts,
        "warnings": list(s.get("warnings", [])),
        "estimated_cost_usd": "unknown",
    }


def _sha12(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def _spec(cfg, search_tools, budget, split, memory_access) -> ConditionSpec:
    from regimes_probe.agent import prompts
    return ConditionSpec(
        answer_model=cfg.get("live", {}).get("answer_model", "gpt-5.4-mini"),
        answer_prompt_version=prompts.fingerprint("answerer"),
        enabled_tools=tuple(sorted(search_tools)), tool_budget=budget,
        split_id=_sha12(split.mode + "|" + split.salt + "|" + "|".join(split.confirm_ids)),
        grader="normalized_match",
        provider_config_id=_sha12(",".join(sorted(search_tools))),
        query_policy=cfg["policy"]["query_mode"], verify_policy="default",
        stop_policy=cfg["policy"]["stop_mode"], memory_access=memory_access)


@dataclass
class LivePlan:
    run_id: str
    dataset_label: str
    dataset_version: str
    is_real: bool
    conditions: list[str]
    budgets: list[int]
    n_optimize: int
    n_confirm: int
    cost: dict[str, Any]
    eligibility_preflight: dict[str, Any]
    manifest: dict[str, Any]
    run_dir: str

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        return d


def build_plan(cfg, items, *, conditions, budgets, optimize, confirm, split_seed,
               run_id, results_root, dataset_label, dataset_version, dataset_path,
               is_real, search_tools, tools_cfg=None, live_settings=None) -> LivePlan:
    """Plan a live run and write a dry-run manifest. No provider calls."""
    mem_cfg = cfg.get("memory", {})
    subset, split, opt, con = subsample_and_split(
        items, optimize=optimize, confirm=confirm, seed=split_seed,
        mode=cfg.get("split", {}).get("mode", "hash"))
    cost = estimate_live(conditions, budgets, n_opt=len(opt), n_con=len(con),
                         passes=mem_cfg.get("experience_passes", 4),
                         exp_budget=mem_cfg.get("experience_budget", 5),
                         judge=cfg.get("grading", {}).get("judge", "exact"),
                         settings=live_settings)
    # Preflight is optimistic about structural checks, but it must be HONEST about
    # which conditions are planned: a plan without policy_memory can never be a
    # headline memory claim.
    planned = list(conditions)
    has_comparison = ("no_memory_search" in planned and "policy_memory" in planned)
    if has_comparison:
        base_spec = _spec(cfg, search_tools, budgets[0], split, "none")
        pol_spec = _spec(cfg, search_tools, budgets[0], split, "frozen_snapshot")
        sc = same_conditions(base_spec, pol_spec)
        sc_ok = sc.ok
    else:
        sc_ok = False
    has_policy = "policy_memory" in planned
    checks = {
        "optimize_confirm_disjoint": True,
        "no_answer_leakage": True, "replay_passed": True,
        "runs_completed": True, "budget_enforced": True,
        "confirm_memory_frozen": bool(mem_cfg.get("confirm_uses_frozen_snapshot")) if has_policy else False,
        "no_live_updates_during_confirm": not mem_cfg.get("confirm_updates_memory"),
        "same_conditions": sc_ok,
    }
    min_confirm = cfg.get("headline", {}).get("min_confirm_size", 20)
    elig = compute_eligibility(checks, dataset_is_real=is_real, conditions_present=planned,
                               confirm_size=len(con), min_confirm=min_confirm)
    manifest = build_manifest(
        run_id=run_id, cfg=cfg, dataset_label=dataset_label, dataset_version=dataset_version,
        dataset_checksum=_dataset_checksum(subset), dataset_path=dataset_path, split=split,
        search_tools=search_tools, tools_cfg=tools_cfg, memory_cfg=mem_cfg,
        eligibility_preflight=elig.to_dict(), cost_estimate=cost, live=False,
        live_settings=live_settings)
    run_dir = Path(results_root) / run_id
    write_manifest(run_dir, manifest)
    (run_dir / "plan.json").write_text(json.dumps({
        "conditions": list(conditions), "budgets": list(budgets),
        "n_optimize": len(opt), "n_confirm": len(con), "dataset": dataset_label,
        "is_real": is_real, "live_settings": live_settings or {}, "cost_estimate": cost,
        "conditions_present": planned,
        "same_conditions": (sc.to_dict() if has_comparison
                            else {"ok": False, "not_applicable": True,
                                  "reason": "policy_memory and/or no_memory_search not planned"}),
        "eligibility_preflight": elig.to_dict(),
        "executed": False}, indent=2), encoding="utf-8")
    import yaml
    (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True),
                                                  encoding="utf-8")
    return LivePlan(run_id=run_id, dataset_label=dataset_label, dataset_version=dataset_version,
                    is_real=is_real, conditions=list(conditions), budgets=list(budgets),
                    n_optimize=len(opt), n_confirm=len(con), cost=cost,
                    eligibility_preflight=elig.to_dict(), manifest=manifest, run_dir=str(run_dir))


def _random_memory(items, agent, params, seed=7) -> PolicyMemory:
    rng = random.Random(seed)
    mem = PolicyMemory(params.copy())
    fams = {"tool": ["news_search", "openai_web_search", "page_fetch"],
            "query": ["direct_question", "freshness_terms", "keyword_compressed"],
            "stop": ["stop_now", "search_more", "fetch_page"]}
    for it in items:
        sig = agent.signature(it)
        rewards = {f: {rng.choice(a): rng.uniform(-1, 1)} for f, a in fams.items()}
        mem.observe(sig, f"rand-{it.id}", rewards, correct=bool(rng.random() < 0.5), tool_calls=1)
    return mem


def run_live_pipeline(cfg, items, *, providers, search_agent, cb_agent, cache,
                      conditions, budgets, optimize, confirm, split_seed, run_id,
                      results_root, dataset_label, dataset_version, dataset_path,
                      is_real, search_tools, weights, params, tools_cfg=None,
                      resume_snapshot: Optional[dict] = None,
                      live_settings: Optional[dict] = None) -> dict[str, Any]:
    """Execute the requested conditions. Providers/agents are injected (mockable)."""
    mem_cfg = cfg.get("memory", {})
    subset, split, opt, con = subsample_and_split(
        items, optimize=optimize, confirm=confirm, seed=split_seed,
        mode=cfg.get("split", {}).get("mode", "hash"))
    ver = dataset_version
    runs: list[ConditionRun] = []
    aligned: dict[int, dict] = {}
    replay_log = None
    budget_ok = True
    gate_budget = 3 if 3 in budgets else budgets[0]

    if "closed_book" in conditions:
        cb = run_condition(con, cb_agent, providers, PolicyMemory(params.copy()),
                           condition="closed_book", budget=0, weights=weights,
                           dataset_version=ver)
        runs.append(ConditionRun("closed_book", 0, cb.outcomes))

    snapshot = None
    if "policy_memory" in conditions:
        if resume_snapshot is not None:
            from regimes_probe.policy.memory import PolicyMemorySnapshot
            snapshot = PolicyMemorySnapshot.from_dict(resume_snapshot)   # skip experience
        else:
            mem = PolicyMemory(params.copy(), nearest_k=mem_cfg.get("nearest_k", 8))
            experience_phase(opt, search_agent, providers, mem,
                             budget=mem_cfg.get("experience_budget", 5),
                             passes=mem_cfg.get("experience_passes", 4),
                             weights=weights, dataset_version=ver)
            # Consolidate raw traces into answer-free policy fragments BEFORE
            # freezing, so the CONFIRM snapshot carries fragments (mirrors
            # scripts/run_experience.py). assert_no_answer_leakage runs inside
            # snapshot() over bandits, traces, and fragments.
            consolidate(mem)
            snapshot = mem.snapshot(meta={"dataset_version": ver, "n_optimize": len(opt)})

    for b in budgets:
        per: dict[str, Any] = {}
        if "no_memory_search" in conditions:
            r = run_condition(con, search_agent, providers, PolicyMemory(params.copy()),
                              condition="no_memory_search", budget=b, weights=weights,
                              dataset_version=ver)
            runs.append(ConditionRun("no_memory_search", b, r.outcomes)); per["no_memory_search"] = r
        if "policy_memory" in conditions:
            r = run_condition(con, search_agent, providers,
                              PolicyMemory.from_snapshot(snapshot, frozen=True),
                              condition="policy_memory", budget=b, weights=weights,
                              dataset_version=ver)
            runs.append(ConditionRun("policy_memory", b, r.outcomes)); per["policy_memory"] = r
            if b == gate_budget:
                replay_log = r.log
        if "random_memory" in conditions:
            r = run_condition(con, search_agent, providers,
                              _random_memory(opt, search_agent, params),
                              condition="random_memory", budget=b, weights=weights,
                              dataset_version=ver)
            runs.append(ConditionRun("random_memory", b, r.outcomes)); per["random_memory"] = r
        if replay_log is None and per:
            replay_log = next(iter(per.values())).log
        aligned[b] = per
        for cr in per.values():
            budget_ok = budget_ok and all(o.tool_calls <= b for o in cr.outcomes)

    significance = {}
    if "no_memory_search" in conditions and "policy_memory" in conditions:
        base_out = aligned[gate_budget]["no_memory_search"].outcomes
        pol_out = aligned[gate_budget]["policy_memory"].outcomes
        bmap = {o.item_id: o for o in base_out}; pmap = {o.item_id: o for o in pol_out}
        ids = [i for i in bmap if i in pmap]
        if ids:
            mc = mcnemar([bmap[i].correct for i in ids], [pmap[i].correct for i in ids])
            ci = bootstrap_correct_per_tool_call([int(pmap[i].correct) for i in ids],
                                                 [pmap[i].tool_calls for i in ids])
            significance = {"budget": gate_budget, "mcnemar": mc.to_dict(),
                            "policy_correct_per_tool_call_ci": ci.to_dict()}

    replay = replay_check(replay_log).to_dict() if replay_log is not None else {}
    snap_dict = snapshot.to_dict() if snapshot is not None else {}
    try:
        assert_no_answer_leakage(snap_dict, "policy_memory_snapshot")
        leak_ok = not any(g in json.dumps(snap_dict)
                          for it in items for g in it.gold_answers() if g)
    except Exception:
        leak_ok = False

    # Conditions that actually completed (one ConditionRun per condition×budget).
    conditions_present = sorted({r.condition for r in runs})
    has_comparison = ("no_memory_search" in conditions_present
                      and "policy_memory" in conditions_present)
    # same-conditions only means something when BOTH compared conditions ran.
    if has_comparison:
        base_spec = _spec(cfg, search_tools, gate_budget, split, "none")
        pol_spec = _spec(cfg, search_tools, gate_budget, split, "frozen_snapshot")
        sc = same_conditions(base_spec, pol_spec)
        sc_ok, sc_dict, specs = sc.ok, sc.to_dict(), {
            "no_memory_search": base_spec.to_dict(), "policy_memory": pol_spec.to_dict()}
    else:
        sc_ok, sc_dict, specs = False, {"ok": False, "not_applicable": True,
                                        "reason": "policy_memory and/or no_memory_search not run"}, {}
    has_policy = "policy_memory" in conditions_present
    checks = {
        "optimize_confirm_disjoint": set(split.optimize_ids).isdisjoint(split.confirm_ids),
        "no_answer_leakage": leak_ok,
        "replay_passed": bool(replay.get("projection_matches")),
        "runs_completed": bool(runs),
        "budget_enforced": budget_ok,
        # memory-claim checks (vacuously fine when no policy_memory; eligibility
        # still fails via missing-conditions, so the verdict is honest):
        "confirm_memory_frozen": True if has_policy else False,
        "no_live_updates_during_confirm": True,
        "same_conditions": sc_ok,
    }
    # Provider-failure stats: overall rate + any condition that failed entirely.
    total_calls = sum(o.tool_calls for r in runs for o in r.outcomes)
    total_failed = sum(o.failed_tool_calls for r in runs for o in r.outcomes)
    pf_rate = (total_failed / total_calls) if total_calls else 0.0
    failed_conditions = []
    for cond in conditions_present:
        cc = sum(o.tool_calls for r in runs if r.condition == cond for o in r.outcomes)
        cf = sum(o.failed_tool_calls for r in runs if r.condition == cond for o in r.outcomes)
        if cc > 0 and cf == cc:
            failed_conditions.append(cond)
    hcfg = cfg.get("headline", {})
    min_confirm = hcfg.get("min_confirm_size", 20)
    elig = compute_eligibility(checks, dataset_is_real=is_real,
                               conditions_present=conditions_present,
                               confirm_size=len(con), min_confirm=min_confirm,
                               provider_failure_rate=pf_rate, failed_conditions=failed_conditions,
                               max_provider_failure_rate=hcfg.get("max_provider_failure_rate", 0.2))

    ls = live_settings or {}
    meta = {
        "answer_model": ls.get("answer_model") or cfg.get("live", {}).get("answer_model"),
        "web_search_model": ls.get("web_search_model"),
        "web_search_context_size": ls.get("web_search_context_size"),
        "provider_mode": ls.get("provider_mode"),
        "openai_web_search_enabled": ls.get("openai_web_search_enabled"),
        "search_baseline": cfg.get("live", {}).get("search_baseline"),
        "tools_enabled": search_tools, "embedder": "hash_embedder",
        "dataset": dataset_label, "dataset_version": ver,
        "split": split.to_dict() | {"optimize_ids": "...", "confirm_ids": "..."},
        "budgets": budgets, "memory_condition": "frozen_policy_memory",
        "policy_condition": f"query={cfg['policy']['query_mode']},stop={cfg['policy']['stop_mode']}",
        "cache": cache.summary() if cache else {},
        "limitations": [
            f"Dataset is `{dataset_label}`." + ("" if is_real else " Synthetic/placeholder — NOT a benchmark."),
            "Live providers were used only if --execute was passed; cache/replay may have served some calls.",
        ],
        "not_claimed": ["No benchmark claim unless headline_eligible is true and criteria in "
                        "docs/FIRST_REAL_RESULT_CRITERIA.md are met."],
    }
    run_dir = write_full_report(run_id, runs=runs, snapshot=snap_dict, replay=replay,
                                significance=significance, meta=meta,
                                eligibility=elig.to_dict(), same_conditions=sc_dict,
                                condition_specs=specs, results_root=results_root)
    cost = estimate_live(conditions, budgets, n_opt=len(opt), n_con=len(con),
                         passes=mem_cfg.get("experience_passes", 4),
                         exp_budget=mem_cfg.get("experience_budget", 5),
                         judge=cfg.get("grading", {}).get("judge", "exact"),
                         settings=live_settings)
    manifest = build_manifest(
        run_id=run_id, cfg=cfg, dataset_label=dataset_label, dataset_version=ver,
        dataset_checksum=_dataset_checksum(subset), dataset_path=dataset_path, split=split,
        search_tools=search_tools, tools_cfg=tools_cfg, memory_cfg=mem_cfg,
        eligibility_preflight=elig.to_dict(), cost_estimate=cost, live=True,
        live_settings=live_settings)
    manifest["cache"] = cache.summary() if cache else {}
    write_manifest(run_dir, manifest)
    import yaml
    (Path(run_dir) / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True),
                                                         encoding="utf-8")
    return {"run_dir": str(run_dir), "eligibility": elig.to_dict(),
            "headline": {b: {k: compute_metrics(v.outcomes) for k, v in aligned[b].items()}
                         for b in budgets},
            "significance": significance, "replay": replay,
            "cache": cache.summary() if cache else {}}
