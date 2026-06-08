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


def build_agent(cfg: dict[str, Any], tools: list[str]) -> EpistemicAgent:
    pol = cfg.get("policy", {})
    agent_cfg = AgentConfig(
        available_tools=tools,
        query_mode=pol.get("query_mode", "learned"),
        stop_mode=pol.get("stop_mode", "learned"),
        as_of=cfg.get("run", {}).get("as_of", "2026-06-01"),
        router=RouterConfig(**cfg.get("router", {})) if cfg.get("router") else RouterConfig(),
        stop=StopConfig(**cfg.get("stopping", {})) if cfg.get("stopping") else StopConfig(),
        verification=VerificationConfig(**cfg.get("verification", {})) if cfg.get("verification") else VerificationConfig(),
    )
    return EpistemicAgent(agent_cfg)


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
import random as _random
from typing import Optional

from regimes_probe.activegraph_pack import replay_check
from regimes_probe.eval.harness import experience_phase, run_condition
from regimes_probe.eval.metrics import compute_metrics
from regimes_probe.eval.report import ConditionRun, write_full_report
from regimes_probe.eval.significance import bootstrap_correct_per_tool_call, mcnemar
from regimes_probe.eval.split import build_split, partition
from regimes_probe.policy.memory import PolicyMemory


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
    snapshot = mem.snapshot(meta={"dataset_version": dataset_version,
                                  "n_optimize": len(opt_items)})

    runs: list[ConditionRun] = []
    aligned: dict[int, tuple] = {}
    replay_log = None
    gate_budget = 3 if 3 in budgets else budgets[0]
    for b in budgets:
        step(f"budget {b}: no_memory / policy_memory / random_memory on CONFIRM")
        base = run_condition(con_items, agent, providers, PolicyMemory(params.copy()),
                             condition="no_memory", budget=b, weights=weights,
                             explore=False, dataset_version=dataset_version)
        frozen = PolicyMemory.from_snapshot(snapshot, frozen=True)
        pol = run_condition(con_items, agent, providers, frozen, condition="policy_memory",
                            budget=b, weights=weights, explore=False, dataset_version=dataset_version)
        rnd = run_condition(con_items, agent, providers,
                            random_memory(opt_items, agent, params), condition="random_memory",
                            budget=b, weights=weights, explore=False, dataset_version=dataset_version)
        runs += [ConditionRun("no_memory", b, base.outcomes),
                 ConditionRun("policy_memory", b, pol.outcomes),
                 ConditionRun("random_memory", b, rnd.outcomes)]
        aligned[b] = (base.outcomes, pol.outcomes, rnd.outcomes)
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

    meta = {
        "answer_model": cfg.get("live", {}).get("answer_model"),
        "search_baseline": cfg.get("live", {}).get("search_baseline"),
        "tools_enabled": search_tools,
        "embedder": "hash_embedder",
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
    step("write report artifacts")
    run_dir = write_full_report(run_id, runs=runs, snapshot=snapshot.to_dict(),
                                replay=replay, significance=significance, meta=meta,
                                results_root=results_root)

    headline = {b: {
        "no_memory": compute_metrics(aligned[b][0]),
        "policy_memory": compute_metrics(aligned[b][1]),
        "random_memory": compute_metrics(aligned[b][2]),
    } for b in budgets}
    return {"run_dir": str(run_dir), "split": split, "headline": headline,
            "significance": significance, "replay": replay, "snapshot": snapshot,
            "budgets": budgets, "gate_budget": gate_budget}
