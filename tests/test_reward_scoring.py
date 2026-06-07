"""Reward computation: components, presets, and credit attribution."""

from __future__ import annotations

from regimes_probe.agent.planner import AgentConfig, EpistemicAgent
from regimes_probe.eval.grader import grade, normalize_answer
from regimes_probe.eval.reward import RewardWeights, compute_rewards
from regimes_probe.policy.contextual_bandit import BanditParams
from regimes_probe.policy.memory import PolicyMemory

TOOLS = ["generic_web_search", "news_search", "official_domain_search", "brave_search"]


def _generic_item(items):
    return next(i for i in items if i.meta.get("regime") == "generic")


def _seed_for(mem, sig, item):
    gt = item.meta["gold_tool"]
    ga = item.meta["gold_query_arm"]
    for _ in range(6):
        mem.bandits["tool"].update_reward(sig.cluster_key, gt, 1.5)
        mem.bandits["query"].update_reward(sig.cluster_key, ga, 1.5)


def _run(items, providers, item, *, seed: bool, budget: int = 3):
    agent = EpistemicAgent(AgentConfig(available_tools=TOOLS + ["page_fetch"]))
    mem = PolicyMemory(BanditParams())
    sig = agent.signature(item)
    if seed:
        _seed_for(mem, sig, item)
    trace = agent.attempt(item, mem, providers, budget=budget, explore=False, attempt_id="t")
    g = grade(item, trace.final_answer)
    gold = [normalize_answer(x) for x in item.gold_answers()]
    rr = compute_rewards(trace, correct=g.correct, gold_norms=gold,
                         weights=RewardWeights.full(),
                         freshness_sensitive=bool(sig.features.get("freshness_sensitive")))
    return trace, g, rr


def test_correct_attempt_rewards_positive(items, providers):
    item = _generic_item(items)
    trace, g, rr = _run(items, providers, item, seed=True)
    assert g.correct is True
    assert rr.flags["found_hit"] is True
    assert rr.components["correctness"] == RewardWeights.full().correctness
    assert rr.attempt_reward > 0
    # the gold tool should receive the highest tool reward
    tool_r = rr.arm_rewards["tool"]
    assert max(tool_r, key=tool_r.get) == item.meta["gold_tool"]


def test_wrong_attempt_no_correctness(items, providers):
    item = _generic_item(items)
    # budget 1: unseeded exploit routes to the alphabetical-first tool
    # (brave_search) which only carries a distractor for a generic-gold item.
    trace, g, rr = _run(items, providers, item, seed=False, budget=1)
    assert g.correct is False
    assert rr.components["correctness"] == 0.0


def test_correctness_only_preset_zeroes_other_terms(items, providers):
    item = _generic_item(items)
    agent = EpistemicAgent(AgentConfig(available_tools=TOOLS + ["page_fetch"]))
    mem = PolicyMemory(BanditParams())
    sig = agent.signature(item)
    _seed_for(mem, sig, item)
    trace = agent.attempt(item, mem, providers, budget=5, explore=False, attempt_id="t")
    gold = [normalize_answer(x) for x in item.gold_answers()]
    rr = compute_rewards(trace, correct=True, gold_norms=gold,
                         weights=RewardWeights.correctness_only(), freshness_sensitive=False)
    assert rr.components["evidence_quality"] == 0.0
    assert rr.components["extra_call_penalty"] == 0.0
    assert rr.components["cost_penalty"] == 0.0


def test_grader_normalization():
    from regimes_probe.datasets.base import Item
    it = Item(id="x", question="q", answer="The White House")
    assert grade(it, "white house").correct
    assert grade(it, "  WHITE   HOUSE ").correct
    assert grade(it, None).abstained
