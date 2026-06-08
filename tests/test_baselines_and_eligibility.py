"""Closed-book baseline, same-conditions validator, and headline eligibility."""

from __future__ import annotations

from regimes_probe.agent.answerer import build_closed_book_knowledge
from regimes_probe.agent.planner import AgentConfig, EpistemicAgent, build_closed_book_agent
from regimes_probe.eval.conditions import ConditionSpec, same_conditions
from regimes_probe.eval.eligibility import REQUIRED_CHECKS, compute_eligibility
from regimes_probe.eval.harness import run_condition
from regimes_probe.policy.contextual_bandit import BanditParams
from regimes_probe.policy.memory import PolicyMemory

TOOLS = ["generic_web_search", "news_search", "official_domain_search", "brave_search"]


# ----------------------------------------------------------------- closed book
def test_closed_book_makes_no_tool_calls(items, providers):
    knowledge = build_closed_book_knowledge(items)
    assert knowledge  # the synthetic fixture marks some items intrinsic_knowable
    cfg = AgentConfig(available_tools=TOOLS + ["page_fetch"])
    cb = build_closed_book_agent(cfg, knowledge)
    res = run_condition(items, cb, providers, PolicyMemory(BanditParams()),
                        condition="closed_book", budget=0)
    assert all(o.tool_calls == 0 for o in res.outcomes)        # no tools used
    # it answers exactly the items it "knows" and abstains on the rest
    knowable = {i.id for i in items if i.meta.get("intrinsic_knowable")}
    correct_ids = {o.item_id for o in res.outcomes if o.correct}
    assert correct_ids <= knowable
    assert any(o.correct for o in res.outcomes)


def test_closed_book_abstains_without_knowledge(items, providers):
    cfg = AgentConfig(available_tools=TOOLS + ["page_fetch"])
    cb = build_closed_book_agent(cfg, knowledge={})
    res = run_condition(items[:5], cb, providers, PolicyMemory(BanditParams()),
                        condition="closed_book", budget=0)
    assert all(o.abstained for o in res.outcomes)
    assert all(o.tool_calls == 0 for o in res.outcomes)


# --------------------------------------------------------- same-conditions
def _spec(**over):
    base = dict(answer_model="gpt-5.5", answer_prompt_version="v0",
                enabled_tools=("a", "b"), tool_budget=3, split_id="S1",
                grader="normalized_match", provider_config_id="P1",
                query_policy="learned", verify_policy="default", stop_policy="learned",
                memory_access="none")
    base.update(over)
    return ConditionSpec(**base)


def test_same_conditions_ok_when_only_memory_differs():
    base = _spec(memory_access="none")
    treat = _spec(memory_access="frozen_snapshot")
    r = same_conditions(base, treat)
    assert r.ok and r.unexpected_diffs == [] and r.diffs == ["memory_access"]


def test_same_conditions_fails_on_unexpected_diff():
    base = _spec(memory_access="none", tool_budget=3)
    treat = _spec(memory_access="frozen_snapshot", tool_budget=5)  # budget differs!
    r = same_conditions(base, treat)
    assert not r.ok and "tool_budget" in r.unexpected_diffs


def test_same_conditions_allows_intended_query_variation():
    base = _spec(memory_access="none", query_policy="fixed")
    treat = _spec(memory_access="frozen_snapshot", query_policy="learned")
    r = same_conditions(base, treat, intended_diffs=("memory_access", "query_policy"))
    assert r.ok


# --------------------------------------------------------------- eligibility
_FOUR = ["closed_book", "no_memory_search", "random_memory", "policy_memory"]


def _all_pass():
    return {c: True for c in REQUIRED_CHECKS}


def test_eligibility_requires_real_dataset():
    e = compute_eligibility(_all_pass(), dataset_is_real=False,
                            conditions_present=_FOUR, confirm_size=24)
    assert e.structurally_valid is True
    assert e.headline_eligible_memory_claim is False
    assert any("synthetic" in r or "placeholder" in r for r in e.headline_eligibility_reasons)


def test_eligibility_true_when_real_all_conditions_and_checks_pass():
    e = compute_eligibility(_all_pass(), dataset_is_real=True,
                            conditions_present=_FOUR, confirm_size=24)
    assert e.headline_eligible_memory_claim is True
    assert e.headline_eligible is True and e.headline_eligibility_reasons == []


def test_eligibility_reports_each_failing_check():
    checks = _all_pass()
    checks["replay_passed"] = False
    checks["no_answer_leakage"] = False
    e = compute_eligibility(checks, dataset_is_real=True, conditions_present=_FOUR, confirm_size=24)
    assert e.headline_eligible_memory_claim is False
    assert e.structurally_valid is False
    assert any("replay" in r for r in e.headline_eligibility_reasons)
    assert any("leakage" in r for r in e.headline_eligibility_reasons)


def test_plumbing_run_structurally_valid_but_not_headline():
    # closed_book + no_memory_search only: structurally valid, NOT a memory headline
    e = compute_eligibility(_all_pass(), dataset_is_real=True,
                            conditions_present=["closed_book", "no_memory_search"],
                            confirm_size=24)
    assert e.structurally_valid is True
    assert e.headline_eligible_memory_claim is False
    assert any("policy_memory" in r for r in e.headline_eligibility_reasons)


def test_small_confirm_blocks_headline():
    e = compute_eligibility(_all_pass(), dataset_is_real=True,
                            conditions_present=_FOUR, confirm_size=10, min_confirm=20)
    assert e.structurally_valid is True
    assert e.headline_eligible_memory_claim is False
    assert any("CONFIRM size" in r for r in e.headline_eligibility_reasons)


def test_missing_random_memory_blocks_headline():
    e = compute_eligibility(_all_pass(), dataset_is_real=True,
                            conditions_present=["closed_book", "no_memory_search", "policy_memory"],
                            confirm_size=24)
    assert e.headline_eligible_memory_claim is False
    assert any("random_memory" in r for r in e.headline_eligibility_reasons)
