"""Budget caps are strictly binding across conditions and modes."""

from __future__ import annotations

import pytest

from regimes_probe.eval.harness import experience_phase, run_condition
from regimes_probe.policy.contextual_bandit import BanditParams
from regimes_probe.policy.memory import PolicyMemory


@pytest.mark.parametrize("budget", [1, 3, 5, 10])
def test_attempt_never_exceeds_budget(items, providers, agent, budget):
    mem = PolicyMemory(BanditParams())
    res = run_condition(items[:8], agent, providers, mem, condition="cap",
                        budget=budget, explore=True, update_memory=True)
    for o in res.outcomes:
        assert o.tool_calls <= budget


def test_budget_binds_after_experience(items, providers, agent):
    mem = PolicyMemory(BanditParams())
    experience_phase(items, agent, providers, mem, budget=5, passes=2)
    snap = mem.snapshot()
    frozen = PolicyMemory.from_snapshot(snap, frozen=True)
    res = run_condition(items, agent, providers, frozen, condition="confirm",
                        budget=1, explore=False)
    assert all(o.tool_calls <= 1 for o in res.outcomes)


def test_always_full_uses_whole_budget(items, providers):
    from regimes_probe.agent.planner import AgentConfig, EpistemicAgent
    tools = ["generic_web_search", "news_search", "official_domain_search", "brave_search"]
    ag = EpistemicAgent(AgentConfig(available_tools=tools + ["page_fetch"],
                                    stop_mode="always_full"))
    mem = PolicyMemory(BanditParams())
    res = run_condition(items[:6], ag, providers, mem, condition="full", budget=3, explore=False)
    # always_full never stops early; with >=budget tools it spends the budget.
    assert all(o.tool_calls == 3 for o in res.outcomes)
