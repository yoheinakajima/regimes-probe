"""Policy memory must never carry final answer text (leakage control)."""

from __future__ import annotations

import json

import pytest

from regimes_probe.eval.harness import experience_phase
from regimes_probe.policy.contextual_bandit import BanditParams
from regimes_probe.policy.memory import FrozenMemoryError, PolicyMemory
from regimes_probe.policy.policy_fragment import assert_no_answer_leakage


def test_forbidden_keys_raise():
    with pytest.raises(ValueError):
        assert_no_answer_leakage({"tool_rewards": {"x": {"answer": "secret"}}})
    with pytest.raises(ValueError):
        assert_no_answer_leakage({"final_answer": "leak"})


def test_snapshot_contains_no_gold_answer(items, providers, agent):
    mem = PolicyMemory(BanditParams())
    experience_phase(items, agent, providers, mem, budget=5, passes=2)
    snap = mem.snapshot()
    blob = json.dumps(snap.to_dict())
    # No gold answer string from any item may appear in the snapshot.
    for it in items:
        for gold in it.gold_answers():
            assert gold not in blob, f"answer leaked into snapshot: {gold!r}"


def test_traces_store_rewards_not_answers(items, providers, agent):
    mem = PolicyMemory(BanditParams())
    experience_phase(items, agent, providers, mem, budget=3, passes=1)
    assert mem.traces
    for t in mem.traces:
        d = t.to_dict()  # raises if a forbidden key sneaks in
        assert "arm_rewards" in d and "embedding" in d
        assert "answer" not in d


def test_frozen_snapshot_rejects_mutation(items, providers, agent):
    mem = PolicyMemory(BanditParams())
    experience_phase(items, agent, providers, mem, budget=3, passes=1)
    frozen = PolicyMemory.from_snapshot(mem.snapshot(), frozen=True)
    sig = agent.signature(items[0])
    with pytest.raises(FrozenMemoryError):
        frozen.observe(sig, "x", {"tool": {"news_search": 1.0}}, correct=True, tool_calls=1)
