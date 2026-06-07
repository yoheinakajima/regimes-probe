"""Event/object/relation schema is explicit, and a recorded run uses it."""

from __future__ import annotations

from regimes_probe.activegraph_pack import (
    ALL_EVENTS,
    ALL_OBJECTS,
    ALL_RELATIONS,
    EventLog,
    Events,
    Objects,
    record_attempt,
    record_grade_and_reward,
    record_run_start,
)
from regimes_probe.eval.reward import RewardWeights
from regimes_probe.policy.contextual_bandit import BanditParams
from regimes_probe.policy.memory import PolicyMemory


def test_canonical_counts():
    assert len(ALL_EVENTS) == 25
    assert len(ALL_OBJECTS) == 20
    assert len(ALL_RELATIONS) == 14
    # no duplicates
    assert len(set(ALL_EVENTS)) == len(ALL_EVENTS)
    assert len(set(ALL_OBJECTS)) == len(ALL_OBJECTS)
    assert len(set(ALL_RELATIONS)) == len(ALL_RELATIONS)


def test_namespaces_match_lists():
    assert Events.TOOL_REQUESTED in ALL_EVENTS
    assert Objects.QUESTION_ATTEMPT in ALL_OBJECTS


def test_recorded_run_emits_expected_events(items, providers, agent):
    log = EventLog(run_id="schema-test")
    record_run_start(log, dataset_version="synthetic@test", condition="unit",
                     config={"budget": 3})
    mem = PolicyMemory(BanditParams())
    item = items[0]
    rec = record_attempt(log, "benchmark_run#1", agent, item, mem, providers,
                         budget=3, explore=False, attempt_id="a1")
    record_grade_and_reward(log, rec, item, weights=RewardWeights.full(),
                            freshness_sensitive=False)
    types = {e.type for e in log.events}
    for required in (
        Events.BENCHMARK_STARTED, Events.ATTEMPT_STARTED, Events.SIGNATURE_CREATED,
        Events.ROUTING_PLAN_CREATED, Events.TOOL_REQUESTED, Events.TOOL_RESPONDED,
        Events.EVIDENCE_OBSERVED, Events.STOP_DECISION_CREATED,
        Events.FINAL_ANSWER_CREATED, Events.GRADE_COMPLETED, Events.REWARD_COMPUTED,
    ):
        assert required in types, f"missing event {required}"
    # every emitted type is a canonical event type (objects/relations excepted)
    domain = types - {"object.created", "relation.created"}
    assert domain <= set(ALL_EVENTS)
