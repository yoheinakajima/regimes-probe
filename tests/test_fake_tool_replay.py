"""Fake tool calls are replayable; the graph is a deterministic projection."""

from __future__ import annotations

from regimes_probe.activegraph_pack import (
    EventLog,
    RecordingInvoker,
    ReplayInvoker,
    record_attempt,
    replay_check,
)
from regimes_probe.policy.contextual_bandit import BanditParams
from regimes_probe.policy.memory import PolicyMemory


def test_recording_then_replay_reproduces_responses(items, providers):
    item = items[0]
    log = EventLog(run_id="rec")
    rinv = RecordingInvoker(providers, log, attempt_id="a")
    r1 = rinv.call("news_search", "acme robotics 2026 latest", limit=5)
    r2 = rinv.call("generic_web_search", "acme robotics", limit=5)

    replay = ReplayInvoker(rinv.recorded)
    rr1 = replay.call("news_search", "acme robotics 2026 latest", limit=5)
    rr2 = replay.call("generic_web_search", "acme robotics", limit=5)
    assert rr1.to_dict() == r1.to_dict()
    assert rr2.to_dict() == r2.to_dict()


def test_replay_divergence_is_detected(items, providers):
    log = EventLog(run_id="rec")
    rinv = RecordingInvoker(providers, log, attempt_id="a")
    rinv.call("news_search", "q1", limit=5)
    replay = ReplayInvoker(rinv.recorded)
    try:
        replay.call("brave_search", "q1", limit=5)  # wrong tool
    except AssertionError:
        return
    raise AssertionError("expected replay divergence to be detected")


def test_recorded_run_projection_replays(items, providers, agent):
    log = EventLog(run_id="proj")
    mem = PolicyMemory(BanditParams())
    record_attempt(log, "benchmark_run#1", agent, items[0], mem, providers,
                   budget=3, explore=False, attempt_id="a1")
    report = replay_check(log)
    assert report.projection_matches is True
    assert report.n_events > 0


def test_attempt_is_deterministic(items, providers, agent):
    mem = PolicyMemory(BanditParams())
    t1 = agent.attempt(items[0], mem, providers, budget=5, explore=False, attempt_id="a")
    mem2 = PolicyMemory(BanditParams())
    t2 = agent.attempt(items[0], mem2, providers, budget=5, explore=False, attempt_id="a")
    assert t1.tools_used() == t2.tools_used()
    assert t1.query_arms_used() == t2.query_arms_used()
    assert t1.final_answer == t2.final_answer
