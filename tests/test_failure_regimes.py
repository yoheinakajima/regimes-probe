"""Generic failure-regime detectors fire, and the fixture exercises a variety."""

from __future__ import annotations

from regimes_probe.agent.planner import AgentConfig, EpistemicAgent
from regimes_probe.eval.harness import run_condition
from regimes_probe.eval.metrics import AttemptOutcome
from regimes_probe.eval.split import build_split, partition
from regimes_probe.policy.contextual_bandit import BanditParams
from regimes_probe.policy.memory import PolicyMemory
from regimes_probe.regimes.detectors import detect_regimes, dominant_regimes

TOOLS = ["generic_web_search", "news_search", "official_domain_search", "brave_search"]


def _outcome(**over) -> AttemptOutcome:
    base = dict(item_id="x", condition="c", budget=3, correct=True, abstained=False,
                tool_calls=3, cost=0.0, latency=0.0, evidence_score=0.8,
                first_tool_hit=False, false_stop=False, over_search=False,
                under_search=False, stale_error=False, found_hit=True,
                cluster_key="k", attempt_reward=0.0, authority_ok=True,
                contradiction=False, support_found=True)
    base.update(over)
    return AttemptOutcome(**base)


def test_each_regime_detectable():
    assert "over_search" in detect_regimes(_outcome(correct=True, over_search=True))
    assert "stop_too_late" in detect_regimes(_outcome(correct=True, over_search=True))
    assert "stop_too_early" in detect_regimes(_outcome(false_stop=True, correct=False))
    assert "under_search" in detect_regimes(_outcome(under_search=True, correct=False))
    assert "stale_evidence" in detect_regimes(_outcome(stale_error=True))
    assert "route_miss" in detect_regimes(_outcome(correct=False, found_hit=False))
    assert "query_miss" in detect_regimes(_outcome(correct=False, found_hit=False))
    assert "evidence_sparse" in detect_regimes(
        _outcome(correct=False, found_hit=False, tool_calls=1))
    assert "answer_extraction_miss" in detect_regimes(
        _outcome(correct=False, found_hit=True))
    assert "support_answer_mismatch" in detect_regimes(
        _outcome(correct=True, evidence_score=0.2))
    assert "verification_miss" in detect_regimes(
        _outcome(correct=False, abstained=False, authority_ok=False))
    assert "contradiction_unresolved" in detect_regimes(
        _outcome(correct=False, contradiction=True))


def test_synthetic_run_exhibits_regime_variety(items, providers):
    agent = EpistemicAgent(AgentConfig(available_tools=TOOLS + ["page_fetch"]))
    split = build_split(items, confirm_fraction=0.4)
    _, con = partition(items, split)
    base = run_condition(con, agent, providers, PolicyMemory(BanditParams()),
                         condition="no_memory_search", budget=3).outcomes
    regimes = dominant_regimes(base)
    # The fixture deliberately produces residual failures across several regimes
    # so the diagnostics are meaningful (not cartoonishly perfect).
    assert len(regimes) >= 4
    assert "route_miss" in regimes
    assert regimes.keys() & {"stale_evidence", "verification_miss", "contradiction_unresolved"}
