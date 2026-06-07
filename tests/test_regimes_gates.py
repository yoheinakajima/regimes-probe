"""Regimes loop: gates, bounded mutations, and promotion events."""

from __future__ import annotations

from collections import Counter

from regimes_probe.activegraph_pack import EventLog, Events
from regimes_probe.regimes.action_space import PolicyConfigBundle, safe_mutations
from regimes_probe.regimes.gates import confirm_gate, optimize_gate
from regimes_probe.regimes.hypothesize import apply_update, propose_update
from regimes_probe.regimes.runner import run_regimes_loop


def test_optimize_gate_requires_min_delta():
    assert optimize_gate(0.50, 0.52, min_delta=0.01).passed
    assert not optimize_gate(0.50, 0.505, min_delta=0.01).passed


def test_confirm_gate_blocks_regression():
    assert confirm_gate(0.50, 0.50, max_regression=0.0).passed
    assert not confirm_gate(0.50, 0.49, max_regression=0.0).passed
    assert confirm_gate(0.50, 0.495, max_regression=0.01).passed


def test_mutations_are_bounded():
    cat = safe_mutations()
    bundle = PolicyConfigBundle()
    # huge delta clamps to the upper bound, not beyond
    m = cat["stop.stop_threshold"](10.0, "test")
    out, rec = m.apply(bundle)
    assert out.stop.stop_threshold <= 0.95
    assert rec["new"] <= 0.95


def test_propose_maps_regimes_to_mutations():
    regimes = Counter({"over_search": 5, "stale_evidence": 2})
    proposal = propose_update(regimes)
    targets = {m.target for m in proposal.mutations}
    assert "stop.stop_threshold" in targets
    # all proposed mutations are in the safe tier in v0
    assert all(m.tier == "safe" for m in proposal.mutations)


def test_apply_update_respects_bounds():
    proposal = propose_update(Counter({"stale_evidence": 3}))
    new_bundle, records = apply_update(PolicyConfigBundle(), proposal)
    assert new_bundle.reward.freshness <= 1.5
    assert records  # something changed


def test_run_loop_emits_promotion_event(items, providers):
    tools = ["generic_web_search", "news_search", "official_domain_search", "brave_search", "page_fetch"]
    log = EventLog(run_id="regimes")
    res = run_regimes_loop(items, providers, tools, passes=2, gate_budget=3, log=log)
    types = {e.type for e in log.events}
    assert Events.REGIME_DETECTED in types
    assert Events.POLICY_UPDATE_PROPOSED in types
    assert (Events.PROMOTION_ACCEPTED in types) or (Events.PROMOTION_REJECTED in types)
    # decision is internally consistent
    p = res.promotion
    assert p.accepted == (p.optimize_gate.passed and p.confirm_gate.passed)
