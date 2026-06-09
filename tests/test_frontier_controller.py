"""Frontier CONTROLLER: shadow vs. active tool-selection (no model/provider calls)."""

from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from regimes_probe.agent.candidate_frontier import CandidateFrontier, StepPlan
from regimes_probe.agent.llm_task_frame import LLMTaskFrameParser, build_task_frame
from regimes_probe.agent.planner import AgentConfig, EpistemicAgent
from regimes_probe.datasets.base import Item
from regimes_probe.policy.contextual_bandit import BanditParams
from regimes_probe.policy.memory import PolicyMemory
from regimes_probe.policy.verification_policy import VerificationConfig
from regimes_probe.tools.base import SearchProvider, SearchResponse, SearchResult
from regimes_probe.tools.fake import FakePageFetch


def _slot(sid, name, role, **kw):
    return {"slot_id": sid, "slot_name": name, "slot_role": role, **kw}


def _con(cid, span, terms, applies, disc, **kw):
    return {"constraint_id": cid, "text_span": span, "normalized_terms": terms,
            "applies_to": applies, "discriminative_score": disc, "specificity_score": disc, **kw}


def _obs(title, snippet, url="https://x.example/p", auth=0.7):
    return SimpleNamespace(title=title, snippet=snippet, url=url, source_authority=auth,
                           failed=False, benchmark_contaminated=False)


TV_Q = "Which 90s TV series starred an actor born in Tennessee and an actor who was a Caribbean immigrant?"
TV = {
    "target_answer_slots": [_slot("series", "90s TV series", "title_or_work",
                                  is_target_answer_slot=True, is_intermediate_slot=False,
                                  depends_on=["a1", "a2"])],
    "latent_slots": [_slot("a1", "actor born in Tennessee", "person"),
                     _slot("a2", "actor who was a Caribbean immigrant", "person")],
    "constraints": [
        _con("c_scope", "90s TV series", ["90s", "tv", "series"], ["series"], 0.3,
             constraint_type="temporal", required=True, priority="high",
             testable_claim="x", how_to_test="read", semantic_label="temporal_work_scope"),
        _con("c_birth", "actor born in Tennessee", ["actor", "born", "Tennessee"], ["a1"], 2.0,
             required=True, priority="high", testable_claim="x", how_to_test="read",
             semantic_label="cast_member_birthplace_relation"),
        _con("c_imm", "actor who was a Caribbean immigrant", ["actor", "Caribbean", "immigrant"],
             ["a2"], 2.0, required=True, priority="high", testable_claim="x", how_to_test="read",
             semantic_label="cast_member_immigration_background")],
    "dependency_edges": [["a1", "series"], ["a2", "series"]],
    "known_context_terms": ["Tennessee", "Caribbean"]}


def _frame(payload, q):
    f, meta = build_task_frame("x", q, use_llm=True,
                               parser=LLMTaskFrameParser(model_fn=lambda _p: json.dumps(payload),
                                                         model="stub"))
    assert meta.parser_used == "llm", meta.fallback_reason
    return f


class _Provider(SearchProvider):
    name = "generic_web_search"
    cost_per_call = Decimal("0.002")
    deterministic = True

    def available(self): return True

    def search(self, q, *, limit=5, **o):
        return SearchResponse(provider=self.name, query=q, results=(
            SearchResult(title="Some series", url="https://gov.example/s",
                         snippet="A 90s TV series featuring an actor born in Tennessee.",
                         source_authority=0.6, rank=0, extra={"item_id": "x"}),),
            cost=self.cost_per_call)


def _agent(*, controller: bool, auto: bool = False, parser_payload=TV):
    return EpistemicAgent(AgentConfig(
        available_tools=["generic_web_search", "page_fetch"],
        enable_task_frame=True, enable_llm_task_frame_parser=True,
        force_task_frame=not auto, auto_epistemic_mode=auto,
        enable_frontier_controller=controller, verification=VerificationConfig(min_support=2)),
        task_frame_parser=LLMTaskFrameParser(model_fn=lambda _p: json.dumps(parser_payload),
                                             model="stub"))


def _run(agent, question=TV_Q, budget=3):
    item = Item(id="x", answer="Some Series", question=question)
    return agent.attempt(item, PolicyMemory(BanditParams()),
                         {"generic_web_search": _Provider(), "page_fetch": FakePageFetch([])},
                         budget=budget, explore=False, attempt_id="t")


# ----------------------------------------------------- unit: planning quality
def test_first_action_prefers_discriminative_over_generic_target_constraint():
    fr = CandidateFrontier(_frame(TV, TV_Q))
    sp = fr.propose_step_action(observations=[], budget_remaining=3, reading_tools=True)
    assert sp.kind == "search"
    assert sp.is_discriminative_constraint is True and sp.is_generic_query is False
    # the discriminative blocking constraint targeted is an actor/casting one, not c_scope.
    assert "c_scope" not in sp.constraint_ids
    assert any(c in sp.constraint_ids for c in ("c_birth", "c_imm"))


def test_does_not_start_with_generic_repeated_descriptor_query():
    fr = CandidateFrontier(_frame(TV, TV_Q))
    sp = fr.propose_step_action(observations=[], budget_remaining=3, reading_tools=True)
    q = sp.query.lower()
    assert q.count("90s tv series") < 2          # never "90s tv series" 90s tv series
    assert "tennessee" in q or "caribbean" in q  # uses a discriminative clue instead


def test_read_action_requires_candidate_slot_constraint_triple():
    fr = CandidateFrontier(_frame(TV, TV_Q))
    fr.ingest_evidence([_obs("Some series", "A 90s TV series with an actor born in Tennessee.")],
                       source_tool="generic_web_search")
    good = fr.frontier_read_value(_obs("series page", "the 90s TV series cast and history"))
    bad = fr.frontier_read_value(_obs("unrelated", "a recipe for tacos"))
    # whichever frontier_read_value selects must carry the full triple.
    if good.selected:
        assert good.candidate_id and good.target_slot_id and good.constraint_ids
    assert not bad.selected and bad.rejected_reason == "no_candidate_slot_constraint_affected"


# ----------------------------------------------------- shadow vs active
def test_shadow_mode_records_recommendation_without_driving():
    tr = _run(_agent(controller=False))
    cf = tr.candidate_frontier
    ctrl = cf.get("controller", {})
    assert ctrl.get("frontier_controller_used") is False
    assert ctrl.get("shadow_total", 0) >= 1                       # recommendations recorded
    # the planner's actual action is recorded alongside the frontier recommendation.
    assert any("frontier_recommended_action" in (c.task_action or {}) for c in tr.calls)
    # no call is attributed to a frontier action in shadow mode.
    assert all(c.frontier_action_id is None for c in tr.calls)


def test_active_mode_drives_tool_selection_and_links_calls():
    tr = _run(_agent(controller=True))
    ctrl = tr.candidate_frontier.get("controller", {})
    assert ctrl.get("frontier_controller_used") is True
    assert ctrl.get("tool_calls_from_frontier_actions", 0) >= 1
    # every executed tool call links to a frontier_action id.
    assert all(c.frontier_action_id for c in tr.calls)
    assert ctrl.get("first_action_discriminative") is True
    assert ctrl.get("first_query_generic") is False


def test_unexecutable_frontier_action_falls_back_to_old_planner(monkeypatch):
    # force every controller step to be unexecutable -> the old planner must still drive.
    monkeypatch.setattr(CandidateFrontier, "propose_step_action",
                        lambda self, **kw: StepPlan("fa_x", "verify_candidate_constraint",
                                                    "unexecutable", reason="forced_test"))
    tr = _run(_agent(controller=True))
    ctrl = tr.candidate_frontier.get("controller", {})
    assert ctrl.get("old_planner_fallback_count", 0) >= 1
    assert len(tr.calls) >= 1                                     # old planner still searched
    assert all(c.frontier_action_id is None for c in tr.calls)   # not controller-driven


def test_easy_direct_question_skips_frontier_controller():
    tr = _run(_agent(controller=True, auto=True),
              question="What is the capital of France?", budget=2)
    cf = tr.candidate_frontier
    assert cf.get("skipped") is True                             # no slates / controller
    assert tr.epistemic_mode["selected_epistemic_mode"] in (
        "direct_answer_possible", "simple_lookup")


# ----------------------------------------------------- replay + offline fork
def _project(tr):
    from regimes_probe.eval.debug import build_debug_record
    from regimes_probe.eval.grader import grade
    from regimes_probe.eval.projection import build_graph_projection
    from regimes_probe.eval.reward import RewardWeights, compute_rewards
    item = Item(id="x", answer="Some Series", question=TV_Q)
    rec = build_debug_record(item=item, trace=tr, grade=grade(item, tr.final_answer),
                             reward=compute_rewards(tr, correct=False, gold_norms=[],
                                                    weights=RewardWeights.full(),
                                                    freshness_sensitive=False),
                             condition="no_memory_search", budget=3)
    return build_graph_projection("rid", runs=[], debug_records=[rec], eligibility_verdict={})


def test_replay_reconstructs_same_frontier_projection():
    tr = _run(_agent(controller=True))
    p1, p2 = _project(tr), _project(tr)
    assert p1["objects"] == p2["objects"] and p1["relations"] == p2["relations"]
    types = {o["type"] for o in p1["objects"]}
    rels = {r["type"] for r in p1["relations"]}
    assert "frontier_action" in types
    assert "tool_call_from_frontier_action" in rels


def test_offline_fork_can_rescore_frontier_actions_without_providers():
    tr = _run(_agent(controller=True))
    actions = tr.candidate_frontier.get("frontier_actions", [])
    assert actions, "expected recorded frontier actions"

    # rescore purely from cached trace data (no providers/models).
    def _pick(acts, *, cost_weight):
        return max(acts, key=lambda a: a["expected_information_gain"]
                   - cost_weight * a["estimated_cost"])["action_id"]
    cheap = _pick(actions, cost_weight=0.5)
    pricey = _pick(actions, cost_weight=5.0)        # heavily penalize cost
    assert isinstance(cheap, str) and isinstance(pricey, str)
    # deterministic: same weighting -> same pick.
    assert _pick(actions, cost_weight=0.5) == cheap


# ----------------------------------------------------- guardrail + memory
def test_controller_requires_task_frame():
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    from _common import (validate_task_frame_flags, build_agent,
                         FRONTIER_CONTROLLER_REQUIRES_TASK_FRAME)
    with pytest.raises(ValueError) as exc:
        validate_task_frame_flags(task_frame=False, llm_parser=False, frontier_controller=True)
    assert FRONTIER_CONTROLLER_REQUIRES_TASK_FRAME in str(exc.value)
    with pytest.raises(ValueError):
        build_agent({"policy": {"enable_task_frame": False,
                                "enable_frontier_controller": True}}, ["generic_web_search"])


def test_policy_memory_answer_free_with_controller():
    from regimes_probe.datasets.synthetic import SyntheticBrowseAdapter
    from regimes_probe.eval.harness import experience_phase
    from regimes_probe.policy.policy_fragment import assert_no_answer_leakage
    from regimes_probe.tools.fake import build_fake_providers, load_corpus
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    items = SyntheticBrowseAdapter().load()[:6]
    providers = build_fake_providers(load_corpus(root / "fixtures" / "fake_search_corpus.json"),
                                     ["generic_web_search", "news_search"])
    agent = EpistemicAgent(AgentConfig(
        available_tools=["generic_web_search", "news_search", "page_fetch"],
        enable_task_frame=True, force_task_frame=True, enable_frontier_controller=True))
    mem = PolicyMemory(BanditParams())
    experience_phase(items, agent, providers, mem, budget=3, passes=2)
    assert_no_answer_leakage(mem.snapshot().to_dict(), "snapshot")
