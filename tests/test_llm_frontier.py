"""Level 5c LLM frontier proposer: the LLM proposes constraint-grounded actions;
deterministic code validates, scores, selects, executes, records.

No live providers, no live models. Every "model" is a local ``model_fn`` stub that
returns canned JSON; replay paths must make zero model calls.
"""

from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from regimes_probe.agent.candidate_frontier import CandidateFrontier, StepPlan
from regimes_probe.agent.llm_frontier import (
    LLMFrontierProposer, ResearchStateCard, _is_generic_query,
    build_research_state_card, validate_proposal)
from regimes_probe.agent.llm_task_frame import LLMTaskFrameParser, ParserCache, build_task_frame
from regimes_probe.agent.planner import AgentConfig, EpistemicAgent
from regimes_probe.datasets.base import Item
from regimes_probe.policy.contextual_bandit import BanditParams
from regimes_probe.policy.memory import PolicyMemory
from regimes_probe.tools.base import SearchProvider, SearchResponse, SearchResult
from regimes_probe.tools.fake import FakePageFetch


# --------------------------------------------------------------------------- fixtures
def _slot(sid, name, role, **kw):
    return {"slot_id": sid, "slot_name": name, "slot_role": role, **kw}


def _con(cid, span, terms, applies, disc, **kw):
    return {"constraint_id": cid, "text_span": span, "normalized_terms": terms,
            "applies_to": applies, "discriminative_score": disc, "specificity_score": disc, **kw}


# A frame whose deterministic query collapses to a bare generic descriptor
# ("restaurant"): the single constraint is a low-discriminative venue *type*, so the
# deterministic planner has nothing specific to compose with -> repair-mode bait.
REST_Q = "What Mexican restaurant in New Mexico is near a hotel and a museum?"
REST = {
    "target_answer_slots": [_slot("rest", "restaurant", "place",
                                  is_target_answer_slot=True, is_intermediate_slot=False)],
    "latent_slots": [],
    "constraints": [
        _con("c_type", "restaurant", ["restaurant"], ["rest"], 0.2,
             constraint_type="type", required=True, priority="high",
             testable_claim="x", how_to_test="read", semantic_label="venue_type")],
    "dependency_edges": [],
    "known_context_terms": ["New Mexico", "hotel", "museum"]}

# A frame with a genuinely discriminative clue -> deterministic query is NOT generic.
TV_Q = "Which 90s TV series starred an actor born in Tennessee?"
TV = {
    "target_answer_slots": [_slot("series", "90s TV series", "title_or_work",
                                  is_target_answer_slot=True, is_intermediate_slot=False,
                                  depends_on=["a1"])],
    "latent_slots": [_slot("a1", "actor born in Tennessee", "person")],
    "constraints": [
        _con("c_birth", "actor born in Tennessee", ["actor", "born", "Tennessee"], ["a1"], 2.0,
             required=True, priority="high", testable_claim="x", how_to_test="read",
             semantic_label="cast_member_birthplace_relation")],
    "dependency_edges": [["a1", "series"]],
    "known_context_terms": ["Tennessee"]}


def _frame(payload, q):
    f, meta = build_task_frame("x", q, use_llm=True,
                               parser=LLMTaskFrameParser(model_fn=lambda _p: json.dumps(payload),
                                                         model="stub"))
    assert meta.parser_used == "llm", meta.fallback_reason
    return f


def _card(frame, frontier, det_plan, **kw):
    base = dict(question=REST_Q, epistemic_mode="multi_constraint_research",
                available_tools=["generic_web_search", "page_fetch"], remaining_budget=3,
                memory_access_mode="no_memory", failed_queries=[], no_progress_queries=[])
    base.update(kw)
    return build_research_state_card(frame, frontier, det_plan=det_plan, **base)


# A canned LLM proposer response: one constraint-grounded restaurant query. The
# parser canonicalises slot ids, so derive the *real* target slot id from the frame.
def _grounded(frame, query="Mexican restaurant New Mexico near hotel museum",
              constraint_ids=("c_type",), pid="pr1",
              action_type="generate_candidates_for_slot"):
    sid = frame.target_answer_slots[0].slot_id
    return json.dumps({"proposals": [{
        "proposal_id": pid, "action_type": action_type,
        "target_slot_id": sid, "constraint_ids": list(constraint_ids),
        "proposed_query": query, "proposed_tool_family": "search",
        "expected_evidence": "restaurant name + its New Mexico location near a hotel/museum",
        "success_criteria": "a named restaurant candidate", "why_this_action": "bind the slot",
        "anchors_used": ["New Mexico", "hotel", "museum"], "avoids_generic_query": True,
        "confidence": 0.8}]})


def _grounded_fn(frame):
    payload = _grounded(frame)
    return lambda _prompt: payload


# The REST payload always canonicalises to the same target slot id, so one stub
# model_fn serves every REST-based scenario (unit + agent integration).
_model_fn = _grounded_fn(_frame(REST, REST_Q))


# --------------------------------------------------------------------------- unit: repair
def test_repair_triggers_on_generic_deterministic_query_and_grounds_it():
    f = _frame(REST, REST_Q)
    fr = CandidateFrontier(f)
    det = fr.propose_step_action(observations=[], budget_remaining=3, reading_tools=True)
    # the deterministic query is a bare generic descriptor — exactly the bottleneck.
    assert _is_generic_query(det.query) and det.query.lower() == "restaurant"

    prop = LLMFrontierProposer(model_fn=_model_fn, cache=ParserCache(), model="stub")
    plan, meta = prop.plan_step(_card(f, fr, det), det, frame=f, frontier=fr, mode="repair")
    assert meta["repaired_generic"] is True and meta["model_called"] is True
    assert plan is not None and plan.kind == "search"
    q = plan.query.lower()
    assert not _is_generic_query(q)
    # uses the known-context anchors, not the bare descriptor.
    assert "new mexico" in q or "hotel" in q or "museum" in q
    assert plan.is_generic_query is False


def test_repair_skipped_when_deterministic_query_is_specific():
    f = _frame(TV, TV_Q)
    fr = CandidateFrontier(f)
    det = fr.propose_step_action(observations=[], budget_remaining=3, reading_tools=True)
    assert det.is_discriminative_constraint and not _is_generic_query(det.query)

    prop = LLMFrontierProposer(model_fn=_model_fn, cache=ParserCache(), model="stub")
    plan, meta = prop.plan_step(_card(f, fr, det, question=TV_Q), det,
                                frame=f, frontier=fr, mode="repair")
    assert plan is None
    assert meta["skipped_reason"] == "deterministic_query_ok"
    assert meta["model_called"] is False and prop.model_calls == 0


# --------------------------------------------------------------------------- unit: validation
def test_generic_llm_proposal_is_rejected():
    f = _frame(REST, REST_Q)
    fr = CandidateFrontier(f)
    sid = f.target_answer_slots[0].slot_id
    raw = json.dumps({"proposals": [{
        "proposal_id": "g1", "action_type": "generate_candidates_for_slot",
        "target_slot_id": sid, "constraint_ids": ["c_type"],
        "proposed_query": "restaurant", "proposed_tool_family": "search"}]})
    prop = LLMFrontierProposer(model_fn=lambda _p: raw, cache=ParserCache(), model="stub")
    plan, meta = prop.plan_step(_card(f, fr, None), None, frame=f, frontier=fr, mode="planner")
    assert plan is None
    statuses = {p["proposal_id"]: p for p in meta["proposals"]}
    assert statuses["g1"]["status"] == "rejected"
    assert statuses["g1"]["rejection_reason"] == "generic_query"


def test_proposal_referencing_nonexistent_slot_is_rejected():
    f = _frame(REST, REST_Q)
    fr = CandidateFrontier(f)
    raw = json.dumps({"proposals": [{
        "proposal_id": "n1", "action_type": "generate_candidates_for_slot",
        "target_slot_id": "ghost_slot", "constraint_ids": ["c_type"],
        "proposed_query": "Mexican restaurant New Mexico hotel"}]})
    prop = LLMFrontierProposer(model_fn=lambda _p: raw, cache=ParserCache(), model="stub")
    plan, meta = prop.plan_step(_card(f, fr, None), None, frame=f, frontier=fr, mode="planner")
    assert plan is None
    p = meta["proposals"][0]
    assert p["status"] == "rejected" and p["rejection_reason"] == "nonexistent_slot:ghost_slot"


def test_duplicate_no_progress_query_is_rejected():
    f = _frame(REST, REST_Q)
    fr = CandidateFrontier(f)
    dup = "Mexican restaurant New Mexico near hotel museum"
    prop = LLMFrontierProposer(model_fn=_model_fn, cache=ParserCache(), model="stub")
    plan, meta = prop.plan_step(_card(f, fr, None, failed_queries=[dup]), None,
                                frame=f, frontier=fr, mode="planner", failed_queries=[dup])
    assert plan is None
    p = meta["proposals"][0]
    assert p["status"] == "rejected"
    assert p["rejection_reason"] == "duplicate_no_progress_query"


def test_disallowed_tool_family_is_rejected():
    f = _frame(REST, REST_Q)
    fr = CandidateFrontier(f)
    sid = f.target_answer_slots[0].slot_id
    raw = json.dumps({"proposals": [{
        "proposal_id": "t1", "action_type": "generate_candidates_for_slot",
        "target_slot_id": sid, "constraint_ids": ["c_type"],
        "proposed_query": "Mexican restaurant New Mexico hotel",
        "proposed_tool_family": "shell_exec"}]})
    ok, reason = validate_proposal(
        next(iter(_proposals(raw))), f, fr, failed_norms=set(),
        available_tools=["generic_web_search"], remaining_budget=3)
    assert not ok and reason == "disallowed_tool:shell_exec"


def _proposals(raw):
    from regimes_probe.agent.llm_frontier import _extract_proposals, _proposal_from_dict
    return [_proposal_from_dict(d, i) for i, d in enumerate(_extract_proposals(raw))]


# --------------------------------------------------------------------------- unit: replay/cache
def test_replay_uses_cached_proposal_and_makes_no_model_call():
    f = _frame(REST, REST_Q)
    fr = CandidateFrontier(f)
    det = fr.propose_step_action(observations=[], budget_remaining=3, reading_tools=True)
    cache = ParserCache()
    # arm: a live-ish proposer populates the cache with one model call.
    armed = LLMFrontierProposer(model_fn=_model_fn, cache=cache, model="stub")
    plan1, meta1 = armed.plan_step(_card(f, fr, det), det, frame=f, frontier=fr, mode="repair")
    assert plan1 is not None and meta1["model_called"] and armed.model_calls == 1

    # replay: no model_fn, replay_only -> must hit cache, zero model calls.
    replay = LLMFrontierProposer(model_fn=None, cache=cache, model="stub", replay_only=True)
    plan2, meta2 = replay.plan_step(_card(f, fr, det), det, frame=f, frontier=fr, mode="repair")
    assert plan2 is not None and meta2["cache_hit"] is True
    assert replay.model_calls == 0 and replay.cache_hits == 1
    assert plan1.query == plan2.query


def test_dry_run_without_cache_falls_back_deterministically():
    f = _frame(REST, REST_Q)
    fr = CandidateFrontier(f)
    det = fr.propose_step_action(observations=[], budget_remaining=3, reading_tools=True)
    replay = LLMFrontierProposer(model_fn=None, cache=ParserCache(), model="stub", replay_only=True)
    plan, meta = replay.plan_step(_card(f, fr, det), det, frame=f, frontier=fr, mode="repair")
    assert plan is None                                  # falls back to deterministic plan
    assert meta["fallback_reason"] == "cache_miss_in_replay"
    assert replay.model_calls == 0


# --------------------------------------------------------------------------- integration: agent
class _Provider(SearchProvider):
    name = "generic_web_search"
    cost_per_call = Decimal("0.002")
    deterministic = True

    def available(self): return True

    def search(self, q, *, limit=5, **o):
        return SearchResponse(provider=self.name, query=q, results=(
            SearchResult(title="El Camino Cantina", url="https://gov.example/r",
                         snippet="A Mexican restaurant in New Mexico near a hotel and museum.",
                         source_authority=0.6, rank=0, extra={"item_id": "x"}),),
            cost=self.cost_per_call)


def _agent(*, planner_mode: bool, controller=True, auto=False, payload=REST,
           model_fn=_model_fn, replay_only=False, cache=None):
    proposer = LLMFrontierProposer(model_fn=model_fn, cache=cache or ParserCache(),
                                   model="stub", replay_only=replay_only)
    cfg = AgentConfig(
        available_tools=["generic_web_search", "page_fetch"],
        enable_task_frame=True, enable_llm_task_frame_parser=True,
        force_task_frame=not auto, auto_epistemic_mode=auto,
        enable_frontier_controller=controller,
        enable_llm_frontier_planner=planner_mode,
        enable_llm_frontier_repair=not planner_mode)
    return EpistemicAgent(
        cfg, task_frame_parser=LLMTaskFrameParser(model_fn=lambda _p: json.dumps(payload),
                                                  model="stub"),
        llm_frontier=proposer), proposer


def _run(agent, question=REST_Q, budget=3):
    item = Item(id="x", answer="El Camino Cantina", question=question)
    return agent.attempt(item, PolicyMemory(BanditParams()),
                         {"generic_web_search": _Provider(), "page_fetch": FakePageFetch([])},
                         budget=budget, explore=False, attempt_id="t")


def test_planner_drives_constraint_grounded_query_and_links_calls():
    agent, proposer = _agent(planner_mode=True)
    tr = _run(agent)
    lf = tr.llm_frontier
    assert lf.get("enabled") is True and lf.get("mode") == "planner"
    # the LLM-proposed, constraint-grounded query actually drove a tool call.
    grounded = [c for c in tr.calls if c.frontier_action_id
                and c.frontier_action_id.startswith("lfp_")]
    assert grounded, "expected an LLM-frontier-driven call"
    q = grounded[0].query.lower()
    assert not _is_generic_query(q)
    assert "new mexico" in q or "hotel" in q or "museum" in q
    assert proposer.selected_count >= 1 and proposer.accepted_count >= 1


def test_accepted_proposal_links_to_frontier_action_in_projection():
    agent, _ = _agent(planner_mode=True)
    tr = _run(agent)
    from regimes_probe.eval.debug import build_debug_record
    from regimes_probe.eval.grader import grade
    from regimes_probe.eval.projection import build_graph_projection
    from regimes_probe.eval.reward import RewardWeights, compute_rewards
    item = Item(id="x", answer="El Camino Cantina", question=REST_Q)
    rec = build_debug_record(item=item, trace=tr, grade=grade(item, tr.final_answer),
                             reward=compute_rewards(tr, correct=False, gold_norms=[],
                                                    weights=RewardWeights.full(),
                                                    freshness_sensitive=False),
                             condition="no_memory_search", budget=3)
    proj = build_graph_projection("rid", runs=[], debug_records=[rec], eligibility_verdict={})
    types = {o["type"] for o in proj["objects"]}
    rels = {r["type"] for r in proj["relations"]}
    assert "llm_frontier_proposal" in types
    assert "research_state_card" in types
    assert "tool_call_from_llm_frontier_proposal" in rels


def test_repair_mode_only_invokes_model_on_generic_query():
    agent, proposer = _agent(planner_mode=False)   # repair mode
    tr = _run(agent)
    lf = tr.llm_frontier
    assert lf.get("enabled") is True and lf.get("mode") == "repair"
    # at least one step's deterministic query was generic and got repaired.
    assert proposer.repair_invoked_count >= 1


def test_easy_direct_question_skips_llm_frontier_planner():
    agent, proposer = _agent(planner_mode=True, auto=True)
    tr = _run(agent, question="What is the capital of France?", budget=2)
    lf = tr.llm_frontier
    assert lf.get("enabled") is False                # no frame/frontier -> proposer never runs
    assert proposer.model_calls == 0
    assert tr.epistemic_mode["selected_epistemic_mode"] in (
        "direct_answer_possible", "simple_lookup")


def test_policy_memory_answer_free_with_llm_frontier():
    from regimes_probe.datasets.synthetic import SyntheticBrowseAdapter
    from regimes_probe.eval.harness import experience_phase
    from regimes_probe.policy.policy_fragment import assert_no_answer_leakage
    from regimes_probe.tools.fake import build_fake_providers, load_corpus
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    items = SyntheticBrowseAdapter().load()[:6]
    providers = build_fake_providers(load_corpus(root / "fixtures" / "fake_search_corpus.json"),
                                     ["generic_web_search", "news_search"])
    proposer = LLMFrontierProposer(model_fn=_model_fn, cache=ParserCache(), model="stub")
    agent = EpistemicAgent(AgentConfig(
        available_tools=["generic_web_search", "news_search", "page_fetch"],
        enable_task_frame=True, force_task_frame=True, enable_frontier_controller=True,
        enable_llm_frontier_planner=True), llm_frontier=proposer)
    mem = PolicyMemory(BanditParams())
    experience_phase(items, agent, providers, mem, budget=3, passes=2)
    assert_no_answer_leakage(mem.snapshot().to_dict(), "snapshot")


# --------------------------------------------------------------------------- guardrails
def test_llm_frontier_requires_task_frame():
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    from _common import (validate_task_frame_flags, build_agent,
                         LLM_FRONTIER_REQUIRES_TASK_FRAME)
    with pytest.raises(ValueError) as exc:
        validate_task_frame_flags(task_frame=False, llm_parser=False, llm_frontier=True)
    assert LLM_FRONTIER_REQUIRES_TASK_FRAME in str(exc.value)
    with pytest.raises(ValueError):
        build_agent({"policy": {"enable_task_frame": False,
                                "enable_llm_frontier_planner": True}}, ["generic_web_search"])


def test_same_conditions_detects_mismatched_llm_frontier_settings():
    from regimes_probe.eval.conditions import ConditionSpec, same_conditions
    base_kw = dict(answer_model="m", answer_prompt_version="v1", enabled_tools=("a",),
                   tool_budget=3, split_id="s", grader="g", provider_config_id="pc",
                   query_policy="q", verify_policy="ve", stop_policy="st")
    baseline = ConditionSpec(**base_kw, memory_access="none",
                             llm_frontier_settings="planner|stub|abc")
    treatment = ConditionSpec(**base_kw, memory_access="frozen_snapshot",
                              llm_frontier_settings="off")
    res = same_conditions(baseline, treatment)        # intended diff: memory_access only
    assert res.ok is False
    assert "llm_frontier_settings" in res.unexpected_diffs

    # identical frontier settings -> only the intended memory_access differs.
    treatment_ok = ConditionSpec(**base_kw, memory_access="frozen_snapshot",
                                 llm_frontier_settings="planner|stub|abc")
    res2 = same_conditions(baseline, treatment_ok)
    assert res2.ok is True and res2.unexpected_diffs == []
