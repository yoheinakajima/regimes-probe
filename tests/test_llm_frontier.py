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


# =========================================================================== #
# Proposal -> action -> evidence integrity (the bug-fix layer).               #
# =========================================================================== #

# Two intermediate slots + a target: the deterministic frontier opens on ONE slot,
# the LLM proposal targets the OTHER. The executed action must carry the PROPOSAL's
# slot/constraints, never the deterministic plan's (the reported bug).
TWO_Q = ("What Mexican restaurant is near a hotel that opened in 1955 "
         "and a museum founded in 2005?")
TWO = {
    "target_answer_slots": [_slot("T", "restaurant", "place", is_target_answer_slot=True,
                                  is_intermediate_slot=False, depends_on=["A", "B"])],
    "latent_slots": [_slot("A", "hotel nearby", "organization"),
                     _slot("B", "museum nearby", "organization")],
    "constraints": [
        _con("C1", "hotel opened 1955", ["hotel", "opened", "1955"], ["A"], 2.0,
             required=True, priority="high", testable_claim="x", how_to_test="read",
             semantic_label="hotel_open_year"),
        _con("C2", "museum founded 2005", ["museum", "founded", "2005"], ["B"], 2.0,
             required=True, priority="high", testable_claim="x", how_to_test="read",
             semantic_label="museum_found_year")],
    "dependency_edges": [["A", "T"], ["B", "T"]],
    "known_context_terms": ["New Mexico"]}


def _two_card(f, fr, det, **kw):
    base = dict(question=TWO_Q, epistemic_mode="task_frame_required",
                available_tools=["serper_search", "exa_search", "page_fetch"],
                remaining_budget=4, memory_access_mode="no_memory",
                failed_queries=[], no_progress_queries=[])
    base.update(kw)
    return build_research_state_card(f, fr, det_plan=det, **base)


def _museum_proposal(f, *, query="museum New Mexico founded 2005 near restaurant",
                     tool_family="serper_search", pid="p1"):
    sidB = next(s.slot_id for s in f.latent_slots
                if "museum" in (s.descriptor_text or s.slot_name))
    return sidB, json.dumps({"proposals": [{
        "proposal_id": pid, "action_type": "generate_candidates_for_slot",
        "target_slot_id": sidB, "constraint_ids": ["C2"], "proposed_query": query,
        "proposed_tool_family": tool_family, "anchors_used": ["New Mexico"],
        "confidence": 0.8}]})


def test_selected_proposal_slot_constraints_propagate_to_executed_action():
    f = _frame(TWO, TWO_Q)
    fr = CandidateFrontier(f)
    det = fr.propose_step_action(observations=[], budget_remaining=4, reading_tools=True)
    sidB, payload = _museum_proposal(f)
    assert det.target_slot_id != sidB, "test needs det + proposal to target different slots"
    prop = LLMFrontierProposer(model_fn=lambda _p: payload, cache=ParserCache(), model="stub")
    plan, meta = prop.plan_step(_two_card(f, fr, det), det, frame=f, frontier=fr, mode="planner")
    assert plan is not None
    # executed action carries the PROPOSAL's slot/constraints, not the det plan's.
    assert plan.target_slot_id == sidB and plan.constraint_ids == ["C2"]
    assert plan.frontier_action_id == "lfp_p1"
    assert meta["integrity_passed"] is True
    assert meta["proposal_slot_id"] == sidB and meta["executed_slot_id"] == sidB
    # and a first-class FrontierAction was registered carrying those fields.
    reg = next(a for a in fr.frontier_actions if a.action_id == "lfp_p1")
    assert reg.target_slot_id == sidB and reg.constraint_ids == ["C2"]


def test_tool_call_records_proposal_and_frontier_action_ids():
    agent, proposer = _agent(planner_mode=True)
    tr = _run(agent)
    driven = [c for c in tr.calls if c.llm_frontier_proposal_id]
    assert driven, "expected a call driven by an LLM proposal"
    c = driven[0]
    assert c.frontier_action_id == f"lfp_{c.llm_frontier_proposal_id}"
    assert c.task_action.get("target_slot_id") and c.task_action.get("tested_constraint_ids")
    assert c.task_action.get("anchors_used") is not None


def test_evidence_linked_to_selected_proposal_slot_and_constraints():
    f = _frame(TWO, TWO_Q)
    fr = CandidateFrontier(f)
    sidB = next(s.slot_id for s in f.latent_slots
                if "museum" in (s.descriptor_text or s.slot_name))
    fr.register_proposal_action(
        proposal_id="p1", action_type="generate_candidates_for_slot", target_slot_id=sidB,
        candidate_id=None, hypothesis_id=None, constraint_ids=["C2"],
        query="museum founded 2005", tool="serper_search", tool_family="search",
        anchors_used=["New Mexico"])
    obs = [SimpleNamespace(
        title="Georgia O'Keeffe Museum", url="https://gov.example/m",
        snippet="The Georgia O'Keeffe Museum is a museum founded 2005 in New Mexico.",
        source_authority=0.7, failed=False, benchmark_contaminated=False)]
    ev = fr.ingest_evidence(obs, source_tool="serper_search", action_id="lfp_p1",
                            directed_slot_id=sidB, directed_constraint_ids=["C2"],
                            proposal_id="p1")
    assert sidB in ev.supports_slot_ids
    assert "C2" in ev.supports_constraint_ids
    # a candidate was bound to the SELECTED museum slot, not some default slot.
    museum_cands = [c for c in fr.slates[sidB].candidates.values()]
    assert any("museum" in c.candidate_text.lower() for c in museum_cands)
    assert ev.progress_components["selected_slot_candidate_count"] >= 1
    assert ev.progress_components["selected_constraint_support_count"] >= 1
    assert any(e["event_type"] == "evidence_linked_to_llm_proposal" for e in fr.events)


def test_integrity_mismatch_triggers_error_and_fallback(monkeypatch):
    f = _frame(TWO, TWO_Q)
    fr = CandidateFrontier(f)
    det = fr.propose_step_action(observations=[], budget_remaining=4, reading_tools=True)
    sidB, payload = _museum_proposal(f)
    prop = LLMFrontierProposer(model_fn=lambda _p: payload, cache=ParserCache(), model="stub")

    # Force a faithless translation: the executed plan drops the proposal's slot.
    def _bad_plan(self, p, det_plan, **kw):
        return StepPlan("lfp_x", p.action_type, "search", query=p.proposed_query,
                        target_slot_id="WRONG", constraint_ids=["C1"])
    monkeypatch.setattr(LLMFrontierProposer, "_to_step_plan", _bad_plan)
    plan, meta = prop.plan_step(_two_card(f, fr, det), det, frame=f, frontier=fr, mode="planner")
    assert plan is None                                   # refused -> fall back to det
    assert meta["integrity_passed"] is False
    assert meta["fallback_reason"] == "frontier_action_integrity_error"
    assert prop.integrity_error_count == 1
    assert any(e["event_type"] == "frontier_action_integrity_error" for e in meta["events"])


# --------------------------------------------------------------------------- candidate promotion
def test_fake_person_result_promoted_to_correct_slot_with_evidence_links():
    # A person/graphic-designer slot with education + employment constraints; the LLM
    # proposal targets that slot. A LinkedIn-style result names the designer.
    Q = "Who is the graphic designer who designed the WHO malaria report cover?"
    payload = {
        "target_answer_slots": [_slot("D", "graphic designer", "graphic_designer",
                                      is_target_answer_slot=True, is_intermediate_slot=False)],
        "latent_slots": [],
        "constraints": [
            _con("C_edu", "Yale Publishing Course Leadership Strategies Book Publishing",
                 ["yale", "publishing", "leadership", "book"], ["D"], 2.0, required=True,
                 priority="high", testable_claim="x", how_to_test="read",
                 semantic_label="education_background"),
            _con("C_emp", "Malaria Consortium employment", ["malaria", "consortium"], ["D"],
                 2.0, required=True, priority="high", testable_claim="x", how_to_test="read",
                 semantic_label="employment_relation")],
        "dependency_edges": [], "known_context_terms": ["WHO", "malaria"]}
    f = _frame(payload, Q)
    fr = CandidateFrontier(f)
    sidD = f.target_answer_slots[0].slot_id
    fr.register_proposal_action(
        proposal_id="p1", action_type="generate_candidates_for_slot", target_slot_id=sidD,
        candidate_id=None, hypothesis_id=None, constraint_ids=["C_edu", "C_emp"],
        query="graphic designer Malaria Consortium Yale Publishing", tool="serper_search",
        tool_family="search", anchors_used=["malaria"])
    obs = [SimpleNamespace(
        title="Cristina Ortiz - Graphic Designer - LinkedIn",
        url="https://linkedin.example/in/cristina",
        snippet=("Graphique Malaria Consortium. Editorial Designer. Malaria Consortium. "
                 "Yale Publishing Course - Leadership Strategies in Book Publishing. 2013."),
        source_authority=0.5, failed=False, benchmark_contaminated=False)]
    ev = fr.ingest_evidence(obs, source_tool="serper_search", action_id="lfp_p1",
                            directed_slot_id=sidD, directed_constraint_ids=["C_edu", "C_emp"],
                            proposal_id="p1")
    cands = [c for c in fr.slates[sidD].candidates.values()]
    cristina = next((c for c in cands if "cristina ortiz" in c.candidate_text.lower()), None)
    assert cristina is not None, "Cristina Ortiz should be extracted + assigned to the slot"
    assert cristina.slot_id == sidD
    assert set(cristina.constraints_supported) >= {"C_edu", "C_emp"}
    assert cristina.evidence_score > 0
    # LinkedIn (platform noise) must NOT have become a candidate.
    assert not any(c.candidate_text.lower() == "linkedin" for c in cands)
    assert sidD in ev.supports_slot_ids


def test_progress_scoring_ignores_unrelated_and_noise_candidates():
    f = _frame(TWO, TWO_Q)
    fr = CandidateFrontier(f)
    sidB = next(s.slot_id for s in f.latent_slots
                if "museum" in (s.descriptor_text or s.slot_name))
    fr.register_proposal_action(
        proposal_id="p1", action_type="generate_candidates_for_slot", target_slot_id=sidB,
        candidate_id=None, hypothesis_id=None, constraint_ids=["C2"], query="museum 2005",
        tool="serper_search", tool_family="search", anchors_used=["New Mexico"])
    # A result with only noise + an unrelated capitalized phrase (no museum match).
    obs = [SimpleNamespace(
        title="Username Generator - LinkedIn", url="https://x.example/u",
        snippet="Merriam Webster Definition. Username Generator Table. Login Menu.",
        source_authority=0.3, failed=False, benchmark_contaminated=False)]
    ev = fr.ingest_evidence(obs, source_tool="serper_search", action_id="lfp_p1",
                            directed_slot_id=sidB, directed_constraint_ids=["C2"],
                            proposal_id="p1")
    pc = ev.progress_components
    assert pc["noise_candidate_count"] >= 1
    assert pc["selected_slot_candidate_count"] == 0     # nothing real bound to the slot
    assert pc["selected_constraint_support_count"] == 0


# --------------------------------------------------------------------------- repair triggers
def _gen_card(f, fr, det, **kw):
    base = dict(question=TWO_Q, epistemic_mode="task_frame_required",
                available_tools=["serper_search", "page_fetch"], remaining_budget=4,
                memory_access_mode="no_memory", failed_queries=[], no_progress_queries=[])
    base.update(kw)
    return build_research_state_card(f, fr, det_plan=det, **base)


def test_repair_triggers_on_repeated_zero_progress_query_even_if_not_generic():
    from regimes_probe.agent.llm_frontier import repair_trigger_reason
    f = _frame(TWO, TWO_Q)
    fr = CandidateFrontier(f)
    det = fr.propose_step_action(observations=[], budget_remaining=4, reading_tools=True)
    assert not _is_generic_query(det.query)              # a specific query...
    from regimes_probe.agent.llm_frontier import _norm_q
    no_prog = {_norm_q(det.query)}                       # ...but already tried w/ no progress
    assert repair_trigger_reason(det, f, fr, no_progress_norms=no_prog) == \
        "repeats_zero_progress_query"


def test_repair_triggers_on_stale_or_no_progress_candidate_query():
    from regimes_probe.agent.candidate_frontier import StepPlan
    from regimes_probe.agent.llm_frontier import repair_trigger_reason
    f = _frame(TWO, TWO_Q)
    fr = CandidateFrontier(f)
    sidB = next(s.slot_id for s in f.latent_slots
                if "museum" in (s.descriptor_text or s.slot_name))
    # seed a stale candidate on the museum slot and point a det plan at it.
    obs = [SimpleNamespace(title="Old Museum", url="https://x.example/o",
                           snippet="A museum in town.", source_authority=0.4,
                           failed=False, benchmark_contaminated=False)]
    fr.ingest_evidence(obs, source_tool="serper_search")
    cand = next(iter(fr.slates[sidB].candidates.values()))
    cand.no_progress_count = 1
    det = StepPlan("fa1", "verify_candidate_constraint", "search",
                   query='"Old Museum" founded year', target_slot_id=sidB,
                   candidate_id=cand.candidate_id, constraint_ids=["C2"])
    assert repair_trigger_reason(det, f, fr, no_progress_norms=set()) == \
        "uses_stale_or_no_progress_candidate"


# --------------------------------------------------------------------------- tool normalization
def test_enabled_concrete_tools_are_accepted():
    from regimes_probe.agent.llm_frontier import normalize_tool
    for t in ("serper_search", "exa_search", "firecrawl_search"):
        r = normalize_tool(t, "", ["serper_search", "exa_search", "firecrawl_search"])
        assert r["ok"] and r["tool"] == t and r["family"] == "search"
    # a concrete tool named in the family field is still recognized + accepted.
    r = normalize_tool("", "serper_search", ["serper_search"])
    assert r["ok"] and r["tool"] == "serper_search"
    # a non-enabled concrete tool is rejected with a clear reason.
    r = normalize_tool("exa_search", "", ["serper_search"])
    assert not r["ok"] and r["reason"] == "tool_not_enabled:exa_search"


def test_family_resolves_to_enabled_search_provider():
    from regimes_probe.agent.llm_frontier import normalize_tool
    r = normalize_tool("", "search", ["serper_search", "page_fetch"])
    assert r["ok"] and r["tool"] == "serper_search" and r["normalized"] is True
    r = normalize_tool("", "web_search", ["exa_search"])      # alias -> search
    assert r["ok"] and r["tool"] == "exa_search"
    r = normalize_tool("", "scrape", ["serper_search"])       # no scrape tool enabled
    assert not r["ok"] and r["reason"] == "no_enabled_tool_in_family:scrape"


def test_serper_proposal_accepted_end_to_end():
    f = _frame(TWO, TWO_Q)
    fr = CandidateFrontier(f)
    det = fr.propose_step_action(observations=[], budget_remaining=4, reading_tools=True)
    _, payload = _museum_proposal(f, tool_family="serper_search")
    prop = LLMFrontierProposer(model_fn=lambda _p: payload, cache=ParserCache(), model="stub")
    card = _two_card(f, fr, det, available_tools=["serper_search", "exa_search", "page_fetch"])
    plan, meta = prop.plan_step(card, det, frame=f, frontier=fr, mode="planner")
    assert plan is not None and plan.tool == "serper_search"
    p = meta["selected"]
    assert p["status"] == "accepted" and meta["normalized_tool"] == "serper_search"


# --------------------------------------------------------------------------- replay
def test_replay_preserves_proposal_action_evidence_links():
    agent, _ = _agent(planner_mode=True)
    tr = _run(agent)

    def _proj(trace):
        from regimes_probe.eval.debug import build_debug_record
        from regimes_probe.eval.grader import grade
        from regimes_probe.eval.projection import build_graph_projection
        from regimes_probe.eval.reward import RewardWeights, compute_rewards
        item = Item(id="x", answer="El Camino Cantina", question=REST_Q)
        rec = build_debug_record(item=item, trace=trace, grade=grade(item, trace.final_answer),
                                 reward=compute_rewards(trace, correct=False, gold_norms=[],
                                                        weights=RewardWeights.full(),
                                                        freshness_sensitive=False),
                                 condition="no_memory_search", budget=3)
        return build_graph_projection("rid", runs=[], debug_records=[rec], eligibility_verdict={})

    p1, p2 = _proj(tr), _proj(tr)
    assert p1["objects"] == p2["objects"] and p1["relations"] == p2["relations"]
    rels = {r["type"] for r in p1["relations"]}
    assert "tool_call_from_llm_frontier_proposal" in rels
    # selection nodes carry the integrity audit fields.
    sels = [o for o in p1["objects"] if o["type"] == "llm_frontier_selection"]
    assert sels and all("integrity_passed" in o["data"] for o in sels)
