"""Level 5d.1: unify evidence-interpretation candidate assertions with the canonical
SlotCandidate registry, attach constraint support from interpreted evidence (not raw
overlap), keep unknown-role observations out of slates, and let the verifier resolve an
extracted candidate by text. No live providers/models.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from regimes_probe.agent.candidate_frontier import CandidateFrontier
from regimes_probe.agent.llm_frontier import (
    _proposal_from_dict, _query_lacks_distinctive_anchor, repair_trigger_reason,
    validate_proposal)
from regimes_probe.agent.llm_task_frame import LLMTaskFrameParser, build_task_frame


def _slot(sid, name, role, **kw):
    return {"slot_id": sid, "slot_name": name, "slot_role": role, **kw}


def _con(cid, span, terms, applies, disc, **kw):
    return {"constraint_id": cid, "text_span": span, "normalized_terms": terms,
            "applies_to": applies, "discriminative_score": disc, "specificity_score": disc, **kw}


def _frame(payload, q):
    f, meta = build_task_frame("x", q, use_llm=True,
                               parser=LLMTaskFrameParser(model_fn=lambda _p: json.dumps(payload),
                                                         model="stub"))
    assert meta.parser_used == "llm", meta.fallback_reason
    return f


def _obs(title, snippet, url, *, auth=0.6, contam=False):
    return SimpleNamespace(title=title, snippet=snippet, url=url, source_authority=auth,
                           failed=False, benchmark_contaminated=contam)


def _ids(f):
    """Remapped slot ids by descriptor (the parser canonicalises P/R/H -> s0/s1/s2)."""
    person = f.target_answer_slots[0].slot_id
    rest = next(s.slot_id for s in f.latent_slots if "restaurant" in (s.slot_name or ""))
    hotel = next(s.slot_id for s in f.latent_slots if "hotel" in (s.slot_name or ""))
    return person, rest, hotel


# A person slot + a hotel(org) slot + a restaurant(place) slot, Santa Fe context.
MULTI_Q = ("Who founded the restaurant near the Pecos Trail Inn (opened 1955) in Santa Fe?")
MULTI = {
    "target_answer_slots": [_slot("P", "founder", "person", is_target_answer_slot=True,
                                  is_intermediate_slot=False, depends_on=["R"])],
    "latent_slots": [_slot("R", "restaurant", "place"),
                     _slot("H", "hotel nearby", "organization")],
    "constraints": [
        _con("c_open", "Pecos Trail Inn opened 1955", ["pecos", "trail", "inn", "opened", "1955"],
             ["H"], 2.0, required=True, priority="high", testable_claim="x", how_to_test="read",
             semantic_label="hotel_open_year"),
        _con("c_rest", "Mexican restaurant Santa Fe", ["mexican", "restaurant", "santa", "fe"],
             ["R"], 1.5, required=True, priority="high", testable_claim="x", how_to_test="read",
             semantic_label="cuisine_location"),
        _con("c_found", "founder of the restaurant", ["founder", "founded"], ["P"], 1.2,
             required=True, priority="high", testable_claim="x", how_to_test="read",
             semantic_label="founding_relation")],
    "dependency_edges": [["R", "P"]], "known_context_terms": ["Santa Fe"]}


# --------------------------------------------------------------------------- registry unify
def test_accepted_assertion_materializes_canonical_candidate():
    f = _frame(MULTI, MULTI_Q)
    fr = CandidateFrontier(f)
    person, _r, _h = _ids(f)
    fr.ingest_evidence([_obs("Maria Lopez - Restaurateur - LinkedIn",
                             "Maria Lopez founded the restaurant in Santa Fe.",
                             "https://linkedin.com/in/maria-lopez")],
                       source_tool="serper",
                       directed_slot_id=person, directed_constraint_ids=["c_found"], proposal_id="p1")
    interp = fr.interpretations[-1]
    acc = [a for a in interp.candidate_assertions if a.accepted
           and "maria lopez" in a.candidate_text.lower()]
    assert acc, "Maria Lopez should be accepted on the person slot"
    a = acc[0]
    assert a.assertion_id and a.canonical_candidate_ids        # linked to a canonical id
    assert person in a.canonical_candidate_ids                 # bound to the selected slot
    cid = a.canonical_candidate_ids[person]
    assert cid in fr.candidates_by_id
    assert fr.assertion_to_candidate.get(a.assertion_id) in a.canonical_candidate_ids.values()


def test_verifier_resolves_extracted_candidate_by_text():
    f = _frame(MULTI, MULTI_Q)
    fr = CandidateFrontier(f)
    person, _r, _h = _ids(f)
    fr.ingest_evidence([_obs("Maria Lopez - Restaurateur - LinkedIn",
                             "Maria Lopez founded the restaurant.",
                             "https://linkedin.com/in/maria-lopez")],
                       source_tool="serper", directed_slot_id=person,
                       directed_constraint_ids=["c_found"], proposal_id="p1")
    # resolve by exact id and by text.
    cid = next(c for c, v in fr.candidates_by_id.items() if "maria" in v.candidate_text.lower())
    assert fr.resolve_candidate(cid)["found"]
    dbg = fr.resolve_candidate("Maria Lopez", slot_id=person)
    assert dbg["found"] and dbg["candidate_id"] == cid


def test_candidate_extracted_then_verified_is_not_nonexistent():
    f = _frame(MULTI, MULTI_Q)
    fr = CandidateFrontier(f)
    person, _r, _h = _ids(f)
    fr.ingest_evidence([_obs("Maria Lopez - Restaurateur - LinkedIn",
                             "Maria Lopez founded the restaurant in Santa Fe.",
                             "https://linkedin.com/in/maria-lopez")],
                       source_tool="serper", directed_slot_id=person,
                       directed_constraint_ids=["c_found"], proposal_id="p1")
    p = _proposal_from_dict({"proposal_id": "v1", "action_type": "verify_candidate_constraint",
                             "target_slot_id": person, "candidate_id": "Maria Lopez",
                             "constraint_ids": ["c_found"],
                             "proposed_query": "Maria Lopez founded restaurant Santa Fe"}, 0)
    ok, reason = validate_proposal(p, f, fr, failed_norms=set(),
                                   available_tools=["serper_search"], remaining_budget=3)
    assert ok, reason
    assert p.candidate_id in fr.candidates_by_id and p.candidate_lookup_debug["found"]


# --------------------------------------------------------------------------- constraint support
def test_cristina_supports_designer_education_employment():
    Q = "Who is the graphic designer who designed the WHO malaria report cover?"
    payload = {
        "target_answer_slots": [_slot("D", "graphic designer", "person",
                                      is_target_answer_slot=True, is_intermediate_slot=False)],
        "latent_slots": [],
        "constraints": [
            _con("C_edu", "Yale Publishing Course", ["yale", "publishing", "leadership", "book"],
                 ["D"], 2.0, required=True, priority="high", testable_claim="x",
                 how_to_test="read", semantic_label="education"),
            _con("C_emp", "Malaria Consortium", ["malaria", "consortium"], ["D"], 2.0,
                 required=True, priority="high", testable_claim="x", how_to_test="read",
                 semantic_label="employment")],
        "dependency_edges": [], "known_context_terms": ["WHO", "malaria"]}
    f = _frame(payload, Q)
    fr = CandidateFrontier(f)
    fr.ingest_evidence([_obs(
        "Cristina Ortiz - Graphic Designer - LinkedIn",
        "Malaria Consortium. Yale Publishing Course - Leadership Strategies in Book Publishing.",
        "https://linkedin.com/in/cristina")],
        source_tool="serper", directed_slot_id=f.target_answer_slots[0].slot_id,
        directed_constraint_ids=["C_edu", "C_emp"], proposal_id="p1")
    cris = next(c for c in fr.candidates_by_id.values() if "cristina ortiz" in c.candidate_text.lower())
    assert set(cris.constraints_supported) >= {"C_edu", "C_emp"}


def test_corroborated_single_anchor_supports_constraint():
    # professional profile + ONE distinctive employer anchor -> support (not 2-term overlap).
    Q = "Who is the engineer trained at Caltech?"
    payload = {
        "target_answer_slots": [_slot("E", "engineer", "person", is_target_answer_slot=True,
                                      is_intermediate_slot=False)],
        "latent_slots": [],
        "constraints": [_con("c_edu", "studied at Caltech", ["caltech"], ["E"], 2.0,
                             required=True, priority="high", testable_claim="x",
                             how_to_test="read", semantic_label="education")],
        "dependency_edges": [], "known_context_terms": []}
    f = _frame(payload, Q)
    fr = CandidateFrontier(f)
    fr.ingest_evidence([_obs("Dana Wu - Engineer - LinkedIn",
                             "Dana Wu studied at Caltech.",
                             "https://linkedin.com/in/dana-wu")],
                       source_tool="serper", directed_slot_id=f.target_answer_slots[0].slot_id,
                       directed_constraint_ids=["c_edu"], proposal_id="p1")
    dana = next(c for c in fr.candidates_by_id.values() if "dana wu" in c.candidate_text.lower())
    assert "c_edu" in dana.constraints_supported


def test_hotel_year_supports_hotel_constraint_not_person_slot():
    f = _frame(MULTI, MULTI_Q)
    fr = CandidateFrontier(f)
    person, _r, hotel = _ids(f)
    fr.ingest_evidence([_obs("Pecos Trail Inn",
                             "The Pecos Trail Inn opened 1955 in Santa Fe.",
                             "https://x.example/inn")],
                       source_tool="serper", directed_slot_id=hotel,
                       directed_constraint_ids=["c_open"], proposal_id="p1")
    # bound to the hotel(org) slot with the opening constraint supported...
    inn_h = [c for c in fr.slates[hotel].candidates.values()
             if "pecos trail inn" in c.candidate_text.lower()]
    assert inn_h and "c_open" in inn_h[0].constraints_supported
    # ...and NOT bound to the person(founder) slot.
    assert not any("pecos trail inn" in c.candidate_text.lower()
                   for c in fr.slates[person].candidates.values())


def test_constraint_support_only_with_local_predicate_for_org():
    f = _frame(MULTI, MULTI_Q)
    fr = CandidateFrontier(f)
    _p, rest, _h = _ids(f)
    fr.ingest_evidence([_obs("Pecos Trail Cafe",
                             "Pecos Trail Cafe is a Mexican restaurant in Santa Fe.",
                             "https://x.example/cafe")],
                       source_tool="serper", directed_slot_id=rest,
                       directed_constraint_ids=["c_rest"], proposal_id="p1")
    cafe = [c for c in fr.slates[rest].candidates.values()
            if "pecos trail cafe" in c.candidate_text.lower()]
    assert cafe and "c_rest" in cafe[0].constraints_supported


# --------------------------------------------------------------------------- hygiene
def test_unknown_role_does_not_enter_all_slots():
    f = _frame(MULTI, MULTI_Q)
    fr = CandidateFrontier(f)
    # "Festervan" — a bare ambiguous token with no role predicate -> weak observation.
    fr.ingest_evidence([_obs("Festervan", "Festervan.", "https://x.example/p")],
                       source_tool="serper")
    assert all(len(s.candidates) == 0 for s in fr.slates.values())
    interp = fr.interpretations[-1]
    assert any(a.rejection_reason == "weak_observation_not_candidate"
               for a in interp.candidate_assertions)


def test_tv_shows_not_assigned_to_person_slot():
    f = _frame(MULTI, MULTI_Q)
    fr = CandidateFrontier(f)
    fr.ingest_evidence([_obs("TV Shows", "Browse TV Shows and series.", "https://x.example/tv")],
                       source_tool="serper")
    assert not any("tv shows" in c.candidate_text.lower()
                   for s in fr.slates.values() for c in s.candidates.values())


def test_weekday_not_assigned_to_person_or_org_slot():
    f = _frame(MULTI, MULTI_Q)
    fr = CandidateFrontier(f)
    fr.ingest_evidence([_obs("Tuesday", "Published on Tuesday.", "https://x.example/d")],
                       source_tool="serper")
    assert not any(c.candidate_text.lower() == "tuesday"
                   for s in fr.slates.values() for c in s.candidates.values())


def test_ev_slot_true_cons_false_records_explanation():
    f = _frame(MULTI, MULTI_Q)
    fr = CandidateFrontier(f)
    # a person with NO constraint anchor in the snippet -> slot-linked but not cons-linked.
    person, _r, _h = _ids(f)
    fr.ingest_evidence([_obs("Jordan Pal - Profile - LinkedIn",
                             "Jordan Pal is a professional.",
                             "https://linkedin.com/in/jordan-pal")],
                       source_tool="serper", directed_slot_id=person,
                       directed_constraint_ids=["c_found"], proposal_id="p1")
    assert any(e["event_type"] == "ev_slot_true_cons_false_explained" for e in fr.events)


# --------------------------------------------------------------------------- repair triggers
def test_repair_triggers_on_noise_term_query():
    from regimes_probe.agent.candidate_frontier import StepPlan
    f = _frame(MULTI, MULTI_Q)
    fr = CandidateFrontier(f)
    det = StepPlan("fa1", "generate_candidates_for_slot", "search",
                   query="Wikipedia founder Santa Fe", target_slot_id="P", constraint_ids=["c_found"])
    assert repair_trigger_reason(det, f, fr, no_progress_norms=set()) == "noise_term_in_query"


def test_repair_triggers_on_prompt_language_query():
    from regimes_probe.agent.candidate_frontier import StepPlan
    f = _frame(MULTI, MULTI_Q)
    fr = CandidateFrontier(f)
    # bag of common words, no proper noun / rare term / year / quote.
    assert _query_lacks_distinctive_anchor("actor born series passed away")
    det = StepPlan("fa1", "generate_candidates_for_slot", "search",
                   query="founder near the restaurant person", target_slot_id="P",
                   constraint_ids=["c_found"])
    reason = repair_trigger_reason(det, f, fr, no_progress_norms=set())
    assert reason in ("query_lacks_distinctive_anchor", "no_high_priority_constraint_anchor")


def test_policy_memory_answer_free_with_interpreter_unify():
    from regimes_probe.agent.planner import AgentConfig, EpistemicAgent
    from regimes_probe.datasets.synthetic import SyntheticBrowseAdapter
    from regimes_probe.eval.harness import experience_phase
    from regimes_probe.policy.contextual_bandit import BanditParams
    from regimes_probe.policy.memory import PolicyMemory
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
