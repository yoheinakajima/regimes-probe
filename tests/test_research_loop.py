"""Level 5e research-loop: read scheduling/forced-read, discriminative-first reasons,
candidate-registry alias resolution + nonexistent breakdown, and support-consistency
(selected_constraint_support_count <-> evidence record, with explicit drop events).

No live providers/models — deterministic interpreter + frontier only.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from regimes_probe.agent.candidate_frontier import CandidateFrontier
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


# A person slot with two constraints (education + employment) — body-only evidence.
PERSON_Q = "Who is the engineer who studied at Caltech in 1990 and worked at Bell Labs?"
PERSON = {
    "target_answer_slots": [_slot("D", "engineer", "person", is_target_answer_slot=True,
                                  is_intermediate_slot=False)],
    "latent_slots": [],
    "constraints": [
        _con("c_edu", "studied at Caltech 1990", ["caltech", "studied", "1990"], ["D"], 2.0,
             required=True, priority="high", testable_claim="x", how_to_test="read",
             semantic_label="education"),
        _con("c_emp", "worked at Bell Labs", ["bell", "labs", "worked"], ["D"], 2.0,
             required=True, priority="high", testable_claim="x", how_to_test="read",
             semantic_label="employment")],
    "dependency_edges": [], "known_context_terms": []}


def _sid(f):
    return f.target_answer_slots[0].slot_id


# --------------------------------------------------------------------------- read scheduling
def test_forced_read_after_repeated_no_support_search():
    f = _frame(PERSON, PERSON_Q)
    fr = CandidateFrontier(f)
    sid = _sid(f)
    # a profile candidate whose SNIPPET gives no constraint support.
    o = _obs("Dana Wu - Engineer - LinkedIn", "Dana Wu is an engineer.",
             "https://linkedin.com/in/dana-wu")
    fr.ingest_evidence([o], source_tool="serper", directed_slot_id=sid,
                       directed_constraint_ids=["c_edu", "c_emp"], proposal_id="p1")
    fr.ingest_evidence([o], source_tool="serper", directed_slot_id=sid,
                       directed_constraint_ids=["c_edu", "c_emp"], proposal_id="p2")
    assert fr._slot_no_support_streak.get(sid) >= fr.force_read_after_n
    fr.generate_frontier_actions()        # propose_step_action regenerates with current streak
    sel = fr.select_frontier_action()
    assert sel.action_type == "read_candidate_source"
    assert sel.selected_reason == "forced_read_after_no_support"
    assert fr.metrics()["forced_read_after_no_support_count"] >= 1


def test_one_read_supports_multiple_constraints():
    f = _frame(PERSON, PERSON_Q)
    fr = CandidateFrontier(f)
    sid = _sid(f)
    # a READ (read_depth>=1) of the profile page body supports BOTH constraints at once.
    page = _obs("Dana Wu - Engineer",
                "Dana Wu studied at Caltech (1990) and worked at Bell Labs.",
                "https://linkedin.com/in/dana-wu")
    ev = fr.ingest_evidence([page], source_tool="firecrawl_scrape", read_depth=2,
                            directed_slot_id=sid, directed_constraint_ids=["c_edu", "c_emp"],
                            proposal_id="p1")
    dana = next(c for c in fr.candidates_by_id.values() if "dana wu" in c.candidate_text.lower())
    assert {"c_edu", "c_emp"} <= set(dana.constraints_supported)
    assert ev.progress_components["selected_constraint_support_count"] >= 2
    assert fr.metrics()["support_from_read_count"] >= 2
    assert fr.metrics()["read_executed_count"] >= 1


def test_support_consistency_no_silent_drop():
    f = _frame(PERSON, PERSON_Q)
    fr = CandidateFrontier(f)
    sid = _sid(f)
    ev = fr.ingest_evidence([_obs("Dana Wu - Engineer - LinkedIn",
                                  "Dana Wu studied at Caltech 1990.",
                                  "https://linkedin.com/in/dana-wu")],
                            source_tool="serper", directed_slot_id=sid,
                            directed_constraint_ids=["c_edu"], proposal_id="p1")
    # the selected-constraint support gain MUST appear on the evidence record (no drop).
    assert ev.progress_components["selected_constraint_support_count"] >= 1
    assert "c_edu" in ev.supports_constraint_ids
    assert fr.metrics()["support_dropped_count"] == 0


# --------------------------------------------------------------------------- registry breakdown
def test_nonexistent_candidate_breakdown():
    f = _frame(PERSON, PERSON_Q)
    fr = CandidateFrontier(f)
    sid = _sid(f)
    # a real candidate on the person slot
    fr.ingest_evidence([_obs("Dana Wu - Engineer - LinkedIn", "Dana Wu studied at Caltech 1990.",
                             "https://linkedin.com/in/dana-wu")],
                       source_tool="serper", directed_slot_id=sid,
                       directed_constraint_ids=["c_edu"], proposal_id="p1")
    # a weak observation (unknown role, no predicate)
    fr.ingest_evidence([_obs("Festervan", "Festervan.", "https://x.example/p")],
                       source_tool="serper")
    assert fr.resolve_candidate("Dana Wu", slot_id=sid)["breakdown"] in (
        "exact_normalized", "normalized_alias_found")
    assert fr.resolve_candidate("Festervan", slot_id=sid)["breakdown"] == \
        "exists_as_weak_observation"
    assert fr.resolve_candidate("Nobody At All", slot_id=sid)["breakdown"] == "truly_nonexistent"


def test_resolve_candidate_other_slot_breakdown():
    # two person-ish slots; a candidate on one resolves from the other as exists_in_other_slot.
    payload = {
        "target_answer_slots": [_slot("T", "the paper", "title_or_work", is_target_answer_slot=True,
                                      is_intermediate_slot=False, depends_on=["A"])],
        "latent_slots": [_slot("A", "co-author", "person")],
        "constraints": [_con("c1", "co-author of paper", ["coauthor", "paper"], ["A"], 1.5,
                             required=True, priority="high", testable_claim="x",
                             how_to_test="read", semantic_label="authorship")],
        "dependency_edges": [["A", "T"]], "known_context_terms": []}
    f = _frame(payload, "Which paper has a co-author who wrote it?")
    fr = CandidateFrontier(f)
    coa = next(s.slot_id for s in f.latent_slots)
    tgt = f.target_answer_slots[0].slot_id
    fr.ingest_evidence([_obs("Jee Park - Researcher - LinkedIn", "Jee Park co-author of paper.",
                             "https://linkedin.com/in/jee-park")],
                       source_tool="serper", directed_slot_id=coa,
                       directed_constraint_ids=["c1"], proposal_id="p1")
    dbg = fr.resolve_candidate("Jee Park", slot_id=tgt)
    assert dbg["found"] and dbg["breakdown"] == "exists_in_other_slot" and dbg["other_slot"] == coa


# --------------------------------------------------------------------------- discriminative-first
def test_discriminative_reason_recorded():
    f = _frame(PERSON, PERSON_Q)
    fr = CandidateFrontier(f)
    sp = fr.propose_step_action(observations=[], budget_remaining=4, reading_tools=True)
    assert sp.kind == "search"
    # the chosen constraint has a year + a proper-noun anchor -> discriminative reasons.
    assert sp.chosen_constraint_specificity > 1.0
    assert ("numeric_or_date_anchor" in sp.discriminative_reason
            or "named_entity_anchor" in sp.discriminative_reason
            or "high_specificity" in sp.discriminative_reason)
    info = sp.action_info()
    assert "discriminative_reason" in info and "chosen_constraint_specificity" in info


def test_streak_resets_on_support():
    f = _frame(PERSON, PERSON_Q)
    fr = CandidateFrontier(f)
    sid = _sid(f)
    fr.ingest_evidence([_obs("Dana Wu - Engineer - LinkedIn", "Dana Wu is an engineer.",
                             "https://linkedin.com/in/dana-wu")],
                       source_tool="serper", directed_slot_id=sid,
                       directed_constraint_ids=["c_edu"], proposal_id="p1")
    assert fr._slot_no_support_streak.get(sid) == 1
    # a supporting result resets the streak.
    fr.ingest_evidence([_obs("Dana Wu - Engineer - LinkedIn", "Dana Wu studied at Caltech 1990.",
                             "https://linkedin.com/in/dana-wu")],
                       source_tool="serper", directed_slot_id=sid,
                       directed_constraint_ids=["c_edu"], proposal_id="p2")
    assert fr._slot_no_support_streak.get(sid) == 0


# --------------------------------------------------------------------------- verifier resolution
def test_verifier_resolves_alias_and_breakdown():
    from regimes_probe.agent.llm_frontier import _proposal_from_dict, validate_proposal
    f = _frame(PERSON, PERSON_Q)
    fr = CandidateFrontier(f)
    sid = _sid(f)
    fr.ingest_evidence([_obs("Dana Wu - Engineer - LinkedIn", "Dana Wu studied at Caltech 1990.",
                             "https://linkedin.com/in/dana-wu")],
                       source_tool="serper", directed_slot_id=sid,
                       directed_constraint_ids=["c_edu"], proposal_id="p1")
    p = _proposal_from_dict({"proposal_id": "v1", "action_type": "verify_candidate_constraint",
                             "target_slot_id": sid, "candidate_id": "dana wu",
                             "constraint_ids": ["c_edu"],
                             "proposed_query": "Dana Wu Caltech 1990"}, 0)
    ok, _ = validate_proposal(p, f, fr, failed_norms=set(),
                              available_tools=["serper_search"], remaining_budget=3)
    assert ok and p.candidate_id in fr.candidates_by_id
    assert p.candidate_lookup_debug.get("breakdown") in (
        "exact_normalized", "normalized_alias_found")
