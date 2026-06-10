"""Level 5i offline tests: source-subject extraction (I), pre-judge triage + explicit-location
filter + batch judging (K), debug-slice runner (N), and coreference safety scaffolding (J).

No live providers/models — local stubs only. Synthetic diagnostic names; no benchmark logic.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from regimes_probe.agent.candidate_frontier import CandidateFrontier
from regimes_probe.agent.coreference import coreference_metrics, propose_coreference
from regimes_probe.agent.evidence_interpreter import EvidenceInterpreter
from regimes_probe.agent.evidence_judge import EvidenceJudge, enforce_hard_rules, EvidenceJudgment
from regimes_probe.agent.llm_task_frame import LLMTaskFrameParser, ParserCache, build_task_frame
from regimes_probe.agent.source_subject import (
    SourceSubject, extract_source_subject, is_promotable_subject, subject_supports_candidate)
from regimes_probe.eval.debug_slice import (
    is_debug_slice_report, select_debug_slice)


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


def _obs(title, snippet, url, *, auth=0.7, contam=False):
    return SimpleNamespace(title=title, snippet=snippet, url=url, source_authority=auth,
                           failed=False, benchmark_contaminated=contam)


def _sid(f, name=None):
    return (f.target_answer_slots[0].slot_id if name is None
            else next(s.slot_id for s in f.all_slots if s.slot_name == name))


PERSON_Q = "Who is the engineer who studied at Caltech in 1990?"
PERSON = {
    "target_answer_slots": [_slot("D", "engineer", "person", is_target_answer_slot=True,
                                  is_intermediate_slot=False)],
    "latent_slots": [],
    "constraints": [_con("c1", "studied at Caltech 1990", ["caltech", "1990"], ["D"], 2.0,
                         required=True, priority="high", blocks_answer_if_unresolved=True,
                         testable_claim="x", how_to_test="read", semantic_label="education")],
    "dependency_edges": [], "known_context_terms": []}

# an explicit-location task: a restaurant in New Mexico (an org target + location facet).
NM_Q = "What restaurant in New Mexico opened in 1985?"
NM = {
    "target_answer_slots": [_slot("R", "restaurant", "organization", is_target_answer_slot=True,
                                  is_intermediate_slot=False)],
    "latent_slots": [],
    "constraints": [_con("cr", "restaurant in New Mexico opened 1985",
                         ["restaurant", "1985"], ["R"], 1.4, required=True, priority="high",
                         blocks_answer_if_unresolved=True, testable_claim="x",
                         how_to_test="read", semantic_label="venue")],
    "dependency_edges": [], "known_context_terms": ["New Mexico"]}


# ----------------------------------------------------------------- I: source-subject
def test_source_subject_identifies_predicate_grounded_person():
    ss = extract_source_subject(
        title="Ken Walker", snippet="Ken Walker is a Kenyan novelist who was born in 1948.",
        url="https://news.example/ken", source_role="article")
    assert ss.subject_name.lower().startswith("ken walker")
    assert ss.subject_type == "person" and ss.is_predicate_grounded
    assert ss.subject_role in ("article_subject", "mentioned_entity")
    assert is_promotable_subject(ss)


def test_source_title_only_subject_rejected_before_promotion():
    # a source-title / generic-topic page (a definition headword) has no real-world subject.
    ss = extract_source_subject(title="Founder Definition & Meaning",
                                snippet="Founder: a person who founds an institution.",
                                url="https://dictionary.example/founder",
                                source_role="generic_definition_page")
    assert ss.is_chrome_or_source_title_only and ss.subject_role == "generic_topic"
    assert not is_promotable_subject(ss)
    assert not subject_supports_candidate(ss, "Founder", "person")


def test_generic_definition_source_does_not_promote_headword():
    f = _frame(PERSON, PERSON_Q)
    fr = CandidateFrontier(f)
    fr.ingest_evidence([_obs("Engineer Definition & Meaning",
                             "Engineer: a person who designs machines.",
                             "https://dictionary.example/engineer")],
                       source_tool="serper", directed_slot_id=_sid(f),
                       directed_constraint_ids=["c1"], proposal_id="p")
    assert all(not c.constraints_supported and c.status != "confirmed"
               for c in fr.candidates_by_id.values())
    m = fr.interpreter.stats()
    assert m["candidate_promoted_from_source_title_only_count"] == 0
    assert m["candidate_promoted_from_chrome_count"] == 0


def test_work_page_not_promoted_for_incompatible_real_world_role():
    ss = extract_source_subject(title="The Founder (2016)",
                                snippet="The Founder is a 2016 biographical drama film.",
                                url="https://imdb.example/title", source_role="database_record")
    assert ss.subject_type == "title_or_work"
    # a film/work subject must not bind a real-world PERSON target slot.
    assert not subject_supports_candidate(ss, "The Founder", "person")


def test_predicate_grounded_org_subject_promotable_for_org_slot():
    ss = extract_source_subject(
        title="Pecos Trail Inn — About Us",
        snippet="Pecos Trail Inn is a historic hotel that opened in 1955.",
        url="https://pecostrailinn.example/about", source_role="official_page")
    assert ss.subject_type == "organization" and ss.is_predicate_grounded
    assert subject_supports_candidate(ss, "Pecos Trail Inn", "organization")


def test_relation_support_requires_object_anchor_in_quote():
    j = EvidenceJudgment(judgment="full_support", quote="Ada Lin founded a lab")
    out = enforce_hard_rules(j, contaminated=False, source_role="article", has_quote=True,
                             candidate_text="Ada Lin", aliases=[], slot_role="person",
                             relational=True, object_anchored=False)
    assert out.judgment == "partial_support"      # object not anchored -> not full relation


# ----------------------------------------------------------------- K1: pre-judge triage
def test_prejudge_triage_blocks_chrome_before_judge():
    f = _frame(PERSON, PERSON_Q)
    judge = EvidenceJudge(model_fn=lambda _p: json.dumps({"judgment": "full_support"}),
                          cache=ParserCache(), model="stub", enabled=True)
    fr = CandidateFrontier(f, interpreter=EvidenceInterpreter(judge=judge))
    fr.ingest_evidence([_obs("Login", "Sign in / Subscribe / Datasets / Trending",
                             "https://x.example/login")],
                       source_tool="serper", directed_slot_id=_sid(f),
                       directed_constraint_ids=["c1"], proposal_id="p")
    assert judge.calls == 0
    s = fr.interpreter.stats()
    assert s["judge_invoked_on_obvious_chrome_count"] == 0


def test_prejudge_triage_records_calls_saved():
    f = _frame(NM, NM_Q)
    judge = EvidenceJudge(model_fn=lambda _p: json.dumps({"judgment": "full_support"}),
                          cache=ParserCache(), model="stub", enabled=True)
    fr = CandidateFrontier(f, interpreter=EvidenceInterpreter(judge=judge))
    # an org clearly in a DIFFERENT state -> rejected pre-judge, saving the judge call.
    fr.ingest_evidence([_obs("Casa Roja Cafe", "Casa Roja Cafe is a restaurant located in Texas.",
                             "https://x.example/casa")],
                       source_tool="serper", directed_slot_id=_sid(f),
                       directed_constraint_ids=["cr"], proposal_id="p")
    s = fr.interpreter.stats()
    assert s["prejudge_rejected_count"] >= 1
    assert s["judge_calls_saved_by_prejudge_triage"] >= 1


# ----------------------------------------------------------------- K2: explicit location
def test_explicit_location_mismatch_rejected_before_judge():
    f = _frame(NM, NM_Q)
    fr = CandidateFrontier(f)
    fr.ingest_evidence([_obs("Casa Roja Cafe", "Casa Roja Cafe is a restaurant located in Texas.",
                             "https://x.example/casa")],
                       source_tool="serper", directed_slot_id=_sid(f),
                       directed_constraint_ids=["cr"], proposal_id="p")
    rej = [a for i in fr.interpretations for a in i.candidate_assertions
           if a.rejection_reason == "explicit_location_mismatch"]
    assert rej
    s = fr.interpreter.stats()
    assert s["explicit_location_mismatch_rejected_count"] >= 1
    assert s["explicit_location_mismatch_promoted_count"] == 0


def test_explicit_location_ambiguous_kept_marked_needs_support():
    f = _frame(NM, NM_Q)
    fr = CandidateFrontier(f)
    fr.ingest_evidence([_obs("Casa Verde Cafe", "Casa Verde Cafe is a well-reviewed restaurant.",
                             "https://x.example/cv")],
                       source_tool="serper", directed_slot_id=_sid(f),
                       directed_constraint_ids=["cr"], proposal_id="p")
    kept = [a for i in fr.interpretations for a in i.candidate_assertions
            if a.needs_location_support]
    assert kept and any(a.accepted for a in kept)
    assert fr.interpreter.stats()["explicit_location_ambiguous_kept_count"] >= 1


def test_explicit_location_supported_allows_promotion():
    f = _frame(NM, NM_Q)
    fr = CandidateFrontier(f)
    fr.ingest_evidence([_obs("Casa Verde Cafe", "Casa Verde Cafe is a restaurant in New Mexico.",
                             "https://x.example/cv")],
                       source_tool="serper", directed_slot_id=_sid(f),
                       directed_constraint_ids=["cr"], proposal_id="p")
    assert fr.interpreter.stats()["explicit_location_supported_count"] >= 1
    assert any("casa verde" in c.candidate_text.lower() for c in fr.candidates_by_id.values())


# ----------------------------------------------------------------- K3: batch judging
def _con_obj(cid, span, terms, applies=("R",)):
    return SimpleNamespace(constraint_id=cid, text_span=span, normalized_terms=list(terms),
                           testable_claim="x", how_to_test="read", applies_to=list(applies),
                           blocks_answer_if_unresolved=True, required=True, priority="high")


def test_batch_judge_matches_per_constraint_behavior():
    cons = [_con_obj("a", "opened in 1985", ["opened", "1985"]),
            _con_obj("b", "in New Mexico", ["new", "mexico"])]
    judge = EvidenceJudge(cache=ParserCache(), model="deterministic", enabled=False,
                          batch_enabled=True)
    det = {"a": ("supports", "Casa Verde opened in 1985"), "b": ("irrelevant", "")}
    out = judge.judge_batch(
        candidate_text="Casa Verde", candidate_id=None, aliases=[], slot_id="R",
        slot_role="organization", slot_descriptor="restaurant", constraints=cons,
        source_id="s1", source_title="t", source_url="https://x", source_domain="x",
        source_role="article", contaminated=False, snippet="Casa Verde opened in 1985.",
        det_by_cid=det)
    # deterministic batch mirrors the per-constraint mapping: supports->full, irrelevant->irr.
    assert out["a"].judgment == "full_support" and out["b"].judgment == "irrelevant"
    assert judge.stats()["per_constraint_judge_calls_avoided_count"] == 1


def test_batch_judge_preserves_contamination_hard_rule():
    cons = [_con_obj("a", "opened in 1985", ["opened", "1985"])]
    judge = EvidenceJudge(cache=ParserCache(), model="deterministic", enabled=False,
                          batch_enabled=True)
    out = judge.judge_batch(
        candidate_text="Casa Verde", candidate_id=None, aliases=[], slot_id="R",
        slot_role="organization", slot_descriptor="restaurant", constraints=cons,
        source_id="s1", source_title="t", source_url="https://x", source_domain="x",
        source_role="benchmark_contaminated", contaminated=True,
        snippet="mirror", det_by_cid={"a": ("supports", "Casa Verde opened in 1985")})
    assert out["a"].judgment not in ("full_support", "partial_support")   # contaminated


def test_batch_blocking_contradiction_early_stops_support():
    f = _frame(NM, NM_Q)

    def model_fn(prompt: str) -> str:
        # contradiction on the (blocking) constraint, regardless of the other.
        return json.dumps({"verdicts": {"cr": {"judgment": "contradiction",
                                               "quote": "closed in 1970"}}})
    judge = EvidenceJudge(model_fn=model_fn, cache=ParserCache(), model="stub", enabled=True,
                          batch_enabled=True)
    fr = CandidateFrontier(f, interpreter=EvidenceInterpreter(judge=judge))
    fr.ingest_evidence([_obs("Casa Verde", "Casa Verde is a restaurant in New Mexico.",
                             "https://x.example/cv")],
                       source_tool="serper", directed_slot_id=_sid(f),
                       directed_constraint_ids=["cr"], proposal_id="p")
    # the blocking constraint is contradicted -> the candidate is not supported/confirmed.
    assert all("cr" not in c.constraints_supported for c in fr.candidates_by_id.values())


def test_batch_judging_records_per_constraint_replayable_events():
    f = _frame(NM, NM_Q)
    judge = EvidenceJudge(
        model_fn=lambda _p: json.dumps({"verdicts": {"cr": {"judgment": "full_support",
                                                            "quote": "Casa Verde opened 1985"}}}),
        cache=ParserCache(), model="stub", enabled=True, batch_enabled=True)
    fr = CandidateFrontier(f, interpreter=EvidenceInterpreter(judge=judge))
    fr.ingest_evidence([_obs("Casa Verde", "Casa Verde, a restaurant in New Mexico, opened 1985.",
                             "https://x.example/cv")],
                       source_tool="serper", directed_slot_id=_sid(f),
                       directed_constraint_ids=["cr"], proposal_id="p")
    created = [e for e in fr.events if e["event_type"] == "llm_evidence_judgment.created"]
    assert any(e.get("data", {}).get("constraint_id") == "cr" for e in created)
    assert fr.interpreter.stats()["batch_judge_calls_count"] >= 1


# ----------------------------------------------------------------- N: debug slice
def test_debug_slice_selection_is_deterministic_and_records_ids():
    pool = [f"q{i:03d}" for i in range(40)]
    a = select_debug_slice(pool, n=12, seed=7, slice_id="mech", method="seeded_sample",
                           dataset="synthetic_browse")
    b = select_debug_slice(pool, n=12, seed=7, slice_id="mech", method="seeded_sample",
                           dataset="synthetic_browse")
    assert a.item_ids == b.item_ids and len(a.item_ids) == 12
    assert a.headline_eligible is False and a.kind == "debug_dev_slice"
    # a pinned slice records exactly the requested ids, order-preserving.
    pinned = select_debug_slice(pool, method="explicit_ids", explicit_ids=["q005", "q001"])
    assert pinned.item_ids == ["q005", "q001"]


def test_claims_refused_for_debug_slice_report():
    report = {"meta": {"debug_slice": {"kind": "debug_dev_slice", "headline_eligible": False}},
              "headline_eligible": True}
    assert is_debug_slice_report(report) is True
    # a normal report is not a debug slice.
    assert is_debug_slice_report({"meta": {}, "headline_eligible": True}) is False


# ----------------------------------------------------------------- J: coreference safety
CO_Q = "A novelist wrote a debut in 1990; this author later won a prize. Who is the author?"
CO = {
    "target_answer_slots": [_slot("A", "the author", "person", is_target_answer_slot=True,
                                  is_intermediate_slot=False)],
    "latent_slots": [_slot("P1", "this author who wrote a debut", "person"),
                     _slot("P2", "this author who won a prize", "person")],
    "constraints": [_con("c1", "wrote a debut in 1990", ["debut", "1990"], ["P1"], 1.2,
                         testable_claim="x", how_to_test="read", semantic_label="work"),
                    _con("c2", "won a prize", ["prize"], ["P2"], 1.0,
                         testable_claim="x", how_to_test="read", semantic_label="award")],
    "dependency_edges": [], "known_context_terms": []}


def test_coreference_proposes_same_referent_but_does_not_auto_merge():
    f = _frame(CO, CO_Q)
    props = propose_coreference(f)            # default: no auto-merge
    person_merges = [p for p in props if p.merge_status == "proposed_only"]
    assert person_merges and person_merges[0].shared_head == "author"
    m = coreference_metrics(props)
    assert m["coreference_applied_count"] == 0
    assert m["invalid_coreference_collapse_count"] == 0


def test_coreference_blocks_distinct_entity_slots():
    DIST = {
        "target_answer_slots": [_slot("Y", "birth year", "date_or_time",
                                      is_target_answer_slot=True, is_intermediate_slot=False)],
        "latent_slots": [_slot("R", "restaurant", "organization"),
                         _slot("H", "hotel", "organization"),
                         _slot("M", "museum", "organization"),
                         _slot("F", "founder", "person")],
        "constraints": [_con("c1", "restaurant", ["restaurant"], ["R"], 0.5,
                             testable_claim="x", how_to_test="read", semantic_label="venue")],
        "dependency_edges": [], "known_context_terms": []}
    f = _frame(DIST, "near a hotel and a museum, who founded the restaurant, and their birth year?")
    props = propose_coreference(f)
    m = coreference_metrics(props)
    # no distinct anchor entities (restaurant/hotel/museum/founder/birth-year) get merged.
    assert m["coreference_applied_count"] == 0 and m["invalid_coreference_collapse_count"] == 0


def test_coreference_blocks_subject_target_merge():
    SUBJ = {
        "target_answer_slots": [_slot("Y", "birth year of the founder", "date_or_time",
                                      is_target_answer_slot=True, is_intermediate_slot=False)],
        "latent_slots": [_slot("F", "the founder of the lab", "person")],
        "constraints": [_con("cf", "founder of the Quanta Lab", ["quanta", "founder"], ["F"],
                             1.4, testable_claim="x", how_to_test="read", semantic_label="role")],
        "dependency_edges": [], "known_context_terms": []}
    f = _frame(SUBJ, "In what year was the founder of the lab born?")
    props = propose_coreference(f)
    m = coreference_metrics(props)
    assert m["target_subject_merge_blocked_count"] >= 1
    assert m["invalid_coreference_collapse_count"] == 0


# ----------------------------------------------------------------- 5h invariants still green
def test_5h_pinned_zero_invariants_remain_green():
    f = _frame(PERSON, PERSON_Q)
    fr = CandidateFrontier(f)
    fr.ingest_evidence([_obs("Dana Wu", "Dana Wu is an engineer who studied at Caltech in 1990.",
                             "https://caltech.edu/people/dana-wu")],
                       source_tool="serper", directed_slot_id=_sid(f),
                       directed_constraint_ids=["c1"], proposal_id="p")
    m = fr.metrics()
    for k in ("target_answer_slot_filled_with_subject_count",
              "generic_single_token_seed_executed_count",
              "generic_definition_source_read_count",
              "premature_founder_search_before_place_supported_count",
              "judge_reused_truncated_excerpt_after_full_read_count",
              "debug_confirmed_label_when_answer_gate_false_count"):
        assert m[k] == 0
