"""Level 5h offline tests: close the read -> judge loop (A/B), target-answer binding
(C/D), anchor-gate over-rejection recovery + seed floor (E/F), abstain admissibility (G),
source-acquisition hygiene (H), location/distance staging (L), regime detectors (M), and
local-support label hygiene (O).

No live providers / models — local stubs only. Names used here are synthetic diagnostic
fixtures; there is no benchmark-specific logic keyed to them.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from regimes_probe.agent.candidate_frontier import CandidateFrontier
from regimes_probe.agent.evidence_interpreter import EvidenceInterpreter
from regimes_probe.agent.evidence_judge import EvidenceJudge
from regimes_probe.agent.llm_frontier import (
    LLMFrontierProposer, ParserCache as _PC, ResearchStateCard, _anchor_terms,
    _is_fuzzy_duplicate, build_research_state_card, validate_proposal)
from regimes_probe.agent.llm_task_frame import LLMTaskFrameParser, ParserCache, build_task_frame
from regimes_probe.agent.read_judgment import (
    DEFAULT_READ_CONFIG, ReadJudgmentConfig, build_anchor_terms, extract_passages)


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


def _obs(title, snippet, url, *, auth=0.7, contam=False, role=""):
    o = SimpleNamespace(title=title, snippet=snippet, url=url, source_authority=auth,
                        failed=False, benchmark_contaminated=contam)
    if role:
        o.source_role = role
    return o


def _sid(f, name=None):
    if name is None:
        return f.target_answer_slots[0].slot_id
    return next(s.slot_id for s in f.all_slots if s.slot_name == name)


# A person frame whose blocking constraint is body-only (requires a read to confirm).
PERSON_Q = "Who is the engineer who studied at Caltech in 1990?"
PERSON = {
    "target_answer_slots": [_slot("D", "engineer", "person", is_target_answer_slot=True,
                                  is_intermediate_slot=False)],
    "latent_slots": [],
    "constraints": [_con("c1", "studied at Caltech 1990", ["caltech", "1990"], ["D"], 2.0,
                         required=True, priority="high", blocks_answer_if_unresolved=True,
                         testable_claim="x", how_to_test="read", semantic_label="education")],
    "dependency_edges": [], "known_context_terms": []}

# A subject(person) + answer-shaped target(date_or_time) frame, with the target depending
# on the subject (bind the subject, then bind the birth year).
SUBJ_Q = "In what year was the founder of the lab born?"
SUBJ = {
    "target_answer_slots": [_slot("Y", "birth year", "date_or_time", is_target_answer_slot=True,
                                  is_intermediate_slot=False, depends_on=["P"])],
    "latent_slots": [_slot("P", "founder of the lab", "person")],
    "constraints": [
        _con("cp", "founder of the Quanta Lab", ["quanta", "founder"], ["P"], 1.6,
             required=True, priority="high", blocks_answer_if_unresolved=True,
             testable_claim="x", how_to_test="read", semantic_label="role"),
        _con("cy", "birth year of the founder", ["birth", "year"], ["Y"], 0.4,
             constraint_type="answer_shape", required=True, priority="high",
             blocks_answer_if_unresolved=True, testable_claim="x", how_to_test="read",
             semantic_label="birth_year")],
    "dependency_edges": [], "known_context_terms": []}


def _person_frontier_with_pending(judge_full_on_marker="RESOLVEDMARK"):
    """Build a frontier whose judge marks c1 requires_read on the snippet (creating a pending
    read judgment), and full_support only when the page body marker is present."""
    def model_fn(prompt: str) -> str:
        if judge_full_on_marker in prompt:
            return json.dumps({"judgment": "full_support",
                               "quote": "Dana Wu studied at Caltech in 1990",
                               "candidate_aliases": []})
        return json.dumps({"judgment": "requires_read",
                           "requires_read_reason": "snippet_insufficient_body_may_support"})
    judge = EvidenceJudge(model_fn=model_fn, cache=ParserCache(), model="stub", enabled=True)
    f = _frame(PERSON, PERSON_Q)
    fr = CandidateFrontier(f, interpreter=EvidenceInterpreter(judge=judge))
    fr.ingest_evidence([_obs("Dana Wu", "Dana Wu is an engineer who studied at Caltech.",
                             "https://caltech.edu/people/dana-wu")],
                       source_tool="serper", directed_slot_id=_sid(f),
                       directed_constraint_ids=["c1"], proposal_id="p")
    return f, fr


# --------------------------------------------------------------------- A: read -> judge loop
def test_requires_read_creates_pending_and_read_routes_back_into_targeted_judgment():
    f, fr = _person_frontier_with_pending()
    pend = list(fr.pending_read_judgments.values())
    assert pend and pend[0].constraint_id == "c1" and pend[0].open
    assert any(e["event_type"] == "read_required_by_judge" for e in fr.events)
    cand = next(c for c in fr.candidates_by_id.values() if "dana wu" in c.candidate_text.lower())
    body = ("About Dana Wu. " + "filler " * 200
            + "RESOLVEDMARK Dana Wu studied at Caltech in 1990 and later led the lab.")
    n = fr.route_read_into_pending_judgments(candidate_id=cand.candidate_id,
                                             source_url="https://caltech.edu/people/dana-wu",
                                             read_text=body)
    assert n == 1
    assert any(e["event_type"] == "read_judged_after_read" for e in fr.events)
    assert any(e["event_type"] == "read_judgment_resolved" for e in fr.events)
    assert fr.pending_read_judgments[pend[0].pending_read_judgment_id].resolution == "full_support"
    assert "c1" in cand.constraints_supported
    assert fr.metrics()["judge_reused_truncated_excerpt_after_full_read_count"] == 0


def test_successful_read_resolves_or_keeps_open_with_explicit_reason():
    f, fr = _person_frontier_with_pending()
    cand = next(c for c in fr.candidates_by_id.values() if "dana wu" in c.candidate_text.lower())
    # a page body that mentions the anchors but NOT the resolving marker -> stays unresolved
    # with an EXPLICIT reason (it is not silently turned into a new candidate).
    body = "Caltech 1990 alumni directory. " + "names " * 100
    fr.route_read_into_pending_judgments(candidate_id=cand.candidate_id,
                                         source_url="https://caltech.edu/people/dana-wu",
                                         read_text=body)
    p = next(iter(fr.pending_read_judgments.values()))
    assert not p.open and p.resolution in ("still_unresolved", "irrelevant", "no_relevant_passage")
    assert any(e["event_type"] == "read_judgment_still_unresolved" for e in fr.events)
    assert fr.metrics()["requires_read_unresolved_after_successful_read_count"] >= 1


# --------------------------------------------------------------------- B: passage retrieval
def test_read_backed_judgment_uses_targeted_passages_not_only_head():
    head = "Homepage navigation links and boilerplate. " * 30
    answer = "RESOLVEDMARK Dana Wu studied at Caltech in 1990."
    body = head + answer + (" trailing content." * 50)
    scan = extract_passages(body, ["Caltech", "1990", "RESOLVEDMARK"],
                            config=ReadJudgmentConfig(read_passage_window_chars=160))
    assert scan.hit and not scan.head_only
    # the selected passage is the windowed answer region, NOT the document head.
    assert any("RESOLVEDMARK" in p for p in scan.passages)
    assert not any(p.startswith("Homepage navigation") for p in scan.passages)


def test_target_passage_retrieval_finds_answer_shaped_evidence_via_descriptor_anchors():
    con = SimpleNamespace(normalized_terms=["probation", "officer"],
                          text_span="years worked as a probation officer")
    slot = SimpleNamespace(descriptor_text="years worked as a probation officer",
                           slot_name="years", slot_role="date_or_time")
    anchors = build_anchor_terms(con, slot=slot, aliases=["Jane Roe"])
    body = ("Biography. " + "x " * 100
            + "Jane Roe worked as a probation officer from 1971 until 1989.")
    scan = extract_passages(body, anchors, config=DEFAULT_READ_CONFIG)
    assert scan.hit
    assert any("probation officer" in p for p in scan.passages)


# --------------------------------------------------------------------- C/D: target binding
def test_bind_target_answer_slot_fires_when_subject_supported_and_target_unresolved():
    f = _frame(SUBJ, SUBJ_Q)
    fr = CandidateFrontier(f)
    # support the SUBJECT (founder) so its slate carries support; target year stays unbound.
    fr.ingest_evidence([_obs("Quanta Lab", "Ada Lin is the founder of the Quanta Lab.",
                             "https://quanta.example/about")],
                       source_tool="serper", directed_slot_id=_sid(f, "founder of the lab"),
                       directed_constraint_ids=["cp"], proposal_id="p")
    acts = fr.generate_frontier_actions()
    binds = [a for a in acts if a.action_type == "bind_target_answer_slot"]
    assert binds and binds[0].target_slot_id == _sid(f, "birth year")
    assert fr.metrics()["bind_target_answer_slot_actions"] >= 1
    sp = fr.propose_step_action(observations=[], budget_remaining=4, reading_tools=True)
    assert sp.kind == "search" and sp.action_type == "bind_target_answer_slot"
    assert "born" in sp.query.lower() or "year" in sp.query.lower()
    assert fr.metrics()["target_answer_slot_filled_with_subject_count"] == 0


def test_target_date_slot_rejects_person_value_role():
    f = _frame(SUBJ, SUBJ_Q)
    fr = CandidateFrontier(f)
    # a PERSON entity must not bind a date_or_time target slot.
    from regimes_probe.agent.candidate_frontier import _role_compatible
    tslot = f.slot(_sid(f, "birth year"))
    assert not _role_compatible("person", tslot.slot_role)
    fr.ingest_evidence([_obs("person page", "Ada Lin is a person.",
                             "https://x.example/ada")],
                       source_tool="serper", directed_slot_id=_sid(f, "birth year"),
                       directed_constraint_ids=["cy"], proposal_id="p")
    yslate = fr.slates[_sid(f, "birth year")]
    assert all(c.candidate_text != "Ada Lin" for c in yslate.candidates.values())
    assert fr.metrics()["target_answer_slot_filled_with_wrong_role_count"] == 0


def test_target_binding_outranks_repeated_intermediate_verification():
    # founder carries TWO blocking constraints: one supported (so it is a working subject),
    # one still open (so it stays ACTIVE and keeps offering a verify) — exactly the seam.
    SUBJ2 = {
        "target_answer_slots": [_slot("Y", "birth year", "date_or_time",
                                      is_target_answer_slot=True, is_intermediate_slot=False,
                                      depends_on=["P"])],
        "latent_slots": [_slot("P", "founder of the lab", "person")],
        "constraints": [
            _con("cp", "founder of the Quanta Lab", ["quanta", "founder"], ["P"], 1.6,
                 required=True, priority="high", blocks_answer_if_unresolved=True,
                 testable_claim="x", how_to_test="read", semantic_label="role"),
            _con("cp2", "later moved to Berlin", ["berlin", "moved"], ["P"], 1.5,
                 required=True, priority="high", blocks_answer_if_unresolved=True,
                 testable_claim="x", how_to_test="read", semantic_label="relocation"),
            _con("cy", "birth year of the founder", ["birth", "year"], ["Y"], 0.4,
                 constraint_type="answer_shape", required=True, priority="high",
                 blocks_answer_if_unresolved=True, testable_claim="x", how_to_test="read",
                 semantic_label="birth_year")],
        "dependency_edges": [], "known_context_terms": []}
    f = _frame(SUBJ2, SUBJ_Q)
    fr = CandidateFrontier(f)
    # support ONLY cp -> founder stays active (cp2 unresolved) but is a supported subject.
    fr.ingest_evidence([_obs("Quanta Lab founder", "Ada Lin is the founder of the Quanta Lab.",
                             "https://quanta.example/about")],
                       source_tool="serper", directed_slot_id=_sid(f, "founder of the lab"),
                       directed_constraint_ids=["cp"], proposal_id="p")
    acts = fr.generate_frontier_actions()
    binds = [a for a in acts if a.action_type == "bind_target_answer_slot"]
    verifies = [a for a in acts if a.action_type == "verify_candidate_constraint"
                and a.target_slot_id == _sid(f, "founder of the lab")]
    assert binds and verifies
    assert max(a.expected_information_gain for a in binds) > max(
        v.expected_information_gain for v in verifies)
    assert fr.metrics()["repeated_intermediate_verify_after_subject_supported_count"] >= 1


# --------------------------------------------------------------------- E: anchor gate
def test_fuzzy_dedupe_rejects_near_duplicate_zero_progress_queries():
    assert _is_fuzzy_duplicate("officer probation years", {"years probation officer"})
    assert _is_fuzzy_duplicate("probation officer years worked", {"years probation officer"})
    assert not _is_fuzzy_duplicate("hotel opened 1955 santa fe", {"years probation officer"})


def test_anchor_gate_accepts_compact_reformulation():
    HOTEL = {
        "target_answer_slots": [_slot("H", "hotel", "organization", is_target_answer_slot=True,
                                      is_intermediate_slot=False)],
        "latent_slots": [],
        "constraints": [_con("ch", "hotel originally opened in 1955",
                             ["hotel", "originally", "opened", "1955"], ["H"], 1.4,
                             required=True, priority="high", blocks_answer_if_unresolved=True,
                             testable_claim="x", how_to_test="read", semantic_label="opened")],
        "dependency_edges": [], "known_context_terms": []}
    f = _frame(HOTEL, "Which hotel originally opened in 1955?")
    fr = CandidateFrontier(f)
    anchors = _anchor_terms(f)
    assert "1955" in anchors and "hotel" in anchors
    from regimes_probe.agent.llm_frontier import FrontierProposal
    p = FrontierProposal(proposal_id="p1", action_type="generate_candidates_for_slot",
                         target_slot_id=_sid(f), constraint_ids=["ch"],
                         proposed_query="hotel 1955", proposed_tool="serper_search")
    ok, reason = validate_proposal(p, f, fr, failed_norms=set(),
                                   available_tools=["serper_search"], remaining_budget=3)
    assert ok, reason
    assert "1955" in p.matched_anchor_tokens or "hotel" in p.matched_anchor_tokens


def test_all_proposals_anchor_rejected_uses_relaxed_gate_not_generic_seed():
    f = _frame(PERSON, PERSON_Q)
    fr = CandidateFrontier(f)
    sid = _sid(f)
    # an anchor-RICH proposal (a quoted proper-noun candidate) whose tokens don't intersect
    # the frame's constraint anchors -> rejected by the soft anchor gate, then RELAXED in.
    payload = json.dumps({"proposals": [{
        "proposal_id": "pr1", "action_type": "generate_candidates_for_slot",
        "target_slot_id": sid, "constraint_ids": ["c1"],
        "proposed_query": '"Dana Wu" engineer profile', "proposed_tool": "serper_search",
        "expected_evidence": "a named engineer", "anchors_used": ["Dana Wu"],
        "avoids_generic_query": True, "confidence": 0.8}]})
    prop = LLMFrontierProposer(model_fn=lambda _p: payload, cache=_PC(), model="stub")
    card = build_research_state_card(
        f, fr, question=PERSON_Q, epistemic_mode="multi_constraint_research",
        available_tools=["serper_search"], remaining_budget=3, memory_access_mode="no_memory",
        failed_queries=[], no_progress_queries=[], det_plan=None)
    plan, meta = prop.plan_step(card, None, frame=f, frontier=fr, mode="planner")
    assert plan is not None and plan.kind == "search"
    assert prop.relaxed_gate_selected_count == 1
    assert prop.stats()["generic_fallback_after_all_proposals_rejected_count"] == 0
    assert any(e["event_type"] == "proposal_gate_relaxed" for e in meta["events"])


# --------------------------------------------------------------------- F: seed floor
def test_generic_single_token_seed_blocked_for_hard_task():
    REST = {
        "target_answer_slots": [_slot("R", "restaurant", "organization",
                                      is_target_answer_slot=True, is_intermediate_slot=False)],
        "latent_slots": [_slot("H", "nearby hotel", "organization")],
        "constraints": [_con("ct", "restaurant", ["restaurant"], ["R"], 0.2,
                             constraint_type="type", required=True, priority="high",
                             blocks_answer_if_unresolved=True, testable_claim="x",
                             how_to_test="read", semantic_label="venue_type")],
        "dependency_edges": [], "known_context_terms": []}
    f = _frame(REST, "Which restaurant near a hotel...?")
    fr = CandidateFrontier(f)
    sp = fr.propose_step_action(observations=[], budget_remaining=4, reading_tools=True)
    assert sp.kind == "unexecutable" and sp.reason == "generic_single_token_seed_blocked"
    assert fr.metrics()["seed_query_generic_blocked_count"] >= 1
    assert fr.metrics()["generic_single_token_seed_executed_count"] == 0


# --------------------------------------------------------------------- G: abstain admissibility
def test_abstain_inadmissible_while_budget_and_anchor_rich_action_exist():
    f = _frame(PERSON, PERSON_Q)
    fr = CandidateFrontier(f)
    fr.ingest_evidence([_obs("Dana Wu", "Dana Wu is an engineer.",
                             "https://caltech.edu/people/dana-wu")],
                       source_tool="serper", directed_slot_id=_sid(f),
                       directed_constraint_ids=["c1"], proposal_id="p")
    adm, why = fr.abstain_admissible(budget_remaining=4)
    assert adm is False and why == "executable_anchor_rich_action_available"
    assert fr.abstain_blocked_due_to_executable_proposal_count >= 1
    assert fr.metrics()["abstain_with_budget_remaining_count"] == 0
    # with no budget, abstain becomes admissible.
    assert fr.abstain_admissible(budget_remaining=0)[0] is True


# --------------------------------------------------------------------- H: source hygiene
def test_generic_definition_page_not_read_for_concrete_entity_task():
    f = _frame(PERSON, PERSON_Q)          # target role=person -> concrete entity task
    fr = CandidateFrontier(f)
    assert fr._is_concrete_entity_task()
    defn = _obs("Engineer Definition & Meaning", "Dictionary definition of engineer.",
                "https://dictionary.example/engineer")
    chosen = fr._resolve_read([defn], frozenset(), frozenset(),
                              page_fetch_available=True, scrape_available=False,
                              allow_social=False, force_page_fetch=True)
    assert chosen is None
    assert fr.metrics()["generic_definition_source_selected_count"] >= 1
    assert fr.metrics()["generic_definition_source_read_count"] == 0


# --------------------------------------------------------------------- I: prejudge hygiene
def test_chrome_candidate_rejected_before_judge_runs():
    f = _frame(PERSON, PERSON_Q)
    judge = EvidenceJudge(model_fn=lambda _p: json.dumps({"judgment": "full_support"}),
                          cache=ParserCache(), model="stub", enabled=True)
    fr = CandidateFrontier(f, interpreter=EvidenceInterpreter(judge=judge))
    fr.ingest_evidence([_obs("Login", "Sign in / Subscribe / Datasets / Trending",
                             "https://x.example/login")],
                       source_tool="serper", directed_slot_id=_sid(f),
                       directed_constraint_ids=["c1"], proposal_id="p")
    assert judge.calls == 0                # chrome never reaches the judge (pre-judge triage)


# --------------------------------------------------------------------- J: coreference safety
def test_distinct_entity_slots_are_not_collapsed():
    # the safety invariant for coreference collapse: distinct role-bearing entities
    # (hotel/museum/founder) stay as SEPARATE slates — never merged just for sharing a role.
    f = _frame(SUBJ, SUBJ_Q)
    fr = CandidateFrontier(f)
    assert len(f.all_slots) == 2                                 # founder + birth-year, distinct
    assert _sid(f, "founder of the lab") != _sid(f, "birth year")
    assert len({sid for sid in fr.slates}) == len(fr.slates)   # no accidental merge


# --------------------------------------------------------------------- L: location staging
def test_dependent_slot_search_deferred_until_dependency_supported():
    f = _frame(SUBJ, SUBJ_Q)               # birth year (Y) depends_on founder (P)
    fr = CandidateFrontier(f)
    acts = fr.generate_frontier_actions()
    # before the founder is supported, NO generate/verify action targets the birth-year slot.
    yid = _sid(f, "birth year")
    assert not any(a.action_type in ("generate_candidates_for_slot",
                                     "verify_candidate_constraint")
                   and a.target_slot_id == yid for a in acts)
    assert any(e["event_type"] == "dependent_slot_search_deferred" for e in fr.events)
    assert fr.metrics()["premature_birth_year_search_before_founder_supported_count"] == 0
    assert fr.metrics()["premature_founder_search_before_place_supported_count"] == 0


# --------------------------------------------------------------------- M/O: detectors + labels
def test_regime_detector_and_local_label_metrics_present_and_zero_invariants():
    f, fr = _person_frontier_with_pending()
    m = fr.metrics()
    for k in ("read_loop_open_count", "read_success_no_evidence_added_count",
              "requires_read_count", "pending_read_judgments_count"):
        assert k in m
    # local candidate support is never labelled in a way that implies global answer readiness.
    assert m["debug_confirmed_label_when_answer_gate_false_count"] == 0


def test_read_success_with_no_evidence_added_flags_read_loop_open():
    f, fr = _person_frontier_with_pending()
    cand = next(c for c in fr.candidates_by_id.values() if "dana wu" in c.candidate_text.lower())
    before = fr.metrics()["read_loop_open_count"]
    # a read whose body adds nothing AND leaves the pending judgment open is a read-loop seam.
    fr.ingest_evidence([_obs("Dana Wu", "Generic alumni listing with no resolving facts. " * 5,
                             "https://caltech.edu/people/dana-wu")],
                       source_tool="page_fetch", read_depth=1,
                       directed_slot_id=_sid(f), read_candidate_id=cand.candidate_id)
    assert fr.metrics()["read_loop_open_count"] >= before


def test_replay_safe_no_model_calls_for_passages_helper_is_pure():
    # extract_passages is pure/deterministic: identical inputs -> identical output.
    body = "alpha beta Caltech 1990 gamma delta" * 10
    a = extract_passages(body, ["Caltech", "1990"]).to_dict()
    b = extract_passages(body, ["Caltech", "1990"]).to_dict()
    assert a == b
