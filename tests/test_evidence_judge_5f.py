"""Level 5f: read-reliable, candidate-safe, support-aware evidence judging.

Confirm-gate strictness, discriminative confirmation, post-model support contract
(named/generic/relational/quote-anchor), contradiction early-stop, and support-aware
no-progress. No live providers/models — local stubs only.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from regimes_probe.agent.candidate_frontier import CandidateFrontier
from regimes_probe.agent.evidence_interpreter import EvidenceInterpreter
from regimes_probe.agent.evidence_judge import EvidenceJudge, EvidenceJudgment, enforce_hard_rules
from regimes_probe.agent.llm_task_frame import LLMTaskFrameParser, ParserCache, build_task_frame


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


# A person slot with a broad (non-discriminative) and a discriminative blocking constraint.
PERSON_Q = "Who is the engineer (broadly) who specifically studied at Caltech in 1990?"
PERSON = {
    "target_answer_slots": [_slot("D", "engineer", "person", is_target_answer_slot=True,
                                  is_intermediate_slot=False)],
    "latent_slots": [],
    "constraints": [
        _con("c_role", "is an engineer", ["engineer"], ["D"], 0.3, required=True,
             priority="high", testable_claim="occupation engineer", how_to_test="read",
             semantic_label="occupation"),
        _con("c_disc", "studied at Caltech 1990", ["caltech", "1990"], ["D"], 2.0, required=True,
             priority="high", testable_claim="studied at Caltech in 1990", how_to_test="read",
             semantic_label="education")],
    "dependency_edges": [], "known_context_terms": []}


def _judge_fn(rules):
    def _fn(prompt: str) -> str:
        for needle, payload in rules.items():
            if needle in prompt:
                return json.dumps(payload)
        return json.dumps({"judgment": "irrelevant"})
    return _fn


def _frontier(frame, model_fn=None):
    judge = (EvidenceJudge(model_fn=model_fn, cache=ParserCache(), model="stub", enabled=True)
             if model_fn is not None else None)
    return CandidateFrontier(frame, interpreter=EvidenceInterpreter(judge=judge)), judge


# --------------------------------------------------------------------------- confirm gate (C/D)
def test_confirm_requires_all_blocking_and_discriminative():
    f = _frame(PERSON, PERSON_Q)
    fr = CandidateFrontier(f)
    sid = f.target_answer_slots[0].slot_id
    from regimes_probe.agent.candidate_frontier import SlotCandidate, _hash
    cand = SlotCandidate(candidate_id="cand1", candidate_text="Dana Wu",
                         normalized_text_hash=_hash("dana wu"), inferred_role="person", slot_id=sid)
    fr.slates[sid].candidates["cand1"] = cand
    fr.candidates_by_id["cand1"] = cand
    # only the broad (non-discriminative) blocking constraint supported -> NOT confirmable.
    cand.constraints_supported = ["c_role"]
    assert not fr._confirmable(cand, contaminated=False)
    # add the discriminative blocking constraint -> confirmable.
    cand.constraints_supported = ["c_role", "c_disc"]
    assert fr._confirmable(cand, contaminated=False)


def test_single_supported_constraint_does_not_confirm():
    f = _frame(PERSON, PERSON_Q)
    fr = CandidateFrontier(f)
    sid = f.target_answer_slots[0].slot_id
    from regimes_probe.agent.candidate_frontier import SlotCandidate, _hash
    cand = SlotCandidate(candidate_id="c1", candidate_text="Dana Wu",
                         normalized_text_hash=_hash("dana wu"), inferred_role="person", slot_id=sid,
                         constraints_supported=["c_disc"])   # one of two blocking
    assert not fr._confirmable(cand, contaminated=False)


def test_partial_support_never_confirms_blocking():
    f = _frame(PERSON, PERSON_Q)
    fr = CandidateFrontier(f)
    sid = f.target_answer_slots[0].slot_id
    from regimes_probe.agent.candidate_frontier import SlotCandidate, _hash
    cand = SlotCandidate(candidate_id="c1", candidate_text="Dana Wu",
                         normalized_text_hash=_hash("dana wu"), inferred_role="person", slot_id=sid,
                         constraints_supported=["c_role"], constraints_partial=["c_disc"])
    assert not fr._confirmable(cand, contaminated=False)     # partial cannot resolve a blocker


def test_confirm_gate_invariants_zero():
    # drive the full agent on the synthetic fixture; the confirm-gate invariants stay 0.
    f = _frame(PERSON, PERSON_Q)
    fr = CandidateFrontier(f)
    sid = f.target_answer_slots[0].slot_id
    fr.ingest_evidence([_obs("Dana Wu - Engineer - LinkedIn",
                             "Dana Wu studied at Caltech in 1990.",
                             "https://linkedin.com/in/dana-wu")],
                       source_tool="serper", directed_slot_id=sid,
                       directed_constraint_ids=["c_role", "c_disc"], proposal_id="p1")
    m = fr.metrics()
    assert m["confirmed_hypothesis_with_unresolved_blocking_count"] == 0
    assert m["confirmed_candidate_with_junk_blocking_slot_count"] == 0
    assert m["blocking_constraint_partial_support_confirmed_count"] == 0


# --------------------------------------------------------------------------- post-model contract (G)
def test_generic_descriptor_never_full_support():
    j = enforce_hard_rules(EvidenceJudgment(judgment="full_support", quote="Graphic Designer"),
                           contaminated=False, source_role="professional_profile", has_quote=True,
                           candidate_text="Graphic Designer", slot_role="person",
                           counters=__import__("collections").Counter())
    assert j.judgment != "full_support"


def test_full_support_requires_named_candidate():
    from collections import Counter
    ctr = Counter()
    j = enforce_hard_rules(EvidenceJudgment(judgment="full_support", quote="the founder"),
                           contaminated=False, source_role="official_page", has_quote=True,
                           candidate_text="the founder", slot_role="person", counters=ctr)
    assert j.judgment == "partial_support"
    # a named candidate whose name is in the quote stays full.
    j2 = enforce_hard_rules(EvidenceJudgment(judgment="full_support", quote="Maria Lopez founded it"),
                            contaminated=False, source_role="official_page", has_quote=True,
                            candidate_text="Maria Lopez", slot_role="person", counters=Counter())
    assert j2.judgment == "full_support"


def test_quote_must_anchor_candidate_for_full():
    # named candidate, but the quote doesn't mention them -> downgrade.
    j = enforce_hard_rules(EvidenceJudgment(judgment="full_support", quote="opened in 1955"),
                           contaminated=False, source_role="official_page", has_quote=True,
                           candidate_text="Maria Lopez", slot_role="person",
                           counters=__import__("collections").Counter())
    assert j.judgment == "partial_support"


def test_relational_full_requires_object_anchor():
    from collections import Counter
    ctr = Counter()
    j = enforce_hard_rules(EvidenceJudgment(judgment="full_support", quote="Maria Lopez is a founder"),
                           contaminated=False, source_role="official_page", has_quote=True,
                           candidate_text="Maria Lopez", slot_role="person",
                           relational=True, object_anchored=False, counters=ctr)
    assert j.judgment == "partial_support"
    assert ctr.get("relational_support_without_object_anchor", 0) == 1


def test_contaminated_never_supports_via_hard_rule():
    j = enforce_hard_rules(EvidenceJudgment(judgment="full_support", quote="Maria Lopez founded it"),
                           contaminated=True, source_role="benchmark_contaminated", has_quote=True,
                           candidate_text="Maria Lopez", slot_role="person",
                           counters=__import__("collections").Counter())
    assert j.judgment not in ("full_support", "partial_support")


# --------------------------------------------------------------------------- contradiction stop (H)
def test_contradiction_early_stop_saves_judge_calls():
    f = _frame(PERSON, PERSON_Q)
    fr, judge = _frontier(f, _judge_fn({
        "occupation engineer": {"judgment": "contradiction",
                                "contradicted_facets": ["occupation"]},
        "studied at Caltech": {"judgment": "full_support", "quote": "Dana Wu studied at Caltech"}}))
    sid = f.target_answer_slots[0].slot_id
    fr.ingest_evidence([_obs("Dana Wu - Chef - LinkedIn", "Dana Wu is a chef, not an engineer.",
                             "https://linkedin.com/in/dana-wu")],
                       source_tool="serper", directed_slot_id=sid,
                       directed_constraint_ids=["c_role", "c_disc"], proposal_id="p1")
    # contradiction on the (blocking) role constraint stops further judging for this candidate.
    assert judge.contradiction_early_stop >= 1
    assert judge.stats()["judge_calls_saved_by_contradiction_stop"] >= 1


# --------------------------------------------------------------------------- support-aware (I)
def test_supported_candidate_not_rejected_no_progress():
    f = _frame(PERSON, PERSON_Q)
    fr = CandidateFrontier(f)
    sid = f.target_answer_slots[0].slot_id
    from regimes_probe.agent.candidate_frontier import SlotCandidate, _hash
    cand = SlotCandidate(candidate_id="c1", candidate_text="Dana Wu",
                         normalized_text_hash=_hash("dana wu"), inferred_role="person", slot_id=sid,
                         constraints_supported=["c_disc"])
    fr.slates[sid].candidates["c1"] = cand
    fr.candidates_by_id["c1"] = cand
    for _ in range(5):                          # repeated no-progress on the slate
        fr.note_no_progress_for_slate(sid)
    assert cand.status != "rejected"            # supported candidate is exempt
    assert fr.metrics()["supported_candidate_rejected_no_progress_count"] == 0


def test_unsupported_candidate_still_rejected_no_progress():
    f = _frame(PERSON, PERSON_Q)
    fr = CandidateFrontier(f)
    sid = f.target_answer_slots[0].slot_id
    from regimes_probe.agent.candidate_frontier import SlotCandidate, _hash
    cand = SlotCandidate(candidate_id="c1", candidate_text="Random Name",
                         normalized_text_hash=_hash("random name"), inferred_role="person",
                         slot_id=sid)
    fr.slates[sid].candidates["c1"] = cand
    fr.candidates_by_id["c1"] = cand
    for _ in range(3):
        fr.note_no_progress_for_slate(sid)
    assert cand.status == "rejected"            # no support -> normal no-progress rejection


# --------------------------------------------------------------------------- read accounting (A)
def test_read_accounting_invariants_present_and_zero():
    f = _frame(PERSON, PERSON_Q)
    fr = CandidateFrontier(f)
    sid = f.target_answer_slots[0].slot_id
    fr.ingest_evidence([_obs("Dana Wu - Engineer - LinkedIn", "Dana Wu studied at Caltech 1990.",
                             "https://linkedin.com/in/dana-wu")],
                       source_tool="serper", read_depth=1, directed_slot_id=sid,
                       directed_constraint_ids=["c_disc"], proposal_id="p1")
    # support_from_read fires when a read produces support.
    assert fr.metrics()["support_from_read_count"] >= 1
