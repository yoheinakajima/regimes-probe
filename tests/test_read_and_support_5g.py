"""Level 5g: read-intent lifecycle (reads actually execute / blocked-with-reason) and the
prompt-text-is-not-evidence safety invariant. No live providers/models — local stubs only.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from regimes_probe.agent.candidate_frontier import CandidateFrontier, _is_clean_url
from regimes_probe.agent.evidence_interpreter import EvidenceInterpreter
from regimes_probe.agent.evidence_judge import EvidenceJudge
from regimes_probe.agent.hypothesis_table import HypothesisTable
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


def _obs(title, snippet, url, *, auth=0.7, contam=False):
    return SimpleNamespace(title=title, snippet=snippet, url=url, source_authority=auth,
                           failed=False, benchmark_contaminated=contam)


PERSON_Q = "Who is the engineer who specifically studied at Caltech in 1990?"
PERSON = {
    "target_answer_slots": [_slot("D", "engineer", "person", is_target_answer_slot=True,
                                  is_intermediate_slot=False)],
    "latent_slots": [],
    "constraints": [_con("c1", "studied at Caltech 1990", ["caltech", "1990"], ["D"], 2.0,
                         required=True, priority="high", blocks_answer_if_unresolved=True,
                         testable_claim="x", how_to_test="read", semantic_label="education")],
    "dependency_edges": [], "known_context_terms": []}

# a frame whose only blocking constraint has GENERIC terms (no proper noun / year).
DIST_Q = "What is the driving distance in miles between the two places?"
DIST = {
    "target_answer_slots": [_slot("D", "distance", "concept", is_target_answer_slot=True,
                                  is_intermediate_slot=False)],
    "latent_slots": [],
    "constraints": [_con("c_dist", "driving distance in miles", ["driving", "distance", "miles"],
                         ["D"], 1.5, required=True, priority="high", blocks_answer_if_unresolved=True,
                         testable_claim="x", how_to_test="read", semantic_label="distance")],
    "dependency_edges": [], "known_context_terms": []}


def _sid(f):
    return f.target_answer_slots[0].slot_id


# --------------------------------------------------------------------------- read lifecycle (req 1)
def test_read_executes_via_candidate_url_not_search():
    f = _frame(PERSON, PERSON_Q)
    fr = CandidateFrontier(f)
    sid = _sid(f)
    o = _obs("Dana Wu", "Dana Wu is an engineer.", "https://caltech.edu/people/dana-wu")
    for _ in range(2):
        fr.ingest_evidence([o], source_tool="serper", directed_slot_id=sid,
                           directed_constraint_ids=["c1"], proposal_id="p")
    cand = next(c for c in fr.candidates_by_id.values() if "dana wu" in c.candidate_text.lower())
    assert cand.source_urls == ["https://caltech.edu/people/dana-wu"]
    # NO observations -> _resolve_read fails; the candidate URL must still drive a read.
    sp = fr.propose_step_action(observations=[], budget_remaining=4, reading_tools=True,
                                page_fetch_available=True)
    assert sp.kind == "read" and sp.action_type == "read_candidate_source"
    assert sp.read_decision.tool == "page_fetch"
    assert fr.metrics()["read_desired_count"] >= 1 and fr.metrics()["read_selected_count"] >= 1


def test_read_without_url_records_blocked_not_search():
    from regimes_probe.agent.candidate_frontier import SlotCandidate, _hash
    f = _frame(PERSON, PERSON_Q)
    fr = CandidateFrontier(f)
    sid = _sid(f)
    # a candidate with a source DOMAIN (so a read action is generated) but NO clean URL.
    cand = SlotCandidate(candidate_id="c1", candidate_text="Dana Wu",
                         normalized_text_hash=_hash("dana wu"), inferred_role="person", slot_id=sid,
                         source_domains=["x.example"], constraints_unknown=["c1"])
    fr.slates[sid].candidates["c1"] = cand
    fr.candidates_by_id["c1"] = cand
    fr._slot_no_support_streak[sid] = 5          # force a read
    sp = fr.propose_step_action(observations=[], budget_remaining=4, reading_tools=True,
                                page_fetch_available=True)
    assert sp.kind == "unexecutable" and sp.reason == "read_blocked_no_url"
    assert fr.metrics()["read_blocked_no_url_count"] >= 1
    assert any(e["event_type"] == "read_blocked_no_url" for e in fr.events)


def test_contaminated_url_not_stored_for_reading():
    f = _frame(PERSON, PERSON_Q)
    fr = CandidateFrontier(f)
    sid = _sid(f)
    # contaminated source -> the candidate is rejected before assignment, so no read URL.
    fr.ingest_evidence([_obs("mirror", "Dana Wu studied at Caltech 1990",
                             "https://hf.co/mirror", contam=True)],
                       source_tool="serper", directed_slot_id=sid,
                       directed_constraint_ids=["c1"], proposal_id="p")
    assert all(not c.source_urls for c in fr.candidates_by_id.values())


def test_is_clean_url_helper():
    assert _is_clean_url("https://example.org/page")
    assert not _is_clean_url("just a search query")
    assert not _is_clean_url("")
    assert not _is_clean_url("ftp://x")


# --------------------------------------------------------------------------- prompt-not-evidence (req 4)
def test_blocking_constraint_not_resolved_from_contaminated_evidence():
    f = _frame(PERSON, PERSON_Q)
    ht = HypothesisTable(f)
    ht.ingest_evidence([_obs("BrowseComp mirror", "engineer studied at Caltech 1990",
                             "https://hf.co/dataset", contam=True)],
                       source_tool="serper", stage=1)
    con = next(c for c in f.constraints if c.constraint_id == "c1")
    assert con.status != "resolved"             # contaminated source cannot resolve a blocker


def test_blocking_constraint_not_resolved_from_generic_overlap_only():
    f = _frame(DIST, DIST_Q)
    ht = HypothesisTable(f)
    # a clean snippet that merely ECHOES the generic question terms (no proper noun/year).
    ht.ingest_evidence([_obs("Driving directions", "driving distance in miles between places",
                             "https://maps.example/x")],
                       source_tool="serper", stage=1)
    con = next(c for c in f.constraints if c.constraint_id == "c_dist")
    assert con.status != "resolved"             # generic question-echo is not evidence


def test_blocking_constraint_resolves_with_distinctive_evidence():
    f = _frame(PERSON, PERSON_Q)
    ht = HypothesisTable(f)
    ht.ingest_evidence([_obs("Dana Wu bio", "Dana Wu studied at Caltech in 1990.",
                             "https://caltech.edu/x")],
                       source_tool="serper", stage=1)
    con = next(c for c in f.constraints if c.constraint_id == "c1")
    assert con.status == "resolved"             # proper noun (Caltech) + year (1990) = real anchor
    assert con.supporting_evidence_ids          # resolution is evidence-backed


def test_initial_blocking_constraint_resolved_without_evidence_is_zero():
    # a freshly-built frame has no resolved blocking constraints (none from prompt text).
    f = _frame(PERSON, PERSON_Q)
    resolved_no_ev = sum(1 for c in f.constraints if c.status == "resolved"
                         and not c.supporting_evidence_ids)
    assert resolved_no_ev == 0


# --------------------------------------------------------------------------- judge pre-gate (req 7)
def test_judge_not_invoked_on_chrome_candidates():
    f = _frame(PERSON, PERSON_Q)
    judge = EvidenceJudge(model_fn=lambda _p: json.dumps({"judgment": "full_support"}),
                          cache=ParserCache(), model="stub", enabled=True)
    fr = CandidateFrontier(f, interpreter=EvidenceInterpreter(judge=judge))
    fr.ingest_evidence([_obs("Login", "Sign in / Subscribe / Related Topics",
                             "https://x.example/login")],
                       source_tool="serper", directed_slot_id=_sid(f),
                       directed_constraint_ids=["c1"], proposal_id="p")
    assert judge.calls == 0                      # chrome is rejected before the judge runs


def test_read_intent_metrics_present():
    f = _frame(PERSON, PERSON_Q)
    fr = CandidateFrontier(f)
    m = fr.metrics()
    for k in ("read_desired_count", "read_selected_count", "read_blocked_no_url_count"):
        assert k in m
