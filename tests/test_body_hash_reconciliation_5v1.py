"""Level 5v.1 offline tests: body-hash / canonicalization reconciliation.

Verification is body-id-first and manifest-authoritative: identity is proven by the strict
entry's body hash matching the manifest body hash under a shared canonical basis; cache
rehydration and passage windows are validated SEPARATELY and never misclassified as a body
hash mismatch. Every unverifiable obligation carries a finite mismatch_class, and N
obligations on one bad body dedupe to one read-body-level mismatch. Synthetic fixtures only —
zero live provider/model calls, no gold/benchmark content.
"""

from __future__ import annotations

import json
from pathlib import Path

from regimes_probe.agent.candidate_frontier import CandidateFrontier, SlotCandidate, _hash
from regimes_probe.agent.evidence_interpreter import EvidenceInterpreter
from regimes_probe.agent.llm_task_frame import LLMTaskFrameParser, ParserCache, build_task_frame
from regimes_probe.agent.read_judgment import (
    READ_BODY_MISMATCH_CLASSES, STRICT_REJUDGE_VERSION, BodyHashBasis,
    canonical_read_body_hash, strict_rejudgment_key)
from regimes_probe.eval.replay_validation import (
    consistency_violations, deterministic_replay_judge, load_legacy_run)

_URL_A = "https://profiles.example/people/alpha-profile-page-with-the-relevant-biography"
_URL_B = "https://records.example/archive/beta-profile-page-with-historic-listings"
_BODY_SUPPORT = (("page boilerplate filler text. " * 40)
                 + " Avery Quinn studied at the Northvale Institute in 1987. ")
_BODY_OTHER = (("different boilerplate text here. " * 40)
               + " Jordan Vale studied at the Eastgate Institute in 1991. ")


def _frame_payload():
    return {"target_answer_slots": [{"slot_id": "D", "slot_name": "engineer",
                                     "slot_role": "person", "is_target_answer_slot": True,
                                     "is_intermediate_slot": False}],
            "latent_slots": [],
            "constraints": [{"constraint_id": "c1", "text_span": "studied at Northvale in 1987",
                             "normalized_terms": ["northvale", "1987"], "applies_to": ["D"],
                             "discriminative_score": 2.0, "specificity_score": 2.0,
                             "required": True, "priority": "high",
                             "blocks_answer_if_unresolved": True, "testable_claim": "x",
                             "how_to_test": "read", "semantic_label": "education"}],
            "dependency_edges": [], "known_context_terms": []}


def _frontier(judge):
    parser = LLMTaskFrameParser(model_fn=lambda _p: json.dumps(_frame_payload()), model="stub")
    frame, _ = build_task_frame("x", "Who is the engineer who studied at Northvale in 1987?",
                                use_llm=True, parser=parser)
    return CandidateFrontier(frame, interpreter=EvidenceInterpreter(judge=judge)), frame


def _sid(fr):
    return next(iter(fr.slates))


def _add_candidate(fr, text="Avery Quinn", cid="cand1"):
    sid = _sid(fr)
    cand = SlotCandidate(candidate_id=cid, candidate_text=text,
                         normalized_text_hash=_hash(text.lower()), inferred_role="person",
                         slot_id=sid, constraints_unknown=["c1"], aliases=[text],
                         source_urls=[_URL_A])
    fr.slates[sid].candidates[cid] = cand
    fr.candidates_by_id[cid] = cand
    return cand


def _add_pending(fr, *, url=_URL_A, candidate_id="cand1", constraint_id="c1"):
    fr._register_pending_read_judgment(candidate_id=candidate_id, slot_id=_sid(fr),
                                       constraint_id=constraint_id, source_url=url,
                                       source_role="professional_profile")
    return next(p for p in fr.pending_read_judgments.values()
                if p.candidate_id == candidate_id and p.constraint_id == constraint_id)


def _persist(tmp_path, fr, *, cache_body=_BODY_SUPPORT, cache_url=_URL_A, slot_id="s0",
             manifest=True, mutate_record=None) -> Path:
    run = tmp_path / "run"
    (run / "cache").mkdir(parents=True, exist_ok=True)
    cache = {"mode": "auto", "store_raw": False, "entries": [
        {"request_hash": "h1", "provider": "page_fetch", "name": "page_fetch",
         "request_meta": {"query": cache_url},
         "response": {"provider": "page_fetch", "query": cache_url,
                      "results": [{"title": "Avery Quinn", "url": cache_url,
                                   "snippet": cache_body, "rank": 0}], "error": None}}]}
    (run / "cache" / "provider_cache.json").write_text(json.dumps(cache))
    dbg = fr.to_debug()
    pj = dbg["pending_read_judgments"]
    for p in pj:
        p["slot_id"] = slot_id
    record = {
        "item_id": "v1-001",
        "question_preview": "Who is the engineer who studied at the Northvale Institute in 1987?",
        "task_frame": {
            "target_answer_slots": [{"slot_id": slot_id, "slot_name": "engineer",
                                     "descriptor": "engineer"}],
            "constraints": [{"constraint_id": "c1",
                             "text_span": "studied at the Northvale Institute in 1987",
                             "normalized_terms": ["northvale", "institute", "1987"],
                             "applies_to": [slot_id]}]},
        "candidate_frontier": {
            "events": dbg["events"], "events_truncated": 0, "interpretations": [],
            "pending_read_judgments": pj,
            "pending_service_urls": dbg["pending_service_urls"],
            "read_bodies": dbg.get("read_bodies", []) if manifest else [],
            "evidence_gaps": dbg.get("evidence_gaps", []),
            "metrics": {"unrelated_read_executed_while_pending_count": 0},
            "slates": [{"slot_role": "person",
                        "top_candidates": [{"candidate_text_preview": "Avery Quinn"}]}]},
        "calls": [], "evidence": []}
    if mutate_record:
        mutate_record(record)
    (run / "debug_questions.jsonl").write_text(json.dumps(record) + "\n")
    if not (run / "llm_evidence_judge_cache.json").exists():
        (run / "llm_evidence_judge_cache.json").write_text("{}")
    return run


def _route(tmp_path, *, read_text=_BODY_SUPPORT, url=_URL_A):
    run = tmp_path / "run"
    (run / "cache").mkdir(parents=True, exist_ok=True)
    judge = deterministic_replay_judge()
    judge.cache = ParserCache(str(run / "llm_evidence_judge_cache.json"))
    fr, _ = _frontier(judge)
    cand = _add_candidate(fr)
    p = _add_pending(fr, url=url)
    fr.route_read_into_pending_judgments(
        candidate_id="cand1", source_url=url, read_text=read_text, judge=judge,
        read_meta={"tool": "page_fetch", "body_provider": "page_fetch"})
    fr.finalize_pending_service(budget_remaining=0, reading_tools_enabled=True)
    return fr, p, judge


# ------------------------------------------------ 1: verify by read_body_id vs manifest hash
def test_strict_entry_verifies_by_read_body_id_against_manifest(tmp_path):
    fr, p, judge = _route(tmp_path)
    run = _persist(tmp_path, fr, cache_body=_BODY_SUPPORT, slot_id=p.slot_id)
    res = load_legacy_run(run)
    o = res.obligations[0]
    assert o.rejudgment_verify_method == "body_id"
    assert o.strict_entry_body_hash == o.manifest_body_hash
    assert o.body_hash_basis == BodyHashBasis.STORED_READ_BODY
    assert o.pipeline_status == "closed"
    pm = res.metrics
    assert pm["read_body_id_verification_success_count"] == 1
    assert pm["read_body_id_verification_failure_count"] == 0
    assert pm["recorded_rejudgment_body_hash_mismatch_count"] == 0
    assert consistency_violations(pm) == []


# ------------------------------------------------ 2: no URL fallback when read_body_id present
def test_read_body_id_present_never_falls_back_to_url(tmp_path):
    fr, p, judge = _route(tmp_path)
    # drop the manifest -> the read_body_id cannot be resolved; MUST NOT URL-fallback.
    run = _persist(tmp_path, fr, cache_body=_BODY_SUPPORT, slot_id=p.slot_id, manifest=False)
    res = load_legacy_run(run)
    o = res.obligations[0]
    assert o.rejudgment_verify_method == "none"            # not url_fallback
    assert o.read_body_mismatch_class == "read_body_id_not_found_in_manifest"
    assert o.rejudgment_unverifiable_reason == "body_not_found"
    assert res.metrics["recorded_rejudgment_verified_by_url_fallback_count"] == 0
    assert consistency_violations(res.metrics) == []


# ------------------------------------------------ 3: same URL, two bodies -> id selects body
def test_two_bodies_same_url_validator_uses_read_body_id_body(tmp_path):
    # register TWO different bodies for the same URL; the strict entry references body #1's
    # read_body_id, and the manifest holds both. The validator must compare against body #1.
    run = tmp_path / "run"
    (run / "cache").mkdir(parents=True)
    judge = deterministic_replay_judge()
    judge.cache = ParserCache(str(run / "llm_evidence_judge_cache.json"))
    fr, _ = _frontier(judge)
    _add_candidate(fr)
    p = _add_pending(fr, url=_URL_A)
    fr.route_read_into_pending_judgments(
        candidate_id="cand1", source_url=_URL_A, read_text=_BODY_SUPPORT, judge=judge,
        read_meta={"tool": "page_fetch", "body_provider": "page_fetch"})
    rbid1 = p.read_body_id
    # a DIFFERENT body for the same URL is registered into the manifest (e.g. a later read).
    fr._register_read_body(source_url=_URL_A, read_text=_BODY_OTHER, pending_ids=[],
                           read_meta={"tool": "page_fetch", "body_provider": "page_fetch"})
    fr.finalize_pending_service(budget_remaining=0, reading_tools_enabled=True)
    assert len(fr.read_bodies) == 2 and rbid1 in fr.read_bodies
    run = _persist(tmp_path, fr, cache_body=_BODY_OTHER, slot_id=p.slot_id)  # cache != body#1
    res = load_legacy_run(run)
    o = res.obligations[0]
    # body identity verifies against body#1's manifest hash (by read_body_id), regardless of
    # the cache holding a different (body#2) text.
    assert o.read_body_id == rbid1
    assert o.rejudgment_verify_method == "body_id"
    assert o.manifest_body_hash == canonical_read_body_hash(
        _BODY_SUPPORT, basis=BodyHashBasis.STORED_READ_BODY)
    assert consistency_violations(res.metrics) == []


# ------------------------------------------------ 4: strict hash != manifest hash (same id)
def test_strict_hash_differs_from_manifest_same_read_body_id(tmp_path):
    fr, p, judge = _route(tmp_path)
    fr.read_bodies[p.read_body_id]["manifest_body_hash"] = "feedfacefeedface"
    fr.read_bodies[p.read_body_id]["body_hash"] = "feedfacefeedface"
    run = _persist(tmp_path, fr, cache_body=_BODY_SUPPORT, slot_id=p.slot_id)
    res = load_legacy_run(run)
    o = res.obligations[0]
    assert o.read_body_mismatch_class == \
        "strict_entry_hash_differs_from_manifest_same_read_body_id"
    assert o.pipeline_status == "rejudgment_unverifiable"
    assert res.metrics["closed_count"] == 0
    assert consistency_violations(res.metrics) == []


# ------------------------------------------------ 5: strict==manifest but cache differs
def test_strict_matches_manifest_but_cache_differs_still_verifies(tmp_path):
    fr, p, judge = _route(tmp_path)
    # the cache holds a DIFFERENT body than the manifest/strict (passage reproduction fails),
    # but identity (strict == manifest) still verifies the strict verdict.
    run = _persist(tmp_path, fr, cache_body=_BODY_OTHER, slot_id=p.slot_id)
    res = load_legacy_run(run)
    o = res.obligations[0]
    assert o.rejudgment_verify_method == "body_id"          # verdict provenance verified
    assert o.cache_body_repro_status == "cache_body_hash_differs_from_manifest"
    assert o.read_body_mismatch_class != \
        "strict_entry_hash_differs_from_manifest_same_read_body_id"
    assert o.pipeline_status == "closed"
    pm = res.metrics
    assert pm["cache_body_differs_from_manifest_count"] == 1
    assert pm["recorded_rejudgment_body_hash_mismatch_count"] == 0   # NOT a body mismatch
    assert consistency_violations(pm) == []


# ------------------------------------------------ 6: stored vs raw basis mismatch
def test_stored_vs_raw_basis_mismatch_is_classified(tmp_path):
    fr, p, judge = _route(tmp_path)
    fr.read_bodies[p.read_body_id]["body_hash_basis"] = BodyHashBasis.RAW_READ_BODY
    run = _persist(tmp_path, fr, cache_body=_BODY_SUPPORT, slot_id=p.slot_id)
    res = load_legacy_run(run)
    o = res.obligations[0]
    assert o.read_body_mismatch_class == "stored_vs_raw_hash_basis_mismatch"
    assert o.pipeline_status == "rejudgment_unverifiable"
    assert consistency_violations(res.metrics) == []


# ------------------------------------------------ 7: passage-window mismatch != body mismatch
def test_passage_window_mismatch_is_not_body_hash_mismatch(tmp_path):
    fr, p, judge = _route(tmp_path)
    # corrupt the persisted passage window hash (body identity still verifies).
    key = strict_rejudgment_key(p.pending_read_judgment_id)
    entry = judge.cache.get(key)
    entry["passage_window_hashes"] = ["deadbeefdeadbeef"]
    judge.cache.put(key, entry)
    run = _persist(tmp_path, fr, cache_body=_BODY_SUPPORT, slot_id=p.slot_id)
    # re-write the (mutated) judge cache into the run dir so the validator reads it.
    (run / "llm_evidence_judge_cache.json").write_text(
        json.dumps({key: entry}))
    res = load_legacy_run(run)
    o = res.obligations[0]
    assert o.rejudgment_verify_method == "body_id"             # identity still verifies
    assert o.passage_window_status == "mismatch"
    assert o.read_body_mismatch_class == "passage_window_hash_mismatch"
    assert o.pipeline_status == "closed"                       # verdict still recorded
    assert res.metrics["recorded_rejudgment_body_hash_mismatch_count"] == 0
    assert consistency_violations(res.metrics) == []


# ------------------------------------------------ 8: N obligations one body -> 1 read-body
def test_multiple_obligations_one_bad_body_dedupes_to_one_read_body(tmp_path):
    run = tmp_path / "run"
    (run / "cache").mkdir(parents=True)
    judge = deterministic_replay_judge()
    judge.cache = ParserCache(str(run / "llm_evidence_judge_cache.json"))
    fr, _ = _frontier(judge)
    sid = _sid(fr)
    # four candidates/obligations all sharing one URL (one read body).
    for i in range(4):
        cand = SlotCandidate(candidate_id=f"cand{i}", candidate_text=f"Person {i}",
                             normalized_text_hash=_hash(f"person {i}"), inferred_role="person",
                             slot_id=sid, constraints_unknown=["c1"], aliases=[f"Person {i}"],
                             source_urls=[_URL_A])
        fr.slates[sid].candidates[cand.candidate_id] = cand
        fr.candidates_by_id[cand.candidate_id] = cand
        fr._register_pending_read_judgment(candidate_id=f"cand{i}", slot_id=sid,
                                           constraint_id="c1", source_url=_URL_A,
                                           source_role="professional_profile")
    fr.route_read_into_pending_judgments(
        candidate_id="cand0", source_url=_URL_A, read_text=_BODY_SUPPORT, judge=judge,
        read_meta={"tool": "page_fetch", "body_provider": "page_fetch"})
    # one read body serves all four; corrupt that ONE manifest body hash.
    assert len(fr.read_bodies) == 1
    rbid = next(iter(fr.read_bodies))
    fr.read_bodies[rbid]["manifest_body_hash"] = "0badc0de0badc0de"
    fr.read_bodies[rbid]["body_hash"] = "0badc0de0badc0de"
    fr.finalize_pending_service(budget_remaining=0, reading_tools_enabled=True)
    run = _persist(tmp_path, fr, cache_body=_BODY_SUPPORT, slot_id=sid)
    res = load_legacy_run(run)
    mismatched = [o for o in res.obligations
                  if o.read_body_mismatch_class
                  == "strict_entry_hash_differs_from_manifest_same_read_body_id"]
    assert len(mismatched) == 4                                 # 4 obligations
    pm = res.metrics
    assert pm["recorded_rejudgment_body_hash_mismatch_obligation_count"] == 4
    assert pm["recorded_rejudgment_body_hash_mismatch_read_body_count"] == 1   # 1 body
    diags = pm["recorded_rejudgment_read_body_mismatch_diagnostics"]
    assert len(diags) == 1
    assert diags[0]["mismatch_entry_count"] == 4
    assert diags[0]["mismatch_class"] in READ_BODY_MISMATCH_CLASSES
    assert consistency_violations(pm) == []


# ------------------------------------------------ 9: snippets/debug cannot verify a rejudgment
def test_search_snippet_body_cannot_verify_strict_rejudgment(tmp_path):
    # a strict entry exists, but the only cache body for the URL is a SEARCH snippet (never an
    # actual read body) and there is no read-body manifest -> cannot verify.
    run = tmp_path / "run"
    (run / "cache").mkdir(parents=True)
    judge = deterministic_replay_judge()
    judge.cache = ParserCache(str(run / "llm_evidence_judge_cache.json"))
    fr, _ = _frontier(judge)
    _add_candidate(fr)
    p = _add_pending(fr)
    fr.finalize_pending_service(budget_remaining=0, reading_tools_enabled=True)
    oid = p.pending_read_judgment_id
    # a strict entry referencing a read_body_id that the manifest will NOT contain.
    (run / "cache").mkdir(parents=True, exist_ok=True)
    cache = {"mode": "auto", "store_raw": False, "entries": [
        {"request_hash": "h1", "provider": "serper_search", "name": "serper_search",
         "request_meta": {"query": "northvale"},
         "response": {"provider": "serper_search", "query": "northvale",
                      "results": [{"title": "x", "url": _URL_A,
                                   "snippet": _BODY_SUPPORT, "rank": 0}], "error": None}}]}
    (run / "cache" / "provider_cache.json").write_text(json.dumps(cache))
    record = {
        "item_id": "v1-snip", "question_preview": "Who studied at Northvale in 1987?",
        "task_frame": {"target_answer_slots": [{"slot_id": p.slot_id, "slot_name": "engineer",
                                                "descriptor": "engineer"}],
                       "constraints": [{"constraint_id": "c1",
                                        "text_span": "studied at Northvale in 1987",
                                        "normalized_terms": ["northvale", "1987"],
                                        "applies_to": [p.slot_id]}]},
        "candidate_frontier": {
            "events": [], "events_truncated": 0, "interpretations": [],
            "pending_read_judgments": fr.to_debug()["pending_read_judgments"],
            "read_bodies": [], "evidence_gaps": [],
            "metrics": {"unrelated_read_executed_while_pending_count": 0},
            "slates": [{"slot_role": "person",
                        "top_candidates": [{"candidate_text_preview": "Avery Quinn"}]}]},
        "calls": [], "evidence": []}
    (run / "debug_questions.jsonl").write_text(json.dumps(record) + "\n")
    (run / "llm_evidence_judge_cache.json").write_text(json.dumps({
        strict_rejudgment_key(oid): {
            "version": STRICT_REJUDGE_VERSION, "verdict": "full_support",
            "read_body_id": "rb_doesnotexist", "strict_entry_body_hash": "abc123",
            "body_hash_basis": BodyHashBasis.STORED_READ_BODY,
            "slot_id": p.slot_id, "constraint_id": "c1"}}))
    res = load_legacy_run(run, allow_live_judge=True, max_judge_calls=20)
    o = res.obligations[0]
    assert not o.body_is_actual_read_body                       # search snippet, never a body
    assert o.pipeline_status not in ("closed", "judged_unclosed", "source_terminal")
    assert res.metrics["live_model_calls"] == 0
    assert res.metrics["closed_count"] == 0
    assert consistency_violations(res.metrics) == []


# ------------------------------------------------ 10: requires_read never increments closed
def test_requires_read_still_open_never_closes(tmp_path):
    run = tmp_path / "run"
    (run / "cache").mkdir(parents=True)
    judge = deterministic_replay_judge()
    judge.cache = ParserCache(str(run / "llm_evidence_judge_cache.json"))
    fr, _ = _frontier(judge)
    _add_candidate(fr)
    # body has the constraint terms but NOT the candidate -> stub judge -> requires_read.
    body = (("filler. " * 60) + " The Northvale Institute opened a 1987 archive page. ")
    p = _add_pending(fr)
    fr.route_read_into_pending_judgments(
        candidate_id="cand1", source_url=_URL_A, read_text=body, judge=judge,
        read_meta={"tool": "page_fetch", "body_provider": "page_fetch"})
    fr.finalize_pending_service(budget_remaining=0, reading_tools_enabled=True)
    run = _persist(tmp_path, fr, cache_body=body, slot_id=p.slot_id)
    res = load_legacy_run(run)
    o = res.obligations[0]
    assert o.closure_code == "requires_read_still_open"
    assert o.pipeline_status == "judged_unclosed"
    assert res.metrics["closed_count"] == 0
    assert res.metrics["requires_read_still_open_counted_closed_count"] == 0
    assert consistency_violations(res.metrics) == []
