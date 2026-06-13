"""Level 5v offline tests: exact read-body provenance + terminal rejudgment semantics.

A recorded targeted rejudgment is verified PRIMARILY by read_body_id + body_hash against the
run's persisted read-body manifest (URL matching is only a fallback when no read_body_id
exists). A body-hash mismatch is an explicit UNVERIFIABLE outcome, never silently missing.
resolved_irrelevant / partial are SOURCE-TERMINAL non-support (not still-open, not
answer-supporting) and emit a generic evidence gap; requires_read_still_open is the only
still-open outcome. Strict gate, contamination handling, and the answer gate are unchanged.
Synthetic fixtures only — zero live provider/model calls, no gold/benchmark content.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from regimes_probe.agent.candidate_frontier import CandidateFrontier, SlotCandidate, _hash
from regimes_probe.agent.evidence_interpreter import EvidenceInterpreter
from regimes_probe.agent.llm_task_frame import LLMTaskFrameParser, ParserCache, build_task_frame
from regimes_probe.agent.read_judgment import (
    STRICT_REJUDGE_VERSION, body_hash, strict_rejudgment_key)
from regimes_probe.eval.replay_validation import (
    consistency_violations, deterministic_replay_judge, load_legacy_run)

ROOT = Path(__file__).resolve().parents[1]
_URL_A = "https://profiles.example/people/alpha-profile-page-with-the-relevant-biography"
_BODY_SUPPORT = (("page boilerplate filler text. " * 40)
                 + " Avery Quinn studied at the Northvale Institute in 1987. ")
#: predicate-relevant (constraint terms present) but the judge stub will call it irrelevant.
_BODY_PREDICATE = (("page boilerplate filler text. " * 40)
                   + " The Northvale Institute in 1987 had many engineering students. ")


def _frame_payload(n_constraints=1):
    cons = [{"constraint_id": "c1", "text_span": "studied at Northvale in 1987",
             "normalized_terms": ["northvale", "1987"], "applies_to": ["D"],
             "discriminative_score": 2.0, "specificity_score": 2.0, "required": True,
             "priority": "high", "blocks_answer_if_unresolved": True, "testable_claim": "x",
             "how_to_test": "read", "semantic_label": "education"}]
    if n_constraints >= 2:
        cons.append({"constraint_id": "c2", "text_span": "later worked in Valdorra",
                     "normalized_terms": ["valdorra", "worked"], "applies_to": ["D"],
                     "discriminative_score": 2.0, "specificity_score": 2.0, "required": True,
                     "priority": "high", "blocks_answer_if_unresolved": True,
                     "testable_claim": "y", "how_to_test": "read", "semantic_label": "career"})
    return {"target_answer_slots": [{"slot_id": "D", "slot_name": "engineer",
                                     "slot_role": "person", "is_target_answer_slot": True,
                                     "is_intermediate_slot": False}],
            "latent_slots": [], "constraints": cons,
            "dependency_edges": [], "known_context_terms": []}


def _frontier(judge=None, n_constraints=1):
    parser = LLMTaskFrameParser(model_fn=lambda _p: json.dumps(_frame_payload(n_constraints)),
                                model="stub")
    frame, _ = build_task_frame("x", "Who is the engineer who studied at Northvale in 1987?",
                                use_llm=True, parser=parser)
    judge = judge or deterministic_replay_judge()
    fr = CandidateFrontier(frame, interpreter=EvidenceInterpreter(judge=judge))
    return fr, judge


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


def _add_pending(fr, *, url=_URL_A, candidate_id="cand1", constraint_id="c1",
                 source_role="professional_profile", contaminated=False):
    fr._register_pending_read_judgment(
        candidate_id=candidate_id, slot_id=_sid(fr), constraint_id=constraint_id,
        source_url=url, source_role=source_role, contaminated=contaminated)
    return next(p for p in fr.pending_read_judgments.values()
                if p.candidate_id == candidate_id and p.constraint_id == constraint_id)


class _VerdictJudge:
    """A judge stub returning a FIXED verdict (to exercise terminal/still-open lifecycles)."""
    enabled = True

    def __init__(self, verdict):
        self.verdict = verdict
        self.cache = ParserCache()

    def judge(self, **kw):
        return SimpleNamespace(judgment=self.verdict)

    def stats(self):
        return {}


def _persist(tmp_path, fr, body, slot_id, *, cache_body=None, manifest=True) -> Path:
    """Persist a frontier's to_debug() into a run dir with a provider cache (mirrors a real
    run). ``manifest`` toggles whether read_bodies is written (to test the URL fallback)."""
    run = tmp_path / "run"
    (run / "cache").mkdir(parents=True, exist_ok=True)
    cache = {"mode": "auto", "store_raw": False, "entries": [
        {"request_hash": "h1", "provider": "page_fetch", "name": "page_fetch",
         "request_meta": {"query": _URL_A},
         "response": {"provider": "page_fetch", "query": _URL_A,
                      "results": [{"title": "Avery Quinn", "url": _URL_A,
                                   "snippet": cache_body if cache_body is not None else body,
                                   "rank": 0}], "error": None}}]}
    (run / "cache" / "provider_cache.json").write_text(json.dumps(cache))
    dbg = fr.to_debug()
    record = {
        "item_id": "v-001",
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
            "pending_read_judgments": dbg["pending_read_judgments"],
            "pending_service_urls": dbg["pending_service_urls"],
            "read_bodies": dbg.get("read_bodies", []) if manifest else [],
            "evidence_gaps": dbg.get("evidence_gaps", []),
            "metrics": {"unrelated_read_executed_while_pending_count": 0},
            "slates": [{"slot_role": "person",
                        "top_candidates": [{"candidate_text_preview": "Avery Quinn"}]}]},
        "calls": [], "evidence": []}
    (run / "debug_questions.jsonl").write_text(json.dumps(record) + "\n")
    if not (run / "llm_evidence_judge_cache.json").exists():
        (run / "llm_evidence_judge_cache.json").write_text("{}")
    return run


# ----------------------------------------------- 1: verify by exact read_body_id + body_hash
def test_recorded_rejudgment_verified_by_read_body_id(tmp_path):
    judge = deterministic_replay_judge()
    run = tmp_path / "run"
    (run / "cache").mkdir(parents=True)
    judge.cache = ParserCache(str(run / "llm_evidence_judge_cache.json"))
    fr, _ = _frontier(judge=judge)
    _add_candidate(fr)
    p = _add_pending(fr)
    fr.route_read_into_pending_judgments(
        candidate_id="cand1", source_url=_URL_A, read_text=_BODY_SUPPORT, judge=judge,
        read_meta={"tool": "page_fetch", "body_provider": "page_fetch"})
    # the obligation references an exact read_body_id; the strict entry carries it + the hash.
    assert p.read_body_id and p.read_body_id in fr.read_bodies
    entry = judge.cache.get(strict_rejudgment_key(p.pending_read_judgment_id))
    assert entry["read_body_id"] == p.read_body_id
    assert entry["judged_body_hash"] == body_hash(_BODY_SUPPORT)
    fr.finalize_pending_service(budget_remaining=0, reading_tools_enabled=True)
    run = _persist(tmp_path, fr, _BODY_SUPPORT, p.slot_id)
    res = load_legacy_run(run)
    o = res.obligations[0]
    assert o.rejudgment_verify_method == "body_id"
    assert o.closure_code == "resolved_full_support"
    assert o.pipeline_status == "closed"
    pm = res.metrics
    assert pm["recorded_rejudgment_verified_by_body_id_count"] == 1
    assert pm["recorded_rejudgment_verified_by_url_fallback_count"] == 0
    assert pm["recorded_rejudgment_body_hash_mismatch_count"] == 0
    assert pm["recorded_rejudgment_unverifiable_count"] == 0
    assert pm["closed_count"] == 1
    assert consistency_violations(pm) == []


# ----------------------------------------------- 2: URL fallback only when no read_body_id
def test_url_fallback_only_when_read_body_id_absent(tmp_path):
    # a LEGACY strict entry (no read_body_id) with a correct body hash + cache_read_body
    # provenance must still verify — via the URL/hash fallback, not the manifest.
    run = tmp_path / "run"
    (run / "cache").mkdir(parents=True)
    fr, _ = _frontier()
    _add_candidate(fr)
    p = _add_pending(fr)
    # mirror a LEGACY (pre-5v) run that attempted + recorded a verdict without a read_body_id.
    p.rejudgment_attempted = True
    p.rejudgment_recorded = True
    p.body_available = True
    fr.finalize_pending_service(budget_remaining=0, reading_tools_enabled=True)
    run = _persist(tmp_path, fr, _BODY_SUPPORT, p.slot_id, manifest=False)
    oid = p.pending_read_judgment_id
    (run / "llm_evidence_judge_cache.json").write_text(json.dumps({
        strict_rejudgment_key(oid): {
            "version": STRICT_REJUDGE_VERSION, "verdict": "full_support",
            "body_source": "cache_read_body", "body_provider": "page_fetch",
            "body_hash": body_hash(_BODY_SUPPORT),
            "slot_id": p.slot_id, "constraint_id": "c1"}}))   # NO read_body_id
    res = load_legacy_run(run)
    o = next(x for x in res.obligations if x.pending_read_judgment_id == oid)
    assert o.rejudgment_verify_method == "url_fallback"
    assert o.closure_code == "resolved_full_support" and o.pipeline_status == "closed"
    pm = res.metrics
    assert pm["recorded_rejudgment_verified_by_url_fallback_count"] == 1
    assert pm["recorded_rejudgment_verified_by_body_id_count"] == 0
    assert consistency_violations(pm) == []


# ----------------------------------------------- 3: body-hash mismatch -> unverifiable
def test_body_hash_mismatch_is_unverifiable_not_missing(tmp_path):
    judge = deterministic_replay_judge()
    run = tmp_path / "run"
    (run / "cache").mkdir(parents=True)
    judge.cache = ParserCache(str(run / "llm_evidence_judge_cache.json"))
    fr, _ = _frontier(judge=judge)
    _add_candidate(fr)
    p = _add_pending(fr)
    fr.route_read_into_pending_judgments(
        candidate_id="cand1", source_url=_URL_A, read_text=_BODY_SUPPORT, judge=judge,
        read_meta={"tool": "page_fetch", "body_provider": "page_fetch"})
    # CORRUPT the manifest body hash so id-verification fails (genuine replacement case).
    fr.read_bodies[p.read_body_id]["body_hash"] = "deadbeefdeadbeef"
    fr.finalize_pending_service(budget_remaining=0, reading_tools_enabled=True)
    run = _persist(tmp_path, fr, _BODY_SUPPORT, p.slot_id)
    res = load_legacy_run(run)
    o = res.obligations[0]
    assert o.pipeline_status == "rejudgment_unverifiable"
    assert o.rejudgment_unverifiable_reason == "body_hash_mismatch"
    assert o.rejudgment_status == "rejudgment_skipped_body_hash_mismatch"
    pm = res.metrics
    assert pm["recorded_rejudgment_body_hash_mismatch_count"] == 1
    assert pm["recorded_rejudgment_unverifiable_count"] == 1
    assert pm["pending_read_targeted_rejudgment_missing_count"] == 0   # NOT "missing"
    assert pm["closed_count"] == 0
    assert pm["recorded_rejudgment_body_mismatch_samples"]
    assert consistency_violations(pm) == []


def test_read_body_id_present_but_manifest_missing_is_body_not_found(tmp_path):
    judge = deterministic_replay_judge()
    run = tmp_path / "run"
    (run / "cache").mkdir(parents=True)
    judge.cache = ParserCache(str(run / "llm_evidence_judge_cache.json"))
    fr, _ = _frontier(judge=judge)
    _add_candidate(fr)
    p = _add_pending(fr)
    fr.route_read_into_pending_judgments(
        candidate_id="cand1", source_url=_URL_A, read_text=_BODY_SUPPORT, judge=judge,
        read_meta={"tool": "page_fetch", "body_provider": "page_fetch"})
    fr.finalize_pending_service(budget_remaining=0, reading_tools_enabled=True)
    # persist WITHOUT the read_bodies manifest -> the entry's read_body_id is unresolvable.
    run = _persist(tmp_path, fr, _BODY_SUPPORT, p.slot_id, manifest=False)
    res = load_legacy_run(run)
    o = res.obligations[0]
    assert o.rejudgment_unverifiable_reason == "body_not_found"
    assert o.rejudgment_status == "rejudgment_skipped_body_not_found"
    assert res.metrics["recorded_rejudgment_body_not_found_count"] == 1
    assert res.metrics["pending_read_targeted_rejudgment_missing_count"] == 0
    assert consistency_violations(res.metrics) == []


# ----------------------------------------------- 4/5/6/7: terminal non-support semantics
def test_resolved_irrelevant_is_terminal_non_support_not_still_open(tmp_path):
    judge = _VerdictJudge("irrelevant")
    run = tmp_path / "run"
    (run / "cache").mkdir(parents=True)
    judge.cache = ParserCache(str(run / "llm_evidence_judge_cache.json"))
    fr, _ = _frontier(judge=judge)
    cand = _add_candidate(fr)
    p = _add_pending(fr)
    fr.route_read_into_pending_judgments(
        candidate_id="cand1", source_url=_URL_A, read_text=_BODY_PREDICATE, judge=judge,
        read_meta={"tool": "page_fetch", "body_provider": "page_fetch"})
    # live frontier: terminal non-support, an evidence gap, and NO constraint support.
    assert p.rejudgment_outcome == "source_terminal_non_support"
    assert p.closure_state == "resolved_irrelevant"
    assert p.evidence_gap_reason == "source_irrelevant"
    assert "c1" not in cand.constraints_supported
    assert fr.rejudgment_terminal_non_support_count == 1
    assert fr.rejudgment_requires_read_still_open_count == 0
    assert any(g["gap_reason"] == "source_irrelevant" for g in fr.evidence_gaps)
    fr.finalize_pending_service(budget_remaining=0, reading_tools_enabled=True)
    run = _persist(tmp_path, fr, _BODY_PREDICATE, p.slot_id)
    res = load_legacy_run(run)
    o = res.obligations[0]
    assert o.pipeline_status == "source_terminal"
    assert o.closure_code == "resolved_irrelevant"
    assert o.stage_reason == "judged_source_irrelevant_terminal"
    pm = res.metrics
    # the irrelevant verdict is NOT counted as still-open and NOT as closed.
    assert pm["rejudgment_terminal_non_support_count"] == 1
    assert pm["rejudgment_requires_read_still_open_count"] == 0
    assert pm["pending_read_targeted_rejudgment_still_open_count"] == 0
    assert pm["closed_count"] == 0
    assert pm["source_terminal_counted_still_open_count"] == 0
    assert pm["source_terminal_counted_closed_count"] == 0
    assert pm["evidence_gap_count"] >= 1
    assert "source_irrelevant" in pm["evidence_gap_reason_counts"]
    assert res.overall_status == "source_terminal_non_support"
    assert consistency_violations(pm) == []


def test_requires_read_is_the_only_still_open_outcome(tmp_path):
    judge = _VerdictJudge("requires_read")
    run = tmp_path / "run"
    (run / "cache").mkdir(parents=True)
    judge.cache = ParserCache(str(run / "llm_evidence_judge_cache.json"))
    fr, _ = _frontier(judge=judge)
    _add_candidate(fr)
    p = _add_pending(fr)
    fr.route_read_into_pending_judgments(
        candidate_id="cand1", source_url=_URL_A, read_text=_BODY_PREDICATE, judge=judge,
        read_meta={"tool": "page_fetch", "body_provider": "page_fetch"})
    assert p.rejudgment_outcome == "still_requires_more_evidence"
    assert p.evidence_gap_reason == "requires_more_evidence"
    fr.finalize_pending_service(budget_remaining=0, reading_tools_enabled=True)
    run = _persist(tmp_path, fr, _BODY_PREDICATE, p.slot_id)
    res = load_legacy_run(run)
    o = res.obligations[0]
    assert o.pipeline_status == "judged_unclosed"
    assert o.closure_code == "requires_read_still_open"
    assert o.stage_reason == "judged_requires_more_evidence"
    pm = res.metrics
    assert pm["rejudgment_requires_read_still_open_count"] == 1
    assert pm["pending_read_targeted_rejudgment_still_open_count"] == 1
    assert pm["closed_count"] == 0
    assert consistency_violations(pm) == []


def test_terminal_non_support_emits_evidence_gap_and_answer_gate_false(tmp_path):
    from regimes_probe.agent.action_planner import evaluate_answer_support
    judge = _VerdictJudge("irrelevant")
    fr, _ = _frontier(judge=judge, n_constraints=2)
    cand = _add_candidate(fr)
    p = _add_pending(fr)
    fr.route_read_into_pending_judgments(
        candidate_id="cand1", source_url=_URL_A, read_text=_BODY_PREDICATE, judge=judge)
    # an evidence gap exists, the candidate is not confirmed, and no constraint is supported.
    assert fr.evidence_gaps and fr.evidence_gaps[0]["gap_reason"] == "source_irrelevant"
    assert fr.evidence_gaps[0]["candidate_remains_viable"] is True
    assert fr.evidence_gaps[0]["constraint_remains_blocking"] is True
    assert cand.status != "confirmed"
    # build a hypothesis table for the gate check — terminal non-support never supports.
    from regimes_probe.agent.hypothesis_table import HypothesisTable
    support = evaluate_answer_support(fr.frame, HypothesisTable(fr.frame))
    assert not (support and support.supported)
    m = fr.metrics()
    assert m["confirmed_hypothesis_with_unresolved_blocking_count"] == 0
    assert m["evidence_gap_count"] >= 1


# ----------------------------------------------- 8: snippets/debug never judged
def test_search_snippet_only_body_is_never_judged(tmp_path):
    run = tmp_path / "run"
    (run / "cache").mkdir(parents=True)
    # a SEARCH-provider snippet for the obligation URL (never an actual read body).
    cache = {"mode": "auto", "store_raw": False, "entries": [
        {"request_hash": "h1", "provider": "serper_search", "name": "serper_search",
         "request_meta": {"query": "northvale"},
         "response": {"provider": "serper_search", "query": "northvale",
                      "results": [{"title": "Avery Quinn", "url": _URL_A,
                                   "snippet": _BODY_SUPPORT, "rank": 0}], "error": None}}]}
    (run / "cache" / "provider_cache.json").write_text(json.dumps(cache))
    fr, _ = _frontier()
    _add_candidate(fr)
    p = _add_pending(fr)
    fr.finalize_pending_service(budget_remaining=0, reading_tools_enabled=True)
    dbg = fr.to_debug()
    record = {
        "item_id": "v-snip",
        "question_preview": "Who is the engineer who studied at the Northvale Institute in 1987?",
        "task_frame": {
            "target_answer_slots": [{"slot_id": p.slot_id, "slot_name": "engineer",
                                     "descriptor": "engineer"}],
            "constraints": [{"constraint_id": "c1",
                             "text_span": "studied at the Northvale Institute in 1987",
                             "normalized_terms": ["northvale", "institute", "1987"],
                             "applies_to": [p.slot_id]}]},
        "candidate_frontier": {
            "events": dbg["events"], "events_truncated": 0, "interpretations": [],
            "pending_read_judgments": dbg["pending_read_judgments"],
            "pending_service_urls": dbg["pending_service_urls"],
            "read_bodies": dbg.get("read_bodies", []),
            "evidence_gaps": dbg.get("evidence_gaps", []),
            "metrics": {"unrelated_read_executed_while_pending_count": 0},
            "slates": [{"slot_role": "person",
                        "top_candidates": [{"candidate_text_preview": "Avery Quinn"}]}]},
        "calls": [], "evidence": []}
    (run / "debug_questions.jsonl").write_text(json.dumps(record) + "\n")
    (run / "llm_evidence_judge_cache.json").write_text("{}")
    res = load_legacy_run(run, allow_live_judge=True, max_judge_calls=20)
    o = res.obligations[0]
    assert o.body_is_search_snippet and not o.body_is_actual_read_body
    assert o.pipeline_status not in ("closed", "judged_unclosed", "source_terminal")
    assert res.metrics["live_model_calls"] == 0
    assert res.metrics["closed_count"] == 0
    assert res.metrics["debug_snippet_judged_count"] == 0
    assert consistency_violations(res.metrics) == []


# ----------------------------------------------- 9: contamination cannot produce support
def test_contaminated_source_cannot_produce_support_or_gap_search():
    judge = _VerdictJudge("full_support")
    fr, _ = _frontier(judge=judge)
    cand = _add_candidate(fr)
    p = _add_pending(fr, contaminated=True)
    # a contaminated requires_read is SUPPRESSED (non-executable) and never routed/judged.
    assert p.suppressed_reason and not p.open
    n = fr.route_read_into_pending_judgments(
        candidate_id="cand1", source_url=_URL_A, read_text=_BODY_SUPPORT, judge=judge)
    assert n == 0
    assert "c1" not in cand.constraints_supported
    fr.finalize_pending_service(budget_remaining=4, reading_tools_enabled=True)
    assert p.service_status == "service_suppressed_non_executable"
    m = fr.metrics()
    assert m["pending_read_targeted_rejudgment_closed_count"] == 0
    assert m["rejudgment_constraint_resolved_count"] == 0


# ----------------------------------------------- 10: full-fixture consistency + zero calls
def test_full_fixture_consistency_and_zero_live_calls(tmp_path):
    judge = deterministic_replay_judge()
    run = tmp_path / "run"
    (run / "cache").mkdir(parents=True)
    judge.cache = ParserCache(str(run / "llm_evidence_judge_cache.json"))
    fr, _ = _frontier(judge=judge, n_constraints=2)
    _add_candidate(fr)
    served = _add_pending(fr, url=_URL_A, candidate_id="cand1", constraint_id="c1")
    fr.route_read_into_pending_judgments(
        candidate_id="cand1", source_url=_URL_A, read_text=_BODY_SUPPORT, judge=judge,
        read_meta={"tool": "page_fetch", "body_provider": "page_fetch"})
    fr.finalize_pending_service(budget_remaining=2, reading_tools_enabled=True)
    run = _persist(tmp_path, fr, _BODY_SUPPORT, served.slot_id)
    res = load_legacy_run(run)
    pm = res.metrics
    assert pm["live_provider_calls"] == 0 and pm["live_model_calls"] == 0
    assert isinstance(pm["consistency_violations"], list)
    assert pm["consistency_violations"] == []
    # the body-id verification is the primary path; no hash mismatch on an honest run.
    assert pm["recorded_rejudgment_verified_by_body_id_count"] == 1
    assert pm["recorded_rejudgment_body_hash_mismatch_count"] == 0
    assert pm["read_body_manifest_count"] == 1
