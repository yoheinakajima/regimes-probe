"""Level 5t offline tests: first-class, budget-aware, auditable pending-read servicing +
post-read targeted rejudgment persistence. The pending-service step governs action selection
before any frontier search/verify/read; one read serves every obligation sharing a URL; a
failed pending read falls back ONCE to the alternate tool on the SAME URL; a successful read
triggers a targeted rejudgment persisted under the strict replay namespace so default replay
validation finds it offline; every pending ends the item with an explicit service reason.
Synthetic fixtures only — zero live provider/model calls, no benchmark/gold content.
"""

from __future__ import annotations

import json
from pathlib import Path

from regimes_probe.agent.candidate_frontier import CandidateFrontier, SlotCandidate, _hash
from regimes_probe.agent.evidence_interpreter import EvidenceInterpreter
from regimes_probe.agent.llm_task_frame import LLMTaskFrameParser, ParserCache, build_task_frame
from regimes_probe.agent.read_judgment import STRICT_REJUDGE_VERSION, strict_rejudgment_key
from regimes_probe.eval.replay_validation import (
    consistency_violations, deterministic_replay_judge, load_legacy_run)

ROOT = Path(__file__).resolve().parents[1]
_URL_A = "https://profiles.example/people/alpha-profile-page-with-the-relevant-biography"
_URL_C = "https://another.example/generic/reference/page-not-tied-to-any-obligation"
_BODY_SUPPORT = (("page boilerplate filler text. " * 40)
                 + " Avery Quinn studied at the Northvale Institute in 1987. ")
_BODY_SUBJECT_ONLY = (("page boilerplate filler text. " * 40)
                      + " Avery Quinn personal homepage with contact links. ")


def _frame_payload(n_constraints: int = 1):
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
                     "testable_claim": "y", "how_to_test": "read",
                     "semantic_label": "career"})
    return {
        "target_answer_slots": [{"slot_id": "D", "slot_name": "engineer",
                                 "slot_role": "person", "is_target_answer_slot": True,
                                 "is_intermediate_slot": False}],
        "latent_slots": [], "constraints": cons,
        "dependency_edges": [], "known_context_terms": []}


def _frontier(n_constraints: int = 1, judge=None):
    parser = LLMTaskFrameParser(
        model_fn=lambda _p: json.dumps(_frame_payload(n_constraints)), model="stub")
    frame, _ = build_task_frame("x", "Who is the engineer who studied at Northvale in 1987?",
                                use_llm=True, parser=parser)
    judge = judge or deterministic_replay_judge()
    fr = CandidateFrontier(frame, interpreter=EvidenceInterpreter(judge=judge))
    return fr, frame, judge


def _sid(fr) -> str:
    return next(iter(fr.slates))


def _add_candidate(fr, text="Avery Quinn", cid="cand1"):
    sid = _sid(fr)
    cand = SlotCandidate(candidate_id=cid, candidate_text=text,
                         normalized_text_hash=_hash(text.lower()), inferred_role="person",
                         slot_id=sid, constraints_unknown=["c1"],
                         aliases=[text], source_urls=[_URL_A])
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


# ------------------------------------------------ 1: service step governs action selection
def test_pending_service_read_selected_before_any_frontier_action():
    fr, _, _ = _frontier()
    _add_pending(fr)
    assert not fr.candidates_by_id            # no candidate/slot state is required (5t-1)
    sp = fr.propose_pending_service_read(budget_remaining=4, page_fetch_available=True)
    assert sp is not None and sp.kind == "read"
    assert getattr(sp.read_obs, "url", "") == _URL_A
    assert sp.reason == "pending_obligation_service"
    assert sp.action_type == "read_candidate_source"
    assert sp.query == ""                     # a concrete URL read, never a search translation
    m = fr.metrics()
    assert m["pending_service_url_count"] == 1
    assert m["pending_service_url_attempted_count"] == 1
    # the same step is what propose_step_action returns first for a fresh frontier.
    fr2, _, _ = _frontier()
    _add_pending(fr2)
    sp2 = fr2.propose_step_action(observations=[], budget_remaining=4, reading_tools=True,
                                  page_fetch_available=True)
    assert sp2.kind == "read" and sp2.reason == "pending_obligation_service"


# ------------------------------------------------ 2: URL dedupe — one read serves all
def test_one_read_serves_all_obligations_sharing_url():
    fr, _, _ = _frontier(n_constraints=2)
    p1 = _add_pending(fr, constraint_id="c1", candidate_id="cand1")
    p2 = _add_pending(fr, constraint_id="c2", candidate_id="cand2")
    sp = fr.propose_pending_service_read(budget_remaining=4, page_fetch_available=True)
    assert sp is not None and sorted(sp.constraint_ids) == ["c1", "c2"]
    fr.record_pending_service_outcome(url=_URL_A, tool="page_fetch",
                                      page_fetch_available=True)
    assert p1.body_available and p2.body_available
    m = fr.metrics()
    assert m["pending_service_url_count"] == 1
    assert m["pending_service_url_success_count"] == 1
    assert m["pending_service_obligations_served_by_successful_read_count"] == 2
    assert m["pending_service_obligation_success_rate"] == 1.0
    # the URL is never re-read by the service step.
    assert fr.propose_pending_service_read(budget_remaining=4,
                                           page_fetch_available=True) is None


# ------------------------------------------------ 3: unrelated reads stay hard-blocked
def test_unrelated_read_hard_blocked_while_pending_open():
    from regimes_probe.agent.reading_policy import normalize_url
    fr, frame, _ = _frontier()
    sid = frame.all_slots[0].slot_id
    _add_pending(fr)
    cand = SlotCandidate(candidate_id="cand_rival", candidate_text="Rival Person",
                         normalized_text_hash=_hash("rival person"), inferred_role="person",
                         slot_id=sid, source_domains=["another.example"],
                         constraints_unknown=["c1"], source_urls=[_URL_C])
    fr.slates[sid].candidates[cand.candidate_id] = cand
    fr.candidates_by_id[cand.candidate_id] = cand
    fr._slot_no_support_streak[sid] = 5
    scraped = frozenset({normalize_url(_URL_A)})   # pending URL already read once
    sp = fr.propose_step_action(observations=[], budget_remaining=4, reading_tools=True,
                                page_fetch_available=True, scraped_urls=scraped)
    assert sp.kind == "unexecutable"
    assert sp.reason == "read_blocked_unrelated_to_pending_obligation"
    m = fr.metrics()
    assert m["unrelated_read_executed_while_pending_count"] == 0     # pinned by construction
    assert m["read_blocked_unrelated_to_pending_obligation_count"] == 1


# ------------------------------------------------ 4: bounded same-URL fallback
def test_failed_pending_read_falls_back_once_to_alternate_tool_same_url():
    fr, _, _ = _frontier()
    _add_pending(fr)
    sp = fr.propose_pending_service_read(budget_remaining=6, page_fetch_available=True,
                                         scrape_available=True)
    t1 = sp.read_decision.tool
    fr.record_pending_service_outcome(url=_URL_A, tool=t1, failed=True,
                                      page_fetch_available=True, scrape_available=True)
    assert fr.pending_read_primary_failed_count == 1
    sp2 = fr.propose_pending_service_read(budget_remaining=5, page_fetch_available=True,
                                          scrape_available=True)
    assert sp2 is not None and getattr(sp2.read_obs, "url", "") == _URL_A   # SAME url
    assert sp2.read_decision.tool != t1                                     # alternate tool
    assert sp2.reason == "pending_obligation_service_fallback"
    assert fr.pending_read_fallback_attempted_count == 1
    fr.record_pending_service_outcome(url=_URL_A, tool=sp2.read_decision.tool,
                                      zero_chars=True, page_fetch_available=True,
                                      scrape_available=True)
    assert fr.pending_read_fallback_failed_count == 1
    assert fr.pending_read_zero_chars_count == 1
    # both tools tried: no third attempt, and the URL record carries the failure status.
    assert fr.propose_pending_service_read(budget_remaining=4, page_fetch_available=True,
                                           scrape_available=True) is None
    rec = fr._service_rec(_URL_A)
    assert rec["status"] == "service_attempted_failed" and rec["zero_chars"]


# ------------------------------------------------ 5: successful read -> persisted rejudgment
def test_successful_read_triggers_and_persists_strict_rejudgment():
    fr, _, judge = _frontier()
    _add_candidate(fr)
    p = _add_pending(fr)
    n = fr.route_read_into_pending_judgments(candidate_id="cand1", source_url=_URL_A,
                                             read_text=_BODY_SUPPORT, judge=judge)
    assert n == 1
    rec = judge.cache.get(strict_rejudgment_key(p.pending_read_judgment_id))
    assert isinstance(rec, dict)
    assert rec["version"] == STRICT_REJUDGE_VERSION
    assert rec["verdict"] == "full_support"
    assert rec["body_source"] == "live_run_read_body"
    assert rec["constraint_id"] == "c1" and rec["slot_id"] == p.slot_id
    assert p.rejudgment_attempted and p.rejudgment_recorded
    assert p.closure_state == "resolved_full_support"
    assert p.body_available and p.passage_relevance == "predicate_relevant"
    m = fr.metrics()
    assert m["pending_read_targeted_rejudgment_attempted_count"] == 1
    assert m["pending_read_targeted_rejudgment_recorded_count"] == 1
    assert m["pending_read_targeted_rejudgment_closed_count"] == 1


# ------------------------------------------------ 6: requires_read stays open, never closed
def test_requires_read_verdict_recorded_but_never_closed():
    fr, _, judge = _frontier()
    _add_candidate(fr)
    p = _add_pending(fr)
    n = fr.route_read_into_pending_judgments(candidate_id="cand1", source_url=_URL_A,
                                             read_text=_BODY_SUBJECT_ONLY, judge=judge)
    assert n == 0                                        # judged, not resolved
    assert p.closure_state == "requires_read_still_open"
    rec = judge.cache.get(strict_rejudgment_key(p.pending_read_judgment_id))
    assert isinstance(rec, dict) and rec["verdict"] == "requires_read"
    m = fr.metrics()
    assert m["pending_read_targeted_rejudgment_still_open_count"] == 1
    assert m["pending_read_targeted_rejudgment_closed_count"] == 0


# ------------------------------------------------ 7: closure never weakens the answer gate
def test_full_support_closes_obligation_without_weakening_answer_gate():
    fr, _, judge = _frontier(n_constraints=2)
    cand = _add_candidate(fr)
    p = _add_pending(fr, constraint_id="c1")
    fr.route_read_into_pending_judgments(candidate_id="cand1", source_url=_URL_A,
                                         read_text=_BODY_SUPPORT, judge=judge)
    assert p.closure_state == "resolved_full_support"
    assert "c1" in cand.constraints_supported
    # c2 (blocking) is still unsupported: the candidate is NOT confirmed and no
    # answer action becomes selectable from this single closure.
    assert cand.status != "confirmed"
    fr.generate_frontier_actions()
    sel = fr.select_frontier_action(budget_remaining=4)
    assert sel is None or sel.action_type != "answer_from_confirmed_hypothesis"
    assert fr.metrics()["confirmed_hypothesis_with_unresolved_blocking_count"] == 0


# ------------------------------------------------ 8: no predicate passage -> reread planning
def test_truncated_subject_only_body_plans_bounded_reread_with_explicit_state():
    fr, _, judge = _frontier()
    _add_candidate(fr)
    p = _add_pending(fr)
    fr.route_read_into_pending_judgments(candidate_id="cand1", source_url=_URL_A,
                                         read_text=_BODY_SUBJECT_ONLY, judge=judge)
    assert p.passage_relevance in ("subject_only", "no_relevant_anchor")
    plans = fr.schedule_predicate_rereads_after_read(
        url=_URL_A, body_truncated_for_storage=True, raw_unavailable=True)
    assert len(plans) == 1
    assert p.closure_state == "body_truncated_before_relevant_passage(raw_unavailable)"
    assert p.reread_pending
    assert fr.predicate_reread_scheduled_count == 1
    # the queued re-read executes as a service read with the bounded higher cap.
    sp = fr.propose_pending_service_read(budget_remaining=4, page_fetch_available=True)
    assert sp is not None and sp.reason == "predicate_reread"
    assert sp.read_opts.get("max_chars") == fr.read_config.reread_max_chars
    assert fr.predicate_reread_attempted_count == 1
    # at most once: a second schedule attempt is blocked, never a second plan.
    assert fr.schedule_predicate_rereads_after_read(
        url=_URL_A, body_truncated_for_storage=True, raw_unavailable=True) == []
    assert fr.predicate_reread_blocked_count >= 1


# ------------------------------------------------ 9: end-of-item explicit reasons
def test_end_of_item_every_pending_has_explicit_service_reason():
    fr, _, _ = _frontier(n_constraints=2)
    clean = _add_pending(fr, constraint_id="c1", candidate_id="cand1")
    suppressed = _add_pending(fr, constraint_id="c2", candidate_id="cand2",
                              source_role="generic_definition_page")
    fr.finalize_pending_service(budget_remaining=0, reading_tools_enabled=True)
    assert clean.service_status == "service_budget_exhausted"
    assert suppressed.service_status == "service_suppressed_non_executable"
    m = fr.metrics()
    assert m["pending_obligation_without_service_reason_count"] == 0
    assert m["pending_service_status_counts"]["service_budget_exhausted"] == 1
    rec = fr._service_rec(_URL_A)
    assert rec["status"] == "service_budget_exhausted"


# ------------------------------------------------ 10/12: live-run -> replay roundtrip
def _live_then_persist_run(tmp_path) -> Path:
    """Simulate the live mechanism end-to-end, persist its artifacts, return the run dir."""
    run = tmp_path / "run"
    (run / "cache").mkdir(parents=True)
    judge = deterministic_replay_judge()
    judge.cache = ParserCache(str(run / "llm_evidence_judge_cache.json"))
    fr, frame, _ = _frontier(judge=judge)
    _add_candidate(fr)
    p = _add_pending(fr)
    sp = fr.propose_pending_service_read(budget_remaining=4, page_fetch_available=True)
    assert sp is not None
    fr.record_pending_service_outcome(url=_URL_A, tool=sp.read_decision.tool,
                                      body_chars=len(_BODY_SUPPORT),
                                      page_fetch_available=True)
    fr.route_read_into_pending_judgments(candidate_id="cand1", source_url=_URL_A,
                                         read_text=_BODY_SUPPORT, judge=judge)
    fr.finalize_pending_service(budget_remaining=2, reading_tools_enabled=True)
    cache = {"mode": "auto", "store_raw": False, "entries": [
        {"request_hash": "h1", "provider": "page_fetch", "name": "page_fetch",
         "request_meta": {"query": _URL_A},
         "response": {"provider": "page_fetch", "query": _URL_A,
                      "results": [{"title": "Avery Quinn", "url": _URL_A,
                                   "snippet": _BODY_SUPPORT, "rank": 0}], "error": None}}]}
    (run / "cache" / "provider_cache.json").write_text(json.dumps(cache))
    dbg = fr.to_debug()
    record = {
        "item_id": "t-001",
        "question_preview": "Who is the engineer who studied at Northvale in 1987?",
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
            "slates": [{"slot_role": "person",
                        "top_candidates": [{"candidate_text_preview": "Avery Quinn"}]}]},
        "calls": [], "evidence": []}
    (run / "debug_questions.jsonl").write_text(json.dumps(record) + "\n")
    return run


def test_validator_reports_service_rates_and_finds_live_recorded_rejudgment(tmp_path):
    run = _live_then_persist_run(tmp_path)
    res = load_legacy_run(run)
    pm = res.metrics
    assert pm["pending_service_url_count"] == 1
    assert pm["pending_service_url_attempted_count"] == 1
    assert pm["pending_service_url_success_count"] == 1
    assert pm["pending_service_url_success_rate"] == 1.0
    assert pm["pending_service_obligations_served_by_successful_read_count"] >= 1
    assert pm["pending_service_obligation_success_rate"] > 0
    # the LIVE-recorded strict rejudgment is found OFFLINE: closed, never
    # rejudgment_prompt_not_in_cache / pending_source_has_search_snippet_only.
    o = res.obligations[0]
    assert o.body_is_actual_read_body
    assert o.closure_code == "resolved_full_support"
    # 5v-2: precise outcome stage reason; 5v-1: verified by exact read_body_id.
    assert o.stage_reason == "judged_resolved_full_support"
    assert o.rejudgment_verify_method == "body_id"
    assert pm["stage_reason_counts"].get("rejudgment_prompt_not_in_cache", 0) == 0
    assert pm["pending_read_targeted_rejudgment_recorded_count"] == 1
    assert pm["pending_read_targeted_rejudgment_missing_count"] == 0
    assert pm["recorded_rejudgment_verified_by_body_id_count"] == 1
    assert pm["recorded_rejudgment_body_hash_mismatch_count"] == 0
    assert pm["closed_count"] == 1
    assert res.overall_status == "validated_closed"
    assert consistency_violations(pm) == []


# ------------------------------------------------ 11: suppressed never a service failure
def test_suppressed_pendings_not_executable_and_not_service_failures():
    fr, _, _ = _frontier()
    _add_pending(fr, source_role="generic_definition_page")
    assert fr.propose_pending_service_read(budget_remaining=4,
                                           page_fetch_available=True) is None
    fr.finalize_pending_service(budget_remaining=4, reading_tools_enabled=True)
    m = fr.metrics()
    assert m["pending_service_url_count"] == 0          # suppressed never enters the registry
    assert m["pending_service_obligation_count"] == 0
    assert m["pending_read_primary_failed_count"] == 0
    assert m["pending_read_fallback_failed_count"] == 0
    assert m["pending_service_status_counts"] == {
        "service_suppressed_non_executable": 1}


# ------------------------------------------------ 12: default validation zero live calls
def test_default_validation_and_mechanism_make_zero_live_calls(tmp_path):
    run = _live_then_persist_run(tmp_path)
    res = load_legacy_run(run)
    assert res.metrics["live_provider_calls"] == 0
    assert res.metrics["live_model_calls"] == 0
    assert res.metrics["live_judge_tier"]["enabled"] is False
    assert consistency_violations(res.metrics) == []
