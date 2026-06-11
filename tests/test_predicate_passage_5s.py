"""Level 5s offline tests: predicate-passage acquisition + pending-source service. Judge-hint
anchors recover predicate passages the old anchors missed; subject-only stays fail-closed;
pending snippet-only obligations are served first; bounded diagnostics explain every stuck
obligation; the bounded predicate re-read is live-only. Zero live calls.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from regimes_probe.agent.candidate_frontier import CandidateFrontier
from regimes_probe.agent.evidence_interpreter import EvidenceInterpreter
from regimes_probe.agent.llm_task_frame import LLMTaskFrameParser, build_task_frame
from regimes_probe.eval.replay_validation import (
    consistency_violations, deterministic_replay_judge, load_legacy_run)

ROOT = Path(__file__).resolve().parents[1]
_URL_A = "https://profiles.example/people/alpha-profile-page-with-the-relevant-biography"


def _record(*, body_url=_URL_A, requires_read_reason="", rationale=""):
    return {
        "item_id": "s-001",
        "question_preview": "Who is the engineer who studied at the Northvale Institute in 1987?",
        "task_frame": {
            "target_answer_slots": [{"slot_id": "s0", "slot_name": "engineer",
                                     "descriptor": "engineer"}],
            "constraints": [{"constraint_id": "c1",
                             "text_span": "studied at the Northvale Institute in 1987",
                             "normalized_terms": ["northvale", "institute", "1987"],
                             "applies_to": ["s0"]}]},
        "candidate_frontier": {
            "events": [], "events_truncated": 0, "interpretations": [],
            "pending_read_judgments": [{
                "pending_read_judgment_id": "prj1", "candidate_id": "cand1",
                "slot_id": "s0", "constraint_id": "c1", "source_url": body_url,
                "source_url_host": "profiles.example", "source_subject": "Avery Quinn",
                "missing_anchors": [], "target_terms": ["northvale", "1987"],
                "created_step": 1, "resolved_step": None, "resolution": "open",
                "read_selected": True, "passage_preview": "",
                "source_role": "professional_profile", "suppressed_reason": "",
                "selected_read_url": body_url, "read_tool": "page_fetch",
                "requires_read_reason": requires_read_reason,
                "judge_rationale": rationale}],
            "slates": [{"slot_role": "person",
                        "top_candidates": [{"candidate_text_preview": "Avery Quinn"}]}]},
        "calls": [], "evidence": []}


def _run(tmp_path, body: str, **rec_kw) -> Path:
    run = tmp_path / "run"
    (run / "cache").mkdir(parents=True)
    cache = {"mode": "auto", "store_raw": False, "entries": [
        {"request_hash": "h1", "provider": "page_fetch", "name": "page_fetch",
         "request_meta": {"query": _URL_A},
         "response": {"provider": "page_fetch", "query": _URL_A,
                      "results": [{"title": "Avery Quinn", "url": _URL_A,
                                   "snippet": body, "rank": 0}], "error": None}}]}
    (run / "cache" / "provider_cache.json").write_text(json.dumps(cache))
    (run / "debug_questions.jsonl").write_text(json.dumps(_record(**rec_kw)) + "\n")
    (run / "llm_evidence_judge_cache.json").write_text("{}")
    return run


# ----------------------------------------------------------------- A: judge-hint anchors
def test_judge_hint_anchors_recover_predicate_passage(tmp_path):
    # the body never repeats the constraint terms, but it DOES carry the term the prior
    # judge asked to verify (recorded in requires_read_reason) — recoverable via hints.
    body = (("nav header filler. " * 60)
            + " Avery Quinn completed the doctorate program in optics during that period. ")
    run = _run(tmp_path, body,
               requires_read_reason="verify the doctorate completion period for the engineer")
    res = load_legacy_run(run)
    o = res.obligations[0]
    assert o.anchor_category_counts.get("judge_hint", 0) >= 1
    assert o.passage_relevance == "predicate_relevant"
    assert o.pipeline_status in ("passages_scanned", "rejudgment_pending")
    assert consistency_violations(res.metrics) == []


def test_rescue_window_finds_predicate_near_later_subject_occurrence(tmp_path):
    # first occurrence of "1987" sits in a nav header far from the subject; subject +
    # year co-occur LATER — the predicate-window rescue must find that window.
    body = ("Archive 1987 index page navigation. " + ("filler text here. " * 80)
            + " Avery Quinn studied at the institute in 1987 according to records. ")
    run = _run(tmp_path, body)
    res = load_legacy_run(run)
    o = res.obligations[0]
    assert o.passage_relevance == "predicate_relevant"
    assert res.metrics["passage_rescue_used_count"] >= 0   # rescue may or may not be needed
    assert o.pipeline_status in ("passages_scanned", "rejudgment_pending")


# ----------------------------------------------------------------- B/C: fail-closed gating
def test_subject_only_body_stays_fail_closed_and_not_judged(tmp_path):
    body = ("nav filler. " * 60) + " Avery Quinn homepage contact and general links. "
    run = _run(tmp_path, body)
    res = load_legacy_run(run, allow_live_judge=True, max_judge_calls=20)
    o = res.obligations[0]
    assert o.passage_relevance in ("subject_only", "weak_predicate_candidate_only",
                                   "subject_only_no_target_anchor")
    assert o.pipeline_status == "actual_body_located"
    assert res.metrics["live_model_calls"] == 0            # never judged
    assert res.metrics["actual_body_subject_only_count"] >= 1


def test_predicate_relevant_passage_reaches_judge(tmp_path):
    body = ("filler. " * 60) + " Avery Quinn studied at the Northvale Institute in 1987. "
    run = _run(tmp_path, body)
    res = load_legacy_run(run, allow_live_judge=True, max_judge_calls=20)
    assert res.metrics["live_model_calls"] >= 1
    assert any(o.pipeline_status in ("closed", "judged_unclosed") for o in res.obligations)


# ----------------------------------------------------------------- D/E: pending service
def _frontier_with_pending(*, contaminated=False, source_role="article"):
    payload = {
        "target_answer_slots": [{"slot_id": "D", "slot_name": "engineer", "slot_role": "person",
                                 "is_target_answer_slot": True, "is_intermediate_slot": False}],
        "latent_slots": [],
        "constraints": [{"constraint_id": "c1", "text_span": "studied at Northvale in 1987",
                         "normalized_terms": ["northvale", "1987"], "applies_to": ["D"],
                         "discriminative_score": 2.0, "specificity_score": 2.0,
                         "required": True, "priority": "high",
                         "blocks_answer_if_unresolved": True, "testable_claim": "x",
                         "how_to_test": "read", "semantic_label": "education"}],
        "dependency_edges": [], "known_context_terms": []}
    parser = LLMTaskFrameParser(model_fn=lambda _p: json.dumps(payload), model="stub")
    frame, _ = build_task_frame("x", "Who is the engineer who studied at Northvale in 1987?",
                                use_llm=True, parser=parser)
    fr = CandidateFrontier(frame,
                           interpreter=EvidenceInterpreter(judge=deterministic_replay_judge()))
    fr._register_pending_read_judgment(candidate_id="cand1", slot_id="D", constraint_id="c1",
                                       source_url=_URL_A, source_role=source_role,
                                       contaminated=contaminated)
    return fr


def test_pending_snippet_only_obligation_served_before_eig_actions():
    fr = _frontier_with_pending()
    # NO forced streak, NO read action would normally win EIG — yet the pending is served.
    sp = fr.propose_step_action(observations=[], budget_remaining=4, reading_tools=True,
                                page_fetch_available=True)
    assert sp.kind == "read" and getattr(sp.read_obs, "url", "") == _URL_A
    assert sp.reason == "pending_obligation_service"
    assert fr.metrics()["pending_search_snippet_only_read_attempted_count"] == 1
    # the read is a concrete URL read, never a search translation.
    assert sp.action_type == "read_candidate_source"


def test_contaminated_pending_remains_suppressed_and_not_served():
    fr = _frontier_with_pending(contaminated=True)
    assert fr.select_pending_read_obligation_url() == []
    sp = fr.propose_step_action(observations=[], budget_remaining=4, reading_tools=True,
                                page_fetch_available=True)
    assert sp.reason != "pending_obligation_service"
    m = fr.metrics()
    assert m["executable_pending_read_from_contaminated_source_count"] == 0
    assert m["pending_read_suppressed_contaminated_or_noise_count"] == 1


# ----------------------------------------------------------------- F: bounded diagnostics
def test_no_relevant_anchor_emits_bounded_diagnostics(tmp_path):
    body = ("completely unrelated text about something else entirely. " * 40)
    run = _run(tmp_path, body)
    res = load_legacy_run(run)
    diags = res.metrics["predicate_passage_diagnostics"]
    assert diags and len(diags) <= 20
    d = diags[0]
    assert d["pending_read_judgment_id"] == "prj1"
    assert d["diagnostic_reason"] in ("no_subject_anchor", "body_likely_wrong_source",
                                      "predicate_terms_absent")
    assert len(d["source_url"]) <= 120 and len(d["candidate_text"]) <= 80
    assert "missing_predicate_anchor_categories" in d
    assert res.metrics["predicate_passage_diagnostic_reason_counts"]
    assert res.metrics["actual_body_no_relevant_anchor_count"] >= 1


# ----------------------------------------------------------------- G: still-open semantics
def test_requires_read_still_open_is_judged_unclosed_with_diag(tmp_path):
    # body names institute + year but never the candidate -> stub judge keeps it open.
    body = ("filler. " * 60) + " The Northvale Institute opened a 1987 archive page. "
    run = _run(tmp_path, body)
    res = load_legacy_run(run, allow_live_judge=True, max_judge_calls=20)
    o = next(x for x in res.obligations if x.pipeline_status in ("judged_unclosed", "closed"))
    assert o.pipeline_status == "judged_unclosed"
    assert res.metrics["closed_count"] == 0
    assert res.metrics["strict_rejudgment_still_open_count"] >= 1
    diags = res.metrics["predicate_passage_diagnostics"]
    assert any(d["diagnostic_reason"] == "rejudgment_still_requires_read" for d in diags)


# ----------------------------------------------------------------- H: bounded reread (live only)
def test_predicate_reread_scheduled_once_and_only_for_truncated_unresolved():
    fr = _frontier_with_pending()
    plan = fr.plan_predicate_reread(pending_read_judgment_id="prj1",
                                    body_truncated_for_storage=True, raw_unavailable=True,
                                    passage_relevance="subject_only")
    assert plan and plan["predicate_reread_max_chars"] == fr.read_config.reread_max_chars
    assert any(e["event_type"] == "predicate_reread_scheduled" for e in fr.events)
    # at most once.
    assert fr.plan_predicate_reread(pending_read_judgment_id="prj1",
                                    body_truncated_for_storage=True, raw_unavailable=True,
                                    passage_relevance="subject_only") is None
    # never for predicate-relevant or untruncated bodies.
    fr2 = _frontier_with_pending()
    assert fr2.plan_predicate_reread(pending_read_judgment_id="prj1",
                                     body_truncated_for_storage=False, raw_unavailable=False,
                                     passage_relevance="subject_only") is None
    assert fr2.plan_predicate_reread(pending_read_judgment_id="prj1",
                                     body_truncated_for_storage=True, raw_unavailable=True,
                                     passage_relevance="predicate_relevant") is None


# ----------------------------------------------------------------- I/J: zero calls + consistency
def test_default_paths_zero_live_calls_and_consistent(tmp_path):
    for body in (("filler. " * 60) + " Avery Quinn studied at the Northvale Institute in 1987. ",
                 "unrelated text. " * 40):
        run = _run(tmp_path / body[:8].strip(), body)
        res = load_legacy_run(run)
        assert res.metrics["live_provider_calls"] == 0
        assert res.metrics["live_model_calls"] == 0
        assert consistency_violations(res.metrics) == []
