"""Level 5q offline tests: pending-read URL targeting + read integrity. Pending obligations
are an EXECUTABLE queue: unrelated read-class calls are hard-blocked while clean pendings are
open; contaminated/noise requires_read never becomes executable; the validator names the
targeting failure precisely (native_pending_reads_not_targeted). Zero live calls.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from regimes_probe.agent.candidate_frontier import CandidateFrontier, SlotCandidate, _hash
from regimes_probe.agent.evidence_interpreter import EvidenceInterpreter
from regimes_probe.agent.llm_task_frame import LLMTaskFrameParser, build_task_frame
from regimes_probe.eval.replay_validation import deterministic_replay_judge, load_legacy_run

ROOT = Path(__file__).resolve().parents[1]
_URL_A = "https://profiles.example/people/alpha-profile-page-with-the-relevant-biography"
_URL_B = "https://elsewhere.example/some/other/page/entirely-unrelated-to-the-obligation"
_URL_C = "https://another.example/generic/definition/page-not-tied-to-any-obligation"


def _frame_payload():
    return {
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


def _frontier():
    parser = LLMTaskFrameParser(model_fn=lambda _p: json.dumps(_frame_payload()), model="stub")
    frame, _ = build_task_frame("x", "Who is the engineer who studied at Northvale in 1987?",
                                use_llm=True, parser=parser)
    fr = CandidateFrontier(frame,
                           interpreter=EvidenceInterpreter(judge=deterministic_replay_judge()))
    return fr, frame


def _add_pending(fr, *, url=_URL_A, candidate_id="cand1"):
    fr._register_pending_read_judgment(candidate_id=candidate_id, slot_id="D",
                                       constraint_id="c1", source_url=url,
                                       source_role="professional_profile")


def _add_candidate(fr, sid, text, url):
    cand = SlotCandidate(candidate_id=f"cand_{text.lower().replace(' ', '_')}",
                         candidate_text=text, normalized_text_hash=_hash(text.lower()),
                         inferred_role="person", slot_id=sid,
                         source_domains=["another.example"], constraints_unknown=["c1"],
                         source_urls=[url])
    fr.slates[sid].candidates[cand.candidate_id] = cand
    fr.candidates_by_id[cand.candidate_id] = cand
    return cand


# ----------------------------------------------------------------- C: hard block unrelated
def test_unrelated_read_is_hard_blocked_while_pending_open():
    fr, frame = _frontier()
    sid = frame.all_slots[0].slot_id
    _add_pending(fr, url=_URL_A, candidate_id="cand1")
    # mark the pending URL as already scraped so the pending read is skipped this step,
    # leaving only an UNRELATED candidate-source read (URL_C) available.
    from regimes_probe.agent.reading_policy import normalize_url
    scraped = frozenset({normalize_url(_URL_A)})
    _add_candidate(fr, sid, "Rival Person", _URL_C)
    fr._slot_no_support_streak[sid] = 5            # force a read action
    sp = fr.propose_step_action(observations=[], budget_remaining=4, reading_tools=True,
                                page_fetch_available=True, scraped_urls=scraped)
    assert sp.kind == "unexecutable"
    assert sp.reason == "read_blocked_unrelated_to_pending_obligation"
    assert fr.read_blocked_unrelated_to_pending_obligation_count == 1
    ev = next(e for e in fr.events
              if e["event_type"] == "read_blocked_unrelated_to_pending_obligation")
    assert ev["data"]["selected_read_url_host"] == "another.example"
    assert ev["data"]["pending_obligation_ids"]
    # the obligation is preserved, never lost.
    assert any(p.open for p in fr.pending_read_judgments.values())
    m = fr.metrics()
    assert m["requires_read_obligation_lost_before_read_count"] == 0


# ----------------------------------------------------------------- D: pending wins over EIG
def test_pending_url_selected_before_unrelated_candidate_read():
    fr, frame = _frontier()
    sid = frame.all_slots[0].slot_id
    _add_pending(fr, url=_URL_A, candidate_id="cand1")
    _add_candidate(fr, sid, "Rival Person", _URL_C)   # an unrelated readable source
    fr._slot_no_support_streak[sid] = 5
    sp = fr.propose_step_action(observations=[], budget_remaining=4, reading_tools=True,
                                page_fetch_available=True)
    assert sp.kind == "read"
    assert getattr(sp.read_obs, "url", "") == _URL_A   # the pending URL, not URL_C
    # 5q-5: the obligation records its execution linkage.
    p = next(p for p in fr.pending_read_judgments.values())
    assert p.read_selected and p.selected_read_url == _URL_A and p.read_tool
    assert fr.metrics()["read_scheduled_for_different_url_than_pending_obligation_count"] == 0


def test_queue_prioritizes_blocking_and_dedupes():
    fr, frame = _frontier()
    _add_pending(fr, url=_URL_A, candidate_id="cand1")
    _add_pending(fr, url=_URL_A, candidate_id="cand2")     # same normalized URL -> deduped
    q = fr.select_pending_read_obligation_url()
    assert len(q) == 1 and q[0].source_url == _URL_A


# ----------------------------------------------------------------- E: contamination suppression
def test_contaminated_requires_read_is_suppressed_not_executable():
    fr, _ = _frontier()
    fr._register_pending_read_judgment(candidate_id="cand1", slot_id="D", constraint_id="c1",
                                       source_url=_URL_B, contaminated=True)
    fr._register_pending_read_judgment(candidate_id="cand2", slot_id="D", constraint_id="c1",
                                       source_url=_URL_C,
                                       source_role="generic_definition_page")
    pend = list(fr.pending_read_judgments.values())
    assert len(pend) == 2 and all(not p.open for p in pend)
    assert all(p.resolution == "suppressed_contaminated_or_noise_source" for p in pend)
    assert {p.suppressed_reason for p in pend} == {
        "benchmark_contaminated", "noise_source_role:generic_definition_page"}
    assert fr.select_pending_read_obligation_url() == []   # never executable
    m = fr.metrics()
    assert m["pending_read_suppressed_contaminated_or_noise_count"] == 2
    assert m["executable_pending_read_from_contaminated_source_count"] == 0
    assert m["executable_pending_read_from_generic_definition_source_count"] == 0
    assert any(e["event_type"] == "pending_read_suppressed_contaminated_or_noise_source"
               for e in fr.events)


# ----------------------------------------------------------------- A/B: validator statuses
def _future_run(tmp_path, *, pending_url, body_url, selected_read_url="") -> Path:
    run = tmp_path / "run"
    (run / "cache").mkdir(parents=True)
    body = ("page boilerplate. " * 80) + " Avery Quinn studied at the Northvale Institute in 1987. "
    cache = {"mode": "auto", "store_raw": False, "entries": [
        {"request_hash": "h1", "provider": "page_fetch", "name": "page_fetch",
         "request_meta": {"query": body_url},
         "response": {"provider": "page_fetch", "query": body_url,
                      "results": [{"title": "Avery Quinn", "url": body_url,
                                   "snippet": body, "rank": 0}], "error": None}}]}
    (run / "cache" / "provider_cache.json").write_text(json.dumps(cache))
    record = {
        "item_id": "future-001",
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
                "slot_id": "s0", "constraint_id": "c1", "source_url": pending_url,
                "source_url_host": "profiles.example", "source_subject": "Avery Quinn",
                "missing_anchors": [], "target_terms": ["northvale", "1987"],
                "created_step": 1, "resolved_step": None, "resolution": "open",
                "read_selected": bool(selected_read_url), "passage_preview": "",
                "source_role": "professional_profile", "suppressed_reason": "",
                "selected_read_url": selected_read_url, "read_tool": ""}],
            "slates": [{"slot_role": "person",
                        "top_candidates": [{"candidate_text_preview": "Avery Quinn"}]}]},
        "calls": [], "evidence": []}
    (run / "debug_questions.jsonl").write_text(json.dumps(record) + "\n")
    (run / "llm_evidence_judge_cache.json").write_text("{}")
    return run


def test_native_unmatched_body_reports_pending_reads_not_targeted(tmp_path):
    run = _future_run(tmp_path, pending_url=_URL_A, body_url=_URL_B)
    res = load_legacy_run(run)
    assert res.metrics["reconstruction_source"] == "native_persisted"
    assert res.overall_status == "native_pending_reads_not_targeted"   # not snippet_only
    assert res.metrics["read_body_unlinked_to_requires_read_obligation"] is True
    assert "pending_read_not_targeted" in res.metrics["stage_reason_counts"]
    assert res.metrics["pending_read_targeting_success_rate"] == 0.0
    assert res.metrics["pending_read_body_link_rate"] == 0.0
    assert res.metrics["obligation_urls_with_no_read_attempt_sample"]
    assert res.metrics["read_urls_unlinked_to_obligation_sample"]
    assert res.metrics["body_located_count"] == 0
    assert res.metrics["live_model_calls"] == 0


def test_native_matching_body_locates_with_good_rates(tmp_path):
    run = _future_run(tmp_path, pending_url=_URL_A, body_url=_URL_A)
    res = load_legacy_run(run)
    assert res.overall_status not in ("native_pending_reads_not_targeted",
                                      "reconstructed_search_snippet_only")
    assert res.metrics["body_located_count"] == 1
    assert res.metrics["pending_read_targeting_success_rate"] == 1.0
    assert res.metrics["pending_read_body_link_rate"] == 1.0
    o = res.obligations[0]
    assert o.read_body_link_source in ("exact_url", "prefix_url", "pending_id")


# ----------------------------------------------------------------- G: pending-id redirect link
def test_body_links_by_pending_id_when_url_redirected(tmp_path):
    # the obligation's source_url is A, but the executed read landed at B (redirect); the
    # pending persisted selected_read_url=B, so the validator links by pending id.
    run = _future_run(tmp_path, pending_url=_URL_A, body_url=_URL_B,
                      selected_read_url=_URL_B)
    res = load_legacy_run(run)
    o = res.obligations[0]
    assert o.body_is_actual_read_body and o.body_source == "cache_read_body"
    assert o.read_body_link_source == "pending_id"
    assert o.matched_pending_read_judgment_id_from_body == "prj1"
    assert o.read_targeted_pending_obligation is True
    assert res.metrics["body_located_count"] == 1


# ----------------------------------------------------------------- default zero live calls
def test_default_validation_zero_live_calls(tmp_path):
    run = _future_run(tmp_path, pending_url=_URL_A, body_url=_URL_A)
    res = load_legacy_run(run)
    assert res.metrics["live_provider_calls"] == 0
    assert res.metrics["live_model_calls"] == 0
    assert res.metrics["live_judge_tier"]["enabled"] is False
