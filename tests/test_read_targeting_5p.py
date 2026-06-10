"""Level 5p offline tests: read-targeting + obligation/read-body linking. A requires_read
obligation pins the EXACT source_url; the validator prefers NATIVE persisted obligations and
reports URL-set diagnostics so an unlinked read body is a precise mechanism failure
(read_body_unlinked_to_requires_read_obligation), never a silent pass. Zero live calls.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from regimes_probe.agent.candidate_frontier import CandidateFrontier
from regimes_probe.agent.evidence_interpreter import EvidenceInterpreter
from regimes_probe.agent.llm_task_frame import LLMTaskFrameParser, build_task_frame
from regimes_probe.eval.replay_validation import deterministic_replay_judge, load_legacy_run

ROOT = Path(__file__).resolve().parents[1]
_LEGACY = ROOT / "fixtures" / "replay" / "legacy_run"
_URL_A = "https://profiles.example/people/alpha-profile-page-with-the-relevant-biography"
_URL_B = "https://elsewhere.example/some/other/page/entirely-unrelated-to-the-obligation"


def _future_run(tmp_path, *, pending_url: str, body_url: str, native: bool = True) -> Path:
    """A synthetic FUTURE-run dir: native persisted pending obligation for ``pending_url``
    + a read-class cache body for ``body_url``. Generic placeholders; no gold."""
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
            "pending_read_judgments": ([{
                "pending_read_judgment_id": "prj1", "candidate_id": "cand1",
                "slot_id": "s0", "constraint_id": "c1", "source_url": pending_url,
                "source_url_host": "profiles.example", "source_subject": "Avery Quinn",
                "missing_anchors": [], "target_terms": ["northvale", "1987"],
                "created_step": 1, "resolved_step": None, "resolution": "open",
                "read_selected": False, "passage_preview": ""}] if native else []),
            "slates": [{"slot_role": "person",
                        "top_candidates": [{"candidate_text_preview": "Avery Quinn"}]}]},
        "calls": [], "evidence": []}
    (run / "debug_questions.jsonl").write_text(json.dumps(record) + "\n")
    (run / "llm_evidence_judge_cache.json").write_text("{}")
    return run


# ----------------------------------------------------------------- A: matching read body
def test_native_obligation_with_matching_read_body_locates_and_scans(tmp_path):
    run = _future_run(tmp_path, pending_url=_URL_A, body_url=_URL_A)
    res = load_legacy_run(run)
    assert res.metrics["reconstruction_source"] == "native_persisted"
    assert res.metrics["native_pending_read_judgment_count"] == 1
    o = res.obligations[0]
    assert o.reconstruction_method == "native_persisted"
    assert o.body_source == "cache_read_body" and o.body_is_actual_read_body
    assert o.pipeline_status in ("passages_scanned", "rejudgment_pending",
                                 "actual_body_located")
    assert res.metrics["body_located_count"] == 1
    assert res.metrics["obligations_without_read_body_count"] == 0
    assert res.metrics["read_body_unlinked_to_requires_read_obligation"] is False
    assert res.metrics["live_model_calls"] == 0


# ----------------------------------------------------------------- B: mismatched read body
def test_obligation_url_a_body_url_b_stays_unmatched_with_diagnostics(tmp_path):
    run = _future_run(tmp_path, pending_url=_URL_A, body_url=_URL_B)
    res = load_legacy_run(run)
    o = res.obligations[0]
    assert not o.body_is_actual_read_body
    assert res.metrics["body_located_count"] == 0
    assert res.metrics["read_body_urls_without_obligations_count"] == 1
    assert res.metrics["obligations_without_read_body_count"] == 1
    # the 5o-smoke failure mode is named precisely: a TARGETING failure, not route_miss.
    assert res.metrics["read_body_unlinked_to_requires_read_obligation"] is True
    assert res.metrics["pending_read_not_targeted_count"] == 1
    assert any(e["event_type"] == "read_body_unlinked_to_requires_read_obligation"
               for e in res.replay_events)
    assert res.metrics["unmatched_obligation_source_urls_sample"]
    assert res.metrics["unmatched_read_body_urls_sample"]


# ----------------------------------------------------------------- C/D: native vs legacy
def test_native_persisted_preferred_over_structured_reconstruction(tmp_path):
    run = _future_run(tmp_path, pending_url=_URL_A, body_url=_URL_A)
    # ALSO add structured interpretations -- native must still win.
    rec = json.loads((run / "debug_questions.jsonl").read_text())
    rec["candidate_frontier"]["interpretations"] = [{
        "interpretation_id": "i1", "source_url": _URL_B, "source_domain": "elsewhere.example",
        "source_role": "article",
        "candidate_assertions": [{"assertion_id": "a0", "candidate_text": "Avery Quinn",
                                  "proposed_slot_ids": ["s0"],
                                  "canonical_candidate_ids": {"s0": "cand1"},
                                  "requires_read_constraint_ids": ["c1"], "judgments": []}]}]
    (run / "debug_questions.jsonl").write_text(json.dumps(rec) + "\n")
    res = load_legacy_run(run)
    assert res.metrics["reconstruction_source"] == "native_persisted"
    assert all(o.reconstruction_method == "native_persisted" for o in res.obligations)
    assert res.metrics["legacy_reconstructed_pending_read_judgment_count"] == 0


def test_fallback_to_structured_interpretations_when_no_native(tmp_path):
    run = _future_run(tmp_path, pending_url=_URL_A, body_url=_URL_A, native=False)
    rec = json.loads((run / "debug_questions.jsonl").read_text())
    rec["candidate_frontier"]["interpretations"] = [{
        "interpretation_id": "i1", "source_url": _URL_A, "source_domain": "profiles.example",
        "source_role": "professional_profile",
        "candidate_assertions": [{"assertion_id": "a0", "candidate_text": "Avery Quinn",
                                  "proposed_slot_ids": ["s0"],
                                  "canonical_candidate_ids": {"s0": "cand1"},
                                  "requires_read_constraint_ids": ["c1"], "judgments": []}]}]
    (run / "debug_questions.jsonl").write_text(json.dumps(rec) + "\n")
    res = load_legacy_run(run)
    assert res.metrics["reconstruction_source"] == "structured_interpretations"
    assert res.metrics["native_pending_read_judgment_count"] == 0
    assert res.metrics["legacy_reconstructed_pending_read_judgment_count"] >= 1


# ----------------------------------------------------------------- E: pending pins the URL
def _frontier_with_pending():
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
    sid = frame.all_slots[0].slot_id
    fr.ingest_evidence([SimpleNamespace(title="Avery Quinn",
                                        snippet="Avery Quinn is an engineer at Northvale.",
                                        url=_URL_A, source_authority=0.8, failed=False,
                                        benchmark_contaminated=False)],
                       source_tool="serper", directed_slot_id=sid,
                       directed_constraint_ids=["c1"], proposal_id="p")
    return fr, sid


def test_read_targets_the_pending_obligations_exact_url():
    fr, sid = _frontier_with_pending()
    pend = [p for p in fr.pending_read_judgments.values() if p.open]
    assert pend and pend[0].source_url == _URL_A
    sp = fr.propose_step_action(observations=[], budget_remaining=4, reading_tools=True,
                                page_fetch_available=True)
    assert sp.kind == "read"                                  # never a search (pinned 0)
    assert getattr(sp.read_obs, "url", "") == _URL_A          # the EXACT pending url
    assert any(e["event_type"] == "read_selected_for_pending_obligation" for e in fr.events)
    m = fr.metrics()
    # kind=read above proves no silent search translation (eval-level metric pinned 0).
    assert m["read_scheduled_for_different_url_than_pending_obligation_count"] == 0
    assert m["pending_read_judgment_without_source_url_count"] == 0
    assert m["requires_read_obligation_lost_before_read_count"] == 0


# ----------------------------------------------------------------- F: unread clean pending
def test_unread_pending_with_clean_url_is_counted():
    fr, sid = _frontier_with_pending()
    m = fr.metrics()                                          # no read executed yet
    assert m["pending_read_obligation_only_search_snippet_after_read_budget_count"] >= 1
    # native persistence: to_debug carries the pending with the FULL source_url.
    dbg = fr.to_debug()
    assert dbg["pending_read_judgments"]
    assert dbg["pending_read_judgments"][0]["source_url"] == _URL_A


# ----------------------------------------------------------------- G: default zero calls
def test_default_validation_zero_live_calls(tmp_path):
    run = _future_run(tmp_path, pending_url=_URL_A, body_url=_URL_A)
    res = load_legacy_run(run)
    assert res.metrics["live_provider_calls"] == 0
    assert res.metrics["live_model_calls"] == 0
    assert res.metrics["live_judge_tier"]["enabled"] is False
