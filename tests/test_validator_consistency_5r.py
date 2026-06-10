"""Level 5r offline tests: validator internal consistency (no contradictory count/stage-reason
pairs), precise stage-reason names, split unlinked counts, read-match metric clarity, and the
no-pending generic-source read gate. Zero live calls.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from regimes_probe.agent.candidate_frontier import CandidateFrontier, SlotCandidate, _hash
from regimes_probe.agent.evidence_interpreter import EvidenceInterpreter
from regimes_probe.agent.llm_task_frame import LLMTaskFrameParser, build_task_frame
from regimes_probe.eval.replay_validation import (
    consistency_violations, deterministic_replay_judge, load_legacy_run)

ROOT = Path(__file__).resolve().parents[1]
_URL_A = "https://profiles.example/people/alpha-profile-page-with-the-relevant-biography"
_URL_B = "https://elsewhere.example/some/other/page/entirely-unrelated-to-the-obligation"
_URL_SNIP = "https://social.example/groups/some-group/posts/123456789/"


def _record_5q_shaped(*, events=None):
    """A 5q-SHAPED record: 3 native pendings — one whose URL has a read body, two whose URLs
    have only search snippets — plus extra unrelated read bodies. Generic placeholders."""
    pend = []
    for i, url in enumerate([_URL_A, _URL_SNIP, _URL_SNIP + "more/"], start=1):
        pend.append({
            "pending_read_judgment_id": f"prj{i}", "candidate_id": "cand1",
            "slot_id": "s0", "constraint_id": "c1", "source_url": url,
            "source_url_host": "x", "source_subject": "Avery Quinn",
            "missing_anchors": [], "target_terms": ["northvale", "1987"],
            "created_step": 1, "resolved_step": None, "resolution": "open",
            "read_selected": i == 1, "passage_preview": "",
            "source_role": "professional_profile", "suppressed_reason": "",
            "selected_read_url": _URL_A if i == 1 else "", "read_tool": ""})
    return {
        "item_id": "r-001",
        "question_preview": "Who is the engineer who studied at the Northvale Institute in 1987?",
        "task_frame": {
            "target_answer_slots": [{"slot_id": "s0", "slot_name": "engineer",
                                     "descriptor": "engineer"}],
            "constraints": [{"constraint_id": "c1",
                             "text_span": "studied at the Northvale Institute in 1987",
                             "normalized_terms": ["northvale", "institute", "1987"],
                             "applies_to": ["s0"]}]},
        "candidate_frontier": {"events": list(events or []), "events_truncated": 0,
                               "interpretations": [], "pending_read_judgments": pend,
                               "slates": [{"slot_role": "person", "top_candidates":
                                           [{"candidate_text_preview": "Avery Quinn"}]}]},
        "calls": [], "evidence": []}


def _run_5q_shaped(tmp_path, *, events=None) -> Path:
    run = tmp_path / "run"
    (run / "cache").mkdir(parents=True)
    body = ("boilerplate. " * 100) + " Avery Quinn studied at the Northvale Institute in 1987. "
    entries = [
        {"request_hash": "h1", "provider": "page_fetch", "name": "page_fetch",
         "request_meta": {"query": _URL_A},
         "response": {"provider": "page_fetch", "query": _URL_A,
                      "results": [{"title": "Avery Quinn", "url": _URL_A,
                                   "snippet": body, "rank": 0}], "error": None}},
        # an UNRELATED read body (generic reference page) — must show up as unlinked.
        {"request_hash": "h2", "provider": "page_fetch", "name": "page_fetch",
         "request_meta": {"query": _URL_B},
         "response": {"provider": "page_fetch", "query": _URL_B,
                      "results": [{"title": "Reference", "url": _URL_B,
                                   "snippet": "generic reference page body " * 40,
                                   "rank": 0}], "error": None}},
        # search snippets for the two snippet-only obligations.
        {"request_hash": "h3", "provider": "serper_search", "name": "serper_search",
         "request_meta": {"query": "q"},
         "response": {"provider": "serper_search", "query": "q",
                      "results": [{"title": "post", "url": _URL_SNIP,
                                   "snippet": "a short search snippet", "rank": 0},
                                  {"title": "post2", "url": _URL_SNIP + "more/",
                                   "snippet": "another snippet", "rank": 1}],
                      "error": None}}]
    (run / "cache" / "provider_cache.json").write_text(
        json.dumps({"mode": "auto", "store_raw": False, "entries": entries}))
    (run / "debug_questions.jsonl").write_text(json.dumps(_record_5q_shaped(events=events)) + "\n")
    (run / "llm_evidence_judge_cache.json").write_text("{}")
    return run


# ----------------------------------------------------------------- 1: consistency invariants
def test_5q_shaped_output_is_internally_consistent(tmp_path):
    res = load_legacy_run(_run_5q_shaped(tmp_path))
    m = res.metrics
    assert consistency_violations(m) == []
    # the 5q contradiction cannot recur: stage reason and count agree by derivation.
    assert m["pending_read_not_targeted_count"] == \
        m["stage_reason_counts"].get("pending_read_not_targeted", 0)
    # unlinked sample non-empty -> explicit positive count + split.
    assert m["unlinked_read_body_count"] >= 1
    assert m["unlinked_read_body_count"] == (m["unlinked_read_body_while_pending_count"]
                                             + m["unlinked_read_body_after_pending_count"])
    assert m["read_urls_unlinked_to_obligation_sample"]
    assert m["live_provider_calls"] == 0 and m["live_model_calls"] == 0


# ----------------------------------------------------------------- 2: precise stage reasons
def test_snippet_only_obligations_get_precise_reason_not_targeting(tmp_path):
    res = load_legacy_run(_run_5q_shaped(tmp_path))
    reasons = res.metrics["stage_reason_counts"]
    # the two snippet-only obligations: pending_source_has_search_snippet_only (no executed-
    # mismatch event in this run), NOT pending_read_not_targeted.
    assert reasons.get("pending_source_has_search_snippet_only", 0) == 2
    assert "pending_read_not_targeted" not in reasons
    assert res.metrics["pending_read_not_targeted_count"] == 0
    # the matched obligation proceeds to rejudgment_pending.
    assert reasons.get("rejudgment_prompt_not_in_cache", 0) == 1


def test_true_not_targeted_event_drives_the_stage_reason(tmp_path):
    events = [{"event_type": "read_scheduled_for_different_url_than_pending_obligation",
               "data": {"chosen_url_host": "elsewhere.example"}}]
    res = load_legacy_run(_run_5q_shaped(tmp_path, events=events))
    reasons = res.metrics["stage_reason_counts"]
    assert reasons.get("pending_read_not_targeted", 0) >= 1
    assert res.metrics["pending_read_not_targeted_count"] == \
        reasons["pending_read_not_targeted"]
    assert res.metrics["not_targeted_read_events_total"] == 1
    assert consistency_violations(res.metrics) == []


def test_body_not_acquired_reason_when_no_snippet_and_no_event(tmp_path):
    run = _run_5q_shaped(tmp_path)
    # remove the search snippets -> the two snippet obligations have NOTHING for their URLs.
    cache = json.loads((run / "cache" / "provider_cache.json").read_text())
    cache["entries"] = [e for e in cache["entries"] if e["provider"] != "serper_search"]
    (run / "cache" / "provider_cache.json").write_text(json.dumps(cache))
    res = load_legacy_run(run)
    reasons = res.metrics["stage_reason_counts"]
    assert reasons.get("pending_obligation_body_not_acquired", 0) == 2
    assert "pending_read_not_targeted" not in reasons
    assert consistency_violations(res.metrics) == []


# ----------------------------------------------------------------- 3: read-match clarity
def test_body_without_read_call_backlink_is_explained(tmp_path):
    res = load_legacy_run(_run_5q_shaped(tmp_path))
    m = res.metrics
    assert m["body_located_count"] == 1
    assert m["matched_read_call_count"] == 0            # no calls persisted in this record
    assert m["body_match_explanation"]                  # the gap is explained, not silent
    assert m["body_located_count"] <= m["matched_read_body_count"]
    assert m["actual_body_located_without_matched_read_body_count"] == 0
    assert m["search_snippet_counted_as_body_count"] == 0


# ----------------------------------------------------------------- 4: generic-source read gate
def _frontier_no_pending():
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
    return fr, frame


def test_generic_definition_read_blocked_when_no_pending():
    fr, frame = _frontier_no_pending()
    sid = frame.all_slots[0].slot_id
    cand = SlotCandidate(candidate_id="cand_dict", candidate_text="Engineer Definition",
                         normalized_text_hash=_hash("engineer definition"),
                         inferred_role="person", slot_id=sid,
                         source_domains=["dictionary.example"], constraints_unknown=["c1"],
                         source_urls=["https://dictionary.example/browse/engineer-term-page"])
    fr.slates[sid].candidates[cand.candidate_id] = cand
    fr.candidates_by_id[cand.candidate_id] = cand
    fr._slot_no_support_streak[sid] = 5
    assert not fr.pending_read_judgments                # no pendings -> gate still applies
    sp = fr.propose_step_action(observations=[], budget_remaining=4, reading_tools=True,
                                page_fetch_available=True)
    assert sp.kind == "unexecutable" and sp.reason == "read_blocked_generic_source"
    assert fr.read_blocked_generic_source_count == 1
    assert any(e["event_type"] == "read_blocked_generic_source" for e in fr.events)
    m = fr.metrics()
    assert m["unrelated_read_executed_while_pending_count"] == 0


def test_concrete_source_read_allowed_when_no_pending():
    fr, frame = _frontier_no_pending()
    sid = frame.all_slots[0].slot_id
    cand = SlotCandidate(candidate_id="cand_real", candidate_text="Avery Quinn",
                         normalized_text_hash=_hash("avery quinn"),
                         inferred_role="person", slot_id=sid,
                         source_domains=["profiles.example"], constraints_unknown=["c1"],
                         source_urls=[_URL_A])
    fr.slates[sid].candidates[cand.candidate_id] = cand
    fr.candidates_by_id[cand.candidate_id] = cand
    fr._slot_no_support_streak[sid] = 5
    sp = fr.propose_step_action(observations=[], budget_remaining=4, reading_tools=True,
                                page_fetch_available=True)
    assert sp.kind == "read" and getattr(sp.read_obs, "url", "") == _URL_A


# ----------------------------------------------------------------- consistency everywhere
def test_consistency_violations_helper_flags_the_5q_contradiction():
    bad = {"stage_reason_counts": {"pending_read_not_targeted": 2},
           "pending_read_not_targeted_count": 0,
           "read_urls_unlinked_to_obligation_sample": ["https://x"],
           "unlinked_read_body_count": 0,
           "unlinked_read_body_while_pending_count": 0,
           "unlinked_read_body_after_pending_count": 0,
           "body_located_count": 1, "matched_read_body_count": 1,
           "matched_read_call_count": 0, "body_match_explanation": "x",
           "search_snippet_counted_as_body_count": 0,
           "actual_body_located_without_matched_read_body_count": 0}
    v = consistency_violations(bad)
    assert "pending_read_not_targeted stage reason and count disagree" in v
    assert "unlinked sample non-empty but unlinked_read_body_count == 0" in v
