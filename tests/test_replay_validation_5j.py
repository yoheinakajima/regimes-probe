"""Level 5j offline validation tests: replay/fork of the read→judge loop (A), truncation
boundary (B), fixture mechanism effects (C), event-derived metric parity (D), read terminology
(F), and parser referent diagnostics (E). Zero live provider/model calls — fixtures + stubs.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from regimes_probe.agent.candidate_frontier import CandidateFrontier
from regimes_probe.agent.coreference import parser_referent_diagnostics
from regimes_probe.agent.evidence_interpreter import EvidenceInterpreter
from regimes_probe.agent.llm_task_frame import LLMTaskFrameParser, build_task_frame
from regimes_probe.agent.read_judgment import ReadJudgmentConfig, extract_passages
from regimes_probe.eval.metric_derivation import (
    derive_metrics_from_events, verify_metric_derivation)
from regimes_probe.eval.replay_validation import (
    deterministic_replay_judge, run_replay, run_triage_promotion_fixture,
    validate_artifacts_dir)

ROOT = Path(__file__).resolve().parents[1]
_READ_FX = ROOT / "fixtures" / "replay" / "read_judge_loop_fixture.json"
_TRIAGE_FX = ROOT / "fixtures" / "replay" / "triage_promotion_fixture.json"


def _frame(payload, q):
    parser = LLMTaskFrameParser(model_fn=lambda _p: json.dumps(payload), model="stub")
    f, meta = build_task_frame("x", q, use_llm=True, parser=parser)
    assert meta.parser_used == "llm"
    return f


# ----------------------------------------------------------------- A: replay closure
def test_replay_closes_pending_read_judgment_via_full_body():
    fx = json.loads(_READ_FX.read_text())
    res = run_replay(fx)
    assert res.overall_status == "validated"
    assert len(res.obligations) == 1
    o = res.obligations[0]
    assert o.closure_code == "resolved_full_support" and o.read_replayed
    m = res.metrics
    assert m["replay_pending_read_judgment_count"] == 1
    assert m["replay_read_judged_after_read_count"] == 1
    assert m["replay_pending_read_closed_count"] == 1
    assert m["replay_requires_read_still_open_count"] == 0
    assert m["live_model_calls"] == 0 and m["live_provider_calls"] == 0
    types = {e["event_type"] for e in res.replay_events}
    assert "pending_read_judgment_replayed" in types
    assert "pending_read_judgment_closed" in types


def test_replay_cache_miss_is_not_a_silent_pass():
    fx = json.loads(_READ_FX.read_text())
    fx = json.loads(json.dumps(fx))
    fx["body_cache"] = {}                     # the read body is missing from the cache
    res = run_replay(fx)
    assert res.overall_status == "unvalidated_cache_miss"
    assert res.metrics["replay_pending_read_unvalidated_cache_miss_count"] >= 1
    assert any(o.closure_code == "unvalidated_cache_miss" for o in res.obligations)


def test_validate_absent_artifacts_dir_returns_unvalidated_cache_miss():
    res = validate_artifacts_dir("results/live/browsecomp-llm-evidence-judge-5g-smoke-001")
    assert res.overall_status == "unvalidated_cache_miss"
    assert any("artifacts_absent" in n for n in res.notes)
    assert res.metrics["live_model_calls"] == 0


# ----------------------------------------------------------------- B: truncation boundary
def test_extract_passages_finds_text_after_char_4000():
    head = "navigation boilerplate filler. " * 200
    assert len(head) > 4000
    body = head + " Avery Quinn studied at Northvale in 1987."
    scan = extract_passages(body, ["Northvale", "1987"],
                            config=ReadJudgmentConfig(read_passage_window_chars=160,
                                                      read_max_chars_total=20000))
    assert scan.hit and scan.first_hit_offset > 4000
    assert any("1987" in p for p in scan.passages)
    assert not scan.head_only


def test_page_fetch_records_truncation_metadata():
    from regimes_probe.tools.fake import FakePageFetch  # noqa: F401 (ensure import path ok)
    from regimes_probe.tools.page_fetch import PageFetch
    pf = PageFetch(max_chars=50)
    # offline: a non-URL query returns the requires_url error path (no network), which is fine
    # for asserting the adapter cap is the configured field, not a provider/storage cap.
    assert pf.max_chars == 50
    resp = pf.search("not a url")
    assert resp.failed and resp.error_meta["error_type"] == "requires_url"


def test_replay_obligation_used_full_body_not_truncated_snippet():
    res = run_replay(json.loads(_READ_FX.read_text()))
    o = res.obligations[0]
    # the resolving anchor is located well past char 4000 in the fixture body.
    assert o.first_hit_offset > 4000 and o.passage_anchor_hits >= 1


# ----------------------------------------------------------------- C: fixture effects
def test_triage_fixture_reduces_judge_calls_and_promotes_nothing_unsafe():
    c = run_triage_promotion_fixture(json.loads(_TRIAGE_FX.read_text()))
    assert c["fixture_judge_call_reduction_ratio"] >= 0.5      # >=50% fewer judge calls
    assert c["fixture_valid_candidate_verdict_regression_count"] == 0
    assert c["fixture_generic_source_candidate_promotion_count"] == 0
    assert c["candidate_promoted_from_source_title_only_count"] == 0
    assert c["candidate_promoted_from_chrome_count"] == 0
    assert c["explicit_location_mismatch_promoted_count"] == 0
    assert c["explicit_location_mismatch_rejected_count"] >= 1
    assert c["live_model_calls"] == 0


# ----------------------------------------------------------------- D: metric derivation
def test_event_derived_metrics_match_inline_counters():
    fx = json.loads(_READ_FX.read_text())
    payload, q = fx["frame"], fx["question"]
    f = _frame(payload, q)
    fr = CandidateFrontier(f, interpreter=EvidenceInterpreter(judge=deterministic_replay_judge()))
    sid = f.all_slots[0].slot_id
    st = fx["steps"][0]
    fr.ingest_evidence([SimpleNamespace(**{**o, "failed": False,
                                           "benchmark_contaminated": False,
                                           "source_authority": o.get("source_authority", 0.7)})
                        for o in st["observations"]],
                       source_tool="serper", directed_slot_id=sid,
                       directed_constraint_ids=["c1"], proposal_id="p")
    cid = fr.resolve_candidate("Avery Quinn", slot_id=sid)["candidate_id"]
    body = fx["body_cache"]["https://profiles.example/avery-quinn"]
    fr.ingest_evidence([SimpleNamespace(title="Avery Quinn", snippet=body,
                                        url="https://profiles.example/avery-quinn",
                                        source_authority=0.7, failed=False,
                                        benchmark_contaminated=False)],
                       source_tool="page_fetch", read_depth=1, directed_slot_id=sid,
                       read_candidate_id=cid)
    chk = verify_metric_derivation(fr)
    assert chk["ok"], chk["mismatches"]
    assert chk["derived"]["requires_read_resolved_by_read_count"] == 1
    # the derivation recomputes from the event log, independent of inline counters.
    derived = derive_metrics_from_events(fr.events)
    assert derived["read_judged_after_read_count"] == 1


# ----------------------------------------------------------------- F: read terminology
def test_read_outcomes_are_distinct_candidates_vs_support_vs_resolution():
    fx = json.loads(_READ_FX.read_text())
    res = run_replay(fx)            # the fixture read RESOLVES a pending judgment
    # rebuild to inspect the read_interpreted event fields directly.
    f = _frame(fx["frame"], fx["question"])
    fr = CandidateFrontier(f, interpreter=EvidenceInterpreter(judge=deterministic_replay_judge()))
    sid = f.all_slots[0].slot_id
    o = fx["steps"][0]["observations"][0]
    fr.ingest_evidence([SimpleNamespace(**{**o, "failed": False, "benchmark_contaminated": False,
                                           "source_authority": 0.8})],
                       source_tool="serper", directed_slot_id=sid,
                       directed_constraint_ids=["c1"], proposal_id="p")
    cid = fr.resolve_candidate("Avery Quinn", slot_id=sid)["candidate_id"]
    body = fx["body_cache"]["https://profiles.example/avery-quinn"]
    fr.ingest_evidence([SimpleNamespace(title="Avery Quinn", snippet=body,
                                        url="https://profiles.example/avery-quinn",
                                        source_authority=0.7, failed=False,
                                        benchmark_contaminated=False)],
                       source_tool="page_fetch", read_depth=1, directed_slot_id=sid,
                       read_candidate_id=cid)
    ri = next(e for e in fr.events if e["event_type"] == "read_interpreted")
    d = ri["data"]
    # a read that resolved a pending obligation is SEPARATELY visible from adding candidates.
    assert d["read_resolved_pending_judgment"] is True
    assert "read_added_candidates" in d and "read_added_constraint_support" in d
    m = fr.metrics()
    assert m["read_resolved_pending_judgment_count"] >= 1


# ----------------------------------------------------------------- E: parser diagnostics
def test_parser_referent_diagnostics_flags_duplicate_author_but_blocks_target_subject():
    CO = {
        "target_answer_slots": [{"slot_id": "Y", "slot_name": "birth year of the founder",
                                 "slot_role": "date_or_time", "is_target_answer_slot": True,
                                 "is_intermediate_slot": False}],
        "latent_slots": [{"slot_id": "P1", "slot_name": "this author who wrote a debut",
                          "slot_role": "person"},
                         {"slot_id": "P2", "slot_name": "this author who won a prize",
                          "slot_role": "person"}],
        "constraints": [{"constraint_id": "c1", "text_span": "wrote a debut",
                         "normalized_terms": ["debut"], "applies_to": ["P1"],
                         "discriminative_score": 1.0, "specificity_score": 1.0,
                         "testable_claim": "x", "how_to_test": "read", "semantic_label": "work"}],
        "dependency_edges": [], "known_context_terms": []}
    f = _frame(CO, "Who is the author and the founder's birth year?")
    d = parser_referent_diagnostics(f)
    assert d["parser_entity_variable_count"] >= 2
    assert d["parser_duplicate_referent_slot_count_estimate"] >= 1     # the two "author" slots
    assert d["parser_target_subject_merge_blocked_count"] >= 1         # birth-year vs founder
    assert d["claims_coreference_cost_reduction"] is False
    assert d["parser_prompt_nudge_status"] == "staged_next_parser_only_increment"
