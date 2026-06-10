"""Level 5k offline tests: real-artifact legacy reconstruction (1/2), passage scan on real
cached bodies incl. raw-payload beyond the 4000 cap (3), opt-in capped live-judge tier (4),
future-run persistence (5), bounded re-read planning (6). Zero live provider/model calls in
default mode; the 'live judge' is a deterministic local stub, never a network call.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from regimes_probe.agent.candidate_frontier import CandidateFrontier
from regimes_probe.agent.evidence_interpreter import EvidenceInterpreter
from regimes_probe.agent.llm_task_frame import LLMTaskFrameParser, build_task_frame
from regimes_probe.agent.read_judgment import DEFAULT_READ_CONFIG
from regimes_probe.eval.replay_validation import (
    deterministic_replay_judge, export_read_judge_replay, load_legacy_run, run_replay,
    validate_artifacts_dir)

ROOT = Path(__file__).resolve().parents[1]
_LEGACY = ROOT / "fixtures" / "replay" / "legacy_run"
_READ_FX = ROOT / "fixtures" / "replay" / "read_judge_loop_fixture.json"


def _copy_legacy(tmp_path) -> Path:
    import shutil
    dst = tmp_path / "legacy_run"
    shutil.copytree(_LEGACY, dst)
    return dst


# ----------------------------------------------------------------- 1/2/3 legacy loader
def test_legacy_loader_reconstructs_and_scans_passages_offline():
    res = load_legacy_run(_LEGACY)                          # default: no live judge
    assert res.metrics["live_model_calls"] == 0 and res.metrics["live_provider_calls"] == 0
    assert len(res.obligations) == 1
    o = res.obligations[0]
    assert o.reconstructed_from_legacy_trace is True
    assert o.pipeline_status == "passages_scanned"
    assert o.stage_reason == "rejudgment_prompt_not_in_cache"   # NOT a generic cache-miss
    assert res.overall_status == "reconstructed_passages_scanned_rejudgment_pending"


def test_legacy_loader_uses_raw_payload_beyond_4000_cap():
    o = load_legacy_run(_LEGACY).obligations[0]
    # the fuller raw provider payload is used, and the resolving fact sits past char 4000.
    assert o.body_source == "raw_cache_payload"
    assert o.cached_payload_chars > o.stored_body_chars
    assert o.passages_found_beyond_4000 is True and o.first_hit_offset > 4000


def test_legacy_loader_reports_no_requires_read_events_per_item():
    # a record with NO requires_read judge events must say so precisely (not pass silently).
    rec = {"item_id": "no-ob", "question_preview": "q", "task_frame": {"constraints": []},
           "candidate_frontier": {"events": []}, "calls": [], "evidence": []}
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "debug_questions.jsonl").write_text(json.dumps(rec) + "\n")
        res = load_legacy_run(p)
        assert res.overall_status == "unvalidated_cache_miss"
        assert any(e["event_type"] == "no_pending_read_judgment_events"
                   for e in res.replay_events)


def test_absent_artifacts_dir_still_unvalidated_cache_miss():
    res = validate_artifacts_dir("results/live/does-not-exist-5k")
    assert res.overall_status == "unvalidated_cache_miss"
    assert any("artifacts_absent" in n for n in res.notes)


# ----------------------------------------------------------------- 4 live-judge tier
def test_live_judge_tier_off_by_default_makes_zero_calls():
    res = load_legacy_run(_LEGACY)                          # allow_live_judge defaults False
    assert res.metrics["live_model_calls"] == 0
    assert all(o.closure_code != "resolved_full_support" for o in res.obligations)


def test_live_judge_tier_closes_and_records_then_replays_offline(tmp_path):
    run = _copy_legacy(tmp_path)
    r1 = load_legacy_run(run, allow_live_judge=True, max_judge_calls=20)
    o = r1.obligations[0]
    assert o.pipeline_status == "judged" and o.stage_reason == "closed_by_live_rejudgment"
    assert r1.metrics["live_model_calls"] == 1
    # the verdict was recorded into the run's judge cache -> a re-run is fully offline.
    r2 = load_legacy_run(run)
    assert r2.metrics["live_model_calls"] == 0
    assert r2.obligations[0].pipeline_status in ("judged", "closed")


def test_live_judge_tier_fail_closed_at_zero_budget(tmp_path):
    run = _copy_legacy(tmp_path)
    res = load_legacy_run(run, allow_live_judge=True, max_judge_calls=0)
    assert res.metrics["live_model_calls"] == 0
    assert res.obligations[0].stage_reason == "rejudgment_call_budget_exhausted"


# ----------------------------------------------------------------- 5 future-run persistence
def _frontier_with_closed_obligation():
    fx = json.loads(_READ_FX.read_text())
    parser = LLMTaskFrameParser(model_fn=lambda _p: json.dumps(fx["frame"]), model="stub")
    frame, _ = build_task_frame("x", fx["question"], use_llm=True, parser=parser)
    fr = CandidateFrontier(frame, interpreter=EvidenceInterpreter(judge=deterministic_replay_judge()))
    sid = frame.all_slots[0].slot_id
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
    return fr


def test_future_run_persistence_is_bounded_and_replayable():
    fr = _frontier_with_closed_obligation()
    rec = export_read_judge_replay(fr, item_id="syn-1", max_passage_chars=300,
                                   fetch_meta_by_url={"https://profiles.example/avery-quinn":
                                                      {"fetched_chars": 6298, "stored_body_chars": 4000,
                                                       "body_truncated_for_storage": True}})
    assert rec["schema"] == "read_judge_replay_v1"
    assert rec["pending_read_judgments"] and rec["pending_read_judgments"][0]["closure_code"] \
        == "resolved_full_support"
    # bounded: passage previews are capped; lifecycle events are persisted.
    assert all(len(p.get("passage_preview", "")) <= 300 for p in rec["pending_read_judgments"])
    types = {e["event_type"] for e in rec["read_judge_events"]}
    assert "read_judged_after_read" in types and "read_required_by_judge" in types
    assert rec["fetch_meta_by_url"]["https://profiles.example/avery-quinn"]["body_truncated_for_storage"]
    # the persisted blob must not be empty and must round-trip through json.
    json.loads(json.dumps(rec))


# ----------------------------------------------------------------- 6 bounded re-read
def test_reread_config_keeps_global_cap_but_allows_higher_judgment_cap():
    c = DEFAULT_READ_CONFIG
    assert c.page_fetch_default_max_chars == 4000          # global default stays conservative
    assert c.read_judgment_max_chars > c.page_fetch_default_max_chars
    assert c.reread_max_chars >= c.read_judgment_max_chars
    assert c.max_rereads_per_pending_judgment == 1


def test_plan_reread_only_when_truncated_unresolved_and_pending():
    fx = json.loads(_READ_FX.read_text())
    parser = LLMTaskFrameParser(model_fn=lambda _p: json.dumps(fx["frame"]), model="stub")
    frame, _ = build_task_frame("x", fx["question"], use_llm=True, parser=parser)
    fr = CandidateFrontier(frame, interpreter=EvidenceInterpreter(judge=deterministic_replay_judge()))
    sid = frame.all_slots[0].slot_id
    url = "https://profiles.example/avery-quinn"
    o = fx["steps"][0]["observations"][0]
    fr.ingest_evidence([SimpleNamespace(**{**o, "failed": False, "benchmark_contaminated": False,
                                           "source_authority": 0.8})],
                       source_tool="serper", directed_slot_id=sid,
                       directed_constraint_ids=["c1"], proposal_id="p")
    cid = fr.resolve_candidate("Avery Quinn", slot_id=sid)["candidate_id"]
    # no re-read when the read resolved the obligation.
    assert fr.plan_reread_for_truncation(candidate_id=cid, source_url=url,
                                         body_truncated_for_storage=True,
                                         pending_resolved=True) is None
    # a pending obligation exists + truncated + unresolved -> exactly one bounded re-read.
    plan = fr.plan_reread_for_truncation(candidate_id=cid, source_url=url,
                                         body_truncated_for_storage=True, pending_resolved=False)
    assert plan and plan["reread_max_chars"] == DEFAULT_READ_CONFIG.reread_max_chars
    assert any(e["event_type"] == "read_reread_due_to_truncation" for e in fr.events)
    # at most once per obligation.
    assert fr.plan_reread_for_truncation(candidate_id=cid, source_url=url,
                                         body_truncated_for_storage=True,
                                         pending_resolved=False) is None
    # never re-read a non-truncated read.
    assert fr.plan_reread_for_truncation(candidate_id=cid, source_url=url,
                                         body_truncated_for_storage=False,
                                         pending_resolved=False) is None


# ----------------------------------------------------------------- regression: 5j fixture green
def test_synthetic_fixture_replay_still_validated():
    res = run_replay(json.loads(_READ_FX.read_text()))
    assert res.overall_status == "validated"
    assert res.metrics["live_model_calls"] == 0
