"""Level 5l offline tests: reconstruct real-5g-shaped obligations from the STRUCTURED
interpretations path (the actual debug-record location of requires_read verdicts), with
truncated-URL prefix matching, honest body/truncation reporting, the missing-judgment-detail
case, the --inspect-schema probe, and unambiguous live-judge tier state. Zero live calls in
default mode; the 'live judge' is a deterministic local stub, never a network call.
"""

from __future__ import annotations

import json
from pathlib import Path

from regimes_probe.eval.replay_validation import (
    _match_url, _norm_url, _reconstruct_obligations, inspect_run_schema, load_legacy_run)

ROOT = Path(__file__).resolve().parents[1]
_LEGACY = ROOT / "fixtures" / "replay" / "legacy_run"


def _record() -> dict:
    line = (_LEGACY / "debug_questions.jsonl").read_text().splitlines()[0]
    return json.loads(line)


# ----------------------------------------------------------------- 1: structured reconstruction
def test_reconstructs_from_structured_interpretations_not_events():
    rec = _record()
    assert rec["candidate_frontier"]["events"] == []      # a 5g record has NO 5h read events
    obs, coverage = _reconstruct_obligations(rec)
    assert obs, "must reconstruct from interpretations[].candidate_assertions[].judgments[]"
    o0 = obs[0]
    assert o0["reconstruction_method"] == "structured_interpretations"
    assert o0["candidate_id"] == "cand1"                  # from canonical_candidate_ids[slot]
    assert o0["slot_id"] == "s0" and o0["constraint_id"] == "c1"
    assert o0["candidate_text"] == "Jordan Vale"
    assert o0["reconstructed_from_legacy_trace"] is True


def test_requires_read_constraint_without_judgment_detail_is_kept():
    obs, _ = _reconstruct_obligations(_record())
    detail_missing = [o for o in obs if "judgment_detail" in o.get("reconstruction_missing_fields", [])]
    assert detail_missing and detail_missing[0]["constraint_id"] == "c2_no_detail"


# ----------------------------------------------------------------- 3: truncated-URL prefix match
def test_truncated_source_url_prefix_matches_full_cache_url():
    rec = _record()
    stored = rec["candidate_frontier"]["interpretations"][0]["source_url"]
    assert len(stored) == 160                            # interpretation.source_url is truncated
    res = load_legacy_run(_LEGACY)
    o = res.obligations[0]
    assert o.url_match_method == "prefix"                 # truncated stored url -> prefix of full
    assert o.source_url == stored


def test_norm_and_match_url_helpers():
    assert _norm_url("HTTPS://Example.com/Path/")[0] == "example.com"
    full = "https://example.com/a/very/long/canonical/slug/here"
    assert _match_url("https://example.com/a/very/long", [full]) == (full, "prefix")
    assert _match_url("https://example.com/a", ["https://other.com/a"])[1] == "none"


# ----------------------------------------------------------------- 4: body + honest truncation
def test_body_located_from_raw_payload_with_store_raw_header():
    o = load_legacy_run(_LEGACY).obligations[0]
    assert o.body_source == "cache_read_body"
    assert o.used_full_body_not_snippet is True       # the fuller raw payload was scanned
    assert o.store_raw_was_enabled is True and o.raw_unavailable is False
    assert o.cached_payload_chars > o.stored_body_chars
    assert o.passages_found_beyond_4000 is True and o.first_hit_offset > 4000
    assert o.matched_read is True and o.read_tool == "firecrawl_scrape"


def test_missing_cache_body_reports_read_event_found_but_body_missing(tmp_path):
    rec = _record()
    # strip BOTH the cache and the debug snippet-preview fallback -> no body locatable.
    rec["evidence"] = []
    run = tmp_path / "run"
    (run / "cache").mkdir(parents=True)
    (run / "cache" / "empty.json").write_text(json.dumps({"entries": []}))
    (run / "debug_questions.jsonl").write_text(json.dumps(rec) + "\n")
    res = load_legacy_run(run)
    reasons = {o.stage_reason for o in res.obligations}
    assert reasons & {"read_event_found_but_body_missing", "source_url_mismatch"}
    # the read call still URL-matches, but with no body nothing advances further (5m-5):
    assert all(o.pipeline_status in ("reconstructed", "read_matched")
               for o in res.obligations)
    assert res.metrics["body_located_count"] == 0
    assert res.metrics["passages_scanned_count"] == 0
    assert all(o.body_source == "not_found" for o in res.obligations)
    assert res.overall_status == "reconstructed_body_missing"
    assert res.metrics["live_model_calls"] == 0


# ----------------------------------------------------------------- 6: live-judge tier clarity
def test_live_judge_tier_enabled_true_with_skip_reason_when_no_obligations(tmp_path):
    # a record with NO requires_read judgments -> obligations == 0.
    rec = {"item_id": "none", "question_preview": "q",
           "task_frame": {"constraints": []},
           "candidate_frontier": {"events": [], "interpretations": []}, "calls": [], "evidence": []}
    run = tmp_path / "run"
    run.mkdir()
    (run / "debug_questions.jsonl").write_text(json.dumps(rec) + "\n")
    res = load_legacy_run(run, allow_live_judge=True, max_judge_calls=20)
    tier = res.metrics["live_judge_tier"]
    assert tier["enabled"] is True                       # the flag state is unambiguous
    assert tier["live_judge_skipped_reason"] == "no_reconstructed_obligations"
    assert res.metrics["live_model_calls"] == 0


def test_live_judge_tier_disabled_state_is_explicit():
    res = load_legacy_run(_LEGACY)                        # default: flag off
    assert res.metrics["live_judge_tier"]["enabled"] is False
    assert res.metrics["live_model_calls"] == 0


# ----------------------------------------------------------------- 7: inspect-schema probe
def test_inspect_schema_reports_verdicts_and_raw_flag():
    info = inspect_run_schema(_LEGACY)
    assert info["exists"] is True
    rec = info["records"][0]
    assert rec["judgment_verdict_counts"].get("requires_read") == 1
    assert rec["calls_by_tool"].get("firecrawl_scrape") == 1
    cache = info["cache_files"][0]
    assert cache["any_raw"] is True and cache["store_raw_header"] is True
    assert info["live_model_calls"] == 0


# ----------------------------------------------------------------- coverage-bounded surfacing
def test_reconstruction_coverage_bounded_is_surfaced():
    rec = _record()
    rec["candidate_frontier"]["events_truncated"] = 123   # simulate a capped event list
    _, coverage = _reconstruct_obligations(rec)
    assert coverage["reconstruction_coverage_bounded"] is True
    assert coverage["events_truncated"] == 123
