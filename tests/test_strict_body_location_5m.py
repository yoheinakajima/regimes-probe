"""Level 5m offline tests: STRICT body-location semantics for real-artifact replay.

A debug snippet is never a body: it must not advance an obligation to body_located /
passages_scanned, must not trigger a truncation claim, and must not be live-judged by
default. matched_read requires a URL relation. The provider recording cache under cache/
is inspected recursively and reported. Zero live provider/model calls in default mode.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from regimes_probe.eval.replay_validation import (
    inspect_run_schema, load_legacy_run)

ROOT = Path(__file__).resolve().parents[1]
_LEGACY = ROOT / "fixtures" / "replay" / "legacy_run"


def _copy(tmp_path) -> Path:
    dst = tmp_path / "run"
    shutil.copytree(_LEGACY, dst)
    return dst


def _strip_provider_cache(run: Path) -> None:
    (run / "cache" / "provider_cache.json").write_text(json.dumps({"entries": []}))


def _record(run: Path) -> dict:
    return json.loads((run / "debug_questions.jsonl").read_text().splitlines()[0])


def _write_record(run: Path, rec: dict) -> None:
    (run / "debug_questions.jsonl").write_text(json.dumps(rec) + "\n")


# --------------------------------------------------------------- 2: locator priority + strictness
def test_cached_body_advances_to_body_located_and_passages_scanned():
    res = load_legacy_run(_LEGACY)
    o = res.obligations[0]
    assert o.body_source == "cache_read_body" and o.body_is_actual_read_body
    assert o.pipeline_status in ("passages_scanned", "rejudgment_pending")
    assert res.metrics["body_located_count"] >= 1
    assert res.metrics["passages_scanned_count"] >= 1
    assert res.metrics["live_provider_calls"] == 0 and res.metrics["live_model_calls"] == 0


def test_call_embedded_body_is_recognised(tmp_path):
    run = _copy(tmp_path)
    _strip_provider_cache(run)                      # no cache body
    rec = _record(run)
    # embed a REAL body on the read call itself (a future run may persist it there).
    url = rec["calls"][0]["evidence_record"]["url"]
    body = ("menu boilerplate " * 60) + " Jordan Vale studied at the Brightmoor Institute in 1992."
    rec["calls"][0]["evidence_record"]["body"] = body
    _write_record(run, rec)
    res = load_legacy_run(run)
    o = res.obligations[0]
    assert o.body_source == "call_embedded_read_body"
    assert o.body_match_source == "read_call"
    assert o.pipeline_status in ("passages_scanned", "rejudgment_pending")
    assert o.passage_anchor_hits >= 1
    assert o.matched_body_url and o.matched_read_url


def test_debug_snippet_does_not_count_as_body_or_passages(tmp_path):
    run = _copy(tmp_path)
    _strip_provider_cache(run)                      # only the debug snippet preview remains
    res = load_legacy_run(run)
    assert res.metrics["body_located_count"] == 0
    assert res.metrics["passages_scanned_count"] == 0
    assert res.metrics["debug_snippet_scanned_count"] >= 1
    o = res.obligations[0]
    assert o.body_source == "debug_snippet_only"
    assert o.pipeline_status == "debug_snippet_scanned"
    assert o.stage_reason in ("read_body_not_persisted_legacy_run",
                              "debug_snippet_only_no_body",
                              "read_event_found_but_body_missing")
    assert res.overall_status == "reconstructed_debug_only"   # never "validated"


# --------------------------------------------------------------- 3: URL matching invariant
def test_matched_read_always_has_url_match_method():
    res = load_legacy_run(_LEGACY)
    assert res.metrics["matched_read_with_no_url_match_method_count"] == 0
    for o in res.obligations:
        if o.matched_read:
            assert o.url_match_method in ("exact", "prefix", "host_only")
            assert o.matched_read_url


def test_read_on_unrelated_host_is_not_matched(tmp_path):
    run = _copy(tmp_path)
    _strip_provider_cache(run)                      # no read-provider cache match either
    rec = _record(run)
    # point the only read call at a DIFFERENT host -> no URL relation -> matched_read=False
    # (5n-2: neither a read-class call nor a read-provider cache entry matches by URL).
    rec["calls"][0]["evidence_record"]["url"] = "https://other.example/unrelated/page"
    rec["calls"][0]["task_action"]["query_text_preview"] = "https://other.example/unrelated/page"
    _write_record(run, rec)
    res = load_legacy_run(run)
    assert all(not o.matched_read for o in res.obligations)
    assert res.metrics["matched_read_with_no_url_match_method_count"] == 0


# --------------------------------------------------------------- 4: truncation requires a body
def test_truncation_claim_never_from_debug_snippet(tmp_path):
    run = _copy(tmp_path)
    _strip_provider_cache(run)
    res = load_legacy_run(run)
    assert res.metrics["body_truncated_before_relevant_passage_count"] == 0
    assert all(not o.body_truncated_before_relevant_passage for o in res.obligations)


def test_truncation_claim_for_actual_capped_body_without_raw(tmp_path):
    run = _copy(tmp_path)
    rec = _record(run)
    url = rec["calls"][0]["evidence_record"]["url"].rstrip(".")
    full_url = json.loads((run / "cache" / "provider_cache.json").read_text())[
        "entries"][0]["response"]["query"]
    # a 4000-char stored body with NO anchors and NO raw payload -> honest truncation claim.
    capped = ("irrelevant filler words here. " * 200)[:4000]
    cache = {"mode": "auto", "store_raw": False, "entries": [
        {"request_hash": "h1", "provider": "firecrawl_scrape", "name": "firecrawl_scrape",
         "request_meta": {"query": full_url},
         "response": {"provider": "firecrawl_scrape", "query": full_url,
                      "results": [{"title": "t", "url": full_url, "snippet": capped,
                                   "rank": 0}],
                      "error": None}}]}
    (run / "cache" / "provider_cache.json").write_text(json.dumps(cache))
    res = load_legacy_run(run)
    o = res.obligations[0]
    assert o.body_source == "cache_read_body" and o.raw_unavailable
    assert o.body_truncated_before_relevant_passage is True
    assert o.stage_reason == "body_truncated_before_relevant_passage(raw_unavailable)"


# --------------------------------------------------------------- 6: live-judge strictness
def test_live_judge_skips_when_only_debug_snippets(tmp_path):
    run = _copy(tmp_path)
    _strip_provider_cache(run)
    res = load_legacy_run(run, allow_live_judge=True, max_judge_calls=20)
    tier = res.metrics["live_judge_tier"]
    assert tier["enabled"] is True
    assert tier["live_judge_skipped_reason"] == "no_actual_body_predicate_relevant_passages_to_judge"
    assert res.metrics["live_model_calls"] == 0
    assert all(not o.live_rejudgment_source or o.live_rejudgment_source != "debug_snippet_only"
               for o in res.obligations)


def test_live_judge_records_source_for_actual_body(tmp_path):
    run = _copy(tmp_path)
    res = load_legacy_run(run, allow_live_judge=True, max_judge_calls=20)
    judged = [o for o in res.obligations
              if o.pipeline_status in ("judged_unclosed", "closed")]
    assert judged
    assert all(o.live_rejudgment_source in ("cache_read_body", "call_embedded_read_body",
                                            "replay_export_body") for o in judged)
    assert res.metrics["live_model_calls"] >= 1


# --------------------------------------------------------------- 1: cache report / inspect-schema
def test_inspect_schema_reports_provider_cache_recursively():
    info = inspect_run_schema(_LEGACY)
    cr = info["cache_report"]
    assert cr["n_files"] >= 2                       # provider cache + llm judge cache
    assert cr["entries_by_provider"].get("firecrawl_scrape") == 1
    assert cr["raw_payload_entries"] >= 1
    assert cr["urls_with_bodies"] >= 1
    files = {f["file"] for f in info["cache_files"]}
    assert any("provider_cache" in f for f in files)   # cache/ contents are surfaced
    assert info["live_model_calls"] == 0


def test_unrecognized_cache_schema_is_counted_not_ignored(tmp_path):
    run = _copy(tmp_path)
    (run / "cache" / "weird.bin").write_text("not json at all {{{")
    info = inspect_run_schema(run)
    assert info["cache_report"]["unrecognized_schema_count"] >= 1
    assert any("weird.bin" in f for f in info["cache_report"]["unrecognized_schema_files"])
