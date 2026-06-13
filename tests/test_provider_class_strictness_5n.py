"""Level 5n offline tests: provider-class body strictness + passage relevance + closure-gated
overall status + versioned rejudgment cache. A search snippet is never a body; subject/title
hits alone are never judgeable; "validated" requires actual closure. Zero live calls in
default mode; the 'live judge' in tests is a deterministic local stub, never a network call.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from regimes_probe.eval.replay_validation import (
    STRICT_REJUDGE_VERSION, _body_hash, _provider_class, inspect_run_schema, load_legacy_run)

ROOT = Path(__file__).resolve().parents[1]
_LEGACY = ROOT / "fixtures" / "replay" / "legacy_run"


def _copy(tmp_path) -> Path:
    dst = tmp_path / "run"
    shutil.copytree(_LEGACY, dst)
    return dst


def _cache(run: Path) -> dict:
    return json.loads((run / "cache" / "provider_cache.json").read_text())


def _write_cache(run: Path, cache: dict) -> None:
    (run / "cache" / "provider_cache.json").write_text(json.dumps(cache))


def _make_search_only(run: Path) -> None:
    """Convert the cached read-body entry into a serper_search snippet entry."""
    cache = _cache(run)
    e = cache["entries"][0]
    e["provider"] = e["name"] = "serper_search"
    e.pop("raw", None)
    cache["store_raw"] = False
    _write_cache(run, cache)


# ----------------------------------------------------------------- 1: provider classes
def test_provider_classification():
    assert _provider_class("firecrawl_scrape") == "read_body"
    assert _provider_class("page_fetch") == "read_body"
    assert _provider_class("serper_search") == "search_snippet"
    assert _provider_class("exa_search") == "search_snippet"
    assert _provider_class("firecrawl_search") == "search_snippet"
    assert _provider_class("openai_responses_evidence_judge") == "model"


# ----------------------------------------------------------------- A: search snippet ≠ body
def test_search_snippet_is_not_body(tmp_path):
    run = _copy(tmp_path)
    _make_search_only(run)
    res = load_legacy_run(run)
    o = res.obligations[0]
    assert o.body_source == "cache_search_snippet_only"
    assert o.body_is_search_snippet and not o.body_is_actual_read_body
    assert o.body_match_source == "search_provider_cache"
    assert res.metrics["body_located_count"] == 0
    assert res.metrics["passages_scanned_count"] == 0
    assert res.metrics["search_snippet_counted_as_body_count"] == 0
    assert res.metrics["search_snippet_only_count"] >= 1
    assert res.overall_status == "reconstructed_search_snippet_only"
    # the live tier skips search snippets entirely.
    r2 = load_legacy_run(run, allow_live_judge=True, max_judge_calls=20)
    tier = r2.metrics["live_judge_tier"]
    assert tier["enabled"] is True
    assert tier["live_judge_skipped_reason"] == \
        "no_actual_body_predicate_relevant_passages_to_judge"
    assert r2.metrics["live_model_calls"] == 0


# ----------------------------------------------------------------- B/H: read body IS body
def test_read_body_is_body_with_read_provider_provenance():
    res = load_legacy_run(_LEGACY)
    o = res.obligations[0]
    assert o.body_source == "cache_read_body" and o.body_is_actual_read_body
    assert o.body_match_source == "read_provider_cache"
    assert o.body_provider == "firecrawl_scrape"
    assert res.metrics["body_located_count"] >= 1
    assert res.metrics["actual_body_located_without_read_body_provenance_count"] == 0
    assert res.metrics["matched_read_with_no_url_match_method_count"] == 0


# ----------------------------------------------------------------- C: subject-only passage
def test_subject_only_passage_is_not_judgeable(tmp_path):
    run = _copy(tmp_path)
    cache = _cache(run)
    # an actual read body that names ONLY the candidate (no constraint/target/year anchors).
    body = ("profile header navigation. " * 40) + " Jordan Vale homepage and contact details. "
    cache["entries"][0]["response"]["results"][0]["snippet"] = body
    cache["entries"][0]["raw"] = body
    _write_cache(run, cache)
    res = load_legacy_run(run)
    o = res.obligations[0]
    assert o.body_is_actual_read_body
    assert o.passage_relevance == "subject_only"
    assert o.pipeline_status == "actual_body_located"          # NOT passages_scanned
    assert o.stage_reason == "subject_only_passage_no_predicate_anchor"
    assert res.metrics["subject_only_passage_count"] >= 1
    assert res.metrics["passages_scanned_count"] == 0
    # live judge refuses subject-only passages.
    r2 = load_legacy_run(run, allow_live_judge=True, max_judge_calls=20)
    assert r2.metrics["live_model_calls"] == 0
    assert r2.metrics["live_judge_tier"]["live_judge_skipped_reason"] == \
        "no_actual_body_predicate_relevant_passages_to_judge"


# ----------------------------------------------------------------- D: predicate-relevant passage
def test_predicate_relevant_passage_scans_and_live_judge_closes(tmp_path):
    run = _copy(tmp_path)
    res = load_legacy_run(run)
    o = res.obligations[0]
    assert o.passage_relevance == "predicate_relevant"          # constraint terms + year hit
    assert o.anchor_category_counts.get("constraint_label", 0) >= 1
    assert res.metrics["predicate_relevant_passage_count"] >= 1
    assert res.metrics["passages_scanned_count"] >= 1
    r2 = load_legacy_run(run, allow_live_judge=True, max_judge_calls=20)
    closed = [x for x in r2.obligations if x.pipeline_status == "closed"]
    assert closed and r2.overall_status == "validated_closed"
    assert all(x.live_rejudgment_source in ("cache_read_body", "call_embedded_read_body",
                                            "replay_export_body") for x in closed)


# ----------------------------------------------------------------- E: overall status gating
def test_overall_status_never_validated_without_closure():
    res = load_legacy_run(_LEGACY)                              # offline: nothing judged
    assert res.metrics["closed_count"] == 0
    assert res.overall_status != "validated"
    assert "validated" not in res.overall_status or res.overall_status == "validated_closed"
    assert res.overall_status == "reconstructed_actual_body_rejudgment_pending"


def test_requires_read_still_open_is_judged_unclosed(tmp_path):
    run = _copy(tmp_path)
    # a stub judge that never closes: candidate is not named in the stored body's passage.
    cache = _cache(run)
    body = ("filler text. " * 100) + " The Brightmoor Institute opened a 1992 archive page. "
    cache["entries"][0]["response"]["results"][0]["snippet"] = body
    cache["entries"][0].pop("raw", None)
    _write_cache(run, cache)
    res = load_legacy_run(run, allow_live_judge=True, max_judge_calls=20)
    judged = [o for o in res.obligations if o.pipeline_status in ("judged_unclosed", "closed")]
    assert judged
    assert any(o.pipeline_status == "judged_unclosed"
               and o.closure_code == "requires_read_still_open" for o in judged)
    if not any(o.pipeline_status == "closed" for o in res.obligations):
        assert res.overall_status == "judged_unclosed"
        assert res.metrics["closed_count"] == 0


# ----------------------------------------------------------------- F/G: rejudgment cache version
def test_legacy_rejudgment_cache_entry_is_ignored_with_version_mismatch(tmp_path):
    # 5v: a legacy pre-strict ``rejudgment::`` entry has no strict-v2 record, so it is simply
    # IGNORED — the obligation stays rejudgment_pending and is NEVER judged/closed.
    run = _copy(tmp_path)
    (run / "llm_evidence_judge_cache.json").write_text(json.dumps(
        {"rejudgment::recon_browsecomp-0000-shaped_0": "full_support"}))
    res = load_legacy_run(run)
    o = res.obligations[0]
    assert o.pipeline_status == "rejudgment_pending"            # NOT judged/closed
    assert o.closure_code != "resolved_full_support"
    assert res.metrics["closed_count"] == 0


def test_strict_rejudgment_cache_entry_with_matching_hash_is_accepted(tmp_path):
    run = _copy(tmp_path)
    cache = _cache(run)
    stored = cache["entries"][0]["response"]["results"][0]["snippet"]
    raw = cache["entries"][0]["raw"]
    body = raw if len(raw) > len(stored) else stored
    oid = "recon_browsecomp-0000-shaped_0"
    (run / "llm_evidence_judge_cache.json").write_text(json.dumps({
        f"{STRICT_REJUDGE_VERSION}::{oid}": {
            "version": STRICT_REJUDGE_VERSION, "verdict": "full_support",
            "body_source": "cache_read_body", "body_provider": "firecrawl_scrape",
            "body_hash": _body_hash(body), "slot_id": "s0", "constraint_id": "c1"}}))
    res = load_legacy_run(run)
    o = next(x for x in res.obligations if x.pending_read_judgment_id == oid)
    assert o.pipeline_status == "closed" and o.closure_code == "resolved_full_support"
    assert o.live_rejudgment_source == "recorded_rejudgment_cache"
    assert res.overall_status == "validated_closed"
    assert res.metrics["live_model_calls"] == 0                  # accepted OFFLINE


def test_strict_entry_with_wrong_body_hash_is_version_mismatch(tmp_path):
    # 5v-1: a strict entry whose body hash does not match the located body is UNVERIFIABLE
    # (body_hash_mismatch) — precise, never judged/closed, never silently pending.
    run = _copy(tmp_path)
    oid = "recon_browsecomp-0000-shaped_0"
    (run / "llm_evidence_judge_cache.json").write_text(json.dumps({
        f"{STRICT_REJUDGE_VERSION}::{oid}": {
            "version": STRICT_REJUDGE_VERSION, "verdict": "full_support",
            "body_source": "cache_read_body", "body_provider": "firecrawl_scrape",
            "body_hash": "deadbeefdeadbeef", "slot_id": "s0", "constraint_id": "c1"}}))
    res = load_legacy_run(run)
    o = next(x for x in res.obligations if x.pending_read_judgment_id == oid)
    assert o.pipeline_status == "rejudgment_unverifiable"
    assert o.rejudgment_unverifiable_reason == "body_hash_mismatch"
    assert res.metrics["recorded_rejudgment_body_hash_mismatch_count"] >= 1
    assert res.metrics["closed_count"] == 0


# ----------------------------------------------------------------- 7: inspect-schema split
def test_inspect_schema_splits_provider_classes(tmp_path):
    run = _copy(tmp_path)
    cache = _cache(run)
    cache["entries"].append({
        "request_hash": "h2", "provider": "serper_search", "name": "serper_search",
        "request_meta": {"query": "some query"},
        "response": {"provider": "serper_search", "query": "some query",
                     "results": [{"title": "t", "url": "https://elsewhere.example/r",
                                  "snippet": "a search snippet", "rank": 0}], "error": None}})
    _write_cache(run, cache)
    info = inspect_run_schema(run)
    cr = info["cache_report"]
    assert cr["read_body_entries_by_provider"].get("firecrawl_scrape") == 1
    assert cr["search_snippet_entries_by_provider"].get("serper_search") == 1
    assert cr["urls_with_actual_read_bodies"] >= 1
    assert cr["urls_with_search_snippets_only"] >= 1
    assert info["live_model_calls"] == 0


# ----------------------------------------------------------------- 8: run_live config exposure
def test_run_live_exposes_read_persistence_flags():
    src = (ROOT / "scripts" / "run_live.py").read_text()
    assert "--read-cache-store-raw" in src
    assert "--read-max-chars" in src
    assert '"read_persistence"' in src                           # captured in plan/manifest


# ----------------------------------------------------------------- I: default-mode zero calls
def test_default_mode_zero_live_calls_everywhere(tmp_path):
    for variant in ("as_is", "search_only"):
        run = _copy(tmp_path / variant)
        if variant == "search_only":
            _make_search_only(run)
        res = load_legacy_run(run)
        assert res.metrics["live_provider_calls"] == 0
        assert res.metrics["live_model_calls"] == 0
        assert res.metrics["live_judge_tier"]["enabled"] is False
