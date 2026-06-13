"""Level 5o offline tests: honest rejudgment stage reasons (closed_by_* only on genuinely
resolving closures), the matched_read_call vs matched_read_body split, and target-anchor
tightening for answer-shaped constraints. Zero live calls in default mode; the 'live judge'
in tests is a deterministic local stub, never a network call.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from regimes_probe.eval.replay_validation import load_legacy_run

ROOT = Path(__file__).resolve().parents[1]
_LEGACY = ROOT / "fixtures" / "replay" / "legacy_run"


def _copy(tmp_path) -> Path:
    dst = tmp_path / "run"
    shutil.copytree(_LEGACY, dst)
    return dst


def _set_cache_body(run: Path, body: str, *, keep_raw: bool = False) -> None:
    cache = json.loads((run / "cache" / "provider_cache.json").read_text())
    cache["entries"][0]["response"]["results"][0]["snippet"] = body
    if keep_raw:
        cache["entries"][0]["raw"] = body
    else:
        cache["entries"][0].pop("raw", None)
    (run / "cache" / "provider_cache.json").write_text(json.dumps(cache))


def _strip_calls(run: Path) -> None:
    rec = json.loads((run / "debug_questions.jsonl").read_text().splitlines()[0])
    rec["calls"] = []
    (run / "debug_questions.jsonl").write_text(json.dumps(rec) + "\n")


# ----------------------------------------------------------------- 1: honest stage reasons
def test_non_closing_rejudgment_never_uses_closed_by_stage_reason(tmp_path):
    run = _copy(tmp_path)
    # body mentions the institute + year but NOT the candidate -> stub judge says
    # requires_read -> judged_unclosed, never "closed_by_*".
    _set_cache_body(run, ("filler. " * 200)
                    + " The Brightmoor Institute opened a 1992 archive page. ")
    res = load_legacy_run(run, allow_live_judge=True, max_judge_calls=20)
    judged = [o for o in res.obligations
              if o.pipeline_status in ("judged_unclosed", "closed")]
    assert judged
    for o in judged:
        if o.closure_code == "requires_read_still_open":
            assert o.pipeline_status == "judged_unclosed"
            assert o.stage_reason == "judged_by_live_rejudgment_still_open"
            assert not o.stage_reason.startswith("closed_by")
    assert res.metrics["closed_count"] == 0
    assert res.overall_status == "judged_unclosed"


def test_closing_rejudgment_uses_closed_by_and_counts(tmp_path):
    run = _copy(tmp_path)                          # fixture body closes (full_support)
    res = load_legacy_run(run, allow_live_judge=True, max_judge_calls=20)
    closed = [o for o in res.obligations if o.pipeline_status == "closed"]
    assert closed
    for o in closed:
        assert o.closure_code in ("resolved_full_support", "resolved_contradiction")
        assert o.stage_reason == "closed_by_live_rejudgment"
    assert res.metrics["closed_count"] == len(closed)
    assert res.overall_status == "validated_closed"


def test_closure_counts_and_judged_unclosed_derived_from_closure_code(tmp_path):
    run = _copy(tmp_path)
    res = load_legacy_run(run, allow_live_judge=True, max_judge_calls=20)
    cc = res.metrics["closure_counts"]
    # closure_counts is a projection of per-obligation closure_code values.
    from collections import Counter
    expected = Counter(o.closure_code for o in res.obligations
                       if o.closure_code != "unvalidated_cache_miss")
    assert cc == dict(expected)
    assert res.metrics["judged_unclosed_count"] == sum(
        1 for o in res.obligations if o.pipeline_status == "judged_unclosed")
    # offline replay after live judging: recorded verdicts respected, still no closed_by_*
    # misuse, zero calls.
    r2 = load_legacy_run(run)
    assert r2.metrics["live_model_calls"] == 0
    for o in r2.obligations:
        # 5v-2: precise rejudgment-outcome stage reasons (still-open is requires_read only).
        if o.pipeline_status == "judged_unclosed":
            assert o.stage_reason == "judged_requires_more_evidence"
            assert o.closure_code == "requires_read_still_open"
        if o.pipeline_status == "closed":
            assert o.stage_reason in ("judged_resolved_full_support",
                                      "judged_resolved_contradiction")
        if o.pipeline_status == "source_terminal":
            assert o.stage_reason in ("judged_source_irrelevant_terminal",
                                      "judged_partial_terminal_non_support")


# ----------------------------------------------------------------- 2: call vs body split
def test_matched_read_call_vs_matched_read_body_split_on_cache_only_body(tmp_path):
    run = _copy(tmp_path)
    _strip_calls(run)                              # legacy run: body in cache, call metadata gone
    res = load_legacy_run(run)
    o = res.obligations[0]
    assert o.matched_read_call is False            # no read call survived in the record
    assert o.matched_read is False                 # back-compat alias of matched_read_call
    assert o.matched_read_body is True             # but the read-class cache body IS located
    assert o.body_is_actual_read_body and o.body_source == "cache_read_body"
    assert o.read_body_chars > 0 and o.read_body_provider == "firecrawl_scrape"
    m = res.metrics
    assert m["matched_read_call_count"] == 0 and m["matched_read_body_count"] >= 1
    assert m["matched_read_true_without_read_call_or_read_body_count"] == 0
    assert m["actual_body_located_without_matched_read_body_count"] == 0


def test_matched_read_invariants_pinned_zero_on_fixture():
    res = load_legacy_run(_LEGACY)
    m = res.metrics
    assert m["matched_read_true_without_read_call_or_read_body_count"] == 0
    assert m["actual_body_located_without_matched_read_body_count"] == 0
    assert m["matched_read_with_no_url_match_method_count"] == 0
    o = res.obligations[0]
    assert o.matched_read_call and o.matched_read_body     # fixture has both


# ----------------------------------------------------------------- 3: target-anchor tightening
def test_target_passage_with_only_alias_and_facet_is_not_judge_eligible(tmp_path):
    run = _copy(tmp_path)
    # candidate named + a generic constraint FACET word ("studied") but NO target descriptor,
    # NO year, NO relation verb tied to the target... "studied" IS a relation verb. Use a
    # facet word that is not a predicate verb: "institute" (constraint_label). Candidate +
    # institute only, no year/descriptor/verb.
    _set_cache_body(run, ("filler. " * 200)
                    + " Jordan Vale institute page overview and general links. ")
    res = load_legacy_run(run)
    o = res.obligations[0]
    assert o.is_target_constraint is True
    assert o.passage_relevance == "weak_predicate_candidate_only"
    assert o.pipeline_status == "actual_body_located"      # NOT passages_scanned
    assert o.stage_reason in ("weak_predicate_candidate_only",
                              "body_truncated_before_relevant_passage(raw_unavailable)")
    assert res.metrics["passages_scanned_count"] == 0
    assert "weak_predicate_candidate_only" in (
        res.metrics["target_passage_relevance_counts"] | {"weak_predicate_candidate_only": 0})
    # live judge refuses it.
    r2 = load_legacy_run(run, allow_live_judge=True, max_judge_calls=20)
    assert r2.metrics["live_model_calls"] == 0
    assert r2.metrics["live_judge_tier"]["live_judge_skipped_reason"] == \
        "no_actual_body_predicate_relevant_passages_to_judge"


def test_target_passage_with_year_anchor_is_judge_eligible(tmp_path):
    run = _copy(tmp_path)
    # candidate + the constraint's YEAR -> a real target anchor -> judge-eligible.
    _set_cache_body(run, ("filler. " * 200)
                    + " Jordan Vale studied at the Brightmoor Institute in 1992. ",
                    keep_raw=True)
    res = load_legacy_run(run)
    o = res.obligations[0]
    assert o.passage_relevance == "predicate_relevant"
    assert o.anchor_category_counts.get("numeric_or_year", 0) >= 1
    assert o.pipeline_status in ("passages_scanned", "rejudgment_pending")
    r2 = load_legacy_run(run, allow_live_judge=True, max_judge_calls=20)
    assert r2.metrics["live_model_calls"] >= 1
    assert any(x.pipeline_status == "closed" for x in r2.obligations)


def test_target_relevance_counts_emitted():
    res = load_legacy_run(_LEGACY)
    assert "target_passage_relevance_counts" in res.metrics
    assert res.metrics["target_passage_relevance_counts"].get("predicate_relevant", 0) >= 1


# ----------------------------------------------------------------- 4: persistence flags
def test_run_live_read_persistence_includes_caps():
    src = (ROOT / "scripts" / "run_live.py").read_text()
    for key in ("read_cache_store_raw", "read_cache_raw_chars", "read_max_chars",
                "read_judgment_max_chars", "firecrawl_scrape_max_chars",
                "page_fetch_max_chars"):
        assert key in src, key


# ----------------------------------------------------------------- default mode zero calls
def test_default_validator_zero_live_calls(tmp_path):
    run = _copy(tmp_path)
    res = load_legacy_run(run)
    assert res.metrics["live_provider_calls"] == 0
    assert res.metrics["live_model_calls"] == 0
    assert res.metrics["live_judge_tier"]["enabled"] is False
