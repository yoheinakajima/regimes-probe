#!/usr/bin/env python
"""Level 5j offline validation: replay the read→judge loop closure + quantify 5i mechanism
effects on FIXED fixtures, with zero live provider/model calls.

    python scripts/validate_read_judge_replay.py
    python scripts/validate_read_judge_replay.py --artifacts results/live/<run-id>

Honesty rule: when the referenced live artifacts/cache are absent, the run reports
``unvalidated_cache_miss`` for that path — it never silently passes. Nothing here claims
benchmark accuracy, memory learning, or generalization; the outputs are mechanism/debug signals.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regimes_probe.eval.replay_validation import (   # noqa: E402
    inspect_run_schema, run_replay, run_triage_promotion_fixture, validate_artifacts_dir)

_READ_FIXTURE = ROOT / "fixtures" / "replay" / "read_judge_loop_fixture.json"
_TRIAGE_FIXTURE = ROOT / "fixtures" / "replay" / "triage_promotion_fixture.json"
#: the live path referenced by the 5j request (gitignored live outputs; usually absent here).
_DEFAULT_ARTIFACTS = "results/live/browsecomp-llm-evidence-judge-5g-smoke-001"


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline read→judge replay validation (5j).")
    ap.add_argument("--artifacts", default=_DEFAULT_ARTIFACTS,
                    help="recorded run dir to replay (reconstructs from debug_questions.jsonl "
                         "+ caches; reports per-stage reasons; unvalidated_cache_miss if absent)")
    ap.add_argument("--json", action="store_true", help="emit a machine-readable JSON blob")
    ap.add_argument("--allow-live-judge", action="store_true",
                    help="OPT-IN: allow ONLY the targeted re-judgment to call the model live "
                         "(all tool/provider data stays from cache); off by default")
    ap.add_argument("--max-judge-calls", type=int, default=20,
                    help="hard cap on live re-judgment calls (fail-closed when reached)")
    ap.add_argument("--inspect-schema", action="store_true",
                    help="light zero-call schema probe of the artifacts dir (drift safety net)")
    args = ap.parse_args()

    # --inspect-schema: a light, zero-call diagnostic (not the primary mechanism).
    if args.inspect_schema:
        info = inspect_run_schema(args.artifacts)
        print(json.dumps(info, indent=2))
        return 0

    out: dict = {}

    # A — replay/fork of the read→judge loop on the committed synthetic fixture.
    a = run_replay(json.loads(_READ_FIXTURE.read_text(encoding="utf-8")))
    out["A_read_judge_replay"] = a.to_dict()
    # B — truncation boundary finding (derived from the obligation's first-hit offset + the
    #     known adapter cap). Proves extract_passages sees beyond char 4000 when the body allows.
    ob = a.obligations[0] if a.obligations else None
    out["B_truncation_boundary"] = {
        "cap_location": "page_fetch_adapter",
        "cap_field": "regimes_probe.tools.page_fetch.PageFetch.max_chars",
        "cap_default_chars": 4000,
        "cap_is_configurable": True,
        "passage_first_hit_offset": (ob.first_hit_offset if ob else None),
        "passage_extraction_sees_beyond_4000": bool(ob and ob.first_hit_offset > 4000),
        "note": ("the 4000-char cap is an ADAPTER cap (page_fetch.max_chars), applied before "
                 "the text is stored/judged; extract_passages can find post-4000 text only when "
                 "the adapter cap is raised or a fuller body is provided — see fetch_meta."),
        "invariants": {"judge_reused_truncated_excerpt_after_full_read_count": 0,
                       "read_head_only_judgment_count": 0},
    }
    # C — fixture-level mechanism effects (triage savings + promotion safety).
    out["C_fixture_effects"] = run_triage_promotion_fixture(
        json.loads(_TRIAGE_FIXTURE.read_text(encoding="utf-8")))
    # Real-artifacts replay: reconstruct from a legacy run dir (per-stage reasons), honest
    # cache-miss only when nothing is inspectable. Live re-judgment is OPT-IN + capped.
    av = validate_artifacts_dir(args.artifacts, allow_live_judge=args.allow_live_judge,
                                max_judge_calls=args.max_judge_calls)
    out["artifacts_replay"] = {"path": args.artifacts, **av.to_dict()}
    out["live_calls"] = {"provider": int(av.metrics.get("live_provider_calls", 0)),
                         "model": int(av.metrics.get("live_model_calls", 0))}
    # 5l-6: the live-judge tier state is taken from the loader (unambiguous: enabled reflects
    # the flag, with a skip reason when there is nothing to judge).
    tier = dict(av.metrics.get("live_judge_tier")
                or {"enabled": bool(args.allow_live_judge), "live_judge_skipped_reason": None})
    tier.update({"max_judge_calls": args.max_judge_calls, "default_is_offline": True})
    out["live_judge_tier"] = tier

    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    print("== 5j offline read→judge validation (zero live calls) ==\n")
    print(f"A. read→judge replay (fixture): overall_status={a.overall_status}")
    for o in a.obligations:
        print(f"   - {o.constraint_id}: {o.closure_code} "
              f"(read_replayed={o.read_replayed}, anchor_hits={o.passage_anchor_hits}, "
              f"first_hit_offset={o.first_hit_offset})")
    print(f"   metrics: {json.dumps(a.metrics)}")
    b = out["B_truncation_boundary"]
    print(f"\nB. truncation boundary: cap={b['cap_default_chars']} @ {b['cap_location']} "
          f"({b['cap_field']}); sees_beyond_4000={b['passage_extraction_sees_beyond_4000']} "
          f"(first_hit_offset={b['passage_first_hit_offset']})")
    c = out["C_fixture_effects"]
    print(f"\nC. fixture effects: judge_calls {c['fixture_judge_calls_before_triage_proxy']}→"
          f"{c['fixture_judge_calls_after_triage']} "
          f"(reduction={c['fixture_judge_call_reduction_ratio']}); "
          f"valid_candidate_regression={c['fixture_valid_candidate_verdict_regression_count']}; "
          f"generic/title/chrome promotions={c['fixture_generic_source_candidate_promotion_count']}; "
          f"location_mismatch_promoted={c['explicit_location_mismatch_promoted_count']}")
    print(f"\nReal artifacts ({args.artifacts}): {av.overall_status}")
    if av.obligations:
        pm = av.metrics
        print(f"   reconstructed={pm.get('reconstructed_count')} "
              f"actual_read_bodies={pm.get('body_located_count')} "
              f"(read_call_matched={pm.get('matched_read_call_count')} "
              f"read_body_matched={pm.get('matched_read_body_count')}) "
              f"passages_scanned={pm.get('passages_scanned_count')} "
              f"(predicate_relevant={pm.get('predicate_relevant_passage_count')} "
              f"subject_only={pm.get('subject_only_passage_count')} "
              f"weak_predicate={pm.get('weak_predicate_candidate_only_count')}) "
              f"judged={pm.get('judged_count')} "
              f"(unclosed={pm.get('judged_unclosed_count')}) closed={pm.get('closed_count')}")
        print(f"   closure_counts: {json.dumps(pm.get('closure_counts', {}))} "
              f"target_relevance: {json.dumps(pm.get('target_passage_relevance_counts', {}))}")
        print(f"   reconstruction_source={pm.get('reconstruction_source')} "
              f"(native={pm.get('native_pending_read_judgment_count')} "
              f"legacy={pm.get('legacy_reconstructed_pending_read_judgment_count')}); "
              f"urls: obligations={pm.get('obligation_source_url_count')} "
              f"read_bodies={pm.get('read_body_url_count')} "
              f"unmatched_obligations={pm.get('obligations_without_read_body_count')} "
              f"unmatched_bodies={pm.get('read_body_urls_without_obligations_count')} "
              f"unlinked_detector={pm.get('read_body_unlinked_to_requires_read_obligation')}")
        if pm.get("unmatched_obligation_source_urls_sample"):
            print(f"   unmatched obligation urls: "
                  f"{pm['unmatched_obligation_source_urls_sample']}")
        if pm.get("unmatched_read_body_urls_sample"):
            print(f"   unmatched read-body urls: {pm['unmatched_read_body_urls_sample']}")
        print(f"   targeting: success_rate={pm.get('pending_read_targeting_success_rate')} "
              f"body_link_rate={pm.get('pending_read_body_link_rate')} "
              f"not_targeted={pm.get('pending_read_not_targeted_count')} "
              f"(event-derived; events={pm.get('not_targeted_read_events_total')})")
        print(f"   unlinked read bodies: total={pm.get('unlinked_read_body_count')} "
              f"while_pending={pm.get('unlinked_read_body_while_pending_count')} "
              f"after_pending={pm.get('unlinked_read_body_after_pending_count')}"
              + (f"; note: {pm.get('body_match_explanation')}"
                 if pm.get('body_match_explanation') else ""))
        print(f"   predicate passages: relevant={pm.get('actual_body_predicate_relevant_count')} "
              f"subject_only={pm.get('actual_body_subject_only_count')} "
              f"no_anchor={pm.get('actual_body_no_relevant_anchor_count')} "
              f"rate={pm.get('predicate_passage_relevance_rate')} "
              f"rescued={pm.get('passage_rescue_used_count')}; "
              f"diag_reasons={json.dumps(pm.get('predicate_passage_diagnostic_reason_counts', {}))}")
        print(f"   snippet-only pendings: total="
              f"{pm.get('pending_obligations_search_snippet_only_count')} "
              f"read_attempted={pm.get('pending_search_snippet_only_read_attempted_count')} "
              f"suppressed={pm.get('pending_search_snippet_only_suppressed_count')}")
        print(f"   pending service (5t): urls={pm.get('pending_service_url_count')} "
              f"attempted={pm.get('pending_service_url_attempted_count')} "
              f"success={pm.get('pending_service_url_success_count')} "
              f"blocked={pm.get('pending_service_url_blocked_count')} "
              f"budget_exhausted={pm.get('pending_service_url_budget_exhausted_count')}; "
              f"obligations={pm.get('pending_service_obligation_count')} "
              f"served={pm.get('pending_service_obligations_served_by_successful_read_count')} "
              f"url_rate={pm.get('pending_service_url_success_rate')} "
              f"obligation_rate={pm.get('pending_service_obligation_success_rate')} "
              f"no_reason={pm.get('pending_obligation_without_service_reason_count')}")
        print(f"   service statuses (5u): {json.dumps(pm.get('service_status_counts', {}))}")
        print(f"   service blocks (5u): "
              f"reasons={json.dumps(pm.get('service_block_reason_counts', {}))} "
              f"urls(suppressed={pm.get('pending_service_url_suppressed_count')} "
              f"already_satisfied={pm.get('pending_service_url_already_satisfied_count')} "
              f"invariant={pm.get('pending_service_url_invariant_violation_count')}); "
              f"service_invariant_violations={pm.get('service_invariant_violation_count')} "
              f"policy_error={pm.get('pending_service_not_attempted_policy_error_count')}")
        if pm.get("blocked_service_urls_sample"):
            print(f"   blocked service urls: "
                  f"{json.dumps(pm['blocked_service_urls_sample'])}")
        if pm.get("service_invariant_violation_samples"):
            print(f"   service invariant violations: "
                  f"{json.dumps(pm['service_invariant_violation_samples'])}")
        print(f"   rejudgment lifecycle (5u): "
              f"{json.dumps(pm.get('rejudgment_status_counts', {}))} "
              f"invariant_violations={pm.get('rejudgment_invariant_violation_count')}")
        if pm.get("rejudgment_invariant_violation_samples"):
            print(f"   rejudgment invariant violations: "
                  f"{json.dumps(pm['rejudgment_invariant_violation_samples'])}")
        if pm.get("rejudgment_cache_write_failed_samples"):
            print(f"   rejudgment cache-write failures: "
                  f"{json.dumps(pm['rejudgment_cache_write_failed_samples'])}")
        print(f"   pinned (5u): unrelated_read_executed_while_pending="
              f"{pm.get('unrelated_read_executed_while_pending_count')} "
              f"debug_snippet_judged={pm.get('debug_snippet_judged_count')} "
              f"still_open_counted_closed="
              f"{pm.get('requires_read_still_open_counted_closed_count')}")
        print(f"   targeted rejudgment (5t): body_available="
              f"{pm.get('pending_read_body_available_count')} "
              f"passages={pm.get('pending_read_body_passage_extracted_count')} "
              f"predicate_relevant={pm.get('pending_read_body_predicate_relevant_count')} "
              f"attempted={pm.get('pending_read_targeted_rejudgment_attempted_count')} "
              f"recorded={pm.get('pending_read_targeted_rejudgment_recorded_count')} "
              f"closed={pm.get('pending_read_targeted_rejudgment_closed_count')} "
              f"still_open={pm.get('pending_read_targeted_rejudgment_still_open_count')} "
              f"missing={pm.get('pending_read_targeted_rejudgment_missing_count')}")
        print(f"   fallback/reread (5t): primary_failed="
              f"{pm.get('pending_read_primary_failed_count')} "
              f"fallback_attempted={pm.get('pending_read_fallback_attempted_count')} "
              f"fallback_success={pm.get('pending_read_fallback_success_count')} "
              f"zero_chars={pm.get('pending_read_zero_chars_count')}; "
              f"reread scheduled={pm.get('predicate_reread_scheduled_count')} "
              f"attempted={pm.get('predicate_reread_attempted_count')} "
              f"blocked={pm.get('predicate_reread_blocked_count')} "
              f"outcomes={pm.get('predicate_reread_outcome_recorded_count')}")
        print(f"   read-body provenance (5v): manifest={pm.get('read_body_manifest_count')} "
              f"verified_by_body_id={pm.get('recorded_rejudgment_verified_by_body_id_count')} "
              f"verified_by_url_fallback="
              f"{pm.get('recorded_rejudgment_verified_by_url_fallback_count')} "
              f"body_hash_mismatch={pm.get('recorded_rejudgment_body_hash_mismatch_count')} "
              f"body_not_found={pm.get('recorded_rejudgment_body_not_found_count')} "
              f"unverifiable={pm.get('recorded_rejudgment_unverifiable_count')}")
        print(f"   body-hash mismatch (5v.1): "
              f"obligations={pm.get('recorded_rejudgment_body_hash_mismatch_obligation_count')} "
              f"read_bodies={pm.get('recorded_rejudgment_body_hash_mismatch_read_body_count')} "
              f"verify_success={pm.get('read_body_id_verification_success_count')} "
              f"verify_failure={pm.get('read_body_id_verification_failure_count')} "
              f"no_class={pm.get('unverifiable_without_mismatch_class_count')}; "
              f"cache_differs={pm.get('cache_body_differs_from_manifest_count')} "
              f"failure_classes="
              f"{json.dumps(pm.get('read_body_id_verification_failure_class_counts', {}))}")
        if pm.get("recorded_rejudgment_read_body_mismatch_diagnostics"):
            print(f"   read-body mismatch diagnostics: "
                  f"{json.dumps(pm['recorded_rejudgment_read_body_mismatch_diagnostics'])}")
        if pm.get("recorded_rejudgment_body_mismatch_samples"):
            print(f"   body-mismatch samples: "
                  f"{json.dumps(pm['recorded_rejudgment_body_mismatch_samples'])}")
        print(f"   rejudgment outcomes (5v): "
              f"constraint_resolving={pm.get('rejudgment_constraint_resolved_count')} "
              f"source_terminal_non_support={pm.get('rejudgment_terminal_non_support_count')} "
              f"requires_read_still_open={pm.get('rejudgment_requires_read_still_open_count')} "
              f"unverifiable={pm.get('rejudgment_unverifiable_count')}; "
              f"closed(constraint_resolving)={pm.get('closed_count')} "
              f"source_obligation_terminal={pm.get('source_obligation_terminal_count')}")
        print(f"   evidence gaps (5v): count={pm.get('evidence_gap_count')} "
              f"reasons={json.dumps(pm.get('evidence_gap_reason_counts', {}))} "
              f"search_alternate={pm.get('evidence_gap_search_alternate_count')}")
        from regimes_probe.eval.replay_validation import consistency_violations
        viol = consistency_violations(pm)
        print(f"   internal consistency: {'OK' if not viol else 'VIOLATIONS: ' + '; '.join(viol)}")
        print(f"   diagnostic-only (never counted as bodies): "
              f"search_snippets={pm.get('search_snippet_only_count')} "
              f"debug_snippets={pm.get('debug_snippet_scanned_count')}; "
              f"beyond_4000={pm.get('passages_found_beyond_4000_count')} "
              f"rejudge_version_mismatch="
              f"{pm.get('recorded_rejudgment_cache_version_mismatch_count')}")
        print(f"   body_sources: {json.dumps(pm.get('body_source_counts', {}))} "
              f"url_match: {json.dumps(pm.get('url_match_method_counts', {}))}")
        print(f"   stage_reasons: {json.dumps(pm.get('stage_reason_counts', {}))}")
        cr = pm.get("cache_report", {})
        if cr:
            print(f"   cache: files={cr.get('n_files')} entries={cr.get('n_entries')} "
                  f"read_body={json.dumps(cr.get('read_body_entries_by_provider', {}))} "
                  f"search={json.dumps(cr.get('search_snippet_entries_by_provider', {}))} "
                  f"raw_entries={cr.get('raw_payload_entries')} "
                  f"unrecognized={cr.get('unrecognized_schema_count')}")
    for nnote in av.notes:
        print(f"   note: {nnote}")
    live_n = int(av.metrics.get("live_model_calls", 0))
    skip = tier.get("live_judge_skipped_reason")
    print(f"\nLive judge tier: enabled={tier.get('enabled')} "
          f"(default offline){f'; skipped: {skip}' if skip else ''}; "
          f"live_model_calls={live_n}, live_provider_calls=0.")
    if not args.allow_live_judge:
        print("No live provider/model calls were made.")
    print("No benchmark/accuracy/memory/generalization claim is made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
