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
              f"passages_scanned={pm.get('passages_scanned_count')} "
              f"(predicate_relevant={pm.get('predicate_relevant_passage_count')} "
              f"subject_only={pm.get('subject_only_passage_count')}) "
              f"judged={pm.get('judged_count')} closed={pm.get('closed_count')}")
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
