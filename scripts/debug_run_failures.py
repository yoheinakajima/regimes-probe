#!/usr/bin/env python
"""Human-readable failure view for a run (no network).

    python scripts/debug_run_failures.py results/live/<run_id> --limit 10
    python scripts/debug_run_failures.py results/live/<run_id> --condition no_memory_search --all

Reads results/{run_id}/debug_questions.jsonl (bounded previews) and prints, per
failed question: the question/gold/model-answer previews, tools used, top
evidence, any failed-tool errors, and the inferred failure seam.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _load(run_dir: Path) -> list[dict]:
    p = run_dir / "debug_questions.jsonl"
    if not p.exists():
        raise SystemExit(f"no debug_questions.jsonl in {run_dir} "
                         "(re-run with a build that writes it).")
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Compact failure view for a run.")
    ap.add_argument("run_dir")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--condition", default=None, help="filter to one condition")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--all", action="store_true", help="include correct rows too")
    ap.add_argument("--seam", default=None, help="filter to one failure seam")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    rows = _load(run_dir)

    rows = [r for r in rows
            if (args.all or not r.get("correct"))
            and (args.condition is None or r.get("condition") == args.condition)
            and (args.budget is None or r.get("budget") == args.budget)
            and (args.seam is None or r.get("failure_seam") == args.seam)]

    seam_counts = Counter(r.get("failure_seam", "unknown") for r in rows)
    print(f"=== {run_dir} — {len(rows)} shown ===")
    print("failure seams:", dict(seam_counts.most_common()))
    print()

    for r in rows[: args.limit]:
        print(f"[{r['item_id']}] {r['condition']}@b{r['budget']}  "
              f"correct={r['correct']} abstained={r['abstained']}  seam={r['failure_seam']}  "
              f"regime={r.get('regime','')} regimes={r.get('regime_names', [])}")
        print(f"  Q   : {r['question_preview']}")
        print(f"  gold: {r['gold_preview']}")
        print(f"  pred: {r['prediction_preview'] or '(none / abstained)'}")
        print(f"  tools: {r['tool_sequence']}  providers: {r['provider_names']}  "
              f"n_results={r['n_results']}")
        print(f"  flags: support_found={r['support_found']} found_hit={r['found_hit']} "
              f"authority_ok={r['authority_ok']} contradiction={r['contradiction']} "
              f"failed_tool_calls={r['failed_tool_calls']} "
              f"contaminated_results={r.get('contaminated_results', 0)} "
              f"stage_depth={r.get('stage_depth_used', 1)} "
              f"followups={r.get('followup_query_count', 0)}")
        if r.get("target_roles"):
            print(f"  target_roles: {r['target_roles']}")
        # Epistemic escalation: which mode was chosen and why.
        em = r.get("epistemic_mode") or {}
        if em:
            print(f"  EPISTEMIC MODE: {em.get('selected_epistemic_mode')} "
                  f"(auto={em.get('auto')}, applied={em.get('applied')}; "
                  f"{em.get('escalation_reason')})")
            if em.get("skipped_heavy_parser_reason"):
                print(f"    skipped_heavy_parser: {em.get('skipped_heavy_parser_reason')}")
        # Level 4 task frame: target/latent slots, constraints, coverage, hypotheses.
        tf = r.get("task_frame") or {}
        if tf:
            pm = r.get("task_frame_parse") or {}
            if pm:
                used = pm.get("parser_used")
                used_label = ("fallback" if (used == "deterministic" and pm.get("fallback_reason"))
                              else used)
                line = (f"  PARSER: used={used_label} model={pm.get('model')} "
                        f"quality={pm.get('parse_quality')} cache_hit={pm.get('cache_hit')}")
                if used == "llm" or pm.get("prompt_hash"):
                    line += f" prompt={pm.get('prompt_version')}#{pm.get('prompt_hash')}"
                if pm.get("fallback_reason"):
                    line += f" fallback_reason={pm.get('fallback_reason')}"
                print(line)
                if pm.get("validation_errors"):
                    print(f"    validation_errors: {pm.get('validation_errors')[:6]}")
                if pm.get("id_mapping"):
                    print(f"    id_mapping (raw->internal): {pm.get('id_mapping')}")
            print(f"  TASK FRAME: targets="
                  f"{[(s['slot_role'], s['slot_name']) for s in tf.get('target_answer_slots', [])]} "
                  f"latent={[(s['slot_role'], s['slot_name']) for s in tf.get('latent_slots', [])]}")
            for c in tf.get("constraints", [])[:8]:
                blk = " BLOCKING" if (c.get("blocks_answer_if_unresolved")
                                      or "can_block_answer" in c.get("affordances", [])) else ""
                print(f"    constraint {c['constraint_id']} [{c.get('status')}{blk}] "
                      f"label={c.get('semantic_label')!r} facets={c.get('semantic_facets', [])[:4]} "
                      f"aff={c.get('affordances', [])} -> slots {c.get('applies_to')}")
            if tf.get("validation_warnings"):
                print(f"    parser_warnings: {tf.get('validation_warnings')[:6]}")
            print(f"    known_context: {tf.get('known_context_terms', [])[:6]}")
            cov = r.get("frame_coverage", {})
            print(f"    coverage: slot_res={cov.get('slot_resolution_rate')} "
                  f"constraint_sup={cov.get('constraint_support_rate')} "
                  f"target_sup={cov.get('target_slot_support_rate')} "
                  f"terminal={cov.get('terminal_action')}")
            print(f"    answer_support_gate={cov.get('answer_support_gate')} "
                  f"answer_supported={cov.get('final_answer_supported_by_constraints')}")
            if cov.get("missing_support_reasons"):
                print(f"    missing_support_reasons: {cov.get('missing_support_reasons')[:6]}")
            for h in (r.get("hypothesis_summary") or {}).get("top_hypotheses", [])[:3]:
                print(f"    hyp {h['hypothesis_id']}: {h['slot_assignments']} "
                      f"support={h['support_score']} conf={h['confidence_score']} active={h['active']}")
            for h in (r.get("hypothesis_summary") or {}).get("rejected_hypotheses", []):
                print(f"    ✗ hyp {h['hypothesis_id']}: {h['rejection_reason']}")
        # Level 5 candidate slates + frontier.
        cf = r.get("candidate_frontier") or {}
        if cf.get("skipped"):
            print(f"  CANDIDATE SLATES: skipped ({cf.get('skipped_candidate_slate_reason')})")
        elif cf:
            print(f"  CANDIDATE SLATES ({cf.get('n_slates')} slots):")
            for sl in cf.get("slates", []):
                tops = ", ".join(f"{c['candidate_text_preview']}[{c['status']}"
                                 + (f":{c['status_reason']}" if c.get('status_reason') else "") + "]"
                                 for c in sl.get("top_candidates", [])[:4]) or "(empty)"
                print(f"    slate {sl.get('slot_role')}/{sl.get('slot_descriptor')!r}: {tops}")
            for h in cf.get("top_hypotheses", [])[:3]:
                print(f"    hyp* {h['hypothesis_id']}: {h.get('slot_candidate_assignments')} "
                      f"support={h.get('support_score')} conf={h.get('confidence_score')} "
                      f"active={h.get('active')}")
            sel = cf.get("selected_frontier_action") or {}
            if sel:
                print(f"    frontier -> {sel.get('action_type')} (eig={sel.get('expected_information_gain')}, "
                      f"{sel.get('selected_reason')})")
            for a in cf.get("frontier_actions", [])[:5]:
                tag = "SEL" if a.get("selected") else "   "
                print(f"      [{tag}] {a.get('action_type')} slot={a.get('target_slot_id')} "
                      f"eig={a.get('expected_information_gain')} "
                      + (f"rej={a.get('rejected_reason')}" if a.get('rejected_reason') else ""))
        # Stage chain (iterative clue resolution): query -> results -> typed candidates.
        for c in r.get("calls", []):
            cands = c.get("candidate_entities", [])
            print(f"  Stage {c.get('stage', 1)} [{c.get('tool')}/{c.get('query_arm')}]: "
                  f"{c.get('query_preview', '')!r} -> {c.get('n_results', 0)} results"
                  + (f", contaminated={c.get('contaminated_results')}"
                     if c.get('contaminated_results') else "")
                  + (f", q={c.get('query_quality')}" if c.get('query_quality') else "")
                  + (f", no_progress" if c.get('no_progress') else "")
                  + (f", sticky={c.get('sticky_penalty')}" if c.get('sticky_penalty') else ""))
            ta = c.get("task_action") or {}
            if ta.get("kind"):
                print(f"        action[{ta['kind']}] slot={ta.get('target_slot_id')} "
                      f"constraints={ta.get('tested_constraint_ids')} "
                      f"eig={ta.get('expected_information_gain')}")
                er = c.get("evidence_record") or {}
                if er:
                    print(f"        evidence: progress={er.get('evidence_progress_score')} "
                          f"new_candidates={er.get('newly_introduced_candidates')} "
                          f"supports_constraints={er.get('supports_constraint_ids')}")
            sc = c.get("scrape") or {}
            if sc.get("read_tool"):
                print(f"        read [{sc.get('read_tool')}]: {sc.get('scrape_url','')}  "
                      f"reason={sc.get('scrape_selected_reason')} "
                      f"success={sc.get('scrape_success')} "
                      f"chars={sc.get('scrape_chars')} "
                      f"evidence_added={sc.get('evidence_added_by_scrape')}"
                      + (f" FAILURE={sc.get('scrape_failure_type')}"
                         if sc.get('scrape_failure_type') else "")
                      + (" (fellback to page_fetch)" if sc.get('fallback_to_page_fetch') else ""))
            if c.get("selected_candidate"):
                print(f"        → selected: {c['selected_candidate']} "
                      f"[{c.get('selected_role')}]  {c.get('selection_reason', '')}")
            if cands:
                print("        candidate hypotheses:")
                for e in cands[:5]:
                    print(f"          {e.get('candidate_text')!r:30} role={e.get('role'):22} "
                          f"raw={e.get('raw_score')} adj={e.get('adjusted_score')}")
            for e in c.get("rejected_candidates", [])[:4]:
                print(f"          ✗ {e.get('candidate_text')!r:28} role={e.get('role'):22} "
                      f"reason={e.get('rejection_reason')}")
            qcands = c.get("query_candidates", [])
            if qcands:
                print(f"        query candidates ({c.get('n_query_candidates', len(qcands))}, "
                      f"{c.get('n_query_candidates_dropped', 0)} dropped): "
                      f"{[(q.get('arm'), q.get('expected_search_quality')) for q in qcands]}")
        for e in r.get("failed_tool_errors", []):
            print(f"  ⚠ tool error [{e['tool']}]: {e.get('error_type')} "
                  f"{e.get('status_code')} — {e.get('message_preview')}")
        for ev in r.get("evidence", [])[:3]:
            mark = "✓" if ev.get("supports") else " "
            tags = []
            if ev.get("contains_gold"):
                tags.append("HAS-GOLD")
            if ev.get("benchmark_contaminated"):
                tags.append(f"CONTAMINATED({ev.get('contamination_reason')})")
            tagstr = ("  [" + ", ".join(tags) + "]") if tags else ""
            print(f"  {mark} {ev.get('url','')}{tagstr}")
            if ev.get("title_preview"):
                print(f"      {ev['title_preview']}")
            if ev.get("snippet_preview"):
                print(f"      {ev['snippet_preview']}")
        print()

    if not rows:
        print("(no matching rows — try --all or different filters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
