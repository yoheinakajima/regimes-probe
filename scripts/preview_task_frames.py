#!/usr/bin/env python
"""Preview the epistemic-mode decision + (open-world) task frame for questions.

    # dry-run (NO model calls): deterministic parser, or replay from a parser cache
    python scripts/preview_task_frames.py --questions-file qs.txt
    python scripts/preview_task_frames.py "Which TV series featured an actor who..."
    python scripts/preview_task_frames.py --use-llm --parser-cache run/task_frame_parser_cache.json "..."

    # parser-only LLM calls (requires OPENAI_API_KEY); recorded + cached
    python scripts/preview_task_frames.py --execute --task-frame-parser-model gpt-5.4-mini "..."
    python scripts/preview_task_frames.py --execute --fresh-parser --show-raw "..."   # ignore stale cache

Prints per question: epistemic-mode decision; PARSER DIAGNOSTICS (parser_used,
fallback_reason + classification, validation_errors/warnings, cache path/key/hit,
model_called); the raw semantic labels/facets/affordances the LLM emitted (even on
fallback); the resulting frame; and the suggested first action. Never answers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regimes_probe.agent.action_planner import ActionPlanner  # noqa: E402
from regimes_probe.agent.epistemic_mode import decide_epistemic_mode  # noqa: E402
from regimes_probe.agent.hypothesis_table import HypothesisTable  # noqa: E402
from regimes_probe.agent.llm_task_frame import (  # noqa: E402
    _parse_edge, _raw_slot_ids, _derive_raw_edges)
from regimes_probe.agent.llm_task_frame import (  # noqa: E402
    LLMTaskFrameParser, ParserCache, build_task_frame)

_SAMPLE = [
    "What is the capital of France?",
    "Who won the 2026 Eurovision Song Contest?",
    "Which TV series featured an actor who immigrated from the Caribbean and won an award in 2019?",
    "There is a Mexican restaurant in New Mexico, near a museum and a hotel, founded by a chef born in which year?",
]

_MAX_RAW = 4000
_MAX_FIELD = 300


def _build_parser(args):
    """Build a parser. dry-run => replay-only (no model); --execute => model_fn.

    ``--fresh-parser`` ignores the existing parser cache file (fresh in-memory
    ParserCache) and, in --execute mode, forces a fresh model call (record mode) so
    a stale FAILED parse is not replayed.
    """
    cache_path = None if args.fresh_parser else args.parser_cache
    cache = ParserCache(cache_path)
    if not args.execute:
        return LLMTaskFrameParser(model_fn=None, cache=cache,
                                  model=args.task_frame_parser_model, replay_only=True)
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        print("--execute requires OPENAI_API_KEY (parser-only model calls). Aborting.")
        raise SystemExit(2)
    from regimes_probe.live.cache import RecordingCache
    from regimes_probe.live.providers import build_task_frame_model_fn
    rec_mode = "record" if args.fresh_parser else "auto"
    rec = RecordingCache(args.recording_cache, mode=rec_mode)
    fn = build_task_frame_model_fn(args.task_frame_parser_model, rec, armed=True)
    return LLMTaskFrameParser(model_fn=fn, cache=cache, model=args.task_frame_parser_model)


def _classify_fallback(meta) -> list[str]:
    """Human categories for WHY the LLM frame was not accepted."""
    fr = meta.fallback_reason or ""
    if not fr:
        return []
    if fr == "invalid_json":
        return ["json_parse"]
    if fr == "low_quality":
        return ["parse_quality_below_floor"]
    if fr in ("no_model_available", "cache_miss_in_replay"):
        return [fr]
    if fr.startswith("model_error"):
        return ["model_error"]
    if fr != "validation_failed":
        return [fr]
    cats: list[str] = []
    errs = meta.validation_errors or []

    def _any(*subs):
        return any(any(s in e for s in subs) for e in errs)
    if _any("no_target_slot"):
        cats.append("missing_target_slot")
    if _any("constraint_refs_unknown_slot", "dependency_edge_refs_unknown_slot"):
        cats.append("bad_slot_references")
    if _any("forbidden_answer_key", "possible_answer_text_in_target_slot"):
        cats.append("gold_or_answer_text_guard")
    if _any("known_context_promoted_to_target"):
        cats.append("known_context_as_target")
    if _any("required_constraint_"):
        cats.append("required_constraint_not_operational")
    if _any("slot_missing_field", "constraint_missing_id", "constraint_missing_text",
            "missing_slot_lists", "slot_empty", "duplicate_slot_id",
            "missing_constraints_list", "payload_not_object", "slot_not_object",
            "constraint_not_object"):
        cats.append("schema_shape")
    return cats or ["validation_failed_uncategorized"]


def _raw_payload(parser, meta):
    """The raw JSON object the LLM emitted (from the parser cache), if available."""
    if parser is None or not meta.input_hash:
        return None
    raw = parser.cache.get(meta.input_hash)
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _truncate(v):
    if isinstance(v, str) and len(v) > _MAX_FIELD:
        return v[:_MAX_FIELD - 1] + "…"
    if isinstance(v, list):
        return [_truncate(x) for x in v[:12]]
    if isinstance(v, dict):
        return {k: _truncate(x) for k, x in v.items()}
    return v


def _bounded_raw(parser, meta) -> str:
    """Bounded raw model output for --show-raw (truncate long fields; cap total)."""
    if parser is None or not meta.input_hash:
        return "(no raw output: parser not used)"
    raw = parser.cache.get(meta.input_hash)
    if not raw:
        return "(no raw output cached for this question)"
    try:
        pretty = json.dumps(_truncate(json.loads(raw)), indent=2, ensure_ascii=False)
    except Exception:
        pretty = raw
    if len(pretty) > _MAX_RAW:
        pretty = pretty[:_MAX_RAW - 1] + "\n…(truncated)"
    return pretty


def _preview(question: str, *, use_llm: bool, parser, budget: int, show_raw: bool) -> None:
    print("=" * 78)
    print(f"Q: {question}")
    dec = decide_epistemic_mode(question, budget=budget)
    print(f"  epistemic_mode: {dec.selected_epistemic_mode}  ({dec.escalation_reason})")
    if dec.skipped_heavy_parser_reason:
        print(f"    skipped_heavy_parser: {dec.skipped_heavy_parser_reason}")
    print(f"    signals: {dec.signals}")

    model_before = parser.model_calls if parser is not None else 0
    frame, meta = build_task_frame("preview", question, use_llm=use_llm, parser=parser)
    model_called = bool(parser is not None and parser.model_calls > model_before)

    # ---- PARSER DIAGNOSTICS ----
    print("  PARSER DIAGNOSTICS:")
    print(f"    parser_used: {meta.parser_used}   parse_quality: {meta.parse_quality}"
          f"   model: {meta.model or '(n/a)'}")
    if use_llm:
        cache_path = parser.cache.path if parser is not None else None
        print(f"    parser_cache_path: {cache_path or '(in-memory)'}")
        print(f"    parser_cache_key: {meta.input_hash or '(none)'}   "
              f"parser_cache_hit: {str(meta.cache_hit).lower()}   "
              f"model_called: {str(model_called).lower()}")
        ow_pass = (meta.parser_used == "llm")
        print(f"    open_world_validation: {'passed' if ow_pass else 'FAILED'}")
        if meta.fallback_reason:
            print(f"    fallback_reason: {meta.fallback_reason}   "
                  f"classification: {_classify_fallback(meta)}")
        if meta.validation_errors:
            print(f"    validation_errors: {list(meta.validation_errors)[:12]}")
        if meta.validation_warnings:
            print(f"    validation_warnings: {list(meta.validation_warnings)[:12]}")
        # Raw semantics the LLM emitted — shown EVEN ON FALLBACK so you can see why.
        raw = _raw_payload(parser, meta)
        if raw is not None:
            cons = raw.get("constraints") or []
            print(f"    LLM emitted: {len(raw.get('target_answer_slots', []))} target / "
                  f"{len(raw.get('latent_slots', []))} latent slots, {len(cons)} constraints")
            for c in cons[:12]:
                print(f"      - {c.get('constraint_id')}: "
                      f"label={c.get('semantic_label')!r} "
                      f"facets={(c.get('semantic_facets') or [])[:5]} "
                      f"affordances_emitted={(c.get('affordances') or [])} "
                      f"applies_to={c.get('applies_to')} required={c.get('required')}")
        else:
            print("    LLM emitted: (no raw output available — replay miss / no model call)")
        # ---- ID / DEPENDENCY-EDGE DIAGNOSTICS ----
        if raw is not None:
            ordered, raw_ids = _raw_slot_ids(raw)
            # mapping actually applied (on accept) else the mapping that WOULD apply.
            mapping = (meta.id_mapping if meta.id_mapping
                       else {rid: f"s{i}" for i, rid in enumerate(ordered)})
            raw_edges = [pe for pe in (_parse_edge(e) for e in raw.get("dependency_edges") or [])
                         if pe is not None]
            derived = _derive_raw_edges(raw)
            print(f"    raw_slot_ids: {ordered}")
            print(f"    id_mapping (raw->internal): {mapping}"
                  + ("" if meta.id_mapping else "  [would-apply; frame fell back]"))
            print(f"    dependency_edges_raw (explicit): {raw_edges or '(none)'}")
            print(f"    dependency_edges_from_depends_on: {derived or '(none)'}")
            print(f"    dependency_edges_remapped (frame): {frame.dependency_edges or '(none)'}")
            ref_err = any(("constraint_refs_unknown_slot" in e
                           or "dependency_edge_refs_unknown_slot" in e)
                          for e in (meta.validation_errors or []))
            if meta.parser_used == "llm":
                stage = "none (accepted; remap is total so post-remap never fails)"
            elif meta.fallback_reason == "validation_failed":
                stage = ("raw_id_reference_validation" if ref_err
                         else "open_world_usability_validation (pre-remap)")
            else:
                stage = f"{meta.fallback_reason} (not an id-validation fallback)"
            print(f"    fallback_stage: {stage}")
    if show_raw:
        print("  RAW PARSER OUTPUT (bounded):")
        for line in _bounded_raw(parser, meta).splitlines():
            print("    " + line)

    # ---- RESULTING FRAME (deterministic or accepted-LLM) ----
    print("  target slots:      " + ", ".join(f"{s.slot_name}({s.slot_role})"
                                               for s in frame.target_answer_slots))
    print("  intermediate slots:" + ", ".join(f"{s.slot_name}({s.slot_role})"
                                               for s in frame.latent_slots))
    blocking = []
    for c in frame.constraints:
        blk = c.blocks_answer_if_unresolved or "can_block_answer" in c.affordances
        if blk:
            blocking.append(c.constraint_id)
        print(f"  constraint {c.constraint_id} [{c.status}{' BLOCKING' if blk else ''}] "
              f"label={c.semantic_label!r} facets={c.semantic_facets[:4]} "
              f"aff_derived={c.affordances} -> {c.applies_to}")
    print(f"  blocking constraints: {blocking or '(none)'}")
    table = HypothesisTable(frame)
    action = ActionPlanner(frame, table).plan(budget_remaining=budget, reading_available=True)
    print(f"  suggested first action: {action.kind}"
          + (f"  query={action.query!r}" if action.query else "")
          + (f"  tests={action.tested_constraint_ids}" if action.tested_constraint_ids else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("questions", nargs="*", help="questions (default: built-in sample)")
    ap.add_argument("--questions-file", default=None, help="one question per line")
    ap.add_argument("--use-llm", "--enable-llm-task-frame-parser", dest="use_llm",
                    action="store_true",
                    help="use the LLM parser (dry-run = replay-only from --parser-cache)")
    ap.add_argument("--execute", action="store_true",
                    help="allow parser-only model calls (requires OPENAI_API_KEY)")
    ap.add_argument("--parser-cache", default=None, help="parser cache JSON (replay)")
    ap.add_argument("--no-cache", "--fresh-parser", dest="fresh_parser", action="store_true",
                    help="ignore the existing parser cache (avoid stale FAILED parses)")
    ap.add_argument("--show-raw", action="store_true",
                    help=f"print bounded raw parser JSON (max ~{_MAX_RAW} chars)")
    ap.add_argument("--recording-cache", default=None)
    ap.add_argument("--task-frame-parser-model", "--answer-model",
                    dest="task_frame_parser_model", default="gpt-5.4-mini",
                    help="parser model (alias: --answer-model)")
    ap.add_argument("--limit", type=int, default=None, help="cap number of questions")
    ap.add_argument("--budget", type=int, default=3)
    args = ap.parse_args()

    questions = list(args.questions)
    if args.questions_file:
        questions += [ln.strip() for ln in Path(args.questions_file).read_text().splitlines()
                      if ln.strip()]
    if not questions:
        questions = list(_SAMPLE)
    if args.limit is not None:
        questions = questions[:args.limit]

    use_llm = args.use_llm or args.execute
    parser = _build_parser(args) if use_llm else None
    if not args.execute:
        print("DRY-RUN: no model is called "
              + ("(LLM parser replays from cache; falls back to deterministic on miss)"
                 if use_llm else "(deterministic parser)")
              + (" [--fresh-parser: ignoring cache file]" if args.fresh_parser else "") + ".\n")
    for q in questions:
        _preview(q, use_llm=use_llm, parser=parser, budget=args.budget, show_raw=args.show_raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
