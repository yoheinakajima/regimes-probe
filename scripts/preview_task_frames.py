#!/usr/bin/env python
"""Preview the epistemic-mode decision + (open-world) task frame for questions.

    # dry-run (NO model calls): deterministic parser, or replay from a parser cache
    python scripts/preview_task_frames.py --questions-file qs.txt
    python scripts/preview_task_frames.py "Which TV series featured an actor who..."
    python scripts/preview_task_frames.py --parser-cache results/live/run/task_frame_parser_cache.json

    # parser-only LLM calls (requires OPENAI_API_KEY); recorded + cached
    python scripts/preview_task_frames.py --execute --task-frame-parser-model gpt-5.4-mini "..."

Prints, per question: epistemic-mode decision; parser used; target + intermediate
slots; constraints with semantic labels / facets / affordances; blocking
constraints; suggested actions; validation warnings; parse quality. Never answers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regimes_probe.agent.action_planner import ActionPlanner  # noqa: E402
from regimes_probe.agent.epistemic_mode import decide_epistemic_mode  # noqa: E402
from regimes_probe.agent.hypothesis_table import HypothesisTable  # noqa: E402
from regimes_probe.agent.llm_task_frame import (  # noqa: E402
    LLMTaskFrameParser, ParserCache, build_task_frame)

_SAMPLE = [
    "What is the capital of France?",
    "Who won the 2026 Eurovision Song Contest?",
    "Which TV series featured an actor who immigrated from the Caribbean and won an award in 2019?",
    "There is a Mexican restaurant in New Mexico, near a museum and a hotel, founded by a chef born in which year?",
]


def _build_parser(args):
    """Build a parser. dry-run => replay-only (no model); --execute => model_fn."""
    cache = ParserCache(args.parser_cache)
    if not args.execute:
        return LLMTaskFrameParser(model_fn=None, cache=cache,
                                  model=args.task_frame_parser_model, replay_only=True)
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        print("--execute requires OPENAI_API_KEY (parser-only model calls). Aborting.")
        raise SystemExit(2)
    from regimes_probe.live.cache import RecordingCache
    from regimes_probe.live.providers import build_task_frame_model_fn
    rec = RecordingCache(args.recording_cache, mode="auto")
    fn = build_task_frame_model_fn(args.task_frame_parser_model, rec, armed=True)
    return LLMTaskFrameParser(model_fn=fn, cache=cache, model=args.task_frame_parser_model)


def _preview(question: str, *, use_llm: bool, parser, budget: int) -> None:
    print("=" * 78)
    print(f"Q: {question}")
    dec = decide_epistemic_mode(question, budget=budget)
    print(f"  epistemic_mode: {dec.selected_epistemic_mode}  ({dec.escalation_reason})")
    if dec.skipped_heavy_parser_reason:
        print(f"    skipped_heavy_parser: {dec.skipped_heavy_parser_reason}")
    print(f"    signals: {dec.signals}")
    if not dec.use_task_frame and not use_llm:
        print("  (mode does not require a task frame; showing frame anyway for inspection)")
    frame, meta = build_task_frame("preview", question, use_llm=use_llm, parser=parser)
    print(f"  parser_used: {meta.parser_used}  parse_quality: {meta.parse_quality}"
          + (f"  fallback: {meta.fallback_reason}" if meta.fallback_reason else ""))
    if meta.validation_warnings:
        print(f"  validation_warnings: {meta.validation_warnings[:8]}")
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
              f"aff={c.affordances} -> {c.applies_to}")
    print(f"  blocking constraints: {blocking or '(none)'}")
    # Suggested next action from the planner (read-only; no providers).
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
    ap.add_argument("--use-llm", action="store_true",
                    help="use the LLM parser (dry-run = replay-only from --parser-cache)")
    ap.add_argument("--execute", action="store_true",
                    help="allow parser-only model calls (requires OPENAI_API_KEY)")
    ap.add_argument("--parser-cache", default=None, help="parser cache JSON (replay)")
    ap.add_argument("--recording-cache", default=None)
    ap.add_argument("--task-frame-parser-model", default="gpt-5.4-mini")
    ap.add_argument("--budget", type=int, default=3)
    args = ap.parse_args()

    questions = list(args.questions)
    if args.questions_file:
        questions += [ln.strip() for ln in Path(args.questions_file).read_text().splitlines()
                      if ln.strip()]
    if not questions:
        questions = list(_SAMPLE)

    use_llm = args.use_llm or args.execute
    parser = _build_parser(args) if use_llm else None
    if not args.execute:
        print("DRY-RUN: no model is called "
              + ("(LLM parser replays from cache; falls back to deterministic on miss)"
                 if use_llm else "(deterministic parser)") + ".\n")
    for q in questions:
        _preview(q, use_llm=use_llm, parser=parser, budget=args.budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
