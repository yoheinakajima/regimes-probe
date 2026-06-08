"""Epistemic escalation controller — a lightweight front door before the heavy parser.

Invoking the task-frame / hypothesis / evidence-graph machinery for *every* question
is wasteful: an easy factual lookup does not need a constraint graph. This module
decides, from cheap question signals, how much epistemic machinery a question
warrants — from ``direct_answer_possible`` up to ``task_frame_required`` — so the
agent escalates effort only when the question is genuinely hard/multi-hop.

Deterministic and gold-free; it reads only the question text + budget. BrowseComp
items will *usually* route to ``task_frame_required``, but that emerges from the
signals (clue density, multi-hop chaining) rather than being hard-coded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from regimes_probe.policy.query_decomposition import extract_clues, split_clauses

#: The escalation ladder, cheapest → heaviest.
EPISTEMIC_MODES = ("direct_answer_possible", "simple_lookup", "decomposed_search",
                   "iterative_research", "task_frame_required")

#: connectives that signal a multi-hop chain (find X, then use X to find Y).
_MULTIHOP = ("who", "whom", "whose", "which", "that", "where", "after", "before",
             "later", "then", "subsequently", "founded by", "authored by",
             "written by", "starred", "featured", "employed by", "published by")
#: recency / currentness cues (favor a fresh lookup over heavy decomposition).
_RECENCY = ("latest", "current", "currently", "now", "today", "this year",
            "recent", "as of", "2024", "2025", "2026")


@dataclass
class EpistemicModeDecision:
    selected_epistemic_mode: str
    escalation_reason: str = ""
    skipped_heavy_parser_reason: str = ""
    estimated_effort: dict[str, Any] = field(default_factory=dict)
    signals: dict[str, Any] = field(default_factory=dict)
    auto: bool = False
    applied: bool = False

    @property
    def use_task_frame(self) -> bool:
        return self.selected_epistemic_mode == "task_frame_required"

    @property
    def use_iterative(self) -> bool:
        return self.selected_epistemic_mode == "iterative_research"

    @property
    def use_decomposition(self) -> bool:
        return self.selected_epistemic_mode in ("decomposed_search", "iterative_research")

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_epistemic_mode": self.selected_epistemic_mode,
            "escalation_reason": self.escalation_reason,
            "skipped_heavy_parser_reason": self.skipped_heavy_parser_reason,
            "estimated_effort": dict(self.estimated_effort),
            "signals": dict(self.signals),
            "auto": self.auto, "applied": self.applied,
        }


def _signals(question: str) -> dict[str, Any]:
    q = question or ""
    ql = q.lower()
    clues = extract_clues(q)
    clauses = [c for c in split_clauses(q) if c.strip()]
    entities = list(dict.fromkeys(clues.entities))
    n_words = len(re.findall(r"[A-Za-z0-9]+", q))
    multihop = sum(1 for k in _MULTIHOP if k in ql)
    recency = any(k in ql for k in _RECENCY)
    # a rough constraint count: clauses carrying a content predicate.
    n_constraints = len([c for c in clauses if len(re.findall(r"[A-Za-z0-9]+", c)) >= 3])
    return {
        "n_words": n_words,
        "n_clauses": len(clauses),
        "n_entities": len(entities),
        "n_constraints": n_constraints,
        "multihop_cues": multihop,
        "recency": recency,
        "answer_shape": list(clues.answer_shape) if hasattr(clues, "answer_shape") else [],
    }


def decide_epistemic_mode(question: str, *, budget: int = 3,
                          force_task_frame: bool = False,
                          disable_direct_answer: bool = False,
                          internal_confidence: Optional[float] = None,
                          auto: bool = True) -> EpistemicModeDecision:
    """Pick the cheapest epistemic mode that fits the question's difficulty signals."""
    s = _signals(question)
    eff = {"budget": budget, "n_clauses": s["n_clauses"], "n_entities": s["n_entities"],
           "multihop_cues": s["multihop_cues"]}

    if force_task_frame:
        return EpistemicModeDecision(
            "task_frame_required", escalation_reason="forced_by_flag",
            estimated_effort=eff | {"expected_searches": budget}, signals=s, auto=auto)

    # Clause count = number of hops; the primary difficulty signal. Entity density and
    # multi-hop connectives only escalate when there are MULTIPLE clauses (so a single
    # title with several proper nouns is not mistaken for a multi-hop chain).
    nc, ne, mh = s["n_clauses"], s["n_entities"], s["multihop_cues"]

    # clue-dense / multi-hop → full task frame.
    if nc >= 3 or (nc >= 2 and (mh >= 2 or ne >= 3)):
        return EpistemicModeDecision(
            "task_frame_required",
            escalation_reason=f"clue_dense(n_clauses={nc},n_entities={ne},multihop={mh})",
            estimated_effort=eff | {"expected_searches": budget}, signals=s, auto=auto)

    # moderate multi-hop → iterative research (resolve an intermediate, then chain).
    if nc >= 2 and mh >= 1:
        return EpistemicModeDecision(
            "iterative_research", escalation_reason="moderate_multihop",
            skipped_heavy_parser_reason="not_clue_dense_enough_for_task_frame",
            estimated_effort=eff | {"expected_searches": min(budget, 3)}, signals=s, auto=auto)

    # a couple of clauses but no chaining → decomposed search.
    if nc >= 2:
        return EpistemicModeDecision(
            "decomposed_search", escalation_reason="multi_clause_single_hop",
            skipped_heavy_parser_reason="no_multihop_chain",
            estimated_effort=eff | {"expected_searches": min(budget, 2)}, signals=s, auto=auto)

    # short, single-clue factual question.
    short = s["n_words"] <= 16 and s["n_constraints"] <= 1
    if short and not disable_direct_answer and not s["recency"]:
        return EpistemicModeDecision(
            "direct_answer_possible", escalation_reason="short_single_fact",
            skipped_heavy_parser_reason="question_is_simple_lookup",
            estimated_effort=eff | {"expected_searches": 0 if budget == 0 else 1},
            signals=s, auto=auto)
    return EpistemicModeDecision(
        "simple_lookup",
        escalation_reason=("recency_lookup" if s["recency"] else "single_clue_lookup"),
        skipped_heavy_parser_reason="question_is_simple_lookup",
        estimated_effort=eff | {"expected_searches": 1}, signals=s, auto=auto)


def explicit_mode_decision(*, task_frame: bool, iterative: bool, decomposition: bool
                           ) -> EpistemicModeDecision:
    """Record the mode implied by EXPLICIT flags (auto controller disabled)."""
    if task_frame:
        mode = "task_frame_required"
    elif iterative:
        mode = "iterative_research"
    elif decomposition:
        mode = "decomposed_search"
    else:
        mode = "simple_lookup"
    return EpistemicModeDecision(mode, escalation_reason="explicit_flags", auto=False,
                                 applied=False)
