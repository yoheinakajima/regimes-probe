"""Operational affordances + open-world semantic facets for task-frame constraints.

The earlier design rejected any constraint whose ``constraint_type`` was not in a
narrow enum — which made the LLM parser brittle and overfit to BrowseComp (rich,
defensible labels like ``employment_relation`` or ``distance_and_temporal_attribute``
forced a fallback). The fix is to make constraint *semantics* open-world while
keeping a small, closed set of **operational affordances** that the planner branches
on. Labels and facets are free-form and preserved verbatim; affordances are the
verbs the planner understands.

This module is dependency-light (operates on primitives / duck-typed constraints) so
both the deterministic parser and the LLM parser can enrich constraints uniformly.
"""

from __future__ import annotations

import re
from typing import Any

#: Lightly-standardized common facets. Open-ended: a parser MAY emit a new facet and
#: it is preserved, never rejected. These are only the well-known ones we map cleanly.
COMMON_FACETS: tuple[str, ...] = (
    "identity", "attribute", "relation", "temporal", "spatial", "quantitative",
    "authorship", "source", "membership", "biographical", "title_or_work",
    "comparison", "distance", "answer_shape",
)

#: Operational affordances — the CLOSED vocabulary the planner consumes. Unlike
#: facets/labels (open-world), these are the only verbs the planner branches on.
AFFORDANCES: tuple[str, ...] = (
    "can_search", "can_verify", "can_read", "can_compare",
    "can_bind_slot", "can_support_answer", "can_block_answer",
)

#: facets (or label substrings) that imply a document/source must be READ.
_READ_FACETS = {"source", "authorship", "title_or_work", "publication_or_source",
                "publication", "document", "record"}
_READ_CUES = ("read", "document", "page", "pdf", "article", "report", "record",
              "archive", "filing", "database", "wikipedia", "profile", "obituary",
              "citation", "bibliograph", "registry", "census", "transcript")
#: facets implying values must be COMPARED (numeric / spatial / temporal ordering).
_COMPARE_FACETS = {"comparison", "distance", "quantitative", "temporal", "spatial",
                   "numeric", "geographic", "chronolog"}

_TOKEN = re.compile(r"[a-z0-9_]+")


def normalize_facets(semantic_label: str, facets: Any) -> list[str]:
    """Open-world facet normalization: keep what the parser said, lightly tidy it.

    Splits a free-form label into facet-like tokens, unions with any explicit
    ``facets`` list, lowercases, de-dupes, and PRESERVES unfamiliar facets. Never
    rejects a facet for being new.
    """
    out: list[str] = []
    seen: set[str] = set()

    def _add(tok: str) -> None:
        t = tok.strip().lower()
        if t and t not in seen and not t.isdigit() and len(t) > 1:
            seen.add(t)
            out.append(t)

    if isinstance(facets, str):
        facets = [facets]
    for f in (facets or []):
        _add(str(f))
    # derive facet tokens from the label too (e.g. "employment_relation" -> employment, relation)
    for tok in _TOKEN.findall((semantic_label or "").lower()):
        if tok not in ("and", "the", "of", "a", "an", "or", "with", "to"):
            _add(tok)
    return out


def novel_facets(facets: list[str]) -> list[str]:
    """Facets the parser introduced that are not in the lightly-standardized set."""
    return [f for f in facets if f not in COMMON_FACETS]


def _has(facets: set[str], cues: set[str]) -> bool:
    return any(any(c in f for c in cues) for f in facets)


def derive_affordances(
    *,
    facets: list[str],
    semantic_label: str = "",
    applies_to: list[str],
    target_slot_ids: set[str],
    intermediate_slot_ids: set[str],
    supports_answer_slot_ids: list[str] | None = None,
    required: bool = False,
    priority: str = "medium",
    blocks_answer_if_unresolved: bool = False,
    testable_claim: str = "",
    evidence_needed: str = "",
    how_to_test: str = "",
    has_terms: bool = True,
    emitted: Any = None,
) -> list[str]:
    """Generic heuristic mapping of (open-world) semantics → planner affordances.

    If the parser emits affordances directly (``emitted``), the KNOWN ones are
    unioned in; unknown affordance strings are ignored (the planner can only act on
    the closed set). The result is returned in stable :data:`AFFORDANCES` order.
    """
    fl = {f.lower() for f in (facets or [])}
    text = " ".join([semantic_label or "", how_to_test or "", evidence_needed or ""]).lower()
    aff: set[str] = set()

    # Almost any constraint with lexical content can seed a search and be verified.
    if has_terms or semantic_label or testable_claim:
        aff.add("can_search")
        aff.add("can_verify")
    # A document/source must be read.
    if (fl & _READ_FACETS) or _has(fl, set(_READ_CUES)) or any(c in text for c in _READ_CUES):
        aff.add("can_read")
    # Values that must be compared (distance/temporal/quantitative/spatial).
    if (fl & _COMPARE_FACETS) or _has(fl, _COMPARE_FACETS):
        aff.add("can_compare")
    # A constraint that names/identifies a slot value can bind that slot.
    if applies_to:
        aff.add("can_bind_slot")
    # Supports the answer when it touches a target slot (directly or via supports list).
    sup = set(supports_answer_slot_ids or [])
    if sup or (set(applies_to) & target_slot_ids):
        aff.add("can_support_answer")
    # Blocks the answer when it is required/high-priority on a target/intermediate slot.
    blocking = required or priority == "high" or blocks_answer_if_unresolved
    if blocking and (set(applies_to) & (target_slot_ids | intermediate_slot_ids)):
        aff.add("can_block_answer")

    if emitted:
        if isinstance(emitted, str):
            emitted = [emitted]
        for a in emitted:
            if a in AFFORDANCES:
                aff.add(a)

    return [a for a in AFFORDANCES if a in aff]
