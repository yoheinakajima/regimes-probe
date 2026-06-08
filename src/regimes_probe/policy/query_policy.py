"""Level 2: query formulation policy.

Turns a question into a *query string* (plus tool options) by selecting a query
template arm. Three modes the benchmark distinguishes:

  * ``fixed``   — always ``direct_question`` (the generic baseline).
  * ``llm``     — a single LLM-generated query (v0: a deterministic stub that is
                  logged/cached and replaceable by templates later).
  * ``learned`` — a contextual bandit selects among template arms.

Arms are defined in ``docs/QUERY_POLICY.md`` and contain no topical knowledge —
each is a generic transformation of the question text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from regimes_probe.policy.contextual_bandit import ContextualBandit
from regimes_probe.policy.signatures import QuerySignature

#: Canonical query-policy arms.
QUERY_ARMS: tuple[str, ...] = (
    "direct_question",
    "keyword_compressed",
    "quoted_entities",
    "source_constrained",
    "freshness_terms",
    "exact_answer_shape",
    "site_or_domain_constrained",
)

_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "what", "which", "who", "whom", "whose", "how", "when",
    "where", "did", "do", "does", "with", "by", "as", "at", "that", "this",
}
_WORD = re.compile(r"[A-Za-z0-9]+")
_PROPER = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")


@dataclass
class QueryPlan:
    arm: str
    query: str
    opts: dict[str, Any] = field(default_factory=dict)
    explanation: dict[str, Any] = field(default_factory=dict)
    clue_ids: list[str] = field(default_factory=list)
    query_text_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"arm": self.arm, "query": self.query, "opts": self.opts,
                "clue_ids": self.clue_ids, "query_text_hash": self.query_text_hash,
                "explanation": self.explanation}


def _keyword_compress(question: str) -> str:
    words = [w for w in _WORD.findall(question) if w.lower() not in _STOPWORDS]
    return " ".join(words)


def _quoted_entities(question: str) -> str:
    propers = _PROPER.findall(question)
    quoted = " ".join(f'"{p}"' for p in propers)
    return (quoted + " " + _keyword_compress(question)).strip()


def _answer_shape_hint(sig: QuerySignature) -> str:
    ql = sig.question.lower()
    if "how many" in ql or "how much" in ql or "percentage" in ql:
        return "number value"
    if "what year" in ql or "what date" in ql or "when" in ql:
        return "date year"
    if "who" in ql or "name" in ql:
        return "name"
    return "exact answer"


def apply_query_arm(
    arm: str,
    sig: QuerySignature,
    *,
    known_domain: Optional[str] = None,
) -> QueryPlan:
    """Apply one query arm. Pure function of the signature (replayable)."""
    q = sig.question
    opts: dict[str, Any] = {}
    if arm == "direct_question":
        query = q
    elif arm == "keyword_compressed":
        query = _keyword_compress(q)
    elif arm == "quoted_entities":
        query = _quoted_entities(q)
    elif arm == "source_constrained":
        query = _keyword_compress(q) + " official"
    elif arm == "freshness_terms":
        query = _keyword_compress(q) + " 2026 latest"
    elif arm == "exact_answer_shape":
        query = _keyword_compress(q) + " " + _answer_shape_hint(sig)
    elif arm == "site_or_domain_constrained":
        query = _keyword_compress(q)
        if known_domain:
            query += f" site:{known_domain}"
            opts["allowed_domains"] = [known_domain]
    else:
        query = q
    return QueryPlan(arm=arm, query=query, opts=opts,
                     explanation={"applied_arm": arm, "known_domain": known_domain})


class QueryPolicy:
    """Select and apply a query arm under one of the three modes."""

    def __init__(self, mode: str = "learned") -> None:
        assert mode in ("fixed", "llm", "learned")
        self.mode = mode

    def formulate(
        self,
        sig: QuerySignature,
        *,
        bandit: Optional[ContextualBandit] = None,
        neighbor: Optional[dict[str, tuple[float, float]]] = None,
        explore: bool = False,
        known_domain: Optional[str] = None,
        arms: Optional[list[str]] = None,
        salt: str = "",
        decompose: bool = False,
        tool: Optional[str] = None,
    ) -> QueryPlan:
        # Query decomposition (Level 2): emit several targeted candidate queries
        # and let the bandit choose the query FORM, conditioned on the tool
        # (tool x query_arm -> reward). Active only in the learned mode so the
        # fixed/llm baselines are preserved. See docs/QUERY_POLICY.md.
        if decompose and self.mode == "learned":
            return self._formulate_decomposed(
                sig, bandit=bandit, neighbor=neighbor, explore=explore,
                tool=tool, salt=salt)

        arms = arms or list(QUERY_ARMS)
        if self.mode == "fixed":
            return apply_query_arm("direct_question", sig, known_domain=known_domain)
        if self.mode == "llm":
            # v0 deterministic stub for the "generic LLM query" condition: a
            # keyword+shape compression. Logged and cached upstream; replaceable.
            plan = apply_query_arm("exact_answer_shape", sig, known_domain=known_domain)
            plan.explanation["llm_stub"] = True
            return plan
        # learned
        if bandit is None:
            return apply_query_arm("direct_question", sig, known_domain=known_domain)
        ranked = bandit.choose_tool_plan(
            sig.cluster_key, arms, k=1, explore=explore, neighbor=neighbor, salt=salt
        )
        chosen = ranked[0].arm if ranked else "direct_question"
        plan = apply_query_arm(chosen, sig, known_domain=known_domain)
        plan.explanation["ranked"] = [s.to_dict() for s in ranked]
        return plan

    def _formulate_decomposed(self, sig, *, bandit, neighbor, explore, tool, salt) -> QueryPlan:
        from regimes_probe.policy.query_decomposition import decompose
        result = decompose(sig.question)
        candidates = result.candidates
        if not candidates:                       # nothing to decompose -> compressed
            plan = apply_query_arm("keyword_compressed", sig)
            plan.explanation["decompose"] = "no_candidates"
            return plan
        by_arm = {c.arm: c for c in candidates}
        arm_list = [c.arm for c in candidates]
        # Condition the query bandit on the tool so it learns tool x query_arm.
        ctx = sig.cluster_key if tool is None else f"{sig.cluster_key}|{tool}"
        if bandit is None:
            chosen = arm_list[0]                  # targeted arm first (never full-question)
            ranked_dump = []
        else:
            ranked = bandit.choose_tool_plan(ctx, arm_list, k=len(arm_list),
                                             explore=explore, neighbor=neighbor, salt=salt)
            chosen = ranked[0].arm if ranked else arm_list[0]
            ranked_dump = [s.to_dict() for s in ranked]
        cand = by_arm[chosen]
        dbg = result.to_debug()
        return QueryPlan(
            arm=cand.arm, query=cand.query, opts={},
            clue_ids=list(cand.clue_ids), query_text_hash=cand.query_text_hash,
            explanation={"decompose": True, "context": ctx, "tool": tool,
                         "n_candidates": len(candidates),
                         "candidate_arms": arm_list, "ranked": ranked_dump,
                         "selected_arm": cand.arm,
                         "selected_query_quality": round(cand.quality, 3),
                         "dropped_count": dbg["dropped_count"],
                         "dropped": dbg["dropped"],
                         "candidate_queries": dbg["candidate_queries"]})
