"""Level 5f: a narrow, cached, replayable LLM **evidence judge**.

This is *not* a planner and *not* an answerer. It is an evidence-fit function over a single
``(candidate, slot, constraint, source-excerpt)`` triple:

    judge(candidate, slot, constraint, source, source_role, contaminated, aliases)
        -> EvidenceJudgment

It decides whether the excerpt ``full_support`` / ``partial_support`` / ``contradiction`` /
``irrelevant`` / ``requires_read`` for that triple, with a quote, rationale, role-fit, and
aliases. It never produces a final answer, never sees gold labels, never weakens the
answer-support gate, and never writes to policy memory.

Discipline (mirrors the other LLM hooks):
- Cached + replayable: keyed by ``prompt_fingerprint | model | triple_hash``; dry-run/replay
  make zero model calls and fall back to the deterministic recognizer.
- Hard rules are enforced *after* the model (so a misbehaving stub cannot break them): a
  contaminated or noise source can never produce full/partial support; full support requires
  a quote; title/chrome-only evidence is downgraded.
- The deterministic recognizer (`recognize_constraint_support`) is the canonical fallback,
  so behaviour with the flag OFF is unchanged.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from regimes_probe.agent.llm_task_frame import _extract_json, _sha

JUDGMENTS = ("full_support", "partial_support", "contradiction", "irrelevant", "requires_read")
_SUPPORT_JUDGMENTS = ("full_support", "partial_support")
#: source roles that can never carry task-specific support (kept local to avoid an import
#: cycle with the interpreter; same generic set).
_NOISE_SOURCE_ROLES = frozenset(
    {"generic_definition_page", "ui_or_navigation_noise", "benchmark_contaminated"})
_CORROBORATING_ROLES = frozenset({"professional_profile", "official_page", "scholarly_paper",
                                  "database_record", "primary_source", "directory_listing",
                                  "article"})


@dataclass
class EvidenceJudgment:
    judgment: str = "irrelevant"
    candidate_role_fit: str = "unclear"           # fits | does_not_fit | unclear
    source_role_fit: str = "weak"                 # acceptable | weak | unacceptable
    supported_facets: list[str] = field(default_factory=list)
    unsupported_facets: list[str] = field(default_factory=list)
    contradicted_facets: list[str] = field(default_factory=list)
    quote: str = ""
    rationale: str = ""
    confidence: float = 0.0
    requires_read_reason: Optional[str] = None
    candidate_aliases: list[str] = field(default_factory=list)
    safety_notes: list[str] = field(default_factory=list)
    # provenance (trace only)
    judgment_id: str = ""
    candidate_id: Optional[str] = None
    slot_id: str = ""
    constraint_id: str = ""
    source_id: str = ""
    mode: str = "deterministic"                   # deterministic | llm
    cache_hit: bool = False
    model_called: bool = False
    input_hash: str = ""
    prompt_hash: str = ""
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"judgment_id": self.judgment_id, "judgment": self.judgment,
                "candidate_id": self.candidate_id, "slot_id": self.slot_id,
                "constraint_id": self.constraint_id, "source_id": self.source_id,
                "candidate_role_fit": self.candidate_role_fit,
                "source_role_fit": self.source_role_fit,
                "supported_facets": list(self.supported_facets)[:8],
                "unsupported_facets": list(self.unsupported_facets)[:8],
                "contradicted_facets": list(self.contradicted_facets)[:8],
                "quote": (self.quote or "")[:200], "rationale": (self.rationale or "")[:200],
                "confidence": round(float(self.confidence or 0.0), 3),
                "requires_read_reason": self.requires_read_reason,
                "candidate_aliases": list(self.candidate_aliases)[:8],
                "safety_notes": list(self.safety_notes)[:6],
                "mode": self.mode, "cache_hit": self.cache_hit,
                "model_called": self.model_called, "input_hash": self.input_hash,
                "prompt_hash": self.prompt_hash, "model": self.model}


def _det_to_judgment(det_status: str, *, has_url: bool, source_role: str) -> str:
    """Map the deterministic recognizer status onto the richer judgment vocabulary."""
    if det_status == "supports":
        return "full_support"
    if det_status == "contradicts":
        return "contradiction"
    if det_status == "insufficient":
        # the snippet hints at fit but cannot establish it: read the body if we can.
        if has_url and source_role in _CORROBORATING_ROLES:
            return "requires_read"
        return "partial_support"
    return "irrelevant"


_GENERIC_DESCRIPTORS = frozenset({
    "kenyan", "east african", "african", "american", "british", "mexican", "christian",
    "buddhist", "religious", "author", "novelist", "founder", "artist", "designer",
    "graphic designer", "actor", "novel", "report", "publication", "restaurant", "hotel",
    "museum", "tv shows", "tv series", "series", "book publishing", "annual report"})
_IDENTITY_ROLES = frozenset({"person", "organization", "title_or_work", "publication_or_source"})


def _is_named_entity(text: str) -> bool:
    """A concrete named entity: a proper-cased token (or multiword phrase), not a lowercase
    descriptor/role label. Generic, no benchmark strings."""
    t = (text or "").strip()
    if not t:
        return False
    if t.lower() in _GENERIC_DESCRIPTORS:
        return False
    words = [w for w in re.findall(r"[A-Za-z0-9'&.-]+", t)]
    proper = [w for w in words if w[:1].isupper() or w[:1].isdigit()]
    return len(proper) >= 1 and t.lower() not in _GENERIC_DESCRIPTORS


def _quote_anchors(quote: str, candidate_text: str, aliases) -> bool:
    q = (quote or "").lower()
    names = [candidate_text] + list(aliases or [])
    return any(n and n.lower() in q for n in names if len(n) >= 3)


def enforce_hard_rules(j: EvidenceJudgment, *, contaminated: bool, source_role: str,
                       has_quote: bool, candidate_text: str = "", aliases=None,
                       slot_role: str = "", relational: bool = False,
                       object_anchored: bool = True, counters=None) -> EvidenceJudgment:
    """Hard, non-negotiable rules applied AFTER the model so a stub cannot break them.

    Beyond contamination + quote: full_support for an identity slot requires a concrete
    NAMED candidate whose name (or a registered alias) appears in the quote; a generic
    descriptor can never be full_support; a relational constraint needs both subject AND
    object anchored. Downgrades are recorded, and ``counters`` (a Counter) tallies them."""
    aliases = list(aliases or [])
    bump = (lambda k: counters.update([k])) if counters is not None else (lambda k: None)
    if contaminated or source_role in _NOISE_SOURCE_ROLES:
        if j.judgment in _SUPPORT_JUDGMENTS:
            j.judgment = "contradiction" if j.contradicted_facets else "irrelevant"
            j.safety_notes.append("contaminated_or_noise_source_cannot_support")
            j.source_role_fit = "unacceptable"
    # full support REQUIRES a quote tying candidate to the predicate.
    if j.judgment == "full_support" and not has_quote:
        j.judgment, _ = "partial_support", j.safety_notes.append("downgraded_full_no_quote")
    # generic descriptors can never be full support for an identity slot.
    if j.judgment in _SUPPORT_JUDGMENTS and candidate_text and \
            candidate_text.strip().lower() in _GENERIC_DESCRIPTORS:
        if j.judgment == "full_support":
            bump("full_support_from_generic_descriptor")
        else:
            bump("partial_support_from_generic_descriptor")
        j.judgment = "irrelevant" if j.judgment == "full_support" else j.judgment
        if j.judgment == "irrelevant":
            j.safety_notes.append("generic_descriptor_not_full_support")
    # full support for an identity slot requires a concrete NAMED candidate + quote anchor.
    if j.judgment == "full_support" and (slot_role in _IDENTITY_ROLES or slot_role == ""
                                         or slot_role not in _IDENTITY_ROLES):
        if candidate_text and not _is_named_entity(candidate_text):
            bump("full_support_without_named_candidate")
            j.judgment = "partial_support"
            j.safety_notes.append("downgraded_full_unnamed_candidate")
        elif has_quote and candidate_text and not _quote_anchors(j.quote, candidate_text, aliases):
            j.judgment = "partial_support"
            j.safety_notes.append("downgraded_full_quote_does_not_anchor_candidate")
    # relational constraints need BOTH subject and object anchored.
    if j.judgment == "full_support" and relational and not object_anchored:
        bump("relational_support_without_object_anchor")
        j.judgment = "partial_support"
        j.safety_notes.append("downgraded_full_relational_object_not_anchored")
    return j


class EvidenceJudge:
    """Deterministic-by-default evidence-fit judge with an optional cached/replayable LLM."""

    def __init__(self, model_fn: Optional[Callable[[str], str]] = None, *,
                 cache=None, model: str = "deterministic", replay_only: bool = False,
                 enabled: bool = False, prompt_name: str = "evidence_judge",
                 max_calls_per_candidate: int = 6) -> None:
        self.model_fn = model_fn
        self.cache = cache
        self.model = model
        self.replay_only = replay_only
        self.enabled = enabled
        self.max_calls_per_candidate = max_calls_per_candidate
        from regimes_probe.agent import prompts
        self._prompt = prompts.get(prompt_name) if prompt_name in prompts.registry_dict() else None
        self._jc = 0
        self.calls = 0
        self.cache_hits = 0
        self.replay_hits = 0
        self.judgment_counts: Counter = Counter()
        self.contaminated_support_blocked = 0
        #: post-model hard-rule downgrade tallies (Level 5f-G) + budget savings (5f-H).
        self.guard_counts: Counter = Counter()
        self.calls_saved_by_pregate = 0
        self.calls_saved_by_relevance_filter = 0
        self.calls_saved_by_contradiction_stop = 0
        self.contradiction_early_stop = 0
        self.max_calls_cap_hit = 0

    def stats(self) -> dict[str, Any]:
        return {
            "llm_evidence_judge_model": self.model,
            "llm_evidence_judge_calls": self.calls,
            "llm_evidence_judge_cache_hits": self.cache_hits,
            "llm_evidence_judge_replay_hits": self.replay_hits,
            "llm_full_support_count": self.judgment_counts.get("full_support", 0),
            "llm_partial_support_count": self.judgment_counts.get("partial_support", 0),
            "llm_contradiction_count": self.judgment_counts.get("contradiction", 0),
            "llm_requires_read_count": self.judgment_counts.get("requires_read", 0),
            "llm_irrelevant_count": self.judgment_counts.get("irrelevant", 0),
            "full_support_from_contaminated_source_count": 0,   # invariant: always 0
            "partial_support_from_contaminated_source_count": 0,
            "full_or_partial_support_from_contaminated_source_count": 0,
            "contaminated_support_blocked_count": self.contaminated_support_blocked,
            # post-model support-contract invariants (Level 5f-G) — must stay 0.
            "full_support_without_named_candidate_count":
                self.guard_counts.get("full_support_without_named_candidate", 0),
            "full_support_from_generic_descriptor_count":
                self.guard_counts.get("full_support_from_generic_descriptor", 0),
            "partial_support_from_generic_descriptor_count":
                self.guard_counts.get("partial_support_from_generic_descriptor", 0),
            "relational_support_without_object_anchor_count":
                self.guard_counts.get("relational_support_without_object_anchor", 0),
            # judge-budget savings (Level 5f-H).
            "judge_calls_saved_by_pregate": self.calls_saved_by_pregate,
            "judge_calls_saved_by_relevance_filter": self.calls_saved_by_relevance_filter,
            "judge_calls_saved_by_contradiction_stop": self.calls_saved_by_contradiction_stop,
            "contradiction_early_stop_count": self.contradiction_early_stop,
            "judge_max_calls_cap_hit_count": self.max_calls_cap_hit,
        }

    def _triple_hash(self, *, candidate_text, slot_id, slot_role, constraint, source_role,
                     contaminated, title, snippet, url) -> str:
        con_key = "|".join(str(x) for x in (
            getattr(constraint, "constraint_id", ""), getattr(constraint, "text_span", ""),
            getattr(constraint, "testable_claim", ""),
            ",".join(getattr(constraint, "normalized_terms", []) or [])))
        body = "\n".join(str(x) for x in (
            candidate_text, slot_id, slot_role, con_key, source_role, int(bool(contaminated)),
            title, snippet[:600], url))
        return _sha(body)

    def judge(self, *, candidate_text: str, candidate_id: Optional[str], aliases,
              slot_id: str, slot_role: str, slot_descriptor: str, constraint,
              source_id: str, source_title: str, source_url: str, source_domain: str,
              source_role: str, contaminated: bool, snippet: str,
              det_status: str, det_quote: str, relational: bool = False,
              object_anchored: bool = True) -> EvidenceJudgment:
        """Judge one (candidate, slot, constraint, source-excerpt) triple."""
        self._jc += 1
        has_url = bool((source_url or "").strip())
        ih = self._triple_hash(candidate_text=candidate_text, slot_id=slot_id,
                               slot_role=slot_role, constraint=constraint,
                               source_role=source_role, contaminated=contaminated,
                               title=source_title, snippet=snippet, url=source_url)
        j = self._judge_inner(
            ih=ih, candidate_text=candidate_text, aliases=list(aliases or []),
            slot_role=slot_role, slot_descriptor=slot_descriptor, constraint=constraint,
            source_title=source_title, source_url=source_url, source_role=source_role,
            contaminated=contaminated, snippet=snippet, det_status=det_status,
            det_quote=det_quote, has_url=has_url)
        j = enforce_hard_rules(j, contaminated=contaminated, source_role=source_role,
                               has_quote=bool((j.quote or "").strip()),
                               candidate_text=candidate_text, aliases=list(aliases or []),
                               slot_role=slot_role, relational=relational,
                               object_anchored=object_anchored, counters=self.guard_counts)
        if contaminated and j.judgment in _SUPPORT_JUDGMENTS:
            self.contaminated_support_blocked += 1     # belt-and-suspenders (should never hit)
            j.judgment = "irrelevant"
        j.judgment_id = f"ej{self._jc}"
        j.candidate_id, j.slot_id = candidate_id, slot_id
        j.constraint_id = getattr(constraint, "constraint_id", "")
        j.source_id, j.input_hash = source_id, ih
        j.model = self.model
        j.prompt_hash = (self._prompt.content_hash if self._prompt is not None else "")
        self.judgment_counts[j.judgment] += 1
        return j

    def _judge_inner(self, *, ih, candidate_text, aliases, slot_role, slot_descriptor,
                     constraint, source_title, source_url, source_role, contaminated,
                     snippet, det_status, det_quote, has_url) -> EvidenceJudgment:
        # deterministic baseline always computed (canonical fallback).
        det = EvidenceJudgment(
            judgment=_det_to_judgment(det_status, has_url=has_url, source_role=source_role),
            candidate_role_fit=("fits" if det_status in ("supports", "insufficient") else "unclear"),
            source_role_fit=("acceptable" if source_role in _CORROBORATING_ROLES
                             else ("unacceptable" if source_role in _NOISE_SOURCE_ROLES else "weak")),
            quote=det_quote, rationale=f"deterministic:{det_status}",
            confidence=0.6 if det_status in ("supports", "contradicts") else 0.3,
            requires_read_reason=("snippet_insufficient_body_may_support"
                                  if det_status == "insufficient" and has_url else None),
            mode="deterministic")
        if not self.enabled or (self.model_fn is None and self.cache is None):
            return det
        raw = self._cached_call(ih, candidate_text=candidate_text, aliases=aliases,
                                slot_role=slot_role, slot_descriptor=slot_descriptor,
                                constraint=constraint, source_title=source_title,
                                source_url=source_url, source_role=source_role,
                                contaminated=contaminated, snippet=snippet, det=det)
        if raw is None:
            return det                                # replay/dry-run miss -> deterministic
        parsed = self._parse(raw)
        if parsed is None:
            return det
        parsed.cache_hit, parsed.model_called = det.cache_hit, det.model_called
        return parsed

    def _cached_call(self, ih, **ctx) -> Optional[str]:
        fp = self._prompt.fingerprint() if self._prompt is not None else "evidence_judge"
        key = _sha(f"{fp}|{self.model}|{ih}")
        raw = self.cache.get(key) if self.cache is not None else None
        if raw is not None:
            self.cache_hits += 1
            if self.replay_only:
                self.replay_hits += 1
            return raw
        if self.replay_only or self.model_fn is None:
            return None                               # never spend on dry-run/replay miss
        con = ctx["constraint"]
        prompt = (
            "You are an EVIDENCE JUDGE. Decide ONLY whether the source excerpt supports a "
            "specific candidate/slot/constraint. Do NOT answer the user's question. Output "
            'JSON with keys: judgment (full_support|partial_support|contradiction|irrelevant|'
            'requires_read), candidate_role_fit, source_role_fit, supported_facets, '
            "unsupported_facets, contradicted_facets, quote, rationale, confidence, "
            "requires_read_reason, candidate_aliases, safety_notes.\n"
            f"SLOT: {ctx['slot_descriptor']} (role={ctx['slot_role']})\n"
            f"CANDIDATE: {ctx['candidate_text']} aliases={ctx['aliases']}\n"
            f"CONSTRAINT: {getattr(con, 'text_span', '')} | testable={getattr(con, 'testable_claim', '')}"
            f" | how_to_test={getattr(con, 'how_to_test', '')}\n"
            f"SOURCE role={ctx['source_role']} contaminated={ctx['contaminated']} "
            f"title={ctx['source_title'][:160]} url={ctx['source_url'][:160]}\n"
            f"EXCERPT: {ctx['snippet'][:600]}\n")
        try:
            self.calls += 1
            raw = self.model_fn(prompt)
        except Exception:
            return None
        if self.cache is not None:
            self.cache.put(key, raw or "")
        return raw

    @staticmethod
    def _parse(raw: str) -> Optional[EvidenceJudgment]:
        obj = _extract_json(raw)
        if not isinstance(obj, dict):
            try:
                obj = json.loads(raw)
            except Exception:
                return None
        if not isinstance(obj, dict):
            return None
        jm = str(obj.get("judgment", "")).strip()
        if jm not in JUDGMENTS:
            return None
        return EvidenceJudgment(
            judgment=jm,
            candidate_role_fit=str(obj.get("candidate_role_fit", "unclear")),
            source_role_fit=str(obj.get("source_role_fit", "weak")),
            supported_facets=[str(x) for x in (obj.get("supported_facets") or [])],
            unsupported_facets=[str(x) for x in (obj.get("unsupported_facets") or [])],
            contradicted_facets=[str(x) for x in (obj.get("contradicted_facets") or [])],
            quote=str(obj.get("quote", "") or ""), rationale=str(obj.get("rationale", "") or ""),
            confidence=float(obj.get("confidence", 0.0) or 0.0),
            requires_read_reason=(str(obj["requires_read_reason"])
                                  if obj.get("requires_read_reason") else None),
            candidate_aliases=[str(x) for x in (obj.get("candidate_aliases") or [])],
            safety_notes=[str(x) for x in (obj.get("safety_notes") or [])],
            mode="llm", model_called=True)
