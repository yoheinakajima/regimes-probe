"""Optional LLM frontier-action / query proposer (Level 5c).

The deterministic frontier controller (`candidate_frontier.py`) genuinely drives tool
calls, but its query *synthesis* is too generic — it emits bare slot descriptors like
"founder" / "report" / "publication" that retrieve dictionary/junk candidates. LLMs
compose constraint-grounded research actions far better. So this module lets an LLM
**propose** next research actions from a bounded, ActiveGraph-derived `ResearchStateCard`
— but the LLM never executes: deterministic code **validates, scores, selects, executes,
and records**. ActiveGraph stays the source of truth.

Discipline (mirrors the gated LLM task-frame parser):
- Answer-free / gold-free: the model sees a bounded state card (no gold, no secrets) and
  proposes ACTIONS, never answers. Nothing here reaches policy memory.
- Cached + replayable: every call is keyed by ``prompt_fingerprint | model | card_hash``;
  a replay (or no ``model_fn``) never calls a model — it falls back to the deterministic
  plan. Missing replay cache refuses rather than spends.
- Validated + scored: every proposal is checked (real slots/constraints, non-generic +
  anchored query, no duplicate, no premature answer, allowed tool, in budget) and scored
  before it can be selected; rejections are recorded with reasons.

Two modes: ``repair`` (only invoked when the deterministic query is generic/blocked) and
``planner`` (ask for top-K proposals from the graph state). Both default off and are
skipped for easy/direct/simple epistemic modes.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from regimes_probe.agent import prompts
from regimes_probe.agent.candidate_frontier import (
    FRONTIER_ACTION_TYPES, StepPlan, _disc, _is_discriminative_constraint)
from regimes_probe.agent.llm_task_frame import ParserCache, _extract_json, _sha
from regimes_probe.policy.query_decomposition import _cap

_WORD = re.compile(r"[A-Za-z0-9]+")
#: one-word / definitional / role-only queries that retrieve junk.
_GENERIC_WORDS = frozenset({
    "founder", "report", "publication", "nickname", "monument", "author", "person",
    "organization", "organisation", "restaurant", "hotel", "museum", "series", "actor",
    "company", "paper", "journal", "book", "film", "movie", "show", "name", "year",
    "definition", "meaning", "wikipedia", "antagonist", "potential"})


def _norm_q(q: str) -> str:
    return " ".join(w.lower() for w in _WORD.findall(q or ""))


def _is_generic_query(q: str) -> bool:
    toks = [w.lower() for w in _WORD.findall(q or "")]
    content = [t for t in toks if len(t) >= 3]
    if len(content) <= 1:
        return True
    # all content tokens are generic role/definition words -> generic.
    return all(t in _GENERIC_WORDS for t in content)


# --------------------------------------------------------------------------- card
@dataclass
class ResearchStateCard:
    """Bounded, ActiveGraph-derived snapshot the LLM reasons over (no gold/secrets)."""
    item_id: str
    question_preview: str
    epistemic_mode: str
    target_slots: list[dict[str, Any]] = field(default_factory=list)
    intermediate_slots: list[dict[str, Any]] = field(default_factory=list)
    known_context_terms: list[str] = field(default_factory=list)
    unresolved_blocking_constraints: list[dict[str, Any]] = field(default_factory=list)
    candidate_slates: list[dict[str, Any]] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    evidence_summaries: list[dict[str, Any]] = field(default_factory=list)
    failed_queries: list[str] = field(default_factory=list)
    no_progress_queries: list[str] = field(default_factory=list)
    available_tools: list[str] = field(default_factory=list)
    remaining_budget: int = 0
    memory_access_mode: str = ""
    deterministic_recommendation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id, "question_preview": self.question_preview[:300],
            "epistemic_mode": self.epistemic_mode,
            "target_slots": self.target_slots, "intermediate_slots": self.intermediate_slots,
            "known_context_terms": list(self.known_context_terms)[:12],
            "unresolved_blocking_constraints": self.unresolved_blocking_constraints[:12],
            "candidate_slates": self.candidate_slates[:8],
            "hypotheses": self.hypotheses[:4], "evidence_summaries": self.evidence_summaries[:6],
            "failed_queries": list(self.failed_queries)[:10],
            "no_progress_queries": list(self.no_progress_queries)[:10],
            "available_tools": list(self.available_tools),
            "remaining_budget": self.remaining_budget,
            "memory_access_mode": self.memory_access_mode,
            "deterministic_recommendation": self.deterministic_recommendation,
        }

    def card_hash(self) -> str:
        return _sha(json.dumps(self.to_dict(), sort_keys=True))


def build_research_state_card(frame, frontier, *, question: str, epistemic_mode: str,
                              available_tools: list[str], remaining_budget: int,
                              memory_access_mode: str, failed_queries: list[str],
                              no_progress_queries: list[str], det_plan) -> ResearchStateCard:
    """Build the card from the persisted TaskFrame + CandidateFrontier state."""
    def _slot(s) -> dict[str, Any]:
        return {"slot_id": s.slot_id, "descriptor": (s.descriptor_text or s.slot_name)[:80],
                "role": s.slot_role, "slot_status": getattr(s, "slot_status", "unbound_variable")}

    blocking = []
    for c in frame.constraints:
        if c.status != "resolved" and (getattr(c, "blocks_answer_if_unresolved", False)
                                       or getattr(c, "priority", "") == "high"
                                       or "can_block_answer" in getattr(c, "affordances", [])):
            blocking.append({"constraint_id": c.constraint_id,
                             "semantic_label": getattr(c, "semantic_label", "") or c.constraint_type,
                             "text_span": c.text_span[:120], "applies_to": list(c.applies_to),
                             "discriminative_score": round(_disc(c), 3),
                             "normalized_terms": list(c.normalized_terms)[:6]})
    fr_dbg = frontier.to_debug() if frontier is not None else {}
    slates = [{"slot_id": s.get("slot_id"), "slot_role": s.get("slot_role"),
               "active": [c.get("candidate_text_preview") for c in s.get("top_candidates", [])
                          if c.get("status") == "active"][:5],
               "confirmed": [c.get("candidate_text_preview") for c in s.get("top_candidates", [])
                             if c.get("status") == "confirmed"][:5],
               "rejected": [c.get("candidate_text_preview") for c in s.get("top_candidates", [])
                            if c.get("status") == "rejected"][:5]}
              for s in fr_dbg.get("slates", [])]
    hyps = [{"hypothesis_id": h.get("hypothesis_id"), "active": h.get("active"),
             "support_score": h.get("support_score"),
             "assignments": h.get("slot_candidate_assignments", {})}
            for h in fr_dbg.get("top_hypotheses", [])]
    ev = []
    for e in getattr(frontier, "evidence", [])[-6:] if frontier is not None else []:
        ev.append({"source_tool": e.source_tool, "domain": e.domain,
                   "title_preview": (e.title_preview or "")[:80],
                   "progress": round(float(getattr(e, "evidence_progress_score", 0.0)), 2)})
    det = {}
    if det_plan is not None:
        det = {"action_type": det_plan.action_type, "query": det_plan.query[:120],
               "target_slot_id": det_plan.target_slot_id, "kind": det_plan.kind,
               "is_generic_query": det_plan.is_generic_query}
    return ResearchStateCard(
        item_id=frame.item_id, question_preview=question, epistemic_mode=epistemic_mode,
        target_slots=[_slot(s) for s in frame.target_answer_slots],
        intermediate_slots=[_slot(s) for s in frame.latent_slots],
        known_context_terms=list(frame.known_context_terms),
        unresolved_blocking_constraints=blocking, candidate_slates=slates, hypotheses=hyps,
        evidence_summaries=ev, failed_queries=list(failed_queries),
        no_progress_queries=list(no_progress_queries), available_tools=list(available_tools),
        remaining_budget=remaining_budget, memory_access_mode=memory_access_mode,
        deterministic_recommendation=det)


# --------------------------------------------------------------------------- proposal
@dataclass
class FrontierProposal:
    proposal_id: str
    action_type: str
    target_slot_id: Optional[str] = None
    candidate_id: Optional[str] = None
    hypothesis_id: Optional[str] = None
    constraint_ids: list[str] = field(default_factory=list)
    proposed_query: str = ""
    proposed_tool_family: str = "search"
    expected_evidence: str = ""
    success_criteria: str = ""
    why_this_action: str = ""
    anchors_used: list[str] = field(default_factory=list)
    avoids_generic_query: bool = True
    risk_flags: list[str] = field(default_factory=list)
    confidence: float = 0.0
    # derived
    status: str = "proposed"                  # proposed | accepted | rejected
    rejection_reason: str = ""
    score: float = 0.0
    score_components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"proposal_id": self.proposal_id, "action_type": self.action_type,
                "target_slot_id": self.target_slot_id, "candidate_id": self.candidate_id,
                "hypothesis_id": self.hypothesis_id, "constraint_ids": list(self.constraint_ids),
                "proposed_query": self.proposed_query[:160],
                "proposed_tool_family": self.proposed_tool_family,
                "expected_evidence": self.expected_evidence[:160],
                "anchors_used": list(self.anchors_used)[:8],
                "avoids_generic_query": self.avoids_generic_query,
                "risk_flags": list(self.risk_flags)[:6], "confidence": round(self.confidence, 3),
                "status": self.status, "rejection_reason": self.rejection_reason,
                "score": round(self.score, 3),
                "score_components": {k: round(v, 3) for k, v in self.score_components.items()}}


def _extract_proposals(raw: str) -> list[dict]:
    obj = _extract_json(raw)
    if isinstance(obj, dict) and isinstance(obj.get("proposals"), list):
        return [p for p in obj["proposals"] if isinstance(p, dict)]
    # tolerate a bare top-level JSON array
    try:
        arr = json.loads(raw)
        if isinstance(arr, list):
            return [p for p in arr if isinstance(p, dict)]
    except Exception:
        pass
    return []


def _anchor_terms(frame) -> set[str]:
    terms: set[str] = set()
    for c in frame.constraints:
        for t in c.normalized_terms:
            if len(t) >= 4:
                terms.add(t.lower())
    for t in frame.known_context_terms:
        for w in _WORD.findall(t.lower()):
            if len(w) >= 3:
                terms.add(w)
    return terms


def validate_proposal(p: FrontierProposal, frame, frontier, *, failed_norms: set[str],
                      available_tools: list[str], remaining_budget: int) -> tuple[bool, str]:
    """Deterministic gate before a proposal may be scored/executed."""
    if p.action_type not in FRONTIER_ACTION_TYPES:
        return False, f"unknown_action_type:{p.action_type}"
    slot_ids = {s.slot_id for s in frame.all_slots}
    con_ids = {c.constraint_id for c in frame.constraints}
    if p.target_slot_id and p.target_slot_id not in slot_ids:
        return False, f"nonexistent_slot:{p.target_slot_id}"
    for cid in p.constraint_ids:
        if cid not in con_ids:
            return False, f"nonexistent_constraint:{cid}"
    if p.candidate_id and frontier is not None and p.candidate_id not in frontier.candidates_by_id:
        return False, f"nonexistent_candidate:{p.candidate_id}"
    if p.hypothesis_id and frontier is not None and p.hypothesis_id not in frontier.hypotheses:
        return False, f"nonexistent_hypothesis:{p.hypothesis_id}"
    if p.action_type == "answer_from_confirmed_hypothesis":
        best = frontier.best_hypothesis() if frontier is not None else None
        ok = bool(best and frontier._answerable(best))
        return (True, "") if ok else (False, "answer_without_supported_hypothesis")
    if p.action_type == "abstain_no_viable_hypothesis":
        return True, ""
    if remaining_budget <= 0:
        return False, "exceeds_budget"
    if p.proposed_tool_family and p.proposed_tool_family not in ("search", "scrape", "fetch"):
        return False, f"disallowed_tool:{p.proposed_tool_family}"
    if p.proposed_tool_family == "scrape" and "firecrawl_scrape" not in available_tools:
        return False, "tool_unavailable:scrape"
    # search/read actions need a real, non-generic, anchored, non-duplicate query.
    q = p.proposed_query.strip()
    if not q:
        return False, "empty_query"
    if _is_generic_query(q):
        return False, "generic_query"
    if _norm_q(q) in failed_norms:
        return False, "duplicate_no_progress_query"
    anchors = _anchor_terms(frame)
    qtok = {w.lower() for w in _WORD.findall(q)}
    if anchors and not (qtok & anchors):
        return False, "query_lacks_constraint_or_context_anchor"
    # must be connected to an unresolved slot or constraint.
    unresolved_slots = {s.slot_id for s in frame.all_slots}
    if frontier is not None:
        unresolved_slots = {sid for sid, sl in frontier.slates.items()
                            if not any(c.status == "confirmed" for c in sl.candidates.values())}
    unresolved_cons = {c.constraint_id for c in frame.constraints if c.status != "resolved"}
    if not ((p.target_slot_id in unresolved_slots) or (set(p.constraint_ids) & unresolved_cons)):
        return False, "not_connected_to_unresolved_slot_or_constraint"
    return True, ""


def score_proposal(p: FrontierProposal, frame, frontier, *, failed_norms: set[str]) -> None:
    cons = [c for c in frame.constraints if c.constraint_id in p.constraint_ids]
    anchors = _anchor_terms(frame)
    qtok = {w.lower() for w in _WORD.findall(p.proposed_query)}
    con_terms = {t.lower() for c in cons for t in c.normalized_terms if len(t) >= 4}
    kc = {w for t in frame.known_context_terms for w in _WORD.findall(t.lower()) if len(w) >= 3}
    comps = {
        "constraint_anchor_score": 1.0 if (qtok & con_terms) else 0.0,
        "known_context_anchor_score": 1.0 if (qtok & kc) else 0.0,
        "discriminative_constraint_score": max((_disc(c) for c in cons), default=0.0),
        "candidate_relevance": 0.5 if p.candidate_id else 0.0,
        "hypothesis_relevance": 0.5 if p.hypothesis_id else 0.0,
        "novelty": 1.0 if _norm_q(p.proposed_query) not in failed_norms else 0.0,
        "duplicate_penalty": -1.0 if _norm_q(p.proposed_query) in failed_norms else 0.0,
        "expected_cost": -1.0 if p.proposed_tool_family in ("scrape", "fetch") else -0.25,
        "tool_reliability": 0.5 if p.proposed_tool_family == "search" else 0.2,
        "no_progress_penalty": -0.5 * len([f for f in p.risk_flags if "no_progress" in f]),
        "confidence": 0.25 * float(p.confidence or 0.0),
    }
    p.score = round(sum(comps.values()), 4)
    p.score_components = comps


# --------------------------------------------------------------------------- proposer
class LLMFrontierProposer:
    """Cached/replayable LLM proposer. Validates + scores + selects; never executes."""

    def __init__(self, model_fn: Optional[Callable[[str], str]] = None, *,
                 cache: Optional[ParserCache] = None, model: str = "stub",
                 prompt_name: str = "frontier_planner", replay_only: bool = False) -> None:
        self.model_fn = model_fn
        self.cache = cache if cache is not None else ParserCache()
        self.model = model
        self.prompt = prompts.get(prompt_name)
        self.replay_only = replay_only
        self.model_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.proposals_count = 0
        self.accepted_count = 0
        self.selected_count = 0
        self.repair_invoked_count = 0
        self.fallback_count = 0
        self.rejection_counts: Counter = Counter()

    def stats(self) -> dict[str, Any]:
        return {"llm_frontier_model": self.model, "llm_frontier_model_calls": self.model_calls,
                "llm_frontier_cache_hits": self.cache_hits,
                "llm_frontier_cache_misses": self.cache_misses,
                "llm_frontier_proposals_count": self.proposals_count,
                "llm_frontier_accepted_count": self.accepted_count,
                "llm_frontier_selected_count": self.selected_count,
                "llm_frontier_repair_invocation_count": self.repair_invoked_count,
                "llm_frontier_fallback_count": self.fallback_count,
                "llm_frontier_rejection_counts": dict(self.rejection_counts)}

    def _input_hash(self, card: ResearchStateCard) -> str:
        return _sha(f"{self.prompt.fingerprint()}|{self.model}|{card.card_hash()}")

    def _call(self, card: ResearchStateCard, meta: dict) -> Optional[str]:
        ih = self._input_hash(card)
        meta["state_card_hash"] = card.card_hash()
        meta["input_hash"] = ih
        raw = self.cache.get(ih)
        if raw is not None:
            self.cache_hits += 1
            meta["cache_hit"] = True
            return raw
        self.cache_misses += 1
        if self.replay_only or self.model_fn is None:
            meta["fallback_reason"] = ("cache_miss_in_replay" if self.replay_only
                                       else "no_model_available")
            return None
        prompt_text = f"{self.prompt.content}\n\nRESEARCH_STATE_CARD:\n{json.dumps(card.to_dict())}"
        try:
            self.model_calls += 1
            raw = self.model_fn(prompt_text)
        except Exception as exc:
            meta["fallback_reason"] = f"model_error:{type(exc).__name__}"
            return None
        meta["model_called"] = True
        self.cache.put(ih, raw or "")
        return raw

    def plan_step(self, card: ResearchStateCard, det_plan, *, frame, frontier, mode: str,
                  observations=(), scraped_urls=frozenset(), no_progress_domains=frozenset(),
                  page_fetch_available: bool = False, scrape_available: bool = False,
                  allow_social: bool = False, force_page_fetch: bool = False,
                  failed_queries=()) -> tuple[Optional[StepPlan], dict[str, Any]]:
        meta: dict[str, Any] = {"mode": mode, "prompt_version": self.prompt.version,
                                "prompt_hash": self.prompt.content_hash, "model": self.model,
                                "cache_hit": False, "model_called": False,
                                "card_hash": card.card_hash(), "proposals": [], "selected": {},
                                "events": [], "repaired_generic": False,
                                "card_summary": {
                                    "epistemic_mode": card.epistemic_mode,
                                    "target_slots": [s.get("descriptor") for s in card.target_slots][:4],
                                    "known_context_terms": list(card.known_context_terms)[:8],
                                    "unresolved_blocking_constraints": [
                                        c.get("semantic_label") for c in
                                        card.unresolved_blocking_constraints][:6],
                                    "n_candidate_slates": len(card.candidate_slates),
                                    "n_hypotheses": len(card.hypotheses),
                                    "remaining_budget": card.remaining_budget,
                                    "memory_access_mode": card.memory_access_mode},
                                "det_recommendation": dict(card.deterministic_recommendation),
                                "det_action_type": (det_plan.action_type if det_plan else None),
                                "det_is_generic": (det_plan.is_generic_query if det_plan else None)}

        def _ev(t, **d):
            meta["events"].append({"event_type": t, **d})

        _ev("llm_frontier_state_card_created", card_hash=card.card_hash())
        # repair mode only fires when the deterministic query is generic / blocked.
        if mode == "repair":
            generic = det_plan is not None and (det_plan.is_generic_query
                                                or det_plan.kind == "unexecutable"
                                                or not det_plan.query.strip()
                                                or _is_generic_query(det_plan.query))
            if not generic:
                meta["skipped_reason"] = "deterministic_query_ok"
                return None, meta
            self.repair_invoked_count += 1
            _ev("llm_frontier_repair_invoked")

        raw = self._call(card, meta)
        if raw is None:
            self.fallback_count += 1
            return None, meta
        failed_norms = {_norm_q(q) for q in failed_queries}
        proposals: list[FrontierProposal] = []
        for i, d in enumerate(_extract_proposals(raw)):
            p = _proposal_from_dict(d, i)
            ok, reason = validate_proposal(p, frame, frontier, failed_norms=failed_norms,
                                           available_tools=card.available_tools,
                                           remaining_budget=card.remaining_budget)
            if ok:
                p.status = "accepted"
                score_proposal(p, frame, frontier, failed_norms=failed_norms)
                _ev("llm_frontier_proposal_validated", proposal_id=p.proposal_id)
            else:
                p.status, p.rejection_reason = "rejected", reason
                self.rejection_counts[reason.split(":")[0]] += 1
                _ev("llm_frontier_proposal_rejected", proposal_id=p.proposal_id, reason=reason)
            proposals.append(p)
        self.proposals_count += len(proposals)
        self.accepted_count += sum(1 for p in proposals if p.status == "accepted")
        _ev("llm_frontier_proposals_generated", n=len(proposals))
        meta["proposals"] = [p.to_dict() for p in proposals]
        meta["proposal_hash"] = _sha(raw or "")

        accepted = [p for p in proposals if p.status == "accepted"]
        if not accepted:
            self.fallback_count += 1
            meta["fallback_reason"] = meta.get("fallback_reason") or "no_accepted_proposal"
            return None, meta
        best = max(accepted, key=lambda p: (p.score, -proposals.index(p)))
        meta["selected"] = best.to_dict()
        self.selected_count += 1
        _ev("llm_frontier_proposal_selected", proposal_id=best.proposal_id)

        plan = self._to_step_plan(best, det_plan, mode=mode, frame=frame, frontier=frontier,
                                  observations=observations, scraped_urls=scraped_urls,
                                  no_progress_domains=no_progress_domains,
                                  page_fetch_available=page_fetch_available,
                                  scrape_available=scrape_available, allow_social=allow_social,
                                  force_page_fetch=force_page_fetch)
        if plan is None:
            self.fallback_count += 1
            meta["fallback_reason"] = "selected_proposal_unexecutable"
            return None, meta
        _ev("llm_frontier_proposal_executed", proposal_id=best.proposal_id, kind=plan.kind)
        meta["selected_proposal_id"] = best.proposal_id
        meta["repaired_generic"] = (mode == "repair")
        return plan, meta

    def _to_step_plan(self, p: FrontierProposal, det_plan, *, mode, frame, frontier, observations,
                      scraped_urls, no_progress_domains, page_fetch_available, scrape_available,
                      allow_social, force_page_fetch) -> Optional[StepPlan]:
        fa_id = f"lfp_{p.proposal_id}"
        if p.action_type == "answer_from_confirmed_hypothesis":
            return StepPlan(fa_id, p.action_type, "answer", reason="llm_frontier_proposal")
        if p.action_type == "abstain_no_viable_hypothesis":
            return StepPlan(fa_id, p.action_type, "abstain", reason="llm_frontier_proposal")
        if p.action_type == "read_candidate_source":
            chosen = frontier._resolve_read(observations, scraped_urls, no_progress_domains,
                                            page_fetch_available, scrape_available, allow_social,
                                            force_page_fetch) if frontier is not None else None
            if chosen is None:
                return None
            o, rd, tested = chosen
            return StepPlan(fa_id, p.action_type, "read", read_obs=o, read_decision=rd,
                            target_slot_id=p.target_slot_id, candidate_id=p.candidate_id,
                            constraint_ids=tested, reason="llm_frontier_proposal")
        # search-like: REPAIR keeps the deterministic action and swaps the query.
        query = _cap(p.proposed_query.strip())
        if mode == "repair" and det_plan is not None:
            return StepPlan(det_plan.frontier_action_id or fa_id, det_plan.action_type, "search",
                            query=query, query_arm="llm_frontier_repair",
                            target_slot_id=det_plan.target_slot_id, candidate_id=det_plan.candidate_id,
                            constraint_ids=list(det_plan.constraint_ids),
                            reason="llm_frontier_repair", is_discriminative_constraint=True,
                            is_generic_query=False)
        return StepPlan(fa_id, p.action_type, "search", query=query, query_arm="llm_frontier_planner",
                        target_slot_id=p.target_slot_id, candidate_id=p.candidate_id,
                        constraint_ids=list(p.constraint_ids), reason="llm_frontier_proposal",
                        is_discriminative_constraint=bool(
                            any(_is_discriminative_constraint(c) for c in frame.constraints
                                if c.constraint_id in p.constraint_ids)),
                        is_generic_query=False)


def _proposal_from_dict(d: dict, i: int) -> FrontierProposal:
    return FrontierProposal(
        proposal_id=str(d.get("proposal_id", f"p{i}")),
        action_type=str(d.get("action_type", "")),
        target_slot_id=(str(d["target_slot_id"]) if d.get("target_slot_id") else None),
        candidate_id=(str(d["candidate_id"]) if d.get("candidate_id") else None),
        hypothesis_id=(str(d["hypothesis_id"]) if d.get("hypothesis_id") else None),
        constraint_ids=[str(x) for x in (d.get("constraint_ids") or [])],
        proposed_query=str(d.get("proposed_query", "") or ""),
        proposed_tool_family=str(d.get("proposed_tool_family", "search") or "search"),
        expected_evidence=str(d.get("expected_evidence", "") or ""),
        success_criteria=str(d.get("success_criteria", "") or ""),
        why_this_action=str(d.get("why_this_action", "") or ""),
        anchors_used=[str(x) for x in (d.get("anchors_used") or [])],
        avoids_generic_query=bool(d.get("avoids_generic_query", True)),
        risk_flags=[str(x) for x in (d.get("risk_flags") or [])],
        confidence=float(d.get("confidence", 0.0) or 0.0))
