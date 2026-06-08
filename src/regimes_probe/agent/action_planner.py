"""Epistemic action planner over task-frame slots and constraints.

Replaces the "top candidate + next clue" follow-up with a planner that asks, each
step: which slot am I binding, which constraint does this query test, which
hypothesis would it distinguish, and what would count as progress. It chooses an
action type, generates a query from slots+constraints (not loose spans), or gates a
read by whether the URL could resolve an unresolved slot/constraint. Deterministic,
gold-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

from regimes_probe.policy.query_decomposition import (
    _cap, _tokens, _is_rare, extract_clues, MAX_QUERY_CHARS, MAX_QUERY_TOKENS)

#: Affordance-driven action kinds the planner emits (plus read_url_for_constraint,
#: emitted by the search loop's frame-grounded read stage).
ACTION_TYPES = ("search_to_bind_slot", "search_to_test_constraint",
                "compare_candidates", "read_to_verify_constraint",
                "read_url_for_constraint", "answer_if_supported", "abstain_if_blocked")
#: A target hypothesis needs at least this much constraint support to answer.
_MIN_SUPPORT_TO_ANSWER = 2


@dataclass
class EpistemicAction:
    action_id: str
    kind: str
    target_slot_id: Optional[str] = None
    tested_constraint_ids: list[str] = field(default_factory=list)
    hypothesis_id: Optional[str] = None
    query: str = ""
    query_arm: str = ""
    expected_information_gain: float = 0.0
    rationale: str = ""
    read_obs_url: Optional[str] = None

    def to_dict(self, *, preview: int = 160) -> dict[str, Any]:
        q = self.query if len(self.query) <= preview else self.query[:preview - 1] + "…"
        return {"action_id": self.action_id, "kind": self.kind,
                "target_slot_id": self.target_slot_id,
                "tested_constraint_ids": list(self.tested_constraint_ids),
                "hypothesis_id": self.hypothesis_id, "query_text_preview": q,
                "query_arm": self.query_arm,
                "expected_information_gain": round(self.expected_information_gain, 3),
                "rationale": self.rationale}


@dataclass
class AnswerSupportResult:
    """Strict gate: may the agent answer from the current hypothesis + evidence?

    Answering must be tied to *supported evidence*, not to a slot merely being
    bound. ``missing_support_reasons`` is empty iff ``supported`` is True.
    """
    supported: bool
    hypothesis_id: Optional[str] = None
    target_candidate: Optional[str] = None
    support_score: float = 0.0
    missing_support_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"answer_supported": self.supported,
                "hypothesis_id": self.hypothesis_id,
                "target_candidate": self.target_candidate,
                "support_score": round(self.support_score, 3),
                "missing_support_reasons": list(self.missing_support_reasons)}


def evaluate_answer_support(frame, table, best=None) -> AnswerSupportResult:
    """Decide whether a final answer is *supported by the question's constraints*.

    ``answer_supported`` is True only when ALL hold (else each failing check adds a
    reason): a target slot candidate exists; non-contaminated evidence tied to the
    selected hypothesis explicitly supports the target slot or a found answer-shape
    hint; the target's most-discriminative constraint(s) are resolved (not
    unresolved); no constraint on a bound slot is contradicted; and the hypothesis
    carries at least ``_MIN_SUPPORT_TO_ANSWER`` supported constraints. This guards
    against the failure where a slot is bound but nothing actually supports it.
    """
    best = best if best is not None else table.best_hypothesis()
    reasons: list[str] = []
    tgt = frame.target_answer_slots[0] if frame.target_answer_slots else None
    if tgt is None:
        return AnswerSupportResult(False, missing_support_reasons=["no_target_slot"])
    if best is None:
        return AnswerSupportResult(False, missing_support_reasons=["no_active_hypothesis"])

    cand_id = best.slot_assignments.get(tgt.slot_id)
    cand_text = table.candidate_text(cand_id) if cand_id else None
    if not cand_id:
        reasons.append("target_slot_unbound")

    # (1) clean, hypothesis-tied evidence that supports the target slot or answer shape.
    ev_ids = set(best.evidence_ids)
    clean = [e for e in getattr(table, "evidence", [])
             if e.evidence_id in ev_ids and float(getattr(e, "contamination_score", 0.0)) == 0.0
             and (tgt.slot_id in getattr(e, "supports_slot_ids", [])
                  or getattr(e, "answer_shape_hints_found", []))]
    if not clean:
        reasons.append("no_clean_evidence_supports_target_or_answer_shape")

    bound = set(best.slot_assignments) | {tgt.slot_id}
    # (2) required BLOCKING constraints (affordance/flag) on a bound slot must be
    #     resolved. Falls back to the most-discriminative target constraint when the
    #     frame declares no explicit blocking constraints.
    def _blocks(c) -> bool:
        return (getattr(c, "blocks_answer_if_unresolved", False)
                or "can_block_answer" in getattr(c, "affordances", []))
    blocking = [c for c in frame.constraints
                if _blocks(c) and (set(c.applies_to) & bound)]
    if not blocking:
        tgt_cons = [c for c in frame.constraints
                    if tgt.slot_id in c.applies_to and c.constraint_type != "answer_shape"]
        if tgt_cons:
            top = max(c.discriminative_score for c in tgt_cons)
            blocking = [c for c in tgt_cons if c.discriminative_score >= top - 1e-9]
    unresolved_blocking = [c.constraint_id for c in blocking if c.status != "resolved"]
    if unresolved_blocking:
        reasons.append("required_blocking_constraint_unresolved:" + ",".join(unresolved_blocking))

    # (3) no constraint on a bound slot is contradicted.
    contradicted = [c.constraint_id for c in frame.constraints
                    if c.status == "contradicted" and (set(c.applies_to) & bound)]
    if contradicted:
        reasons.append("constraints_contradicted:" + ",".join(contradicted))

    # (4) the hypothesis must carry enough supported constraints (tied to it).
    if best.support_score < _MIN_SUPPORT_TO_ANSWER:
        reasons.append("insufficient_constraint_support")

    return AnswerSupportResult(
        supported=not reasons, hypothesis_id=best.hypothesis_id,
        target_candidate=cand_text, support_score=best.support_score,
        missing_support_reasons=reasons)


@dataclass
class ReadValueDecision:
    should_read: bool
    target_slot_id: Optional[str] = None
    tested_constraint_ids: list[str] = field(default_factory=list)
    hypothesis_id: Optional[str] = None
    read_selected_reason: Optional[str] = None
    read_rejected_reason: Optional[str] = None
    read_candidate_score: float = 0.0
    no_evidence_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {"should_read": self.should_read, "target_slot_id": self.target_slot_id,
                "tested_constraint_ids": list(self.tested_constraint_ids),
                "hypothesis_id": self.hypothesis_id,
                "read_selected_reason": self.read_selected_reason,
                "read_rejected_reason": self.read_rejected_reason,
                "read_candidate_score": round(self.read_candidate_score, 3),
                "no_evidence_reason": self.no_evidence_reason}


def _distinctive_phrase(constraint) -> str:
    """The most distinctive contiguous span of a constraint (for a query)."""
    spans = extract_clues(constraint.text_span).phrase_spans
    if spans:
        return spans[0]
    rare = [t for t in constraint.normalized_terms if _is_rare(t)]
    return " ".join((rare or constraint.normalized_terms)[:4])


def _constraint_text_match(obs, constraint) -> bool:
    text_l = f"{getattr(obs, 'title', '') or ''} {getattr(obs, 'snippet', '') or ''}".lower()
    terms = [t for t in constraint.normalized_terms if len(t) >= 4]
    return bool(terms and sum(1 for t in terms if t in text_l) >= max(2, (len(terms) + 1) // 2))


def frame_read_value(obs, frame, table, *, cross_provider: bool = False,
                     allow_social: bool = False) -> ReadValueDecision:
    """Gate a read by whether the URL could resolve an unresolved slot/constraint."""
    from regimes_probe.agent.reading_policy import _is_social, normalize_url, _STRUCTURED_MARKERS
    url = getattr(obs, "url", "") or ""
    host = (urlparse(url).hostname or "").lower()
    text = f"{getattr(obs, 'title', '') or ''} {getattr(obs, 'snippet', '') or ''}"
    text_l = text.lower()
    if getattr(obs, "benchmark_contaminated", False):
        return ReadValueDecision(False, read_rejected_reason="benchmark_contaminated")
    if _is_social(host) and not allow_social:
        return ReadValueDecision(False, read_rejected_reason="social_or_platform_page")
    matched = [c.constraint_id for c in frame.unresolved_constraints
               if _constraint_text_match(obs, c)]
    shape = any(h.lower() in text_l for h in frame.answer_shape_hints)
    authority = float(getattr(obs, "source_authority", 0.0))
    structured = url.lower().endswith(".pdf") or any(m in text_l for m in _STRUCTURED_MARKERS)
    # supports the current best hypothesis's target candidate?
    best = table.best_hypothesis()
    supports_hyp = False
    if best:
        for cid in best.slot_assignments.values():
            ct = table.candidate_text(cid).lower()
            if ct and ct in text_l:
                supports_hyp = True
                break
    reason = None
    if supports_hyp:
        reason = "supports_candidate_hypothesis"
    elif matched:
        reason = "contains_unresolved_clue"
    elif shape:
        reason = "contains_answer_shape_hint"
    elif cross_provider:
        reason = "cross_provider_same_url_or_domain"
    elif structured and authority >= 0.6 and matched:
        reason = "structured_or_pdf_and_high_relevance"
    if reason is None:
        # 'generic authoritative + insufficient snippet' is NOT a valid reason alone.
        return ReadValueDecision(False, read_rejected_reason="no_unresolved_slot_or_constraint",
                                 no_evidence_reason="url_matches_no_constraint")
    tgt = frame.target_answer_slots[0].slot_id if frame.target_answer_slots else None
    score = float(len(matched) + (1.0 if supports_hyp else 0.0) + (0.5 if shape else 0.0)
                  + authority)
    return ReadValueDecision(True, target_slot_id=tgt, tested_constraint_ids=matched,
                             hypothesis_id=(best.hypothesis_id if best else None),
                             read_selected_reason=reason, read_candidate_score=score)


class ActionPlanner:
    """Pick the next epistemic action from the frame + hypothesis table."""

    def __init__(self, frame, table) -> None:
        self.frame = frame
        self.table = table
        self._ac = 0
        self._query_hashes: set[str] = set()
        self._slot_attempts: dict[str, int] = {}
        self.last_answer_support: Optional[AnswerSupportResult] = None

    def _next_id(self) -> str:
        self._ac += 1
        return f"a{self._ac}"

    def _unbound_slots(self) -> list:
        bound = {sid for h in self.table.hypotheses.values() for sid in h.slot_assignments}
        return [s for s in self.frame.all_slots if s.slot_id not in bound]

    def _query(self, q: str) -> tuple[str, bool]:
        q = _cap(q)
        import hashlib
        h = hashlib.sha256(q.lower().encode()).hexdigest()[:16]
        repeat = h in self._query_hashes
        self._query_hashes.add(h)
        return q, repeat

    def _constraint_priority_key(self, con) -> tuple:
        """Order constraints by expected information gain for the planner.

        Unresolved high-priority BLOCKING constraints first; then constraints that
        support the answer; then by declared priority and specificity. The planner
        branches on these *affordances*, not on the free-form semantic label.
        """
        pr = {"high": 2, "medium": 1, "low": 0}.get(getattr(con, "priority", "medium"), 1)
        blocks = 1 if ("can_block_answer" in getattr(con, "affordances", [])
                       or getattr(con, "blocks_answer_if_unresolved", False)) else 0
        supports = 1 if "can_support_answer" in getattr(con, "affordances", []) else 0
        return (blocks, supports, pr, con.specificity_score)

    def plan(self, *, budget_remaining: int, reading_available: bool) -> EpistemicAction:
        f, t = self.frame, self.table
        best = t.best_hypothesis()
        tgt_slot = f.target_answer_slots[0] if f.target_answer_slots else None

        # 1. Answer only when the strict support gate passes.
        supp = evaluate_answer_support(f, t, best)
        self.last_answer_support = supp
        if supp.supported and tgt_slot:
            return EpistemicAction(self._next_id(), "answer_if_supported",
                                   target_slot_id=tgt_slot.slot_id,
                                   hypothesis_id=best.hypothesis_id if best else None,
                                   rationale="target bound + evidence supports target/constraints")
        # 2. Abstain when out of budget or there is no actionable path.
        if budget_remaining <= 0 or (not f.unresolved_constraints and not f.all_slots):
            return self._abstain(supp)

        unbound_ids = {s.slot_id for s in self._unbound_slots()}
        # Candidate constraints to act on, ordered by expected information gain.
        actionable = [c for c in f.unresolved_constraints
                      if c.constraint_type != "answer_shape"
                      and ("can_verify" in c.affordances or "can_search" in c.affordances
                           or "can_bind_slot" in c.affordances)]
        actionable.sort(key=self._constraint_priority_key, reverse=True)

        for con in actionable:
            aff = con.affordances
            tgt_bound = bool(best and tgt_slot and tgt_slot.slot_id in best.slot_assignments)

            # 3a. compare candidates when the constraint is comparative and the slot it
            #     applies to has competing candidates to disambiguate.
            if "can_compare" in aff:
                sid = next((s for s in con.applies_to), None)
                if sid and self.table_has_multiple_candidates(sid):
                    cand = t.candidate_text(best.slot_assignments.get(sid, "")) if best else ""
                    phrase = _distinctive_phrase(con)
                    q, repeat = self._query(f'"{phrase}" {cand}'.strip())
                    if not repeat and q:
                        return EpistemicAction(
                            self._next_id(), "compare_candidates", target_slot_id=sid,
                            tested_constraint_ids=[con.constraint_id],
                            hypothesis_id=(best.hypothesis_id if best else None), query=q,
                            query_arm="compare", expected_information_gain=con.discriminative_score + 0.5,
                            rationale=f"compare candidates for slot {sid} via {con.constraint_id}")

            # 3b. test a bound target candidate against this constraint.
            if tgt_bound and ("can_verify" in aff) and (tgt_slot.slot_id in con.applies_to
                                                        or "can_support_answer" in aff):
                cand = t.candidate_text(best.slot_assignments[tgt_slot.slot_id])
                phrase = _distinctive_phrase(con)
                q, repeat = self._query(f'"{cand}" {phrase}')
                if not repeat and q:
                    return EpistemicAction(
                        self._next_id(), "search_to_test_constraint",
                        target_slot_id=tgt_slot.slot_id, tested_constraint_ids=[con.constraint_id],
                        hypothesis_id=best.hypothesis_id, query=q, query_arm="candidate_constraint",
                        expected_information_gain=con.discriminative_score + 1.0,
                        rationale=f"verify candidate '{cand}' against {con.constraint_id}")

            # 3c. search to bind an unbound slot this constraint identifies.
            if "can_bind_slot" in aff:
                slot = next((f.slot(sid) for sid in con.applies_to if sid in unbound_ids), None)
                slot = slot or (tgt_slot if tgt_slot and tgt_slot.slot_id in unbound_ids else None)
                phrase = _distinctive_phrase(con)
                obj = slot.slot_name if slot else (tgt_slot.slot_name if tgt_slot else "")
                q, repeat = self._query(f'"{phrase}" {obj}' if phrase else obj)
                if not repeat and q:
                    sid = slot.slot_id if slot else "?"
                    self._slot_attempts[sid] = self._slot_attempts.get(sid, 0) + 1
                    return EpistemicAction(
                        self._next_id(), "search_to_bind_slot",
                        target_slot_id=(slot.slot_id if slot else None),
                        tested_constraint_ids=[con.constraint_id], query=q, query_arm="slot_seeking",
                        expected_information_gain=con.specificity_score,
                        rationale=f"bind slot via constraint {con.constraint_id}")

        # 4. Nothing distinguishing left to try → abstain (blocked).
        return self._abstain(supp)

    def table_has_multiple_candidates(self, slot_id: str) -> int:
        return sum(1 for c in self.table.candidates.values()
                   if not getattr(c, "rejected", False)
                   and slot_id in getattr(c, "support_for_slot_ids", []))

    def _abstain(self, supp) -> EpistemicAction:
        reasons = supp.missing_support_reasons if supp else []
        return EpistemicAction(
            self._next_id(), "abstain_if_blocked",
            rationale=("blocked: " + "; ".join(reasons[:3])) if reasons
            else "no unresolved distinguishing constraint remains")
