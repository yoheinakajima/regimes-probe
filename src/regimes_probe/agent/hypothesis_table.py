"""Hypothesis table + evidence ledger for constraint-satisfaction search.

A hypothesis is a *partial assignment of candidates to task-frame slots*, scored by
how many constraints it supports/contradicts and how many slots it covers. Every
search/scrape/fetch result becomes an :class:`EvidenceRecord` linked to the slots,
constraints, and candidates it touches. All of this is gold-free and answer-free
(only result text + the question's own constraints are used); nothing here is ever
written into policy memory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

from regimes_probe.agent.clue_resolution import (
    _domain_label, _entities_in_field, _is_generic_entity, _norm, classify_entity_role)
from regimes_probe.agent.task_frame import _ROLE_TRIGGERS
from regimes_probe.policy.query_decomposition import _tokens

#: a distinctive term carries a real anchor (a year or a rare/proper token), so a blocking
#: constraint cannot resolve from generic glue-word overlap with a question-echo (req 4).
_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")


def _is_rare_term(t: str) -> bool:
    try:
        from regimes_probe.policy.query_decomposition import _is_rare
        return bool(_is_rare(t))
    except Exception:
        return False


def _host(url: str) -> str:
    return (urlparse(url or "").hostname or "").lower()


def _context_role(entity: str, *texts: str) -> tuple[str, bool]:
    """Role-type a candidate using nearby role-trigger nouns when available.

    A bare multi-word proper noun ("Casa Verde") cannot be reliably typed in
    isolation, but the surrounding evidence usually names its kind ("Casa Verde
    restaurant", "actor Jane Doe"). When a role-trigger noun sits immediately
    beside the mention we trust it over the isolated guess. Returns
    ``(role, from_context)``.
    """
    el = (entity or "").lower()
    for text in texts:
        tl = (text or "").lower()
        idx = tl.find(el)
        if idx < 0:
            continue
        after = re.findall(r"[a-z]+", tl[idx + len(el): idx + len(el) + 32])[:2]
        before = re.findall(r"[a-z]+", tl[max(0, idx - 32): idx])[-2:]
        for w in after + before:
            role = _ROLE_TRIGGERS.get(w)
            if role:
                return role, True
    return classify_entity_role(entity), False


def _prev(s: str, n: int = 160) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n - 1] + "…"


@dataclass
class Candidate:
    candidate_id: str
    candidate_text: str
    normalized_text: str
    inferred_role: str
    source_result_ids: list[str] = field(default_factory=list)
    source_domain: str = ""
    role_confidence: float = 0.5
    extraction_confidence: float = 0.5
    not_gold_answer: bool = True          # we NEVER know the gold answer
    observed_in_stage: int = 1
    support_for_slot_ids: list[str] = field(default_factory=list)
    rejected: bool = False
    rejection_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, "candidate_text": self.candidate_text,
                "inferred_role": self.inferred_role, "source_domain": self.source_domain,
                "role_confidence": round(self.role_confidence, 3),
                "observed_in_stage": self.observed_in_stage,
                "support_for_slot_ids": list(self.support_for_slot_ids),
                "rejected": self.rejected, "rejection_reason": self.rejection_reason}


@dataclass
class EvidenceRecord:
    evidence_id: str
    source_tool: str
    url: str = ""
    domain: str = ""
    title_preview: str = ""
    snippet_preview: str = ""
    supports_candidate_ids: list[str] = field(default_factory=list)
    supports_slot_ids: list[str] = field(default_factory=list)
    supports_constraint_ids: list[str] = field(default_factory=list)
    contradicts_constraint_ids: list[str] = field(default_factory=list)
    newly_introduced_candidates: list[str] = field(default_factory=list)
    answer_shape_hints_found: list[str] = field(default_factory=list)
    source_authority_score: float = 0.0
    contamination_score: float = 0.0
    read_depth: int = 0                   # 0=snippet, 1=fetch, 2=scrape
    evidence_progress_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"evidence_id": self.evidence_id, "source_tool": self.source_tool,
                "domain": self.domain, "title_preview": _prev(self.title_preview),
                "supports_slot_ids": list(self.supports_slot_ids),
                "supports_constraint_ids": list(self.supports_constraint_ids),
                "contradicts_constraint_ids": list(self.contradicts_constraint_ids),
                "newly_introduced_candidates": list(self.newly_introduced_candidates)[:6],
                "answer_shape_hints_found": list(self.answer_shape_hints_found),
                "source_authority_score": round(self.source_authority_score, 3),
                "contamination_score": round(self.contamination_score, 3),
                "read_depth": self.read_depth,
                "evidence_progress_score": round(self.evidence_progress_score, 3)}


@dataclass
class Hypothesis:
    hypothesis_id: str
    slot_assignments: dict[str, str] = field(default_factory=dict)   # slot_id -> candidate_id
    constraint_status: dict[str, str] = field(default_factory=dict)  # constraint_id -> status
    evidence_ids: list[str] = field(default_factory=list)
    support_score: float = 0.0
    contradiction_score: float = 0.0
    coverage_score: float = 0.0
    confidence_score: float = 0.0
    last_action_id: Optional[str] = None
    active: bool = True
    rejection_reason: Optional[str] = None
    no_progress_count: int = 0

    def recompute(self, n_slots: int, n_constraints: int) -> None:
        self.support_score = sum(1 for v in self.constraint_status.values() if v == "supported")
        self.contradiction_score = sum(1 for v in self.constraint_status.values() if v == "contradicted")
        self.coverage_score = (len(self.slot_assignments) / n_slots) if n_slots else 0.0
        self.confidence_score = (self.support_score - 1.5 * self.contradiction_score
                                 + 0.5 * self.coverage_score)

    def to_dict(self, table: "HypothesisTable") -> dict[str, Any]:
        return {"hypothesis_id": self.hypothesis_id,
                "slot_assignments": {sid: table.candidate_text(cid)
                                     for sid, cid in self.slot_assignments.items()},
                "constraint_status": dict(self.constraint_status),
                "support_score": round(self.support_score, 3),
                "contradiction_score": round(self.contradiction_score, 3),
                "coverage_score": round(self.coverage_score, 3),
                "confidence_score": round(self.confidence_score, 3),
                "active": self.active, "rejection_reason": self.rejection_reason}


class HypothesisTable:
    MAX_NO_PROGRESS = 2

    def __init__(self, frame) -> None:
        self.frame = frame
        self.candidates: dict[str, Candidate] = {}        # keyed by normalized text
        self._by_id: dict[str, Candidate] = {}            # keyed by candidate_id
        self.hypotheses: dict[str, Hypothesis] = {}
        self.evidence: list[EvidenceRecord] = []
        self._cc = 0
        self._ec = 0
        self._hc = 0
        self._known = {_norm(t) for t in frame.known_context_terms}

    # ---------- lookups ----------
    def candidate_text(self, cid: str) -> str:
        c = self._by_id.get(cid) or self.candidates.get(cid)
        return c.candidate_text if c else cid

    def _target_slot_ids(self) -> list[str]:
        return [s.slot_id for s in self.frame.target_answer_slots]

    def _slots_for_role(self, role: str) -> list:
        return [s for s in self.frame.all_slots if s.slot_role == role]

    # ---------- ingest ----------
    def ingest_evidence(self, observations, *, source_tool: str, stage: int,
                        read_depth: int = 0, action_id: Optional[str] = None) -> EvidenceRecord:
        """Turn results into an evidence record; extract candidates, assign to
        role-compatible slots, support/contradict constraints, advance hypotheses."""
        self._ec += 1
        ev = EvidenceRecord(evidence_id=f"e{self._ec}", source_tool=source_tool,
                            read_depth=read_depth)
        progressed = False
        for o in observations:
            if getattr(o, "failed", False):
                continue
            title = getattr(o, "title", "") or ""
            snippet = getattr(o, "snippet", "") or ""
            url = getattr(o, "url", "") or ""
            ev.url = ev.url or url
            ev.domain = ev.domain or _host(url)
            ev.title_preview = ev.title_preview or title
            ev.snippet_preview = ev.snippet_preview or snippet
            ev.source_authority_score = max(ev.source_authority_score,
                                            float(getattr(o, "source_authority", 0.0)))
            if getattr(o, "benchmark_contaminated", False):
                ev.contamination_score = 1.0
            text_l = f"{title} {snippet}".lower()
            for e in _entities_in_field(title) + _entities_in_field(snippet):
                if _is_generic_entity(e) or len(e) < 3:
                    continue
                key = _norm(e)
                cand = self.candidates.get(key)
                if cand is None:
                    self._cc += 1
                    role, from_ctx = _context_role(e, title, snippet)
                    cand = Candidate(candidate_id=f"cand{self._cc}", candidate_text=e,
                                     normalized_text=key, inferred_role=role,
                                     source_domain=_host(url), observed_in_stage=stage,
                                     role_confidence=0.75 if from_ctx else (0.6 if " " in e else 0.4))
                    # Assign to role-compatible slots; NEVER promote a known context
                    # term into a target slot.
                    is_known = key in self._known
                    for s in self._slots_for_role(role):
                        if s.is_target_answer_slot and is_known:
                            continue
                        cand.support_for_slot_ids.append(s.slot_id)
                    self.candidates[key] = cand
                    self._by_id[cand.candidate_id] = cand
                    ev.newly_introduced_candidates.append(cand.candidate_id)
                    progressed = True
                if cand.candidate_id not in ev.supports_candidate_ids:
                    ev.supports_candidate_ids.append(cand.candidate_id)
                for sid in cand.support_for_slot_ids:
                    if sid not in ev.supports_slot_ids:
                        ev.supports_slot_ids.append(sid)
            # answer-shape hints present?
            for hint in self.frame.answer_shape_hints:
                if hint.lower() in text_l and hint not in ev.answer_shape_hints_found:
                    ev.answer_shape_hints_found.append(hint)
            # constraint support: most normalized terms present in this evidence text.
            # SAFETY (Level 5g/req 4): never resolve a blocking constraint from a
            # contaminated source (a benchmark mirror echoing the question is NOT evidence),
            # and require at least one DISTINCTIVE term (proper noun / year), so a generic
            # question-echo cannot resolve a constraint by glue-word overlap alone.
            contaminated_ev = bool(getattr(o, "benchmark_contaminated", False))
            for con in self.frame.constraints:
                if con.status == "contradicted":
                    continue
                terms = [t for t in con.normalized_terms if len(t) >= 4]
                if not (terms and sum(1 for t in terms if t in text_l) >= max(2, (len(terms) + 1) // 2)):
                    continue
                # a distinctive anchor = a year, or a term that is PROPER-CASED in the
                # constraint's own text_span (a named entity), present in this evidence.
                proper = {w.lower() for w in re.findall(r"[A-Z][A-Za-z'&]{3,}",
                                                        getattr(con, "text_span", "") or "")}
                distinctive = any(
                    t in text_l and (bool(_YEAR_RE.search(t)) or t.lower() in proper)
                    for t in con.normalized_terms if len(t) >= 4)
                blocking = bool(getattr(con, "blocks_answer_if_unresolved", False)
                                or getattr(con, "required", False)
                                or getattr(con, "priority", "") == "high")
                if con.constraint_id not in ev.supports_constraint_ids:
                    ev.supports_constraint_ids.append(con.constraint_id)
                # a blocking constraint resolves only from clean evidence with a real anchor.
                if blocking and (contaminated_ev or not distinctive):
                    continue
                if ev.evidence_id not in con.supporting_evidence_ids:
                    con.supporting_evidence_ids.append(ev.evidence_id)
                if con.status != "resolved":
                    con.status = "resolved"
                    progressed = True
        ev.evidence_progress_score = float(
            len(ev.newly_introduced_candidates) + len(ev.supports_constraint_ids)
            + len(ev.answer_shape_hints_found))
        self.evidence.append(ev)
        self._advance_hypotheses(ev, action_id)
        ev._progressed = progressed  # type: ignore[attr-defined]
        return ev

    # ---------- hypotheses ----------
    def _advance_hypotheses(self, ev: EvidenceRecord, action_id: Optional[str]) -> None:
        """Create/extend a hypothesis per distinct target-slot candidate."""
        target_ids = self._target_slot_ids()
        if not target_ids:
            return
        tgt = target_ids[0]
        for cid in ev.supports_candidate_ids:
            cand = self._by_id.get(cid)
            if cand is None or cand.rejected or tgt not in cand.support_for_slot_ids:
                continue
            hyp = self._hyp_for_target(cand.candidate_id, tgt)
            hyp.evidence_ids.append(ev.evidence_id)
            hyp.last_action_id = action_id
            # bind best intermediate candidates (role-compatible) too.
            for s in self.frame.latent_slots:
                if s.slot_id in hyp.slot_assignments:
                    continue
                bc = next((c for c in self.candidates.values()
                           if not c.rejected and s.slot_id in c.support_for_slot_ids), None)
                if bc is not None:
                    hyp.slot_assignments[s.slot_id] = bc.candidate_id
            # constraint status for this hypothesis's slots.
            for con in self.frame.constraints:
                if any(sid in hyp.slot_assignments for sid in con.applies_to):
                    hyp.constraint_status[con.constraint_id] = (
                        "supported" if con.status == "resolved" else
                        ("contradicted" if con.status == "contradicted" else "unknown"))
            hyp.recompute(len(self.frame.all_slots), len(self.frame.constraints))

    def _hyp_for_target(self, cid: str, tgt: str) -> Hypothesis:
        for h in self.hypotheses.values():
            if h.slot_assignments.get(tgt) == cid:
                return h
        self._hc += 1
        h = Hypothesis(hypothesis_id=f"h{self._hc}", slot_assignments={tgt: cid})
        self.hypotheses[h.hypothesis_id] = h
        return h

    def _cid_key(self, candidate_id: str) -> str:
        return next((k for k, c in self.candidates.items() if c.candidate_id == candidate_id),
                    candidate_id)

    def note_no_progress(self, hypothesis_id: str) -> None:
        h = self.hypotheses.get(hypothesis_id)
        if not h or not h.active:
            return
        h.no_progress_count += 1
        if h.no_progress_count >= self.MAX_NO_PROGRESS:
            h.active = False
            h.rejection_reason = "repeated_no_progress"

    def reject_contradicted(self) -> None:
        for h in self.hypotheses.values():
            if h.active and h.contradiction_score > 0 and h.support_score == 0:
                h.active = False
                h.rejection_reason = "contradicted_by_evidence"

    def best_hypothesis(self) -> Optional[Hypothesis]:
        active = [h for h in self.hypotheses.values() if h.active]
        if not active:
            return None
        return max(active, key=lambda h: (h.confidence_score, h.support_score,
                                          h.coverage_score))

    # ---------- coverage / metrics ----------
    def coverage(self) -> dict[str, float]:
        slots = self.frame.all_slots
        cons = self.frame.constraints
        tgt = self._target_slot_ids()
        best = self.best_hypothesis()
        target_supported = 0.0
        if best and tgt:
            tslot = tgt[0]
            if tslot in best.slot_assignments:
                # target slot is "supported" if any constraint applying to it is supported
                target_supported = 1.0 if any(
                    best.constraint_status.get(c.constraint_id) == "supported"
                    for c in cons if tslot in c.applies_to) else 0.0
        assigned = len({sid for h in self.hypotheses.values() for sid in h.slot_assignments})
        return {
            "slot_resolution_rate": (assigned / len(slots)) if slots else 0.0,
            "constraint_support_rate": (sum(1 for c in cons if c.status == "resolved")
                                        / len(cons)) if cons else 0.0,
            "target_slot_support_rate": target_supported,
            "hypothesis_coverage_score": best.coverage_score if best else 0.0,
            "n_hypotheses": len(self.hypotheses),
            "n_rejected_hypotheses": sum(1 for h in self.hypotheses.values() if not h.active),
        }

    def to_debug(self, *, top: int = 3) -> dict[str, Any]:
        ranked = sorted(self.hypotheses.values(),
                        key=lambda h: (-h.confidence_score, h.hypothesis_id))
        return {
            "n_candidates": len(self.candidates),
            "n_evidence": len(self.evidence),
            "top_hypotheses": [h.to_dict(self) for h in ranked[:top]],
            "rejected_hypotheses": [
                {"hypothesis_id": h.hypothesis_id, "rejection_reason": h.rejection_reason}
                for h in self.hypotheses.values() if not h.active][:top],
        }
