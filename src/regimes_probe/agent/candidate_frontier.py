"""ActiveGraph-native candidate-slate / frontier layer for multi-hop search.

A human solving a hard multi-hop question does not carry one global "best candidate":
they keep a **candidate slate per unresolved variable** (possible hotels, possible
museums, possible founders…), test candidates against that slot's constraints, reject
bad ones, expand promising ones to dependent slots, and only answer from a supported
hypothesis. This module makes that pattern first-class.

It is **generic** (no fixed slot names / constraint labels / gold answers), **event
sourced** (every candidate/status/merge/promotion/frontier decision is an event with a
deterministic id), and **projectable** (`to_debug()` feeds `eval/projection.py`, so the
slates and frontier are replayable, forkable, and learnable from traces — not a
sidecar that only lives in the Python loop). It is **skipped** for easy/direct/simple
epistemic modes (see `agent/epistemic_mode.py`). Nothing here is written to policy
memory; only the question's own constraints + result text are used.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from regimes_probe.agent.clue_resolution import (
    ROLES, _entities_in_field, _is_generic_entity, _norm, classify_entity_role)
from regimes_probe.agent.hypothesis_table import EvidenceRecord, _context_role, _host

CANDIDATE_STATUSES = ("active", "rejected", "confirmed", "merged", "stale")
FRONTIER_ACTION_TYPES = (
    "generate_candidates_for_slot", "verify_candidate_constraint",
    "expand_candidate_to_dependent_slot", "compare_candidates_for_slot",
    "read_candidate_source", "reject_candidate", "merge_duplicate_candidate",
    "promote_candidate_to_confirmed", "answer_from_confirmed_hypothesis",
    "abstain_no_viable_hypothesis")
#: epistemic modes that DO use the heavy slate layer (others skip it).
HEAVY_MODES = ("decomposed_search", "iterative_research", "task_frame_required")
_REJECT_NO_PROGRESS = 2
_CONFIRM_SUPPORT = 1                       # supported required constraints to confirm
_YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")

#: Candidate *texts* that are retrieval noise, not real entities (used to (a) not count
#: them as progress and (b) trigger LLM repair when a deterministic query leans on them).
_NOISE_DEFINITION = frozenset({
    "definition", "meaning", "dictionary", "thesaurus", "merriam", "webster", "wikipedia",
    "wiktionary", "synonym", "synonyms", "antonym", "encyclopedia", "encyclopaedia"})
_NOISE_PLATFORM = frozenset({
    "linkedin", "facebook", "twitter", "instagram", "youtube", "tiktok", "pinterest",
    "reddit", "quora", "tumblr", "medium", "substack", "github", "tripadvisor", "yelp"})
_NOISE_UI = frozenset({
    "login", "log", "signin", "sign", "signup", "register", "menu", "home", "search",
    "username", "generator", "table", "tables", "page", "click", "next", "previous",
    "settings", "profile", "account", "navigation", "cookie", "cookies", "subscribe"})


def _noise_kind(text: str) -> str:
    """Classify a candidate's *text* as a retrieval-noise kind, or '' if it looks real."""
    words = {w.lower() for w in re.findall(r"[A-Za-z]+", text or "")}
    if not words:
        return ""
    if words & _NOISE_DEFINITION:
        return "generic_definition_noise"
    if words & _NOISE_PLATFORM and len(words) <= 2:
        return "source_platform_noise"
    if words & _NOISE_UI and len(words) <= 2:
        return "ui_navigation_noise"
    return ""


def _role_compatible(entity_role: str, slot_role: str) -> bool:
    """Whether an extracted-entity role may bind a slot. Permissive for free-form slot
    roles (a person entity binds a `graphic_designer`/`author`/`founder` slot), but never
    forces an incompatible type (a date never binds a person slot)."""
    if entity_role == slot_role:
        return True
    if entity_role == "unknown" or slot_role == "unknown":
        return True
    # free-form / custom slot roles (not in the closed ROLES vocabulary) are person-like.
    if slot_role not in ROLES and entity_role == "person":
        return True
    return False


def _prev(s: str, n: int = 120) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n - 1] + "…"


def _hash(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- data
@dataclass
class SlotCandidate:
    candidate_id: str
    candidate_text: str
    normalized_text_hash: str
    inferred_role: str
    slot_id: str
    status: str = "active"
    status_reason: str = ""
    source_evidence_ids: list[str] = field(default_factory=list)
    source_action_ids: list[str] = field(default_factory=list)
    source_domains: list[str] = field(default_factory=list)
    constraints_supported: list[str] = field(default_factory=list)
    constraints_contradicted: list[str] = field(default_factory=list)
    constraints_unknown: list[str] = field(default_factory=list)
    evidence_score: float = 0.0
    role_match_score: float = 0.0
    source_authority_score: float = 0.0
    novelty_score: float = 1.0
    no_progress_count: int = 0
    duplicate_of: Optional[str] = None
    next_test_action_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id,
                "candidate_text_preview": _prev(self.candidate_text, 80),
                "normalized_text_hash": self.normalized_text_hash,
                "inferred_role": self.inferred_role, "slot_id": self.slot_id,
                "status": self.status, "status_reason": self.status_reason,
                "source_evidence_ids": list(self.source_evidence_ids)[:8],
                "source_action_ids": list(self.source_action_ids)[:8],
                "source_domains": list(self.source_domains)[:6],
                "constraints_supported": list(self.constraints_supported),
                "constraints_contradicted": list(self.constraints_contradicted),
                "constraints_unknown": list(self.constraints_unknown),
                "evidence_score": round(self.evidence_score, 3),
                "role_match_score": round(self.role_match_score, 3),
                "source_authority_score": round(self.source_authority_score, 3),
                "novelty_score": round(self.novelty_score, 3),
                "no_progress_count": self.no_progress_count,
                "duplicate_of": self.duplicate_of}


@dataclass
class CandidateSlate:
    slate_id: str
    slot_id: str
    slot_role: str
    slot_descriptor: str
    candidates: dict[str, SlotCandidate] = field(default_factory=dict)   # by candidate_id
    slate_confidence: float = 0.0
    unresolved_constraint_ids: list[str] = field(default_factory=list)
    next_frontier_action_ids: list[str] = field(default_factory=list)

    def _ids(self, status: str) -> list[str]:
        return [c.candidate_id for c in self.candidates.values() if c.status == status]

    def to_dict(self, *, top: int = 6) -> dict[str, Any]:
        ranked = sorted(self.candidates.values(),
                        key=lambda c: (-c.evidence_score, c.candidate_id))
        return {"slate_id": self.slate_id, "slot_id": self.slot_id,
                "slot_role": self.slot_role, "slot_descriptor": _prev(self.slot_descriptor, 80),
                "active_candidate_ids": self._ids("active"),
                "rejected_candidate_ids": self._ids("rejected"),
                "confirmed_candidate_ids": self._ids("confirmed"),
                "merged_candidate_ids": self._ids("merged"),
                "slate_confidence": round(self.slate_confidence, 3),
                "unresolved_constraint_ids": list(self.unresolved_constraint_ids),
                "next_frontier_action_ids": list(self.next_frontier_action_ids),
                "top_candidates": [c.to_dict() for c in ranked[:top]]}


@dataclass
class FrontierAction:
    action_id: str
    action_type: str
    target_slot_id: Optional[str] = None
    candidate_id: Optional[str] = None
    hypothesis_id: Optional[str] = None
    constraint_ids: list[str] = field(default_factory=list)
    expected_information_gain: float = 0.0
    estimated_cost: float = 1.0
    selected: bool = False
    selected_reason: str = ""
    rejected_reason: str = ""
    plan: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"action_id": self.action_id, "action_type": self.action_type,
                "target_slot_id": self.target_slot_id, "candidate_id": self.candidate_id,
                "hypothesis_id": self.hypothesis_id, "constraint_ids": list(self.constraint_ids),
                "expected_information_gain": round(self.expected_information_gain, 3),
                "estimated_cost": round(self.estimated_cost, 3),
                "selected": self.selected, "selected_reason": self.selected_reason,
                "rejected_reason": self.rejected_reason, "plan": dict(self.plan)}


@dataclass
class StepPlan:
    """A translated, executable (or not) frontier action for the controller loop."""
    frontier_action_id: str
    action_type: str
    kind: str                                # search | read | answer | abstain | unexecutable
    tool: Optional[str] = None
    query: str = ""
    query_arm: str = ""
    target_slot_id: Optional[str] = None
    candidate_id: Optional[str] = None
    constraint_ids: list[str] = field(default_factory=list)
    read_obs: Any = None
    read_decision: Any = None
    reason: str = ""
    is_discriminative_constraint: bool = False
    is_generic_query: bool = False
    # LLM-frontier-proposal provenance (Level 5c) — set when the plan came from a proposal.
    llm_frontier_proposal_id: Optional[str] = None
    anchors_used: list[str] = field(default_factory=list)
    expected_evidence: str = ""
    proposed_tool_family: str = ""

    _PLANNER_KIND = {
        "answer_from_confirmed_hypothesis": "answer_if_supported",
        "abstain_no_viable_hypothesis": "abstain_if_blocked",
        "read_candidate_source": "read_url_for_constraint",
        "generate_candidates_for_slot": "search_to_bind_slot",
        "verify_candidate_constraint": "search_to_test_constraint",
        "expand_candidate_to_dependent_slot": "search_to_bind_slot",
        "compare_candidates_for_slot": "compare_candidates"}

    @property
    def executable(self) -> bool:
        return self.kind in ("search", "read", "answer", "abstain")

    @property
    def planner_kind(self) -> str:
        return self._PLANNER_KIND.get(self.action_type, "search_to_bind_slot")

    def action_info(self) -> dict[str, Any]:
        info = {"action_id": self.frontier_action_id,
                "frontier_action_id": self.frontier_action_id,
                "kind": self.planner_kind, "frontier_action_type": self.action_type,
                "target_slot_id": self.target_slot_id, "candidate_id": self.candidate_id,
                "tested_constraint_ids": list(self.constraint_ids),
                "query_text_preview": self.query[:160], "query_arm": self.query_arm,
                "selected_reason": self.reason, "driven_by": "frontier_controller",
                "is_discriminative_constraint": self.is_discriminative_constraint,
                "is_generic_query": self.is_generic_query}
        if self.llm_frontier_proposal_id:
            info.update({"llm_frontier_proposal_id": self.llm_frontier_proposal_id,
                         "anchors_used": list(self.anchors_used),
                         "expected_evidence": self.expected_evidence[:160],
                         "proposed_tool_family": self.proposed_tool_family,
                         "driven_by": "llm_frontier_proposal"})
        return info


@dataclass
class FrontierHypothesis:
    hypothesis_id: str
    slot_candidate_assignments: dict[str, str] = field(default_factory=dict)
    unresolved_slots: list[str] = field(default_factory=list)
    constraints_supported: list[str] = field(default_factory=list)
    constraints_contradicted: list[str] = field(default_factory=list)
    constraints_unknown: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    support_score: float = 0.0
    contradiction_score: float = 0.0
    coverage_score: float = 0.0
    source_diversity_score: float = 0.0
    confidence_score: float = 0.0
    active: bool = True
    rejection_reason: Optional[str] = None
    last_frontier_action_id: Optional[str] = None

    def to_dict(self, frontier: "CandidateFrontier") -> dict[str, Any]:
        return {"hypothesis_id": self.hypothesis_id,
                "slot_candidate_assignments": {sid: frontier.candidate_text(cid)
                                               for sid, cid in self.slot_candidate_assignments.items()},
                "unresolved_slots": list(self.unresolved_slots),
                "constraints_supported": list(self.constraints_supported),
                "constraints_contradicted": list(self.constraints_contradicted),
                "support_score": round(self.support_score, 3),
                "contradiction_score": round(self.contradiction_score, 3),
                "coverage_score": round(self.coverage_score, 3),
                "source_diversity_score": round(self.source_diversity_score, 3),
                "confidence_score": round(self.confidence_score, 3),
                "active": self.active, "rejection_reason": self.rejection_reason,
                "last_frontier_action_id": self.last_frontier_action_id}


# --------------------------------------------------------------------------- core
class CandidateFrontier:
    """Per-slot candidate slates + frontier scheduler, event-sourced and projectable."""

    def __init__(self, frame, *, attempt_id: str = "", item_id: str = "",
                 run_id: str = "", clock: Optional[Callable[[], int]] = None) -> None:
        self.frame = frame
        self.attempt_id, self.item_id, self.run_id = attempt_id, item_id, run_id
        self._seq = 0
        self._clock = clock or (lambda: self._seq)
        self.slates: dict[str, CandidateSlate] = {}
        self.candidates_by_id: dict[str, SlotCandidate] = {}
        self.hypotheses: dict[str, FrontierHypothesis] = {}
        self.evidence: list[EvidenceRecord] = []
        self.frontier_actions: list[FrontierAction] = []
        self.events: list[dict[str, Any]] = []
        self._cc = self._ec = self._hc = self._ac = self._evc = 0
        self._known = {_norm(t) for t in frame.known_context_terms}
        # one slate per slot that NEEDS binding (every unbound variable).
        for s in frame.all_slots:
            if getattr(s, "slot_status", "unbound_variable") == "known_constant":
                continue
            self._sc = len(self.slates)
            slate = CandidateSlate(
                slate_id=f"slate_{s.slot_id}", slot_id=s.slot_id, slot_role=s.slot_role,
                slot_descriptor=getattr(s, "descriptor_text", "") or s.slot_name,
                unresolved_constraint_ids=[c.constraint_id for c in frame.constraints
                                           if s.slot_id in c.applies_to])
            self.slates[s.slot_id] = slate
            self._emit("candidate_slate.created", slate_id=slate.slate_id,
                       slot_id=s.slot_id, data={"slot_role": s.slot_role})

    # ---------- events ----------
    def _emit(self, event_type: str, **payload) -> dict[str, Any]:
        self._seq += 1
        self._evc += 1
        ev = {"event_id": f"fe{self._evc}", "event_type": event_type,
              "seq": self._clock(), "run_id": self.run_id, "attempt_id": self.attempt_id,
              "item_id": self.item_id}
        ev.update({k: v for k, v in payload.items() if v is not None})
        self.events.append(ev)
        return ev

    # ---------- lookups ----------
    def candidate_text(self, cid: str) -> str:
        c = self.candidates_by_id.get(cid)
        return c.candidate_text if c else cid

    def _slots_for_role(self, role: str) -> list:
        return [s for s in self.frame.all_slots
                if s.slot_id in self.slates and (s.slot_role == role or role == "unknown"
                                                 or s.slot_role == "unknown")]

    def _constraints_for_slot(self, slot_id: str) -> list:
        return [c for c in self.frame.constraints if slot_id in c.applies_to]

    def _dependents(self, slot_id: str) -> list[str]:
        return [s.slot_id for s in self.frame.all_slots if slot_id in getattr(s, "depends_on", [])]

    # ---------- ingest ----------
    def ingest_evidence(self, observations, *, source_tool: str, stage: int = 1,
                        read_depth: int = 0, action_id: Optional[str] = None,
                        directed_slot_id: Optional[str] = None,
                        directed_constraint_ids: Optional[list[str]] = None,
                        proposal_id: Optional[str] = None) -> EvidenceRecord:
        """Fold a tool call's observations into the slates.

        When the call was driven by an LLM frontier proposal, ``directed_slot_id`` /
        ``directed_constraint_ids`` / ``proposal_id`` *direct* the linkage: a
        role-compatible candidate is bound to the SELECTED slot (even when its role label
        differs from the strict role index) and the SELECTED constraints are evaluated +
        linked, so evidence ties to the slot/constraint the proposal actually searched
        for — not a stale/default one. Returns the record with ``progress_components``."""
        self._ec += 1
        ev = EvidenceRecord(evidence_id=f"e{self._ec}", source_tool=source_tool,
                            read_depth=read_depth)
        directed_constraint_ids = list(directed_constraint_ids or [])
        dslot = self.frame.slot(directed_slot_id) if directed_slot_id else None
        before_best = self.best_hypothesis()
        before_conf = before_best.confidence_score if before_best else 0.0
        before_sel_support = self._slot_constraint_support(directed_slot_id, directed_constraint_ids)
        noise_count = 0
        for o in observations:
            if getattr(o, "failed", False):
                continue
            title = getattr(o, "title", "") or ""
            snippet = getattr(o, "snippet", "") or ""
            url = getattr(o, "url", "") or ""
            authority = float(getattr(o, "source_authority", 0.0))
            contaminated = bool(getattr(o, "benchmark_contaminated", False))
            ev.url = ev.url or url
            ev.domain = ev.domain or _host(url)
            ev.title_preview = ev.title_preview or title
            ev.snippet_preview = ev.snippet_preview or snippet
            ev.source_authority_score = max(ev.source_authority_score, authority)
            if contaminated:
                ev.contamination_score = 1.0
            text_l = f"{title} {snippet}".lower()
            for e in _entities_in_field(title) + _entities_in_field(snippet):
                if _is_generic_entity(e) or len(e) < 3:
                    continue
                if _noise_kind(e):
                    noise_count += 1
                    continue                # retrieval noise is never a candidate
                norm = _norm(e)
                is_known = norm in self._known
                role, from_ctx = _context_role(e, title, snippet)
                # A KNOWN CONSTANT keeps its ISOLATED role (context must not flip a
                # given location into an organization candidate) and is only admitted
                # to a role-compatible slot whose constraint the evidence supports.
                if is_known:
                    role, from_ctx = classify_entity_role(e), False
                target_slots = list(self._slots_for_role(role))
                # DIRECTED: also admit the candidate to the proposal's slot when role
                # compatible, so the slot the LLM searched for actually gets the binding.
                if (dslot is not None and dslot.slot_id in self.slates
                        and dslot not in target_slots
                        and _role_compatible(role, dslot.slot_role)):
                    target_slots.append(dslot)
                for slot in target_slots:
                    cons = self._constraints_for_slot(slot.slot_id)
                    if slot.slot_id == directed_slot_id:
                        cons = cons + [c for c in (self._con(cid) for cid in directed_constraint_ids)
                                       if c is not None and c not in cons]
                    supported = [c for c in cons if _supports(c, text_l)
                                 and any(t in norm for t in c.normalized_terms if len(t) >= 4)]
                    if is_known and not supported:
                        continue            # not bound to this slot -> not a candidate
                    self._assign(slot, e, norm, role, ev, source_tool, action_id,
                                 stage, _host(url), authority, contaminated, from_ctx,
                                 text_l, cons, directed_slot_id=directed_slot_id,
                                 directed_constraint_ids=directed_constraint_ids,
                                 proposal_id=proposal_id)
            for hint in self.frame.answer_shape_hints:
                if hint.lower() in text_l and hint not in ev.answer_shape_hints_found:
                    ev.answer_shape_hints_found.append(hint)
        ev.evidence_progress_score = float(len(ev.newly_introduced_candidates)
                                           + len(ev.supports_constraint_ids))
        self.evidence.append(ev)
        self._advance_hypotheses(ev, action_id)
        self.generate_frontier_actions()
        # progress components (honest, slot/constraint-aware — req: don't count noise).
        after_best = self.best_hypothesis()
        after_conf = after_best.confidence_score if after_best else 0.0
        after_sel_support = self._slot_constraint_support(directed_slot_id, directed_constraint_ids)
        slot_compat = sum(
            1 for cid in ev.newly_introduced_candidates
            if (c := self.candidates_by_id.get(cid)) is not None
            and _role_compatible(c.inferred_role,
                                 self.slates[c.slot_id].slot_role if c.slot_id in self.slates else "unknown"))
        sel_slot_new = sum(1 for cid in ev.newly_introduced_candidates
                           if (c := self.candidates_by_id.get(cid)) is not None
                           and c.slot_id == directed_slot_id)
        ev.progress_components = {
            "raw_candidate_count": len(ev.newly_introduced_candidates),
            "slot_compatible_candidate_count": slot_compat,
            "selected_slot_candidate_count": sel_slot_new,
            "selected_constraint_support_count": max(0, after_sel_support - before_sel_support),
            "hypothesis_score_delta": round(after_conf - before_conf, 3),
            "target_support_path_delta": round(after_conf - before_conf, 3),
            "noise_candidate_count": noise_count,
        }
        return ev

    def _slot_constraint_support(self, slot_id: Optional[str],
                                 constraint_ids: list[str]) -> int:
        """Count how many of ``constraint_ids`` are currently supported by some active/
        confirmed candidate on ``slot_id`` (for selected-constraint progress deltas)."""
        if not slot_id or slot_id not in self.slates:
            return 0
        cset = set(constraint_ids)
        supported: set[str] = set()
        for c in self.slates[slot_id].candidates.values():
            if c.status in ("rejected", "merged"):
                continue
            supported.update(set(c.constraints_supported) & cset)
        return len(supported)

    def _assign(self, slot, text, norm, role, ev, source_tool, action_id, stage,
                domain, authority, contaminated, from_ctx, text_l, cons,
                directed_slot_id=None, directed_constraint_ids=None,
                proposal_id=None) -> None:
        slate = self.slates[slot.slot_id]
        existing = next((c for c in slate.candidates.values()
                         if c.normalized_text_hash == _hash(norm)), None)
        cand_l = text.lower()
        supported = [c.constraint_id for c in cons if _supports(c, text_l)]
        contradicted = [c.constraint_id for c in cons if _contradicts(c, text_l, cand_l)]
        if existing is None:
            self._cc += 1
            cand = SlotCandidate(
                candidate_id=f"cand{self._cc}", candidate_text=text,
                normalized_text_hash=_hash(norm), inferred_role=role, slot_id=slot.slot_id,
                role_match_score=1.0 if role == slot.slot_role else 0.5,
                source_authority_score=authority,
                novelty_score=0.75 if from_ctx else 1.0)
            slate.candidates[cand.candidate_id] = cand
            self.candidates_by_id[cand.candidate_id] = cand
            ev.newly_introduced_candidates.append(cand.candidate_id)
            self._emit("candidate.extracted", candidate_id=cand.candidate_id,
                       slot_id=slot.slot_id, evidence_id=ev.evidence_id, action_id=action_id,
                       data={"text_preview": _prev(text, 60), "role": role})
            self._emit("candidate.assigned_to_slot", candidate_id=cand.candidate_id,
                       slot_id=slot.slot_id, evidence_id=ev.evidence_id)
        else:
            cand = existing
            # same normalized text arriving again = a duplicate; merge provenance.
            if domain and domain not in cand.source_domains:
                self._emit("candidate.merged", candidate_id=cand.candidate_id,
                           slot_id=slot.slot_id, evidence_id=ev.evidence_id,
                           data={"merged_provenance_domain": domain})
        # provenance + constraint support/contradiction (idempotent unions).
        _union(cand.source_evidence_ids, ev.evidence_id)
        if action_id:
            _union(cand.source_action_ids, action_id)
        if domain:
            _union(cand.source_domains, domain)
        cand.source_authority_score = max(cand.source_authority_score, authority)
        for cid in supported:
            _union(cand.constraints_supported, cid)
            _union(ev.supports_constraint_ids, cid)
            self._emit("evidence.linked_to_constraint", candidate_id=cand.candidate_id,
                       evidence_id=ev.evidence_id, constraint_id=cid)
        for cid in contradicted:
            _union(cand.constraints_contradicted, cid)
        cand.constraints_unknown = [c.constraint_id for c in cons
                                    if c.constraint_id not in cand.constraints_supported
                                    and c.constraint_id not in cand.constraints_contradicted]
        cand.evidence_score = float(len(cand.constraints_supported)) + 0.25 * len(cand.source_domains)
        _union(ev.supports_candidate_ids, cand.candidate_id)
        _union(ev.supports_slot_ids, slot.slot_id)
        self._emit("evidence.linked_to_candidate", candidate_id=cand.candidate_id,
                   evidence_id=ev.evidence_id, slot_id=slot.slot_id)
        # DIRECTED: record that this evidence/candidate ties back to the LLM proposal's
        # selected slot + constraints (auditable proposal -> evidence link).
        if proposal_id and slot.slot_id == directed_slot_id:
            self._emit("evidence_linked_to_llm_proposal", candidate_id=cand.candidate_id,
                       evidence_id=ev.evidence_id, slot_id=slot.slot_id,
                       action_id=action_id,
                       data={"proposal_id": proposal_id,
                             "constraint_ids": list(directed_constraint_ids or []),
                             "supported_constraint_ids": list(supported)})
        was_confirmed = cand.status == "confirmed"
        self._update_status(cand, contaminated)
        if (proposal_id and slot.slot_id == directed_slot_id
                and cand.status == "confirmed" and not was_confirmed):
            self._emit("candidate_promoted_from_llm_frontier_evidence",
                       candidate_id=cand.candidate_id, slot_id=slot.slot_id,
                       data={"proposal_id": proposal_id})

    def _update_status(self, cand: SlotCandidate, contaminated: bool) -> None:
        old = cand.status
        if cand.status in ("merged", "rejected"):
            return
        reason = ""
        new = cand.status
        if cand.constraints_contradicted:
            new, reason = "rejected", "contradicted_constraint"
        elif cand.no_progress_count >= _REJECT_NO_PROGRESS:
            new, reason = "rejected", "repeated_no_progress"
        elif self._confirmable(cand, contaminated):
            new, reason = "confirmed", "required_constraints_supported"
        if new != old:
            cand.status, cand.status_reason = new, reason
            if new == "rejected":
                self._emit("candidate.rejected", candidate_id=cand.candidate_id,
                           slot_id=cand.slot_id, data={"reason": reason})
            elif new == "confirmed":
                self._emit("candidate.promoted", candidate_id=cand.candidate_id,
                           slot_id=cand.slot_id, data={"reason": reason})
            else:
                self._emit("candidate.status_changed", candidate_id=cand.candidate_id,
                           slot_id=cand.slot_id, data={"status": new, "reason": reason})

    def _confirmable(self, cand: SlotCandidate, contaminated: bool) -> bool:
        if contaminated or cand.constraints_contradicted:
            return False
        required = [c for c in self._constraints_for_slot(cand.slot_id)
                    if getattr(c, "required", False) or getattr(c, "priority", "") == "high"]
        if required:
            if not all(c.constraint_id in cand.constraints_supported for c in required):
                return False
        return len(cand.constraints_supported) >= _CONFIRM_SUPPORT

    def note_no_progress_for_slate(self, slot_id: str) -> None:
        slate = self.slates.get(slot_id)
        if not slate:
            return
        for c in slate.candidates.values():
            if c.status == "active":
                c.no_progress_count += 1
                self._update_status(c, contaminated=False)

    # ---------- hypotheses ----------
    def _advance_hypotheses(self, ev: EvidenceRecord, action_id: Optional[str]) -> None:
        targets = [s.slot_id for s in self.frame.target_answer_slots if s.slot_id in self.slates]
        if not targets:
            return
        tgt = targets[0]
        for cid in list(self.slates.get(tgt, CandidateSlate("", "", "", "")).candidates):
            cand = self.candidates_by_id.get(cid)
            if cand is None or cand.status in ("rejected", "merged"):
                continue
            hyp = self._hyp_for(cid, tgt)
            _union(hyp.evidence_ids, ev.evidence_id)
            hyp.last_frontier_action_id = action_id
            for s in self.frame.all_slots:
                if s.slot_id == tgt or s.slot_id not in self.slates:
                    continue
                best = self._best_candidate(s.slot_id)
                if best is not None:
                    hyp.slot_candidate_assignments[s.slot_id] = best.candidate_id
            self._recompute_hypothesis(hyp)
            self._emit("hypothesis.updated", hypothesis_id=hyp.hypothesis_id, action_id=action_id)

    def _hyp_for(self, cid: str, tgt: str) -> FrontierHypothesis:
        for h in self.hypotheses.values():
            if h.slot_candidate_assignments.get(tgt) == cid:
                return h
        self._hc += 1
        h = FrontierHypothesis(hypothesis_id=f"h{self._hc}",
                               slot_candidate_assignments={tgt: cid})
        self.hypotheses[h.hypothesis_id] = h
        self._emit("hypothesis.created", hypothesis_id=h.hypothesis_id)
        return h

    def _best_candidate(self, slot_id: str) -> Optional[SlotCandidate]:
        active = [c for c in self.slates[slot_id].candidates.values()
                  if c.status in ("active", "confirmed")]
        return max(active, key=lambda c: (c.evidence_score, c.candidate_id), default=None)

    def _recompute_hypothesis(self, hyp: FrontierHypothesis) -> None:
        supported, contradicted, domains = set(), set(), set()
        for sid, cid in hyp.slot_candidate_assignments.items():
            c = self.candidates_by_id.get(cid)
            if not c:
                continue
            supported.update(c.constraints_supported)
            contradicted.update(c.constraints_contradicted)
            domains.update(c.source_domains)
        hyp.constraints_supported = sorted(supported)
        hyp.constraints_contradicted = sorted(contradicted)
        n_slots = len(self.slates) or 1
        hyp.unresolved_slots = [sid for sid in self.slates
                                if sid not in hyp.slot_candidate_assignments]
        hyp.support_score = float(len(supported))
        hyp.contradiction_score = float(len(contradicted))
        hyp.coverage_score = len(hyp.slot_candidate_assignments) / n_slots
        hyp.source_diversity_score = float(len(domains))
        no_progress = sum(self.candidates_by_id[cid].no_progress_count
                          for cid in hyp.slot_candidate_assignments.values()
                          if cid in self.candidates_by_id)
        hyp.confidence_score = (hyp.support_score - 1.5 * hyp.contradiction_score
                                + 0.5 * hyp.coverage_score + 0.25 * hyp.source_diversity_score
                                - 0.5 * no_progress)
        if hyp.contradiction_score > 0 and hyp.support_score == 0 and hyp.active:
            hyp.active, hyp.rejection_reason = False, "contradicted_by_evidence"
            self._emit("hypothesis.rejected", hypothesis_id=hyp.hypothesis_id,
                       data={"reason": hyp.rejection_reason})

    def best_hypothesis(self) -> Optional[FrontierHypothesis]:
        active = [h for h in self.hypotheses.values() if h.active]
        return max(active, key=lambda h: (h.confidence_score, h.support_score,
                                          h.coverage_score), default=None)

    # ---------- frontier ----------
    def generate_frontier_actions(self) -> list[FrontierAction]:
        actions: list[FrontierAction] = []

        def _add(atype, *, slot=None, cand=None, cons=None, eig, cost=1.0, reason="") -> FrontierAction:
            self._ac += 1
            a = FrontierAction(action_id=f"fa{self._ac}", action_type=atype,
                               target_slot_id=slot, candidate_id=cand,
                               constraint_ids=list(cons or []),
                               expected_information_gain=eig, estimated_cost=cost,
                               selected_reason=reason)
            actions.append(a)
            self._emit("frontier_action.generated", action_id=a.action_id,
                       data={"action_type": atype, "target_slot_id": slot, "eig": round(eig, 3)})
            return a

        for slate in self.slates.values():
            blocking = [c for c in self._constraints_for_slot(slate.slot_id)
                        if c.status != "resolved"
                        and (getattr(c, "blocks_answer_if_unresolved", False)
                             or getattr(c, "priority", "") == "high"
                             or "can_block_answer" in getattr(c, "affordances", []))]
            blocking.sort(key=lambda c: -_disc(c))    # most discriminative first
            active = [c for c in slate.candidates.values() if c.status == "active"]
            confirmed = [c for c in slate.candidates.values() if c.status == "confirmed"]
            if blocking:
                top = blocking[0]
                # Prefer DISCRIMINATIVE blocking constraints over generic answer-type
                # ones, so the controller does not open with a generic target query.
                disc_bonus = (0.6 if _is_discriminative_constraint(top)
                              else (-0.6 if _is_generic_answer_constraint(top) else 0.0))
                cons_ids = [c.constraint_id for c in blocking]
                if active:
                    _add("verify_candidate_constraint", slot=slate.slot_id,
                         cand=active[0].candidate_id, cons=cons_ids,
                         eig=3.0 + 0.75 * _disc(top) + disc_bonus, cost=1.0,
                         reason="resolve_blocking_constraint")
                else:
                    _add("generate_candidates_for_slot", slot=slate.slot_id, cons=cons_ids,
                         eig=2.5 + 0.75 * _disc(top) + disc_bonus, cost=1.0,
                         reason="bind_slot_via_discriminative_constraint")
            elif not slate.candidates:
                _add("generate_candidates_for_slot", slot=slate.slot_id, eig=2.0, cost=1.0,
                     reason="slot_unbound")
            if len(active) >= 2:
                _add("compare_candidates_for_slot", slot=slate.slot_id, eig=1.5, cost=1.0,
                     reason="distinguish_competing_candidates")
            # reads are expensive — cheap verification is preferred (lower net score).
            for c in active:
                if c.source_domains and c.constraints_unknown:
                    _add("read_candidate_source", slot=slate.slot_id, cand=c.candidate_id,
                         cons=list(c.constraints_unknown), eig=1.0, cost=2.0,
                         reason="deepen_candidate_evidence")
            for c in confirmed:
                for dep in self._dependents(slate.slot_id):
                    if dep in self.slates and not self._best_candidate(dep):
                        _add("expand_candidate_to_dependent_slot", slot=dep,
                             cand=c.candidate_id, eig=2.0, cost=1.0,
                             reason="upstream_confirmed_unlocks_dependent")
        best = self.best_hypothesis()
        if best is not None and self._answerable(best):
            _add("answer_from_confirmed_hypothesis", eig=5.0, cost=0.0,
                 reason="confirmed_supported_hypothesis")
        self.frontier_actions = actions
        for slate in self.slates.values():
            slate.next_frontier_action_ids = [a.action_id for a in actions
                                              if a.target_slot_id == slate.slot_id]
        return actions

    def _answerable(self, hyp: FrontierHypothesis) -> bool:
        targets = [s.slot_id for s in self.frame.target_answer_slots if s.slot_id in self.slates]
        if not targets:
            return False
        tgt = targets[0]
        tcand = hyp.slot_candidate_assignments.get(tgt)
        c = self.candidates_by_id.get(tcand) if tcand else None
        return bool(c and c.status == "confirmed" and not hyp.constraints_contradicted)

    def select_frontier_action(self, *, budget_remaining: int = 99,
                               reading_available: bool = True) -> Optional[FrontierAction]:
        cands = list(self.frontier_actions) or self.generate_frontier_actions()
        if not cands:
            return None
        # expected information gain net of cost; reads cost more (verify before read).
        def _score(a: FrontierAction) -> float:
            return a.expected_information_gain - 0.5 * a.estimated_cost
        best = max(cands, key=lambda a: (_score(a), -self.frontier_actions.index(a)))
        for a in cands:
            if a is best:
                a.selected, a.selected_reason = True, a.selected_reason or "max_expected_information_gain"
                self._emit("frontier_action.selected", action_id=a.action_id,
                           data={"action_type": a.action_type,
                                 "selected_reason": a.selected_reason})
                self._emit("frontier_action.scored", action_id=a.action_id,
                           data={"score": round(_score(a), 3)})
            else:
                a.rejected_reason = a.rejected_reason or "lower_expected_information_gain"
        return best

    # ---------- read-value gate (candidate-slot-constraint triple) ----------
    def frontier_read_value(self, obs) -> FrontierAction:
        self._ac += 1
        a = FrontierAction(action_id=f"fa{self._ac}", action_type="read_candidate_source",
                           estimated_cost=2.0, plan={"url": getattr(obs, "url", "")})
        if getattr(obs, "benchmark_contaminated", False):
            a.rejected_reason = "contaminated_source"
            return a
        text_l = f"{getattr(obs, 'title', '') or ''} {getattr(obs, 'snippet', '') or ''}".lower()
        # find a (candidate, slot, constraint) the page could affect.
        for slate in self.slates.values():
            cons = [c for c in self._constraints_for_slot(slate.slot_id) if c.status != "resolved"]
            for cand in slate.candidates.values():
                if cand.status in ("rejected", "merged"):
                    continue
                if cand.candidate_text.lower() in text_l:
                    tested = [c.constraint_id for c in cons
                              if any(t in text_l for t in c.normalized_terms if len(t) >= 4)]
                    if tested or cons:
                        a.candidate_id = cand.candidate_id
                        a.target_slot_id = slate.slot_id
                        a.constraint_ids = tested or [c.constraint_id for c in cons]
                        a.selected = True
                        a.selected_reason = "page_affects_candidate_slot_constraint"
                        return a
        a.rejected_reason = "no_candidate_slot_constraint_affected"
        return a

    # ---------- controller (optional: frontier drives tool selection) ----------
    def propose_step_action(self, *, observations=(), budget_remaining: int = 99,
                            reading_tools: bool = False, scraped_urls=frozenset(),
                            no_progress_domains=frozenset(), page_fetch_available: bool = False,
                            scrape_available: bool = False, allow_social: bool = False,
                            force_page_fetch: bool = False) -> "StepPlan":
        """Pick + TRANSLATE the next frontier action into an executable step plan.

        Search-like actions build a query from the slot's most DISCRIMINATIVE blocking
        constraint (never a bare repeat of the target descriptor); a read resolves a
        candidate+slot+constraint URL from the observations; answer/abstain are
        terminal. If a selected action cannot be executed it returns kind
        ``unexecutable`` so the loop falls back to the old planner."""
        self.generate_frontier_actions()
        sel = self.select_frontier_action(budget_remaining=budget_remaining,
                                          reading_available=reading_tools)
        if sel is None:
            return StepPlan("", "", "unexecutable", reason="no_frontier_action")
        at = sel.action_type
        if at == "answer_from_confirmed_hypothesis":
            return StepPlan(sel.action_id, at, "answer", reason=sel.selected_reason)
        if at == "abstain_no_viable_hypothesis":
            return StepPlan(sel.action_id, at, "abstain", reason=sel.selected_reason)
        if at == "read_candidate_source":
            if not reading_tools:
                return StepPlan(sel.action_id, at, "unexecutable", reason="no_reading_tool")
            chosen = self._resolve_read(observations, scraped_urls, no_progress_domains,
                                        page_fetch_available, scrape_available, allow_social,
                                        force_page_fetch)
            if chosen is None:
                return StepPlan(sel.action_id, at, "unexecutable",
                                reason="no_candidate_slot_constraint_url")
            o, rd, tested = chosen
            return StepPlan(sel.action_id, at, "read", read_obs=o, read_decision=rd,
                            target_slot_id=sel.target_slot_id, candidate_id=sel.candidate_id,
                            constraint_ids=tested, reason="read_candidate_source")
        query, arm, disc, generic = self._build_query(sel)
        if not query:
            return StepPlan(sel.action_id, at, "unexecutable", reason="empty_query")
        return StepPlan(sel.action_id, at, "search", query=query, query_arm=arm,
                        target_slot_id=sel.target_slot_id, candidate_id=sel.candidate_id,
                        constraint_ids=list(sel.constraint_ids), reason=sel.selected_reason,
                        is_discriminative_constraint=disc, is_generic_query=generic)

    def _resolve_read(self, observations, scraped_urls, no_progress_domains,
                      page_fetch_available, scrape_available, allow_social, force_page_fetch):
        from urllib.parse import urlparse as _up
        from regimes_probe.agent.reading_policy import normalize_url, select_reading_tool
        for o in observations:
            if getattr(o, "failed", False) or not getattr(o, "url", ""):
                continue
            host = (_up(o.url).hostname or "").lower()
            if normalize_url(o.url) in scraped_urls or host in no_progress_domains:
                continue
            rv = self.frontier_read_value(o)
            if not rv.selected:
                continue
            rd = select_reading_tool(
                url=o.url, title=getattr(o, "title", ""), snippet=getattr(o, "snippet", ""),
                source_authority=float(getattr(o, "source_authority", 0.0)),
                contaminated=False, unresolved_clue_terms=[], answer_shape=[],
                cross_provider_domains=set(), page_fetch_available=page_fetch_available,
                scrape_available=scrape_available, scraped_urls=scraped_urls,
                no_progress_domains=no_progress_domains, allow_social=allow_social,
                prefer_page_fetch=force_page_fetch, force_read=True)
            if rd.tool:
                return o, rd, list(rv.constraint_ids)
        return None

    def _head_noun(self, descriptor: str, slot) -> str:
        from regimes_probe.agent.task_frame import _ROLE_TRIGGERS
        toks = re.findall(r"[A-Za-z][A-Za-z'&]+", descriptor or "")
        role = getattr(slot, "slot_role", "") if slot else ""
        for t in toks:                                 # a role-type noun matching the slot
            if _ROLE_TRIGGERS.get(t.lower()) == role:
                return t
        for t in reversed(toks):                       # else the last content word (the type)
            if _ROLE_TRIGGERS.get(t.lower()):
                return t
        return toks[-1] if toks else ""

    def _build_query(self, action) -> tuple[str, str, bool, bool]:
        from regimes_probe.agent.action_planner import _distinctive_phrase
        from regimes_probe.policy.query_decomposition import _cap, _is_rare
        slot = self.frame.slot(action.target_slot_id) if action.target_slot_id else None
        descriptor = (getattr(slot, "descriptor_text", "") or getattr(slot, "slot_name", "")) if slot else ""
        cons = [self._con(cid) for cid in action.constraint_ids]
        cons = [c for c in cons if c is not None]
        cons.sort(key=lambda c: -_disc(c))
        top = cons[0] if cons else None
        phrase = _distinctive_phrase(top) if top is not None else ""
        is_disc = bool(top is not None and _is_discriminative_constraint(top))
        cand = self.candidate_text(action.candidate_id) if action.candidate_id else ""
        head = self._head_noun(descriptor, slot)
        parts: list[str] = []
        if cand:
            parts.append(f'"{cand}"')
        if phrase:
            parts.append(phrase)
        words = set(re.findall(r"[a-z0-9]+", " ".join(parts).lower()))
        # add the constraint's distinctive (rare/proper) terms not already present.
        if top is not None:
            for t in top.normalized_terms:
                if len(t) >= 4 and t.lower() not in words and (_is_rare(t) or t[:1].isupper()):
                    parts.append(t)
                    words.add(t.lower())
        if head and head.lower() not in words:
            parts.append(head)
        generic = not phrase                            # no discriminative clue used
        if not parts:                                   # last resort: the descriptor once
            parts.append(descriptor)
            generic = True
        return _cap(" ".join(p for p in parts if p).strip()), \
            ("candidate_constraint" if cand else "frontier_search"), is_disc, generic

    def _con(self, cid: str):
        return next((c for c in self.frame.constraints if c.constraint_id == cid), None)

    def register_proposal_action(self, *, proposal_id: str, action_type: str,
                                 target_slot_id: Optional[str], candidate_id: Optional[str],
                                 hypothesis_id: Optional[str], constraint_ids: list[str],
                                 query: str = "", tool: str = "", tool_family: str = "",
                                 anchors_used: Optional[list[str]] = None) -> FrontierAction:
        """Materialize a selected LLM proposal as a first-class FrontierAction so the
        executed tool call carries the PROPOSAL's slot/constraints (not a stale default).
        Action id is ``lfp_<proposal_id>`` so the tool call links straight to its proposal.
        """
        self._ac += 1
        a = FrontierAction(
            action_id=f"lfp_{proposal_id}", action_type=action_type,
            target_slot_id=target_slot_id, candidate_id=candidate_id,
            hypothesis_id=hypothesis_id, constraint_ids=list(constraint_ids),
            selected=True, selected_reason="llm_frontier_proposal",
            plan={"proposal_id": proposal_id, "query_preview": _prev(query, 120),
                  "proposed_tool": tool, "proposed_tool_family": tool_family,
                  "anchors_used": list(anchors_used or [])})
        self.frontier_actions = [fa for fa in self.frontier_actions
                                 if fa.action_id != a.action_id] + [a]
        self._emit("selected_proposal_translated_to_action", action_id=a.action_id,
                   data={"proposal_id": proposal_id, "action_type": action_type,
                         "target_slot_id": target_slot_id,
                         "constraint_ids": list(constraint_ids)})
        return a

    def record_execution(self, action_id: str, *, success: bool, kind: str = "",
                         evidence_progress: float = 0.0) -> None:
        for a in self.frontier_actions:
            if a.action_id == action_id:
                a.plan["executed"] = True
                a.plan["execution_success"] = success
        self._emit("frontier_action.executed", action_id=action_id,
                   data={"success": success, "kind": kind,
                         "evidence_progress": round(float(evidence_progress), 3)})

    def record_unexecutable(self, action_id: str, *, reason: str) -> None:
        self._emit("frontier_action_unexecutable", action_id=action_id,
                   data={"reason": reason})

    # ---------- merge / promote (explicit) ----------
    def merge_candidates(self, keep_id: str, dup_id: str, *, reason: str = "duplicate") -> None:
        keep, dup = self.candidates_by_id.get(keep_id), self.candidates_by_id.get(dup_id)
        if not keep or not dup:
            return
        for d in dup.source_domains:
            _union(keep.source_domains, d)
        for e in dup.source_evidence_ids:
            _union(keep.source_evidence_ids, e)
        dup.status, dup.status_reason, dup.duplicate_of = "merged", reason, keep_id
        self._emit("candidate.merged", candidate_id=dup_id, slot_id=dup.slot_id,
                   data={"merged_into": keep_id, "reason": reason})

    # ---------- projection / debug / metrics ----------
    def to_debug(self, *, top: int = 4) -> dict[str, Any]:
        ranked = sorted(self.hypotheses.values(),
                        key=lambda h: (-h.confidence_score, h.hypothesis_id))
        return {
            "skipped": False,
            "n_slates": len(self.slates),
            "slates": [s.to_dict() for s in self.slates.values()],
            "top_hypotheses": [h.to_dict(self) for h in ranked[:top]],
            "frontier_actions": [a.to_dict() for a in self.frontier_actions][:12],
            "selected_frontier_action": next(
                (a.to_dict() for a in self.frontier_actions if a.selected), {}),
            "events": list(self.events),
            "metrics": self.metrics(),
        }

    def metrics(self) -> dict[str, Any]:
        from collections import Counter
        n_slots = len(self.slates) or 1
        active = sum(len(s._ids("active")) for s in self.slates.values())
        confirmed = sum(len(s._ids("confirmed")) for s in self.slates.values())
        rejected = sum(len(s._ids("rejected")) for s in self.slates.values())
        merged = sum(len(s._ids("merged")) for s in self.slates.values())
        total = len(self.candidates_by_id) or 1
        atype = Counter(a.action_type for a in self.frontier_actions)
        return {
            "candidate_slate_size_mean": len(self.candidates_by_id) / n_slots,
            "active_candidates_per_slot": active / n_slots,
            "confirmed_candidates_per_slot": confirmed / n_slots,
            "rejected_candidates_per_slot": rejected / n_slots,
            "candidate_promotion_rate": confirmed / total,
            "candidate_rejection_rate": rejected / total,
            "candidate_merge_rate": merged / total,
            "hypothesis_branching_factor": float(len(self.hypotheses)),
            "frontier_action_counts": dict(atype),
            "frontier_expected_gain_mean": (
                sum(a.expected_information_gain for a in self.frontier_actions)
                / len(self.frontier_actions)) if self.frontier_actions else 0.0,
            "n_candidate_events": len(self.events),
        }


# --------------------------------------------------------------------------- helpers
def _union(lst: list, v) -> None:
    if v and v not in lst:
        lst.append(v)


def _disc(con) -> float:
    return float(getattr(con, "discriminative_score", 0.0)
                 or getattr(con, "specificity_score", 0.0) or 0.0)


def _is_discriminative_constraint(con) -> bool:
    """A specific, distinguishing constraint — not a generic answer-type/scope clue."""
    return (getattr(con, "constraint_type", "") != "answer_shape" and _disc(con) >= 1.0)


def _is_generic_answer_constraint(con) -> bool:
    return (getattr(con, "constraint_type", "") == "answer_shape" or _disc(con) < 1.0)


def _supports(con, text_l: str) -> bool:
    terms = [t for t in con.normalized_terms if len(t) >= 4]
    return bool(terms and sum(1 for t in terms if t in text_l) >= max(2, (len(terms) + 1) // 2))


def _contradicts(con, text_l: str, cand_l: str) -> bool:
    """Generic contradiction: the constraint anchors a specific year and the evidence
    about THIS candidate carries a different year for the same attribute."""
    cyears = set(_YEAR.findall(con.text_span + " " + " ".join(con.normalized_terms)))
    if not cyears:
        return False
    tyears = set(_YEAR.findall(text_l))
    if not tyears:
        return False
    attr = [t for t in con.normalized_terms if t.isalpha() and len(t) >= 4]
    if cand_l and cand_l in text_l and any(a in text_l for a in attr):
        return not (cyears & tyears)
    return False
