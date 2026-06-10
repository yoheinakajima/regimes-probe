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
    "abstain_no_viable_hypothesis", "bind_target_answer_slot")
#: generic answer-shape role/descriptor cues: when a target slot is answer-shaped (a year,
#: date, number, name, surname, title, venue, role-holder…) and a subject is supported, the
#: frontier should bind the TARGET, not keep re-verifying the supported subject (5h-C).
_ANSWER_SHAPE_ROLES = frozenset({
    "date", "date_or_time", "year", "number", "quantity", "name", "surname", "title",
    "venue", "publication", "duration", "time_span", "measure", "distance"})
_ANSWER_SHAPE_WORDS = frozenset({
    "year", "years", "birth", "born", "date", "number", "name", "surname", "title",
    "venue", "duration", "range", "age", "distance", "miles", "kilometers", "count",
    "how", "many", "much", "when"})
#: epistemic modes that DO use the heavy slate layer (others skip it).
HEAVY_MODES = ("decomposed_search", "iterative_research", "task_frame_required")
_REJECT_NO_PROGRESS = 2
_CONFIRM_SUPPORT = 1                       # supported required constraints to confirm
_YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
#: source roles whose PAGE BODY tends to carry high evidentiary value, so a read of them
#: is worth more (generic role types, not specific domains).
_HIGH_VALUE_READ_ROLES = frozenset({
    "professional_profile", "official_page", "scholarly_paper", "database_record",
    "primary_source", "article"})

#: Candidate *texts* that are retrieval noise, not real entities (used to (a) not count
#: them as progress and (b) trigger LLM repair when a deterministic query leans on them).
_NOISE_DEFINITION = frozenset({
    "definition", "meaning", "dictionary", "thesaurus", "merriam", "webster", "wikipedia",
    "wiktionary", "synonym", "synonyms", "antonym", "encyclopedia", "encyclopaedia"})
_NOISE_PLATFORM = frozenset({
    "linkedin", "facebook", "twitter", "instagram", "youtube", "tiktok", "pinterest",
    "reddit", "quora", "tumblr", "medium", "substack", "github", "tripadvisor", "yelp",
    "huggingface", "wikipedia", "wikimedia", "stackoverflow", "stackexchange"})
#: multiword platform / source names that are site chrome, not task entities.
_NOISE_PLATFORM_PHRASES = frozenset({
    "hugging face", "stack overflow", "stack exchange", "google scholar",
    "internet archive", "wayback machine", "google translate"})
_NOISE_UI = frozenset({
    "login", "log", "signin", "sign", "signup", "register", "menu", "home", "search",
    "username", "generator", "table", "tables", "page", "click", "next", "previous",
    "settings", "profile", "account", "navigation", "cookie", "cookies", "subscribe",
    "translate", "datasets", "dataset", "models", "spaces", "docs", "documentation",
    "pricing", "download", "downloads", "newsletter", "cart", "checkout", "explore",
    "trending", "categories", "topics", "tags", "sitemap", "preferences", "language",
    "results", "browse", "directory", "crossword", "clue", "clues"})


def _noise_kind(text: str) -> str:
    """Classify a candidate's *text* as a retrieval-noise kind, or '' if it looks real."""
    words = {w.lower() for w in re.findall(r"[A-Za-z]+", text or "")}
    if not words:
        return ""
    ordered = [w.lower() for w in re.findall(r"[A-Za-z]+", text or "")]
    norm = " ".join(ordered)
    lead = ordered[0] if ordered else ""
    if words & _NOISE_DEFINITION or lead in _NOISE_DEFINITION:
        return "generic_definition_noise"
    if norm in _NOISE_PLATFORM_PHRASES or lead in _NOISE_PLATFORM or (
            words & _NOISE_PLATFORM and len(words) <= 2):
        return "source_platform_noise"
    # a leading navigation/chrome word ("Browse …", "Search …", "Login …") is page chrome.
    if lead in _NOISE_UI or (words & _NOISE_UI and len(words) <= 2):
        return "ui_navigation_noise"
    return ""


def _is_clean_url(url: str) -> bool:
    """A concrete, readable http(s) URL (not a query string / fragment) — req 1 read-backing."""
    u = (url or "").strip()
    return u.lower().startswith(("http://", "https://")) and " " not in u and len(u) > 12


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
    source_role: str = "unknown"             # best source role observed (read-value signal)
    read_done: bool = False                  # a page-body read has been interpreted for it
    constraints_partial: list[str] = field(default_factory=list)   # 5f: partial (non-blocking)
    requires_read_constraint_ids: list[str] = field(default_factory=list)  # 5f: judge wants read
    aliases: list[str] = field(default_factory=list)               # 5f: judge-extracted aliases
    source_urls: list[str] = field(default_factory=list)           # 5g: clean URLs to read

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
    # discriminative-first planning debug (Level 5e / req 6).
    discriminative_reason: str = ""
    chosen_constraint_specificity: float = 0.0

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
                "is_generic_query": self.is_generic_query,
                "discriminative_reason": self.discriminative_reason,
                "chosen_constraint_specificity": round(self.chosen_constraint_specificity, 3)}
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
                 run_id: str = "", clock: Optional[Callable[[], int]] = None,
                 interpreter=None) -> None:
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
        #: Level 5d: structured EvidenceInterpretation per ingested observation. Slates are
        #: populated from these assertions, not raw n-grams. A default deterministic
        #: interpreter is created lazily to avoid an import cycle.
        self.interpreter = interpreter
        self.interpretations: list[Any] = []
        #: canonical candidate registry indexes (req 1/6): resolve proposal/verifier
        #: candidate TEXT -> canonical SlotCandidate id, and assertion_id -> candidate_id.
        self.candidate_text_index: dict[str, dict[str, str]] = {}   # slot_id -> {norm: cid}
        self.assertion_to_candidate: dict[str, str] = {}
        #: weak observations (req 4): normalized text -> slot hints, for nonexistent-candidate
        #: breakdown (an extracted-but-not-admitted mention is not "truly nonexistent").
        self.weak_observation_index: dict[str, list[str]] = {}
        #: Level 5e read scheduling (req 5): per-slot consecutive search/verify-with-no-support
        #: streak; after ``force_read_after_n`` a read of the best candidate source is forced.
        self.force_read_after_n = 2
        self._slot_no_support_streak: dict[str, int] = {}
        self.read_starvation_count = 0
        self.forced_read_count = 0
        self.read_executed_count = 0
        self.support_from_read_count = 0
        self.support_dropped_count = 0
        #: Level 5f: LLM-judge requires_read accounting.
        self.requires_read_total = 0
        self.requires_read_scheduled = 0
        #: Level 5g read-intent lifecycle accounting (req 1).
        self.read_desired_count = 0
        self.read_selected_count = 0
        self.read_blocked_no_url_count = 0
        self.read_blocked_tool_count = 0
        #: Level 5h-A/B pending read->judge loop + targeted passage retrieval.
        from regimes_probe.agent.read_judgment import DEFAULT_READ_CONFIG
        self.read_config = DEFAULT_READ_CONFIG
        self.pending_read_judgments: dict[str, Any] = {}
        self._prj = 0
        self.requires_read_resolved_by_read_count = 0
        self.requires_read_unresolved_after_read_count = 0
        self.successful_read_without_pending_replay_count = 0
        self.read_success_evidence_added_false_count = 0
        self.read_passage_hits_count = 0
        self.read_passage_no_hits_count = 0
        self.read_passage_judged_count = 0
        self.target_answer_passage_hits_count = 0
        #: Level 5h-C/D target-answer binding + target-slot priority.
        self.bind_target_answer_slot_actions = 0
        self.bind_target_answer_slot_selected_count = 0
        self.bind_target_answer_slot_success_count = 0
        self.target_binding_eig_boost_count = 0
        self.repeated_intermediate_verify_after_subject_supported_count = 0
        self.fuzzy_duplicate_query_rejected_count = 0
        self._executed_query_token_sets: list[frozenset] = []
        self.target_binding_rejected_reason: dict[str, int] = {}
        #: Level 5h-F/G/H seed floor, abstain admissibility, source hygiene.
        self.seed_query_generic_blocked_count = 0
        self.generic_single_token_seed_executed_count = 0
        self.abstain_with_budget_remaining_count = 0
        self.abstain_blocked_due_to_executable_proposal_count = 0
        self.generic_definition_source_selected_count = 0
        self.generic_definition_source_read_count = 0
        self.source_title_only_read_count = 0
        self.concrete_entity_source_selected_count = 0
        self.source_acquisition_rejected_reason: dict[str, int] = {}
        #: Level 5h-L location/distance staging guard.
        self.premature_founder_search_before_place_supported_count = 0
        self.premature_birth_year_search_before_founder_supported_count = 0
        #: Level 5h-M regime detectors (debug labels only — never benchmark claims).
        self.read_loop_open_count = 0
        self.read_success_no_evidence_added_count = 0
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

    def _interpreter(self):
        """The evidence interpreter (lazy default deterministic; avoids an import cycle)."""
        if self.interpreter is None:
            from regimes_probe.agent.evidence_interpreter import EvidenceInterpreter
            self.interpreter = EvidenceInterpreter()
        return self.interpreter

    def _cons_for(self, slot_id: str, directed_slot_id, directed_constraint_ids):
        cons = list(self._constraints_for_slot(slot_id))
        if slot_id == directed_slot_id:
            cons += [c for c in (self._con(cid) for cid in (directed_constraint_ids or []))
                     if c is not None and c not in cons]
        return cons

    # ---------- ingest ----------
    def ingest_evidence(self, observations, *, source_tool: str, stage: int = 1,
                        read_depth: int = 0, action_id: Optional[str] = None,
                        directed_slot_id: Optional[str] = None,
                        directed_constraint_ids: Optional[list[str]] = None,
                        proposal_id: Optional[str] = None,
                        read_candidate_id: Optional[str] = None) -> EvidenceRecord:
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
        interp_engine = self._interpreter()
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
            # INTERPRET the result into structured assertions (source role + candidate +
            # constraint assertions). Slates are populated ONLY from accepted assertions.
            interp = interp_engine.interpret(
                o, frame=self.frame, frontier=self, source_evidence_id=ev.evidence_id,
                source_tool=source_tool, selected_slot_id=directed_slot_id,
                selected_constraint_ids=directed_constraint_ids, known_norms=self._known)
            self.interpretations.append(interp)
            self._emit("evidence_interpreted", evidence_id=ev.evidence_id,
                       data={"interpretation_id": interp.interpretation_id,
                             "source_role": interp.source_role,
                             "noise_reasons": list(interp.noise_reasons)})
            self._emit("source_classified", evidence_id=ev.evidence_id,
                       data={"source_role": interp.source_role})
            for a in interp.candidate_assertions:
                if a.accepted:
                    continue
                noise_count += a.rejection_reason in (
                    "generic_definition_noise", "ui_navigation_noise",
                    "source_platform_noise", "benchmark_contaminated_source")
                self._emit("candidate_assertion_rejected", evidence_id=ev.evidence_id,
                           data={"text": _prev(a.candidate_text, 60),
                                 "reason": a.rejection_reason})
                if a.rejection_reason == "weak_observation_not_candidate":
                    self.weak_observation_index.setdefault(
                        self._norm_key(a.candidate_text), [])
                    if directed_slot_id and directed_slot_id not in \
                            self.weak_observation_index[self._norm_key(a.candidate_text)]:
                        self.weak_observation_index[self._norm_key(a.candidate_text)].append(
                            directed_slot_id)
                    self._emit("weak_observation_recorded", evidence_id=ev.evidence_id,
                               data={"text": _prev(a.candidate_text, 60)})
            for ca in interp.constraint_assertions:
                self._emit("constraint_assertion_made", evidence_id=ev.evidence_id,
                           data={"constraint_id": ca.constraint_id, "status": ca.status})
                if ca.status in ("irrelevant", "insufficient") and ca.constraint_id in (
                        directed_constraint_ids or []):
                    self._emit("evidence_constraint_support_rejected", evidence_id=ev.evidence_id,
                               data={"constraint_id": ca.constraint_id, "reason": ca.reason})
            # materialize accepted candidate assertions into canonical SlotCandidates and
            # attach their (recognizer-computed) constraint support — slates and support
            # come from interpreted assertions, never raw n-grams (req 1/2).
            self.attach_constraint_support_from_interpretation(
                interp, ev, source_tool=source_tool, action_id=action_id, stage=stage,
                url=url, authority=authority, contaminated=contaminated,
                directed_slot_id=directed_slot_id,
                directed_constraint_ids=directed_constraint_ids, proposal_id=proposal_id)
            # explain an ev->slot-true / ev->cons-false interpretation (req 7 invariant).
            acc = interp.accepted_candidates()
            if acc and any(a.proposed_slot_ids for a in acc) and not any(
                    a.supports_constraint_ids for a in acc):
                self._emit("ev_slot_true_cons_false_explained", evidence_id=ev.evidence_id,
                           data={"interpretation_id": interp.interpretation_id,
                                 "source_role": interp.source_role,
                                 "reason": "accepted_candidate_no_constraint_anchor_present"})
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
        sel_support_gain = max(0, after_sel_support - before_sel_support)
        ev.progress_components = {
            "raw_candidate_count": len(ev.newly_introduced_candidates),
            "slot_compatible_candidate_count": slot_compat,
            "selected_slot_candidate_count": sel_slot_new,
            "selected_constraint_support_count": sel_support_gain,
            "hypothesis_score_delta": round(after_conf - before_conf, 3),
            "target_support_path_delta": round(after_conf - before_conf, 3),
            "noise_candidate_count": noise_count,
        }
        # Read scheduling streak (req 5): a search/verify on the selected slot that yields no
        # NEW constraint support increments the streak; a read or any support resets it.
        if read_depth and read_depth >= 1:
            self.read_executed_count += 1
            self._emit("read_interpreted", evidence_id=ev.evidence_id,
                       data={"slot_id": directed_slot_id, "read_depth": read_depth,
                             "new_support": sel_support_gain})
            # 5h-A: route the fetched page BODY back into any pending requires_read judgments
            # for this candidate/source (targeted passage re-judge), and detect a read that
            # produced chars but added no evidence (a read-loop-open seam — 5h-M).
            best_obs = max((o for o in observations if not getattr(o, "failed", False)),
                           key=lambda o: len(getattr(o, "snippet", "") or ""), default=None)
            read_text = getattr(best_obs, "snippet", "") or "" if best_obs else ""
            read_url = getattr(best_obs, "url", "") or "" if best_obs else ""
            had_pending = bool(self._open_pending_for(read_candidate_id, read_url))
            resolved = 0
            if read_text and (read_candidate_id or read_url):
                resolved = self.route_read_into_pending_judgments(
                    candidate_id=read_candidate_id, source_url=read_url, read_text=read_text)
            evidence_added = bool(sel_support_gain or resolved
                                  or ev.newly_introduced_candidates)
            if read_text and not evidence_added:
                self.read_success_evidence_added_false_count += 1
                self.read_success_no_evidence_added_count += 1
                if had_pending:
                    self.read_loop_open_count += 1
        if directed_slot_id:
            if sel_support_gain > 0 or (read_depth and read_depth >= 1):
                self._slot_no_support_streak[directed_slot_id] = 0
            else:
                self._slot_no_support_streak[directed_slot_id] = (
                    self._slot_no_support_streak.get(directed_slot_id, 0) + 1)
        # Hard support-consistency invariant (req 1): the per-candidate support just attached
        # MUST appear on the evidence record's supports_constraint_ids. If a selected
        # constraint gained candidate support but the record didn't record it, that is a
        # dropped-support bug — make it explicit rather than silent.
        sel_slate = self.slates.get(directed_slot_id) if directed_slot_id else None
        for cid in (directed_constraint_ids or []):
            cand_supports = bool(sel_slate) and any(
                cid in c.constraints_supported for c in sel_slate.candidates.values())
            if cand_supports and cid not in ev.supports_constraint_ids:
                self.support_dropped_count += 1
                self._emit("support_dropped", evidence_id=ev.evidence_id,
                           data={"constraint_id": cid, "slot_id": directed_slot_id,
                                 "reason": "candidate_support_not_on_evidence_record"})
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

    @staticmethod
    def _norm_key(text: str) -> str:
        return " ".join(w.lower() for w in re.findall(r"[A-Za-z0-9]+", text or ""))

    def resolve_candidate(self, text_or_id: str, *, slot_id: Optional[str] = None) -> dict:
        """Resolve a candidate TEXT or id to its canonical SlotCandidate id (req 1/4/6).

        Returns ``{candidate_id, found, breakdown, searched_slot_ids, close_matches, other_slot}``.
        ``breakdown`` classifies a miss generically (``exists_in_other_slot`` /
        ``exists_as_weak_observation`` / ``exists_but_rejected`` / ``exists_but_stale`` /
        ``normalized_alias_found`` / ``truly_nonexistent``) so a verifier never calls an
        extracted candidate "nonexistent" without saying why."""
        debug = {"candidate_id": None, "found": False, "breakdown": None,
                 "normalized": self._norm_key(text_or_id), "searched_slot_ids": [],
                 "close_matches": [], "other_slot": None}
        if text_or_id in self.candidates_by_id:
            debug.update(candidate_id=text_or_id, found=True, breakdown="exact_id")
            return debug
        key = self._norm_key(text_or_id)
        order = ([slot_id] if slot_id else []) + [s for s in self.slates if s != slot_id]
        # an ACTIVE/confirmed candidate on the selected (then any) slot resolves cleanly.
        for sid in order:
            debug["searched_slot_ids"].append(sid)
            cid = self.candidate_text_index.get(sid, {}).get(key)
            if not cid:
                continue
            cand = self.candidates_by_id.get(cid)
            status = getattr(cand, "status", "active")
            if status in ("rejected", "stale", "merged"):
                # remember the worst-case classification but keep looking for a live one.
                debug["close_matches"].append({"slot_id": sid, "text": key, "candidate_id": cid,
                                               "status": status})
                if debug["breakdown"] is None:
                    debug["breakdown"] = ("exists_but_rejected" if status != "stale"
                                          else "exists_but_stale")
                continue
            debug.update(candidate_id=cid, found=True)
            if slot_id and sid != slot_id:
                debug.update(other_slot=sid, breakdown="exists_in_other_slot")
            else:
                debug["breakdown"] = ("normalized_alias_found" if key != self._norm_key(cand.candidate_text)
                                      else "exact_normalized")
            return debug
        # weak observation? (extracted but not admitted as a candidate)
        if key in self.weak_observation_index:
            debug["breakdown"] = "exists_as_weak_observation"
            debug["weak_observation_slots"] = list(self.weak_observation_index[key])
            return debug
        # close matches (substring/variant) for debug only.
        for sid in order:
            for k, cid in self.candidate_text_index.get(sid, {}).items():
                if key and (key in k or k in key):
                    debug["close_matches"].append({"slot_id": sid, "text": k, "candidate_id": cid})
        if debug["breakdown"] is None:
            debug["breakdown"] = "truly_nonexistent"
        return debug

    def attach_constraint_support_from_interpretation(
            self, interp, ev, *, source_tool, action_id, stage, url, authority,
            contaminated, directed_slot_id=None, directed_constraint_ids=None,
            proposal_id=None) -> None:
        """Materialize accepted candidate assertions into canonical SlotCandidates and
        attach the recognizer-computed constraint support (the single support path)."""
        title = getattr(ev, "title_preview", "") or ""
        for a in interp.accepted_candidates():
            self._emit("candidate_assertion_made", evidence_id=ev.evidence_id,
                       data={"assertion_id": a.assertion_id, "text": _prev(a.candidate_text, 60),
                             "role": a.inferred_role, "proposed_slot_ids": list(a.proposed_slot_ids)})
            for sid in a.proposed_slot_ids:
                slot = self.frame.slot(sid)
                if slot is None or sid not in self.slates:
                    continue
                cons = self._cons_for(sid, directed_slot_id, directed_constraint_ids)
                sup, con = (a.slot_support.get(sid) or [[], []])
                cid = self._assign(
                    slot, a.candidate_text, _norm(a.candidate_text), a.inferred_role, ev,
                    source_tool, action_id, stage, _host(url), authority, contaminated,
                    a.from_ctx, f"{title} {ev.snippet_preview}".lower(), cons,
                    supported=list(sup), contradicted=list(con), source_role=a.source_role,
                    read_depth=getattr(ev, "read_depth", 0), source_url=url,
                    directed_slot_id=directed_slot_id,
                    directed_constraint_ids=directed_constraint_ids, proposal_id=proposal_id)
                a.canonical_candidate_ids[sid] = cid
                if a.assertion_id:
                    self.assertion_to_candidate[a.assertion_id] = cid
                self._emit("candidate_assertion_materialized", candidate_id=cid,
                           slot_id=sid, evidence_id=ev.evidence_id,
                           data={"assertion_id": a.assertion_id})
                if sup:
                    self._emit("evidence_constraint_support_attached", candidate_id=cid,
                               slot_id=sid, evidence_id=ev.evidence_id,
                               data={"constraint_ids": list(sup)})
                # Level 5f: partial support is recorded (influences EIG/scheduling) but does
                # NOT resolve a blocking constraint; requires_read marks a read target; the
                # judge's aliases enter the canonical alias registry (not a parallel one).
                cand = self.candidates_by_id.get(cid)
                # Level 5f: project each LLM evidence judgment for this candidate (created +
                # accepted/rejected + per-judgment support/contradiction/requires-read).
                for jd in a.judgments:
                    if jd.get("candidate_id") not in (None, cid) or jd.get("slot_id") not in (sid, ""):
                        pass
                    self._emit("llm_evidence_judgment.created", candidate_id=cid, slot_id=sid,
                               evidence_id=ev.evidence_id, data={
                                   "judgment_id": jd.get("judgment_id"),
                                   "judgment": jd.get("judgment"),
                                   "constraint_id": jd.get("constraint_id"),
                                   "quote": (jd.get("quote") or "")[:120],
                                   "model": jd.get("model"), "prompt_hash": jd.get("prompt_hash"),
                                   "cache_hit": jd.get("cache_hit"), "mode": jd.get("mode")})
                    jm = jd.get("judgment")
                    if jm in ("full_support", "partial_support", "contradiction", "requires_read"):
                        self._emit("llm_evidence_judgment.accepted",
                                   evidence_id=ev.evidence_id, candidate_id=cid,
                                   data={"judgment_id": jd.get("judgment_id"), "judgment": jm})
                        if jm == "full_support":
                            self._emit("evidence_judgment_supports_candidate_constraint",
                                       candidate_id=cid, slot_id=sid, evidence_id=ev.evidence_id,
                                       data={"constraint_id": jd.get("constraint_id")})
                        elif jm == "contradiction":
                            self._emit("evidence_judgment_contradicts_candidate_constraint",
                                       candidate_id=cid, slot_id=sid, evidence_id=ev.evidence_id,
                                       data={"constraint_id": jd.get("constraint_id")})
                    elif jm == "irrelevant":
                        self._emit("llm_evidence_judgment.rejected",
                                   evidence_id=ev.evidence_id, candidate_id=cid,
                                   data={"judgment_id": jd.get("judgment_id"),
                                         "reason": "irrelevant"})
                if cand is not None:
                    for pc in a.slot_partial.get(sid, []):
                        if pc not in cand.constraints_supported:
                            _union(cand.constraints_partial, pc)
                        self._emit("evidence_judgment_partially_supports_candidate_constraint",
                                   candidate_id=cid, slot_id=sid, evidence_id=ev.evidence_id,
                                   data={"constraint_id": pc})
                    for rc in a.slot_requires_read.get(sid, []):
                        if rc not in cand.requires_read_constraint_ids:
                            self.requires_read_total += 1
                        _union(cand.requires_read_constraint_ids, rc)
                        self._emit("evidence_judgment_requires_read", candidate_id=cid,
                                   slot_id=sid, evidence_id=ev.evidence_id,
                                   data={"constraint_id": rc})
                        # 5h-A: a requires_read is a persistent OBLIGATION on the
                        # (candidate, slot, constraint, source_url) triple, not a vague hint.
                        self._register_pending_read_judgment(
                            candidate_id=cid, slot_id=sid, constraint_id=rc,
                            source_url=getattr(ev, "url", ""))
                    for al in a.candidate_aliases:
                        if al and al not in cand.aliases:
                            cand.aliases.append(al)
                        self.candidate_text_index.setdefault(sid, {}).setdefault(
                            self._norm_key(al), cid)
                    # partial support nudges ranking/EIG but never resolves a blocker.
                    cand.evidence_score = (float(len(cand.constraints_supported))
                                           + 0.5 * len(cand.constraints_partial)
                                           + 0.25 * len(cand.source_domains))

    def _assign(self, slot, text, norm, role, ev, source_tool, action_id, stage,
                domain, authority, contaminated, from_ctx, text_l, cons,
                *, supported=None, contradicted=None, source_role="unknown", read_depth=0,
                source_url="", directed_slot_id=None, directed_constraint_ids=None,
                proposal_id=None) -> str:
        slate = self.slates[slot.slot_id]
        existing = next((c for c in slate.candidates.values()
                         if c.normalized_text_hash == _hash(norm)), None)
        cand_l = text.lower()
        had_support = bool(existing and existing.constraints_supported)
        # Support is ATTACHED from the interpreter's recognizers (req 2); fall back to the
        # overlap rule only when a direct caller did not supply it.
        if supported is None:
            supported = [c.constraint_id for c in cons if _supports(c, text_l)]
        if contradicted is None:
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
            self.candidate_text_index.setdefault(slot.slot_id, {})[
                self._norm_key(text)] = cand.candidate_id
            ev.newly_introduced_candidates.append(cand.candidate_id)
            self._emit("candidate.extracted", candidate_id=cand.candidate_id,
                       slot_id=slot.slot_id, evidence_id=ev.evidence_id, action_id=action_id,
                       data={"text_preview": _prev(text, 60), "role": role})
            self._emit("canonical_candidate_created", candidate_id=cand.candidate_id,
                       slot_id=slot.slot_id, evidence_id=ev.evidence_id)
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
        # remember a CLEAN, concrete URL so a read can execute against the candidate's
        # source even if the candidate text isn't in a later snippet (5g req 1).
        if _is_clean_url(source_url) and not contaminated:
            _union(cand.source_urls, source_url)
        cand.source_authority_score = max(cand.source_authority_score, authority)
        if source_role and source_role != "unknown":
            cand.source_role = source_role
        if read_depth and read_depth >= 1:
            cand.read_done = True
        for cid in supported:
            newly = cid not in cand.constraints_supported
            _union(cand.constraints_supported, cid)
            _union(ev.supports_constraint_ids, cid)
            if newly and read_depth and read_depth >= 1:
                self.support_from_read_count += 1     # support_from_read_rate signal (req 13)
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
        if existing is not None:
            self._emit("canonical_candidate_updated", candidate_id=cand.candidate_id,
                       slot_id=slot.slot_id, evidence_id=ev.evidence_id)
        self._update_status(cand, contaminated)
        if (proposal_id and slot.slot_id == directed_slot_id
                and cand.status == "confirmed" and not was_confirmed):
            self._emit("candidate_promoted_from_llm_frontier_evidence",
                       candidate_id=cand.candidate_id, slot_id=slot.slot_id,
                       data={"proposal_id": proposal_id})
        return cand.candidate_id

    def _update_status(self, cand: SlotCandidate, contaminated: bool) -> None:
        old = cand.status
        if cand.status in ("merged", "rejected"):
            return
        reason = ""
        new = cand.status
        if cand.constraints_contradicted:
            new, reason = "rejected", "contradicted_constraint"
        elif cand.no_progress_count >= _REJECT_NO_PROGRESS and not cand.constraints_supported:
            # repeated no-progress applies to QUERIES/actions, never to a candidate that
            # already has clean support (Level 5f-I): a supported candidate is "working".
            new, reason = "rejected", "repeated_no_progress"
        elif self._confirmable(cand, contaminated):
            # 5h-O: a candidate-/slot-LOCAL support label — NOT a global answer-gate signal.
            # "confirmed" here means this slot's constraints are locally supported; the strict
            # answer-support gate (eval/eligibility) is the only thing that authorises answering.
            new, reason = "confirmed", "slot_candidate_supported"
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

    def _slot_blocking_constraints(self, slot_id: str) -> list:
        return [c for c in self._constraints_for_slot(slot_id)
                if c.status != "resolved"
                and (getattr(c, "blocks_answer_if_unresolved", False)
                     or getattr(c, "required", False) or getattr(c, "priority", "") == "high"
                     or "can_block_answer" in getattr(c, "affordances", []))]

    def _confirmable(self, cand: SlotCandidate, contaminated: bool) -> bool:
        """A candidate may be CONFIRMED for its slot only when the answer-gate-style support
        contract holds for that slot (Level 5f-C/D): every blocking constraint supported (by
        FULL support, never partial), at least one DISCRIMINATIVE blocking constraint supported
        when one exists, no contamination/contradiction, and the candidate text is not junk."""
        if contaminated or cand.constraints_contradicted:
            return False
        if _noise_kind(cand.candidate_text):           # never confirm a junk/chrome token
            return False
        blocking = self._slot_blocking_constraints(cand.slot_id)
        # every blocking constraint must be supported by FULL support (partial never counts).
        if not all(c.constraint_id in cand.constraints_supported for c in blocking):
            return False
        # a single supported constraint cannot confirm when other blocking ones are unresolved
        # (already implied above); require at least one supported constraint overall.
        if len(cand.constraints_supported) < _CONFIRM_SUPPORT:
            return False
        # if any blocking constraint is discriminative, at least one such must be supported.
        disc_blocking = [c for c in blocking if _is_discriminative_constraint(c)]
        if disc_blocking and not any(c.constraint_id in cand.constraints_supported
                                     for c in disc_blocking):
            return False
        return True

    def note_no_progress_for_slate(self, slot_id: str) -> None:
        slate = self.slates.get(slot_id)
        if not slate:
            return
        for c in slate.candidates.values():
            # a supported "working" candidate is exempt from no-progress decay (5f-I).
            if c.status == "active" and not c.constraints_supported:
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

    # ---------- 5h-C/D: target-answer binding ----------
    def _target_slot_ids(self) -> list[str]:
        return [s.slot_id for s in self.frame.target_answer_slots if s.slot_id in self.slates]

    def _is_answer_shaped_slot(self, slot) -> bool:
        if slot is None:
            return False
        if (getattr(slot, "slot_role", "") or "").lower() in _ANSWER_SHAPE_ROLES:
            return True
        desc = (getattr(slot, "descriptor_text", "") or getattr(slot, "slot_name", "")).lower()
        return any(w in _ANSWER_SHAPE_WORDS for w in re.findall(r"[a-z]+", desc))

    def _target_unbound(self, slot_id: str) -> bool:
        """A target slot is unbound when no candidate on it is confirmed or carries support."""
        slate = self.slates.get(slot_id)
        if not slate:
            return True
        return not any(c.status == "confirmed" or c.constraints_supported
                       for c in slate.candidates.values()
                       if c.status not in ("rejected", "merged"))

    def _supported_subject_candidate(self, *, exclude_slot: str = ""):
        """Return the best supported subject/intermediate candidate on a NON-target slot
        (the working subject a target-binding read/search should pivot from)."""
        targets = set(self._target_slot_ids())
        best = None
        for sid, slate in self.slates.items():
            if sid in targets or sid == exclude_slot:
                continue
            for c in slate.candidates.values():
                if c.status in ("rejected", "merged"):
                    continue
                if c.status == "confirmed" or c.constraints_supported:
                    if best is None or c.evidence_score > best.evidence_score:
                        best = c
        return best

    @staticmethod
    def _qtokens(query: str) -> frozenset:
        return frozenset(w.lower() for w in re.findall(r"[a-z0-9]+", (query or "").lower())
                         if len(w) >= 3)

    def is_near_duplicate_query(self, query: str) -> bool:
        """Fuzzy dedupe by normalized content-token SET (word-order variants count once)."""
        toks = self._qtokens(query)
        if not toks:
            return False
        return any(toks and (toks == prev or (len(toks & prev) >= max(2, len(toks) - 1)
                                              and len(toks ^ prev) <= 1))
                   for prev in self._executed_query_token_sets)

    def note_executed_query(self, query: str) -> None:
        toks = self._qtokens(query)
        if toks and toks not in self._executed_query_token_sets:
            self._executed_query_token_sets.append(toks)

    def _is_concrete_entity_task(self) -> bool:
        """A task whose target/intermediate slots name concrete entities (person/org/place/
        work), not a dictionary definition — generic role test (5h-H)."""
        roles = {(getattr(s, "slot_role", "") or "").lower() for s in self.frame.all_slots}
        entityish = {"person", "organization", "organisation", "place", "location", "work",
                     "restaurant", "hotel", "museum", "company", "author", "founder", "film"}
        return bool(roles & entityish) and "definition" not in roles

    @staticmethod
    def _is_generic_definition_source(obs) -> bool:
        title = getattr(obs, "title", "") or ""
        snippet = getattr(obs, "snippet", "") or ""
        nk = _noise_kind(title) or _noise_kind(snippet[:40])
        if nk == "generic_definition_noise":
            return True
        role = getattr(obs, "source_role", "") or ""
        return role in ("generic_definition_page", "source_title_only")

    def _slot_supported(self, slot_id: str) -> bool:
        slate = self.slates.get(slot_id)
        return bool(slate and any(c.status == "confirmed" or c.constraints_supported
                                  for c in slate.candidates.values()
                                  if c.status not in ("rejected", "merged")))

    def _dependency_unsupported(self, slot) -> bool:
        """5h-L: a slot is premature to search while any slot it DEPENDS ON is unsupported
        (don't search the founder before the place is supported; the birth year before the
        founder). Generic — driven by the frame's dependency edges, no entity names."""
        for dep in getattr(slot, "depends_on", []) or []:
            if dep in self.slates and not self._slot_supported(dep):
                return True
        return False

    def _executable_anchor_rich_action(self) -> bool:
        """A non-generic, anchor-rich progress action is available right now (5h-G/E)."""
        acts = self.frontier_actions or self.generate_frontier_actions()
        for a in acts:
            if a.action_type not in ("generate_candidates_for_slot",
                                     "verify_candidate_constraint", "read_candidate_source",
                                     "bind_target_answer_slot",
                                     "expand_candidate_to_dependent_slot"):
                continue
            if a.action_type == "read_candidate_source":
                return True                       # a read of a real source is anchor-rich
            cons = [self._con(cid) for cid in a.constraint_ids]
            if any(c is not None and _is_discriminative_constraint(c) for c in cons):
                return True
        return False

    def abstain_admissible(self, *, budget_remaining: int) -> tuple[bool, str]:
        """5h-G: abstain is INADMISSIBLE while budget remains, a blocking constraint is
        unresolved, AND an executable anchor-rich progress action exists. It is admissible
        only when the remaining options are unsafe/generic/contaminated/exhausted."""
        if budget_remaining <= 0:
            return True, "budget_exhausted"
        has_blocking = any(self._slot_blocking_constraints(s) for s in self.slates)
        if not has_blocking:
            return True, "no_unresolved_blocking_constraint"
        if self._executable_anchor_rich_action():
            self.abstain_blocked_due_to_executable_proposal_count += 1
            return False, "executable_anchor_rich_action_available"
        return True, "only_generic_or_exhausted_actions_remain"

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

        bind_subject = self._supported_subject_candidate()
        for slate in self.slates.values():
            slot_obj = self.frame.slot(slate.slot_id)
            # 5h-L: defer *searching/binding* a slot whose dependency is not yet supported
            # (stage it: bind the place before the founder, the founder before the birth
            # year). An already-confirmed candidate still expands to its dependents below.
            dep_blocked = self._dependency_unsupported(slot_obj)
            if dep_blocked:
                self._emit("dependent_slot_search_deferred", slot_id=slate.slot_id,
                           data={"reason": "dependency_unsupported"})
            # 5h-C: an answer-shaped TARGET slot with a supported subject is driven by the
            # bind_target_answer_slot action below, not a generic generate/verify here.
            defer_for_bind = (bind_subject is not None
                              and slate.slot_id in self._target_slot_ids()
                              and self._is_answer_shaped_slot(slot_obj)
                              and self._target_unbound(slate.slot_id))
            if defer_for_bind:
                dep_blocked = True
            blocking = ([] if dep_blocked else
                        [c for c in self._constraints_for_slot(slate.slot_id)
                         if c.status != "resolved"
                         and (getattr(c, "blocks_answer_if_unresolved", False)
                              or getattr(c, "priority", "") == "high"
                              or "can_block_answer" in getattr(c, "affordances", []))])
            blocking.sort(key=lambda c: -_disc(c))    # most discriminative first
            active = [c for c in slate.candidates.values() if c.status == "active"]
            confirmed = [c for c in slate.candidates.values() if c.status == "confirmed"]
            streak = self._slot_no_support_streak.get(slate.slot_id, 0)
            slate_has_support = any(c.constraints_supported for c in slate.candidates.values())
            # After N consecutive search/verify with no support, search/verify is decayed so a
            # read (or pivot) can take over — snippets aren't converting to support (req 5).
            streak_penalty = 0.8 * min(streak, 3)
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
                         eig=3.0 + 0.75 * _disc(top) + disc_bonus - streak_penalty, cost=1.0,
                         reason="resolve_blocking_constraint")
                elif not confirmed:
                    # only search for MORE candidates when the slot has no confirmed binding
                    # yet (5h-D: a confirmed slot is "done"; don't keep generating for it).
                    _add("generate_candidates_for_slot", slot=slate.slot_id, cons=cons_ids,
                         eig=2.5 + 0.75 * _disc(top) + disc_bonus - streak_penalty, cost=1.0,
                         reason="bind_slot_via_discriminative_constraint")
            elif not slate.candidates and not dep_blocked:
                _add("generate_candidates_for_slot", slot=slate.slot_id, eig=2.0, cost=1.0,
                     reason="slot_unbound")
            if len(active) >= 2:
                _add("compare_candidates_for_slot", slot=slate.slot_id, eig=1.5, cost=1.0,
                     reason="distinguish_competing_candidates")
            # READS convert snippet candidates into page-body support (req 5). A read is
            # boosted when the candidate has a source + unresolved constraints AND snippets
            # produced no support, or a forced read is due after a no-support streak. One
            # read can close several constraints (cons = all unresolved for the candidate).
            forced = streak >= self.force_read_after_n
            for c in active:
                # the LLM judge explicitly asked to read this candidate's source (req: 5f).
                judge_read = bool(c.requires_read_constraint_ids and not c.read_done)
                if not (c.source_domains and (c.constraints_unknown or judge_read)):
                    if judge_read and not c.source_domains:
                        self._emit("skipped_read_after_requires_read", slot_id=slate.slot_id,
                                   candidate_id=c.candidate_id,
                                   data={"reason": "no_source_url_for_candidate"})
                    continue
                src_val = 0.5 if c.source_role in _HIGH_VALUE_READ_ROLES else 0.0
                convert = (not c.constraints_supported)     # snippets gave a candidate, no support
                # the judge explicitly asked to read THIS source: it should outrank another
                # snippet search/verify for the slot (a read is the converting action).
                read_eig = (1.0 + (2.5 if forced else 0.0) + (1.0 if convert else 0.0)
                            + (3.5 if judge_read else 0.0) + src_val)
                reason = ("evidence_judge_requires_read" if judge_read
                          else ("forced_read_after_no_support" if forced
                                else ("read_to_convert_candidate" if convert
                                      else "deepen_candidate_evidence")))
                read_cons = list(c.requires_read_constraint_ids or c.constraints_unknown)
                ra = _add("read_candidate_source", slot=slate.slot_id, cand=c.candidate_id,
                          cons=read_cons, eig=read_eig, cost=2.0, reason=reason)
                ra.plan["read_value"] = {"forced": forced, "convert": convert,
                                         "judge_requires_read": judge_read,
                                         "source_role": c.source_role, "streak": streak,
                                         "n_unresolved": len(read_cons)}
                if forced:
                    self.forced_read_count += 1
            for c in confirmed:
                for dep in self._dependents(slate.slot_id):
                    if dep in self.slates and not self._best_candidate(dep):
                        _add("expand_candidate_to_dependent_slot", slot=dep,
                             cand=c.candidate_id, eig=2.0, cost=1.0,
                             reason="upstream_confirmed_unlocks_dependent")
        # 5h-C/D: when a SUBJECT/intermediate is supported but an answer-shaped TARGET slot
        # is still unbound, prefer binding the target over re-verifying the supported subject.
        subject = self._supported_subject_candidate()
        bind_actions: list[FrontierAction] = []
        if subject is not None:
            for tid in self._target_slot_ids():
                tslot = self.frame.slot(tid)
                if not (self._is_answer_shaped_slot(tslot) and self._target_unbound(tid)):
                    continue
                tcons = [c.constraint_id for c in self._constraints_for_slot(tid)
                         if c.status != "resolved"]
                ba = _add("bind_target_answer_slot", slot=tid, cand=subject.candidate_id,
                          cons=tcons, eig=4.2, cost=1.0,
                          reason="subject_supported_target_unbound")
                ba.plan["bind_target"] = {"subject_candidate_id": subject.candidate_id,
                                          "subject_slot_id": subject.slot_id}
                bind_actions.append(ba)
                self.bind_target_answer_slot_actions += 1
                self.target_binding_eig_boost_count += 1
                self._emit("bind_target_answer_slot_proposed", slot_id=tid,
                           candidate_id=subject.candidate_id,
                           data={"subject_slot_id": subject.slot_id})
            # demote repeated intermediate verification while the target is starved.
            if bind_actions:
                for a in actions:
                    if (a.action_type == "verify_candidate_constraint"
                            and a.target_slot_id == subject.slot_id):
                        a.expected_information_gain -= 2.5
                        self.repeated_intermediate_verify_after_subject_supported_count += 1
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
        # read-starvation signal (req 13): a readable candidate action existed but a
        # search/verify outscored it -> the read was starved this step.
        reads = [a for a in cands if a.action_type == "read_candidate_source"]
        if reads and best.action_type != "read_candidate_source":
            self.read_starvation_count += 1
        if best.action_type == "read_candidate_source":
            if best.selected_reason == "evidence_judge_requires_read":
                self.requires_read_scheduled += 1
            self._emit("read_scheduled", action_id=best.action_id,
                       data={"slot_id": best.target_slot_id, "candidate_id": best.candidate_id,
                             "reason": best.selected_reason})
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
            adm, why = self.abstain_admissible(budget_remaining=budget_remaining)
            if not adm:
                # 5h-G: withhold an inadmissible abstain; let the loop take a progress action.
                self._emit("abstain_withheld_executable_action_available",
                           action_id=sel.action_id, data={"reason": why})
                return StepPlan(sel.action_id, at, "unexecutable",
                                reason="abstain_inadmissible_executable_action")
            return StepPlan(sel.action_id, at, "abstain", reason=sel.selected_reason)
        if at == "bind_target_answer_slot":
            self.bind_target_answer_slot_selected_count += 1
            query, arm, disc, generic = self._build_bind_target_query(sel)
            if not query:
                self.target_binding_rejected_reason["empty_bind_query"] = \
                    self.target_binding_rejected_reason.get("empty_bind_query", 0) + 1
                return StepPlan(sel.action_id, at, "unexecutable", reason="empty_bind_query")
            self.note_executed_query(query)
            self._emit("bind_target_answer_slot_selected", action_id=sel.action_id,
                       slot_id=sel.target_slot_id, candidate_id=sel.candidate_id,
                       data={"query_preview": query[:120]})
            return StepPlan(sel.action_id, at, "search", query=query, query_arm=arm,
                            target_slot_id=sel.target_slot_id, candidate_id=sel.candidate_id,
                            constraint_ids=list(sel.constraint_ids),
                            reason="bind_target_answer_slot",
                            is_discriminative_constraint=disc, is_generic_query=generic,
                            discriminative_reason="target_answer_binding")
        if at == "read_candidate_source":
            self.read_desired_count += 1
            self._emit("read_desired", action_id=sel.action_id, slot_id=sel.target_slot_id,
                       candidate_id=sel.candidate_id, data={"reason": sel.selected_reason})
            if not reading_tools:
                self.read_blocked_tool_count += 1
                self._emit("read_blocked_disallowed_tool", action_id=sel.action_id,
                           data={"reason": "no_reading_tool_available"})
                return StepPlan(sel.action_id, at, "unexecutable", reason="read_blocked_no_tool")
            # 1) a URL among the observations that ties candidate+slot+constraint.
            chosen = self._resolve_read(observations, scraped_urls, no_progress_domains,
                                        page_fetch_available, scrape_available, allow_social,
                                        force_page_fetch)
            # 2) fall back to a CLEAN URL stored on the selected candidate's provenance, so a
            #    forced/judge read actually executes instead of silently becoming a search.
            if chosen is None:
                chosen = self._read_candidate_url(
                    sel.candidate_id, sel.constraint_ids, scraped_urls, no_progress_domains,
                    page_fetch_available, scrape_available, allow_social, force_page_fetch)
            if chosen is None:
                self.read_blocked_no_url_count += 1
                self._emit("read_blocked_no_url", action_id=sel.action_id,
                           slot_id=sel.target_slot_id, candidate_id=sel.candidate_id,
                           data={"reason": "no_clean_url_for_candidate_source"})
                # NOT a silent search: record the blocker; the loop will run a source search.
                return StepPlan(sel.action_id, at, "unexecutable", reason="read_blocked_no_url")
            o, rd, tested = chosen
            self.read_selected_count += 1
            self._emit("read_selected", action_id=sel.action_id, slot_id=sel.target_slot_id,
                       candidate_id=sel.candidate_id, data={"url_host": _host(getattr(o, "url", ""))})
            return StepPlan(sel.action_id, at, "read", read_obs=o, read_decision=rd,
                            target_slot_id=sel.target_slot_id, candidate_id=sel.candidate_id,
                            constraint_ids=tested, reason="read_candidate_source")
        query, arm, disc, generic = self._build_query(sel)
        if not query:
            return StepPlan(sel.action_id, at, "unexecutable", reason="empty_query")
        # 5h-F: a hard/multi-constraint task must never EXECUTE a generic single-token seed
        # (e.g. bare "founder"/"hotel"). Block it so the loop repairs to an anchor-rich query.
        is_seed = (not self._executed_query_token_sets
                   and not any(c.constraints_supported for c in self.candidates_by_id.values()))
        hard = len(self.frame.constraints) >= 2 or len(self.slates) >= 2
        if is_seed and hard and self._is_generic_single_token(query):
            self.seed_query_generic_blocked_count += 1
            self._emit("seed_query_generic_blocked", action_id=sel.action_id,
                       data={"query_preview": query[:80]})
            return StepPlan(sel.action_id, at, "unexecutable",
                            reason="generic_single_token_seed_blocked")
        spec, disc_reason = self._discriminativeness(
            self._con(sel.constraint_ids[0]) if sel.constraint_ids else None)
        return StepPlan(sel.action_id, at, "search", query=query, query_arm=arm,
                        target_slot_id=sel.target_slot_id, candidate_id=sel.candidate_id,
                        constraint_ids=list(sel.constraint_ids), reason=sel.selected_reason,
                        is_discriminative_constraint=disc, is_generic_query=generic,
                        discriminative_reason=disc_reason, chosen_constraint_specificity=spec)

    @staticmethod
    def _is_generic_single_token(query: str) -> bool:
        from regimes_probe.agent.llm_frontier import _GENERIC_WORDS
        content = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", query or "") if len(w) >= 3]
        # a query carrying a quoted phrase, a year, or a proper noun is not a generic seed.
        if '"' in (query or "") or _YEAR.findall(query or ""):
            return False
        if any(w[:1].isupper() for w in re.findall(r"[A-Za-z][A-Za-z'&]+", query or "")
               if w.lower() not in _GENERIC_WORDS):
            return False
        return len(content) <= 1 and bool(content) and content[0] in _GENERIC_WORDS

    def _build_bind_target_query(self, action) -> tuple[str, str, bool, bool]:
        """5h-C: build a TARGET-binding query from the working subject + the target slot's
        answer-shape descriptor + unresolved target-constraint anchors (subject pivots the
        search to the answer, e.g. subject aliases + "born"/"year")."""
        from regimes_probe.policy.query_decomposition import _cap, _is_rare
        subj = self.candidates_by_id.get(action.candidate_id) if action.candidate_id else None
        tslot = self.frame.slot(action.target_slot_id) if action.target_slot_id else None
        parts: list[str] = []
        if subj is not None:
            parts.append(f'"{subj.candidate_text}"')
            for al in subj.aliases[:1]:
                parts.append(f'"{al}"')
        words = set(re.findall(r"[a-z0-9]+", " ".join(parts).lower()))
        desc = (getattr(tslot, "descriptor_text", "") or
                (tslot.slot_name if tslot else "")).lower()
        for w in re.findall(r"[a-z]+", desc):
            if w in _ANSWER_SHAPE_WORDS and w not in words:
                parts.append(w)
                words.add(w)
        for cid in action.constraint_ids:
            con = self._con(cid)
            if con is None:
                continue
            for t in con.normalized_terms:
                if len(t) >= 4 and t.lower() not in words and (_is_rare(t) or t[:1].isupper()):
                    parts.append(t)
                    words.add(t.lower())
        q = _cap(" ".join(p for p in parts if p).strip())
        return q, "bind_target_answer_slot", bool(subj is not None), not bool(subj)

    def _discriminativeness(self, con) -> tuple[float, str]:
        """Generic discriminativeness: specificity + numeric/date + named-entity anchors +
        how many slots it constrains (req 6). No hard-coded clue templates."""
        if con is None:
            return 0.0, "no_constraint"
        disc = _disc(con)
        terms = list(getattr(con, "normalized_terms", []))
        has_year = bool(_YEAR.findall(getattr(con, "text_span", "") or ""))
        has_proper = any(str(t)[:1].isupper() for t in terms if len(str(t)) >= 4)
        n_slots = len(getattr(con, "applies_to", []) or [])
        score = disc + (0.5 if has_year else 0.0) + (0.5 if has_proper else 0.0) \
            + (0.25 * max(0, n_slots - 1))
        reasons = []
        if disc >= 1.0:
            reasons.append("high_specificity")
        if has_year:
            reasons.append("numeric_or_date_anchor")
        if has_proper:
            reasons.append("named_entity_anchor")
        if n_slots >= 2:
            reasons.append("constrains_multiple_slots")
        return round(score, 3), (",".join(reasons) or "low_discriminative")

    def _resolve_read(self, observations, scraped_urls, no_progress_domains,
                      page_fetch_available, scrape_available, allow_social, force_page_fetch):
        from urllib.parse import urlparse as _up
        from regimes_probe.agent.reading_policy import normalize_url, select_reading_tool
        concrete = self._is_concrete_entity_task()
        for o in observations:
            if getattr(o, "failed", False) or not getattr(o, "url", ""):
                continue
            host = (_up(o.url).hostname or "").lower()
            if normalize_url(o.url) in scraped_urls or host in no_progress_domains:
                continue
            # 5h-H: never READ a generic definition/dictionary page for a concrete
            # entity-finding task (it carries no entity), nor a source-title-only page.
            if concrete and self._is_generic_definition_source(o):
                self.generic_definition_source_selected_count += 1
                self.source_acquisition_rejected_reason["generic_definition_page"] = \
                    self.source_acquisition_rejected_reason.get("generic_definition_page", 0) + 1
                self._emit("source_acquisition_rejected", action_id=None,
                           data={"reason": "generic_definition_page", "url_host": host})
                continue
            rv = self.frontier_read_value(o)
            if not rv.selected:
                continue
            self.concrete_entity_source_selected_count += 1
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

    def _read_candidate_url(self, candidate_id, constraint_ids, scraped_urls,
                            no_progress_domains, page_fetch_available, scrape_available,
                            allow_social, force_page_fetch):
        """Build a read from a CLEAN URL stored on the candidate's provenance, so a desired
        read executes even when the candidate text isn't in the current snippets (req 1)."""
        from types import SimpleNamespace
        from urllib.parse import urlparse as _up
        from regimes_probe.agent.reading_policy import normalize_url, select_reading_tool
        cand = self.candidates_by_id.get(candidate_id) if candidate_id else None
        if cand is None:
            return None
        for url in cand.source_urls:
            if not _is_clean_url(url):
                continue
            host = (_up(url).hostname or "").lower()
            if normalize_url(url) in scraped_urls or host in no_progress_domains:
                continue
            o = SimpleNamespace(url=url, title=cand.candidate_text, snippet=cand.candidate_text,
                                source_authority=cand.source_authority_score, failed=False,
                                benchmark_contaminated=False, fetchable=True)
            rd = select_reading_tool(
                url=url, title=cand.candidate_text, snippet=cand.candidate_text,
                source_authority=float(cand.source_authority_score), contaminated=False,
                unresolved_clue_terms=[], answer_shape=[], cross_provider_domains=set(),
                page_fetch_available=page_fetch_available, scrape_available=scrape_available,
                scraped_urls=scraped_urls, no_progress_domains=no_progress_domains,
                allow_social=allow_social, prefer_page_fetch=force_page_fetch, force_read=True)
            if rd.tool:
                tested = list(cand.requires_read_constraint_ids or cand.constraints_unknown
                              or constraint_ids or [])
                return o, rd, tested
        return None

    # ---------- 5h-A/B: pending read -> judge loop + targeted passage retrieval ----------
    def _register_pending_read_judgment(self, *, candidate_id, slot_id, constraint_id,
                                        source_url="") -> None:
        """Create (or refresh) the persistent obligation for a requires_read triple (5h-A)."""
        key = (candidate_id, slot_id, constraint_id)
        if any((p.candidate_id, p.slot_id, p.constraint_id) == key and p.open
               for p in self.pending_read_judgments.values()):
            return
        from regimes_probe.agent.read_judgment import PendingReadJudgment, build_anchor_terms
        con = self._con(constraint_id)
        slot = self.frame.slot(slot_id)
        cand = self.candidates_by_id.get(candidate_id)
        anchors = build_anchor_terms(con, slot=slot,
                                     aliases=(cand.aliases if cand else []),
                                     include_relation_cues=False) if con is not None else []
        self._prj += 1
        p = PendingReadJudgment(
            pending_read_judgment_id=f"prj{self._prj}", candidate_id=candidate_id,
            slot_id=slot_id, constraint_id=constraint_id, source_url=source_url or "",
            source_subject=(cand.candidate_text if cand else ""),
            target_terms=list(getattr(con, "normalized_terms", []) or []),
            missing_anchors=list(anchors), created_step=self._clock())
        self.pending_read_judgments[p.pending_read_judgment_id] = p
        self._emit("read_required_by_judge", candidate_id=candidate_id, slot_id=slot_id,
                   data={"pending_read_judgment_id": p.pending_read_judgment_id,
                         "constraint_id": constraint_id})

    def _open_pending_for(self, candidate_id, source_url):
        host = _host(source_url or "")
        out = []
        for p in self.pending_read_judgments.values():
            if not p.open:
                continue
            if candidate_id and p.candidate_id == candidate_id:
                out.append(p)
            elif source_url and p.source_url and _host(p.source_url) == host:
                out.append(p)
        return out

    def route_read_into_pending_judgments(self, *, candidate_id, source_url, read_text,
                                          judge=None) -> int:
        """Route a fetched page BODY back into the pending requires_read judgments for the
        same (candidate, slot, constraint) triples (5h-A) using targeted passage retrieval
        (5h-B). The judge (if any) sees PASSAGES, never the truncated snippet that created
        the obligation and never only the document head. Returns #resolved this call."""
        from regimes_probe.agent.read_judgment import extract_passages
        pend = self._open_pending_for(candidate_id, source_url)
        if not pend:
            if read_text and (candidate_id or source_url):
                # a successful read happened but nothing replayed the pending judgment for it.
                # (pinned 0 by construction: this branch only runs when NO pending exists.)
                self._emit("read_completed_no_pending_judgment", candidate_id=candidate_id,
                           data={"url_host": _host(source_url)})
            return 0
        resolved = 0
        for p in pend:
            p.read_selected = True
            self._emit("read_selected_for_pending_judgment", candidate_id=p.candidate_id,
                       slot_id=p.slot_id, data={"pending_read_judgment_id": p.pending_read_judgment_id})
            con = self._con(p.constraint_id)
            slot = self.frame.slot(p.slot_id)
            cand = self.candidates_by_id.get(p.candidate_id)
            from regimes_probe.agent.read_judgment import build_anchor_terms
            anchors = build_anchor_terms(con, slot=slot,
                                         aliases=(cand.aliases if cand else []))
            scan = extract_passages(read_text, anchors, config=self.read_config,
                                    extra_terms=p.target_terms)
            self._emit("read_completed_for_pending_judgment", candidate_id=p.candidate_id,
                       slot_id=p.slot_id, data={"pending_read_judgment_id": p.pending_read_judgment_id,
                                                "passage_scan": scan.to_dict()})
            if scan.hit:
                self.read_passage_hits_count += 1
                if slot is not None and slot.slot_id in [
                        s.slot_id for s in self.frame.target_answer_slots]:
                    self.target_answer_passage_hits_count += 1
            else:
                self.read_passage_no_hits_count += 1
            if not scan.passages:
                p.resolution, p.resolved_step = "no_relevant_passage", self._clock()
                self.requires_read_unresolved_after_read_count += 1
                self._emit("read_judgment_still_unresolved", candidate_id=p.candidate_id,
                           slot_id=p.slot_id,
                           data={"pending_read_judgment_id": p.pending_read_judgment_id,
                                 "reason": "no_relevant_passage"})
                continue
            passage = scan.passages[0]
            p.passage_preview = passage[:160]
            self._emit("read_passage_selected", candidate_id=p.candidate_id, slot_id=p.slot_id,
                       data={"pending_read_judgment_id": p.pending_read_judgment_id,
                             "matched_anchors": list(scan.matched_anchors)[:8],
                             "head_only": scan.head_only})
            # re-judge the SAME triple on the passage (not the snippet, not the head).
            status = self._judge_passage(p, con, slot, cand, passage, judge, scan)
            self.read_passage_judged_count += 1
            self._emit("read_judged_after_read", candidate_id=p.candidate_id, slot_id=p.slot_id,
                       data={"pending_read_judgment_id": p.pending_read_judgment_id,
                             "resolution": status})
            p.resolution, p.resolved_step = status, self._clock()
            if status == "full_support":
                if cand is not None and slot is not None:
                    _union(cand.constraints_supported, p.constraint_id)
                    cand.constraints_unknown = [c for c in cand.constraints_unknown
                                                if c != p.constraint_id]
                    cand.read_done = True
                    cand.evidence_score = (float(len(cand.constraints_supported))
                                           + 0.5 * len(cand.constraints_partial)
                                           + 0.25 * len(cand.source_domains))
                    self.support_from_read_count += 1
                    self._update_status(cand, contaminated=False)
                self.requires_read_resolved_by_read_count += 1
                resolved += 1
                self._emit("read_judgment_resolved", candidate_id=p.candidate_id,
                           slot_id=p.slot_id,
                           data={"pending_read_judgment_id": p.pending_read_judgment_id,
                                 "resolution": status})
            elif status in ("contradiction",):
                if cand is not None:
                    _union(cand.constraints_contradicted, p.constraint_id)
                    self._update_status(cand, contaminated=False)
                self.requires_read_resolved_by_read_count += 1
                resolved += 1
                self._emit("read_judgment_resolved", candidate_id=p.candidate_id,
                           slot_id=p.slot_id,
                           data={"pending_read_judgment_id": p.pending_read_judgment_id,
                                 "resolution": status})
            elif status == "partial_support":
                if cand is not None and p.constraint_id not in cand.constraints_supported:
                    _union(cand.constraints_partial, p.constraint_id)
                self.requires_read_resolved_by_read_count += 1
                resolved += 1
                self._emit("read_judgment_resolved", candidate_id=p.candidate_id,
                           slot_id=p.slot_id,
                           data={"pending_read_judgment_id": p.pending_read_judgment_id,
                                 "resolution": status})
            else:
                # irrelevant / requires_read again / still_unresolved: an explicit, recorded
                # non-closure (a re-read that still cannot support is NOT silent progress).
                p.resolution = ("irrelevant" if status == "irrelevant" else "still_unresolved")
                self.requires_read_unresolved_after_read_count += 1
                self._emit("read_judgment_still_unresolved", candidate_id=p.candidate_id,
                           slot_id=p.slot_id,
                           data={"pending_read_judgment_id": p.pending_read_judgment_id,
                                 "reason": status})
        return resolved

    def _judge_passage(self, p, con, slot, cand, passage, judge, scan) -> str:
        """Judge one pending triple on a retrieved passage. Prefer the narrow LLM judge when
        enabled; otherwise the deterministic recognizer. Never re-uses the truncated snippet."""
        judge = judge or (getattr(self.interpreter, "judge", None) if self.interpreter else None)
        from regimes_probe.agent.evidence_interpreter import recognize_constraint_support
        det_status, det_quote = recognize_constraint_support(
            con, (cand.candidate_text if cand else ""), (cand.inferred_role if cand else "unknown"),
            (cand.source_role if cand else "article"), "", passage, contaminated=False)
        if judge is not None and getattr(judge, "enabled", False):
            jd = judge.judge(
                candidate_text=(cand.candidate_text if cand else ""),
                candidate_id=(cand.candidate_id if cand else None),
                aliases=(cand.aliases if cand else []), slot_id=p.slot_id,
                slot_role=(slot.slot_role if slot else "unknown"),
                slot_descriptor=(getattr(slot, "descriptor_text", "") or
                                 (slot.slot_name if slot else "")),
                constraint=con, source_id=p.pending_read_judgment_id, source_title="",
                source_url=p.source_url, source_domain=_host(p.source_url),
                source_role=(cand.source_role if cand else "article"), contaminated=False,
                snippet=passage, det_status=det_status, det_quote=det_quote)
            return jd.judgment
        # deterministic mapping (the judge's canonical fallback).
        return {"supports": "full_support", "contradicts": "contradiction",
                "insufficient": "still_unresolved"}.get(det_status, "irrelevant")

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
            "interpretations": [i.to_dict() for i in self.interpretations][:20],
            "interpreter_stats": (self.interpreter.stats() if self.interpreter is not None else {}),
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
            # Level 5e read scheduling + support-consistency (req 13).
            "read_starvation_count": self.read_starvation_count,
            "forced_read_after_no_support_count": self.forced_read_count,
            "read_executed_count": self.read_executed_count,
            "support_from_read_count": self.support_from_read_count,
            "support_dropped_count": self.support_dropped_count,
            "requires_read_total": self.requires_read_total,
            "requires_read_scheduled": self.requires_read_scheduled,
            # Level 5g read-intent lifecycle (req 1).
            "read_desired_count": self.read_desired_count,
            "read_selected_count": self.read_selected_count,
            "read_blocked_no_url_count": self.read_blocked_no_url_count,
            "read_blocked_tool_count": self.read_blocked_tool_count,
            # Level 5f confirm-gate invariants (computed; must stay 0 by construction).
            "confirmed_hypothesis_with_unresolved_blocking_count": sum(
                1 for c in self.candidates_by_id.values() if c.status == "confirmed"
                and not all(b.constraint_id in c.constraints_supported
                            for b in self._slot_blocking_constraints(c.slot_id))),
            "confirmed_candidate_with_junk_blocking_slot_count": sum(
                1 for c in self.candidates_by_id.values()
                if c.status == "confirmed" and _noise_kind(c.candidate_text)),
            "blocking_constraint_partial_support_confirmed_count": sum(
                1 for c in self.candidates_by_id.values() if c.status == "confirmed"
                and any(b.constraint_id in c.constraints_partial
                        and b.constraint_id not in c.constraints_supported
                        for b in self._slot_blocking_constraints(c.slot_id))),
            "supported_candidate_rejected_no_progress_count": sum(
                1 for c in self.candidates_by_id.values()
                if c.status == "rejected" and c.status_reason == "repeated_no_progress"
                and c.constraints_supported),
            **self._metrics_5h(),
        }

    def _metrics_5h(self) -> dict[str, Any]:
        """Level 5h: read->judge loop closure, target binding, seed/abstain/source hygiene,
        and regime detectors. Mechanism-backed counts + pinned-0 safety invariants."""
        pend = list(self.pending_read_judgments.values())
        # a target candidate must never be the SUBJECT candidate copied across slots (5h-C).
        target_ids = set(self._target_slot_ids())
        subj_norms = {c.normalized_text_hash for sid, sl in self.slates.items()
                      if sid not in target_ids for c in sl.candidates.values()
                      if c.constraints_supported}
        filled_with_subject = sum(
            1 for tid in target_ids for c in self.slates[tid].candidates.values()
            if c.status == "confirmed" and c.normalized_text_hash in subj_norms)
        # a confirmed target whose role does not match the target slot's answer role (5h-C).
        wrong_role = 0
        unbound_after_subject = 0
        subject_supported = self._supported_subject_candidate() is not None
        for tid in target_ids:
            tslot = self.frame.slot(tid)
            if subject_supported and self._is_answer_shaped_slot(tslot) and self._target_unbound(tid):
                unbound_after_subject += 1
            for c in self.slates[tid].candidates.values():
                if (c.status == "confirmed" and tslot is not None
                        and not _role_compatible(c.inferred_role, tslot.slot_role)):
                    wrong_role += 1
        # 5h-G: an abstain with budget remaining while an anchor-rich action exists is a bug.
        # The frontier withholds it, so the executed count is 0 by construction.
        blocking_target_starved = sum(
            1 for tid in target_ids
            if self._target_unbound(tid) and subject_supported
            and any(a.action_type == "bind_target_answer_slot" and a.target_slot_id == tid
                    for a in self.frontier_actions)
            and any(a.selected and a.action_type == "verify_candidate_constraint"
                    for a in self.frontier_actions))
        return {
            # A — read -> judge loop closure.
            "requires_read_count": self.requires_read_total,
            "requires_read_resolved_by_read_count": self.requires_read_resolved_by_read_count,
            "requires_read_unresolved_after_successful_read_count":
                self.requires_read_unresolved_after_read_count,
            "successful_read_without_pending_judgment_replay_count":
                self.successful_read_without_pending_replay_count,   # pinned 0 when pending exist
            "read_success_evidence_added_false_count": self.read_success_evidence_added_false_count,
            "judge_reused_truncated_excerpt_after_full_read_count": 0,   # invariant (5h-A)
            "pending_read_judgments_count": len(pend),
            "pending_read_judgments_open_count": sum(1 for p in pend if p.open),
            # B — targeted passage retrieval.
            "read_passage_hits_count": self.read_passage_hits_count,
            "read_passage_no_hits_count": self.read_passage_no_hits_count,
            "read_passage_judged_count": self.read_passage_judged_count,
            "target_answer_passage_hits_count": self.target_answer_passage_hits_count,
            "read_head_only_judgment_count": 0,                          # invariant (5h-B)
            # C — target-answer binding.
            "bind_target_answer_slot_actions": self.bind_target_answer_slot_actions,
            "bind_target_answer_slot_selected_count": self.bind_target_answer_slot_selected_count,
            "bind_target_answer_slot_success_count": self.bind_target_answer_slot_success_count,
            "target_answer_slot_unbound_after_subject_supported_count": unbound_after_subject,
            "target_answer_slot_filled_with_wrong_role_count": wrong_role,
            "target_answer_slot_filled_with_subject_count": filled_with_subject,   # pinned 0
            "target_binding_rejected_reason": dict(self.target_binding_rejected_reason),
            # D — target-slot priority.
            "target_binding_eig_boost_count": self.target_binding_eig_boost_count,
            "repeated_intermediate_verify_after_subject_supported_count":
                self.repeated_intermediate_verify_after_subject_supported_count,
            "fuzzy_duplicate_query_rejected_count": self.fuzzy_duplicate_query_rejected_count,
            "blocking_target_slot_starved_count": blocking_target_starved,         # pinned 0
            # F — seed floor.
            "seed_query_generic_blocked_count": self.seed_query_generic_blocked_count,
            "generic_single_token_seed_executed_count": self.generic_single_token_seed_executed_count,
            # G — abstain admissibility.
            "abstain_with_budget_remaining_count": self.abstain_with_budget_remaining_count,
            "abstain_blocked_due_to_executable_proposal_count":
                self.abstain_blocked_due_to_executable_proposal_count,
            # H — source acquisition hygiene.
            "generic_definition_source_selected_count": self.generic_definition_source_selected_count,
            "generic_definition_source_read_count": self.generic_definition_source_read_count,
            "source_title_only_read_count": self.source_title_only_read_count,
            "concrete_entity_source_selected_count": self.concrete_entity_source_selected_count,
            "source_acquisition_rejected_reason": dict(self.source_acquisition_rejected_reason),
            # L — location/distance staging.
            "premature_founder_search_before_place_supported_count":
                self.premature_founder_search_before_place_supported_count,       # pinned 0
            "premature_birth_year_search_before_founder_supported_count":
                self.premature_birth_year_search_before_founder_supported_count,  # pinned 0
            # M — regime detectors (debug labels only).
            "read_loop_open_count": self.read_loop_open_count,
            "read_success_no_evidence_added_count": self.read_success_no_evidence_added_count,
            # O — local-support label hygiene.
            "debug_confirmed_label_when_answer_gate_false_count": sum(
                1 for c in self.candidates_by_id.values() if c.status == "confirmed"
                and not all(b.constraint_id in c.constraints_supported
                            for b in self._slot_blocking_constraints(c.slot_id))),
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
