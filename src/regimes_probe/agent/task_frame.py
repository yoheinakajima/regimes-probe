"""Generic task-frame representation for constraint-satisfaction search tasks.

BrowseComp-style questions are constraint-satisfaction problems over latent
variables: a target answer slot, intermediate slots, and constraints linking them.
This module parses a question into a generic :class:`TaskFrame` (deterministic v0,
gold-free, no model) so the agent can reason about *which hidden variable each
search/read is meant to resolve* rather than chasing high-frequency entities.

Not BrowseComp-specific: the slot roles, constraint types, and inference rules are
generic. An LLM parser could be layered later (cached + prompt-versioned), but v0
is heuristic and always available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from regimes_probe.agent.clue_resolution import (
    ROLES, _BROAD_LOCATIONS, _norm, classify_entity_role, infer_target_roles)
from regimes_probe.policy.query_decomposition import (
    extract_clues, split_clauses, _tokens, _is_rare)

#: Constraint types (generic).
CONSTRAINT_TYPES = ("identity", "attribute", "relation", "temporal", "location",
                    "distance", "source", "authorship", "membership", "title_work",
                    "answer_shape")

#: Role-trigger nouns → the slot role they imply (generic, not topical).
_ROLE_TRIGGERS: dict[str, str] = {}
for _role, _words in {
    "organization": ("restaurant", "hotel", "museum", "university", "college",
                     "school", "institute", "foundation", "department", "association",
                     "agency", "ministry", "monastery", "church", "company",
                     "corporation", "organization", "society", "team", "club",
                     "hospital", "bureau", "council", "committee", "league"),
    "person": ("founder", "author", "writer", "journalist", "ceo", "director",
               "president", "inventor", "scientist", "artist", "actor", "actress",
               "musician", "leader", "owner", "chairman", "monk", "soldier",
               "architect", "engineer", "composer", "painter", "poet", "novelist",
               "chef", "cook", "baker", "bartender", "sommelier", "photographer",
               "person", "individual"),
    "location": ("city", "town", "country", "village", "monument", "building",
                 "street", "river", "mountain", "island", "region", "state",
                 "province", "county", "address", "neighborhood", "park", "place"),
    "title_or_work": ("series", "film", "movie", "show", "book", "novel", "manga",
                      "album", "song", "paper", "report", "article", "painting",
                      "sculpture", "play", "poem", "story", "episode", "comic",
                      "documentary", "anime"),
    "publication_or_source": ("journal", "magazine", "newspaper", "press",
                              "publisher", "publication", "website", "platform"),
    "date_or_time": ("year", "date", "century", "decade", "day", "month"),
    "number": ("population", "percentage", "distance", "height", "length", "age",
               "count", "amount", "quantity", "total", "number"),
    "event": ("war", "battle", "festival", "competition", "conference", "ceremony",
              "election", "tournament", "exhibition"),
}.items():
    for _w in _words:
        _ROLE_TRIGGERS[_w] = _role

_TEMPORAL = ("year", "born", "founded", "published", "released", "established",
             "built", "completed", "died", "created", "date", "century", "when",
             "between")
_SOURCE_CUES = ("released by", "published by", "issued by", "produced by",
                "report by", "according to", "from the")
_AUTHOR_CUES = ("authored by", "written by", "author of", "by the author")
_DISTANCE_CUES = ("near", "within", "miles", "kilometers", "km", "distance", "close to")
_MEMBERSHIP_CUES = ("member of", "part of", "belongs to", "affiliated with")
_LOC_PREP = ("in", "near", "at", "from", "around", "outside", "inside", "located")
_ACRONYM = re.compile(r"\b[A-Z]{2,6}\b")
_WH = re.compile(r"\b(who|whom|whose|where|when|which|what|how)\b")


def _interrogative_target_role(question: str) -> Optional[str]:
    """Primary target role from the interrogative head noun.

    The question asks for *one thing*; the wh-phrase names it. "Which TV series…"
    → the series (``title_or_work``); "…in which year" → ``date_or_time``; "who…"
    → ``person``. Everything else describing it is an intermediate slot/constraint,
    not the target. Generic (no topical rules); returns ``None`` if undetectable.
    """
    ql = question.lower()
    m = _WH.search(ql)
    if not m:
        return None
    wh = m.group(1)
    if wh in ("who", "whom", "whose"):
        return "person"
    if wh == "where":
        return "location"
    if wh == "when":
        return "date_or_time"
    words = re.findall(r"[a-z]+", ql[m.end():])
    if wh == "how":
        return "number" if {"many", "much"} & set(words[:2]) else None
    # which / what: scan forward to the first role-trigger noun (the head noun).
    for w in words[:8]:
        if w in _ROLE_TRIGGERS:
            return _ROLE_TRIGGERS[w]
    return None


@dataclass
class Slot:
    slot_id: str
    slot_name: str
    slot_role: str                       # one of ROLES + "number"
    is_target_answer_slot: bool = False
    is_intermediate_slot: bool = False
    depends_on: list[str] = field(default_factory=list)
    expected_evidence_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"slot_id": self.slot_id, "slot_name": self.slot_name,
                "slot_role": self.slot_role,
                "is_target_answer_slot": self.is_target_answer_slot,
                "is_intermediate_slot": self.is_intermediate_slot,
                "depends_on": list(self.depends_on),
                "expected_evidence_type": self.expected_evidence_type}


@dataclass
class Constraint:
    constraint_id: str
    text_span: str
    normalized_terms: list[str]
    constraint_type: str
    applies_to: list[str] = field(default_factory=list)        # slot ids
    specificity_score: float = 0.0
    discriminative_score: float = 0.0
    status: str = "unresolved"           # resolved | unresolved | contradicted
    supporting_evidence_ids: list[str] = field(default_factory=list)
    contradicting_evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self, *, preview: int = 160) -> dict[str, Any]:
        span = self.text_span if len(self.text_span) <= preview else self.text_span[:preview - 1] + "…"
        return {"constraint_id": self.constraint_id, "text_span": span,
                "normalized_terms": self.normalized_terms[:8],
                "constraint_type": self.constraint_type, "applies_to": list(self.applies_to),
                "specificity_score": round(self.specificity_score, 3),
                "discriminative_score": round(self.discriminative_score, 3),
                "status": self.status,
                "supporting_evidence_ids": list(self.supporting_evidence_ids)[:8],
                "contradicting_evidence_ids": list(self.contradicting_evidence_ids)[:8]}


@dataclass
class TaskFrame:
    item_id: str
    target_answer_slots: list[Slot] = field(default_factory=list)
    latent_slots: list[Slot] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    dependency_edges: list[tuple[str, str]] = field(default_factory=list)
    known_context_terms: list[str] = field(default_factory=list)
    answer_shape_hints: list[str] = field(default_factory=list)
    source_requirements: list[str] = field(default_factory=list)
    parse_quality: float = 0.0

    @property
    def all_slots(self) -> list[Slot]:
        return self.target_answer_slots + self.latent_slots

    def slot(self, slot_id: str) -> Optional[Slot]:
        return next((s for s in self.all_slots if s.slot_id == slot_id), None)

    @property
    def unresolved_constraints(self) -> list[Constraint]:
        return [c for c in self.constraints if c.status == "unresolved"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "target_answer_slots": [s.to_dict() for s in self.target_answer_slots],
            "latent_slots": [s.to_dict() for s in self.latent_slots],
            "constraints": [c.to_dict() for c in self.constraints],
            "dependency_edges": [list(e) for e in self.dependency_edges],
            "known_context_terms": list(self.known_context_terms)[:12],
            "answer_shape_hints": list(self.answer_shape_hints),
            "source_requirements": list(self.source_requirements),
            "parse_quality": round(self.parse_quality, 3),
        }


def _constraint_type(clause: str) -> str:
    cl = clause.lower()
    if any(c in cl for c in _AUTHOR_CUES):
        return "authorship"
    if any(c in cl for c in _SOURCE_CUES):
        return "source"
    if any(c in cl for c in _DISTANCE_CUES):
        return "distance"
    if any(c in cl for c in _MEMBERSHIP_CUES):
        return "membership"
    if any(w in cl.split() for w in _TEMPORAL) or re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", cl):
        return "temporal"
    toks = cl.split()
    if any(t in _LOC_PREP for t in toks) and re.search(r"\b[A-Z][a-z]+\b", clause):
        return "location"
    if any(_ROLE_TRIGGERS.get(t) == "title_or_work" for t in toks):
        return "title_work"
    if '"' in clause or "named" in cl or "called" in cl:
        return "identity"
    if any(_ROLE_TRIGGERS.get(t) in ("person", "organization", "location") for t in toks):
        return "relation"
    return "attribute"


def _specificity(terms: list[str], clause: str) -> float:
    rare = sum(1 for t in terms if _is_rare(t))
    caps = len(re.findall(r"\b[A-Z][a-zA-Z]+\b", clause))
    quoted = clause.count('"') // 2
    return float(rare * 1.0 + caps * 0.5 + quoted * 1.0 + (0.5 if len(terms) >= 3 else 0.0))


def parse_task_frame(item_id: str, question: str) -> TaskFrame:
    """Deterministically parse a question into a generic task frame (no gold)."""
    clues = extract_clues(question)
    target_roles, _ = infer_target_roles(question)
    # The interrogative head noun names the single thing being asked for; prefer it
    # so secondary inferred roles (e.g. an actor in "which series featured an actor")
    # become intermediate slots rather than competing target answers.
    primary = _interrogative_target_role(question)
    if primary is not None:
        target_roles = [primary]
    ql = question.lower()
    tokens = _tokens(ql)

    frame = TaskFrame(item_id=item_id)
    sid = 0

    # Target answer slot(s): one per inferred target role; name it from a matching
    # role-trigger noun in the question if present.
    used_roles: set[str] = set()
    for role in target_roles:
        name = next((t for t in tokens if _ROLE_TRIGGERS.get(t) == role), role)
        frame.target_answer_slots.append(Slot(
            slot_id=f"s{sid}", slot_name=name, slot_role=role,
            is_target_answer_slot=True, expected_evidence_type=role))
        used_roles.add(role)
        sid += 1

    # Intermediate (latent) slots: other role-trigger nouns present in the question.
    seen_trigger_roles: set[str] = set()
    for t in tokens:
        role = _ROLE_TRIGGERS.get(t)
        if role and role not in used_roles and role not in seen_trigger_roles:
            frame.latent_slots.append(Slot(
                slot_id=f"s{sid}", slot_name=t, slot_role=role,
                is_intermediate_slot=True, expected_evidence_type=role))
            seen_trigger_roles.add(role)
            sid += 1

    # Known context terms (GIVEN in the question): named entities + quoted phrases +
    # acronyms (e.g. WHO). These are constraints/context, never a target candidate.
    acronyms = [a for a in _ACRONYM.findall(question) if a not in ("PDF",)]
    frame.known_context_terms = list(dict.fromkeys(clues.entities + clues.quoted + acronyms))

    # Answer-shape + source requirements.
    from regimes_probe.policy.query_decomposition import answer_shape_terms
    frame.answer_shape_hints = answer_shape_terms(question)
    frame.source_requirements = list(clues.source_hints)

    # Constraints: one per clause, attached to the best-matching slot.
    cid = 0
    for clause in split_clauses(question):
        terms = [t for t in _tokens(clause) if len(t) >= 3][:10]
        if not terms:
            continue
        ctype = _constraint_type(clause)
        # attach to a slot whose role-trigger noun appears in the clause, else target.
        applies = []
        for s in frame.all_slots:
            if s.slot_name.lower() in clause.lower():
                applies.append(s.slot_id)
        if not applies and frame.target_answer_slots:
            applies = [frame.target_answer_slots[0].slot_id]
        spec = _specificity(terms, clause)
        frame.constraints.append(Constraint(
            constraint_id=f"c{cid}", text_span=clause.strip(), normalized_terms=terms,
            constraint_type=ctype, applies_to=applies,
            specificity_score=spec, discriminative_score=spec))
        cid += 1

    # Dependency edges: intermediate slots feed the target (generic chain).
    for tgt in frame.target_answer_slots:
        for lat in frame.latent_slots:
            frame.dependency_edges.append((lat.slot_id, tgt.slot_id))
            tgt.depends_on.append(lat.slot_id)

    # Parse quality: do we have a target slot + at least one specific constraint?
    has_specific = any(c.specificity_score >= 1.0 for c in frame.constraints)
    frame.parse_quality = round(
        0.5 * bool(frame.target_answer_slots) + 0.3 * has_specific
        + 0.2 * bool(frame.latent_slots), 3)
    return frame
