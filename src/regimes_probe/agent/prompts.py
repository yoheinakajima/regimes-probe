"""Prompt registry with version + content-hash pinning.

Every prompt-like artifact the agent *would* use on a live run is pinned here so
that (a) the same-conditions validator can assert prompts are identical across
the compared conditions, and (b) the run manifest records exactly which prompt
bytes were in force. v0's offline path uses deterministic answerers and an
exact-match grader, so these prompts are not yet sent to any model — but they are
pinned now so the first live run is auditable from the start.

Each :class:`Prompt` declares whether it is ``allowed_to_vary`` across
conditions. The answerer and judge prompts must NOT vary in a headline
comparison; the LLM query-generation prompt MAY vary (it is a Level 2 ablation
knob).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    content: str
    intended_use: str
    allowed_to_vary: bool

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()[:16]

    def fingerprint(self) -> str:
        """Stable ``name@version#hash8`` identifier used in ConditionSpec/manifest."""
        return f"{self.name}@{self.version}#{self.content_hash[:8]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "content_hash": self.content_hash,
            "fingerprint": self.fingerprint(),
            "intended_use": self.intended_use,
            "allowed_to_vary": self.allowed_to_vary,
        }


# ---------------------------------------------------------------------------
# The pinned prompts. Content is intentionally minimal but REAL: editing it
# changes the hash, which the same-conditions validator and manifest will catch.
# ---------------------------------------------------------------------------
_ANSWERER_V1 = (
    "You are a careful web-research answerer. Using ONLY the provided evidence, "
    "give the single most-supported short answer. If the evidence does not "
    "support an answer, respond exactly with ABSTAIN. Do not use prior knowledge "
    "beyond the evidence. Return only the answer text."
)

_CLOSED_BOOK_V1 = (
    "You are answering WITHOUT any tools or search. Using only your own internal "
    "knowledge, give the single best short answer, or respond exactly with "
    "ABSTAIN if you are not confident. Return only the answer text."
)

_JUDGE_V1 = (
    "You are a strict grader. Given a question, a gold answer, and a candidate "
    "answer, respond CORRECT if the candidate matches the gold answer's meaning, "
    "otherwise INCORRECT. Ignore formatting and case. Return only one word."
)

_QUERY_LLM_V1 = (
    "Rewrite the question into a single effective web-search query. Keep salient "
    "entities and add precise terms. Return only the query string."
)

_TASK_FRAME_PARSER_V1 = (
    "You parse a research question into a constraint-satisfaction TASK FRAME. "
    "These questions are constraint-satisfaction problems over latent (hidden) "
    "variables: a target answer plus intermediate variables, linked by constraints.\n"
    "\n"
    "RULES — read carefully:\n"
    "- Do NOT answer the question. Do NOT solve it. Do NOT guess any entity, name, "
    "date, or value. You only PARSE.\n"
    "- Output ONLY structure: slots and constraints derived from the question text.\n"
    "- Distinguish KNOWN CONTEXT TERMS (entities/values GIVEN in the question, e.g. "
    "an organization that released a report) from UNKNOWN VARIABLES (what must be "
    "found).\n"
    "- The TARGET slot is the single thing being ASKED FOR (the interrogative head: "
    "'which series…' -> the series; '…in which year' -> the year; 'who…' -> a "
    "person). It is NOT every entity mentioned.\n"
    "- INTERMEDIATE slots are things that must be found BEFORE the answer.\n"
    "- Attach each clue/clause to the slot it constrains via applies_to.\n"
    "- Preserve multi-hop dependencies in dependency_edges (intermediate -> target).\n"
    "- Do NOT promote a source/organization given in the question to a TARGET slot "
    "unless the question explicitly asks for that source/organization.\n"
    "- Never put a concrete answer (a name/date/value not present in the question) "
    "into any slot_name or term.\n"
    "\n"
    "GENERIC EXAMPLES (illustrative shapes only — do not copy, do not answer):\n"
    "- 'Which TV series starred an actor born in <place>…' -> target = series "
    "(title_or_work); intermediate = the actor (person); constraints attach the "
    "birthplace/era to the actor, the actor to the series.\n"
    "- 'A restaurant near a hotel and a museum, founded by a chef born in which "
    "year' -> target = the birth year (date_or_time); intermediate = restaurant, "
    "hotel, museum (organization/location) and the founder (person); distance/"
    "location constraints attach to the restaurant.\n"
    "- 'A report by <organization>; foreword by a person; introduction by a person; "
    "who is the introduction author' -> known context = the organization; "
    "intermediate = report (title_or_work), foreword author (person); target = the "
    "introduction author (person).\n"
    "- 'A paper using a census sample asks which journal published it' -> target = "
    "the journal/publication (publication_or_source); intermediate = the paper "
    "(title_or_work), authors (person), country/sample as constraints.\n"
    "\n"
    "OUTPUT: a single JSON object with keys target_answer_slots, latent_slots, "
    "constraints, dependency_edges, known_context_terms, unresolved_slots, "
    "answer_shape_hints, source_requirements, parse_quality. Each slot has slot_id, "
    "slot_name, slot_role, is_target_answer_slot, is_intermediate_slot, depends_on, "
    "expected_evidence_type. Each constraint has constraint_id, text_span, "
    "normalized_terms, constraint_type, applies_to, specificity_score, "
    "discriminative_score, status. slot_role is one of person, organization, "
    "location, title_or_work, publication_or_source, event, concept, date_or_time, "
    "number, unknown. Return ONLY the JSON object, no prose."
)

_TASK_FRAME_PARSER_V2 = (
    "You parse a research question into an OPEN-WORLD, OPERATIONAL task frame. "
    "These questions are constraint-satisfaction problems over latent (hidden) "
    "variables: a target answer plus intermediate variables, linked by constraints. "
    "Your job is to expose what must be FOUND and how each piece can be TESTED — not "
    "to answer.\n"
    "\n"
    "PRINCIPLES:\n"
    "- Do NOT answer or solve the question. Do NOT guess any entity, name, date, or "
    "value. Parse only.\n"
    "- Parse into variables (slots), constraints, and TESTABLE CLAIMS.\n"
    "- Use NATURAL semantic labels freely (e.g. 'employment_relation', 'authorship + "
    "educational background', 'distance_and_temporal_attribute'). There is NO fixed "
    "label list — pick the most descriptive label and add semantic_facets.\n"
    "- For EACH constraint also give OPERATIONAL AFFORDANCES: how it can be tested — "
    "what evidence would support/refute it, what action tests it, and suggested "
    "search query templates.\n"
    "- Distinguish KNOWN CONTEXT (given in the question) from UNKNOWN VARIABLES "
    "(what must be found). Never promote a given source/org to a target slot unless "
    "the question explicitly asks for it.\n"
    "- The TARGET slot is the single thing being ASKED FOR (the interrogative head). "
    "Intermediate slots are things that must be found before the answer.\n"
    "- Mark which constraints BLOCK answer support if unresolved (required + high "
    "priority), and which answer slot ids each constraint supports.\n"
    "- Preserve multi-hop dependencies in dependency_edges (intermediate -> target). "
    "Never put a concrete answer (a value absent from the question) into any field.\n"
    "\n"
    "OUTPUT: one JSON object with keys target_answer_slots, latent_slots, "
    "constraints, dependency_edges, known_context_terms, unresolved_slots, "
    "answer_shape_hints, source_requirements, parse_quality.\n"
    "Each slot: slot_id, slot_name, slot_role (person/organization/location/"
    "title_or_work/publication_or_source/event/concept/date_or_time/number/unknown), "
    "is_target_answer_slot, is_intermediate_slot, depends_on, expected_evidence_type.\n"
    "Each constraint: constraint_id, text_span, semantic_label (free-form), "
    "semantic_facets (list, free-form; common ones: identity, attribute, relation, "
    "temporal, spatial, quantitative, authorship, source, membership, biographical, "
    "title_or_work, comparison, distance, answer_shape), applies_to (slot ids), "
    "required (bool), priority (high/medium/low), testable_claim, evidence_needed, "
    "how_to_test, suggested_query_templates (list), supports_answer_slot_ids (list), "
    "blocks_answer_if_unresolved (bool), source_quote_or_span, parser_confidence, "
    "and OPTIONALLY affordances (subset of: can_search, can_verify, can_read, "
    "can_compare, can_bind_slot, can_support_answer, can_block_answer).\n"
    "GENERIC SHAPES (do not copy, do not answer): 'which TV series starred an actor "
    "born in X' -> target=series, intermediate=actor; 'a restaurant near a hotel/"
    "museum, founder born in which year' -> target=birth year, intermediate="
    "restaurant/hotel/museum/founder; 'report by ORG, foreword by P1, introduction "
    "by P2 — who wrote the introduction' -> known context=ORG, intermediate=report/"
    "P1, target=P2; 'a paper using a census sample — which journal published it' -> "
    "target=journal, intermediate=paper/authors, census/sample as constraints.\n"
    "Return ONLY the JSON object, no prose."
)

_VARIABLES_BLOCK = (
    "\nVARIABLES vs CONSTANTS (critical — avoids both leakage and over-rejection):\n"
    "- Target and intermediate slots are UNKNOWN VARIABLES to be found. Their "
    "slot_name is a DESCRIPTOR of the unknown (e.g. '90s TV series', 'founder full "
    "name', 'person who wrote the introduction', 'hotel originally opened in 1955'). "
    "Reuse the question's wording freely — a descriptor is NOT an answer and is NOT "
    "leakage.\n"
    "- Do NOT put a concrete value in a target slot. Leave bound_value empty for "
    "targets; candidate values are produced LATER from evidence, never during "
    "parsing.\n"
    "- Constants GIVEN in the question (a specific named org/place/source/date such "
    "as 'WHO', 'Tennessee', 'New Mexico', 'Gracie Award') go in known_context_terms "
    "or as constraints — NEVER as a target answer slot, unless the question asks for "
    "that exact constant type.\n"
    "- Optionally set per slot: slot_status (unbound_variable | known_constant), "
    "descriptor_text, bound_value (leave empty), evidence_required_to_bind.\n")
_TASK_FRAME_PARSER_V3 = _TASK_FRAME_PARSER_V2.replace(
    "OUTPUT: one JSON object", _VARIABLES_BLOCK + "OUTPUT: one JSON object")

PROMPTS: dict[str, Prompt] = {
    "answerer": Prompt(
        name="answerer", version="v1", content=_ANSWERER_V1,
        intended_use="evidence-grounded short-answer extraction (search conditions)",
        allowed_to_vary=False,
    ),
    "closed_book": Prompt(
        name="closed_book", version="v1", content=_CLOSED_BOOK_V1,
        intended_use="intrinsic-knowledge answer with no tools (closed_book baseline)",
        allowed_to_vary=False,
    ),
    "judge": Prompt(
        name="judge", version="v1", content=_JUDGE_V1,
        intended_use="optional LLM judge for free-form grading",
        allowed_to_vary=False,
    ),
    "query_llm": Prompt(
        name="query_llm", version="v1", content=_QUERY_LLM_V1,
        intended_use="LLM query generation (Level 2; replaceable by templates)",
        allowed_to_vary=True,
    ),
    "task_frame_parser": Prompt(
        name="task_frame_parser", version="v3", content=_TASK_FRAME_PARSER_V3,
        intended_use="open-world operational task-frame parsing (Level 4b/4e; never answers)",
        allowed_to_vary=True,
    ),
}


def get(name: str) -> Prompt:
    if name not in PROMPTS:
        raise KeyError(f"unknown prompt: {name!r}; known: {sorted(PROMPTS)}")
    return PROMPTS[name]


def fingerprint(name: str) -> str:
    return get(name).fingerprint()


def registry_dict() -> dict[str, Any]:
    """Answer-free, secret-free snapshot of the prompt registry (for manifests)."""
    return {name: p.to_dict() for name, p in PROMPTS.items()}
