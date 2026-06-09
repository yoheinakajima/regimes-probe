"""Optional cached/replayable LLM task-frame parser (Level 4).

The deterministic parser (`task_frame.parse_task_frame`) proved the
constraint-satisfaction architecture, but its parse *quality* is the limiting
factor: it mis-types slots, attaches constraints to the wrong slot, and promotes
source/context entities to targets. This module adds an **optional** LLM parser
that emits the **same** `TaskFrame` schema — it produces task *state* only and is
forbidden from answering — validated against a schema and **falling back to the
deterministic parser** whenever the output is invalid or low quality.

Discipline (mirrors the gated LLM query policy in `docs/QUERY_POLICY.md`):
- Gold-free / answer-free: the parser sees only the question text. It must not
  receive or emit final-answer text; nothing here is written to policy memory.
- Cached + replayable: every call goes through a parser cache keyed by the prompt
  hash + model + question. A replay (no `model_fn`, or `replay_only`) never calls
  a model — a cache miss falls back to the deterministic parser.
- Schema-validated + fallback-safe: `validate_payload` enforces the contract; any
  failure records a `fallback_reason` and uses the deterministic frame.

This module never performs network I/O itself; a `model_fn` (if any) is injected
by the live runner. Tests inject a stub `model_fn` or a pre-seeded cache.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from regimes_probe.agent import prompts
from regimes_probe.agent.affordances import (
    AFFORDANCES, COMMON_FACETS, derive_affordances, normalize_facets, novel_facets)
from regimes_probe.agent.clue_resolution import ROLES, _norm
from regimes_probe.agent.task_frame import (
    SLOT_STATUSES, Constraint, Slot, TaskFrame, _ROLE_TRIGGERS,
    _interrogative_target_role, parse_task_frame)

#: Slot roles map to evidence types — a small standardized set. Unknown roles are
#: coerced to "unknown" with a WARNING (never a hard rejection).
VALID_ROLES: frozenset[str] = frozenset(ROLES) | {"number"}
#: A parsed frame below this aggregate quality is rejected (deterministic fallback).
MIN_LLM_PARSE_QUALITY: float = 0.5

#: Minimal required fields. Constraint SEMANTICS are open-world: we never require a
#: particular constraint_type/facet, only that the object is operationally usable.
_REQUIRED_SLOT_FIELDS = ("slot_id", "slot_name", "slot_role")
_ANSWER_KEYS = ("answer", "final_answer", "solution", "gold", "gold_answer", "result")
_WORD = re.compile(r"[A-Za-z0-9]+")


def _as_list(v: Any) -> list:
    """Normalize a scalar/None/list into a list (e.g. applies_to: 's0' -> ['s0'])."""
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


#: key-pairs the LLM might use for a dependency edge object (prerequisite -> dependent).
_EDGE_KEYS = (("from", "to"), ("source", "target"), ("parent", "child"),
              ("prerequisite", "dependent"), ("src", "dst"), ("head", "tail"),
              ("depends_on", "slot"))


def _parse_edge(edge: Any) -> Optional[tuple[str, str]]:
    """Extract a (prerequisite, dependent) id pair from any reasonable edge shape.

    The LLM may emit edges as ``["I1","T1"]`` OR as an object
    ``{"from":"I1","to":"T1"}`` (or source/target, parent/child, …). Returns the two
    raw ids, or ``None`` if the shape is unrecognizable."""
    if isinstance(edge, (list, tuple)) and len(edge) == 2:
        return (str(edge[0]), str(edge[1]))
    if isinstance(edge, dict):
        for a, b in _EDGE_KEYS:
            if a in edge and b in edge:
                return (str(edge[a]), str(edge[b]))
        vals = list(edge.values())
        if len(vals) == 2:
            return (str(vals[0]), str(vals[1]))
    return None


def _raw_slot_ids(payload: Any) -> tuple[list[str], set[str]]:
    """All unique raw slot ids the parser emitted, in target-then-latent order."""
    ordered: list[str] = []
    if not isinstance(payload, dict):
        return ordered, set()
    for s in (payload.get("target_answer_slots") or []) + (payload.get("latent_slots") or []):
        if isinstance(s, dict):
            sid = s.get("slot_id")
            if isinstance(sid, str) and sid and sid not in ordered:
                ordered.append(sid)
    return ordered, set(ordered)


def _derive_raw_edges(payload: dict) -> list[tuple[str, str]]:
    """Edges implied by per-slot ``depends_on`` (prerequisite -> dependent), raw ids."""
    edges: list[tuple[str, str]] = []
    for s in (payload.get("target_answer_slots") or []) + (payload.get("latent_slots") or []):
        if isinstance(s, dict) and s.get("slot_id"):
            for p in _as_list(s.get("depends_on")):
                edges.append((str(p), str(s["slot_id"])))
    return edges


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _q_tokens(question: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(question)}


# --------------------------------------------------------------------------- cache
class ParserCache:
    """Equivalent parser cache: input_hash -> raw model output string.

    Optionally file-backed (JSON) so a live run's parses are replayable without a
    model. Pure in-memory by default (tests)."""

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path
        self._store: dict[str, str] = {}
        if path:
            try:
                import os
                if os.path.exists(path):
                    with open(path, encoding="utf-8") as fh:
                        self._store = dict(json.load(fh))
            except Exception:
                self._store = {}

    @property
    def path(self) -> Optional[str]:
        return self._path

    def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    def put(self, key: str, value: str) -> None:
        self._store[key] = value
        if self._path:
            try:
                with open(self._path, "w", encoding="utf-8") as fh:
                    json.dump(self._store, fh, sort_keys=True)
            except Exception:
                pass

    def __contains__(self, key: str) -> bool:
        return key in self._store


# --------------------------------------------------------------------------- meta
@dataclass
class ParseMeta:
    """Answer-free provenance for one task-frame parse (logged + replayable)."""

    parser_used: str = "deterministic"      # deterministic | llm
    prompt_version: str = ""
    prompt_hash: str = ""
    model: str = ""
    input_hash: str = ""
    output_hash: str = ""
    parse_quality: float = 0.0
    parse_quality_components: dict[str, float] = field(default_factory=dict)
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    fallback_reason: str = ""
    cache_hit: bool = False
    id_mapping: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parser_used": self.parser_used,
            "prompt_version": self.prompt_version,
            "prompt_hash": self.prompt_hash,
            "model": self.model,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "parse_quality": round(self.parse_quality, 3),
            "parse_quality_components": {k: round(v, 3)
                                         for k, v in self.parse_quality_components.items()},
            "validation_errors": list(self.validation_errors)[:12],
            "validation_warnings": list(self.validation_warnings)[:12],
            "fallback_reason": self.fallback_reason,
            "cache_hit": self.cache_hit,
            "id_mapping": dict(self.id_mapping),
        }


# ------------------------------------------------- variable / constant / binding
# A target slot is an UNBOUND VARIABLE described by question language — not a leaked
# answer. The validator must reject only a *concrete known constant* or a *premature
# binding*, never a descriptor that happens to reuse question text (e.g. "90s TV
# series", "founder full name", "person who wrote the introduction").

#: words that signal a relational / answer-type DESCRIPTOR (vs. a concrete entity).
_RELATIONAL_WORDS = frozenset({
    "who", "whom", "whose", "which", "that", "where", "when", "matching", "with",
    "from", "by", "near", "born", "founded", "released", "wrote", "authored",
    "starred", "featuring", "featured", "published", "located", "containing",
    "about", "involving", "described"})
_ANSWER_TYPE_WORDS = frozenset({
    "name", "surname", "full", "title", "year", "date", "number", "value",
    "identity", "profile", "amount", "count", "winner", "author"})
#: small words that don't break a Title-Case proper-noun phrase.
_CAP_STOPWORDS = frozenset({"of", "the", "and", "for", "in", "on", "at", "to", "a",
                            "an", "de", "la", "el", "du", "von", "van", "by"})
_NAME_TOK = re.compile(r"[A-Za-z0-9&'.]+")


def _is_named_entity(name: str) -> bool:
    """The slot name is a CONCRETE proper-noun entity (Title-Case phrase or acronym),
    NOT a type descriptor. A single lowercase generic word makes it a descriptor:
    'World Health Organisation' / 'Tennessee' / 'Gracie Award' -> True;
    '90s TV series' / 'Mexican restaurant in NM' / 'founder full name' -> False."""
    toks = [t for t in _NAME_TOK.findall(name or "") if any(c.isalpha() for c in t)]
    significant = [t for t in toks if t.lower() not in _CAP_STOPWORDS]
    if not significant:
        return False
    if len(significant) == 1 and significant[0].isupper() and len(significant[0]) <= 6:
        return True                                  # acronym (WHO, NASA, FBI)
    return all(t[0].isupper() for t in significant)


def _has_type_or_relational(name: str, role: str) -> bool:
    toks = _WORD.findall((name or "").lower())
    if any(t in _ROLE_TRIGGERS for t in toks):       # a generic type head (series, report…)
        return True
    return any((t in _RELATIONAL_WORDS or t in _ANSWER_TYPE_WORDS) for t in toks)


def _is_descriptor(name: str, role: str) -> bool:
    """A variable DESCRIPTOR: type/relational language, and NOT a concrete entity."""
    return (not _is_named_entity(name)) and _has_type_or_relational(name, role)


def _role_matches_head(slot_role: str, question: str) -> bool:
    head = _interrogative_target_role(question)
    return head is None or slot_role == head


def _slot_has_binding_constraints(slot_id: Any, payload: dict) -> bool:
    sid = str(slot_id)
    for c in payload.get("constraints") or []:
        if isinstance(c, dict) and sid in [str(x) for x in _as_list(c.get("applies_to"))]:
            return True
    return False


def _equals_known_context(name: str, kct_norms: set[str]) -> bool:
    n = _norm(name or "")
    return bool(n) and n in kct_norms


def infer_slot_status(slot: dict, payload: dict, kct_norms: set[str]) -> str:
    """Infer the binding lifecycle of a slot when the parser did not state it.

    A target/intermediate slot is an ``unbound_variable`` unless it is clearly a
    constant given in the question (a concrete entity equal to a known-context term
    with no binding constraints) or already carries a concrete ``bound_value``."""
    raw = str(slot.get("slot_status", "") or "").lower()
    if raw in SLOT_STATUSES:
        return raw
    if str(slot.get("bound_value", "") or "").strip():
        return "candidate_binding"
    name = str(slot.get("slot_name", ""))
    if (_equals_known_context(name, kct_norms) and _is_named_entity(name)
            and not _slot_has_binding_constraints(slot.get("slot_id"), payload)):
        return "known_constant"
    return "unbound_variable"


def classify_target_slot(slot: dict, payload: dict, question: str,
                         kct_norms: set[str]) -> tuple[str, str]:
    """Classify a TARGET slot as (decision, reason).

    decision ∈ {"pass","warn","fail"}. A descriptor variable passes even if it reuses
    question text; only a premature binding or a concrete known constant fails; a
    genuinely ambiguous case warns (never a fallback)."""
    name = str(slot.get("slot_name", ""))
    role = str(slot.get("slot_role", "unknown"))
    if str(slot.get("bound_value", "") or "").strip():
        return "fail", "premature_bound_value"
    status = infer_slot_status(slot, payload, kct_norms)
    if status == "known_constant":
        return "fail", "concrete_known_constant"
    if status == "candidate_binding":
        return "fail", "premature_candidate_binding"
    # --- unbound variable: prefer to ACCEPT descriptors ---
    if (_role_matches_head(role, question)
            or _slot_has_binding_constraints(slot.get("slot_id"), payload)
            or _is_descriptor(name, role)):
        return "pass", "unbound_variable_descriptor"
    if _equals_known_context(name, kct_norms):
        return "fail", "exact_context_promoted_to_target"
    if _is_named_entity(name):
        return "fail", "concrete_known_constant"
    return "warn", "ambiguous_descriptor"


# --------------------------------------------------------------------------- json
def _extract_json(raw: str) -> Optional[dict]:
    """Tolerantly pull the first top-level JSON object out of a model response."""
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(raw[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


# --------------------------------------------------------------------------- validate
def validate_payload(payload: Any, question: str) -> list[str]:
    """Open-world validation: return hard ERRORS (empty == valid → accept the frame).

    What is checked (operational usability + safety), NOT semantic vocabulary:
    - JSON is a well-formed object with slot lists and a target slot.
    - ``applies_to`` (scalar normalized to a list) references real slot ids.
    - ``required`` constraints have a ``testable_claim`` and ``evidence_needed`` or
      ``how_to_test``, and apply to a target/intermediate slot.
    - No gold/final-answer text (forbidden keys; entity-named target slot).

    A constraint is NEVER rejected for an unfamiliar ``semantic_label`` or facet, or
    an unfamiliar ``slot_role`` — those become :func:`parser_warnings`.
    """
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["payload_not_object"]

    for k in payload:                                   # no gold-like top-level key
        if str(k).lower() in _ANSWER_KEYS:
            errors.append(f"forbidden_answer_key:{k}")

    targets = payload.get("target_answer_slots")
    latents = payload.get("latent_slots")
    constraints = payload.get("constraints")
    if not isinstance(targets, list) or not isinstance(latents, list):
        errors.append("missing_slot_lists")
        targets = targets if isinstance(targets, list) else []
        latents = latents if isinstance(latents, list) else []
    if not isinstance(constraints, list):
        errors.append("missing_constraints_list")
        constraints = []

    all_slots = list(targets) + list(latents)
    slot_ids: set[str] = set()
    target_ids: set[str] = set()
    inter_ids: set[str] = set()
    qtok = _q_tokens(question)

    if not targets:
        errors.append("no_target_slot")

    for s in all_slots:
        if not isinstance(s, dict):
            errors.append("slot_not_object")
            continue
        for f in _REQUIRED_SLOT_FIELDS:
            if f not in s:
                errors.append(f"slot_missing_field:{f}")
        sid = s.get("slot_id")
        if not sid or not isinstance(sid, str):
            errors.append("slot_empty_slot_id")
        else:
            if sid in slot_ids:
                errors.append(f"duplicate_slot_id:{sid}")
            slot_ids.add(sid)
        if not str(s.get("slot_name", "")).strip():
            errors.append("slot_empty_name")
    for s in targets:
        if isinstance(s, dict) and s.get("slot_id"):
            target_ids.add(s["slot_id"])
    for s in latents:
        if isinstance(s, dict) and s.get("slot_id"):
            inter_ids.add(s["slot_id"])

    # no gold-like answer text in a TARGET slot: a multiword capitalized name that
    # introduces tokens absent from the question is almost certainly a guessed answer.
    role_words = VALID_ROLES | {"answer", "name", "value", "target", "the"}
    for s in targets:
        if not isinstance(s, dict):
            continue
        name = str(s.get("slot_name", ""))
        novel = [t for t in _WORD.findall(name.lower())
                 if t not in qtok and t not in role_words and not t.isdigit()]
        if novel and re.search(r"[A-Z][a-z]+\s+[A-Z][a-z]+", name):
            errors.append("possible_answer_text_in_target_slot")

    for c in constraints:
        if not isinstance(c, dict):
            errors.append("constraint_not_object")
            continue
        if not c.get("constraint_id"):
            errors.append("constraint_missing_id")
        if not str(c.get("text_span", "")).strip() and not str(c.get("semantic_label", "")).strip():
            errors.append("constraint_missing_text_and_label")
        applies = _as_list(c.get("applies_to"))
        for ref in applies:
            if ref not in slot_ids:
                errors.append(f"constraint_refs_unknown_slot:{ref}")
        # operational usability for REQUIRED constraints (open-world, affordance-first).
        if bool(c.get("required")) or str(c.get("priority", "")).lower() == "high":
            if not str(c.get("testable_claim", "")).strip():
                errors.append(f"required_constraint_missing_testable_claim:{c.get('constraint_id')}")
            if not (str(c.get("evidence_needed", "")).strip()
                    or str(c.get("how_to_test", "")).strip()):
                errors.append(f"required_constraint_missing_evidence_or_how_to_test:{c.get('constraint_id')}")
            if not (set(applies) & (target_ids | inter_ids)):
                errors.append(f"required_constraint_no_target_or_intermediate_slot:{c.get('constraint_id')}")

    # Dependency edges may be lists OR objects (from/to, source/target, …); parse the
    # endpoints flexibly and validate them against the RAW slot-id namespace. An
    # unparseable SHAPE is not fatal (we fall back to per-slot depends_on); only a
    # genuinely unknown endpoint id is an error.
    for edge in payload.get("dependency_edges", []) or []:
        pe = _parse_edge(edge)
        if pe is None:
            continue
        a, b = pe
        if a not in slot_ids or b not in slot_ids:
            errors.append(f"dependency_edge_refs_unknown_slot:{a}->{b}")

    # Variable/constant validation (replaces string-overlap): a target slot fails
    # only if it is a PREMATURE BINDING or a CONCRETE KNOWN CONSTANT, never because a
    # descriptor reuses question text. Ambiguous cases warn (see parser_warnings).
    kct_norms = {_norm(str(t)) for t in (payload.get("known_context_terms") or [])
                 if _norm(str(t))}
    for s in targets:
        if not isinstance(s, dict):
            continue
        decision, reason = classify_target_slot(s, payload, question, kct_norms)
        if decision == "fail":
            errors.append(f"known_context_promoted_to_target:{reason}:{s.get('slot_id')}")

    return errors


def parser_warnings(payload: Any, question: str) -> list[str]:
    """Non-fatal observations (preserved, never trigger fallback): novel facets,
    unfamiliar slot roles (coerced to 'unknown'), interrogative-head mismatch,
    unknown affordances (dropped), non-required constraints lacking a testable claim."""
    warns: list[str] = []
    if not isinstance(payload, dict):
        return warns
    targets = payload.get("target_answer_slots") or []
    for s in (targets + (payload.get("latent_slots") or [])):
        if isinstance(s, dict) and s.get("slot_role") not in VALID_ROLES:
            warns.append(f"unfamiliar_slot_role_coerced_to_unknown:{s.get('slot_role')}")
    head_role = _interrogative_target_role(question)
    if head_role is not None and targets and not any(
            isinstance(s, dict) and s.get("slot_role") == head_role for s in targets):
        warns.append(f"target_role_mismatch:expected_{head_role}")
    # ambiguous (neither clearly a descriptor nor clearly a constant) -> warn, not fail.
    kct_norms = {_norm(str(t)) for t in (payload.get("known_context_terms") or [])
                 if _norm(str(t))}
    for s in targets:
        if isinstance(s, dict):
            decision, reason = classify_target_slot(s, payload, question, kct_norms)
            if decision == "warn":
                warns.append(f"ambiguous_target_descriptor:{reason}:{s.get('slot_id')}")
    for c in (payload.get("constraints") or []):
        if not isinstance(c, dict):
            continue
        facets = normalize_facets(str(c.get("semantic_label", "")), c.get("semantic_facets"))
        for nf in novel_facets(facets):
            warns.append(f"novel_facet:{nf}")
        for a in _as_list(c.get("affordances")):
            if a not in AFFORDANCES:
                warns.append(f"unknown_affordance_dropped:{a}")
        if not bool(c.get("required")) and not str(c.get("testable_claim", "")).strip():
            warns.append(f"constraint_missing_testable_claim:{c.get('constraint_id')}")

    # --- id / dependency-edge robustness (advisory; never a fallback) ---
    _, raw_ids = _raw_slot_ids(payload)
    for s in (targets + (payload.get("latent_slots") or [])):
        if isinstance(s, dict):
            for ref in _as_list(s.get("depends_on")):
                if str(ref) not in raw_ids:
                    warns.append(f"unknown_depends_on_ref_dropped:{ref}")
    for c in (payload.get("constraints") or []):
        if isinstance(c, dict):
            for ref in _as_list(c.get("supports_answer_slot_ids")):
                if str(ref) not in raw_ids:
                    warns.append(f"unknown_supports_answer_ref_dropped:{ref}")
    explicit = [pe for pe in (_parse_edge(e) for e in payload.get("dependency_edges") or [])
                if pe is not None]
    if any(_parse_edge(e) is None for e in payload.get("dependency_edges") or []):
        warns.append("dependency_edge_malformed_shape_dropped")
    derived = _derive_raw_edges(payload)
    if derived and explicit and set(derived) != set(explicit):
        warns.append("dependency_edges_inconsistent_with_depends_on:prefer_depends_on")
    return list(dict.fromkeys(warns))


# --------------------------------------------------------------------------- build
def _frame_from_payload(item_id: str, payload: dict, warnings: Optional[list[str]] = None) -> TaskFrame:
    """Construct a TaskFrame from a validated open-world payload.

    The parser emits its OWN slot-id namespace (e.g. T1/T2/I1/I2). Those raw ids are
    validated (above) and then **remapped consistently** to a clean internal namespace
    (``s0``, ``s1``, … in target-then-latent order); the mapping is applied to every
    reference — ``slot_id``, ``depends_on``, ``applies_to``, ``supports_answer_slot_ids``,
    and ``dependency_edges`` — and recorded as ``frame.id_mapping`` (raw -> internal).
    ``dependency_edges`` are derived from per-slot ``depends_on`` when omitted, and
    ``depends_on`` is preferred when the two disagree. Free-form semantics are
    preserved; the closed-set ``affordances`` are derived.
    """
    warnings = warnings or []
    ordered_raw, raw_ids = _raw_slot_ids(payload)
    id_map: dict[str, str] = {rid: f"s{i}" for i, rid in enumerate(ordered_raw)}
    kct_terms = [str(t) for t in _as_list(payload.get("known_context_terms"))]
    kct_norms = {_norm(t) for t in kct_terms if _norm(t)}

    def _m(rid: Any) -> Optional[str]:           # raw id -> internal id (None if unknown)
        return id_map.get(str(rid))

    def _mlist(raw: Any) -> list[str]:           # remap a ref list, dropping unknowns
        return [m for m in (_m(x) for x in _as_list(raw)) if m]

    def _slot(d: dict, *, target: bool) -> Slot:
        role = str(d.get("slot_role", "unknown"))
        if role not in VALID_ROLES:
            role = "unknown"
        raw_id = str(d["slot_id"])
        name = str(d.get("slot_name", "")).strip()
        # which constants does this descriptor reference? (e.g. "report by WHO" -> WHO)
        nl = name.lower()
        refs = [t for t in kct_terms if t and _norm(t) and _norm(t) in _norm(name)
                and _norm(t) != _norm(name)] or [t for t in kct_terms if t.lower() in nl]
        ev = str(d.get("evidence_required_to_bind", "")
                 or d.get("expected_evidence_type", role) or "")
        return Slot(
            slot_id=id_map.get(raw_id, raw_id),
            slot_name=name,
            slot_role=role,
            is_target_answer_slot=bool(d.get("is_target_answer_slot", target)),
            is_intermediate_slot=bool(d.get("is_intermediate_slot", not target)),
            depends_on=_mlist(d.get("depends_on")),
            expected_evidence_type=str(d.get("expected_evidence_type", role) or role),
            raw_slot_id=raw_id,
            slot_status=infer_slot_status(d, payload, kct_norms),
            descriptor_text=str(d.get("descriptor_text", "") or name),
            bound_value=str(d.get("bound_value", "") or "").strip(),
            known_context_refs=list(dict.fromkeys(refs))[:6],
            evidence_required_to_bind=ev,
            parser_confidence=float(d.get("parser_confidence", 0.0) or 0.0))

    frame = TaskFrame(item_id=item_id)
    frame.id_mapping = dict(id_map)
    frame.target_answer_slots = [_slot(s, target=True) for s in payload["target_answer_slots"]]
    frame.latent_slots = [_slot(s, target=False) for s in payload.get("latent_slots", [])]
    target_ids = {s.slot_id for s in frame.target_answer_slots}
    inter_ids = {s.slot_id for s in frame.latent_slots}

    for c in payload.get("constraints", []):
        terms = [str(t) for t in _as_list(c.get("normalized_terms"))][:10]
        label = str(c.get("semantic_label", "") or c.get("constraint_type", "") or "attribute")
        facets = normalize_facets(label, c.get("semantic_facets"))
        applies = _mlist(c.get("applies_to"))
        sup = _mlist(c.get("supports_answer_slot_ids"))
        con = Constraint(
            constraint_id=str(c["constraint_id"]),
            text_span=str(c.get("text_span", "")).strip(),
            normalized_terms=terms,
            # keep a legacy facet for back-compat code paths; primary facet or label.
            constraint_type=str(c.get("constraint_type") or (facets[0] if facets else "attribute")),
            applies_to=applies,
            specificity_score=float(c.get("specificity_score", 0.0) or 0.0),
            discriminative_score=float(c.get("discriminative_score",
                                             c.get("specificity_score", 0.0)) or 0.0),
            status=str(c.get("status", "unresolved") or "unresolved"),
            semantic_label=label, semantic_facets=facets,
            required=bool(c.get("required", False)),
            priority=str(c.get("priority", "medium") or "medium").lower(),
            testable_claim=str(c.get("testable_claim", "") or "").strip(),
            evidence_needed=str(c.get("evidence_needed", "") or "").strip(),
            how_to_test=str(c.get("how_to_test", "") or "").strip(),
            suggested_query_templates=[str(t) for t in _as_list(c.get("suggested_query_templates"))][:6],
            supports_answer_slot_ids=sup,
            blocks_answer_if_unresolved=bool(c.get("blocks_answer_if_unresolved", False)),
            source_quote_or_span=str(c.get("source_quote_or_span", "") or "")[:200],
            parser_confidence=float(c.get("parser_confidence", 0.0) or 0.0),
            raw_parser_output={k: c.get(k) for k in (
                "semantic_label", "semantic_facets", "constraint_type", "priority",
                "required", "affordances") if k in c})
        con.affordances = derive_affordances(
            facets=con.semantic_facets, semantic_label=con.semantic_label,
            applies_to=con.applies_to, target_slot_ids=target_ids,
            intermediate_slot_ids=inter_ids,
            supports_answer_slot_ids=con.supports_answer_slot_ids,
            required=con.required, priority=con.priority,
            blocks_answer_if_unresolved=con.blocks_answer_if_unresolved,
            testable_claim=con.testable_claim, evidence_needed=con.evidence_needed,
            how_to_test=con.how_to_test, has_terms=bool(con.normalized_terms),
            emitted=c.get("affordances"))
        con.planner_interpretation = {"affordances": list(con.affordances),
                                      "priority": con.priority, "required": con.required,
                                      "blocks_answer_if_unresolved": con.blocks_answer_if_unresolved}
        if warnings:
            con.validation_warnings = [w for w in warnings
                                       if w.endswith(con.constraint_id) or w.startswith("novel_facet")]
        frame.constraints.append(con)

    # Dependency edges: prefer per-slot depends_on (the more reliable signal); fall
    # back to explicit edges when no depends_on was emitted. Remap to internal ids and
    # drop any endpoint that did not survive remapping.
    derived = _derive_raw_edges(payload)
    explicit = [pe for pe in (_parse_edge(e) for e in payload.get("dependency_edges") or [])
                if pe is not None]
    chosen = derived if derived else explicit
    seen: set[tuple[str, str]] = set()
    for a, b in chosen:
        ia, ib = _m(a), _m(b)
        if ia and ib and (ia, ib) not in seen:
            seen.add((ia, ib))
            frame.dependency_edges.append((ia, ib))
    frame.validation_warnings = list(warnings)
    return frame


def score_parse_quality(frame: TaskFrame, question: str) -> tuple[float, dict[str, float]]:
    """Component-wise parse-quality scoring (all components in [0,1])."""
    slots = frame.all_slots
    head_role = _interrogative_target_role(question)
    # target identification: exactly-ish one target whose role matches the head.
    if not frame.target_answer_slots:
        target_id = 0.0
    else:
        role_ok = (head_role is None
                   or any(s.slot_role == head_role for s in frame.target_answer_slots))
        few = 1.0 if len(frame.target_answer_slots) <= 2 else 0.5
        target_id = (0.6 if role_ok else 0.2) + 0.4 * few
        target_id = min(1.0, target_id)
    # constraint attachment: fraction attached to existing slots.
    ids = {s.slot_id for s in slots}
    if frame.constraints:
        attach = sum(1 for c in frame.constraints
                     if c.applies_to and all(a in ids for a in c.applies_to))
        constraint_attachment = attach / len(frame.constraints)
    else:
        constraint_attachment = 0.0
    # dependency consistency: edges reference real slots and connect latent->target.
    edges = frame.dependency_edges
    if frame.latent_slots:
        valid_edges = [e for e in edges if e[0] in ids and e[1] in ids]
        dependency = (len(valid_edges) / len(edges)) if edges else 0.0
    else:
        dependency = 1.0 if not edges else 0.5
    # known-context separation: no known term is a target slot name.
    tnames = {_norm(s.slot_name) for s in frame.target_answer_slots}
    kct = [_norm(t) for t in frame.known_context_terms]
    bad = sum(1 for t in kct if t and t in tnames)
    known_sep = 1.0 if not kct else max(0.0, 1.0 - bad / max(1, len(kct)))
    # slot coverage: fraction of question role-trigger nouns covered by some slot.
    from regimes_probe.agent.task_frame import _ROLE_TRIGGERS
    qwords = _q_tokens(question)
    trig = {w for w in qwords if w in _ROLE_TRIGGERS}
    covered_roles = {s.slot_role for s in slots}
    if trig:
        want_roles = {_ROLE_TRIGGERS[w] for w in trig}
        slot_coverage = len(want_roles & covered_roles) / len(want_roles)
    else:
        slot_coverage = 1.0 if slots else 0.0
    # ambiguity: fewer "unknown" roles is better.
    unknown = sum(1 for s in slots if s.slot_role == "unknown")
    ambiguity = 1.0 if not slots else max(0.0, 1.0 - unknown / len(slots))

    comps = {
        "target_identification_score": round(target_id, 3),
        "constraint_attachment_score": round(constraint_attachment, 3),
        "dependency_score": round(dependency, 3),
        "known_context_separation_score": round(known_sep, 3),
        "slot_coverage_score": round(slot_coverage, 3),
        "ambiguity_score": round(ambiguity, 3),
    }
    weights = {
        "target_identification_score": 0.30,
        "constraint_attachment_score": 0.25,
        "dependency_score": 0.10,
        "known_context_separation_score": 0.15,
        "slot_coverage_score": 0.10,
        "ambiguity_score": 0.10,
    }
    agg = sum(comps[k] * w for k, w in weights.items())
    return round(agg, 3), comps


# --------------------------------------------------------------------------- parser
class LLMTaskFrameParser:
    """Cached/replayable LLM task-frame parser. Validates + falls back to determ."""

    def __init__(self, model_fn: Optional[Callable[[str], str]] = None, *,
                 cache: Optional[ParserCache] = None, model: str = "stub",
                 prompt_name: str = "task_frame_parser", replay_only: bool = False) -> None:
        self.model_fn = model_fn
        self.cache = cache if cache is not None else ParserCache()
        self.model = model
        self.prompt = prompts.get(prompt_name)
        self.replay_only = replay_only
        # accounting (read into report/manifest by the runner)
        self.model_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.fallback_count = 0

    def _input_hash(self, question: str) -> str:
        return _sha(f"{self.prompt.fingerprint()}|{self.model}|{question.strip()}")

    def stats(self) -> dict[str, Any]:
        """Answer-free parser accounting (model calls / cache / fallbacks)."""
        return {"task_frame_parser_model": self.model,
                "parser_model_calls": self.model_calls,
                "parser_cache_hits": self.cache_hits,
                "parser_cache_misses": self.cache_misses,
                "parser_fallback_count": self.fallback_count}

    def parse(self, item_id: str, question: str) -> tuple[Optional[TaskFrame], ParseMeta]:
        """Attempt an LLM parse. Returns (frame|None, meta); None means: fall back."""
        meta = ParseMeta(parser_used="llm", prompt_version=self.prompt.version,
                         prompt_hash=self.prompt.content_hash, model=self.model)
        meta.input_hash = self._input_hash(question)
        raw = self.cache.get(meta.input_hash)
        if raw is not None:
            meta.cache_hit = True
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            if self.replay_only or self.model_fn is None:
                meta.fallback_reason = ("cache_miss_in_replay" if self.replay_only
                                        else "no_model_available")
                self.fallback_count += 1
                return None, meta
            prompt_text = f"{self.prompt.content}\n\nQUESTION:\n{question.strip()}"
            try:
                self.model_calls += 1
                raw = self.model_fn(prompt_text)
            except Exception as exc:  # model error -> deterministic fallback
                meta.fallback_reason = f"model_error:{type(exc).__name__}"
                self.fallback_count += 1
                return None, meta
            self.cache.put(meta.input_hash, raw or "")
        meta.output_hash = _sha(raw or "")

        payload = _extract_json(raw or "")
        if payload is None:
            meta.fallback_reason = "invalid_json"
            self.fallback_count += 1
            return None, meta
        errors = validate_payload(payload, question)
        meta.validation_errors = errors
        if errors:
            meta.fallback_reason = "validation_failed"
            self.fallback_count += 1
            return None, meta
        warnings = parser_warnings(payload, question)
        meta.validation_warnings = warnings
        frame = _frame_from_payload(item_id, payload, warnings)
        meta.id_mapping = dict(frame.id_mapping)
        frame.known_context_terms = [str(t) for t in _as_list(payload.get("known_context_terms"))]
        frame.answer_shape_hints = [str(t) for t in _as_list(payload.get("answer_shape_hints"))]
        frame.source_requirements = [str(t) for t in _as_list(payload.get("source_requirements"))]
        frame.validation_warnings = warnings
        quality, comps = score_parse_quality(frame, question)
        frame.parse_quality = quality
        meta.parse_quality = quality
        meta.parse_quality_components = comps
        if quality < MIN_LLM_PARSE_QUALITY:
            meta.fallback_reason = "low_quality"
            self.fallback_count += 1
            return None, meta
        return frame, meta


# --------------------------------------------------------------------------- coordinator
def build_task_frame(item_id: str, question: str, *, use_llm: bool = False,
                     parser: Optional[LLMTaskFrameParser] = None
                     ) -> tuple[TaskFrame, ParseMeta]:
    """Build a task frame: LLM parser first (if enabled+valid), else deterministic.

    Always returns a usable `TaskFrame`. The deterministic parser is the default
    and the fallback; the LLM parser only *replaces* it when enabled and its output
    validates above the quality floor. ``meta`` carries answer-free provenance.
    """
    if use_llm and parser is not None:
        frame, meta = parser.parse(item_id, question)
        if frame is not None:
            return frame, meta
        # fall through to deterministic, preserving llm provenance for debug.
        det = parse_task_frame(item_id, question)
        _, comps = score_parse_quality(det, question)
        meta.parser_used = "deterministic"
        if not meta.fallback_reason:
            meta.fallback_reason = "llm_frame_none"
        det_meta = meta
        det_meta.parse_quality = det.parse_quality
        det_meta.parse_quality_components = comps
        return det, det_meta

    det = parse_task_frame(item_id, question)
    quality, comps = score_parse_quality(det, question)
    meta = ParseMeta(parser_used="deterministic", parse_quality=det.parse_quality,
                     parse_quality_components=comps)
    return det, meta
