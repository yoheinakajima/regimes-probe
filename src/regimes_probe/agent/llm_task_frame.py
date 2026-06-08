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
from regimes_probe.agent.clue_resolution import ROLES, _norm
from regimes_probe.agent.task_frame import (
    Constraint, Slot, TaskFrame, _interrogative_target_role, parse_task_frame)

#: Valid slot roles (the frame vocabulary + the numeric role).
VALID_ROLES: frozenset[str] = frozenset(ROLES) | {"number"}
#: Valid constraint types.
from regimes_probe.agent.task_frame import CONSTRAINT_TYPES as _CT
VALID_CONSTRAINT_TYPES: frozenset[str] = frozenset(_CT)
#: A parsed frame below this aggregate quality is rejected (deterministic fallback).
MIN_LLM_PARSE_QUALITY: float = 0.5

_REQUIRED_SLOT_FIELDS = ("slot_id", "slot_name", "slot_role", "is_target_answer_slot",
                         "is_intermediate_slot", "depends_on", "expected_evidence_type")
_REQUIRED_CONSTRAINT_FIELDS = ("constraint_id", "text_span", "normalized_terms",
                               "constraint_type", "applies_to", "specificity_score",
                               "discriminative_score", "status")
_ANSWER_KEYS = ("answer", "final_answer", "solution", "gold", "gold_answer", "result")
_WORD = re.compile(r"[A-Za-z0-9]+")


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
    fallback_reason: str = ""
    cache_hit: bool = False

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
            "fallback_reason": self.fallback_reason,
            "cache_hit": self.cache_hit,
        }


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
    """Return a list of validation errors (empty == valid). Schema + safety checks."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["payload_not_object"]

    # forbid any answer-bearing top-level key (no gold-like text).
    for k in payload:
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
        role = s.get("slot_role")
        if role not in VALID_ROLES:
            errors.append(f"unsupported_role:{role}")

    # target slot role should match the interrogative head where detectable.
    head_role = _interrogative_target_role(question)
    if head_role is not None and targets:
        if not any(isinstance(s, dict) and s.get("slot_role") == head_role for s in targets):
            errors.append(f"target_role_mismatch:expected_{head_role}")

    # no gold-like answer text in a TARGET slot: every token of a target slot_name
    # must be a question token or a generic role descriptor (not a novel entity).
    role_words = VALID_ROLES | {"answer", "name", "value", "target", "the"}
    for s in targets:
        if not isinstance(s, dict):
            continue
        name = str(s.get("slot_name", ""))
        novel = [t for t in _WORD.findall(name.lower())
                 if t not in qtok and t not in role_words and not t.isdigit()]
        # a capitalized multiword name introducing tokens absent from the question
        # is almost certainly a guessed answer, not a parse.
        if novel and re.search(r"[A-Z][a-z]+\s+[A-Z][a-z]+", name):
            errors.append("possible_answer_text_in_target_slot")

    # constraints attach to existing slot ids; required fields present.
    n_attached = 0
    for c in constraints:
        if not isinstance(c, dict):
            errors.append("constraint_not_object")
            continue
        for f in _REQUIRED_CONSTRAINT_FIELDS:
            if f not in c:
                errors.append(f"constraint_missing_field:{f}")
        ctype = c.get("constraint_type")
        if ctype is not None and ctype not in VALID_CONSTRAINT_TYPES:
            errors.append(f"unsupported_constraint_type:{ctype}")
        applies = c.get("applies_to") or []
        if not isinstance(applies, list):
            errors.append("constraint_applies_to_not_list")
            applies = []
        for ref in applies:
            if ref not in slot_ids:
                errors.append(f"constraint_refs_unknown_slot:{ref}")
        if applies:
            n_attached += 1

    # dependency edges reference existing slots.
    for edge in payload.get("dependency_edges", []) or []:
        if (not isinstance(edge, (list, tuple)) or len(edge) != 2
                or edge[0] not in slot_ids or edge[1] not in slot_ids):
            errors.append("dependency_edge_refs_unknown_slot")

    # known context terms must be GIVEN in the question, and must not be a target
    # slot name (the WHO-as-target failure) unless the question asks for that source.
    kct = payload.get("known_context_terms", []) or []
    target_names = {_norm(str(s.get("slot_name", ""))) for s in targets if isinstance(s, dict)}
    for term in kct:
        tnorm = _norm(str(term))
        if tnorm and tnorm in target_names:
            errors.append(f"known_context_promoted_to_target:{term}")

    return errors


# --------------------------------------------------------------------------- build
def _frame_from_payload(item_id: str, payload: dict) -> TaskFrame:
    """Construct a TaskFrame from a validated payload (fills derived defaults)."""
    def _slot(d: dict, *, target: bool) -> Slot:
        return Slot(
            slot_id=str(d["slot_id"]),
            slot_name=str(d.get("slot_name", "")).strip(),
            slot_role=str(d.get("slot_role", "unknown")),
            is_target_answer_slot=bool(d.get("is_target_answer_slot", target)),
            is_intermediate_slot=bool(d.get("is_intermediate_slot", not target)),
            depends_on=[str(x) for x in (d.get("depends_on") or [])],
            expected_evidence_type=str(d.get("expected_evidence_type",
                                              d.get("slot_role", "")) or ""))

    frame = TaskFrame(item_id=item_id)
    frame.target_answer_slots = [_slot(s, target=True) for s in payload["target_answer_slots"]]
    frame.latent_slots = [_slot(s, target=False) for s in payload.get("latent_slots", [])]
    for c in payload.get("constraints", []):
        terms = [str(t) for t in (c.get("normalized_terms") or [])][:10]
        frame.constraints.append(Constraint(
            constraint_id=str(c["constraint_id"]),
            text_span=str(c.get("text_span", "")).strip(),
            normalized_terms=terms,
            constraint_type=str(c.get("constraint_type", "attribute")),
            applies_to=[str(x) for x in (c.get("applies_to") or [])],
            specificity_score=float(c.get("specificity_score", 0.0) or 0.0),
            discriminative_score=float(c.get("discriminative_score",
                                             c.get("specificity_score", 0.0)) or 0.0),
            status=str(c.get("status", "unresolved") or "unresolved")))
    frame.dependency_edges = _build_edges(payload)
    return frame


def _build_edges(payload: dict) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    for e in payload.get("dependency_edges", []) or []:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            edges.append((str(e[0]), str(e[1])))
    return edges


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
        frame = _frame_from_payload(item_id, payload)
        frame.dependency_edges = _build_edges(payload)
        frame.known_context_terms = [str(t) for t in (payload.get("known_context_terms") or [])]
        frame.answer_shape_hints = [str(t) for t in (payload.get("answer_shape_hints") or [])]
        frame.source_requirements = [str(t) for t in (payload.get("source_requirements") or [])]
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
