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
