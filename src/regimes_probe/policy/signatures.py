"""Query signatures: generic *epistemic* features of a question.

A signature is the contextual-bandit context. It is deliberately built from
**epistemic properties** of the question — freshness sensitivity, authority
sensitivity, answer-shape constraint, multihop structure, … — and NOT from
topical categories. This is what lets a learned policy transfer across topics:
"recent-fact lookups want a freshness tool + freshness query terms" is a
procedural lesson, not a fact about any subject.

Feature extraction is deterministic (pure function of the text) so signatures
are stable under replay. See ``docs/POLICY_MEMORY.md``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from typing import Any

from regimes_probe.policy.embeddings import HashEmbedder

#: The eight generic epistemic features. Order is canonical (used as a vector).
EPISTEMIC_FEATURES: tuple[str, ...] = (
    "freshness_sensitive",
    "source_authority_sensitive",
    "answer_shape_constrained",
    "query_fragile",
    "evidence_sparse",
    "staleness_prone",
    "multihop_likely",
    "verification_heavy",
)

_FRESHNESS = ("current", "latest", "now", "today", "recent", "recently", "this year",
              "currently", "2024", "2025", "2026", "as of")
_AUTHORITY = ("official", "government", "regulation", "regulatory", "sec ", "law",
              "statute", "standard", "specification", "filing", "patent", "treaty")
_SHAPE = ("how many", "what year", "what date", "which", "what is the name",
          "who is", "how much", "what percentage", "in what")
_STALENESS = ("price", "population", "ceo", "ranking", "rank", "record", "champion",
              "winner", "owner", "capital of", "leader")
_VERIFY = ("is it true", "verify", "confirm", "rumor", "rumour", "allegedly",
           "reportedly", "disputed", "controversial")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_PROPER_RE = re.compile(r"\b[A-Z][a-zA-Z]+\b")


def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace. Used for the dedupe/overlap hash."""
    return " ".join(_TOKEN_RE.findall(text.lower()))


def text_hash(text: str) -> str:
    """Stable hash of the normalized question text (exact-overlap guard)."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()[:16]


def _contains_any(text_l: str, needles: tuple[str, ...]) -> bool:
    return any(n in text_l for n in needles)


@dataclass
class QuerySignature:
    """The contextual key for a question. Carries NO answer information."""

    question: str
    norm_hash: str
    features: dict[str, float]
    lexical: dict[str, Any]
    embedding: list[float] = field(default_factory=list)

    @property
    def cluster_key(self) -> str:
        """Discrete latent-regime bucket from the active epistemic features.

        This is a *learned-clusters proxy*, not a topical label: questions with
        the same epistemic shape land in the same bucket regardless of subject.
        """
        active = [f for f in EPISTEMIC_FEATURES if self.features.get(f, 0.0) >= 0.5]
        return "+".join(active) if active else "generic"

    def feature_vector(self) -> list[float]:
        return [self.features.get(f, 0.0) for f in EPISTEMIC_FEATURES]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cluster_key"] = self.cluster_key
        return d


class SignatureExtractor:
    """Compute a :class:`QuerySignature` from a question. Deterministic."""

    def __init__(self, embedder: HashEmbedder | None = None) -> None:
        self.embedder = embedder or HashEmbedder()

    def compute(self, question: str) -> QuerySignature:
        text_l = question.lower()
        tokens = _TOKEN_RE.findall(text_l)
        n_tokens = len(tokens)
        propers = _PROPER_RE.findall(question)
        commas = question.count(",")
        clauses = max(1, question.count(" that ") + question.count(" who ")
                      + question.count(" which ") + commas)

        feats: dict[str, float] = {}
        feats["freshness_sensitive"] = 1.0 if _contains_any(text_l, _FRESHNESS) else 0.0
        feats["source_authority_sensitive"] = 1.0 if _contains_any(text_l, _AUTHORITY) else 0.0
        feats["answer_shape_constrained"] = 1.0 if _contains_any(text_l, _SHAPE) else 0.0
        feats["query_fragile"] = 1.0 if (n_tokens >= 22 or len(propers) >= 4) else 0.0
        feats["evidence_sparse"] = 1.0 if (commas >= 3 or "obscure" in text_l or len(propers) >= 5) else 0.0
        feats["staleness_prone"] = 1.0 if _contains_any(text_l, _STALENESS) else 0.0
        feats["multihop_likely"] = 1.0 if clauses >= 2 else 0.0
        feats["verification_heavy"] = 1.0 if _contains_any(text_l, _VERIFY) else 0.0

        lexical = {
            "n_tokens": n_tokens,
            "n_proper_nouns": len(propers),
            "n_commas": commas,
            "has_quotes": ('"' in question or "'" in question),
            "has_year": bool(re.search(r"\b(19|20)\d{2}\b", question)),
            "n_clauses": clauses,
        }
        return QuerySignature(
            question=question,
            norm_hash=text_hash(question),
            features=feats,
            lexical=lexical,
            embedding=self.embedder.embed(question),
        )
