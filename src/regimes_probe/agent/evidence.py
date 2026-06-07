"""Evidence observations and gold-free evidence scoring.

An :class:`EvidenceObservation` is the agent's structured read of one search/
fetch result. Scoring (``supports``, ``fresh``, ``source_authority``) is derived
from the result and the *question's* properties — never from the gold answer.
Whether an asserted answer is actually correct is the grader's job, downstream.

The ``asserts`` field (the answer a document claims) is transient: it is used by
the answerer and to detect contradictions, but it is never written into policy
memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from regimes_probe.datasets.base import Item
from regimes_probe.tools.base import SearchResult


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


@dataclass
class EvidenceObservation:
    call_index: int
    tool: str
    query_arm: str
    url: str
    snippet: str
    source_authority: float
    published_at: Optional[str]
    fresh: bool
    supports: bool
    asserts: Optional[str]
    content_hash: str
    fetchable: bool = False   # relevant doc whose assertion is hidden until fetched

    def to_public_dict(self) -> dict[str, Any]:
        """Answer-free projection for logging (drops the asserted answer)."""
        return {
            "call_index": self.call_index,
            "tool": self.tool,
            "query_arm": self.query_arm,
            "url": self.url,
            "source_authority": round(self.source_authority, 3),
            "published_at": self.published_at,
            "fresh": self.fresh,
            "supports": self.supports,
            "content_hash": self.content_hash,
        }

    def to_verify_dict(self) -> dict[str, Any]:
        """Includes ``asserts`` for verification (transient, not persisted)."""
        d = self.to_public_dict()
        d["asserts"] = self.asserts
        return d


def score_observation(
    result: SearchResult,
    item: Item,
    *,
    call_index: int,
    tool: str,
    query_arm: str,
    as_of: str,
    fresh_window_days: int = 150,
) -> EvidenceObservation:
    """Build an :class:`EvidenceObservation` from a raw result (gold-free)."""
    extra = result.extra or {}
    relevant = extra.get("item_id") == item.id
    asserts = extra.get("asserts")
    supports = bool(relevant and asserts is not None)
    fetchable = bool(relevant and asserts is None and extra.get("answer_on_fetch_only"))

    pub = _parse_date(result.published_at)
    today = _parse_date(as_of) or date.today()
    fresh = bool(pub is not None and (today - pub).days <= fresh_window_days)

    return EvidenceObservation(
        fetchable=fetchable,
        call_index=call_index,
        tool=tool,
        query_arm=query_arm,
        url=result.url,
        snippet=result.snippet,
        source_authority=float(result.source_authority),
        published_at=result.published_at,
        fresh=fresh,
        supports=supports,
        asserts=asserts,
        content_hash=result.content_hash(),
    )
