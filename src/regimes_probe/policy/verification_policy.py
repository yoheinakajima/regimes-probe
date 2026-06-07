"""Level 2: verification policy.

Given the evidence gathered so far and a candidate answer, compute the
verification state: is there a candidate, is it supported, is the source
authoritative, is it fresh, are there unresolved contradictions. This state
feeds the stopping policy and the reward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VerificationState:
    candidate_found: bool = False
    support_found: bool = False
    authority_ok: bool = False
    freshness_ok: bool = False
    contradiction: bool = False
    support_count: int = 0
    score: float = 0.0

    def context_code(self) -> str:
        """Compact code used as extra context for the stopping bandit."""
        return (
            f"c{int(self.candidate_found)}"
            f"s{int(self.support_found)}"
            f"a{int(self.authority_ok)}"
            f"f{int(self.freshness_ok)}"
            f"x{int(self.contradiction)}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_found": self.candidate_found,
            "support_found": self.support_found,
            "authority_ok": self.authority_ok,
            "freshness_ok": self.freshness_ok,
            "contradiction": self.contradiction,
            "support_count": self.support_count,
            "score": round(self.score, 4),
            "context_code": self.context_code(),
        }


@dataclass
class VerificationConfig:
    """Mutable thresholds (regimes loop may tune these)."""

    authority_threshold: float = 0.7
    freshness_required_when_sensitive: bool = True
    min_support: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority_threshold": self.authority_threshold,
            "freshness_required_when_sensitive": self.freshness_required_when_sensitive,
            "min_support": self.min_support,
        }


def verify(
    candidate: dict[str, Any] | None,
    evidence: list[dict[str, Any]],
    *,
    freshness_sensitive: bool,
    config: VerificationConfig | None = None,
) -> VerificationState:
    """Compute the verification state from candidate + evidence observations.

    ``candidate`` is ``{"answer": ..., "support_urls": [...]}`` (transient — not
    persisted to policy memory). Each evidence dict carries ``supports`` (bool),
    ``source_authority`` (0..1), ``fresh`` (bool), ``contradicts`` (bool).
    """
    cfg = config or VerificationConfig()
    st = VerificationState()
    st.candidate_found = candidate is not None and bool(candidate.get("answer"))
    supports = [e for e in evidence if e.get("supports")]
    st.support_count = len(supports)
    st.support_found = st.support_count >= cfg.min_support
    st.authority_ok = any(
        float(e.get("source_authority", 0.0)) >= cfg.authority_threshold for e in supports
    )
    if freshness_sensitive and cfg.freshness_required_when_sensitive:
        st.freshness_ok = any(bool(e.get("fresh")) for e in supports)
    else:
        st.freshness_ok = True
    # A contradiction is two supporting sources asserting *different* answers —
    # derived from the evidence itself, never from the gold answer.
    asserted = {e.get("asserts") for e in supports if e.get("asserts")}
    st.contradiction = len(asserted) > 1 or any(e.get("contradicts") for e in evidence)

    # Verification score in [0,1]: a soft sufficiency signal.
    score = 0.0
    score += 0.4 if st.candidate_found else 0.0
    score += 0.3 if st.support_found else 0.0
    score += 0.15 if st.authority_ok else 0.0
    score += 0.15 if st.freshness_ok else 0.0
    score -= 0.3 if st.contradiction else 0.0
    st.score = max(0.0, min(1.0, score))
    return st
