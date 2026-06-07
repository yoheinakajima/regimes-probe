"""Answerer: turn gathered evidence into a candidate answer.

The synthetic/default answerer is deterministic and models a *competent
extractor*: given the right evidence, it reports the answer the evidence
asserts. It selects the assertion with the highest authority-weighted support,
so corroboration and source authority matter. It abstains when no evidence
asserts anything (e.g. a multihop answer that was never fetched).

This deliberately isolates the thing we are studying — *retrieval/epistemic
policy* — from answer extraction. A live answerer (OpenAI Responses) can be
swapped in behind the same interface for real runs; in v0 we do NOT optimize the
answer prompt (see ``docs/LEAKAGE_CONTROLS.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from regimes_probe.agent.evidence import EvidenceObservation


@dataclass
class CandidateAnswer:
    answer: Optional[str]
    support_count: int
    support_authority: float
    support_urls: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "support_count": self.support_count,
            "support_authority": round(self.support_authority, 3),
            "support_urls": self.support_urls,
        }


class DeterministicAnswerer:
    """Authority-weighted vote over asserted answers in the evidence."""

    name = "deterministic_answerer"

    def answer(self, observations: list[EvidenceObservation]) -> CandidateAnswer:
        votes: dict[str, float] = {}
        counts: dict[str, int] = {}
        urls: dict[str, list[str]] = {}
        for obs in observations:
            if not (obs.supports and obs.asserts):
                continue
            w = 0.5 + obs.source_authority  # authority-weighted vote
            votes[obs.asserts] = votes.get(obs.asserts, 0.0) + w
            counts[obs.asserts] = counts.get(obs.asserts, 0) + 1
            urls.setdefault(obs.asserts, []).append(obs.url)
        if not votes:
            return CandidateAnswer(None, 0, 0.0, [])
        # Deterministic argmax: weight desc, then answer text asc.
        best = max(votes.items(), key=lambda kv: (kv[1], kv[0]))[0]
        best_auth = max(
            (o.source_authority for o in observations if o.asserts == best and o.supports),
            default=0.0,
        )
        return CandidateAnswer(
            answer=best,
            support_count=counts[best],
            support_authority=best_auth,
            support_urls=urls[best],
        )
