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

    def answer(self, observations: list[EvidenceObservation], *, item=None) -> CandidateAnswer:
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


class ClosedBookAnswerer:
    """Answer from *intrinsic knowledge* with NO tool calls (closed-book).

    For the offline harness this is a deterministic stand-in: a fixed knowledge
    map of ``item_id -> answer`` simulates "the model already knows this fact".
    For a live run, closed-book is simply the real answerer with tools disabled
    (no knowledge map needed) — this class is the no-key simulation only, and is
    clearly labelled in reports.
    """

    name = "closed_book_answerer"

    def __init__(self, knowledge: Optional[dict[str, str]] = None) -> None:
        self.knowledge = knowledge or {}

    def answer(self, observations: list[EvidenceObservation], *, item=None) -> CandidateAnswer:
        # Closed-book ignores any (absent) evidence; it answers only what it
        # "knows" intrinsically.
        if item is not None and item.id in self.knowledge:
            return CandidateAnswer(self.knowledge[item.id], 0, 1.0, ["intrinsic:closed_book"])
        return CandidateAnswer(None, 0, 0.0, [])


def build_closed_book_knowledge(items, *, flag: str = "intrinsic_knowable") -> dict[str, str]:
    """Build the simulated intrinsic-knowledge map for synthetic items.

    Only items whose ``meta[flag]`` is true are "known" — modelling a model that
    knows a subset of facts without searching. This is a deliberate harness
    simulation, NOT answer leakage into policy memory.
    """
    return {it.id: it.answer for it in items if it.meta.get(flag) and it.answer}
