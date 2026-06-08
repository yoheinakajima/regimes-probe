"""Agent layer: signature → plans → budgeted search loop → answer."""

from __future__ import annotations

from regimes_probe.agent.answerer import (
    CandidateAnswer,
    ClosedBookAnswerer,
    DeterministicAnswerer,
    build_closed_book_knowledge,
)
from regimes_probe.agent.evidence import EvidenceObservation, score_observation
from regimes_probe.agent.planner import AgentConfig, EpistemicAgent, build_closed_book_agent
from regimes_probe.agent.search_loop import (
    AttemptTrace,
    CallRecord,
    DirectInvoker,
    SearchLoop,
    SearchLoopConfig,
    ToolInvoker,
)

__all__ = [
    "CandidateAnswer", "ClosedBookAnswerer", "DeterministicAnswerer",
    "build_closed_book_knowledge",
    "EvidenceObservation", "score_observation",
    "AgentConfig", "EpistemicAgent", "build_closed_book_agent",
    "AttemptTrace", "CallRecord", "DirectInvoker", "SearchLoop",
    "SearchLoopConfig", "ToolInvoker",
]
