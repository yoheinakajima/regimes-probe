"""Agent layer: signature → plans → budgeted search loop → answer."""

from __future__ import annotations

from regimes_probe.agent.answerer import CandidateAnswer, DeterministicAnswerer
from regimes_probe.agent.evidence import EvidenceObservation, score_observation
from regimes_probe.agent.planner import AgentConfig, EpistemicAgent
from regimes_probe.agent.search_loop import (
    AttemptTrace,
    CallRecord,
    DirectInvoker,
    SearchLoop,
    SearchLoopConfig,
    ToolInvoker,
)

__all__ = [
    "CandidateAnswer", "DeterministicAnswerer",
    "EvidenceObservation", "score_observation",
    "AgentConfig", "EpistemicAgent",
    "AttemptTrace", "CallRecord", "DirectInvoker", "SearchLoop",
    "SearchLoopConfig", "ToolInvoker",
]
