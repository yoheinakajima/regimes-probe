"""Canonical object-type names for the regimes-probe ActiveGraph pack.

The graph is a deterministic projection of the event log. These are the object
types projected from ``object.created`` events. See ``docs/ACTIVEGRAPH_DESIGN.md``.
"""

from __future__ import annotations


class Objects:
    """Namespace of canonical object-type strings."""

    BENCHMARK_RUN = "benchmark_run"
    BENCHMARK_ITEM = "benchmark_item"
    QUESTION_ATTEMPT = "question_attempt"
    QUERY_SIGNATURE = "query_signature"
    POLICY_MEMORY_SNAPSHOT = "policy_memory_snapshot"
    POLICY_FRAGMENT = "policy_fragment"
    ROUTING_PLAN = "routing_plan"
    QUERY_PLAN = "query_plan"
    TOOL_CALL = "tool_call"
    EVIDENCE_OBSERVATION = "evidence_observation"
    CANDIDATE_ANSWER = "candidate_answer"
    FINAL_ANSWER = "final_answer"
    GRADE_RESULT = "grade_result"
    EVIDENCE_REWARD = "evidence_reward"
    TOOL_REWARD = "tool_reward"
    TRACE_REWARD = "trace_reward"
    REGIME_LABEL = "regime_label"
    POLICY_UPDATE = "policy_update"
    PROMOTION_DECISION = "promotion_decision"
    REPORT = "report"


ALL_OBJECTS: tuple[str, ...] = (
    Objects.BENCHMARK_RUN,
    Objects.BENCHMARK_ITEM,
    Objects.QUESTION_ATTEMPT,
    Objects.QUERY_SIGNATURE,
    Objects.POLICY_MEMORY_SNAPSHOT,
    Objects.POLICY_FRAGMENT,
    Objects.ROUTING_PLAN,
    Objects.QUERY_PLAN,
    Objects.TOOL_CALL,
    Objects.EVIDENCE_OBSERVATION,
    Objects.CANDIDATE_ANSWER,
    Objects.FINAL_ANSWER,
    Objects.GRADE_RESULT,
    Objects.EVIDENCE_REWARD,
    Objects.TOOL_REWARD,
    Objects.TRACE_REWARD,
    Objects.REGIME_LABEL,
    Objects.POLICY_UPDATE,
    Objects.PROMOTION_DECISION,
    Objects.REPORT,
)
