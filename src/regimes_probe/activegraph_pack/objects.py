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
    TOOL_RESPONSE = "tool_response"
    EVIDENCE_OBSERVATION = "evidence_observation"
    CANDIDATE_ANSWER = "candidate_answer"
    FINAL_ANSWER = "final_answer"
    ANSWER_ATTEMPT = "answer_attempt"
    GRADE_RESULT = "grade_result"
    EVIDENCE_REWARD = "evidence_reward"
    TOOL_REWARD = "tool_reward"
    TRACE_REWARD = "trace_reward"
    REWARD_ASSIGNMENT = "reward_assignment"
    REGIME_LABEL = "regime_label"
    FAILURE_REGIME = "failure_regime"
    MEMORY_SNAPSHOT = "memory_snapshot"
    ELIGIBILITY_VERDICT = "eligibility_verdict"
    CLAIM_CANDIDATE = "claim_candidate"
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

#: Standardized object types for the exported graph projection
#: (``results/{run_id}/graph_projection.json``). Several map 1:1 onto the
#: event-log object types above; a few (``tool_response``, ``answer_attempt``,
#: ``reward_assignment``, ``failure_regime``, ``memory_snapshot``,
#: ``eligibility_verdict``, ``claim_candidate``) are promoted to first-class
#: projection objects here. See ``docs/ACTIVEGRAPH_DESIGN.md``.
PROJECTION_OBJECTS: tuple[str, ...] = (
    Objects.BENCHMARK_RUN,
    Objects.BENCHMARK_ITEM,
    Objects.QUESTION_ATTEMPT,
    Objects.ROUTING_PLAN,
    Objects.QUERY_PLAN,
    Objects.TOOL_CALL,
    Objects.TOOL_RESPONSE,
    Objects.EVIDENCE_OBSERVATION,
    Objects.ANSWER_ATTEMPT,
    Objects.GRADE_RESULT,
    Objects.REWARD_ASSIGNMENT,
    Objects.FAILURE_REGIME,
    Objects.POLICY_FRAGMENT,
    Objects.MEMORY_SNAPSHOT,
    Objects.ELIGIBILITY_VERDICT,
    Objects.CLAIM_CANDIDATE,
    Objects.REPORT,
)

#: Map an event-log object type onto its standardized projection type.
CANONICAL_PROJECTION_TYPE: dict[str, str] = {
    Objects.FINAL_ANSWER: Objects.ANSWER_ATTEMPT,
    Objects.CANDIDATE_ANSWER: Objects.ANSWER_ATTEMPT,
    Objects.TRACE_REWARD: Objects.REWARD_ASSIGNMENT,
    Objects.TOOL_REWARD: Objects.REWARD_ASSIGNMENT,
    Objects.EVIDENCE_REWARD: Objects.REWARD_ASSIGNMENT,
    Objects.REGIME_LABEL: Objects.FAILURE_REGIME,
    Objects.POLICY_MEMORY_SNAPSHOT: Objects.MEMORY_SNAPSHOT,
}
