"""Canonical relation-type names for the regimes-probe ActiveGraph pack.

Relations are projected from ``relation.created`` events and encode the
provenance graph that makes a run auditable. See ``docs/ACTIVEGRAPH_DESIGN.md``.
"""

from __future__ import annotations


class Relations:
    """Namespace of canonical relation-type strings."""

    ATTEMPT_FOR_ITEM = "attempt_for_item"
    SIGNATURE_FOR_ATTEMPT = "signature_for_attempt"
    SNAPSHOT_USED_BY_ATTEMPT = "snapshot_used_by_attempt"
    FRAGMENT_SELECTED_FOR_ATTEMPT = "fragment_selected_for_attempt"
    PLAN_FOR_ATTEMPT = "plan_for_attempt"
    QUERY_FOR_TOOL_CALL = "query_for_tool_call"
    RESPONSE_FOR_TOOL_CALL = "response_for_tool_call"
    EVIDENCE_FROM_TOOL_CALL = "evidence_from_tool_call"
    EVIDENCE_FROM_TOOL_RESPONSE = "evidence_from_tool_response"
    ANSWER_SUPPORTED_BY_EVIDENCE = "answer_supported_by_evidence"
    GRADE_FOR_ANSWER = "grade_for_answer"
    REWARD_FOR_CALL = "reward_for_call"
    REWARD_FROM_TOOL_CALL = "reward_from_tool_call"
    REWARD_FOR_ATTEMPT = "reward_for_attempt"
    REGIME_FOR_ATTEMPT = "regime_for_attempt"
    POLICY_FRAGMENT_FROM_TRACES = "policy_fragment_from_traces"
    MEMORY_SNAPSHOT_CONTAINS_FRAGMENT = "memory_snapshot_contains_fragment"
    ELIGIBILITY_FOR_RUN = "eligibility_for_run"
    CLAIM_SUPPORTED_BY_ARTIFACT = "claim_supported_by_artifact"
    UPDATE_FROM_REWARD = "update_from_reward"
    REGIME_FOR_FAILURE = "regime_for_failure"
    PROMOTION_FOR_UPDATE = "promotion_for_update"


ALL_RELATIONS: tuple[str, ...] = (
    Relations.ATTEMPT_FOR_ITEM,
    Relations.SIGNATURE_FOR_ATTEMPT,
    Relations.SNAPSHOT_USED_BY_ATTEMPT,
    Relations.FRAGMENT_SELECTED_FOR_ATTEMPT,
    Relations.PLAN_FOR_ATTEMPT,
    Relations.QUERY_FOR_TOOL_CALL,
    Relations.EVIDENCE_FROM_TOOL_CALL,
    Relations.ANSWER_SUPPORTED_BY_EVIDENCE,
    Relations.GRADE_FOR_ANSWER,
    Relations.REWARD_FOR_CALL,
    Relations.REWARD_FOR_ATTEMPT,
    Relations.UPDATE_FROM_REWARD,
    Relations.REGIME_FOR_FAILURE,
    Relations.PROMOTION_FOR_UPDATE,
)

#: Standardized causal relations for the exported graph projection. Names follow
#: the public schema in ``docs/ACTIVEGRAPH_DESIGN.md``; several are aliases of the
#: event-log relations above with a clearer public name.
PROJECTION_RELATIONS: tuple[str, ...] = (
    Relations.ATTEMPT_FOR_ITEM,
    Relations.PLAN_FOR_ATTEMPT,
    Relations.QUERY_FOR_TOOL_CALL,
    Relations.RESPONSE_FOR_TOOL_CALL,
    Relations.EVIDENCE_FROM_TOOL_RESPONSE,
    Relations.ANSWER_SUPPORTED_BY_EVIDENCE,
    Relations.GRADE_FOR_ANSWER,
    Relations.REWARD_FROM_TOOL_CALL,
    Relations.REWARD_FOR_ATTEMPT,
    Relations.REGIME_FOR_ATTEMPT,
    Relations.POLICY_FRAGMENT_FROM_TRACES,
    Relations.MEMORY_SNAPSHOT_CONTAINS_FRAGMENT,
    Relations.ELIGIBILITY_FOR_RUN,
    Relations.CLAIM_SUPPORTED_BY_ARTIFACT,
)
