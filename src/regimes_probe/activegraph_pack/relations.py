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
    EVIDENCE_FROM_TOOL_CALL = "evidence_from_tool_call"
    ANSWER_SUPPORTED_BY_EVIDENCE = "answer_supported_by_evidence"
    GRADE_FOR_ANSWER = "grade_for_answer"
    REWARD_FOR_CALL = "reward_for_call"
    REWARD_FOR_ATTEMPT = "reward_for_attempt"
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
