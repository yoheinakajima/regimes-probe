"""Canonical event-type names for the regimes-probe ActiveGraph pack.

The event log is the source of truth (ActiveGraph CONTRACT #2). Every major
action in a benchmark run emits one of these event types. See
``docs/EVENT_SCHEMA.md`` for the payload shape of each.

These are plain string constants so the pure-Python layer can emit and read
them without importing the ActiveGraph runtime. ``activegraph_pack.behaviors``
wires the same strings into ``activegraph.Event`` objects.
"""

from __future__ import annotations


class Events:
    """Namespace of canonical event-type strings."""

    BENCHMARK_STARTED = "benchmark.started"
    ITEM_QUEUED = "item.queued"
    ATTEMPT_STARTED = "attempt.started"
    SIGNATURE_CREATED = "signature.created"
    MEMORY_SNAPSHOT_LOADED = "memory_snapshot.loaded"
    POLICY_FRAGMENT_SELECTED = "policy_fragment.selected"
    ROUTING_PLAN_CREATED = "routing_plan.created"
    QUERY_PLAN_CREATED = "query_plan.created"
    TOOL_REQUESTED = "tool.requested"
    TOOL_RESPONDED = "tool.responded"
    EVIDENCE_OBSERVED = "evidence.observed"
    CANDIDATE_ANSWER_CREATED = "candidate_answer.created"
    VERIFICATION_COMPLETED = "verification.completed"
    STOP_DECISION_CREATED = "stop_decision.created"
    FINAL_ANSWER_CREATED = "final_answer.created"
    GRADE_COMPLETED = "grade.completed"
    REWARD_COMPUTED = "reward.computed"
    REGIME_DETECTED = "regime.detected"
    POLICY_UPDATE_PROPOSED = "policy_update.proposed"
    POLICY_UPDATE_APPLIED = "policy_update.applied"
    MEMORY_CONSOLIDATED = "memory.consolidated"
    EVALUATION_COMPLETED = "evaluation.completed"
    PROMOTION_ACCEPTED = "promotion.accepted"
    PROMOTION_REJECTED = "promotion.rejected"
    REPORT_CREATED = "report.created"


#: All canonical event-type strings, in causal order.
ALL_EVENTS: tuple[str, ...] = (
    Events.BENCHMARK_STARTED,
    Events.ITEM_QUEUED,
    Events.ATTEMPT_STARTED,
    Events.SIGNATURE_CREATED,
    Events.MEMORY_SNAPSHOT_LOADED,
    Events.POLICY_FRAGMENT_SELECTED,
    Events.ROUTING_PLAN_CREATED,
    Events.QUERY_PLAN_CREATED,
    Events.TOOL_REQUESTED,
    Events.TOOL_RESPONDED,
    Events.EVIDENCE_OBSERVED,
    Events.CANDIDATE_ANSWER_CREATED,
    Events.VERIFICATION_COMPLETED,
    Events.STOP_DECISION_CREATED,
    Events.FINAL_ANSWER_CREATED,
    Events.GRADE_COMPLETED,
    Events.REWARD_COMPUTED,
    Events.REGIME_DETECTED,
    Events.POLICY_UPDATE_PROPOSED,
    Events.POLICY_UPDATE_APPLIED,
    Events.MEMORY_CONSOLIDATED,
    Events.EVALUATION_COMPLETED,
    Events.PROMOTION_ACCEPTED,
    Events.PROMOTION_REJECTED,
    Events.REPORT_CREATED,
)
