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
    # Candidate-slate / frontier lifecycle (Level 5). These are recorded into the
    # attempt trace + graph projection (NOT the canonical event log), so they are
    # deliberately kept OUT of ALL_EVENTS (which the schema test pins at 25).
    CANDIDATE_SLATE_CREATED = "candidate_slate.created"
    CANDIDATE_EXTRACTED = "candidate.extracted"
    CANDIDATE_ASSIGNED_TO_SLOT = "candidate.assigned_to_slot"
    CANDIDATE_STATUS_CHANGED = "candidate.status_changed"
    CANDIDATE_REJECTED = "candidate.rejected"
    CANDIDATE_PROMOTED = "candidate.promoted"
    CANDIDATE_MERGED = "candidate.merged"
    HYPOTHESIS_CREATED = "hypothesis.created"
    HYPOTHESIS_UPDATED = "hypothesis.updated"
    HYPOTHESIS_REJECTED = "hypothesis.rejected"
    FRONTIER_ACTION_GENERATED = "frontier_action.generated"
    FRONTIER_ACTION_SELECTED = "frontier_action.selected"
    FRONTIER_ACTION_EXECUTED = "frontier_action.executed"
    FRONTIER_ACTION_SCORED = "frontier_action.scored"
    EVIDENCE_LINKED_TO_CANDIDATE = "evidence.linked_to_candidate"
    EVIDENCE_LINKED_TO_SLOT = "evidence.linked_to_slot"
    EVIDENCE_LINKED_TO_CONSTRAINT = "evidence.linked_to_constraint"
    # LLM frontier proposer (Level 5c)
    LLM_FRONTIER_STATE_CARD_CREATED = "llm_frontier_state_card_created"
    LLM_FRONTIER_PROPOSALS_GENERATED = "llm_frontier_proposals_generated"
    LLM_FRONTIER_PROPOSAL_VALIDATED = "llm_frontier_proposal_validated"
    LLM_FRONTIER_PROPOSAL_REJECTED = "llm_frontier_proposal_rejected"
    LLM_FRONTIER_PROPOSAL_SELECTED = "llm_frontier_proposal_selected"
    LLM_FRONTIER_PROPOSAL_EXECUTED = "llm_frontier_proposal_executed"
    LLM_FRONTIER_REPAIR_INVOKED = "llm_frontier_repair_invoked"
    LLM_FRONTIER_CACHE_HIT = "llm_frontier_cache_hit"
    # Level 5c proposal->action integrity + evidence linkage (the bug-fix layer)
    FRONTIER_ACTION_INTEGRITY_CHECKED = "frontier_action_integrity_checked"
    FRONTIER_ACTION_INTEGRITY_ERROR = "frontier_action_integrity_error"
    SELECTED_PROPOSAL_TRANSLATED_TO_ACTION = "selected_proposal_translated_to_action"
    EVIDENCE_LINKED_TO_LLM_PROPOSAL = "evidence_linked_to_llm_proposal"
    CANDIDATE_PROMOTED_FROM_LLM_FRONTIER_EVIDENCE = "candidate_promoted_from_llm_frontier_evidence"
    LLM_FRONTIER_REPAIR_TRIGGERED_REASON = "llm_frontier_repair_triggered_reason"
    TOOL_FAMILY_NORMALIZED = "tool_family_normalized"
    # Level 5d evidence interpretation (retrieval -> structured assertions)
    EVIDENCE_INTERPRETED = "evidence_interpreted"
    SOURCE_CLASSIFIED = "source_classified"
    CANDIDATE_ASSERTION_MADE = "candidate_assertion_made"
    CANDIDATE_ASSERTION_REJECTED = "candidate_assertion_rejected"
    CONSTRAINT_ASSERTION_MADE = "constraint_assertion_made"
    # Level 5d.1 canonical candidate registry + constraint-support attachment
    CANDIDATE_ASSERTION_MATERIALIZED = "candidate_assertion_materialized"
    CANONICAL_CANDIDATE_CREATED = "canonical_candidate_created"
    CANONICAL_CANDIDATE_UPDATED = "canonical_candidate_updated"
    EVIDENCE_CONSTRAINT_SUPPORT_ATTACHED = "evidence_constraint_support_attached"
    EVIDENCE_CONSTRAINT_SUPPORT_REJECTED = "evidence_constraint_support_rejected"
    WEAK_OBSERVATION_RECORDED = "weak_observation_recorded"
    CANDIDATE_LOOKUP_RESOLVED = "candidate_lookup_resolved"
    CANDIDATE_LOOKUP_FAILED = "candidate_lookup_failed"
    EV_SLOT_TRUE_CONS_FALSE_EXPLAINED = "ev_slot_true_cons_false_explained"


#: Candidate-slate / frontier event-type strings (trace + projection only).
SLATE_EVENTS: tuple[str, ...] = (
    Events.CANDIDATE_SLATE_CREATED, Events.CANDIDATE_EXTRACTED,
    Events.CANDIDATE_ASSIGNED_TO_SLOT, Events.CANDIDATE_STATUS_CHANGED,
    Events.CANDIDATE_REJECTED, Events.CANDIDATE_PROMOTED, Events.CANDIDATE_MERGED,
    Events.HYPOTHESIS_CREATED, Events.HYPOTHESIS_UPDATED, Events.HYPOTHESIS_REJECTED,
    Events.FRONTIER_ACTION_GENERATED, Events.FRONTIER_ACTION_SELECTED,
    Events.FRONTIER_ACTION_EXECUTED, Events.FRONTIER_ACTION_SCORED,
    Events.EVIDENCE_LINKED_TO_CANDIDATE, Events.EVIDENCE_LINKED_TO_SLOT,
    Events.EVIDENCE_LINKED_TO_CONSTRAINT,
)

#: LLM frontier proposer event-type strings (trace + projection only).
LLM_FRONTIER_EVENTS: tuple[str, ...] = (
    Events.LLM_FRONTIER_STATE_CARD_CREATED, Events.LLM_FRONTIER_PROPOSALS_GENERATED,
    Events.LLM_FRONTIER_PROPOSAL_VALIDATED, Events.LLM_FRONTIER_PROPOSAL_REJECTED,
    Events.LLM_FRONTIER_PROPOSAL_SELECTED, Events.LLM_FRONTIER_PROPOSAL_EXECUTED,
    Events.LLM_FRONTIER_REPAIR_INVOKED, Events.LLM_FRONTIER_CACHE_HIT,
    Events.FRONTIER_ACTION_INTEGRITY_CHECKED, Events.FRONTIER_ACTION_INTEGRITY_ERROR,
    Events.SELECTED_PROPOSAL_TRANSLATED_TO_ACTION, Events.EVIDENCE_LINKED_TO_LLM_PROPOSAL,
    Events.CANDIDATE_PROMOTED_FROM_LLM_FRONTIER_EVIDENCE,
    Events.LLM_FRONTIER_REPAIR_TRIGGERED_REASON, Events.TOOL_FAMILY_NORMALIZED,
)

#: Evidence-interpretation event-type strings (trace + projection only).
EVIDENCE_INTERPRETATION_EVENTS: tuple[str, ...] = (
    Events.EVIDENCE_INTERPRETED, Events.SOURCE_CLASSIFIED,
    Events.CANDIDATE_ASSERTION_MADE, Events.CANDIDATE_ASSERTION_REJECTED,
    Events.CONSTRAINT_ASSERTION_MADE,
    Events.CANDIDATE_ASSERTION_MATERIALIZED, Events.CANONICAL_CANDIDATE_CREATED,
    Events.CANONICAL_CANDIDATE_UPDATED, Events.EVIDENCE_CONSTRAINT_SUPPORT_ATTACHED,
    Events.EVIDENCE_CONSTRAINT_SUPPORT_REJECTED, Events.WEAK_OBSERVATION_RECORDED,
    Events.CANDIDATE_LOOKUP_RESOLVED, Events.CANDIDATE_LOOKUP_FAILED,
    Events.EV_SLOT_TRUE_CONS_FALSE_EXPLAINED,
)


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
