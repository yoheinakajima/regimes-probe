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
    # Level 4 task-frame / constraint-graph relations
    CONSTRAINT_APPLIES_TO_SLOT = "constraint_applies_to_slot"
    ACTION_TESTS_CONSTRAINT = "action_tests_constraint"
    ACTION_TARGETS_SLOT = "action_targets_slot"
    EVIDENCE_SUPPORTS_CONSTRAINT = "evidence_supports_constraint"
    EVIDENCE_SUPPORTS_SLOT = "evidence_supports_slot"
    HYPOTHESIS_ASSIGNS_CANDIDATE = "hypothesis_assigns_candidate"
    HYPOTHESIS_SUPPORTED_BY_EVIDENCE = "hypothesis_supported_by_evidence"
    HYPOTHESIS_REJECTED_BY_EVIDENCE = "hypothesis_rejected_by_evidence"
    ANSWER_SUPPORTED_BY_HYPOTHESIS = "answer_supported_by_hypothesis"
    FRAME_FOR_ATTEMPT = "frame_for_attempt"
    SLOT_IN_FRAME = "slot_in_frame"
    # Open-world semantic / affordance + escalation relations (Level 4c)
    CONSTRAINT_HAS_FACET = "constraint_has_facet"
    CONSTRAINT_HAS_AFFORDANCE = "constraint_has_affordance"
    HYPOTHESIS_ASSIGNS_SLOT = "hypothesis_assigns_slot"
    ANSWER_SUPPORTED_BY_PATH = "answer_supported_by_path"
    UNRESOLVED_CONSTRAINT_BLOCKS_ANSWER = "unresolved_constraint_blocks_answer"
    EPISTEMIC_MODE_FOR_ATTEMPT = "epistemic_mode_for_attempt"
    # Variable / constant / binding distinction (Level 4e)
    SLOT_HAS_DESCRIPTOR = "slot_has_descriptor"
    SLOT_BOUND_BY_CANDIDATE = "slot_bound_by_candidate"
    SLOT_DEPENDS_ON_CONTEXT = "slot_depends_on_context"
    CONTEXT_CONSTRAINS_SLOT = "context_constrains_slot"
    CANDIDATE_SUPPORTED_BY_EVIDENCE = "candidate_supported_by_evidence"
    TARGET_SLOT_UNBOUND_UNTIL_EVIDENCE = "target_slot_unbound_until_evidence"
    KNOWN_CONTEXT_NOT_ANSWER = "known_context_not_answer"
    # Candidate-slate / frontier relations (Level 5)
    SLATE_FOR_SLOT = "slate_for_slot"
    CANDIDATE_IN_SLATE = "candidate_in_slate"
    CANDIDATE_ASSIGNED_TO_SLOT = "candidate_assigned_to_slot"
    CANDIDATE_SUPPORTS_CONSTRAINT = "candidate_supports_constraint"
    CANDIDATE_CONTRADICTS_CONSTRAINT = "candidate_contradicts_constraint"
    CANDIDATE_REJECTED_BY_EVIDENCE = "candidate_rejected_by_evidence"
    CANDIDATE_CONFIRMED_BY_EVIDENCE = "candidate_confirmed_by_evidence"
    CANDIDATE_MERGED_INTO = "candidate_merged_into"
    HYPOTHESIS_USES_CANDIDATE = "hypothesis_uses_candidate"
    HYPOTHESIS_REJECTED_BY_CONSTRAINT = "hypothesis_rejected_by_constraint"
    ACTION_TESTS_CANDIDATE = "action_tests_candidate"
    ACTION_EXPANDS_CANDIDATE = "action_expands_candidate"
    FRONTIER_ACTION_SELECTED_BECAUSE = "frontier_action_selected_because"
    EVIDENCE_UPDATES_CANDIDATE_STATUS = "evidence_updates_candidate_status"
    CANDIDATE_UNLOCKS_DEPENDENT_SLOT = "candidate_unlocks_dependent_slot"
    # Frontier controller (Level 5b)
    TOOL_CALL_FROM_FRONTIER_ACTION = "tool_call_from_frontier_action"
    HYPOTHESIS_UPDATED_AFTER_FRONTIER_ACTION = "hypothesis_updated_after_frontier_action"
    # LLM frontier proposer (Level 5c)
    PROPOSAL_TARGETS_SLOT = "proposal_targets_slot"
    PROPOSAL_TESTS_CONSTRAINT = "proposal_tests_constraint"
    PROPOSAL_USES_CANDIDATE = "proposal_uses_candidate"
    PROPOSAL_BASED_ON_STATE_CARD = "proposal_based_on_state_card"
    PROPOSAL_SELECTED_FOR_ACTION = "proposal_selected_for_action"
    PROPOSAL_REJECTED_BECAUSE = "proposal_rejected_because"
    TOOL_CALL_FROM_LLM_FRONTIER_PROPOSAL = "tool_call_from_llm_frontier_proposal"
    EVIDENCE_LINKED_TO_LLM_PROPOSAL = "evidence_linked_to_llm_proposal"
    CANDIDATE_PROMOTED_FROM_LLM_FRONTIER = "candidate_promoted_from_llm_frontier"
    # Level 5d evidence interpretation
    EVIDENCE_INTERPRETED_AS = "evidence_interpreted_as"
    INTERPRETATION_ASSERTS_CANDIDATE = "interpretation_asserts_candidate"
    INTERPRETATION_SUPPORTS_CONSTRAINT = "interpretation_supports_constraint"
    INTERPRETATION_CONTRADICTS_CONSTRAINT = "interpretation_contradicts_constraint"
    CANDIDATE_ASSERTION_ASSIGNED_TO_SLOT = "candidate_assertion_assigned_to_slot"
    CANDIDATE_ASSERTION_REJECTED_BECAUSE = "candidate_assertion_rejected_because"
    SOURCE_CLASSIFIED_AS = "source_classified_as"
    EVIDENCE_UPDATES_CANDIDATE_SLATE = "evidence_updates_candidate_slate"
    EVIDENCE_UPDATES_CONSTRAINT_STATUS = "evidence_updates_constraint_status"
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
    Relations.FRAME_FOR_ATTEMPT,
    Relations.SLOT_IN_FRAME,
    Relations.CONSTRAINT_APPLIES_TO_SLOT,
    Relations.ACTION_TARGETS_SLOT,
    Relations.ACTION_TESTS_CONSTRAINT,
    Relations.EVIDENCE_SUPPORTS_CONSTRAINT,
    Relations.EVIDENCE_SUPPORTS_SLOT,
    Relations.HYPOTHESIS_ASSIGNS_CANDIDATE,
    Relations.HYPOTHESIS_SUPPORTED_BY_EVIDENCE,
    Relations.HYPOTHESIS_REJECTED_BY_EVIDENCE,
    Relations.ANSWER_SUPPORTED_BY_HYPOTHESIS,
    Relations.CONSTRAINT_HAS_FACET,
    Relations.CONSTRAINT_HAS_AFFORDANCE,
    Relations.HYPOTHESIS_ASSIGNS_SLOT,
    Relations.ANSWER_SUPPORTED_BY_PATH,
    Relations.UNRESOLVED_CONSTRAINT_BLOCKS_ANSWER,
    Relations.EPISTEMIC_MODE_FOR_ATTEMPT,
    Relations.SLOT_HAS_DESCRIPTOR,
    Relations.SLOT_BOUND_BY_CANDIDATE,
    Relations.SLOT_DEPENDS_ON_CONTEXT,
    Relations.CONTEXT_CONSTRAINS_SLOT,
    Relations.CANDIDATE_SUPPORTED_BY_EVIDENCE,
    Relations.TARGET_SLOT_UNBOUND_UNTIL_EVIDENCE,
    Relations.KNOWN_CONTEXT_NOT_ANSWER,
    Relations.SLATE_FOR_SLOT,
    Relations.CANDIDATE_IN_SLATE,
    Relations.CANDIDATE_ASSIGNED_TO_SLOT,
    Relations.CANDIDATE_SUPPORTS_CONSTRAINT,
    Relations.CANDIDATE_CONTRADICTS_CONSTRAINT,
    Relations.CANDIDATE_REJECTED_BY_EVIDENCE,
    Relations.CANDIDATE_CONFIRMED_BY_EVIDENCE,
    Relations.CANDIDATE_MERGED_INTO,
    Relations.HYPOTHESIS_USES_CANDIDATE,
    Relations.HYPOTHESIS_REJECTED_BY_CONSTRAINT,
    Relations.ACTION_TESTS_CANDIDATE,
    Relations.ACTION_EXPANDS_CANDIDATE,
    Relations.FRONTIER_ACTION_SELECTED_BECAUSE,
    Relations.EVIDENCE_UPDATES_CANDIDATE_STATUS,
    Relations.CANDIDATE_UNLOCKS_DEPENDENT_SLOT,
    Relations.TOOL_CALL_FROM_FRONTIER_ACTION,
    Relations.HYPOTHESIS_UPDATED_AFTER_FRONTIER_ACTION,
    Relations.PROPOSAL_TARGETS_SLOT,
    Relations.PROPOSAL_TESTS_CONSTRAINT,
    Relations.PROPOSAL_USES_CANDIDATE,
    Relations.PROPOSAL_BASED_ON_STATE_CARD,
    Relations.PROPOSAL_SELECTED_FOR_ACTION,
    Relations.PROPOSAL_REJECTED_BECAUSE,
    Relations.TOOL_CALL_FROM_LLM_FRONTIER_PROPOSAL,
    Relations.EVIDENCE_LINKED_TO_LLM_PROPOSAL,
    Relations.CANDIDATE_PROMOTED_FROM_LLM_FRONTIER,
    Relations.EVIDENCE_INTERPRETED_AS,
    Relations.INTERPRETATION_ASSERTS_CANDIDATE,
    Relations.INTERPRETATION_SUPPORTS_CONSTRAINT,
    Relations.INTERPRETATION_CONTRADICTS_CONSTRAINT,
    Relations.CANDIDATE_ASSERTION_ASSIGNED_TO_SLOT,
    Relations.CANDIDATE_ASSERTION_REJECTED_BECAUSE,
    Relations.SOURCE_CLASSIFIED_AS,
    Relations.EVIDENCE_UPDATES_CANDIDATE_SLATE,
    Relations.EVIDENCE_UPDATES_CONSTRAINT_STATUS,
)
