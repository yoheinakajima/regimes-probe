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
    # Level 4 task-frame / constraint-graph objects
    TASK_FRAME = "task_frame"
    LATENT_SLOT = "latent_slot"
    CONSTRAINT = "constraint"
    HYPOTHESIS = "hypothesis"
    SLOT_ASSIGNMENT = "slot_assignment"
    EVIDENCE_RECORD = "evidence_record"
    EPISTEMIC_ACTION = "epistemic_action"
    READ_VALUE_DECISION = "read_value_decision"
    # Open-world semantic / affordance + escalation objects (Level 4c)
    EPISTEMIC_MODE_DECISION = "epistemic_mode_decision"
    SEMANTIC_CONSTRAINT = "semantic_constraint"
    CONSTRAINT_FACET = "constraint_facet"
    OPERATIONAL_AFFORDANCE = "operational_affordance"
    ANSWER_SUPPORT_PATH = "answer_support_path"
    # Variable / constant / binding distinction (Level 4e)
    KNOWN_CONTEXT_TERM = "known_context_term"
    SLOT_VARIABLE = "slot_variable"
    CANDIDATE_BINDING = "candidate_binding"
    SLOT_DESCRIPTOR = "slot_descriptor"
    BINDING_STATUS = "binding_status"
    # Candidate-slate / frontier layer (Level 5)
    CANDIDATE_SLATE = "candidate_slate"
    SLOT_CANDIDATE = "slot_candidate"
    CANDIDATE_STATUS = "candidate_status"
    CANDIDATE_REJECTION = "candidate_rejection"
    CANDIDATE_PROMOTION = "candidate_promotion"
    CANDIDATE_MERGE = "candidate_merge"
    HYPOTHESIS_STATE = "hypothesis_state"
    FRONTIER_ACTION = "frontier_action"
    FRONTIER_DECISION = "frontier_decision"
    FRONTIER_SCORE = "frontier_score"
    CANDIDATE_EVIDENCE_LINK = "candidate_evidence_link"
    CANDIDATE_CONSTRAINT_STATUS = "candidate_constraint_status"
    # LLM frontier proposer (Level 5c)
    RESEARCH_STATE_CARD = "research_state_card"
    LLM_FRONTIER_PROPOSAL = "llm_frontier_proposal"
    LLM_FRONTIER_VALIDATION = "llm_frontier_validation"
    LLM_FRONTIER_SELECTION = "llm_frontier_selection"
    LLM_FRONTIER_PROMPT = "llm_frontier_prompt"
    # Level 5d evidence interpretation
    EVIDENCE_INTERPRETATION = "evidence_interpretation"
    CANDIDATE_ASSERTION = "candidate_assertion"
    CONSTRAINT_ASSERTION = "constraint_assertion"
    SOURCE_ROLE_CLASSIFICATION = "source_role_classification"
    EVIDENCE_NOISE_CLASSIFICATION = "evidence_noise_classification"
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
    Objects.TASK_FRAME,
    Objects.LATENT_SLOT,
    Objects.CONSTRAINT,
    Objects.HYPOTHESIS,
    Objects.SLOT_ASSIGNMENT,
    Objects.EVIDENCE_RECORD,
    Objects.EPISTEMIC_ACTION,
    Objects.READ_VALUE_DECISION,
    Objects.EPISTEMIC_MODE_DECISION,
    Objects.SEMANTIC_CONSTRAINT,
    Objects.CONSTRAINT_FACET,
    Objects.OPERATIONAL_AFFORDANCE,
    Objects.ANSWER_SUPPORT_PATH,
    Objects.KNOWN_CONTEXT_TERM,
    Objects.SLOT_VARIABLE,
    Objects.CANDIDATE_BINDING,
    Objects.SLOT_DESCRIPTOR,
    Objects.BINDING_STATUS,
    Objects.CANDIDATE_SLATE,
    Objects.SLOT_CANDIDATE,
    Objects.CANDIDATE_STATUS,
    Objects.CANDIDATE_REJECTION,
    Objects.CANDIDATE_PROMOTION,
    Objects.CANDIDATE_MERGE,
    Objects.HYPOTHESIS_STATE,
    Objects.FRONTIER_ACTION,
    Objects.FRONTIER_DECISION,
    Objects.FRONTIER_SCORE,
    Objects.CANDIDATE_EVIDENCE_LINK,
    Objects.CANDIDATE_CONSTRAINT_STATUS,
    Objects.RESEARCH_STATE_CARD,
    Objects.LLM_FRONTIER_PROPOSAL,
    Objects.LLM_FRONTIER_VALIDATION,
    Objects.LLM_FRONTIER_SELECTION,
    Objects.LLM_FRONTIER_PROMPT,
    Objects.EVIDENCE_INTERPRETATION,
    Objects.CANDIDATE_ASSERTION,
    Objects.CONSTRAINT_ASSERTION,
    Objects.SOURCE_ROLE_CLASSIFICATION,
    Objects.EVIDENCE_NOISE_CLASSIFICATION,
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
