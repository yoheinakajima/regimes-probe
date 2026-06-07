# Event, object, and relation schema

The event log is the source of truth; the graph is a deterministic projection of
it. This document is the authoritative payload schema, matching
`src/regimes_probe/activegraph_pack/{events,objects,relations}.py` and the emits
in `activegraph_pack/behaviors.py` / `tools.py`.

All identifiers are stable strings. Object ids are `"<type>#<n>"`, event ids
`"evt_<n>"`, relation ids `"rel_<n>"` — monotonic counters under a frozen clock,
so a recorded run is byte-reproducible.

## Event types (25)

Causal order within one attempt is top-to-bottom.

| event type | when | key payload fields |
| --- | --- | --- |
| `benchmark.started` | run begins | `dataset_version`, `condition`, `config` |
| `item.queued` | item enters the run | `item_id` |
| `attempt.started` | an attempt begins | `item_id`, `budget`, `explore`, `attempt_id` |
| `signature.created` | signature computed | `cluster_key`, `norm_hash`, `features` |
| `memory_snapshot.loaded` | snapshot attached (CONFIRM) | `snapshot_id`, `nearest_k` |
| `policy_fragment.selected` | fragment chosen for cluster | `cluster_key`, `fragment_id` |
| `routing_plan.created` | Level 1 routing decided | `sequence`, `attempt` |
| `query_plan.created` | Level 2 query formed | `step`, `arm`, `query` |
| `tool.requested` | a tool call is issued | `tool`, `query`, `opts`, `limit`, `attempt_id` |
| `tool.responded` | a tool returns | `tool`, `query`, `n_results`, `cost`, `response` |
| `evidence.observed` | a result is scored | `step`, `tool`, `url`, `supports`, `fresh`, `source_authority`, `content_hash` |
| `candidate_answer.created` | candidate formed | `step`, `has_answer`, `support_count` |
| `verification.completed` | verification state computed | `step`, `candidate_found`, `support_found`, `authority_ok`, `freshness_ok`, `contradiction`, `score`, `context_code` |
| `stop_decision.created` | stop/continue decided | `step`, `arm`, `stop`, `explanation` |
| `final_answer.created` | answer finalized | `has_answer`, `tool_calls` |
| `grade.completed` | grading done | `item_id`, `correct`, `abstained`, `method` |
| `reward.computed` | reward attributed | `attempt_reward`, `flags` |
| `regime.detected` | failure regime found | `regime`, `count` |
| `policy_update.proposed` | regimes loop proposes | `dominant_regimes`, `mutations`, `rationale` |
| `policy_update.applied` | memory updated / mutation applied | `attempt_id`, `cluster_key`, `families` |
| `memory.consolidated` | fragments rebuilt | `n_clusters`, `n_traces` |
| `evaluation.completed` | a gate evaluation finished | `accepted`, `metric`, `baseline_confirm`, `candidate_confirm` |
| `promotion.accepted` | update promoted | `accepted`, `optimize_gate`, `confirm_gate`, `update` |
| `promotion.rejected` | update rejected | (same shape as `promotion.accepted`) |
| `report.created` | report artifacts written | `run_id`, `paths` |

> **Note.** `object.created` and `relation.created` are ActiveGraph's own
> projection events; every object/relation below is created via those. They are
> not in the 25 domain event types but appear in the log.

## Object types (20)

| object type | data fields (answer-free unless noted) |
| --- | --- |
| `benchmark_run` | `dataset_version`, `condition`, `config` |
| `benchmark_item` | `id`, `question`, `released_at`, `source`, `meta` (gold answer **excluded** via `Item.public_dict`) |
| `question_attempt` | `item_id`, `budget`, `mode` |
| `query_signature` | `cluster_key`, `norm_hash`, `features`, `lexical` |
| `policy_memory_snapshot` | `bandits`, `traces`, `fragments`, `params`, `nearest_k` (answer-free; leakage-guarded) |
| `policy_fragment` | `cluster_key`, `tool_rewards`, `query_rewards`, `verify_rewards`, `stop_rewards`, `support_count`, `correct_count` |
| `routing_plan` | `sequence`, `explanation` (ranked tools + numeric scores) |
| `query_plan` | `step`, `arm`, `query` |
| `tool_call` | `tool`, `query`, `n_results`, `cost`, `attempt_id` |
| `evidence_observation` | `tool`, `query_arm`, `url`, `source_authority`, `published_at`, `fresh`, `supports`, `content_hash` |
| `candidate_answer` | `has_answer`, `support_count` (raw-trace layer; transient answer not persisted to memory) |
| `final_answer` | `has_answer`, `support_count`, `tool_calls` |
| `grade_result` | `item_id`, `correct`, `abstained`, `method`, `gold_norm` (grade audit only, never in memory) |
| `evidence_reward` | per-observation reward components |
| `tool_reward` | `tool`, `reward` |
| `trace_reward` | `attempt_reward`, `arm_rewards`, `flags`, `components`, `hit_index` |
| `regime_label` | `regime`, `count` |
| `policy_update` | `dominant_regimes`, `mutations`, `rationale` |
| `promotion_decision` | `accepted`, `optimize_gate`, `confirm_gate`, `update` |
| `report` | `run_id`, artifact paths |

## Relation types (14)

| relation | source → target | meaning |
| --- | --- | --- |
| `attempt_for_item` | question_attempt → benchmark_item | which item the attempt answers |
| `signature_for_attempt` | query_signature → question_attempt | the attempt's signature |
| `snapshot_used_by_attempt` | policy_memory_snapshot → question_attempt | snapshot consulted (CONFIRM) |
| `fragment_selected_for_attempt` | policy_fragment → question_attempt | fragment prior used |
| `plan_for_attempt` | routing_plan → question_attempt | routing plan |
| `query_for_tool_call` | query_plan → tool_call | query that drove a call |
| `evidence_from_tool_call` | evidence_observation → tool_call | provenance of evidence |
| `answer_supported_by_evidence` | final_answer → evidence_observation | supporting evidence |
| `grade_for_answer` | grade_result → final_answer | grade of the answer |
| `reward_for_call` | tool_reward → tool_call | reward attributed to a call |
| `reward_for_attempt` | trace_reward → question_attempt | attempt-level reward |
| `update_from_reward` | policy_update → trace_reward | what a reward changed |
| `regime_for_failure` | regime_label → question_attempt | regime label of a failure |
| `promotion_for_update` | promotion_decision → policy_update | promotion of an update |

## Determinism rules (enforced)

- Behavior bodies are pure: no `random`, `datetime.now`, `uuid.uuid4`, or
  network I/O. Time comes from a frozen clock; ids from a monotonic generator.
- The only I/O in a recorded run is the provider call inside
  `RecordingInvoker` (`activegraph_pack/tools.py`).
- `replay_check()` re-projects the log into a fresh graph and asserts the
  object/relation projection matches — the proof that the graph is a function of
  the log. `ReplayInvoker` replays recorded tool responses in order with no
  network, and raises on any request divergence.
