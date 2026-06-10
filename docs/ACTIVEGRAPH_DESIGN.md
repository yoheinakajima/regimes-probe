# ActiveGraph design

How `regimes-probe` maps onto ActiveGraph, and how it honors the framework's
contract so every run is replayable, forkable, and auditable.

## Principles (from ActiveGraph)

1. **The event log is the source of truth.** Every major action emits an event.
2. **The graph is a deterministic projection of the log.** Objects and relations
   are materialized from `object.created` / `relation.created` events only.
3. **Behaviors react to events, mutate the graph, and emit more events.**
4. **Behavior bodies are deterministic.** No `random`, `datetime.now`,
   `uuid.uuid4`, or direct network I/O inside a behavior body.
5. **All network calls go through tools.** Tools are recorded as
   `tool.requested` / `tool.responded` pairs; replay reads them back.
6. **Replay verifies determinism.**

## The pack

`src/regimes_probe/activegraph_pack/` is the pack:

- `objects.py`, `events.py`, `relations.py` — the canonical type names
  (20 objects, 25 events, 14 relations). See `EVENT_SCHEMA.md`.
- `behaviors.py` — the **EventLog** and the recording behaviors.
- `tools.py` — the **recorded-tool boundary**: `RecordingInvoker` /
  `ReplayInvoker`, plus an optional `@tool` registration path for a future
  Runtime-driven integration.

### EventLog: native + faithful fallback

`EventLog` wraps the real ActiveGraph `Graph` when the `activegraph` package is
installed (the native path), constructing it with a `FrozenClock` and an
`IDGen` so timestamps and ids are deterministic. When `activegraph` is not
installed, it falls back to a small, equivalent append-only log + projector with
identical semantics (log is truth; `apply`-style projection of `object.created` /
`relation.created`). This keeps the whole project — including the replay check —
runnable on the standard library + PyYAML alone, while preferring the real
runtime when present.

```
question ─▶ EventLog.emit(domain events)        # source of truth
              │
              ├─ add_object(...) ─▶ object.created ─▶ projection
              └─ add_relation(...) ─▶ relation.created ─▶ projection
```

### Behaviors (deterministic recorders)

The recording functions in `behaviors.py` are behavior bodies:

- `record_run_start` → `benchmark.started` + `benchmark_run`.
- `record_attempt` → `item.queued`, `attempt.started`, `signature.created`,
  then drives the agent's `SearchLoop` with an `_AGLoopRecorder` that emits
  `routing_plan.created`, `query_plan.created`, `evidence.observed`,
  `candidate_answer.created`, `verification.completed`, `stop_decision.created`
  in causal order, and finally `final_answer.created`.
- `record_grade_and_reward` → `grade.completed`, `reward.computed` + the
  `grade_result` / `trace_reward` / `tool_reward` objects and reward relations.
- `record_policy_update` → applies `memory.observe(...)` and emits
  `policy_update.applied`.

Each is pure given its inputs. The only I/O is the provider call inside the
recording tool invoker.

### Tools: recorded-tool pattern

`RecordingInvoker.call(tool, query, ...)`:

1. emit `tool.requested`,
2. call `provider.search(...)` — **the only network in a recorded run**,
3. emit `tool.responded` (with the serialized response) and create a
   `tool_call` object,
4. append the response to an in-order `recorded` list.

`ReplayInvoker` consumes that `recorded` list, returning each response in order
and raising on any `(tool, query)` divergence — so a replay never touches the
network and is provably the same request sequence.

This mirrors ActiveGraph's CONTRACT v0.7 recorded-tool semantics: replay reads
responses from the log instead of re-invoking.

## Replay, fork, audit

- **Replay.** `replay_check(log)` builds a fresh graph, replays every event
  (via ActiveGraph's `_replay_event` on the native path, or the fallback
  projector), and asserts the object/relation projection matches the live one.
  `results/{run_id}/replay_check.md` records the outcome.
- **Fork.** Because state is a pure function of the log, a run can be forked at
  any event boundary by truncating the log and continuing — the native
  ActiveGraph `Runtime.fork` does exactly this; the fallback supports the same
  by re-projection.
- **Audit.** Every answer traces back through `answer_supported_by_evidence` →
  `evidence_from_tool_call` → `tool.responded` (URL + page `content_hash`), and
  every policy change through `reward_for_attempt` → `update_from_reward`.

## Three layers (kept distinct)

1. **Raw event trace** — the full log, including transient answer text in the
   raw-trace layer (e.g. `candidate_answer`). This is the audit record.
2. **Policy memory / priors** — answer-free reward statistics, embeddings, and
   policy fragments (`POLICY_MEMORY.md`). This is what the hot path reads.
3. **Optional natural-language lessons** — a future layer that *describes*
   fragments in prose; not implemented in v0.

Leakage controls (`LEAKAGE_CONTROLS.md`) live at the boundary between layer 1
and layer 2: `assert_no_answer_leakage` runs on every object written into policy
memory and on the frozen snapshot.

## Exported graph projection (`graph_projection.json`)

Every completed run exports a compact, standardized, secret-free projection of
the run graph to `results/{run_id}/graph_projection.json`
(`eval/projection.py`). It is the artifact a reviewer (or an offline forked
ablation) reads instead of transient in-memory objects.

**Object types** (`activegraph_pack/objects.py:PROJECTION_OBJECTS`):
`benchmark_run`, `benchmark_item`, `question_attempt`, `routing_plan`,
`query_plan`, `tool_call`, `tool_response`, `evidence_observation`,
`answer_attempt`, `grade_result`, `reward_assignment`, `failure_regime`,
`policy_fragment`, `memory_snapshot`, `eligibility_verdict`, `claim_candidate`,
`report`. Several are the public names of event-log objects
(`final_answer`→`answer_attempt`, `trace_reward`/`tool_reward`→`reward_assignment`,
`regime_label`→`failure_regime`, `policy_memory_snapshot`→`memory_snapshot`); see
`CANONICAL_PROJECTION_TYPE`. The Level-4 task-frame layer (see `QUERY_POLICY.md`)
adds `task_frame`, `latent_slot`, `constraint`, `hypothesis`, `slot_assignment`,
`evidence_record`, `epistemic_action`, and `read_value_decision` — emitted only for
attempts run with `policy.enable_task_frame` on (off in the byte-identical demo).

**Relation types** (`relations.py:PROJECTION_RELATIONS`): `attempt_for_item`,
`plan_for_attempt`, `query_for_tool_call`, `response_for_tool_call`,
`evidence_from_tool_response`, `answer_supported_by_evidence`, `grade_for_answer`,
`reward_from_tool_call`, `reward_for_attempt`, `regime_for_attempt`,
`policy_fragment_from_traces`, `memory_snapshot_contains_fragment`,
`eligibility_for_run`, `claim_supported_by_artifact`. Level-4 relations:
`frame_for_attempt`, `slot_in_frame`, `constraint_applies_to_slot`,
`action_targets_slot`, `action_tests_constraint`, `evidence_supports_constraint`,
`evidence_supports_slot`, `hypothesis_assigns_candidate`,
`hypothesis_supported_by_evidence`, `hypothesis_rejected_by_evidence`,
`answer_supported_by_hypothesis` — making the constraint graph an auditable
subgraph: which action tested which constraint on which slot, and which evidence
advanced or rejected which hypothesis. The `task_frame` node also carries parser
provenance (`parser_used` deterministic|llm, `prompt_version`, `prompt_hash`,
`fallback_reason`) so a reviewer can see whether a frame came from the deterministic
v0 parser or the optional cached/validated LLM parser (`QUERY_POLICY.md` Level 4b),
and why it fell back when it did. The LLM parser's model calls flow through the same
`RecordingCache` as the answerer, so parser parses are recorded/replayed like any
other tool call (dry-run and replay never call the model), and the report carries
answer-free parser accounting (`task_frame_parser`: model, model calls, cache
hits/misses, fallbacks). Whether a final answer is constraint-supported is recorded
on the attempt's `frame_coverage` (`answer_support_gate` + `missing_support_reasons`)
from the strict `evaluate_answer_support` gate — so an `answer_supported_by_hypothesis`
edge reflects supported evidence, not a merely-bound slot.

**Open-world semantics + affordances in the graph (Level 4c/4d).** Constraints
project as `semantic_constraint` nodes carrying BOTH the raw free-form
`semantic_label`/`semantic_facets` AND the derived closed-set `affordances` — kept
side by side, never collapsed. Each gets `constraint_has_facet` edges to
`constraint_facet` nodes and `constraint_has_affordance` edges to
`operational_affordance` nodes; an unresolved blocking constraint gets an
`unresolved_constraint_blocks_answer` edge. The escalation decision projects as an
`epistemic_mode_decision` node (`epistemic_mode_for_attempt`), and a supported answer
projects an explicit `answer_support_path` node (`answer_supported_by_path`) capturing
the answer → hypothesis → slot → evidence → constraint chain; hypotheses also get
`hypothesis_assigns_slot` edges. This is deliberate: by recording **raw semantics,
derived affordances, the actions taken, and the outcome** together, the ontology can
**evolve from traces** — future learning can ask which affordances actually paid off,
which facets needed which tools, and which epistemic modes avoided wasted work,
without freezing a constraint enum up front.

**Variables vs. constants vs. bindings in the graph (Level 4e).** Slots project as
`slot_variable` nodes carrying `slot_status` (unbound_variable/known_constant/
candidate_binding/derived_value), `raw_slot_id`, and `bound_value`; each gets a
`slot_has_descriptor` edge to a `slot_descriptor` node (the raw text describing the
unknown) and a `binding_status` node — a target variable is linked
`target_slot_unbound_until_evidence` until a candidate binds it. Constants GIVEN in
the question are `known_context_term` nodes tied `known_context_not_answer` to the
frame, and a descriptor that references one gets `slot_depends_on_context`. When a
hypothesis proposes a value, it projects a `candidate_binding` node
(`slot_bound_by_candidate`) — the moment a slot stops being unbound. This records, in
one auditable place, the **raw descriptor, the inferred slot status, the validation
decision, and the later evidence binding**, so future learning can distinguish
parsers that write good variable descriptors from parsers that prematurely bind
answers, and planners that bind slots correctly from evidence from validators that
over-reject useful frames.

**Bounding / safety.** Every attempt gets compact `question_attempt`/
`answer_attempt`/`grade_result`/`reward_assignment` (+ `failure_regime` when it
failed) nodes; heavy sub-nodes (`query_plan`/`tool_call`/`tool_response`/
`evidence_observation`) are expanded only for the first `detail_limit` attempts.
All text is a length-bounded preview — no full pages, no secrets. The projection
carries model **predictions** (bounded) but **not gold answers** — gold lives
only in the audit `debug_questions.jsonl`. The projection records the event-log
replay summary (`event_log.projection_matches`) as its provenance.

### First-class verdict / regime / fragment objects

- **`eligibility_verdict`** — one flat object per run with every gate that feeds
  the two verdicts (structural + headline). Mirrored into `report.json`
  (`eligibility_verdict`) and used by `scripts/generate_claims.py`. See
  `REPORTING.md`.
- **`failure_regime`** — a structured object per failed attempt
  (`eval/failure_regime.py`): the canonical regime name, the answer-free flags
  that triggered it, condition/budget/item/attempt ids, and (in the projection)
  a `regime_for_attempt` edge to its attempt. Provider failures surface here as
  `provider_error`, not bare strings.
- **`policy_fragment` lineage** — consolidation records, per fragment,
  `fragment_id`, `source_trace_ids` (answer-free trace content hashes),
  `source_attempt_ids`, `top_tool_rewards`/`top_query_rewards`, and
  `consolidation_event_id`. Edges `memory_snapshot_contains_fragment` and
  `policy_fragment_from_traces` make the lineage explicit. Ids/hashes only —
  the leakage scan still passes.

## Offline forked ablations

Because state is a pure function of the log and every provider/model/tool outcome
is in the recording cache, a run can be **forked offline**: reuse the cached
outcomes in `replay` mode and rerun policy variants without spending. Forks are
marked `offline_fork=true` + `parent_run_id` and are never headline-eligible. See
`OFFLINE_FORK_ABLATIONS.md`.

## Candidate slates + frontier as native graph state (Level 5)

The multi-hop candidate-slate / frontier layer (`agent/candidate_frontier.py`,
`QUERY_POLICY.md` Level 5) is deliberately **not** a sidecar that only lives in the
Python loop — it is event-sourced and projected so it is replayable, forkable, and
learnable from traces. Lifecycle **events** (kept in `Events` + the `SLATE_EVENTS`
tuple, recorded into the attempt trace + projection rather than the canonical 25-event
log) cover the whole slate life: `candidate_slate.created`, `candidate.extracted`,
`candidate.assigned_to_slot`, `candidate.status_changed`, `candidate.rejected`,
`candidate.promoted`, `candidate.merged`, `hypothesis.created/updated/rejected`,
`frontier_action.generated/selected/scored`, and the three `evidence.linked_to_*`
edges — each with a deterministic id and an injected clock, no secrets, no unbounded
page text (previews/hashes/ids only). The **projection** adds first-class objects
(`candidate_slate`, `slot_candidate`, `candidate_status`, `candidate_rejection`,
`candidate_promotion`, `candidate_merge`, `hypothesis_state`, `frontier_action`,
`frontier_decision`, `frontier_score`, `candidate_evidence_link`,
`candidate_constraint_status`) and relations (`slate_for_slot`, `candidate_in_slate`,
`candidate_assigned_to_slot`, `candidate_supports_constraint`,
`candidate_contradicts_constraint`, `candidate_rejected_by_evidence`,
`candidate_confirmed_by_evidence`, `candidate_merged_into`, `hypothesis_uses_candidate`,
`hypothesis_rejected_by_constraint`, `action_tests_candidate`,
`action_expands_candidate`, `frontier_action_selected_because`,
`evidence_updates_candidate_status`, `candidate_unlocks_dependent_slot`, …). Because
the projection is a deterministic function of the trace (a pure function of the
recorded tool/model outputs), candidate slates and frontier decisions **re-project
identically**; an offline fork can re-score frontier actions under different
reward/priority settings from the cached outcomes, and refuses rather than spends if a
frontier action needs a missing provider output. This is what lets future learning ask
which candidate/frontier strategies actually work, over many traces, rather than
guessing.

When the **frontier controller** is enabled (Level 5b, `QUERY_POLICY.md`), the same
source-of-truth discipline holds for the *control* decisions: the selected action is
recorded (`frontier_action.selected`/`scored`), its outcome is recorded
(`frontier_action.executed` with success/evidence-progress, or
`frontier_action_unexecutable` on a safe fallback), every executed tool call carries a
`frontier_action_id` and projects a `tool_call_from_frontier_action` edge to its
driving action, and a hypothesis update projects
`hypothesis_updated_after_frontier_action`. So a reviewer can reconstruct, purely from
`graph_projection.json`, *which* frontier action drove *which* tool call and *what* it
achieved — and an offline fork can replay the cached tool outputs while re-scoring the
frontier under a different EIG/reward setting, refusing to spend if a needed output is
absent. Controller decisions are never hidden Python state.

### LLM frontier proposer as native graph state (Level 5c)

The optional LLM frontier proposer (`QUERY_POLICY.md` Level 5c, `agent/llm_frontier.py`)
keeps the same discipline: **the LLM proposes, ActiveGraph records, deterministic code
disposes.** The bounded state the model reasons over is itself a graph object — the
`research_state_card` (projected from the `TaskFrame` + `CandidateFrontier`, never from
gold) — and so is everything downstream of it. New **projection objects**
(`activegraph_pack/objects.py:PROJECTION_OBJECTS`): `research_state_card`,
`llm_frontier_prompt`, `llm_frontier_proposal`, `llm_frontier_validation`,
`llm_frontier_selection`. New **events** (`LLM_FRONTIER_EVENTS`, kept out of the frozen
`ALL_EVENTS` tuple): `llm_frontier_state_card_created`, `llm_frontier_repair_invoked`,
`llm_frontier_proposals_generated`, `llm_frontier_proposal_validated`,
`llm_frontier_proposal_rejected`, `llm_frontier_proposal_selected`,
`llm_frontier_proposal_executed`, `llm_frontier_model_called`. New **projection relations**
(`PROJECTION_RELATIONS`): `proposal_targets_slot`, `proposal_tests_constraint`,
`proposal_uses_candidate`, `proposal_based_on_state_card`, `proposal_selected_for_action`,
`proposal_rejected_because`, and `tool_call_from_llm_frontier_proposal` (the `lfp_`-prefixed
`frontier_action_id` on a `CallRecord` links the executed tool call back to its proposal).

A reviewer can therefore reconstruct, purely from `graph_projection.json`: the exact card
the model saw (by hash), every proposal it returned, **why** each was accepted or rejected
(reason on the `proposal_rejected_because` edge), which one was selected and scored, and
which tool call it ultimately drove. Because each model call is keyed by
`prompt_fingerprint | model | card_hash` and cached, the whole layer **re-projects
identically** on replay and an offline fork can re-validate/re-score the cached proposals
under a different policy **without any model call** — refusing to spend if the proposal
cache is absent. The proposer's own settings are pinned into `ConditionSpec` so a
comparison cannot silently differ in how queries were composed.

**Proposal → action → evidence integrity is itself event-sourced.** The first repair run
revealed the selected proposal's slot/constraints were not being carried into the executed
action; the fix keeps the whole chain in the graph so a reviewer can prove faithfulness
from `graph_projection.json` alone. New **events** (added to `LLM_FRONTIER_EVENTS`, still
outside the frozen `ALL_EVENTS`): `selected_proposal_translated_to_action`,
`frontier_action_integrity_checked`, `frontier_action_integrity_error`,
`evidence_linked_to_llm_proposal`, `candidate_promoted_from_llm_frontier_evidence`,
`llm_frontier_repair_triggered_reason`, `tool_family_normalized`. New **relations**
(`PROJECTION_RELATIONS`): `evidence_linked_to_llm_proposal`,
`candidate_promoted_from_llm_frontier`. The `llm_frontier_selection` node now exposes the
audit fields directly — `proposal_slot_id` vs `executed_slot_id`, `proposal_constraint_ids`
vs `executed_constraint_ids`, `integrity_passed`, `repair_trigger_reason`,
`normalized_tool`/`normalized_tool_family`, and `progress_components` — and the executed
tool call links to its proposal via the `lfp_`-prefixed `frontier_action_id`
(`tool_call_from_llm_frontier_proposal`). When integrity fails, the proposal is **refused**
(the deterministic planner drives that step instead), so a faithless translation can never
reach the evidence layer; and because every field is persisted on the step trace, the whole
chain **re-projects identically on replay** and an offline fork can re-audit it with zero
model calls.

### Evidence interpretation as native graph state (Level 5d)

The evidence interpreter (`QUERY_POLICY.md` Level 5d, `agent/evidence_interpreter.py`) makes
the step from *retrieval* to *understanding* a first-class, replayable part of the graph:
candidate slates and constraint support are now derived from recorded **assertions**, not
from transient n-gram extraction. Each ingested result emits an `EvidenceInterpretation`
recorded on the frontier and projected. New **projection objects**
(`activegraph_pack/objects.py:PROJECTION_OBJECTS`): `evidence_interpretation`,
`candidate_assertion`, `constraint_assertion`, `source_role_classification`,
`evidence_noise_classification`. New **events** (`EVIDENCE_INTERPRETATION_EVENTS`, outside
the frozen `ALL_EVENTS`): `evidence_interpreted`, `source_classified`,
`candidate_assertion_made`, `candidate_assertion_rejected`, `constraint_assertion_made`. New
**relations** (`PROJECTION_RELATIONS`): `evidence_interpreted_as`,
`interpretation_asserts_candidate`, `interpretation_supports_constraint`,
`interpretation_contradicts_constraint`, `candidate_assertion_assigned_to_slot`,
`candidate_assertion_rejected_because`, `source_classified_as`,
`evidence_updates_candidate_slate`, `evidence_updates_constraint_status`.

So a reviewer can reconstruct, purely from `graph_projection.json`: how each result was
typed (source role), which extracted entities became candidates vs were rejected and **why**
(the `candidate_assertion_rejected_because` edge), which constraints each result supported
(with a quote), and exactly which slate/constraint updates each interpretation caused. The
classification is a pure, deterministic function of the recorded result text + frame, so the
slates/assertions **re-project identically** on replay; the optional LLM source-role hook is
cached by `prompt_hash | evidence_hash | frame_hash | model`, so even with it enabled a
replay makes no model call and an offline fork can re-interpret cached results for free.
Because slates are populated from assertions on non-noise sources only, no answer text from
a definition/UI/contaminated page can leak into policy memory.

**One canonical candidate registry (Level 5d.1).** An accepted candidate assertion is not a
sidecar record — it *materializes* the canonical `SlotCandidate` the frontier, hypotheses,
and verifier all share, and that link is event-sourced. New **events**
(`EVIDENCE_INTERPRETATION_EVENTS`): `candidate_assertion_materialized`,
`canonical_candidate_created`, `canonical_candidate_updated`,
`evidence_constraint_support_attached`, `evidence_constraint_support_rejected`,
`weak_observation_recorded`, `candidate_lookup_resolved`, `candidate_lookup_failed`,
`ev_slot_true_cons_false_explained`. New **relations** (`PROJECTION_RELATIONS`):
`assertion_materializes_candidate`, `canonical_candidate_for_slot`,
`evidence_supports_selected_constraint`, `proposal_resolves_candidate`,
`weak_observation_not_candidate`. So `graph_projection.json` shows, for each result, which
assertion became which canonical candidate, which selected constraint it supported (with a
quote), which proposal a verifier resolved to that same candidate, and why an
`ev→slot-true / ev→cons-false` interpretation produced no constraint support. The
resolution (`resolve_candidate`) and recognizer support are pure deterministic functions of
the recorded text + frame, so the whole registry re-projects identically on replay.

**Read scheduling + support-consistency are event-sourced (Level 5e).** New **events**:
`read_scheduled` (a `read_candidate_source` was selected — often `forced_read_after_no_support`),
`read_interpreted` (a page-body read was folded in, with its new-support delta), and
`support_dropped` (a per-candidate support that failed to land on the evidence record, with a
reason). The frontier's `metrics()` now projects `read_executed_count`,
`forced_read_after_no_support_count`, `read_starvation_count`, `support_from_read_count`, and
`support_dropped_count`; the verifier's `candidate_lookup_failed` carries the
`nonexistent_candidate` breakdown. A reviewer can replay, from `graph_projection.json`,
which step forced a read, whether that read produced support, and exactly where (if
anywhere) support was dropped — the support path is auditable end to end, and the read/streak
logic is a pure function of the recorded support deltas so it re-projects identically.

**The LLM evidence judge is projected like any other judgment (Level 5f).** Each
`EvidenceJudgment` is a first-class object (`evidence_judgment`) linked to its source
interpretation (`evidence_judgment_from_source`), its candidate assertion
(`evidence_judgment_for_candidate_constraint`), and the constraint it
supports/partially-supports/contradicts (`evidence_judgment_supports_constraint` /
`_partial_constraint` / `_contradicts_constraint`). New **events**:
`llm_evidence_judgment.created/accepted/rejected`, `evidence_judgment_requires_read`,
`evidence_judgment_supports/partially_supports/contradicts_candidate_constraint`,
`evidence_judgment_rejected_reason`, `skipped_read_after_requires_read`. Each judgment node
carries `judgment_id`, `model`, `prompt_hash`, `input_hash`, `cache_hit`, and `mode`
(deterministic|llm), so a reviewer can replay exactly which model call (or cached entry)
produced each support decision and verify — from `graph_projection.json` alone — that no
support came from a contaminated source and that every `full_support` is quote-backed.
Because the judge is keyed by `prompt_fingerprint | model | triple_hash` and replay reads
the cache, the whole judged support path re-projects identically with zero model calls.

The Level-5f hardening adds **invariant metrics** (not new object types): read-path
accounting (`read_failed_zero_chars_count`, `read_fallback_*`, pinned-0
`read_requires_url_violation_count`) is attached to the candidate-frontier trace
(`read_accounting`) so it aggregates per cell; confirm-gate invariants
(`confirmed_hypothesis_with_unresolved_blocking_count`,
`blocking_constraint_partial_support_confirmed_count`, …) and post-model support-contract
invariants (`full_support_without_named_candidate_count`,
`full_support_from_generic_descriptor_count`, `relational_support_without_object_anchor_count`,
`full_or_partial_support_from_contaminated_source_count`) are computed deterministically from
the projected slate/judgment state, so a reviewer can confirm from artifacts alone that no
hypothesis confirmed with an unresolved blocking constraint and no support came from a
generic descriptor or contaminated source.

Level 5g makes the **read-intent lifecycle** event-sourced: `read_desired`, `read_selected`,
`read_blocked_no_url`, `read_blocked_disallowed_tool`, and
`read_blocked_unsafe_or_contaminated_url` are registered events emitted by the frontier, so a
reviewer can reconstruct from the log *why* a read was desired, whether it had a clean URL, and
whether it executed — closing the gap where forced reads silently became searches. The
safety invariant that **prompt text is not evidence** is enforced at the projection level too:
a blocking constraint carries `supporting_evidence_ids` only when a non-contaminated evidence
event with a distinctive anchor supported it, so
`initial_blocking_constraint_resolved_without_evidence_count` is a deterministic function of
the projected constraint state and is pinned 0. Reads against a candidate's recorded clean
`source_urls` re-project identically. (Projecting each read-intent state as its own graph
*object* is the documented follow-up; the events are already in the log.)

Level 5h promotes the read→judge **obligation** to a persistent, event-sourced object: a judge
`requires_read` emits `read_required_by_judge` and creates a `PendingReadJudgment` keyed by the
`(candidate, slot, constraint, source_url)` triple. When the source is later read, the body is
routed back into a re-judgment of that *same* triple — `read_selected_for_pending_judgment` →
`read_completed_for_pending_judgment` → `read_passage_selected` → `read_judged_after_read` →
`read_judgment_resolved`/`read_judgment_still_unresolved` — so a reviewer can reconstruct, from
the log alone, that a requires_read was actually *closed against the fetched page*, which
passage closed it, and why (or why not). The targeted passage selection is deterministic and
pure (`extract_passages`), so the re-judgment re-projects identically; the judge never sees the
truncated snippet that created the obligation, which makes
`judge_reused_truncated_excerpt_after_full_read_count` and `read_head_only_judgment_count`
deterministic projection invariants pinned 0. The new target-binding, seed-floor,
abstain-admissibility, source-hygiene, dependency-staging, and anchor-gate-relaxation decisions
are all event-backed too (`bind_target_answer_slot_*`, `seed_query_generic_blocked`,
`abstain_withheld_executable_action_available`, `source_acquisition_rejected`,
`dependent_slot_search_deferred`, `proposal_gate_relaxed`/`generic_fallback_blocked`), and the
candidate-local support label is `slot_candidate_supported` (never confused, in the projection,
with the global answer-support gate).
