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
