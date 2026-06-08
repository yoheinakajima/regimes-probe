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
and why it fell back when it did.

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
