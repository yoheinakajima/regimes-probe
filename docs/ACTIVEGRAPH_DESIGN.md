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
