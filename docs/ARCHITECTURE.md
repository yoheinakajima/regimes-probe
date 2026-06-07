# Architecture

`regimes-probe` is an [ActiveGraph](https://github.com/yoheinakajima)-native benchmark
and improvement loop for **learning epistemic search behavior from traces**. It learns
**procedural** epistemic policy — *where to look, how to query, when to verify, when to
stop* — and explicitly **not** factual answers. No benchmark answer is ever stored in
hot-path policy memory.

It extends the spirit of `yoheinakajima/regimes` (which tested failure-regime repair via
`OPTIMIZE`/`CONFIRM` transforms) and broadens the fix space from a narrow repair target to
a set of **epistemic policy seams**: route, query, verify, stop, assemble, answer,
consolidate memory.

## Core hypothesis

> An otherwise identical web-search agent equipped with ActiveGraph **procedural memory**
> and a **contextual-bandit epistemic policy** can improve **correct answers per tool
> call** (`correct_per_tool_call`) on held-out browsing questions — **without changing
> base model weights** and **without storing benchmark answers**.

- **Level 1** = contextual-bandit **tool routing**.
- **Level 2** = contextual-bandit **epistemic policy** (query formulation, verification,
  stopping).

## ActiveGraph invariants

`regimes-probe` is built on ActiveGraph, an event-sourced graph runtime. The following
invariants are load-bearing for this project and constrain every component:

1. **The event log is the source of truth.** The graph is a *deterministic projection* of
   the event log.
2. **Behaviors react to events and must be deterministic.** No `random`, no
   `datetime.now()`, no `uuid`, no network access inside behaviors.
3. **All network goes through ActiveGraph tools.** Search providers, page fetches, and
   model calls are wrapped as tools so their requests/responses are recorded as events.
4. **Replay verifies determinism.** Re-projecting the event log must reproduce the graph
   and all derived rewards/decisions byte-for-byte (modulo recorded tool responses).

Concretely: any nondeterminism (sampling, timestamps, IDs, search results) is injected at
the **event boundary** (recorded `tool.responded`, `attempt.started`, etc.), never
generated inside a behavior. This is what makes the cold-path reward attribution and the
`regimes`-style improvement loop reproducible.

## The three layers (do not conflate)

| Layer | What it is | Retrievable by answerer in main condition? |
| --- | --- | --- |
| **(1) Raw event trace** | The full event log of every attempt: prompts, `tool.requested`/`tool.responded`, URLs, page hashes, `evidence.observed`, grades. Source of truth, archived. | **No.** Archived for analysis/replay only. Direct retrieval by the answerer is a leakage path. |
| **(2) Policy memory / priors** | Compressed, *answer-free* procedural statistics keyed by `query_signature`: bandit parameters, per-seam priors, `policy_fragment`s. This is what the hot path reads. | **Yes** — this is the learned object under test. Contains no final benchmark answers. |
| **(3) Optional natural-language lessons** | Human/LLM-readable summaries distilled from regimes analysis (e.g. "freshness-sensitive signatures favor `news_search` then verify"). Generated cold-path, gated. | Optional, advisory; must be answer-free and gated like any policy update. |

The benchmark integrity argument rests on this separation: **only Layer 2 (and optionally
Layer 3) feed the hot path, and neither contains benchmark answers.** Layer 1 exists for
reward attribution, replay, and offline analysis.

## Hot path vs cold path

### Hot path (per `question_attempt`, online)

```
question (benchmark_item)
   |
   v
[embed] --> query_signature              # HashEmbedder (tests) / OpenAI embeddings (live)
   |          (freshness-sensitive, source-authority-sensitive,
   |           answer-shape-constrained, query-fragile, evidence-sparse,
   |           staleness-prone, multihop-likely, verification-heavy)
   v
[policy memory lookup]                    # load frozen policy_memory_snapshot, priors for signature
   |
   v
[contextual bandit scoring]               # score seams: route / query / verify / stop
   |
   v
[budgeted plan]                           # routing_plan + query_plan under budget in {1,3,5,10}
   |
   v
[tool calls] --(tool.requested/responded)--> evidence_observation(s)
   |                                          (all via ActiveGraph tools)
   v
[assemble + answer]                       # candidate_answer -> verification -> stop_decision
   |
   v
final_answer
```

### Cold path (per `benchmark_run`, offline / after attempts)

```
final_answer + grade_result
   |
   v
[grading]                                 # grade.completed (correct / incorrect / abstain)
   |
   v
[evidence-gain scoring]                   # evidence_reward per evidence_observation
   |
   v
[reward attribution]                      # tool_reward (per call), trace_reward (per attempt)
   |                                       # reward_for_call, reward_for_attempt
   v
[regime detection]                        # regime_label: route_miss, query_miss, ...
   |
   v
[policy update]                           # policy_update.proposed -> applied (OPTIMIZE only)
   |
   v
[consolidation]                           # memory.consolidated -> next policy_memory_snapshot
   |
   v
[optional lesson generation]              # Layer 3, advisory, gated
   |
   v
[promotion gate]                          # promotion.accepted / rejected via CONFIRM
```

The two paths meet at the `policy_memory_snapshot`: the cold path produces snapshots, the
hot path consumes a **frozen** snapshot. `CONFIRM` always runs against a frozen snapshot
(`confirm_uses_frozen_snapshot=true`) and never mutates memory
(`confirm_updates_memory=false`). `OPTIMIZE` may explore (`optimize_allows_exploration=true`).

## Policy seams

| Level | Seam | Hot-path object | Cold-path reward signal |
| --- | --- | --- | --- |
| 1 | **route** (which tool) | `routing_plan` | `tool_reward`, `first_tool_hit_rate` |
| 2 | **query** (how to phrase) | `query_plan` | `evidence_reward`, `evidence_gain_per_call` |
| 2 | **verify** (confirm candidate) | `verification.completed` | `verification_miss` regime, support match |
| 2 | **stop** (when to halt) | `stop_decision.created` | `over_search`/`under_search`, `false_stop_rate` |
| (assemble / answer / consolidate are supporting seams) | | `candidate_answer`, `final_answer`, `memory.consolidated` | `trace_reward` |

## Regimes-style improvement

Failure attribution uses **generic, non-topical** regimes (see canonical schema):
`route_miss`, `query_miss`, `evidence_sparse`, `stale_evidence`, `over_search`,
`under_search`, `verification_miss`, `stop_too_early`, `stop_too_late`,
`answer_extraction_miss`, `contradiction_unresolved`, `support_answer_mismatch`.

The loop:

1. **Detect** dominant `regime_label`s on the `OPTIMIZE` split (`regime.detected`).
2. **Propose** parameter changes to the contextual bandit (exploration rate, priors),
   reward weights, query-template set, or stop/verify thresholds
   (`policy_update.proposed`).
3. **Apply** under `OPTIMIZE` only (`policy_update.applied`), where exploration is allowed.
4. **Promote** only if a held-out **`CONFIRM`** run against a frozen snapshot improves the
   primary metric (`promotion.accepted`); otherwise `promotion.rejected`.

This mirrors `regimes`' `OPTIMIZE`/`CONFIRM` gating, but the transform space is the full
set of epistemic seams rather than a single repair operator.

## Directory layout

```
src/regimes_probe/
  activegraph_pack/   # ActiveGraph object/event/relation schema, behaviors, projection, replay harness
  datasets/           # SyntheticBrowseAdapter, BrowseCompAdapter, LiveBrowseCompAdapter; checksum/version capture
  tools/              # ActiveGraph-wrapped search/fetch tools (openai_web_search, brave_search, page_fetch, ...)
  policy/             # query_signature embedding, policy_memory, contextual bandit, seam scoring, consolidation
  agent/              # hot-path orchestration: routing_plan, query_plan, candidate/final answer, verify, stop
  regimes/            # regime detection, policy_update proposals, OPTIMIZE/CONFIRM promotion gating
  eval/               # benchmark_run driver, grading, reward attribution, metrics, reports, replay check
```

Supporting top-level directories: `config/` (run configs), `fixtures/` (synthetic
deterministic data for Study 0 and unit tests), `results/` (reports, snapshots), `scripts/`
(entry points), `tests/`.

## Canonical objects, events, relations

Every component emits and consumes the canonical ActiveGraph schema. Object types:
`benchmark_run`, `benchmark_item`, `question_attempt`, `query_signature`,
`policy_memory_snapshot`, `policy_fragment`, `routing_plan`, `query_plan`, `tool_call`,
`evidence_observation`, `candidate_answer`, `final_answer`, `grade_result`,
`evidence_reward`, `tool_reward`, `trace_reward`, `regime_label`, `policy_update`,
`promotion_decision`, `report`.

Event types: `benchmark.started`, `item.queued`, `attempt.started`, `signature.created`,
`memory_snapshot.loaded`, `policy_fragment.selected`, `routing_plan.created`,
`query_plan.created`, `tool.requested`, `tool.responded`, `evidence.observed`,
`candidate_answer.created`, `verification.completed`, `stop_decision.created`,
`final_answer.created`, `grade.completed`, `reward.computed`, `regime.detected`,
`policy_update.proposed`, `policy_update.applied`, `memory.consolidated`,
`evaluation.completed`, `promotion.accepted`, `promotion.rejected`, `report.created`.

Relation types: `attempt_for_item`, `signature_for_attempt`, `snapshot_used_by_attempt`,
`fragment_selected_for_attempt`, `plan_for_attempt`, `query_for_tool_call`,
`evidence_from_tool_call`, `answer_supported_by_evidence`, `grade_for_answer`,
`reward_for_call`, `reward_for_attempt`, `update_from_reward`, `regime_for_failure`,
`promotion_for_update`.
