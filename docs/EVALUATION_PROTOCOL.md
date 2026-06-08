# Evaluation Protocol

This is the exact, ordered protocol for a `regimes-probe` evaluation. It is designed so
that learning happens only on `OPTIMIZE`, scoring happens against a **frozen**
`policy_memory_snapshot` on `CONFIRM`, and every step is replayable. See
[LEAKAGE_CONTROLS.md](./LEAKAGE_CONTROLS.md) for why each control exists.

## Configuration constants

| Setting | Value |
| --- | --- |
| `main_metric` | `correct_per_tool_call` |
| `budgets` | `[1, 3, 5, 10]` tool calls per `question_attempt` |
| `confirm_uses_frozen_snapshot` | `true` |
| `confirm_updates_memory` | `false` |
| `optimize_allows_exploration` | `true` |

## Protocol steps

### 1. Build / load dataset
Load via the appropriate adapter (`SyntheticBrowseAdapter`, `BrowseCompAdapter`,
`LiveBrowseCompAdapter`). Capture dataset checksum/version provenance and attach to the
`benchmark_run` (`benchmark.started`, then `item.queued` per `benchmark_item`).

### 2. Build deterministic OPTIMIZE/CONFIRM split
Partition items into `OPTIMIZE` (experience) and `CONFIRM` (held-out scoring) using a
**deterministic** rule (fixed seed / stable hash of item id). Prefer **entity-disjoint** or
**time-disjoint** splits where the dataset allows it. Exact-question overlap between splits
is forbidden.

### 3. Run `no_memory_baseline` on CONFIRM
Score the agent on `CONFIRM` with policy memory disabled (fixed/uniform policy). This is the
reference the learned policy must beat.

### 4. Run experience phase on OPTIMIZE for `policy_memory`
Run attempts on `OPTIMIZE` with the contextual-bandit policy learning enabled
(`optimize_allows_exploration=true`). Cold-path attribution produces rewards
(`evidence_reward`, `tool_reward`, `trace_reward`), `policy_update`s, and consolidated
memory (`memory.consolidated`).

### 5. Freeze memory snapshot
Emit a `policy_memory_snapshot` capturing the learned, **answer-free** priors. This snapshot
is immutable for the scoring phase (`confirm_uses_frozen_snapshot=true`).

### 6. Run `policy_memory` on CONFIRM
Score on `CONFIRM` using the frozen snapshot (`memory_snapshot.loaded`). Memory is **not**
updated during this phase (`confirm_updates_memory=false`).

### 7. Run `random_memory_control` on CONFIRM
Score on `CONFIRM` using a frozen snapshot whose parameters are randomized. Detects whether
gains come from *learning* vs merely *having a non-default policy structure*.

### 8. (Optional) `naive_full_trace_memory_control` on CONFIRM
A **clearly labeled, leakage-prone** control where the answerer is allowed to retrieve raw
prior Q/A traces. Used only to upper-bound/contrast; **never** the main condition. Reports
must flag its results as leakage-prone.

### 9. (Optional) `regimes_style_policy_update` gated through OPTIMIZE/CONFIRM
Detect dominant `regime_label`s on `OPTIMIZE`, propose `policy_update`s, apply them under
`OPTIMIZE` only, then evaluate a frozen post-update snapshot on `CONFIRM`. Promote
(`promotion.accepted`) only if `CONFIRM` improves beyond the bootstrap CI; else
`promotion.rejected`.

### 10. Produce reports
Emit `report` objects (`report.created`, `evaluation.completed`) with per-condition,
per-budget metrics, statistical tests, and dataset/run provenance.

### 11. Run replay check
Re-project the event log and re-derive the graph and all rewards/decisions. The run is valid
only if replay reproduces results byte-for-byte (modulo recorded tool responses).

## Conditions summary

| Condition | Split scored | Memory |
| --- | --- | --- |
| `no_memory_baseline` | CONFIRM | none |
| `policy_memory` | CONFIRM | frozen learned snapshot |
| `random_memory_control` | CONFIRM | frozen random snapshot |
| `naive_full_trace_memory_control` *(optional)* | CONFIRM | raw-trace retrieval (leakage-prone) |
| `regimes_style_policy_update` *(optional)* | CONFIRM | frozen post-update snapshot, gated |

## Budget tracks

Every condition is reported at each `budget in {1, 3, 5, 10}` tool calls. Budget caps are
hard limits on the number of `tool_call`s per `question_attempt`; exceeding a cap is a
harness bug (checked in Study 0). Comparisons include the full **budget curve**, not just a
single budget.

## Metrics

### Primary
- **`correct_per_tool_call`** — correct answers divided by tool calls consumed.

### Secondary
| Metric | Meaning |
| --- | --- |
| `accuracy` | fraction of items answered correctly |
| `correct_per_dollar` | correctness per dollar of tool/model spend |
| `correct_per_second` | correctness per wall-clock second |
| `first_tool_hit_rate` | fraction where the first routed tool yields usable evidence |
| `evidence_gain_per_call` | marginal evidence quality per tool call |
| `primary_or_authoritative_source_rate` | fraction of answers backed by primary/authoritative sources |
| `stale_source_error_rate` | errors attributable to stale evidence |
| `abstention precision/recall` | quality of "I don't know" decisions |
| `false_stop_rate` | stops that were premature (answer not yet supported) |
| `over_search_rate` | continued searching after sufficient evidence existed |

## Statistical tests

- **Paired per-question comparisons** between conditions (same `benchmark_item`s).
- **McNemar's test** for accuracy flips (correct↔incorrect) between paired conditions.
- **Bootstrap confidence intervals** for `correct_per_tool_call`.
- **Budget-curve comparison** across `{1,3,5,10}` (compare whole curves, not one point).

A learned-policy improvement is reported as real only when it (a) is positive on `CONFIRM`,
(b) exceeds the `random_memory_control`, and (c) clears the bootstrap CI / paired-test
threshold.

## Addendum (v0.2): baselines and headline gating

The protocol's baseline set is now explicit and offline-runnable:

- **`closed_book`** (no tools, budget 0) — estimates intrinsic model knowledge.
  Required for real runs so apparent "routing" gains cannot be intrinsic recall.
- **`no_memory_search`** — search + tools, no policy memory (the Level 1
  baseline; was `no_memory`).
- **`random_memory`** — uninformative-priors control.
- **`policy_memory`** — frozen snapshot (treatment).

Two guards gate whether a run may be reported as a headline:

1. **Same-conditions** (`eval/conditions.py`): baseline and policy must be
   identical except memory access.
2. **Headline eligibility** (`eval/eligibility.py`): OPTIMIZE/CONFIRM disjoint,
   CONFIRM memory frozen, no-answer-leakage passes, same-conditions passes,
   replay passes, both runs completed, budgets enforced, no live updates during
   CONFIRM — and the dataset is real (a synthetic fixture is never a headline).

`scripts/run_synthetic_full.py` runs all of the above offline;
`scripts/run_ablations.py` runs the Level-1/Level-2/reward/online ablations on
the fixture. See `docs/FIRST_REAL_RESULT_CRITERIA.md`.

The two verdicts are emitted as one flat `eligibility_verdict` object (in
`report.json` and as a node in `graph_projection.json`), so every gate is a
named field rather than buried in nested checks. See `docs/REPORTING.md` and
`docs/ACTIVEGRAPH_DESIGN.md`.

## Offline forked ablations (no re-spend)

To compare reward/router/stop variants after a paid run, fork it offline: the
recording cache is replayed (no provider calls) and only policy parameters
change. A fork is marked `offline_fork=true`/`parent_run_id` and is **never**
headline-eligible. The fork's natural scope is the memory-variant condition
(`policy_memory`); read the fixed `no_memory_search` baseline from the parent
report. See `docs/OFFLINE_FORK_ABLATIONS.md`.

## Query-formulation ladder for BrowseComp (why iterative resolution)

The BrowseComp dry runs isolated the bottleneck step by step:

1. **Level 1 routing only** → accuracy 0; the exact answer was missing from
   snippets (`exact_answer_missing`) because the whole long prompt was searched as
   one query and surfaced spam / benchmark-mirroring pages.
2. **Query decomposition v1 (clue spans + quality scoring)** → contamination fell
   to ~1% and queries became sane, but accuracy stayed 0: the agent still issued
   *parallel* clue queries and hoped the answer appeared in a snippet.
3. **Iterative clue resolution** (`--enable-iterative-clue-resolution`) → BrowseComp
   typically needs to **resolve an intermediate entity** from one clue's results,
   then search that entity with the next clue. Stage 1 issues the best clue query;
   later stages combine the top extracted candidate entity with the next clue span
   or an answer-shape hint, under the same budget. Stage chains, candidate entities,
   and per-stage evidence improvement are recorded (`debug_questions.jsonl`,
   `scripts/debug_run_failures.py`) and measured (`mean_stage_depth_used`,
   `evidence_improved_after_followup_rate`, …).

**Scrape/fetch comes last.** Full-page scrape (Firecrawl) should be evaluated only
*after* candidate-entity targeting works — scraping a page the staged search would
not have found just adds cost without addressing `exact_answer_missing`. All three
levels are off by default and are recorded in the manifest/report so a run states
exactly which were active.
