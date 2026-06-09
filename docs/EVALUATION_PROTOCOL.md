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

### Candidate-hypothesis gating (iterative v0 → v1)

Iterative v0 (staged search) introduced multi-hop resolution but **over-selected
wrong intermediate candidates** and re-exploited them (sources, broad orgs, broad
locations, generic concepts). The next generic layer types each candidate by role,
infers the target role(s) from the question, and carries a candidate forward only
if it is role-compatible AND its follow-up improves evidence — with an anti-sticky
beam that forces exploration after repeated no-progress. This is a general
epistemic-policy rule for multi-step search, not a BrowseComp-specific patch.
Metrics (`candidate_role_match_rate`, `sticky_candidate_count`,
`evidence_improved_after_candidate_rate`, `candidate_switch_count`, …) make the
selection behavior measurable. Scrape/fetch is evaluated only after candidate
targeting improves, since scraping wrong pages only produces richer wrong evidence.

### Level 4: task-frame / constraint-graph search

The deepest layer reframes the bottleneck: a BrowseComp item is **constraint
satisfaction over latent variables**, not a clue bag. The agent parses the question
into a generic `TaskFrame` (typed target + intermediate slots, typed constraints,
dependency edges, known-context terms), then drives search/candidate-selection/
reading/stopping over an explicit hypothesis table + evidence ledger
(`QUERY_POLICY.md` Level 4). Evaluate it as an **ablation** (`--enable-task-frame`
on vs off, same tool set), reading the frame metrics: `slot_resolution_rate`,
`constraint_support_rate`, `target_slot_support_rate`, `hypothesis_coverage_score`,
`evidence_progress_per_action`, `read_value_precision`, `no_progress_action_rate`,
`repeated_equivalent_query_rate`, and `final_answer_supported_by_constraints_rate`.
The success signal is not just answer rate but whether reads/queries are
**frame-grounded** (each action tests a specific unresolved constraint/slot) and the
final answer is **supported by the question's own constraints** — a guard against
answering with an ungrounded high-frequency entity. Per-item/per-topic rules are
forbidden (that would overfit the benchmark, see `METHODOLOGY_RISKS.md`).

The first structurally-successful Level-4 live run
(`browsecomp-taskframe-debug-001`) confirmed the wiring works, runs are
headline-eligible, policy memory stays answer-free, and the planner uses far less
budget than scrape-read (`policy_memory@12` mean tool calls dropped 12.0 → 5.9) —
but accuracy was still 0 because the **deterministic parser built the wrong frame**
(wrong target, wrong constraint attachment, source/org promoted to target). The
limiting factor is parse quality, not the planner.

**Level 4b: LLM task-frame parser as an ablation.** `--enable-llm-task-frame-parser`
(requires `--enable-task-frame`) swaps the deterministic parser for a cached,
schema-validated LLM parser that emits the **same** frame and **does not answer**
(`QUERY_POLICY.md` Level 4b). Evaluate it as a parser ablation holding everything
else fixed, and read the new provenance metrics: `parser_used`,
`task_frame_parse_quality_mean`, `deterministic_fallback_rate`,
`parser_validation_failure_counts`, `target_slot_role_distribution`. Because the
parser is cached and fallback-safe, a replay never calls a model and a bad/invalid
parse silently reverts to the deterministic frame — so a comparison stays valid even
when the model is unavailable. The honest read: does a **higher-quality frame**
(better target identification + constraint attachment) raise frame-grounded action
rates and `final_answer_supported_by_constraints_rate`, and only then accuracy?

**Live wiring + cost.** The parser model is wired through the same `RecordingCache`
as the answerer (`build_task_frame_model_fn`); the parser model defaults to
`--answer-model` (override with `--task-frame-parser-model`). The dry-run plan
prints and records `parser_model_calls_estimated` (≈ distinct optimize+confirm
questions, since the parser cache dedups), and the report records the realized
`parser_model_calls`, `parser_cache_hits`, `parser_cache_misses`,
`parser_fallback_count`, and `task_frame_parser_model` so the parser's spend and
fallback rate are auditable alongside the answerer's. `--enable-llm-task-frame-parser`
without `--enable-task-frame` is a hard configuration error (non-zero exit, no
artifacts) — flags are never silently downgraded.

**Answer-support is gated, not assumed.** `answer_supported` /
`final_answer_supported_by_constraints` now comes from a strict gate
(`evaluate_answer_support`): a bound slot is not enough — non-contaminated evidence
tied to the hypothesis must support the target, the target's discriminative
constraint must be resolved, nothing contradicted, and ≥2 constraints supported.
Audit `missing_support_reasons` to see why an attempt did not earn a supported
answer; a run whose `answer_support_gate` rate is high but accuracy is low indicates
the gate is being satisfied by the wrong hypothesis (re-check parse quality and
constraint resolution), not that the gate is too lax.

### Level 4c/4d: open-world constraints + escalation controller

The parser contract is now **open-world** (`QUERY_POLICY.md` Level 4c): constraints
carry free-form `semantic_label`/`semantic_facets` plus a closed set of operational
`affordances`. Evaluate the change by its intended effect — **the LLM-parser
fallback rate should drop sharply** (rich labels like `employment_relation` no longer
force a fallback) **without** loosening the safety checks. Read
`deterministic_fallback_rate`, `parser_validation_failure_counts` (should no longer
contain label/facet rejections), and the per-constraint `affordances` /
`validation_warnings` in `debug_questions.jsonl`. A correctness gain is only credible
if it comes from *better-grounded actions* (affordance-driven), not from the gate or
parser being more permissive.

The **epistemic escalation controller** (Level 4d, `--auto-epistemic-mode`) is an
ablation in its own right: with it on, easy questions should consume far less budget
(direct/simple modes, `parser_model_calls` near zero on those items) while hard
multi-hop items still escalate to `task_frame_required`. Compare wall-clock/cost and
accuracy with the controller on vs. off; the win is "same accuracy, less wasted
work" on the easy tail, not a headline accuracy change. The default benchmark run
keeps explicit flags so conditions stay comparable; the controller is for the
generic-agent setting and is recorded per attempt (`selected_epistemic_mode`).

### Level 4e: descriptor variables are not leakage

The known-context-as-target validator now tests **variable vs. constant**, not string
overlap (`QUERY_POLICY.md` Level 4e). When auditing parser quality, do NOT treat a
target slot name that reuses question wording ("90s TV series", "founder full name")
as leakage — that is a *descriptor* of the unknown and is expected. The real failures
to watch are the opposite two: a parser that **prematurely binds** a target
(`bound_value` populated, or a Title-Case constant like "World Health Organisation"
promoted to the answer) — caught by validation and recorded as
`known_context_promoted_to_target:<reason>` — and a validator that **over-rejects**
useful descriptors (now surfaced as `validation_warnings`, not fallbacks). Read the
per-slot `slot_status` and the preview's `known_context_target_check` to tell these
apart. Because raw descriptors, inferred `slot_status`, and the validation decision
are all recorded in the trace/graph, "does the parser create good variable
descriptors vs. prematurely bind answers?" becomes a measurable, learnable signal —
not a hand judgment.

### Level 5: candidate slates + frontier (multi-hop search behavior)

Hard multi-hop questions need **possible mid-hop answers maintained per slot**, not a
single global candidate (`QUERY_POLICY.md` Level 5). Evaluate this layer by its
*process* metrics, since correctness on the final answer is downstream of many
intermediate decisions: `candidate_slate_size_mean`, `active/confirmed/rejected_
candidates_per_slot`, `candidate_promotion_rate`/`rejection_rate`/`merge_rate`,
`hypothesis_branching_factor`, `frontier_action_counts`, `frontier_expected_gain_mean`,
`read_on_candidate_rate`, and `answer_from_confirmed_hypothesis_rate`. Good behavior:
candidates rejected when they contradict a constraint, upstream slots bound before
downstream targets, cheap verification chosen before expensive reads, duplicates
merged, and broad known-context terms NOT searched as if they were unresolved slots.
The layer is **skipped** for easy questions (`skipped_candidate_slate_count` +
`skipped_heavy_candidate_slate_reason_counts`), so a generic-agent run should show the
heavy machinery firing only on the hard tail. Because the whole slate/frontier state
is projected and replayable, an **offline fork** can re-score frontier actions under a
different reward/priority setting without re-calling providers — letting you ask
"which frontier strategy resolves slots with the least wasted work?" from traces.

**Shadow vs. controller is a clean A/B (Level 5b).** Run the frontier in shadow
(default) and in `--enable-frontier-controller` with everything else fixed. In shadow
mode read `frontier_planner_agreement_rate` — how often the frontier's recommendation
matched the old planner's actual action — to see where the two diverge before handing
over control. In controller mode read `frontier_controller_used_rate`,
`frontier_action_execution_success_rate`, `frontier_fallback_to_old_planner_count`, and
`tool_calls_from_frontier_actions`. Two planning-quality signals matter most for the
"don't open with a generic query" requirement: `first_action_discriminative_constraint_rate`
(should be high — the first action targets an actor/casting-style discriminative
constraint, not the generic answer-type target constraint) and `generic_first_query_rate`
(should be near zero). A controller win is "same/better accuracy with fewer wasted
tool calls and a higher discriminative-first rate," audited from the per-call
`frontier_action_id` links — not a black-box accuracy delta.

**LLM frontier proposer is another clean A/B (Level 5c).** With the controller fixed on,
turn on `--enable-llm-frontier-repair` or `--enable-llm-frontier-planner` (both require
`--enable-task-frame`) and hold everything else constant — including across the
`no_memory` / `policy_memory` arms, which the same-conditions validator now enforces via
`ConditionSpec.llm_frontier_settings` (`mode|model|prompt-fingerprint`). The question this
A/B answers is narrow and mechanistic: *does letting an LLM compose the query (under
deterministic validation) reduce generic junk-retrieving queries without changing anything
else?* Read it from the projected metrics, not a black-box accuracy delta:
`generic_query_repaired_count` and `generic_query_blocked_count` (the bottleneck the layer
targets), `proposal_accept_rate` / `proposal_rejection_counts` (how disciplined the
validator is — a healthy run rejects generic / duplicate / nonexistent-reference
proposals), `selected_query_constraint_anchor_rate` and
`selected_query_known_context_anchor_rate` (the executed query is grounded in a real
constraint / known-context term, not a bare descriptor), `duplicate_query_blocked_count`,
`evidence_progress_by_llm_frontier_action` and `llm_frontier_realized_eig` (the proposed
action actually advanced evidence), and `llm_frontier_vs_deterministic_agreement_rate`
(where the LLM diverged from the deterministic plan). Because every proposal, its
validation verdict, and its driven tool call are projected and the model call is cached by
`card_hash`, the whole comparison is **replayable with zero model calls** and an offline
fork can re-score the cached proposals under a different policy. A win is "fewer generic
queries and more anchored, evidence-advancing tool calls at equal/better accuracy,"
auditable from the `tool_call_from_llm_frontier_proposal` edges — and the model never
touches policy memory, so it cannot leak an answer into the learned signal.

**Audit proposal → action → evidence integrity, not just query quality (Level 5c).** The
first repair run produced good queries but mistranslated them — the selected proposal's
slot/constraints were dropped, so evidence attached to the wrong slot and support stayed
at zero (a correct `Cristina Ortiz` candidate even went un-promoted). So a Level-5c run is
only trustworthy if the integrity metrics are clean: `llm_proposal_to_action_integrity_rate`
and `llm_proposal_slot_match_rate` / `llm_proposal_constraint_match_rate` should be **1.0**
(any drop means executed actions diverged from the proposals, with
`frontier_action_integrity_error` events marking the refused-and-fell-back steps);
`evidence_linked_to_selected_constraint_rate`,
`llm_frontier_progress_slot_compatible_candidate_rate`, and
`llm_frontier_selected_constraint_support_rate` show the evidence actually advanced the
*selected* slot/constraint rather than an arbitrary candidate; `candidate_promoted_from_llm_
frontier_count` counts genuine promotions; `llm_frontier_repair_trigger_reason_counts`
shows *why* repair fired (now also on repeated zero-progress / stale-candidate / noise /
missing-anchor queries, not just one-word generics); and `llm_frontier_tool_normalization_
count` shows enabled concrete tools (`serper_search`/`exa_search`) being accepted rather
than spuriously rejected. Read these together with `has_gold_but_not_promoted` in the
analysis artifacts (gold lives only there, never in the card) to catch the
"right answer present but never scored" failure the bug produced.

**Evidence interpretation turns "did retrieval work?" into "was the result understood?"
(Level 5d).** The deterministic interpreter is always on, so judge a run by whether results
were interpreted into the *right* assertions, not by how many were retrieved:
`source_role_counts` / `source_role_noise_rate` show how much of the result stream is
definition/UI/contaminated chrome; `noise_candidate_rejection_rate` and
`candidate_assertion_rejection_counts` show the slate being protected from n-gram junk
("Datasets", "Login", "Merriam", …); `accepted_candidate_assertion_count` vs
`candidate_assertion_count` shows the accept/reject discipline;
`constraint_assertion_support_count` and `constraint_support_from_interpretation_rate` show
constraint support flowing **only** from explicit assertions on non-noise sources (never
arbitrary overlap); and `candidate_promotion_from_evidence_count` shows real entities making
it onto slates. The debug-only `has_gold_but_not_promoted_count` (computed in
`debug_run_failures.py` where gold is available — never in policy memory) is the direct
check for the WHO `Cristina Ortiz` failure: a correct candidate present in results but not
promoted. A healthy interpreter run rejects most chrome, accepts role-compatible entities on
the selected slot, and raises constraint support via quoted assertions.

**The evidence→candidate→constraint path has first-class link invariants (Level 5d.1).** A
trace where evidence reaches slots but never constraints is a *bug to surface, not absorb*:
read `evidence_to_slot_link_rate` vs `evidence_to_constraint_link_rate` together with
`ev_slot_true_cons_false_count` and `accepted_candidate_without_constraint_support_count` —
a large slot/constraint gap means accepted candidates carry no constraint anchor and should
be investigated, and every such case is explained by an `ev_slot_true_cons_false_explained`
event. The verifier↔registry desync (an extracted candidate later called "nonexistent") is
caught by `canonical_candidate_resolution_rate` (should be ≈1.0) and
`verifier_nonexistent_but_extracted_candidate_count` (should be 0); a non-zero count means
the proposal and registry namespaces diverged. `deterministic_query_marked_ok_but_repaired_count`
and `deterministic_query_bad_but_not_repaired_count` measure whether repair fires on
evidence-quality failures (noise/no-progress/prompt-language queries), not just one-word
generics.

**Judge the research *loop* by reads and support yield, not accuracy alone (Level 5e).**
The success target for the next trace is consistency, not score: `support_dropped_count`
(should be 0 — support never disappears silently), `support_from_read_count` /
`support_from_read_rate` (reads should convert snippet candidates into body-evidence
support on at least some cases), `forced_read_after_no_support_count` and
`read_starvation_count` (reads scheduled when snippets can't close body-only constraints,
not starved), `first_action_discriminative_constraint_rate` with the recorded
`discriminative_reason`, and the `nonexistent_candidate_breakdown` /
`canonical_candidate_alias_resolution_rate` (an extracted candidate resolves canonically
instead of failing). `read_after_search_rate` and the per-attempt `stage_depth`/`followup`
counts should rise for `task_frame_required` cases as reads add depth. Because read-heavy
policies cost more, weigh support yield **per read and per dollar**, and treat learned
provider metrics as observations, not fixed truths.
