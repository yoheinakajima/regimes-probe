# Leakage Controls

`regimes-probe` claims that gains come from **learned procedural search policy**, not from
memorized benchmark answers or intrinsic model knowledge. That claim is only credible if
leakage is actively controlled, measured, and reported. This document lists the mandatory
controls.

## Core invariant

> **No benchmark answer ever enters hot-path policy memory.** The hot path consumes only
> Layer 2 (answer-free procedural priors) and live `evidence_observation`s.

## Memory-content controls

1. **No final benchmark answer in hot-path policy memory.** `policy_memory_snapshot`s store
   procedural statistics keyed by `query_signature` (bandit params, seam priors,
   `policy_fragment`s) — never the correct answer string or anything from which it can be
   recovered.
2. **No prior full Q/A retrieval in the main `policy_memory` condition.** The answerer
   cannot fetch prior question→answer pairs. Retrieval is restricted to answer-free priors.
3. **Raw traces are archived, not retrieved by the answerer.** The full event trace
   (Layer 1) exists for reward attribution, replay, and offline analysis only. Direct
   retrieval of raw traces by the answerer is a leakage path and is allowed **only** in the
   explicitly labeled `naive_full_trace_memory_control` (never the main condition).

## Split / freezing controls

4. **Freeze memory before CONFIRM.** Memory is frozen into a `policy_memory_snapshot` after
   the `OPTIMIZE` experience phase; `CONFIRM` scores against the frozen snapshot and does
   not update memory (`confirm_uses_frozen_snapshot=true`, `confirm_updates_memory=false`).
5. **Exact question overlap forbidden.** No `benchmark_item` may appear in both `OPTIMIZE`
   and `CONFIRM`. The split is deterministic and checked.
6. **Prefer entity-disjoint / time-disjoint splits.** Where the dataset permits, split so
   that entities (or time windows) in `CONFIRM` are absent from `OPTIMIZE`, preventing
   per-entity answer transfer through priors.

## Baseline / control conditions

7. **Closed-book baseline.** Run the answerer with **no search** to estimate intrinsic
   (parametric) knowledge. Gains over closed-book attributable to search are the quantity
   of interest; this is why the primary target is recent-facts `LiveBrowseComp`.
8. **Random-memory control.** `random_memory_control` (frozen randomized priors) confirms
   that gains require *learning*, not just a non-default policy structure.
9. **No-search baseline.** A separate budget-0 / no-tool baseline bounds how much is
   answerable without browsing at all and frames the `correct_per_tool_call` numerator.

## Logging / provenance

10. **Log everything.** Every run logs: all prompts, every `tool.requested`/`tool.responded`
    with full args, resolved **URLs**, **page content hashes**, **timestamps**, **model
    IDs**, and **tool configs** (provider, context size, allowed domains). This is what
    makes replay and post-hoc leakage audits possible.

## Cache / repetition risks

11. **Report repeated-run / cache risks.** Hosted search and page caches can make a second
    run cheaper or different from the first, and repeated runs over the same items can leak
    information across runs. Reports must disclose: whether responses were cached vs live,
    how many times each item was run, and any provider-side caching that could bias
    `correct_per_tool_call`, `correct_per_dollar`, or `correct_per_second`.

## Summary table

| Control | Mechanism | Where enforced |
| --- | --- | --- |
| No answers in memory | answer-free `policy_memory_snapshot` | `policy/`, consolidation |
| No prior Q/A retrieval | restricted retrieval interface | `agent/`, `policy/` |
| Raw traces archived only | Layer 1 not exposed to answerer | `activegraph_pack/`, `agent/` |
| Freeze before CONFIRM | frozen snapshot, no CONFIRM updates | `eval/` protocol steps 5–6 |
| No exact overlap | deterministic split check | `eval/`, dataset split |
| Disjoint splits | entity/time-disjoint partitioning | `datasets/`, `eval/` |
| Intrinsic-knowledge estimate | closed-book baseline | `eval/` |
| Learning vs structure | `random_memory_control` | `eval/` step 7 |
| Browsing contribution | no-search baseline | `eval/` |
| Auditability | full prompt/tool/URL/hash/timestamp logging | `activegraph_pack/`, `eval/` |
| Cache/repetition disclosure | cache + run-count reporting | `eval/`, `report` |
