# Regimes Improvement Loop

This document specifies the **regimes layer** of `regimes-probe`: the cold-path
mechanism that diagnoses *why* the epistemic policy failed and proposes **bounded**
policy-parameter mutations, gated by `OPTIMIZE` (in-sample) then `CONFIRM` (held-out).

It mirrors the discipline of [`yoheinakajima/regimes`](https://github.com/yoheinakajima):
`OPTIMIZE` is exploratory and **not** the headline; `CONFIRM` is **held out**; every
promotion is recorded as a `promotion.accepted` / `promotion.rejected` event carrying a
`promotion_decision` object; and **every claim is grounded in a committed report
artifact** (`report.created`).

## What this layer is (and is not)

The regimes layer **sits above the contextual-bandit epistemic policy**. It does **not**
route, query, verify, stop, or answer questions. It never touches the hot path directly.

| The regimes layer DOES | The regimes layer does NOT |
| --- | --- |
| Read cold-path rewards and grades (Layer 1 trace, archived) | Answer benchmark questions |
| Detect dominant `regime_label`s over a run | Run tool calls or fetch pages |
| Propose **bounded** `policy_update`s to bandit parameters / rules | Mutate `policy_memory_snapshot` in place during `CONFIRM` |
| Gate every change through `OPTIMIZE` then `CONFIRM` | Store benchmark answers in policy memory |
| Record every promotion decision and ground it in a report | Make headline claims from in-sample (`OPTIMIZE`) numbers |

The object it consumes is the cold-path reward attribution
(`tool_reward`, `evidence_reward`, `trace_reward`, `grade_result`); the objects it
produces are `regime_label`, `policy_update`, `promotion_decision`, and `report`, all of
which become the next `policy_memory_snapshot` only after a passing `CONFIRM`.

## Candidate mutation space (three tiers)

The regimes layer proposes changes only from an explicit, tiered catalogue. Each proposal
names the tier; the promotion gate applies stricter thresholds and smaller bounded deltas
to higher tiers.

### Tier 1 — Safe numeric mutations (allowed in v0)

Purely numeric knobs on the contextual bandit and reward function. Bounded by a configured
`max_delta` per parameter; monotone and reversible.

| Parameter | Seam affected | Typical bounded delta |
| --- | --- | --- |
| `reward_weights` (correctness / evidence-gain / cost) | reward attribution | ±0.10 per weight, renormalized |
| `router_blend_weights` (per-tool prior blend) | route | ±0.10, renormalized over tools |
| `knn_k` (K for nearest-neighbor signature retrieval) | policy memory lookup | ±1, clamped to `[1, 16]` |
| `exploration_coeff` (UCB / epsilon term) | route / query | ±0.05, clamped to `[0, 1]` |
| `recency_decay` (freshness half-life) | query / verify | ±0.10× current |
| `confidence_penalty` (penalty on low-support answers) | answer / verify | ±0.05 |
| `stop_threshold` (sufficiency score to halt) | stop | ±0.05, clamped to `[0, 1]` |
| `verification_threshold` (support score to accept) | verify | ±0.05, clamped to `[0, 1]` |

### Tier 2 — Medium-risk mutations (gated, conservative in v0)

Structured rule/library edits. Allowed but with the strictest `CONFIRM` threshold and a
mandatory diff review recorded in the report.

| Mutation | Seam affected |
| --- | --- |
| `query_template_library` (add / remove / reweight templates) | query |
| `evidence_sufficiency_rule` (when evidence is "enough") | stop |
| `source_authority_scoring_rule` (how authority is scored) | route / verify |
| `stale_source_detection_rule` (what counts as stale) | verify |

### Tier 3 — High-risk mutations (NOT in v0)

> **AVOID high-risk mutations in v0.** The following are explicitly out of scope for the
> initial study and MUST NOT be proposed by the regimes layer in v0:
>
> - `answer_prompt` (the prompt that drafts `candidate_answer` / `final_answer`)
> - `judge_prompt` (the grading / verification prompt)
> - arbitrary code (behaviors, tools, projection logic)
>
> These change the measurement apparatus or open uncontrolled degrees of freedom, which
> would invalidate the `correct_per_tool_call` comparison. They are listed only to mark
> them as deliberately excluded.

## Failure regimes (generic, not topical)

Regimes are **generic** — they describe *epistemic-search failure modes*, never a topic or
a specific question. The twelve canonical regimes, their one-line definitions, and the
bounded mutation(s) each suggests:

| `regime_label` | Definition | Bounded mutation(s) suggested |
| --- | --- | --- |
| `route_miss` | The first/dominant tool chosen was wrong for the signature. | Adjust `router_blend_weights`; raise `exploration_coeff`. |
| `query_miss` | The query phrasing failed to surface available evidence. | Reweight / extend `query_template_library`; raise `exploration_coeff` on query seam. |
| `evidence_sparse` | Too little evidence gathered to support any answer. | Raise `nearest_k` (borrow more priors); raise `stop_threshold` so the loop keeps gathering; raise evidence-gain term in `reward_weights`. |
| `stale_evidence` | Evidence used was outdated / superseded. | Raise the `freshness` reward weight; lower `verification.authority_threshold` so fresh non-top-authority sources qualify; strengthen `recency_decay`. |
| `over_search` | More tool calls than needed; correct answer already in hand. | **Lower** `stop_threshold` (halt once evidence suffices); raise `extra_call_penalty` in `reward_weights`. |
| `under_search` | Stopped before enough evidence was gathered. | **Raise** `stop_threshold` (require more before halting); raise evidence-gain term in `reward_weights`. |
| `verification_miss` | Candidate answer accepted without adequate support check. | Raise `verification.authority_threshold`; raise `confidence_penalty`. |
| `stop_too_early` | Halted with sufficiency below the true need. | **Raise** `stop_threshold`. |
| `stop_too_late` | Continued after sufficiency was reached. | **Lower** `stop_threshold`. |
| `answer_extraction_miss` | Evidence supported a correct answer but extraction failed. | Raise `confidence_penalty` to force re-read; reweight query templates toward extraction. |
| `contradiction_unresolved` | Conflicting evidence left unreconciled. | Raise `verification_threshold`; adjust `source_authority_scoring_rule`. |
| `support_answer_mismatch` | Final answer not entailed by cited evidence. | Raise `verification_threshold`; raise `confidence_penalty`. |

A run typically surfaces a **mixture**; the loop acts on the **dominant** regime(s) for the
split (those over a configured share of attributed failures) to keep each proposal
single-purpose and reversible.

## The regimes loop (steps)

```
                 OPTIMIZE split                          CONFIRM split (held out, frozen)
                 ------------------                      --------------------------------
  1. baseline run ---------------+
                                 |
  2. detect dominant regimes <---+   (regime.detected, regime_label)
                                 |
  3. propose bounded update <----+   (policy_update.proposed, policy_update)
                                 |
  4. eval-diff (apply on OPT) ---+   (policy_update.applied, evaluation.completed)
                                 |
  5. in-sample gates pass? ------+--> freeze update (snapshot frozen)
                                 |                       |
                                 |                       v
  6.                             |        before/after CONFIRM, same split, frozen snapshot
                                 |                       |   (evaluation.completed x2)
                                 |                       v
  7.                             +-------------> promote? (promotion.accepted / rejected,
                                                          promotion_decision, report.created)
```

1. **Run baseline on `OPTIMIZE`.** Establish current-policy metrics on the in-sample split.
2. **Detect dominant failure regimes.** Attribute failures to `regime_label`s; emit
   `regime.detected`. Rank by share of attributed failures.
3. **Propose a bounded policy update.** Pick the mutation(s) the dominant regime suggests,
   within `max_delta`. Emit `policy_update.proposed` with a `policy_update` object.
4. **Run eval-diff on `OPTIMIZE`.** Apply the update under `OPTIMIZE` only
   (`policy_update.applied`; `optimize_allows_exploration=true`), re-run, and compute the
   before/after delta in `correct_per_tool_call`. Emit `evaluation.completed`.
5. **If in-sample gates pass, freeze the update.** Gates: primary metric does not regress
   in-sample beyond noise, and the proposal stayed within bounds. Freeze the resulting
   `policy_memory_snapshot`.
6. **Run `CONFIRM` before/after with the same split.** Against the **frozen** snapshot
   (`confirm_uses_frozen_snapshot=true`, `confirm_updates_memory=false`), evaluate the
   baseline snapshot and the candidate snapshot on the identical held-out `CONFIRM` split.
   Emit two `evaluation.completed` events.
7. **Promote only if `CONFIRM` improves or does not regress beyond threshold.** Compare
   held-out before/after `correct_per_tool_call`. If it improves, or stays within the
   configured `confirm_regression_tolerance`, emit `promotion.accepted`; otherwise
   `promotion.rejected`. Either way, write a `promotion_decision` and a committed `report`.

## Discipline (mirrored from `yoheinakajima/regimes`)

- **`OPTIMIZE` is not the headline.** In-sample improvement is necessary but never
  reported as the result. It only authorizes a `CONFIRM` attempt.
- **`CONFIRM` is held out.** The `CONFIRM` split is disjoint from `OPTIMIZE`, evaluated
  against a **frozen** snapshot, and never mutates memory.
- **All promotions are recorded.** Every decision emits `promotion.accepted` or
  `promotion.rejected` with a `promotion_decision` object referencing the originating
  `policy_update` and both `evaluation.completed` runs.
- **All claims are grounded in committed report artifacts.** No number is claimed unless it
  appears in a committed `report` (`report.created`); see [`REPORTING.md`](./REPORTING.md).

## Worked example: `over_search` dominance

**1–2. Detect.** Baseline on `OPTIMIZE` shows `over_search` as the dominant regime: many
attempts reached a correct answer but kept issuing tool calls, depressing
`correct_per_tool_call`. Emit `regime.detected`.

**3. Propose.** **Lower** `stop_threshold` by a bounded delta (`-0.10`, within `max_delta`)
so the agent halts as soon as the evidence is sufficient. (`stop_now` fires when the
verification score ≥ `stop_threshold`, so a *lower* threshold halts *sooner*.)
Emit `policy_update.proposed`:

```json
{
  "object_type": "policy_update",
  "policy_update_id": "pu_2026_06_07_over_search_001",
  "proposed_by": "regimes_layer",
  "tier": "safe_numeric",
  "regime_label": "over_search",
  "from_snapshot_id": "snap_baseline_0007",
  "mutation": {
    "parameter": "stop_threshold",
    "op": "increment",
    "delta": -0.10,
    "before": 0.60,
    "after": 0.50,
    "bounds": [0.1, 0.95],
    "max_delta": 0.10
  },
  "rationale": "over_search dominant on OPTIMIZE; correct answer reached before final calls.",
  "status": "proposed",
  "created_event": "policy_update.proposed"
}
```

**4. Eval-diff on `OPTIMIZE`.** Apply (`policy_update.applied`) and re-run. The in-sample
diff shows `correct_per_tool_call` rising (e.g. `0.182 -> 0.205`) with accuracy flat —
fewer wasted calls, same correctness. Emit `evaluation.completed`. *(Not headline.)*

**5. Freeze.** In-sample gates pass (metric up, within bounds); freeze the candidate
snapshot `snap_candidate_0008`.

**6. `CONFIRM` before/after.** On the held-out `CONFIRM` split, against frozen snapshots,
evaluate baseline `snap_baseline_0007` then candidate `snap_candidate_0008`. Emit two
`evaluation.completed` events.

**7. Promote or reject.** Compare held-out `correct_per_tool_call`. If it improves (or does
not regress beyond `confirm_regression_tolerance`), promote. Emit `promotion.accepted` with
a `promotion_decision`:

```json
{
  "object_type": "promotion_decision",
  "promotion_decision_id": "prom_2026_06_07_0008",
  "policy_update_id": "pu_2026_06_07_over_search_001",
  "regime_label": "over_search",
  "decision": "accepted",
  "decision_event": "promotion.accepted",
  "from_snapshot_id": "snap_baseline_0007",
  "to_snapshot_id": "snap_candidate_0008",
  "optimize_eval": {
    "run_id": "run_opt_0042",
    "metric": "correct_per_tool_call",
    "before": 0.182,
    "after": 0.205,
    "headline": false
  },
  "confirm_eval": {
    "run_id_before": "run_conf_0043a",
    "run_id_after": "run_conf_0043b",
    "metric": "correct_per_tool_call",
    "before": 0.171,
    "after": 0.189,
    "delta": 0.018,
    "confirm_regression_tolerance": 0.01,
    "uses_frozen_snapshot": true,
    "updates_memory": false
  },
  "report_id": "report_run_conf_0043b",
  "grounded_in_report": true
}
```

Had the held-out `confirm_eval.delta` fallen below `-confirm_regression_tolerance`, the
decision would be `"rejected"` with `decision_event: "promotion.rejected"`, the candidate
snapshot discarded, and the baseline retained — still recorded and grounded in a committed
report.
