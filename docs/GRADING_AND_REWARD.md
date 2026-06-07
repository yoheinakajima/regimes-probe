# GRADING_AND_REWARD.md

## Purpose

This file defines how a `question_attempt` is graded and how the scalar reward is
computed and **attributed** to the bandit arm families (tool, query, verification, stop).
The reward decomposition is **identical** to `docs/CONTEXTUAL_BANDIT.md` and is consumed
by `update_reward` to update trace-derived priors in the `policy_memory_snapshot`
(`docs/POLICY_MEMORY.md`). Grading uses answer/evidence text only from the **cold raw
trace archive**; no answer text enters policy memory.

---

## Answer grading (`grade_result`)

| Mode | When used | Output |
|---|---|---|
| exact / normalized match | A reference answer exists | `correct ∈ {0,1}` after normalization (case, whitespace, punctuation, simple synonyms/units) |
| LLM judge (optional) | No exact reference, or shape-flexible answers | `correct ∈ [0,1]`; prompt+model+output **logged and replayable** (cached, fixed decoding) |

```
function grade_answer(final_answer, reference, judge_enabled):
    if reference != NULL:
        return normalized_match(final_answer, reference)        # {0,1}
    elif judge_enabled:
        return llm_judge(final_answer, item, cache=True)        # [0,1], logged+replayable
    else:
        return NULL                                             # ungradable
```

`correctness_reward = W_CORRECT * grade_result.correct`.

---

## Evidence grading

For each `evidence_observation`, grade independently of final correctness:

| Signal | Meaning |
|---|---|
| `candidate_surfaced` | Did this evidence surface a `candidate_answer`? |
| `supports_answer` | Does the source support the committed candidate? |
| `source_authority` | Authority score of the source ∈ [0,1] |
| `freshness` | Recency score of the source ∈ [0,1] |
| `contradiction_or_noise` | Contradiction/noise penalty ∈ [0,1] |

### Per-call `evidence_gain`

The marginal information a single `tool_call` adds, used for `first_tool_hit_reward` and
to detect over-search:

```
evidence_gain(call) =   G_SURFACE   * candidate_surfaced
                      + G_SUPPORT   * supports_answer
                      + G_AUTHORITY * source_authority
                      + G_FRESH     * freshness
                      - G_NOISE     * contradiction_or_noise
```

`first_tool_hit_reward = FTH * evidence_gain(first_call)` when the first call yields
`candidate_surfaced` or `supports_answer`.

---

## Reward decomposition (canonical, shared)

Per-attempt scalar reward — **same terms as `docs/CONTEXTUAL_BANDIT.md`**:

```
final_reward =   correctness_reward
               + evidence_quality_reward
               + source_authority_reward
               + freshness_reward
               + first_tool_hit_reward
               - cost_penalty
               - latency_penalty
               - stale_penalty
               - contradiction_penalty
               - extra_call_penalty
```

| Term | Formula |
|---|---|
| `correctness_reward` | `W_CORRECT * grade_result.correct` |
| `evidence_quality_reward` | `W_EVID * mean(supports_answer over supporting evidence)` |
| `source_authority_reward` | `W_AUTH * max(source_authority over supporting evidence)` |
| `freshness_reward` | `W_FRESH * max(freshness over supporting evidence)` |
| `first_tool_hit_reward` | `FTH * evidence_gain(first_call)` if first call hit, else 0 |
| `cost_penalty` | `C_COST * sum(call_cost)` |
| `latency_penalty` | `C_LAT * total_latency` |
| `stale_penalty` | `C_STALE * 1[freshness_sensitive and not freshness_acceptable]` |
| `contradiction_penalty` | `C_CONTRA * 1[contradictions_unresolved]` |
| `extra_call_penalty` | `C_EXTRA * max(0, calls_used - calls_needed)` |

`calls_needed` is the number of calls up to and including the first call after which the
candidate had sufficient support (see `support_confidence` in
`docs/VERIFICATION_AND_STOPPING.md`). Calls beyond that are over-search.

The benchmark's `main_metric = correct_per_tool_call = grade_result.correct / calls_used`.

---

## Reward attribution

The scalar `final_reward` (and its component terms) is attributed to each arm family that
made a decision in the attempt, so `update_reward` can credit the right priors:

| Arm family | Attributed reward (`*_reward`) |
|---|---|
| `tool_reward` (tool arm) | Weighted toward `first_tool_hit_reward`, `evidence_quality_reward`, `source_authority_reward`, minus `cost_penalty`/`latency_penalty` for that call |
| `query_reward` (query arm) | Weighted toward `evidence_quality_reward` and `evidence_gain` of calls using that query arm; penalized by `query_miss` (no usable evidence) |
| `verification_reward` (verification arm) | Credited `contradiction` resolution and `source_authority`/`freshness` gains; penalized by `contradiction_penalty` if left unresolved |
| `stop_reward` (stop arm) | Balances `correctness_reward`+support vs `cost`/`latency`/`extra_call` penalties (see `docs/VERIFICATION_AND_STOPPING.md`) |

```
function attribute_rewards(attempt, terms):
    for call in attempt.tool_calls:
        tool_reward = ( terms.first_tool_hit_reward * 1[call.is_first]
                      + terms.evidence_quality_reward * call.share
                      + terms.source_authority_reward * call.share
                      - C_COST * call.cost - C_LAT * call.latency )
        update_reward(snapshot, attempt.ctx, "tool", call.tool, terms_of(tool_reward), now)

    for qp in attempt.query_plans:
        query_reward = ( terms.evidence_quality_reward * qp.share
                       + G_SURFACE * qp.candidate_surfaced
                       - (Q_MISS if qp.no_usable_evidence else 0) )
        update_reward(snapshot, attempt.ctx, "query_template", qp.arm, terms_of(query_reward), now)

    for vc in attempt.verification_checks:
        verification_reward = ( terms.source_authority_reward * vc.share
                              + terms.freshness_reward * vc.share
                              - terms.contradiction_penalty * vc.left_unresolved )
        update_reward(snapshot, attempt.ctx, "verification", vc.arm, terms_of(verification_reward), now)

    stop_reward = ( terms.correctness_reward + terms.evidence_quality_reward
                  - terms.cost_penalty - terms.latency_penalty - terms.extra_call_penalty )
    update_reward(snapshot, attempt.ctx, "stop", attempt.stop_arm, terms_of(stop_reward), now)
```

All updates are skipped during CONFIRM (`confirm_updates_memory=false`) unless an
explicit online-learning variant.

---

## JSON example: `tool_reward`

```json
{
  "object_type": "tool_reward",
  "attempt_id": "qa_88421",
  "tool": "news_search",
  "call_index": 0,
  "is_first_call": true,
  "terms": {
    "first_tool_hit_reward": 0.30,
    "evidence_quality_reward": 0.22,
    "source_authority_reward": 0.18,
    "freshness_reward": 0.12,
    "cost_penalty": 0.05,
    "latency_penalty": 0.03
  },
  "scalar_reward": 0.74,
  "regime_label": "regime_07",
  "query_signature": { "features": ["freshness-sensitive", "answer-shape-constrained"] },
  "tick": 184120
}
```

## JSON example: `evidence_reward`

```json
{
  "object_type": "evidence_reward",
  "attempt_id": "qa_88421",
  "evidence_observation_id": "ev_0031",
  "tool": "news_search",
  "signals": {
    "candidate_surfaced": 1,
    "supports_answer": 1,
    "source_authority": 0.81,
    "freshness": 0.74,
    "contradiction_or_noise": 0.05
  },
  "evidence_gain": 0.69,
  "regime_label": "regime_07",
  "tick": 184120
}
```

> Note: `evidence_reward` records numeric signals and the source id reference, but the
> evidence **text** itself lives only in the cold raw trace archive, never in policy
> memory.

## JSON example: `trace_reward`

```json
{
  "object_type": "trace_reward",
  "attempt_id": "qa_88421",
  "benchmark_item": "item_5102",
  "budget_cap": 5,
  "calls_used": 3,
  "calls_needed": 2,
  "grade_result": { "correct": 1.0, "mode": "normalized_match" },
  "terms": {
    "correctness_reward": 1.00,
    "evidence_quality_reward": 0.22,
    "source_authority_reward": 0.18,
    "freshness_reward": 0.12,
    "first_tool_hit_reward": 0.30,
    "cost_penalty": 0.15,
    "latency_penalty": 0.09,
    "stale_penalty": 0.00,
    "contradiction_penalty": 0.00,
    "extra_call_penalty": 0.10
  },
  "final_reward": 1.40,
  "main_metric": { "correct_per_tool_call": 0.333 },
  "attribution": {
    "tool":         { "news_search": 0.74, "web_search": 0.20 },
    "query_template": { "freshness_terms": 0.41 },
    "verification": { "corroborate": 0.18 },
    "stop":         { "stop_now": 0.66 }
  },
  "regime_labels": ["regime_07"],
  "tick": 184120
}
```

The `terms` block in every reward object uses the **canonical decomposition** shared with
`docs/CONTEXTUAL_BANDIT.md`, guaranteeing that grading, attribution, and bandit updates
stay consistent. No object on this page stores `final_answer`, `candidate_answer`, or
evidence text in policy memory; answer/evidence text is confined to the cold archive.
