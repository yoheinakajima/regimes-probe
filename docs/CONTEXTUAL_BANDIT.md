# CONTEXTUAL_BANDIT.md

## Purpose

The contextual bandit is the core decision mechanism that selects arms at every choice
point: **Level 1** tool routing (`docs/ROUTING_POLICY.md`) and **Level 2** query
formulation (`docs/QUERY_POLICY.md`), verification, and stopping
(`docs/VERIFICATION_AND_STOPPING.md`).

It scores arms from **trace-derived reward statistics** in the
`policy_memory_snapshot` (`docs/POLICY_MEMORY.md`) — never from answer text. It does not
call an LLM to make routing decisions.

---

## Context

The context vector `ctx` provided at each decision:

| Field | Description |
|---|---|
| `question_embedding` | Dense vector of normalized question |
| `features` | Generic epistemic `query_signature` features + lexical/result features |
| `nn_stats` | Nearest-neighbor trace statistics from the snapshot (per-arm decayed reward) |
| `budget_cap` | Current remaining budget cap (from `budgets=[1,3,5,10]`) |
| `tool_availability` | Set of currently available tools/arms |
| `regime_label` | Latent regime cluster id for conditioning |

---

## Arms

| Level | Arm family | Arms |
|---|---|---|
| 1 | `tool` | search tools, `fetch_page` and other fetch tools (`tool_availability`-gated) |
| 2 | `query_template` | `direct_question`, `keyword_compressed`, `quoted_entities`, `source_constrained`, `freshness_terms`, `exact_answer_shape`, `site_or_domain_constrained` |
| 2 | `verification` | `corroborate`, `contradiction_check` (verification checks) |
| 2 | `stop` | `stop_now`, `search_more`, `fetch_page` |

---

## Reward decomposition

The scalar reward used to update arm statistics is decomposed into additive terms. This
**same decomposition** is used in `docs/GRADING_AND_REWARD.md`.

```
reward =   correctness_reward
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

| Term | Sign | Meaning |
|---|---|---|
| `correctness_reward` | + | Final answer matches reference / passes judge |
| `evidence_quality_reward` | + | Retrieved evidence supports the candidate answer |
| `source_authority_reward` | + | Sources are authoritative |
| `freshness_reward` | + | Sources are sufficiently recent |
| `first_tool_hit_reward` | + | First tool call surfaced usable evidence |
| `cost_penalty` | − | Per-call monetary/budget cost |
| `latency_penalty` | − | Wall-clock / step latency |
| `stale_penalty` | − | Sources too old for a freshness-sensitive question |
| `contradiction_penalty` | − | Unresolved contradictions among sources |
| `extra_call_penalty` | − | Calls beyond what was needed (over-search) |

Reward attribution to arm families is defined in `docs/GRADING_AND_REWARD.md`.

---

## Estimators and adjustments

| Mechanism | Role |
|---|---|
| epsilon-greedy | Default exploration during OPTIMIZE |
| UCB | Optimistic exploration: `mean + c*sqrt(ln(N)/n)` |
| Thompson sampling (optional) | Beta-sampled exploration, seeded deterministically |
| recency decay | Down-weights old observations via half-life multiplier on counts |
| confidence adjustment | Widen/narrow estimate by count and variance |
| nearest-neighbor weighting | Weight neighbor traces by cosine similarity |
| global prior fallback | Back off to global arm priors when local support is thin |

### Exploration regime gating

- **OPTIMIZE:** `optimize_allows_exploration=true`. epsilon-greedy / UCB / Thompson are
  active; memory may update.
- **CONFIRM:** `confirm_uses_frozen_snapshot=true`, `confirm_updates_memory=false`. Use
  the **frozen** snapshot and **deterministic exploitation** (epsilon=0, no Thompson
  sampling) — unless the run is explicitly configured as an online-learning variant.

---

## Recency decay

For an arm stat with last update tick `last_tick` evaluated at `now_tick` and half-life
`H` (`recency_half_life_ticks`):

```
decay(stat) = 0.5 ** ((now_tick - stat.last_tick) / H)
decayed_n   = stat.n * decay(stat)            # effective count
```

`decayed_n` (not raw `n`) is used in confidence and UCB/Thompson computations.

---

## Confidence adjustment

Confidence shrinks toward the prior when the **decayed** count is small or variance is
high:

```
# Standard error of the mean, floored count to avoid div-by-zero
se(stat)        = sqrt(max(stat.var_reward, VAR_FLOOR) / max(stat.decayed_n, 1))
confidence(stat) = 1 / (1 + se(stat))         # in (0,1], higher = more trustworthy
```

---

## Nearest-neighbor weighted reward

Given `ctx.question_embedding` and the snapshot's stored trace vectors, retrieve up to
`nn_k` neighbors with cosine similarity `>= nn_min_similarity`. Each neighbor `j`
contributes its per-arm decayed reward with **weight equal to its similarity**:

```
weighted_mean(arm) = sum_j ( sim_j * decay_j * neighbor_reward_j(arm) )
                     / sum_j ( sim_j * decay_j )
```

This produces a context-local reward estimate `nn_mean(arm)` that is blended with the
regime-bucket estimate.

---

## Global prior fallback

When local support is thin (`decayed_n(arm) < MIN_SUPPORT`), blend the local estimate
with the `global_prior` for that arm using a support-weighted mix:

```
alpha   = decayed_n(arm) / (decayed_n(arm) + PRIOR_STRENGTH)
mean(arm) = alpha * local_mean(arm) + (1 - alpha) * global_prior_mean(arm)
```

---

## PSEUDOCODE

Constants (overridable per `policy_fragment.policy_params`):
`EPSILON`, `UCB_C`, `VAR_FLOOR=1e-3`, `MIN_SUPPORT=5`, `PRIOR_STRENGTH=10`,
`NN_K`, `NN_MIN_SIM`, `H = recency_half_life_ticks`, `NN_BLEND=0.5`.

### `score_tool` (general arm scorer)

```
function score_tool(arm, ctx, snapshot, mode, now_tick):
    # 1. Pull regime-bucket stat and global prior for this arm
    bucket = snapshot.lookup(ctx.regime_label, ctx.features)   # may be empty
    stat   = bucket.arm_stats.get(arm) or EMPTY_STAT
    gprior = snapshot.global_prior.get(arm) or EMPTY_STAT

    # 2. Recency decay
    d            = 0.5 ** ((now_tick - stat.last_tick) / H)
    decayed_n    = stat.n * d
    local_mean   = stat.mean_reward
    local_var    = max(stat.var_reward, VAR_FLOOR)

    # 3. Nearest-neighbor weighted reward (cosine-similarity weighted)
    neighbors = snapshot.nn_query(ctx.question_embedding, k=NN_K, min_sim=NN_MIN_SIM)
    num = 0.0 ; den = 0.0
    for nb in neighbors:
        nd = 0.5 ** ((now_tick - nb.last_tick) / H)
        w  = nb.similarity * nd                      # weight = similarity * decay
        if nb.has_arm(arm):
            num += w * nb.reward(arm)
            den += w
    nn_mean = (num / den) if den > 0 else local_mean

    # 4. Blend bucket estimate with NN estimate
    blended_mean = (1 - NN_BLEND) * local_mean + NN_BLEND * nn_mean
    blended_n    = decayed_n + den                   # effective support

    # 5. Global prior fallback (support-weighted)
    alpha     = blended_n / (blended_n + PRIOR_STRENGTH)
    mean_hat  = alpha * blended_mean + (1 - alpha) * gprior.mean_reward
    n_hat     = max(blended_n, 1e-6)

    # 6. Estimator-specific score
    if mode == EXPLOIT:                              # CONFIRM / deterministic
        score = mean_hat
    elif mode == UCB:
        N = max(snapshot.total_pulls(ctx.regime_label), 2)
        score = mean_hat + UCB_C * sqrt(ln(N) / n_hat)
    elif mode == THOMPSON:
        # Beta sampling on reward mapped to [0,1]; seeded deterministically
        a = 1 + mean_hat * n_hat
        b = 1 + (1 - mean_hat) * n_hat
        rng = seeded_rng(hash(signature_key(ctx)) ^ hash(arm))
        score = beta_sample(a, b, rng)
    else:  # EPSILON_GREEDY base score is the mean; exploration handled in chooser
        score = mean_hat

    return ScoreResult(arm=arm, mean=mean_hat, score=score,
                       n=n_hat, var=local_var,
                       confidence = 1 / (1 + sqrt(local_var / n_hat)))
```

### `choose_tool_plan` (deterministic tie-break + epsilon-greedy)

```
function choose_tool_plan(ctx, snapshot, mode, now_tick, available_arms):
    scores = []
    for arm in available_arms:                       # tool_availability-gated
        scores.append(score_tool(arm, ctx, snapshot, mode, now_tick))

    # Deterministic ordering: score desc, then arm name asc (stable tie-break)
    scores = sort(scores, key = (-s.score, s.arm))

    # epsilon-greedy exploration only during OPTIMIZE
    if mode == EPSILON_GREEDY and OPTIMIZE_ALLOWS_EXPLORATION:
        rng = seeded_rng(hash(signature_key(ctx)) ^ now_tick)
        if rng.uniform() < EPSILON:
            # explore: pick deterministically among non-top arms by seeded index
            idx     = 1 + (rng.randint() mod max(len(scores) - 1, 1))
            chosen  = scores[min(idx, len(scores)-1)]
        else:
            chosen  = scores[0]
    else:
        chosen = scores[0]                            # EXPLOIT/UCB/THOMPSON: top score

    return ToolPlan(chosen_arm   = chosen.arm,
                    ranked        = scores,
                    explanation   = numeric_explanation(scores, chosen, mode))
```

Notes:
- **Deterministic tie-break:** equal scores are broken by ascending arm name so replays
  are reproducible.
- All randomness uses `seeded_rng` keyed by the signature hash (and `now_tick` where an
  exploration roll is needed), so OPTIMIZE runs are reproducible and CONFIRM is fully
  deterministic.

### `update_reward`

```
function update_reward(snapshot, ctx, arm_family, arm, reward_terms, now_tick):
    if CONFIRM_MODE and not ONLINE_VARIANT:
        return                                       # confirm_updates_memory=false

    r = ( reward_terms.correctness_reward
        + reward_terms.evidence_quality_reward
        + reward_terms.source_authority_reward
        + reward_terms.freshness_reward
        + reward_terms.first_tool_hit_reward
        - reward_terms.cost_penalty
        - reward_terms.latency_penalty
        - reward_terms.stale_penalty
        - reward_terms.contradiction_penalty
        - reward_terms.extra_call_penalty )

    bucket = snapshot.get_or_create_bucket(ctx.regime_label, ctx.features)
    stat   = bucket.arm_stats(arm_family).get_or_create(arm)

    # Apply recency decay to existing aggregates before incorporating new sample
    d                = 0.5 ** ((now_tick - stat.last_tick) / H)
    prev_n           = stat.n * d
    prev_mean        = stat.mean_reward
    prev_m2          = stat.var_reward * max(prev_n, 1)   # decayed sum of sq. dev.

    new_n    = prev_n + 1
    delta    = r - prev_mean
    new_mean = prev_mean + delta / new_n
    new_m2   = prev_m2 + delta * (r - new_mean)           # Welford update
    new_var  = new_m2 / max(new_n, 1)

    stat.n            = new_n
    stat.decayed_n    = new_n
    stat.mean_reward  = new_mean
    stat.var_reward   = new_var
    stat.last_tick    = now_tick

    # Raw trace + answer/evidence text go to the cold archive, NOT here.
    archive_raw_trace(ctx, arm_family, arm, reward_terms)
```

### `consolidate_policy_fragment`

```
function consolidate_policy_fragment(snapshot, regime_label, signature, now_tick):
    bucket = snapshot.lookup(regime_label, signature)
    support = bucket.total_decayed_n()
    if support < FRAGMENT_MIN_SUPPORT:
        return None                                  # not stable enough to promote

    arm_priors = {}
    for family in ["tool", "query_template", "verification", "stop"]:
        arm_priors[family] = {}
        for arm, stat in bucket.arm_stats(family):
            d = 0.5 ** ((now_tick - stat.last_tick) / H)
            arm_priors[family][arm] = {
                "n":         stat.n,
                "decayed_n": stat.n * d,
                "mean_reward": stat.mean_reward,
                "var_reward":  stat.var_reward
            }

    fragment = PolicyFragment(
        regime_label = regime_label,
        query_signature = signature,
        policy_params = inherit_or_default_params(bucket),  # epsilon, ucb_c, nn_k, ...
        arm_priors = arm_priors,
        support_n = support,
        stable = (support >= FRAGMENT_STABLE_SUPPORT
                  and max_arm_variance(arm_priors) <= STABLE_VAR_MAX),
        provenance = { "consolidated_from_snapshot": snapshot.id,
                       "last_tick": now_tick }
    )
    # No answer/evidence text is ever placed in the fragment.
    return fragment
```

### Regimes-layer mutation gating (OPTIMIZE then CONFIRM)

The regimes layer proposes **bounded** mutations to `policy_fragment.policy_params`
(e.g., nudge `epsilon`, `ucb_c`, `stop_confidence_threshold`, `nn_k` within clamps),
emits a `policy_update`, and gates promotion:

```
function gate_promotion(candidate_fragment, optimize_set, confirm_set):
    # OPTIMIZE: in-sample, exploration allowed, memory may update
    opt = evaluate(candidate_fragment, optimize_set,
                   mode=EPSILON_GREEDY, allow_explore=true)
    if opt.main_metric <= baseline.main_metric + OPT_MARGIN:   # correct_per_tool_call
        return PromotionDecision(promote=false, reason="optimize_no_gain")

    # CONFIRM: held-out, frozen snapshot, deterministic exploitation, no memory writes
    conf = evaluate(freeze(candidate_fragment), confirm_set,
                    mode=EXPLOIT, allow_explore=false, update_memory=false)
    promote = conf.main_metric >= baseline.main_metric + CONFIRM_MARGIN
    return PromotionDecision(promote=promote,
                             optimize=opt, confirm=conf,
                             reason = "confirmed" if promote else "confirm_failed")
```

The `main_metric` for gating is `correct_per_tool_call` and budget caps `[1,3,5,10]`
remain binding throughout.
