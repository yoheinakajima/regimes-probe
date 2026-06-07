# POLICY_MEMORY.md

## Purpose

`regimes-probe` learns a **procedural epistemic search policy** (where to look, how to
query, when to verify, when to stop) — it does **not** learn or memorize factual
answers. Policy memory is the ActiveGraph-native procedural store that holds
**trace-derived priors and reward statistics**, never answer text.

The event log is the source of truth. Behaviors are deterministic. Policy memory is a
**derived, consolidated projection** of the raw trace archive used on the hot path to
score arms in the contextual bandit (see `docs/CONTEXTUAL_BANDIT.md`).

> **Hard rule:** No answer memory. Policy memory stores *how to search*, not *what the
> answer was*.

---

## What policy memory stores

Policy memory stores **generic, trace-derived priors** keyed by latent regime and by
query signature — not by topic. There are **no predefined topical categories**. Clusters
are *learned* (latent regimes) rather than hand-labeled domains.

| Stored item | Description | Answer text? |
|---|---|---|
| `question_embedding` | Dense vector of the normalized question, for nearest-neighbor retrieval | No |
| `question_text_hash` | Hash of normalized question text (dedup / signature key) | No (hash only) |
| `lexical_features` | Token count, entity count, has-quotes, has-date, wh-word, comparative, superlative, numeric-answer-shape | No |
| `result_features` | Result-set stats from past attempts: hit count, mean source authority, mean freshness, contradiction rate, sparsity | No |
| `tool_rewards` | Reward statistics keyed by **tool arm** (search/fetch tools) | No |
| `query_template_rewards` | Reward statistics keyed by **query policy arm** | No |
| `verification_rewards` | Reward statistics keyed by **verification check / stop arm** | No |
| `stopping_rewards` | Reward statistics keyed by **stop policy arm** | No |
| `policy_fragments` | Consolidated, promotable units of policy (priors over arms for a signature) | No |

Each `*_rewards` entry holds **aggregated reward statistics** (count, mean, variance,
recency-decayed count, last-update tick) keyed by **arm**, never raw answer strings.

### What is explicitly NOT stored in policy memory

- `final_answer` text or `candidate_answer` text
- Full question/answer pairs for retrieval
- Verbatim evidence passages or page bodies
- Any field that would let the system "look up the answer" instead of "learn to search"

Raw evidence text and answer text live only in the **raw trace archive** (cold storage,
for grading/replay/audit), which is **separated** from the policy memory snapshot.

---

## Latent regimes, not topical categories

The benchmark must not encode fixed domains (e.g., "sports", "finance"). Instead:

- **Latent regimes** are discovered by clustering over `question_embedding` +
  `lexical_features` + `result_features`. A `regime_label` is a cluster id, not a topic
  name.
- Priors are conditioned on `(regime_label, query_signature)` and on nearest-neighbor
  traces, so generalization is by **epistemic structure**, not subject matter.

---

## Query signatures: generic epistemic features

A `query_signature` is a set of **epistemic properties** of the question — properties
about *how the answer must be searched for and verified*, NOT what the question is about.

| Feature | Epistemic meaning (NOT a topic) |
|---|---|
| `freshness-sensitive` | Correct answer changes over time; recent sources required |
| `source-authority-sensitive` | Answer correctness depends on authoritative sources |
| `answer-shape-constrained` | Answer must match a constrained shape (date, number, name, enum) |
| `query-fragile` | Small query wording changes drastically change recall |
| `evidence-sparse` | Supporting evidence is rare across the corpus |
| `staleness-prone` | High-ranked sources are frequently outdated |
| `multihop-likely` | Answer requires composing multiple evidence observations |
| `verification-heavy` | High risk of contradiction; corroboration usually pays off |

These are computed from features and trace statistics, are domain-agnostic, and form the
**conditioning context** for arm selection. Two questions on completely different topics
that share `{freshness-sensitive, answer-shape-constrained}` should share priors.

---

## Leakage controls

1. **No final answer in hot-path policy memory.** The policy memory snapshot schema has
   no field that can hold `final_answer` / `candidate_answer` text. Enforced at write
   time (schema validation rejects answer-text fields).
2. **No full Q/A retrieval.** Retrieval returns reward priors and arm statistics for
   nearest signatures, never the answers those neighbors produced.
3. **Separate raw trace archive from policy memory snapshot.** Raw traces (with evidence
   and answer text) are written to a cold archive used only for grading, audit, and
   offline replay. The snapshot is a sanitized projection.
4. **Freeze memory before CONFIRM.** Before a CONFIRM run, the active
   `policy_memory_snapshot` is frozen (`frozen=true`, immutable hash). CONFIRM reads the
   frozen snapshot and `confirm_updates_memory=false`; no writes occur during CONFIRM.
   OPTIMIZE may update memory and allows exploration
   (`optimize_allows_exploration=true`).

```
raw trace archive (cold)            policy_memory_snapshot (hot)
---------------------               ----------------------------
evidence text          --consolidate-->   reward stats by arm
answer text            (sanitize/strip)    embeddings + features
full tool_call I/O                         policy_fragments
                                           (frozen before CONFIRM)
```

---

## Consolidation flow

1. A `question_attempt` produces `tool_call`, `evidence_observation`, `candidate_answer`,
   `final_answer`, `grade_result`, and reward objects (`tool_reward`, `evidence_reward`,
   `trace_reward`). These go to the **raw trace archive**.
2. Consolidation strips answer/evidence text and updates **reward statistics by arm** for
   the matching `(regime_label, query_signature)` bucket and nearest-neighbor index.
3. When statistics cross a stability threshold, a `policy_fragment` is emitted/updated.
4. Regimes layer proposes bounded mutations to policy-parameter fields, gated by OPTIMIZE
   then CONFIRM (see `docs/CONTEXTUAL_BANDIT.md` → `consolidate_policy_fragment` and the
   `promotion_decision` flow).

---

## JSON schema example: `policy_memory_snapshot`

```json
{
  "object_type": "policy_memory_snapshot",
  "snapshot_id": "pms_2026_06_07_a1",
  "created_tick": 184213,
  "frozen": true,
  "frozen_hash": "sha256:7b3c…e91",
  "embedding_dim": 768,
  "recency_half_life_ticks": 50000,
  "regimes": [
    {
      "regime_label": "regime_07",
      "centroid_embedding_ref": "vec://pms_2026_06_07_a1/centroids/07",
      "query_signature": {
        "features": ["freshness-sensitive", "answer-shape-constrained", "staleness-prone"]
      },
      "result_features": {
        "mean_hit_count": 4.2,
        "mean_source_authority": 0.71,
        "mean_freshness": 0.55,
        "contradiction_rate": 0.18,
        "sparsity": 0.22
      },
      "tool_rewards": {
        "web_search": { "n": 412, "decayed_n": 188.4, "mean_reward": 0.63, "var_reward": 0.041, "last_tick": 184010 },
        "news_search": { "n": 230, "decayed_n": 121.9, "mean_reward": 0.71, "var_reward": 0.038, "last_tick": 184120 },
        "fetch_page": { "n": 190, "decayed_n": 88.0, "mean_reward": 0.52, "var_reward": 0.060, "last_tick": 183990 }
      },
      "query_template_rewards": {
        "freshness_terms":     { "n": 150, "decayed_n": 80.1, "mean_reward": 0.69, "var_reward": 0.033, "last_tick": 184118 },
        "exact_answer_shape":  { "n": 121, "decayed_n": 64.7, "mean_reward": 0.66, "var_reward": 0.045, "last_tick": 184101 },
        "direct_question":     { "n": 98,  "decayed_n": 41.2, "mean_reward": 0.48, "var_reward": 0.052, "last_tick": 183900 }
      },
      "verification_rewards": {
        "corroborate":         { "n": 88, "decayed_n": 47.3, "mean_reward": 0.58, "var_reward": 0.049, "last_tick": 184090 },
        "contradiction_check": { "n": 60, "decayed_n": 31.0, "mean_reward": 0.61, "var_reward": 0.055, "last_tick": 184050 }
      },
      "stopping_rewards": {
        "stop_now":    { "n": 300, "decayed_n": 140.0, "mean_reward": 0.40, "var_reward": 0.070, "last_tick": 184100 },
        "search_more": { "n": 210, "decayed_n": 99.5,  "mean_reward": 0.55, "var_reward": 0.061, "last_tick": 184095 }
      },
      "policy_fragment_refs": ["pf_regime07_fresh_shape_v3"]
    }
  ],
  "nn_index_ref": "ann://pms_2026_06_07_a1/traces",
  "global_prior": {
    "tool_rewards":           { "web_search": { "n": 50000, "mean_reward": 0.50, "var_reward": 0.08 } },
    "query_template_rewards": { "direct_question": { "n": 50000, "mean_reward": 0.47, "var_reward": 0.08 } },
    "stopping_rewards":       { "search_more": { "n": 50000, "mean_reward": 0.49, "var_reward": 0.08 } }
  }
}
```

> Note every leaf is a **reward statistic keyed by arm**. There is no `answer`,
> `final_answer`, or evidence-text field anywhere in the snapshot.

---

## JSON schema example: `policy_fragment`

A `policy_fragment` is a promotable unit of consolidated policy for a
`(regime_label, query_signature)`. It stores **reward statistics keyed by arm** plus the
policy-parameter values the regimes layer may mutate. It stores **NO answer text**.

```json
{
  "object_type": "policy_fragment",
  "fragment_id": "pf_regime07_fresh_shape_v3",
  "regime_label": "regime_07",
  "query_signature": {
    "features": ["freshness-sensitive", "answer-shape-constrained", "staleness-prone"]
  },
  "version": 3,
  "stable": true,
  "support_n": 612,
  "policy_params": {
    "epsilon": 0.07,
    "ucb_c": 1.10,
    "recency_half_life_ticks": 50000,
    "nn_k": 12,
    "nn_min_similarity": 0.60,
    "stop_confidence_threshold": 0.62,
    "corroboration_min_sources": 2
  },
  "arm_priors": {
    "tool": {
      "news_search": { "n": 230, "decayed_n": 121.9, "mean_reward": 0.71, "var_reward": 0.038 },
      "web_search":  { "n": 412, "decayed_n": 188.4, "mean_reward": 0.63, "var_reward": 0.041 }
    },
    "query_template": {
      "freshness_terms":    { "n": 150, "decayed_n": 80.1, "mean_reward": 0.69, "var_reward": 0.033 },
      "exact_answer_shape": { "n": 121, "decayed_n": 64.7, "mean_reward": 0.66, "var_reward": 0.045 }
    },
    "verification": {
      "corroborate":         { "n": 88, "decayed_n": 47.3, "mean_reward": 0.58, "var_reward": 0.049 },
      "contradiction_check": { "n": 60, "decayed_n": 31.0, "mean_reward": 0.61, "var_reward": 0.055 }
    },
    "stop": {
      "search_more": { "n": 210, "decayed_n": 99.5, "mean_reward": 0.55, "var_reward": 0.061 },
      "stop_now":    { "n": 300, "decayed_n": 140.0, "mean_reward": 0.40, "var_reward": 0.070 }
    }
  },
  "provenance": {
    "consolidated_from_snapshot": "pms_2026_06_07_a1",
    "last_tick": 184120,
    "promotion_decision_ref": "promo_2026_06_07_07"
  }
}
```

> The fragment is fully describable as: *for this latent regime and epistemic signature,
> these are the reward-weighted preferences over arms and these are the policy
> parameters.* No factual content is present.
