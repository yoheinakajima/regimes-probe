# regimes-probe run `demo`

## Eligibility

- **structurally_valid: True**
- **headline_eligible_memory_claim: False**
- dataset_is_real: False
- conditions_present: ['closed_book', 'no_memory_search', 'policy_memory', 'random_memory']
- reasons NOT headline-eligible (memory claim):
  - dataset is a synthetic/placeholder fixture — not a benchmark headline
- same_conditions (no_memory_search vs policy_memory) ok: True (unexpected diffs: [])

## Headline result

Frozen policy memory vs no_memory_search baseline, correct_per_tool_call — budget 1: 0.000 → 0.750; budget 3: 0.095 → 0.656; budget 5: 0.086 → 0.618; budget 10: 0.021 → 0.618.

## Models, providers, tools

- answer_model: `gpt-5.4-mini`
- search_baseline: `openai_web_search`
- tools_enabled: ['generic_web_search', 'news_search', 'official_domain_search', 'brave_search']
- embedder: `hash_embedder`

## Benchmark / split

- dataset: `synthetic_browse`  version: `synthetic_browse@44963c46dfcf`
- split: {'mode': 'hash', 'salt': 'regimes-probe-v0', 'n_optimize': 35, 'n_confirm': 28, 'optimize_ids': '...', 'confirm_ids': '...'}
- budget caps: [1, 3, 5, 10]
- memory condition: `frozen_policy_memory`
- policy condition: `query=learned,stop=learned`

## Metrics by condition and budget

| condition | budget | accuracy | correct_per_tool_call | mean_calls | first_tool_hit |
|---|---|---|---|---|---|
| closed_book | 0 | 0.036 | 0.000 | 0.00 | 0.000 |
| no_memory_search | 1 | 0.000 | 0.000 | 1.00 | 0.000 |
| policy_memory | 1 | 0.750 | 0.750 | 1.00 | 0.750 |
| random_memory | 1 | 0.286 | 0.286 | 1.00 | 0.286 |
| no_memory_search | 3 | 0.286 | 0.095 | 3.00 | 0.000 |
| policy_memory | 3 | 0.750 | 0.656 | 1.14 | 0.750 |
| random_memory | 3 | 0.500 | 0.275 | 1.82 | 0.286 |
| no_memory_search | 5 | 0.429 | 0.086 | 5.00 | 0.000 |
| policy_memory | 5 | 0.750 | 0.618 | 1.21 | 0.750 |
| random_memory | 5 | 0.500 | 0.222 | 2.25 | 0.286 |
| no_memory_search | 10 | 0.214 | 0.021 | 10.00 | 0.000 |
| policy_memory | 10 | 0.750 | 0.618 | 1.21 | 0.750 |
| random_memory | 10 | 0.464 | 0.157 | 2.96 | 0.286 |

## Efficiency & epistemic-error rates

| condition | budget | over_search | false_stop | stale_err | evidence_gain/call | provider_fail_rate |
|---|---|---|---|---|---|---|
| closed_book | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| no_memory_search | 1 | 0.000 | 0.000 | 0.000 | 0.112 | 0.000 |
| policy_memory | 1 | 0.000 | 0.000 | 0.000 | 0.913 | 0.000 |
| random_memory | 1 | 0.000 | 0.000 | 0.250 | 0.530 | 0.000 |
| no_memory_search | 3 | 0.000 | 0.000 | 0.250 | 0.201 | 0.000 |
| policy_memory | 3 | 0.000 | 0.179 | 0.000 | 0.798 | 0.000 |
| random_memory | 3 | 0.036 | 0.321 | 0.250 | 0.436 | 0.000 |
| no_memory_search | 5 | 0.214 | 0.000 | 0.250 | 0.122 | 0.000 |
| policy_memory | 5 | 0.000 | 0.250 | 0.000 | 0.793 | 0.000 |
| random_memory | 5 | 0.036 | 0.321 | 0.250 | 0.375 | 0.000 |
| no_memory_search | 10 | 0.214 | 0.000 | 0.250 | 0.061 | 0.000 |
| policy_memory | 10 | 0.000 | 0.250 | 0.000 | 0.793 | 0.000 |
| random_memory | 10 | 0.000 | 0.393 | 0.250 | 0.285 | 0.000 |

## Provider failures

- none recorded (no provider errors during tool calls).

## Statistical tests

```json
{
  "budget": 3,
  "mcnemar": {
    "b_only_baseline_correct": 0,
    "c_only_treatment_correct": 13,
    "statistic": 11.0769,
    "p_value": 0.0009,
    "n_discordant": 13
  },
  "policy_correct_per_tool_call_ci": {
    "point": 0.65625,
    "ci_lo": 0.447368,
    "ci_hi": 0.866667,
    "level": 0.95
  }
}
```

## Failure regimes (dominant on evaluated set)

- route_miss: 193
- query_miss: 193
- verification_miss: 114
- evidence_sparse: 109
- contradiction_unresolved: 51
- stale_evidence: 49
- stop_too_early: 48
- under_search: 48

## Promotions

- none

## Replay

- projection_matches: **True** (998 events)

## Limitations

- Dataset is `synthetic_browse` — a synthetic/placeholder harness, NOT a real benchmark score.
- The synthetic answerer models a competent extractor; it isolates retrieval/epistemic policy.
- No live providers were called; only fixture-backed, deterministic tools.

## What is NOT claimed

- No claim of BrowseComp or LiveBrowseComp performance.
- No claim base model weights changed (they do not).
- No claim benchmark answers are stored in policy memory (they are not).
