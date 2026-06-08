# regimes-probe run `demo`

## Headline result

Frozen policy memory vs no-memory baseline, correct_per_tool_call — budget 1: 0.000 → 0.792; budget 3: 0.083 → 0.655; budget 5: 0.100 → 0.655; budget 10: 0.025 → 0.655.

## Models, providers, tools

- answer_model: `gpt-5.5`
- search_baseline: `openai_web_search`
- tools_enabled: ['generic_web_search', 'news_search', 'official_domain_search', 'brave_search']
- embedder: `hash_embedder`

## Benchmark / split

- dataset: `synthetic_browse`  version: `synthetic_browse@e4fbb8440607`
- split: {'mode': 'hash', 'salt': 'regimes-probe-v0', 'n_optimize': 32, 'n_confirm': 24, 'optimize_ids': '...', 'confirm_ids': '...'}
- budget caps: [1, 3, 5, 10]
- memory condition: `frozen_policy_memory`
- policy condition: `query=learned,stop=learned`

## Metrics by condition and budget

| condition | budget | accuracy | correct_per_tool_call | mean_calls | first_tool_hit |
|---|---|---|---|---|---|
| no_memory | 1 | 0.000 | 0.000 | 1.00 | 0.000 |
| policy_memory | 1 | 0.792 | 0.792 | 1.00 | 0.792 |
| random_memory | 1 | 0.250 | 0.250 | 1.00 | 0.250 |
| no_memory | 3 | 0.250 | 0.083 | 3.00 | 0.000 |
| policy_memory | 3 | 0.792 | 0.655 | 1.21 | 0.792 |
| random_memory | 3 | 0.500 | 0.308 | 1.62 | 0.250 |
| no_memory | 5 | 0.500 | 0.100 | 5.00 | 0.000 |
| policy_memory | 5 | 0.792 | 0.655 | 1.21 | 0.792 |
| random_memory | 5 | 0.500 | 0.267 | 1.88 | 0.250 |
| no_memory | 10 | 0.250 | 0.025 | 10.00 | 0.000 |
| policy_memory | 10 | 0.792 | 0.655 | 1.21 | 0.792 |
| random_memory | 10 | 0.375 | 0.150 | 2.50 | 0.250 |

## Efficiency & epistemic-error rates

| condition | budget | over_search | false_stop | stale_err | evidence_gain/call |
|---|---|---|---|---|---|
| no_memory | 1 | 0.000 | 0.000 | 0.000 | 0.106 |
| policy_memory | 1 | 0.000 | 0.000 | 0.000 | 0.823 |
| random_memory | 1 | 0.000 | 0.000 | 0.292 | 0.523 |
| no_memory | 3 | 0.000 | 0.000 | 0.292 | 0.183 |
| policy_memory | 3 | 0.000 | 0.208 | 0.000 | 0.802 |
| random_memory | 3 | 0.125 | 0.500 | 0.292 | 0.519 |
| no_memory | 5 | 0.250 | 0.000 | 0.292 | 0.116 |
| policy_memory | 5 | 0.000 | 0.208 | 0.000 | 0.802 |
| random_memory | 5 | 0.125 | 0.500 | 0.292 | 0.450 |
| no_memory | 10 | 0.250 | 0.000 | 0.292 | 0.058 |
| policy_memory | 10 | 0.000 | 0.208 | 0.000 | 0.802 |
| random_memory | 10 | 0.000 | 0.500 | 0.292 | 0.338 |

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
    "point": 0.655172,
    "ci_lo": 0.454545,
    "ci_hi": 0.92,
    "level": 0.95
  }
}
```

## Failure regimes (dominant on evaluated set)

- route_miss: 140
- query_miss: 140
- evidence_sparse: 68
- stop_too_early: 51
- under_search: 51
- stale_evidence: 49
- over_search: 18
- stop_too_late: 18

## Promotions

- none

## Replay

- projection_matches: **True** (912 events)

## Limitations

- Dataset is `synthetic_browse` — a synthetic/placeholder harness, NOT a real benchmark score.
- The synthetic answerer models a competent extractor; it isolates retrieval/epistemic policy.
- No live providers were called; only fixture-backed, deterministic tools.

## What is NOT claimed

- No claim of BrowseComp or LiveBrowseComp performance.
- No claim base model weights changed (they do not).
- No claim benchmark answers are stored in policy memory (they are not).
