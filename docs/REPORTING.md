# Reporting

This document specifies every artifact `regimes-probe` writes for a run, the required
sections of the human-readable summary, the column schema of every CSV, and the JSON
skeleton that ties the run together.

The reporting discipline is load-bearing: **no claim is made unless it appears in a
committed report artifact** (`report.created`). `OPTIMIZE` numbers are recorded but never
headlined; `CONFIRM` numbers are the only basis for headline claims. See
[`REGIMES_IMPROVEMENT_LOOP.md`](./REGIMES_IMPROVEMENT_LOOP.md).

## Generated outputs

All artifacts for a run land under `results/{run_id}/`. `report.created` is emitted once
every file below is written.

| Path | Type | Purpose |
| --- | --- | --- |
| `results/{run_id}/report.json` | JSON | Machine-readable master artifact: run metadata, per-condition metrics, statistical tests, promotions. Source of truth for all claims. |
| `results/{run_id}/summary.md` | Markdown | Human-readable headline report. Required-sections checklist below. |
| `results/{run_id}/budget_curve.csv` | CSV | Primary metric and accuracy as a function of budget `{1,3,5,10}` per condition. |
| `results/{run_id}/per_question.csv` | CSV | One row per (item, condition, budget): correctness, cost, evidence gain, regime. |
| `results/{run_id}/tool_rewards.csv` | CSV | Per-`signature_cluster`, per-tool reward attribution (route seam). |
| `results/{run_id}/query_rewards.csv` | CSV | Per-`signature_cluster`, per-query-template evidence-gain attribution (query seam). |
| `results/{run_id}/stop_verify_rewards.csv` | CSV | Stop/verify seam rewards: over/under-search and verification outcomes. |
| `results/{run_id}/memory_snapshot.json` | JSON | The `policy_memory_snapshot` used (and/or produced) by the run. Answer-free Layer-2 object. |
| `results/{run_id}/policy_updates.json` | JSON | All `policy_update` proposals and their `promotion_decision`s for the run. |
| `results/{run_id}/replay_check.md` | Markdown | Determinism attestation: re-projecting the event log reproduces graph + rewards byte-for-byte. |

## `summary.md` — required-sections checklist

`summary.md` MUST contain every section below. A run report is invalid (and must not be
cited) if any section is missing. Headline result is drawn from `CONFIRM` only.

- [ ] **Headline result** — held-out (`CONFIRM`) `correct_per_tool_call`, main vs control, with CI. Marked as the only headline claim.
- [ ] **Model and provider versions** — base model id(s), provider, embedding model, judge model, with versions/pins.
- [ ] **Tools enabled** — the ActiveGraph-wrapped tools available this run (e.g. `openai_web_search`, `brave_search`, `page_fetch`).
- [ ] **Benchmark / dataset version** — dataset adapter, version tag, content checksum.
- [ ] **Split details** — `OPTIMIZE` and `CONFIRM` sizes, disjointness attestation, split seed/hash.
- [ ] **Budget caps** — budgets evaluated `{1,3,5,10}` and per-attempt tool-call cap.
- [ ] **Memory condition** — none / frozen-snapshot / consolidating; snapshot id.
- [ ] **Policy condition** — bandit level (L1 route / L2 epistemic), exploration setting, frozen vs exploring.
- [ ] **Correctness metrics** — accuracy and abstain rate per condition and budget.
- [ ] **Efficiency metrics** — `correct_per_tool_call` (primary), mean tool calls, evidence-gain per call.
- [ ] **Failure regimes** — dominant `regime_label`s with shares, per condition.
- [ ] **Promoted / rejected updates** — every `policy_update` with its `promotion_decision` (accepted/rejected) and the `CONFIRM` delta.
- [ ] **Limitations** — sample size, provider drift, replay caveats, synthetic-vs-live caveats.
- [ ] **What is not claimed** — explicit non-claims (e.g. no causal claim beyond the tested seams; `OPTIMIZE` gains are not headline; no factual-knowledge claim).

## CSV column schemas

### `budget_curve.csv`

| Column | Type | Description |
| --- | --- | --- |
| `budget` | int | Tool-call budget; one of `{1,3,5,10}`. |
| `condition` | string | Experimental condition (e.g. `control`, `memory_frozen`, `memory_consolidating`). |
| `accuracy` | float | Fraction correct at this budget/condition (`[0,1]`). |
| `correct_per_tool_call` | float | **Primary metric**: correct answers / total tool calls. |
| `n` | int | Number of attempts contributing to the row. |

### `per_question.csv`

| Column | Type | Description |
| --- | --- | --- |
| `item_id` | string | `benchmark_item` id. |
| `condition` | string | Experimental condition. |
| `budget` | int | Budget for this attempt; one of `{1,3,5,10}`. |
| `correct` | int | `1` correct, `0` incorrect; abstain encoded separately (`-1`) if used. |
| `tool_calls` | int | Number of tool calls consumed by the attempt. |
| `evidence_gain` | float | Total evidence-gain reward attributed to the attempt. |
| `regime` | string | Dominant `regime_label` for the attempt (empty if correct/no failure). |

### `tool_rewards.csv` (route seam)

| Column | Type | Description |
| --- | --- | --- |
| `signature_cluster` | string | `query_signature` cluster key. |
| `tool` | string | Tool name (e.g. `openai_web_search`). |
| `mean_reward` | float | Mean `tool_reward` for (cluster, tool). |
| `count` | int | Number of `tool_call`s in the cell. |
| `first_tool_hit_rate` | float | Fraction where this tool, used first, yielded usable evidence. |
| `condition` | string | Experimental condition. |

### `query_rewards.csv` (query seam)

| Column | Type | Description |
| --- | --- | --- |
| `signature_cluster` | string | `query_signature` cluster key. |
| `query_template` | string | Query-template id from the template library. |
| `mean_evidence_reward` | float | Mean `evidence_reward` for (cluster, template). |
| `evidence_gain_per_call` | float | Evidence gain normalized by tool calls. |
| `count` | int | Number of `query_plan`s using the template. |
| `condition` | string | Experimental condition. |

### `stop_verify_rewards.csv` (stop / verify seams)

| Column | Type | Description |
| --- | --- | --- |
| `signature_cluster` | string | `query_signature` cluster key. |
| `seam` | string | `stop` or `verify`. |
| `mean_reward` | float | Mean `trace_reward` contribution from the seam. |
| `over_search_rate` | float | Fraction of attempts flagged `over_search`. |
| `under_search_rate` | float | Fraction flagged `under_search`. |
| `false_stop_rate` | float | Fraction halted before sufficiency was reached. |
| `verification_miss_rate` | float | Fraction accepted without adequate support check. |
| `count` | int | Number of attempts in the cell. |
| `condition` | string | Experimental condition. |

## `report.json` skeleton

`report.json` is the master machine-readable artifact. Every claim in `summary.md` must be
derivable from it. Headline claims use `conditions[*].confirm` only.

```json
{
  "object_type": "report",
  "report_id": "report_run_conf_0043b",
  "run_id": "run_conf_0043b",
  "created_event": "report.created",
  "run_metadata": {
    "created_at": "2026-06-07T00:00:00Z",
    "git_commit": "<sha>",
    "model": {
      "answerer_model": "claude-...",
      "provider": "anthropic",
      "embedding_model": "openai:text-embedding-3-small | HashEmbedder",
      "judge_model": "claude-..."
    },
    "tools_enabled": ["openai_web_search", "brave_search", "page_fetch"],
    "dataset": {
      "adapter": "BrowseCompAdapter",
      "version": "<tag>",
      "checksum": "<sha256>"
    },
    "split": {
      "optimize_n": 0,
      "confirm_n": 0,
      "disjoint": true,
      "split_hash": "<sha256>"
    },
    "budgets": [1, 3, 5, 10],
    "tool_call_cap": 10,
    "memory_condition": "memory_frozen",
    "policy_condition": "L2_epistemic_frozen",
    "snapshot_id": "snap_candidate_0008",
    "replay": { "deterministic": true, "replay_check": "replay_check.md" }
  },
  "conditions": [
    {
      "condition": "control",
      "optimize": { "headline": false, "accuracy": 0.0, "correct_per_tool_call": 0.0, "mean_tool_calls": 0.0, "n": 0 },
      "confirm":  { "headline": true,  "accuracy": 0.0, "correct_per_tool_call": 0.0, "mean_tool_calls": 0.0, "n": 0 },
      "by_budget": [
        { "budget": 1,  "accuracy": 0.0, "correct_per_tool_call": 0.0, "n": 0 },
        { "budget": 3,  "accuracy": 0.0, "correct_per_tool_call": 0.0, "n": 0 },
        { "budget": 5,  "accuracy": 0.0, "correct_per_tool_call": 0.0, "n": 0 },
        { "budget": 10, "accuracy": 0.0, "correct_per_tool_call": 0.0, "n": 0 }
      ],
      "regime_shares": { "over_search": 0.0, "route_miss": 0.0 }
    },
    {
      "condition": "memory_frozen",
      "optimize": { "headline": false, "accuracy": 0.0, "correct_per_tool_call": 0.0, "mean_tool_calls": 0.0, "n": 0 },
      "confirm":  { "headline": true,  "accuracy": 0.0, "correct_per_tool_call": 0.0, "mean_tool_calls": 0.0, "n": 0 },
      "by_budget": [],
      "regime_shares": {}
    }
  ],
  "statistical_tests": {
    "primary_metric": "correct_per_tool_call",
    "comparison": "memory_frozen_vs_control_on_confirm",
    "mcnemar": {
      "split": "confirm",
      "b": 0,
      "c": 0,
      "statistic": 0.0,
      "p_value": 0.0,
      "corrected": true
    },
    "bootstrap_ci": {
      "split": "confirm",
      "metric": "correct_per_tool_call",
      "point_estimate_delta": 0.0,
      "ci_lower": 0.0,
      "ci_upper": 0.0,
      "confidence": 0.95,
      "n_resamples": 10000,
      "seed": 0
    }
  },
  "promotions": [
    {
      "promotion_decision_id": "prom_2026_06_07_0008",
      "policy_update_id": "pu_2026_06_07_over_search_001",
      "regime_label": "over_search",
      "decision": "accepted",
      "decision_event": "promotion.accepted",
      "from_snapshot_id": "snap_baseline_0007",
      "to_snapshot_id": "snap_candidate_0008",
      "optimize_delta": 0.023,
      "confirm_delta": 0.018,
      "confirm_regression_tolerance": 0.01,
      "grounded_in_report": true
    }
  ],
  "claims": {
    "headline_source": "conditions[*].confirm",
    "optimize_is_headline": false,
    "not_claimed": [
      "No factual-knowledge claim; only procedural policy is evaluated.",
      "OPTIMIZE (in-sample) gains are not headline.",
      "No causal claim beyond the tested epistemic seams."
    ]
  }
}
```
