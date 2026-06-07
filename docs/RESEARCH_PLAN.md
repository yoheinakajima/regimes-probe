# Research Plan

This document defines the research sequence for `regimes-probe`. Each study is run as one
or more `benchmark_run`s and produces a `report`. All studies hold the **base model
weights fixed** and store **no benchmark answers** in hot-path policy memory.

## Shared conventions

- **Primary metric:** `correct_per_tool_call` (`main_metric`).
- **Budget tracks:** `budgets = [1, 3, 5, 10]` tool calls per `question_attempt`.
- **Splits:** deterministic `OPTIMIZE`/`CONFIRM` split (see
  [EVALUATION_PROTOCOL.md](./EVALUATION_PROTOCOL.md)). Learning happens on `OPTIMIZE`;
  scoring happens on a **frozen** `policy_memory_snapshot` over `CONFIRM`.
- **Determinism:** every run is replayable; the replay check must pass before a `report`
  is considered valid.
- **Default config:** `answer_model=gpt-5.5`, `cheaper_answer_model=gpt-5.4-mini`,
  `search_baseline=openai_web_search`, `benchmark_primary=LiveBrowseComp`,
  `benchmark_secondary=BrowseComp`.

---

## Study 0 — Harness validity

**Goal.** Prove the harness is correct, deterministic, replayable, and budget-bounded
*before* spending any API budget. This is the gate for all later studies.

**Setup.**
- `SyntheticBrowseAdapter` provides a synthetic fixture of `benchmark_item`s with known
  ground truth and known "correct tool / correct query" so reward attribution can be
  checked against an oracle.
- **Deterministic fake tools**: search and `page_fetch` tools return fixed responses keyed
  by request; **no API keys required**, no network.
- `HashEmbedder` for `query_signature` so embeddings are deterministic.

**Conditions.**
- `no_memory_baseline` vs `policy_memory` on the synthetic fixture.
- A `random_memory_control` to confirm the harness can detect *no* learning.
- A deliberate replay re-run to confirm byte-for-byte reproduction of graph + rewards.

**Metrics / checks.**
- Replay check passes (event log re-projects identically).
- Budget caps are never exceeded for any `budget in {1,3,5,10}`.
- Reward attribution matches the synthetic oracle (`tool_reward`, `evidence_reward`,
  `trace_reward` land on the right calls/attempts).
- `policy_memory` strictly beats `random_memory_control` on the oracle fixture
  (sanity that learning is detectable).

---

## Study 1 — Level 1 routing memory

**Goal.** Test whether a contextual bandit can learn **tool routing** keyed by
`query_signature`, improving `correct_per_tool_call` with the base model, tools, and
prompts held identical.

**Setup.**
- Same `answer_model`, same tool roster, same prompts across conditions.
- The only thing that varies is whether the router consults `policy_memory`.
- Experience phase on `OPTIMIZE`; freeze snapshot; score on `CONFIRM`.

**Conditions.**
| Condition | Router behavior |
| --- | --- |
| `no_memory` | route with a fixed default (e.g. `openai_web_search`) / uniform policy |
| `policy_memory` | contextual-bandit routing over the adapter roster, keyed by `query_signature` |
| `random_memory_control` | bandit parameters drawn from a frozen random snapshot |

**Metrics.**
- Primary: `correct_per_tool_call`, reported per budget cap `{1,3,5,10}`.
- Secondary: `first_tool_hit_rate`, `accuracy`, `correct_per_dollar`,
  `correct_per_second`, `evidence_gain_per_call`.
- Paired per-question comparison `policy_memory` vs `no_memory`; bootstrap CIs on
  `correct_per_tool_call`; budget-curve comparison.

---

## Study 2 — Level 2 query policy

**Goal.** Add the **query-formulation** seam: learn which query template/phrasing a
signature should use, measured by evidence gain and downstream correctness.

**Setup.**
- Build on Study 1's routing. Add `query_plan` selection from a learned bandit over query
  templates, keyed by `query_signature`.
- All conditions share routing policy from Study 1 to isolate the query seam.

**Conditions.**
| Condition | Query seam |
| --- | --- |
| `fixed_generic_query` | use the raw question text / one fixed generic template |
| `llm_query_generation` | answer_model generates the query each attempt (no memory) |
| `learned_query_template_bandit` | contextual bandit selects a query template per signature |

**Metrics.**
- Primary: `correct_per_tool_call`.
- Secondary: `evidence_gain_per_call`, `primary_or_authoritative_source_rate`,
  `query_miss` regime rate, `correct_per_dollar`.
- Especially track `query-fragile` and `evidence-sparse` signatures.

---

## Study 3 — Level 2 verification / stopping policy

**Goal.** Add **verification** and **stopping** seams: learn when to verify a
`candidate_answer` and when to stop searching, trading tool calls for correctness.

**Setup.**
- Build on Studies 1–2. Add `verification.completed` and `stop_decision.created` seams
  driven by a learned policy keyed by signature and current evidence state.

**Conditions.**
| Condition | Verify/stop behavior |
| --- | --- |
| `always_full_budget` | always exhaust the budget cap, no early stop |
| `stop_after_first_candidate` | stop as soon as a `candidate_answer` is produced |
| `learned_stop_verify` | contextual-bandit stopping + verification decisions |

**Metrics.**
- Primary: `correct_per_tool_call`.
- Secondary: `over_search_rate`, `false_stop_rate`, `verification_miss` regime rate,
  `stop_too_early`/`stop_too_late` rates, `support_answer_mismatch` rate,
  abstention precision/recall.

---

## Study 4 — Reward ablation

**Goal.** Determine which reward shaping best drives the epistemic policy, comparing pure
correctness against cost- and evidence-aware rewards.

**Setup.**
- Fix the policy architecture (best from Studies 1–3). Vary only the reward function used
  in cold-path attribution (`evidence_reward` / `tool_reward` / `trace_reward` weights).

**Conditions.**
| Condition | Reward signal |
| --- | --- |
| `correctness_only` | reward = grade correctness |
| `correctness_minus_cost` | correctness penalized by tool-call / dollar cost |
| `correctness_plus_evidence` | correctness + evidence-quality bonus |
| `full_epistemic_reward` | correctness + cost + evidence quality + source authority/freshness |

**Metrics.**
- Primary: `correct_per_tool_call`.
- Secondary: `correct_per_dollar`, `evidence_gain_per_call`,
  `primary_or_authoritative_source_rate`, `stale_source_error_rate`, `over_search_rate`.
- Report whether richer rewards improve generalization on `CONFIRM` or merely overfit
  `OPTIMIZE`.

---

## Study 5 — Regimes-style policy improvement

**Goal.** Close the loop: detect dominant failure regimes on `OPTIMIZE`, propose targeted
changes to bandit/reward/query/stop parameters, and **promote only if held-out `CONFIRM`
improves**.

**Setup.**
- Run the best configuration from Studies 1–4 on `OPTIMIZE`.
- `regimes/` detects dominant `regime_label`s (`regime.detected`) and emits
  `policy_update.proposed`.
- Apply updates under `OPTIMIZE` (`optimize_allows_exploration=true`); evaluate a frozen
  snapshot on `CONFIRM` (`confirm_uses_frozen_snapshot=true`,
  `confirm_updates_memory=false`).

**Conditions.**
| Condition | Update behavior |
| --- | --- |
| `baseline_frozen` | no regimes update; current best frozen snapshot |
| `regimes_proposed_applied` | regimes updates applied on OPTIMIZE only |
| `regimes_promoted` | only updates that pass the CONFIRM gate (`promotion.accepted`) are kept |

**Promotion rule.** A `policy_update` is promoted (`promotion.accepted`) iff the frozen
post-update snapshot improves `correct_per_tool_call` on `CONFIRM` by a margin exceeding
the bootstrap CI; otherwise `promotion.rejected`.

**Metrics.**
- Primary: `correct_per_tool_call` on `CONFIRM`, per budget cap.
- Secondary: reduction in the targeted `regime_label` rate, McNemar test on accuracy flips,
  budget-curve comparison `regimes_promoted` vs `baseline_frozen`.
- Report promotion accept/reject counts and which regimes drove accepted updates.
