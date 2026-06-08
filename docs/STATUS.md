# Status — grounded claim ledger

Every claim below is tagged with its supporting artifact. Claims without a
committed artifact are listed as *not yet verified* or *to avoid*. This file is
the honesty contract for the project (mirrors the `regimes` discipline).

Last updated for: v0 (synthetic harness, Study 0 + Study 1).

**Readiness: LIVE EXECUTOR WIRED BUT NOT EXECUTED; NOT BENCHMARK-CLAIMED.** The
no-key scaffold is frozen and auditable, **and** the live executor
(`scripts/run_live.py` + `src/regimes_probe/live/`) is now wired into the same
harness — safe by default (dry-run; refuses to spend without `--execute`). **No
provider-calling run has been performed; the first one is still pending.** The
ladder is `docs/LIVE_LADDER.md` (A–F); start with the tiny 10/20, budgets [1, 3].
**Defaults are cheap-first** (`gpt-5.4-mini`, provider-diverse routing); OpenAI
hosted `web_search` + `gpt-5.5` is an opt-in strong baseline (rung F), not the
default experiment. Until a run clears `docs/FIRST_REAL_RESULT_CRITERIA.md` with
`headline_eligible = true`, **no benchmark performance is claimed.**

**One-line summary:** the scaffold demonstrates the intended mechanism on a
synthetic fixture; **no BrowseComp/LiveBrowseComp performance is claimed.** The
skeptic's companion to this ledger is `docs/METHODOLOGY_RISKS.md`; the path to a
real result is `docs/REAL_BENCHMARK_READINESS.md` + `docs/NEXT_LIVE_RUN.md`.

**Three distinct things — do not conflate them:**

1. **Synthetic-harness result** (`results/demo/`, Study 0/1) — numbers on an
   *engineered* fixture. Proves the mechanism + plumbing learn the choices the
   fixture rewards. `headline_eligible = false`. **Not a benchmark.**
2. **Real-data-shaped smoke test** (`tests/test_real_data_shape.py`,
   `fixtures/real_shaped/`) — runs the *real adapter path* (decode → split →
   report → leakage → replay) on *fictional placeholder* data. Proves the
   pipeline accepts real-benchmark *structure*. **No performance meaning at all.**
3. **Future real benchmark result** (LiveBrowseComp / BrowseComp) — not yet run.
   Requires keys + data and must clear `docs/FIRST_REAL_RESULT_CRITERIA.md` with
   `headline_eligible = true`. **The only thing that could be a benchmark claim.**

**⛔ Active blocker — LiveBrowseComp obfuscation.** The HF dataset
`Forival/LiveBrowseComp` has been *downloaded* but its `problem`/`answer` fields
are **encoded, not plaintext**. The adapter now **fails closed** (refuses to pass
encrypted text as a question; `scripts/run_live.py` refuses with no placeholder
fallback when a real path is supplied), so **LiveBrowseComp cannot be executed
until the official decode/plaintext path is resolved** — provide a plaintext
export or the official canary/decode utility (validated). The decode scheme could
not be verified from this environment (network restricted). See
`docs/BENCHMARK_TARGETS.md` and `docs/NEXT_LIVE_RUN.md`.

## ✅ Verified claims (with artifacts)

| claim | evidence |
| --- | --- |
| The harness runs with no API keys and no network. | `python -m pytest -q` → 68 passing; `scripts/*` run offline; `scripts/run_synthetic_full.py` runs the whole pipeline offline. |
| The **real-benchmark adapter path** works on real-data-shaped inputs (no keys). | `tests/test_real_data_shape.py` (BrowseComp decode + LiveBrowseComp JSONL + split + report + leakage + replay on `fixtures/real_shaped/` placeholders). |
| **Four baselines/controls** are distinguished: `closed_book`, `no_memory_search`, `random_memory`, `policy_memory`. | `scripts/run_synthetic_full.py` output + `results/demo/{report.json,budget_curve.csv}`; `tests/test_baselines_and_eligibility.py`. |
| The **closed-book baseline makes no tool calls** and answers only intrinsic knowledge. | `tests/test_baselines_and_eligibility.py::test_closed_book_makes_no_tool_calls`. |
| A **same-conditions validator** enforces baseline/policy differ only in memory access. | `eval/conditions.py`; `tests/test_baselines_and_eligibility.py`. |
| Reports carry **`headline_eligible`** with reasons; synthetic runs are correctly **ineligible**. | `eval/eligibility.py`; `results/demo/report.json` (`headline_eligible: false`, reason = synthetic); `tests/test_baselines_and_eligibility.py`. |
| **Ablation scaffolding** runs no-key (routing_only … online_learning_confirm). | `scripts/run_ablations.py`, `config/default.yaml` `ablations:`; reports under `results/ablations/`. |
| The synthetic fixture exercises a **variety of failure regimes** (not cartoonishly perfect). | `tests/test_failure_regimes.py`; `dominant_regimes` on a baseline shows ≥4 regimes incl. route_miss + verification/stale/contradiction. |
| Policy memory snapshots are **inspectable and answer-free**. | `scripts/inspect_memory_snapshot.py` (leakage scan PASS). |
| Two runs can be **compared** (budget curves, deltas, McNemar). | `scripts/compare_runs.py`. |
| A **first-live-run preflight** validates everything (split, conditions, eligibility, budgets, env-var names, cost) and writes a dry-run manifest **without calling any provider**. | `scripts/preflight_first_live_run.py`; 0 blockers on the synthetic default. |
| **Call/cost is estimable** before any spend; dollar cost stays "unknown" unless prices are configured (no vendor prices hard-coded). | `eval/cost.py`, `scripts/estimate_live_cost.py`; `tests/test_live_readiness_scaffold.py`. |
| A **run manifest** records git/dataset/split/models/prompts/tools/budgets/memory/eligibility/cost with **no secrets** (env-var names only). | `eval/manifest.py`; `results/demo/run_manifest.json`; `tests/test_live_readiness_scaffold.py`. |
| **Prompts are version+hash pinned** and folded into the same-conditions fingerprint. | `agent/prompts.py`; `ConditionSpec.answer_prompt_version`; `tests/test_live_readiness_scaffold.py`. |
| Run artifacts have a **hash ledger**; a **conservative claim generator** REFUSES performance claims when `headline_eligible=false`. | `scripts/hash_artifacts.py` (`results/demo/artifact_hashes.json`), `scripts/generate_claims.py` (`results/demo/claims_candidates.md`). |
| Docs lint + no-overclaim check passes; no-key Make targets run. | `scripts/docs_check.py` (25 docs), `Makefile` (`make check`). |
| The **live executor is wired** into the provider-agnostic harness, **safe by default** (dry-run; `--execute` required to spend). | `scripts/run_live.py`, `src/regimes_probe/live/{providers,cache,runner}.py`; `tests/test_run_live_safety.py`. |
| Live calls are **cache/replay-guarded** (no re-spend) and **un-armed in dry-run** (refuse to call). | `RecordingCache`, `CachedProvider`/`NotArmed`; `tests/test_run_live_safety.py`. |
| The execute pipeline passes **same-conditions + headline-eligibility** plumbing (verified with mock providers, no network). | `tests/test_run_live_safety.py::test_run_live_pipeline_with_mocks_passes_plumbing`. |
| Secrets never enter cache/manifest/artifacts (env-var names only; key-like fields redacted). | `live/cache.sanitize`; `tests/test_run_live_safety.py`. |
| LiveBrowseComp **fails closed** on obfuscated rows — encrypted `problem` is never passed through as a question; `run_live` refuses (no placeholder fallback when a real path is given). | `datasets/livebrowsecomp.py`; `scripts/run_live.py`; `tests/test_livebrowsecomp_failclosed.py` (9 tests). |
| BrowseComp decode matches openai/simple-evals **byte-for-byte** (`derive_key` = repeated `sha256(canary)`). | `datasets/browsecomp.py`; `tests/test_browsecomp_decrypt.py`. |
| Live config is **cheap-first**: answerer/web_search default `gpt-5.4-mini`; OpenAI hosted `web_search` is **opt-in** (modes cheap/diverse/openai-hosted); no silent fallback to expensive hosted search; gpt-5.5+hosted warns. | `tools/openai_web_search.py` (no gpt-5.5 default), `live/settings.py`, `scripts/run_live.py`; `tests/test_live_provider_modes.py` (12 tests). |
| Extended provider arms (Monid agentic-discovery, Firecrawl search/scrape/interact, Wokelo specialized-research) are **optional, never default, and safety-gated**; stateful/paid and browser-like tools are off without explicit flags; browser-use is **deferred**; Wokelo **fails closed** without base URL/path; per-arm metadata (family/cost/stateful/safe_default) is recorded in the manifest. | `tools/{metadata,monid,firecrawl,wokelo}.py`, `live/settings.py`; `tests/test_extended_providers.py` (15 tests); `docs/TOOL_ABSTRACTIONS.md`. |
| Reports separate **`structurally_valid`** from **`headline_eligible_memory_claim`**: a run without `policy_memory` (e.g. a plumbing run) is never a memory headline, the "differ only in memory access" phrase only appears when `policy_memory` ran, and claims generation refuses memory-performance claims when `policy_memory` is missing. Headline also requires all four conditions on a real dataset with CONFIRM ≥ `headline.min_confirm_size` (default 20). | `eval/eligibility.py`, `eval/report.py`, `scripts/generate_claims.py`; `tests/test_eligibility_conditions.py`, `tests/test_baselines_and_eligibility.py`. |
| A **provider/API error during a tool call does not crash the run**: it becomes a recorded failed observation (sanitized, no secrets), the bandit is penalized for that arm, per-tool failure counts are reported, and a high failure rate (> `headline.max_provider_failure_rate`, default 0.2) or an entirely-failed required condition blocks the headline while keeping the run structurally valid. | `tools/base.safe_search`, `live/providers.py`, `activegraph_pack/tools.py`, `eval/{reward,metrics,eligibility,report}.py`; `tests/test_tool_failures.py` (10 tests). |
| The graph is a deterministic projection of the event log (replay passes). | `results/demo/replay_check.md` (`projection_matches: true`); `tests/test_fake_tool_replay.py`. |
| Fake tool calls are replayable; replay detects divergence. | `tests/test_fake_tool_replay.py::test_recording_then_replay_reproduces_responses`, `::test_replay_divergence_is_detected`. |
| The project also runs with the standard library + PyYAML only (no `activegraph`). | `EventLog` fallback in `activegraph_pack/behaviors.py`; verified by blocking the import. |
| OPTIMIZE/CONFIRM splits are deterministic and disjoint. | `tests/test_split.py`. |
| The default router uses no LLM and is deterministic given a snapshot. | `tests/test_router_determinism.py`; `policy/router.py`. |
| Budget caps are strictly binding across modes. | `tests/test_budget_caps.py` (budgets 1/3/5/10). |
| Policy memory stores no final answer text; the frozen snapshot is answer-free. | `tests/test_policy_memory_no_answer_leakage.py`; `assert_no_answer_leakage`. |
| On the synthetic fixture, frozen policy memory improves `correct_per_tool_call` vs `no_memory_search` at every budget, improves accuracy at every budget, and far exceeds the `random_memory` control at budget 1. | `results/demo/budget_curve.csv` (budget 1: no_memory_search 0.000, random_memory 0.286, policy_memory 0.750; budget 3 accuracy: 0.214 / 0.500 / 0.857). |
| The improvement is statistically detectable (paired). | `results/demo/report.json` → McNemar at budget 3: 18 wrong→correct flips, 0 reverse, p ≈ 0.0001; bootstrap CI for policy `correct_per_tool_call` excludes the baseline. |
| The synthetic result is **not** perfect — residual failures remain so diagnostics are meaningful. | `results/demo/summary.md` (policy accuracy 0.857 at budget 3, not 1.0); `per_question.csv` regime labels. |
| On this fixture the learned stop policy does not always stop early, so at large budgets `policy_memory` trades calls for accuracy and `correct_per_tool_call` can trail `random_memory` (recorded honestly; thresholds were not tuned to flatter the number). | `results/demo/budget_curve.csv` (budget 5/10: policy_memory mean_tool_calls = budget; stop fires only when the verification score reaches `stopping.stop_threshold`). |
| The regimes loop detects regimes, proposes bounded updates, gates them, and records a promotion decision. | `tests/test_regimes_gates.py`; `results/regimes-demo/policy_updates.json`. |
| Base model weights are unchanged by construction. | No training/fine-tuning code exists; the only learned state is the bandit/memory. |
| Frozen-memory leakage is checked the same way the inspector checks it, and is layer-separated: only frozen policy memory gates the headline; the raw audit trace may contain gold by design. | `src/regimes_probe/eval/leakage.py` (`leakage_check_details`); `results/demo/report.json` → `leakage_check_details`; `tests/test_debug_artifacts.py::test_report_leakage_matches_inspector_for_clean_snapshot`, `::test_raw_archive_gold_does_not_fail_policy_memory_leakage`. |
| `report.json` carries non-empty per-condition metrics plus provider-failure accounting. | `results/demo/report.json` → `metrics`, `provider_failure_rate`, `tool_failures`; `tests/test_debug_artifacts.py::test_report_metrics_populated`, `::test_provider_failures_in_report_and_summary`. |
| Each run emits a bounded, secret-free `debug_questions.jsonl` and two offline triage scripts read it. | `results/demo/debug_questions.jsonl`; `scripts/debug_run_failures.py`, `scripts/summarize_provider_returns.py`; `tests/test_debug_artifacts.py::test_debug_jsonl_bounded_and_has_previews`, `::test_debug_scripts_run`. |
| Every run exports a compact, secret-free, answer-free typed graph projection with all 17 object + 14 relation types, provenance-linked to the event-log replay. | `results/demo/graph_projection.json`; `eval/projection.py`; `tests/test_graph_native.py::test_graph_projection_has_expected_object_and_relation_types`, `::test_projection_is_bounded_to_detail_sample`. |
| The two eligibility verdicts are one flat first-class `eligibility_verdict` object, mirrored in `report.json` and the projection and consumed by the claims generator. | `results/demo/report.json` → `eligibility_verdict`; `eval/eligibility.py:build_eligibility_verdict`; `tests/test_graph_native.py::test_eligibility_verdict_object_in_report_and_projection`, `::test_plumbing_run_structurally_valid_but_not_headline`, `::test_all_four_real_can_be_headline_eligible`, `::test_synthetic_cannot_be_headline_eligible`. |
| Failures are first-class `failure_regime` objects attached to attempts (provider failures become `provider_error`, not strings). | `eval/failure_regime.py`; `results/demo/report.json` → `failure_regime_summary`; `tests/test_graph_native.py::test_failure_regime_objects_attach_to_attempts`. |
| Policy fragments carry trace lineage (source trace/attempt ids, top arms) and stay answer-free. | `results/demo/memory_snapshot.json` fragments; `policy/consolidation.py`; `scripts/inspect_memory_snapshot.py`; `tests/test_graph_native.py::test_policy_fragments_link_to_source_traces_without_answers`. |
| Offline forked ablations rerun policy variants on cached outcomes and refuse to call providers; they are never headline-eligible. | `live/fork.py`, `scripts/fork_offline_ablation.py`, `docs/OFFLINE_FORK_ABLATIONS.md`; `tests/test_graph_native.py::test_offline_fork_refuses_live_calls`, `::test_offline_fork_missing_cache_refuses`, `::test_offline_fork_end_to_end_not_headline_no_spend`. |
| `page_fetch` is a follow-up (URL) tool, never a first-hop search arm: the router routes only over search-family tools, page_fetch runs only on a URL from evidence, and a non-URL call fails gracefully with `requires_url`. | `tools/metadata.py` (`FOLLOWUP_FAMILIES`, `is_first_hop`); `agent/search_loop.py` (first-hop/follow-up split); `tools/page_fetch.py`; `tests/test_page_fetch_routing.py` (6 tests incl. provider-failure-rate falls when page_fetch is excluded from first-hop). |
| Level 2 query decomposition turns a long clue-dense question into 3–6 targeted, length-capped clue queries; the bandit learns the query form (`tool × query_arm`) and the long full-question query is never the cold-start default. | `policy/query_decomposition.py`; `policy/query_policy.py` (`--enable-query-decomposition`); `tests/test_query_decomposition.py` (long→multiple short queries, length cap, quoted/proper-noun/date clues preserved, full-question not default, arms logged). |
| Benchmark-contaminated results (eval-host / question-mirroring pages) are detected, penalized in reward, and reported. | `eval/contamination.py`; `eval/reward.py` (`contamination_penalty`); `results/{run}/report.json` → `contamination` (count, rate, by provider/domain); `tests/test_query_decomposition.py::test_contamination_*`. |

## Commands that pass with no keys/network

```
python -m pytest -q                              # 68 passed
python scripts/run_synthetic_full.py --run-id demo
python -m pytest -q tests/test_real_data_shape.py
python scripts/run_ablations.py
python scripts/validate_live_readiness.py        # 0 failures (warnings only)
python scripts/preflight_first_live_run.py       # 0 blockers; writes dry-run manifest
python scripts/estimate_live_cost.py --optimize 10 --confirm 20
python scripts/inspect_memory_snapshot.py results/demo/memory_snapshot.json   # leakage PASS
python scripts/hash_artifacts.py results/demo
python scripts/generate_claims.py results/demo/report.json   # REFUSES perf claims
python scripts/docs_check.py                      # required docs present, no overclaim
python scripts/fork_offline_ablation.py results/live/<run_id> --out <id>-fork  # offline; replay-only, refuses to spend
python scripts/run_live.py --dataset real-shaped --optimize 5 --confirm 10 \
    --budgets 1 --conditions closed_book,no_memory_search   # DRY-RUN, no calls
make live-dry-run                                 # no provider calls
make check
```

**Live executor: wired, not executed.** `scripts/run_live.py` calls providers
ONLY with `--execute` (never run here). First provider-calling run is pending;
see `docs/LIVE_LADDER.md`.

## ◑ Partially verified claims

| claim | status |
| --- | --- |
| BrowseComp / LiveBrowseComp adapters load real data. | Decode/round-trip helpers tested (`datasets/browsecomp.py`); real-data load is implemented but not exercised in CI (no data committed). Missing data raises a clear `DatasetUnavailable`. |
| Live search/answer adapters work end-to-end. | Adapters are implemented and env-gated (`tools/*.py`, `policy/embeddings.OpenAIEmbedder`), but no live network run is recorded here. |
| Level 2 query/stop policies improve over fixed baselines. | Mechanisms implemented and toggleable (`query_mode`, `stop_mode`); the synthetic demo shows the learned policy stopping early (mean 1 call at budget 1). A dedicated Study 2/3 runner is future work. |

## ⬜ Not yet verified

- Any claim about a **real** LiveBrowseComp or BrowseComp score. None is made.
- Reward-ablation (Study 4) and Thompson-sampling comparisons as committed reports.
- Entity-disjoint splits on real data (time-disjoint mode is implemented).

## 🚫 Unsupported claims to avoid

- "The agent memorizes / recalls facts." — It does not; it learns procedural
  policy. Policy memory is answer-free.
- "Improves on BrowseComp/LiveBrowseComp." — Not demonstrated; only the
  synthetic harness is measured here.
- "Beats hosted web search." — No live comparison is recorded.
- "Improvement holds in-sample, therefore it is real." — OPTIMIZE is never the
  headline; only CONFIRM results are reported.
- "The regimes loop reliably finds promotions." — On this synthetic set the
  baseline policy is already strong, so the loop frequently (and correctly)
  **rejects** updates rather than promoting noise.

## Result artifacts index

- `results/demo/report.json` — per-condition metrics, significance, replay.
- `results/demo/summary.md` — human-readable headline + limitations.
- `results/demo/budget_curve.csv` — `correct_per_tool_call` by budget/condition.
- `results/demo/per_question.csv` — per-item correctness, calls, regime label.
- `results/demo/{tool,query,stop_verify}_rewards.csv` — learned reward tables.
- `results/demo/memory_snapshot.json` — the frozen, answer-free policy memory.
- `results/demo/replay_check.md` — determinism proof.
- `results/regimes-demo/policy_updates.json` — promotion/rejection record.
- `results/demo/debug_questions.jsonl` — bounded, secret-free per-question debug previews (tool sequence, providers, evidence, inferred failure seam, regime names); read by `scripts/debug_run_failures.py` and `scripts/summarize_provider_returns.py`.
- `results/demo/graph_projection.json` — standardized, compact, secret-free typed graph projection of the run (17 object + 14 relation types incl. `eligibility_verdict`, `failure_regime`, `policy_fragment` lineage); provenance-linked to the event-log replay. In the artifact hash ledger.
