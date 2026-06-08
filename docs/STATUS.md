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
| The graph is a deterministic projection of the event log (replay passes). | `results/demo/replay_check.md` (`projection_matches: true`); `tests/test_fake_tool_replay.py`. |
| Fake tool calls are replayable; replay detects divergence. | `tests/test_fake_tool_replay.py::test_recording_then_replay_reproduces_responses`, `::test_replay_divergence_is_detected`. |
| The project also runs with the standard library + PyYAML only (no `activegraph`). | `EventLog` fallback in `activegraph_pack/behaviors.py`; verified by blocking the import. |
| OPTIMIZE/CONFIRM splits are deterministic and disjoint. | `tests/test_split.py`. |
| The default router uses no LLM and is deterministic given a snapshot. | `tests/test_router_determinism.py`; `policy/router.py`. |
| Budget caps are strictly binding across modes. | `tests/test_budget_caps.py` (budgets 1/3/5/10). |
| Policy memory stores no final answer text; the frozen snapshot is answer-free. | `tests/test_policy_memory_no_answer_leakage.py`; `assert_no_answer_leakage`. |
| On the synthetic fixture, frozen policy memory improves `correct_per_tool_call` vs `no_memory_search` at every budget, and beats the `random_memory` control. | `results/demo/budget_curve.csv` (budget 1: no_memory_search 0.000, random_memory 0.286, policy_memory 0.750). |
| The improvement is statistically detectable (paired). | `results/demo/report.json` → McNemar at budget 3: 13 wrong→correct flips, 0 reverse, p ≈ 0.0009; bootstrap CI for policy `correct_per_tool_call` excludes the baseline. |
| The synthetic result is **not** perfect — residual failures remain so diagnostics are meaningful. | `results/demo/summary.md` (policy accuracy 0.75 at budget 3, not 1.0); `per_question.csv` regime labels. |
| The regimes loop detects regimes, proposes bounded updates, gates them, and records a promotion decision. | `tests/test_regimes_gates.py`; `results/regimes-demo/policy_updates.json`. |
| Base model weights are unchanged by construction. | No training/fine-tuning code exists; the only learned state is the bandit/memory. |

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
python scripts/docs_check.py                      # 25 docs, no overclaim
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
