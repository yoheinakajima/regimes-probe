# Status — grounded claim ledger

Every claim below is tagged with its supporting artifact. Claims without a
committed artifact are listed as *not yet verified* or *to avoid*. This file is
the honesty contract for the project (mirrors the `regimes` discipline).

Last updated for: v0 (synthetic harness, Study 0 + Study 1).

**One-line summary:** the scaffold demonstrates the intended mechanism on a
synthetic fixture; **no BrowseComp/LiveBrowseComp performance is claimed.** The
skeptic's companion to this ledger is `docs/METHODOLOGY_RISKS.md`; the path to a
real result is `docs/REAL_BENCHMARK_READINESS.md` + `docs/NEXT_LIVE_RUN.md`.

## ✅ Verified claims (with artifacts)

| claim | evidence |
| --- | --- |
| The harness runs with no API keys and no network. | `python -m pytest -q` → 50 passing; `scripts/*` run offline; `scripts/run_synthetic_full.py` runs the whole pipeline offline. |
| The **real-benchmark adapter path** works on real-data-shaped inputs (no keys). | `tests/test_real_data_shape.py` (BrowseComp decode + LiveBrowseComp JSONL + split + report + leakage + replay on `fixtures/real_shaped/` placeholders). |
| The graph is a deterministic projection of the event log (replay passes). | `results/demo/replay_check.md` (`projection_matches: true`); `tests/test_fake_tool_replay.py`. |
| Fake tool calls are replayable; replay detects divergence. | `tests/test_fake_tool_replay.py::test_recording_then_replay_reproduces_responses`, `::test_replay_divergence_is_detected`. |
| The project also runs with the standard library + PyYAML only (no `activegraph`). | `EventLog` fallback in `activegraph_pack/behaviors.py`; verified by blocking the import. |
| OPTIMIZE/CONFIRM splits are deterministic and disjoint. | `tests/test_split.py`. |
| The default router uses no LLM and is deterministic given a snapshot. | `tests/test_router_determinism.py`; `policy/router.py`. |
| Budget caps are strictly binding across modes. | `tests/test_budget_caps.py` (budgets 1/3/5/10). |
| Policy memory stores no final answer text; the frozen snapshot is answer-free. | `tests/test_policy_memory_no_answer_leakage.py`; `assert_no_answer_leakage`. |
| On the synthetic fixture, frozen policy memory improves `correct_per_tool_call` vs no-memory at every budget. | `results/demo/budget_curve.csv`, `results/demo/summary.md` (e.g. budget 1: 0.000 → 0.792). |
| The improvement is statistically detectable (paired). | `results/demo/report.json` → McNemar at budget 3: 13 wrong→correct flips, 0 reverse, p ≈ 0.0009; bootstrap CI for policy `correct_per_tool_call` excludes the baseline. |
| The regimes loop detects regimes, proposes bounded updates, gates them, and records a promotion decision. | `tests/test_regimes_gates.py`; `results/regimes-demo/policy_updates.json`. |
| Base model weights are unchanged by construction. | No training/fine-tuning code exists; the only learned state is the bandit/memory. |

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
