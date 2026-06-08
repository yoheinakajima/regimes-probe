# Implementation plan

Status of each phase in this v0, with the modules that satisfy it. Legend:
✅ done · ◑ partial · ⬜ not started.

## Phase 0 — docs and schemas ✅
- All `docs/*` exist; README references ActiveGraph, regimes, LiveBrowseComp,
  BrowseComp, and OpenAI search docs.
- Event/object/relation schema is explicit (`EVENT_SCHEMA.md`,
  `activegraph_pack/{events,objects,relations}.py`).

## Phase 1 — synthetic fixture ✅
- `fixtures/synthetic_browse.json` + `fixtures/fake_search_corpus.json`
  (generated deterministically by `datasets/synthetic.generate_synthetic_fixtures`).
- Deterministic fake tools (`tools/fake.py`); no keys, no network.
- End-to-end baseline + memory agent (`agent/`, `eval/harness.py`).
- Budget caps enforced (`tests/test_budget_caps.py`).

## Phase 2 — ActiveGraph event logging ✅
- All major actions emit events (`activegraph_pack/behaviors.py`).
- Replay check passes (`replay_check`, `tests/test_fake_tool_replay.py`).
- No direct network I/O in behaviors; the only I/O is the recorded tool
  invoker. Fake tool calls are replayable (`RecordingInvoker`/`ReplayInvoker`).

## Phase 3 — Level 1 contextual-bandit router ✅
- Router chooses tools from policy memory (`policy/router.py`); no LLM;
  deterministic given a snapshot (`tests/test_router_determinism.py`).
- UCB / epsilon-greedy / (optional) Thompson scoring + updates
  (`policy/contextual_bandit.py`, `tests/test_bandit_update.py`).

## Phase 4 — Level 1 evaluation ✅
- `no_memory` vs `policy_memory` on the synthetic fixture, budget curve, and
  `correct_per_tool_call` (`scripts/run_budget_curve.py`, `scripts/make_report.py`).

## Phase 5 — Level 2 query policy ✅
- Query-template arms + learned selection, enable/disable via `query_mode`
  (`policy/query_policy.py`); query rewards tracked (`stop_verify`/`query_rewards.csv`).

## Phase 6 — Level 2 verification/stopping ✅
- Stop/verify arms (`policy/stopping_policy.py`, `policy/verification_policy.py`),
  `stop_mode` toggle, false-stop / over-search metrics (`eval/metrics.py`).

## Phase 7 — Regimes-style improvement ✅
- Generic failure-regime detectors (`regimes/detectors.py`), bounded mutations
  (`regimes/action_space.py`, `regimes/hypothesize.py`), OPTIMIZE + CONFIRM gates
  (`regimes/gates.py`), promotion/rejection events (`regimes/runner.py`,
  `tests/test_regimes_gates.py`).

## Phase 8 — BrowseComp / LiveBrowseComp adapters ◑
- Adapters load available data; tests don't require data; dataset
  version/checksum stored; missing data raises a clear error
  (`datasets/browsecomp.py`, `datasets/livebrowsecomp.py`). Decode helpers are
  round-trip tested.
- A no-key **real-data-shaped smoke test** exercises the full adapter path
  (decode → split → report → leakage → replay) on placeholder fixtures
  (`fixtures/real_shaped/`, `tests/test_real_data_shape.py`,
  `scripts/build_real_shaped_fixtures.py`).
- Loading the *actual* benchmark data and confirming the real field/canary
  schema remains future work (needs dataset access). See
  `docs/REAL_BENCHMARK_READINESS.md`.

## Phase 9 — live tool adapters ◑
- OpenAI Responses `web_search` (+ low/unlimited context), page fetch, and
  independent adapters (Brave/Tavily/Exa/Serper, generic/news/official) all
  present and env-gated (`tools/*.py`). Network tests are opt-in only (none in
  the suite). End-to-end live wiring in scripts is future work.

## Phase 10 — reporting and statistical analysis ✅
- `report.json`, `summary.md`, all CSVs, `replay_check.md` generated
  (`eval/report.py`). McNemar + bootstrap CI implemented
  (`eval/significance.py`). `STATUS.md` maintained.

## Not built in v0 (by design)
- Arbitrary prompt/code mutation, answer-prompt or judge-prompt optimization,
  storing answers in memory. The regimes mutation space is restricted to safe
  numeric (and a couple of medium-risk threshold) parameters.

## v0.2 — credibility scaffolding (no keys) ✅
- `closed_book` + `no_memory_search` baselines alongside `random_memory` and
  `policy_memory` (`agent/answerer.ClosedBookAnswerer`, `_common.full_pipeline`).
- Same-conditions validator (`eval/conditions.py`) and headline-eligibility
  (`eval/eligibility.py`), surfaced in `report.json` / `summary.md`.
- Ablation scaffolding (`scripts/run_ablations.py`, `config/default.yaml`).
- `scripts/inspect_memory_snapshot.py`, `scripts/compare_runs.py`,
  `scripts/run_synthetic_full.py`, `scripts/validate_live_readiness.py`.
- Real-data-shaped smoke test + placeholder fixtures (`fixtures/real_shaped/`).
- Synthetic fixture enriched with failure-regime examples (residual failures by
  design) so regime diagnostics are meaningful.
- Docs: `REAL_BENCHMARK_READINESS`, `NEXT_LIVE_RUN`, `METHODOLOGY_RISKS`,
  `FIRST_REAL_RESULT_CRITERIA`.

## Suggested next steps
1. Wire `scripts/` to live providers behind `config/tools.yaml` (Phase 9 finish).
2. Add a small recorded LiveBrowseComp/BrowseComp subset fixture for an offline
   real-data smoke test (Phase 8 finish).
3. Layer-3 natural-language "lessons" generated from policy fragments.
4. Thompson sampling study and reward-ablation study runners (Studies 2–4 as
   dedicated scripts).
