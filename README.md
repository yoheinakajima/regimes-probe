# regimes-probe

**An ActiveGraph-native benchmark and improvement loop for learning epistemic
search *policy* from traces.**

`regimes-probe` tests one hypothesis:

> An otherwise identical web-search agent with ActiveGraph procedural memory and
> a contextual-bandit epistemic policy can improve **correct answers per tool
> call** on held-out browsing questions — **without changing base model weights**
> and **without storing benchmark answers**.

> **Current status (be precise):** the scaffold **demonstrates the intended
> mechanism on a synthetic fixture**. It has **not** been run on BrowseComp or
> LiveBrowseComp, and **no real benchmark performance is claimed**. See
> [`docs/STATUS.md`](docs/STATUS.md), [`docs/METHODOLOGY_RISKS.md`](docs/METHODOLOGY_RISKS.md),
> and [`docs/REAL_BENCHMARK_READINESS.md`](docs/REAL_BENCHMARK_READINESS.md).

It learns a *procedural* policy for **where to look, how to search, when to
verify, and when to stop** — not factual answers.

- **Level 1** — contextual-bandit **tool routing**: which search/fetch tool to
  use for a question.
- **Level 2** — contextual-bandit **epistemic policy**: query formulation,
  verification, and stopping.
- **Regimes layer** — diagnoses generic failure regimes and proposes *bounded*
  policy-parameter updates, promoted only if they hold on a held-out split
  (OPTIMIZE → CONFIRM), mirroring [`yoheinakajima/regimes`](https://github.com/yoheinakajima/regimes).

## What it is

- A reproducible harness where the **only** difference between conditions is a
  **frozen, answer-free policy memory**.
- ActiveGraph-native: the event log is the source of truth, the graph is a
  deterministic projection, behaviors are deterministic, all network goes
  through recorded tools, and every run is **replayable**.
- Provider-agnostic: search tools are adapters behind one interface, so the
  contextual bandit can learn tool-selection differences.

## What it is not

- It is **not** a factual-memory store. Policy memory holds traces, rewards,
  priors, and policy fragments — **never answer text** (`docs/LEAKAGE_CONTROLS.md`).
- It is **not** a fine-tuning project. Base model weights never change.
- It does **not** optimize answer/judge prompts or mutate arbitrary code in v0.

## How it relates to ActiveGraph

The benchmark is implemented as an ActiveGraph pack
(`src/regimes_probe/activegraph_pack/`). It uses the real
[`activegraph`](https://github.com/yoheinakajima/activegraph) runtime when
installed and a faithful standard-library fallback otherwise, so it runs with no
extra dependencies. See `docs/ACTIVEGRAPH_DESIGN.md`.

- ActiveGraph docs: https://docs.activegraph.ai/
- Behaviors: https://docs.activegraph.ai/concepts/behaviors/
- Replay: https://docs.activegraph.ai/concepts/replay/

## How it relates to `regimes`

[`yoheinakajima/regimes`](https://github.com/yoheinakajima/regimes) tested
failure-regime repair via OPTIMIZE/CONFIRM code/prompt transforms.
`regimes-probe` keeps that **discipline** (OPTIMIZE is not the headline; CONFIRM
is held out; all promotions recorded; claims grounded in committed artifacts) but
**broadens the fix space** to epistemic policy seams: route, query, verify, stop,
assemble, answer, consolidate memory.

## Why LiveBrowseComp / BrowseComp

- **Primary: LiveBrowseComp** — targets *recent* facts, reducing intrinsic-
  knowledge dependence. Paper https://arxiv.org/abs/2605.28721 · dataset
  https://huggingface.co/datasets/Forival/LiveBrowseComp
- **Secondary: BrowseComp** — public, short-answer, easy to grade. Paper
  https://arxiv.org/abs/2504.12516 · eval
  https://github.com/openai/simple-evals/blob/main/browsecomp_eval.py
- **Future: BrowseComp-Plus** — fixed-corpus controlled retrieval.
  https://arxiv.org/abs/2508.06600

See `docs/BENCHMARK_TARGETS.md`. Unit tests never require real benchmark data.

## Quickstart (no API keys, no network)

```bash
pip install -e .            # or: pip install -e '.[dev,activegraph]'
python -m pytest -q         # 60 tests, no keys/network

# The whole no-key pipeline in one command (the first demo):
python scripts/run_synthetic_full.py --run-id demo   # split -> baseline -> experience ->
                                                     # freeze -> CONFIRM -> controls ->
                                                     # report -> replay -> STATUS print

# ...or step by step:
python scripts/build_split.py    --run-id demo   # deterministic OPTIMIZE/CONFIRM split
python scripts/run_baseline.py   --budget 1      # no-memory baseline on CONFIRM
python scripts/run_experience.py --run-id demo   # experience phase on OPTIMIZE -> frozen snapshot
python scripts/run_confirm.py    --run-id demo --budget 1   # frozen policy memory on CONFIRM
python scripts/run_budget_curve.py               # budget curve: no_memory vs policy_memory
python scripts/make_report.py    --run-id demo   # full report + replay check -> results/demo/
python scripts/run_regimes_loop.py --run-id rl   # regimes improvement loop with OPTIMIZE/CONFIRM gating
python scripts/run_ablations.py                  # Level1/Level2/reward/online ablations
python scripts/inspect_memory_snapshot.py results/demo/memory_snapshot.json   # audit a snapshot
python scripts/compare_runs.py results/a/report.json results/b/report.json    # diff two runs

# Real-benchmark readiness (validates config only; calls NO providers):
python scripts/validate_live_readiness.py        # add --strict to gate on gaps
```

Representative **synthetic-fixture** result (committed under `results/demo/`) —
this validates the *mechanism and harness*, **not** any real benchmark. The
report's `headline_eligible` is **false** (dataset is synthetic), by design:

| budget | closed_book | no_memory_search | random_memory (control) | policy_memory |
| --- | --- | --- | --- | --- |
| 1 | 0.036 acc | 0.000 | 0.286 | **0.750** |
| 3 | 0.036 acc | 0.095 | 0.275 | **0.656** |

(`correct_per_tool_call`, except `closed_book` which makes 0 tool calls so only
its accuracy is shown.) Paired McNemar at budget 3 shows 13 answers flipping
wrong→correct with no reverse flips; `random_memory` lands well below
`policy_memory`; the replay check confirms the graph is a deterministic
projection of the log. **These are numbers on an engineered synthetic fixture
(Study 0/1)** — the policy is intentionally *not* perfect (residual failures feed
the regime diagnostics). They say nothing about BrowseComp or LiveBrowseComp.
Read `docs/STATUS.md` (claim ledger) and `docs/METHODOLOGY_RISKS.md` first.

## Toward a real run (no keys needed yet)

The path to a credible real result is documented, and the real-data *adapter
path* is already exercised offline on placeholder fixtures
(`fixtures/real_shaped/`, `tests/test_real_data_shape.py`):

- `docs/REAL_BENCHMARK_READINESS.md` — what's done vs. what's blocked on
  keys/data/decisions.
- `docs/NEXT_LIVE_RUN.md` — exact step-by-step for the first live run.
- `python scripts/validate_live_readiness.py` — config validation, no calls.

## Live runs

Set the relevant env var(s) and copy `config/tools.example.yaml` to
`config/tools.yaml`. Default live answerer is `gpt-5.5` (OpenAI Responses API),
default hosted-search baseline is `openai_web_search`; add independent adapters
(Brave/Tavily/Exa/Serper) for tool diversity. All credentials are optional and
read from the environment:

```
OPENAI_API_KEY, BRAVE_SEARCH_API_KEY, TAVILY_API_KEY, EXA_API_KEY, SERPER_API_KEY
```

OpenAI references: https://developers.openai.com/api/docs/models ·
https://developers.openai.com/api/docs/guides/tools-web-search

## Documentation

Start with `docs/ARCHITECTURE.md` and `docs/RESEARCH_PLAN.md`. The full set:
architecture, research plan, benchmark targets, model/tool choices, ActiveGraph
design, event schema, policy memory, contextual bandit, regimes loop, routing /
query / verification-and-stopping policies, grading and reward, evaluation
protocol, leakage controls, reporting, implementation plan, and `STATUS.md`.

For the path to a real benchmark: `docs/REAL_BENCHMARK_READINESS.md`,
`docs/NEXT_LIVE_RUN.md`, `docs/FIRST_REAL_RESULT_CRITERIA.md`, and
`docs/METHODOLOGY_RISKS.md` (read this one before believing any number).

## Repository layout

```
docs/                      precise design docs (read these first)
src/regimes_probe/
  activegraph_pack/        event log, behaviors, recorded tools, schema
  datasets/                synthetic + BrowseComp + LiveBrowseComp adapters
  tools/                   provider adapters (only place network I/O happens)
  policy/                  signatures, embeddings, memory, bandit, policy seams
  agent/                   planner, search loop, answerer, evidence
  regimes/                 detectors, action space, hypothesize, gates, runner
  eval/                    split, grader, reward, metrics, significance, report
scripts/                   runnable CLIs (no keys needed for the synthetic path)
tests/                     deterministic, no keys/network
fixtures/                  synthetic dataset + fake search corpus
config/                    default.yaml + tools.example.yaml
results/                   generated run artifacts
```

License: MIT.
