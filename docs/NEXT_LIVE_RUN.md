# Next live run — step-by-step playbook

A precise plan for the **first live run** once API keys are available. Until then
nothing here is executed, and no live call is made by the test suite.

Run `python scripts/validate_live_readiness.py --strict` first; fix every `✗`
before proceeding.

## 0. Prerequisites

- A real dataset (BrowseComp CSV or LiveBrowseComp via Hugging Face / a local
  JSONL cache).
- At least `OPENAI_API_KEY` (default answerer `gpt-5.5` + hosted baseline
  `openai_web_search`). Optionally one independent search key for tool diversity.

## 1. Install

```bash
pip install -e '.[activegraph,openai,datasets]'
python -m pytest -q                 # still must pass with no keys
```

## 2. Environment variables

Export only what you have. None are required by tests; they are read at runtime.

```bash
export OPENAI_API_KEY=...           # answerer + openai_web_search baseline
export BRAVE_SEARCH_API_KEY=...     # optional independent adapter
export TAVILY_API_KEY=...           # optional
export EXA_API_KEY=...              # optional
export SERPER_API_KEY=...           # optional
```

## 3. Dataset setup

**BrowseComp (secondary, easiest to grade):**
```bash
# Download the obfuscated CSV from openai/simple-evals, then:
#   config/default.yaml -> dataset.browsecomp_csv: /path/to/browse_comp_test_set.csv
python -c "from regimes_probe.datasets.browsecomp import BrowseCompAdapter as A; \
           a=A('/path/to/browse_comp_test_set.csv'); print(len(a.load()), a.version())"
```

**LiveBrowseComp (primary, recent facts) — ⛔ BLOCKED on obfuscation.**
The HF release `Forival/LiveBrowseComp` ships `problem`/`answer` **encoded**, not
plaintext. The adapter **fails closed** and will NOT pass encrypted text through
as a question, so a live run cannot spend on encrypted blobs. **This benchmark
cannot be executed until the official decode/plaintext path is resolved.** You
will hit a clear `DatasetUnavailable` / `REFUSING` message until you provide one
of:

```bash
# (1) a PLAINTEXT export as local JSONL (fields: question, answer, released_at):
python scripts/run_live.py --dataset livebrowsecomp \
    --dataset-path data/livebrowsecomp_plaintext.jsonl --optimize 10 --confirm 20 --budgets 1,3
#     (dry-run first; add --execute to actually call providers)

# (2) the OFFICIAL canary/decode from the dataset authors, then configure it:
python -c "from regimes_probe.datasets.livebrowsecomp import LiveBrowseCompAdapter as A; \
           print(len(A(local_jsonl='data/lbc.jsonl', canary='<OFFICIAL_CANARY>').load()))"
#     the adapter validates that the decode yields plausible plaintext, else fails closed.
```

A raw HF dump (`problem`/`answer`, no `question`, no canary) is **refused** — by
design. Confirm the field names and the official decode scheme against the
dataset card/paper before configuring `canary=` (the obfuscation could not be
verified from this environment; network was restricted).

## 4. Config to edit

Copy `config/tools.example.yaml` → `config/tools.yaml` and enable the baseline
plus **≥2 independent** search adapters. In `config/default.yaml`:

- `dataset.default: BrowseComp` (or LiveBrowseComp) and the dataset path.
- `live.answer_model: gpt-5.5` (or `gpt-5.4-mini` to economize).
- `live.embedder: openai_embedder` (or keep `hash_embedder`).
- `split.mode: time` for LiveBrowseComp (time-disjoint CONFIRM).
- Keep `memory.confirm_uses_frozen_snapshot: true`,
  `memory.confirm_updates_memory: false`.

## 5. First tiny live run — **budgets [1, 3], OPTIMIZE=10, CONFIRM=20**

The exact escalating commands (A–E) live in **[`docs/LIVE_LADDER.md`](./LIVE_LADDER.md)**
— each is dry-run unless you add `--execute`. The executor is `scripts/run_live.py`.

Start small to validate plumbing and **cost**, not to make a claim. These are the
exact parameters for the first run; do not scale up until it works.

```bash
# Preflight (NO provider calls) — confirms split/conditions/eligibility/cost and
# writes results/{run_id}/run_manifest.json. Inspect the cost estimate first.
python scripts/preflight_first_live_run.py \
    --dataset livebrowsecomp --optimize 10 --confirm 20 --budgets 1 3 --strict
python scripts/estimate_live_cost.py --optimize 10 --confirm 20 --budgets 1 3
```

Then the live run executes this sequence (live wiring — passing live providers +
a live answerer into the same harness the synthetic path uses — is the one
remaining integration step):

```bash
# 1) closed_book baseline (no tools)  — bounds intrinsic knowledge — REQUIRED
# 2) no_memory_search baseline: openai_web_search, no memory
# 3) experience on the 10 OPTIMIZE items (exploration on)
# 4) freeze snapshot
# 5) policy_memory on the 20 CONFIRM items (frozen, exploit)
# 6) random_memory control on CONFIRM
```

Keep budgets at `[1, 3]` for the first run to cap spend (the full `[1, 3, 5, 10]`
curve comes later, after the tiny run works). Log every prompt, tool call, URL,
page hash, timestamp, model id, and tool config (already emitted to the event
log). A 10/20 run is a **plumbing + cost check, not a measurement** (§8).

## 6. Expected artifacts

Same shape as the synthetic demo, under `results/{run_id}/`:
`report.json`, `summary.md`, `budget_curve.csv`, `per_question.csv`,
`tool_rewards.csv`, `query_rewards.csv`, `stop_verify_rewards.csv`,
`memory_snapshot.json`, `policy_updates.json`, `replay_check.md`.

## 7. How to inspect whether results are meaningful

- **Replay check passes** (`replay_check.md`): graph == projection of the log.
- **Leakage**: grep the committed `memory_snapshot.json` for any gold answer
  string — there must be none. Re-run `assert_no_answer_leakage` on real traces.
- **Controls ordering**: `policy_memory` should beat `random_memory`, and
  `no_search`/`closed_book` should be clearly worse than search conditions. If
  `random_memory` ≈ `policy_memory`, the signal is suspect (see Methodology
  risks).
- **Held-out only**: read `correct_per_tool_call` on **CONFIRM**. Ignore OPTIMIZE
  for headlines.
- **Significance**: McNemar p and the bootstrap CI for `correct_per_tool_call`
  must exclude the baseline, not just have a higher point estimate.
- **Budget curve**: improvement should hold across budgets, not at one cherry-
  picked cap.

## 8. What NOT to claim from the first run

- Do **not** report a "BrowseComp/LiveBrowseComp score" from 20 CONFIRM items —
  it is a plumbing/cost check, not a measurement.
- Do **not** claim improvement from OPTIMIZE numbers.
- Do **not** claim improvement without the closed-book/no-search baselines.
- Do **not** compare against published leaderboard numbers; the answerer, tools,
  prompts, and grading differ.

## 9. Scaling to a real run

- Increase to the full split with a fixed seed; keep OPTIMIZE/CONFIRM disjoint
  (time-disjoint for LiveBrowseComp).
- Pin exact `model` ids and tool configs; record them in `report.json` meta.
- Run multiple seeds; report mean ± bootstrap CI on **CONFIRM** only.
- Keep all conditions identical except the frozen policy memory (the one
  independent variable).
- Only then, if CONFIRM improves beyond the CI and survives the controls, report
  it — grounded in the committed `results/{run_id}/` artifacts, with limitations.
