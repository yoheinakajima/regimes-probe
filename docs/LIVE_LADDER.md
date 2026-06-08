# Live run ladder

The exact, escalating commands for the first provider-calling runs. **Every
command is dry-run (no spend) unless you add `--execute`.** `run_live.py` builds
live providers + a live OpenAI Responses answerer into the same provider-agnostic
harness the no-key path uses; every live call flows through a recording cache so
reruns/replay never re-spend.

Conventions:
- Run dir: `results/live/<run_id>/` (`run_id` is auto-derived or `--run-id`).
- Default tools: `openai_web_search` + `page_fetch` → only `OPENAI_API_KEY` needed.
- Estimated calls below are **worst case** (every attempt uses its full budget);
  the cache reduces real spend on reruns.
- After any run: generate the audit ledger and conservative claims.

> Estimates assume the full pipeline structure (experience passes = 4). Verify
> with `python scripts/estimate_live_cost.py --optimize N --confirm M --budgets ...`.

---

## A. Preflight only — **NO provider calls**
```bash
python scripts/validate_live_readiness.py --strict
python scripts/preflight_first_live_run.py \
    --dataset livebrowsecomp --optimize 10 --confirm 20 --budgets 1 3 --strict
```
- Calls providers: **no.** Run dir: `results/pre-<hash8>/` (manifest only).
- Calls: 0. Env: none required to run (`--strict` checks `OPENAI_API_KEY` presence).

## B. Tiny plumbing run — closed_book + no_memory_search, budget 1, 5/10
```bash
# dry-run (no spend):
python scripts/run_live.py --dataset livebrowsecomp --dataset-path PATH \
    --optimize 5 --confirm 10 --budgets 1 \
    --conditions closed_book,no_memory_search
# execute (spends): append --execute and a cache
python scripts/run_live.py --dataset livebrowsecomp --dataset-path PATH \
    --optimize 5 --confirm 10 --budgets 1 \
    --conditions closed_book,no_memory_search \
    --recording-cache results/live/cache.json --execute
```
- Calls providers: **yes (only with `--execute`).** Run dir: `results/live/<run_id>/`.
- Est. calls: **~20 answerer + ≤10 web_search** (no experience, no memory).
- Env: `OPENAI_API_KEY`.

## C. Tiny full run — all four conditions, budgets 1,3, 10/20
```bash
python scripts/run_live.py --dataset livebrowsecomp --dataset-path PATH \
    --optimize 10 --confirm 20 --budgets 1,3 \
    --recording-cache results/live/cache.json --execute
```
- Calls providers: **yes (with `--execute`).** Run dir: `results/live/<run_id>/`.
- Est. calls: **~180 answerer + ≤440 tool** (incl. experience 10×4).
- Env: `OPENAI_API_KEY` (+ optional Brave/Tavily/Exa/Serper if you add `--tools`).

## D. Small credible run — all four, budgets 1,3,5, 25/50
```bash
python scripts/run_live.py --dataset livebrowsecomp --dataset-path PATH \
    --optimize 25 --confirm 50 --budgets 1,3,5 \
    --recording-cache results/live/cache.json --execute
```
- Calls providers: **yes (with `--execute`).** Run dir: `results/live/<run_id>/`.
- Est. calls: **~600 answerer + ≤1,850 tool**. Env: `OPENAI_API_KEY` (+ optional).

## E. First serious run — all four, budgets 1,3,5, 50/100
```bash
python scripts/run_live.py --dataset livebrowsecomp --dataset-path PATH \
    --optimize 50 --confirm 100 --budgets 1,3,5 \
    --recording-cache results/live/cache.json --execute
```
- Calls providers: **yes (with `--execute`).** Run dir: `results/live/<run_id>/`.
- Est. calls: **~1,200 answerer + ≤3,700 tool** (real money). Env: `OPENAI_API_KEY` (+ optional).

---

## After any run — report / hashes / claims (no provider calls)
The run writes the report itself; then:
```bash
python scripts/hash_artifacts.py   results/live/<run_id>
python scripts/generate_claims.py  results/live/<run_id>/report.json   # refuses perf claims unless headline_eligible
python scripts/inspect_memory_snapshot.py results/live/<run_id>/memory_snapshot.json
python scripts/compare_runs.py results/live/<run_id>/report.json results/live/<other>/report.json
```

## Resume / no-respend
- **Replay (no spend):** add `--cache-mode replay` to reuse a populated cache;
  a cache miss raises rather than calling a provider.
- **Resume policy memory:** `--resume-from-snapshot results/live/<run>/memory_snapshot.json`
  skips the (expensive) experience phase and reuses a frozen snapshot.
- **Compare runs:** `scripts/compare_runs.py` (budget curves, deltas, McNemar).

## Safety recap
- Dry-run is the default; **`--execute` is the only way to spend.**
- Un-armed providers (dry-run) raise instead of calling.
- Missing keys → execute refuses before any call.
- Secrets are env-only; the cache and manifest redact key-like fields.
- A synthetic/placeholder dataset is always `headline_eligible=false`. Until a run
  clears `docs/FIRST_REAL_RESULT_CRITERIA.md`, **no benchmark result is claimed.**
