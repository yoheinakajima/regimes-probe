# Live run ladder

The exact, escalating commands for the first provider-calling runs. **Every
command is dry-run (no spend) unless you add `--execute`.** `run_live.py` builds
live providers + a live OpenAI Responses answerer into the same provider-agnostic
harness the no-key path uses; every live call flows through a recording cache so
reruns/replay never re-spend.

**Cheap-first.** The default is `--search-provider-mode cheap` with
`gpt-5.4-mini`. OpenAI hosted `web_search` is **opt-in** (`--search-provider-mode
openai-hosted` or `--tools …,openai_web_search`); `gpt-5.5` + hosted web_search is
a **later strong/expensive baseline only** (rung F), not the default experiment.

Conventions:
- Run dir: `results/live/<run_id>/` (`run_id` is auto-derived or `--run-id`).
- Default models: answerer + web_search tool both `gpt-5.4-mini`, context `low`.
- Default tools (cheap mode): a low-cost external **search** adapter if its key is
  present (Serper/Brave/Tavily/Exa) as the first-hop bandit arm(s), plus
  `page_fetch` as a **follow-up** tool (URL-only; not a bandit arm). With **no**
  external key, the run explains which env var to set — it does **not** fall back
  to OpenAI web_search.
- `diverse` mode (the main experiment) makes each **search** provider a separate
  first-hop bandit arm; follow-up tools (`page_fetch`/scrape) are not arms.
- Dry-run prints `first-hop bandit arms`, `follow-up tools`, and `all enabled
  tools` separately; the manifest records `first_hop_tools` / `followup_tools` /
  `all_enabled_tools` (`tools_enabled` is a back-compat alias for the first-hop
  search arms).
- **`--enable-query-decomposition` (Level 2)**: decompose each long question into
  several targeted clue queries and let the bandit learn the query form
  (`tool × query_arm`). On BrowseComp this is **required** — Level 1 routing alone
  scored 0 because searching the whole prompt missed the exact answer (see
  `docs/METHODOLOGY_RISKS.md`). The flag is recorded as `query_decomposition_enabled`
  in the dry-run print / manifest / report. **Tune search targeting before adding
  Firecrawl scrape** — scraping a contaminated page just adds cost.
  Recommended first real-query run:
  ```bash
  python scripts/run_live.py --dataset livebrowsecomp --dataset-path PATH \
      --optimize 10 --confirm 20 --budgets 1,3 --search-provider-mode diverse \
      --enable-query-decomposition \
      --recording-cache results/live/cache.json --execute
  ```
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

## C. Tiny full run — all four conditions, budgets 1,3, 10/20 (cheap)
```bash
python scripts/run_live.py --dataset livebrowsecomp --dataset-path PATH \
    --optimize 10 --confirm 20 --budgets 1,3 \
    --recording-cache results/live/cache.json --execute
```
- Calls providers: **yes (with `--execute`).** Run dir: `results/live/<run_id>/`.
- Est. calls: **~180 answerer + ≤440 tool** (incl. experience 10×4), all `gpt-5.4-mini`.
- Env: `OPENAI_API_KEY` + ≥1 external search key (e.g. `SERPER_API_KEY`).

## C′. The MAIN experiment — provider-diverse routing
```bash
python scripts/run_live.py --dataset livebrowsecomp --dataset-path PATH \
    --optimize 10 --confirm 20 --budgets 1,3 --search-provider-mode diverse \
    --recording-cache results/live/cache.json --execute
```
- Each enabled **search** provider (Serper/Brave/Tavily/Exa/OpenAI) is a separate
  first-hop bandit arm — this is what regimes-probe is actually testing.
  `page_fetch` is a **follow-up** tool (runs on a URL from evidence), not a bandit
  arm, and is not given the first-hop search budget in the cost estimate.
- Env: `OPENAI_API_KEY` + the search-provider keys you want as arms.

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
- Est. calls: **~1,200 answerer + ≤3,700 tool** (real money). Env: `OPENAI_API_KEY` + search keys.

## F. OpenAI-hosted strong baseline — LATER, expensive, opt-in only
```bash
python scripts/run_live.py --dataset livebrowsecomp --dataset-path PATH \
    --optimize 10 --confirm 20 --budgets 1,3 \
    --search-provider-mode openai-hosted \
    --answer-model gpt-5.5 --web-search-model gpt-5.5 \
    --recording-cache results/live/cache.json --execute
```
- A **strong/expensive comparison point**, NOT the default regimes-probe test.
  `run_live` prints an explicit ⚠️ EXPENSIVE warning for `gpt-5.5 + openai_web_search`.
- Env: `OPENAI_API_KEY`. Run this only after the cheap/diverse runs work.

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

## Extra tool families (opt-in; add only after cheap search works)
- `--enable-scrape-tools` adds `firecrawl_scrape` (full-page markdown — richer
  evidence than snippets).
- `--enable-agentic-tool-discovery` adds Monid `monid_discover`/`monid_inspect`
  (a learned *tool-discovery* arm; auto-on in `diverse`).
- `--allow-stateful-or-paid-tools` (Monid `monid_run`) and
  `--enable-browserish-tools` (`firecrawl_interact`) are **off by default** and are
  NOT part of the first BrowseComp runs.
- `browser_use` is **deferred** (browser control; prompt-injection/state risk).
- Wokelo (`wokelo_research`/`wokelo_company_lookup`) needs `WOKELO_BASE_URL` +
  endpoint path; otherwise it fails closed.

See **[TOOL_ABSTRACTIONS.md](./TOOL_ABSTRACTIONS.md)** for the full taxonomy,
per-arm metadata, and the safety gates. Recommended order: cheap search-only →
add scrape → add agentic discovery → (later) specialized research.

## Step 6 — Offline ablations (after a paid run; NO new spend)

Always pass `--recording-cache results/live/cache.json` on the paid run so its
provider/model/tool outcomes are stored. Then compare policy variants for free:

```
python scripts/fork_offline_ablation.py results/live/<run_id> \
    --out <run_id>-fork-a --policy-config ablations/variant_a.yaml
```

The fork replays the cache (refuses on any miss — it never calls a provider),
writes a full artifact set marked `offline_fork=true`/`parent_run_id`, and is
never headline-eligible. Fork the memory variant (`policy_memory`); read the
fixed `no_memory_search` baseline from the parent report. See
**[OFFLINE_FORK_ABLATIONS.md](./OFFLINE_FORK_ABLATIONS.md)**.

## Step 7 — Level 3 evidence reading (page_fetch + firecrawl_scrape)

`--enable-scrape-tools` adds `firecrawl_scrape` (needs `FIRECRAWL_API_KEY`) as a
**follow-up reading tool alongside `page_fetch`** — never a first-hop search arm.
The reading policy picks `firecrawl_scrape` for PDFs / structured / authoritative-
but-thin pages and `page_fetch` for simple pages or as the fallback. Reads count
against budget, are cached/replayable, and Firecrawl 402/quota errors fail closed
to `page_fetch`. Evaluate scrape **after** search + candidate targeting work
(do not add it first):

```bash
python scripts/run_live.py --dataset livebrowsecomp --dataset-path PATH \
    --optimize 10 --confirm 20 --budgets 1,3 --search-provider-mode diverse \
    --enable-query-decomposition --enable-iterative-clue-resolution \
    --enable-scrape-tools \
    --recording-cache results/live/cache.json --execute
```

See **[QUERY_POLICY.md](./QUERY_POLICY.md)** (Level 3) and watch
`scrape_to_answer_rate` / `scrape_failure_counts` in the report.
