# Tool abstractions and the provider-diverse arm set

regimes-probe treats every search/fetch/scrape/discovery tool as a **separate
contextual-bandit arm**. The point of the experiment is to learn *which kind of
tool to reach for*, so the roster spans several **families**, each with metadata
and safety gates. The single source of truth is
`src/regimes_probe/tools/metadata.py` (`TOOL_META`).

## Tool families

| family | meaning | examples |
| --- | --- | --- |
| `search` | ordinary web search (snippets + urls) | `serper_search`, `brave_search`, `tavily_search`, `exa_search`, `firecrawl_search`, `openai_web_search` |
| `fetch` | retrieve a known URL (stdlib) | `page_fetch` |
| `scrape` | full-page extraction (markdown/metadata) | `firecrawl_scrape` |
| `agentic_discovery` | discover/inspect/run *tools* for a task | `monid_discover`, `monid_inspect`, `monid_run` |
| `specialized_research` | company/market research providers | `wokelo_research`, `wokelo_company_lookup` |
| `browserish` | browser-like interaction / control | `firecrawl_interact`, `browser_use` (deferred) |

## Per-arm metadata

Each arm carries: `tool_family`, `expected_cost_class` (low/medium/high/unknown),
`stateful` (can it take paid/state-changing actions?), `requires_network`,
`requires_api_key` (env-var **name** only), `safe_default` (may auto-enable in
cheap/diverse without a flag), and `implemented`. The cost estimator, manifest,
and report all surface this so a reviewer can see exactly what was in play.

The router ranks arms by learned reward, but **it only ever sees arms that the
settings resolver allowed** — unsafe families never enter the arm set unless their
explicit flag is set (see gates below).

## The providers added here

### Monid — agentic tool discovery (not a search engine)
- `monid_discover(query)` → candidate tools for a task (name, docs/url, pricing,
  input-schema summary), normalized into evidence-like records.
- `monid_inspect(tool_id)` → a tool's schema/pricing/docs.
- `monid_run(tool_id, input=…)` → **executes** a discovered tool. **Stateful/paid;
  DISABLED by default**; needs `--allow-stateful-or-paid-tools`.
- Use it to test a **learned tool-discovery policy** as one arm — *not* as a normal
  search provider. Env: `MONID_API_KEY`. Docs: <https://docs.monid.ai/>

### Firecrawl — search + scrape (+ interact)
- `firecrawl_search(query)` → search results (+ optional page content).
- `firecrawl_scrape(url)` → full-page markdown + metadata — **richer evidence than
  a snippet**, which may improve answer support. Needs `--enable-scrape-tools`.
- `firecrawl_interact` → browser-like; **SCAFFOLD, disabled by default**; needs
  `--enable-browserish-tools`.
- Env: `FIRECRAWL_API_KEY`. Docs: <https://docs.firecrawl.dev/>

### Wokelo — specialized company/market research (SCAFFOLD, fails closed)
- `wokelo_research`, `wokelo_company_lookup` — promising for company/market
  research, but `docs.wokelo.ai` is JS-rendered and the endpoint shape is unknown
  here, so these are a **guarded generic HTTP scaffold**: **unavailable unless**
  `WOKELO_BASE_URL` + an endpoint path (`WOKELO_RESEARCH_PATH` /
  `WOKELO_COMPANY_PATH`) are configured. Otherwise they **fail closed** with a clear
  message. `WOKELO_OPENAPI_PATH` is reserved for loading a spec later.
- Env: `WOKELO_API_KEY` (+ base url/path). Docs: <https://docs.wokelo.ai/>

### browser-use — DEFERRED (not a live adapter)
- <https://github.com/browser-use/browser-use> — intentionally **not** enabled in
  v0. It turns the task into **browser control** and introduces
  **prompt-injection / state** risks far beyond BrowseComp search-routing. It
  appears in the registry as `implemented: false` and is always dropped from any
  tool set; this doc is the scaffold note for a future browser-agent arm.

## Safety gates (mode- and `--tools`-derived sets alike)

| tool(s) | requires |
| --- | --- |
| `firecrawl_scrape` | `--enable-scrape-tools` |
| `monid_discover` / `monid_inspect` | `--enable-agentic-tool-discovery` **or** `diverse` mode |
| `monid_run` (stateful/paid) | `--allow-stateful-or-paid-tools` (never auto) |
| `firecrawl_interact` (browser-like) | `--enable-browserish-tools` (never auto) |
| `browser_use` | never (deferred) |

## What each mode enables

- **cheap** (default): `page_fetch` + cheap external `search` adapters with keys +
  `firecrawl_search` (if key). NO openai, scrape, discovery, stateful, or browserish.
- **diverse** (main experiment): cheap set + `monid_discover/inspect` (if key) +
  `openai_web_search` as one arm (unless disabled). `scrape` only with the flag;
  `monid_run`/`firecrawl_interact`/`browser_use` still off.
- **openai-hosted**: explicit `openai_web_search` strong/expensive baseline.

## Rollout order (do this in sequence)

1. Get a **cheap, search-only** provider-diverse run working first.
2. Then add `firecrawl_scrape` (`--enable-scrape-tools`) to test richer evidence.
3. Then add `monid_discover` (`--enable-agentic-tool-discovery` or diverse) to test
   a learned tool-discovery arm.
4. Wokelo only once the official endpoint shape/OpenAPI is known.
5. `monid_run`, `firecrawl_interact`, and browser-use are **later/again-gated** and
   are not part of the first BrowseComp search-routing experiments.
