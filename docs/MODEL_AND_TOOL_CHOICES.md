# Model and Tool Choices

This document defines the default **live** configuration and the tool roster. Everything
here is wrapped behind common interfaces so the contextual-bandit router never depends on a
specific provider's behavior, and so every network call is replayable through ActiveGraph.

## Cheap-first by default (important)

`regimes-probe` is about **learned epistemic policy across multiple search
providers/tools** — NOT OpenAI hosted browsing. The live defaults are
**cheap-first**, and OpenAI hosted `web_search` (especially with `gpt-5.5`) is an
**opt-in strong baseline**, never the silent center of the experiment.

Three `--search-provider-mode`s:

First-hop **search** providers are the bandit arms (one per provider). `page_fetch`
(and scrape tools) are **follow-up** tools: they operate on a URL returned by a
search and are *not* bandit arms — the router never routes them first-hop.

- **`cheap`** (default) — answerer `gpt-5.4-mini`; first-hop search arms = any
  low-cost external adapter whose key is present (Serper/Brave/Tavily/Exa), with
  `page_fetch` available as a follow-up tool. OpenAI `web_search` is **not** added.
  With no external key, the run **explains which env vars are needed** instead of
  falling back to expensive hosted search.
- **`diverse`** (the **main experiment**) — every external search adapter with a
  key, plus OpenAI `web_search` as **one arm among many** (unless disabled); the
  router/bandit sees each **search** provider as a separate first-hop arm.
  `page_fetch` remains a follow-up tool, not an arm.
- **`openai-hosted`** (later **strong/expensive baseline** only) — explicit
  `openai_web_search` first-hop arm + `page_fetch` follow-up; may use `gpt-5.5`.

## Defaults at a glance

| Role | Default (cheap-first) | Opt-in baseline | Knob |
| --- | --- | --- | --- |
| Answerer | `gpt-5.4-mini` (OpenAI Responses) | `gpt-5.5` | `--answer-model` / `live.answer_model` |
| Web-search tool model | `gpt-5.4-mini`, ctx `low` | `gpt-5.5` | `--web-search-model` / `--web-search-context-size` |
| First-hop search arms | provider-diverse cheap externals (`page_fetch` is follow-up only) | `openai_web_search` | `--tools` / `--search-provider-mode` |
| Embeddings | `HashEmbedder` (tests) | OpenAI embeddings (live) | `live.embedder` |

OpenAI model and tool references:

- Models: <https://developers.openai.com/api/docs/models>
- Web search tool: <https://developers.openai.com/api/docs/guides/tools-web-search>

## Answerer

- **Default:** `gpt-5.4-mini` via the **OpenAI Responses API** (cheap-first).
  Configurable with `--answer-model` / `live.answer_model`.
- **Strong baseline:** `gpt-5.5` (`live.strong_answer_model`) — expensive; use only
  for the explicit hosted baseline, not the learning runs.
- The answerer consumes only **Layer 2 policy memory** (answer-free priors) and
  `evidence_observation`s from tools — never raw archived traces or benchmark answers
  (see [LEAKAGE_CONTROLS.md](./LEAKAGE_CONTROLS.md)).

## Embeddings

- **Tests:** deterministic `HashEmbedder` so `query_signature` creation is reproducible and
  needs no API keys.
- **Live:** optional OpenAI embeddings for higher-fidelity signatures. The embedder is
  selected by config; the bandit and signature features are agnostic to which is used.

## Search and browsing tools

### Why this roster

- **`openai_web_search` is a strong hosted browsing baseline** and matches what these
  benchmarks (BrowseComp / LiveBrowseComp) are typically run against, so the no-memory
  baseline is competitive and the learned-policy gain is meaningful rather than an artifact
  of a weak baseline.
- **Independent search adapters (Brave/Tavily/Exa/Serper/etc.) create real tool diversity**
  for the contextual bandit. Routing is only interesting if tools genuinely differ in
  coverage, freshness, and authority; a single provider gives the router nothing to learn.
- **Every provider is wrapped as an ActiveGraph tool** so requests/responses are recorded
  as `tool.requested`/`tool.responded` events and the whole run is replayable
  deterministically. No provider SDK is called from inside a behavior.

### No provider-specific logic in the router

The router scores an **abstract action set** (the adapter names below) keyed by
`query_signature`. Provider-specific request shaping, auth, pagination, and response
normalization live **inside each adapter behind a common interface**. The bandit must never
branch on a provider's quirks; swapping `brave_search` for `tavily_search` is a config
change, not a router change.

### Adapter roster

| Adapter | Purpose |
| --- | --- |
| `openai_web_search` | hosted Responses API web search; default baseline |
| `openai_web_search_low_context` | hosted web search, reduced context (cheaper/faster) |
| `openai_web_search_unlimited_context` | hosted web search, large context (**optional/expensive**) |
| `generic_web_search` | provider-neutral general web search |
| `news_search` | recency-oriented news search (freshness-sensitive signatures) |
| `page_fetch` | fetch a specific URL/page (records page hash) |
| `official_domain_search` | constrained search restricted to an allowed-domains list (authority-sensitive) |
| `brave_search` | Brave Search API |
| `tavily_search` | Tavily API |
| `exa_search` | Exa API |
| `serper_search` | Serper (Google SERP) API |
| `academic_search` | scholarly/paper search |
| `code_search` | code/repository search |
| `firecrawl_search` | Firecrawl web search (+ optional page content) — `search` family |
| `firecrawl_scrape` | Firecrawl full-page markdown/metadata — `scrape` family (needs `--enable-scrape-tools`) |
| `firecrawl_interact` | Firecrawl browser-like interaction — `browserish`, **disabled by default** |
| `monid_discover` / `monid_inspect` | Monid agentic **tool discovery** (needs `--enable-agentic-tool-discovery`, auto in diverse) |
| `monid_run` | Monid tool **execution** — stateful/paid, **disabled by default** |
| `wokelo_research` / `wokelo_company_lookup` | Wokelo specialized research — **scaffold; fails closed** without base URL/path |
| `browser_use` | **DEFERRED** — browser control; not a live adapter in v0 |

These extra families (scrape / agentic-discovery / specialized-research /
browserish) are documented in **[TOOL_ABSTRACTIONS.md](./TOOL_ABSTRACTIONS.md)**,
including per-arm metadata and the safety gates. Add them only **after** a cheap
search-only provider-diverse run works.

All adapters implement the same interface (request signature, normalized
`evidence_observation` output, cost/latency reporting) so they are interchangeable from the
router's perspective.

## Environment variables

All keys are **optional**. **Tests never require them** (tests use deterministic fake tools
and `HashEmbedder`). An adapter whose key is unset is simply unavailable to the router for
that run and is excluded from its action set.

| Env var | Used by | Required? |
| --- | --- | --- |
| `OPENAI_API_KEY` | answerer, OpenAI embeddings, `openai_web_search*` | optional (needed only for live OpenAI features) |
| `BRAVE_SEARCH_API_KEY` | `brave_search` | optional |
| `TAVILY_API_KEY` | `tavily_search` | optional |
| `EXA_API_KEY` | `exa_search` | optional |
| `SERPER_API_KEY` | `serper_search` | optional |
| `FIRECRAWL_API_KEY` | `firecrawl_search` / `firecrawl_scrape` / `firecrawl_interact` | optional |
| `MONID_API_KEY` | `monid_discover` / `monid_inspect` / `monid_run` | optional |
| `WOKELO_API_KEY` | `wokelo_research` / `wokelo_company_lookup` | optional |
| `WOKELO_BASE_URL` | Wokelo endpoint base (no default; required to call) | optional |
| `WOKELO_RESEARCH_PATH` / `WOKELO_COMPANY_PATH` | Wokelo endpoint paths | optional |
| `WOKELO_OPENAPI_PATH` | reserved for loading a Wokelo OpenAPI spec later | optional |

If a key referenced by a configured adapter is missing at run time, the harness logs the
exclusion in the `benchmark_run`/`report` and continues with the remaining roster, keeping
runs reproducible and key-optional.
