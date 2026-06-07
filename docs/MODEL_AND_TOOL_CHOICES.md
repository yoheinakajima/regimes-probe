# Model and Tool Choices

This document defines the default **live** configuration and the tool roster. Everything
here is wrapped behind common interfaces so the contextual-bandit router never depends on a
specific provider's behavior, and so every network call is replayable through ActiveGraph.

## Defaults at a glance

| Role | Default | Cheaper / alt | Notes |
| --- | --- | --- | --- |
| Answerer | `gpt-5.5` (OpenAI Responses API) | `gpt-5.4-mini` | `answer_model` / `cheaper_answer_model`; configurable |
| Embeddings | `HashEmbedder` (tests) | OpenAI embeddings (live) | deterministic in tests |
| Search baseline | `openai_web_search` (Responses API `web_search`) | independent adapters | `search_baseline=openai_web_search` |

OpenAI model and tool references:

- Models: <https://developers.openai.com/api/docs/models>
- Web search tool: <https://developers.openai.com/api/docs/guides/tools-web-search>

## Answerer

- **Default:** `gpt-5.5` via the **OpenAI Responses API**. Configurable per `benchmark_run`.
- **Cheaper option:** `gpt-5.4-mini` for cost-bounded runs and the `correct_per_dollar`
  ablations.
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

If a key referenced by a configured adapter is missing at run time, the harness logs the
exclusion in the `benchmark_run`/`report` and continues with the remaining roster, keeping
runs reproducible and key-optional.
