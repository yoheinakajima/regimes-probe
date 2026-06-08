# Real benchmark readiness checklist

What stands between today's **synthetic-harness scaffold** and a **credible real
benchmark result** on LiveBrowseComp / BrowseComp. Each item is tagged with one
status. Run `python scripts/validate_live_readiness.py` for an automated subset
of these checks (no providers called).

> The current claim is: *the scaffold demonstrates the intended mechanism on a
> synthetic fixture.* It is **not** a BrowseComp/LiveBrowseComp result. See
> `docs/STATUS.md` and `docs/METHODOLOGY_RISKS.md`.

## Legend
- ✅ **Implemented** — done and tested offline.
- 🟡 **Implemented but unverified on real data** — code exists; never run against
  the actual benchmark.
- 🔑 **Requires API keys / network** — blocked until credentials are available.
- 📦 **Requires dataset access** — blocked until the real dataset is downloaded
  or HF access is configured.
- 🧭 **Requires methodological decision** — a choice must be made before a result
  is credible.
- ⏸️ **Not needed for v0** — out of scope now.

## Adapter & data path
| item | status | notes |
| --- | --- | --- |
| BrowseComp obfuscated-CSV decode (`problem`/`answer`/`canary`) | ✅ | `datasets/browsecomp.py`; round-trip + graceful-skip tested. |
| BrowseComp loads the real CSV | 📦 | Download from openai/simple-evals; set `dataset.browsecomp_csv`. |
| LiveBrowseComp local-JSONL loader | ✅ | `datasets/livebrowsecomp.py`; tested on placeholder JSONL. |
| LiveBrowseComp Hugging Face loader (`Forival/LiveBrowseComp`) | 🟡 / 📦 | Code present; needs `datasets` pkg + network. |
| Dataset version + checksum capture | ✅ | `DatasetAdapter.version()/checksum()`. |
| Graceful failure on missing/encrypted data | ✅ | `DatasetUnavailable`; tested. |
| Real-data-SHAPED smoke test (no keys) | ✅ | `tests/test_real_data_shape.py`, `fixtures/real_shaped/`. |
| Confirm field names match the real release | 🧭 / 📦 | `_row_to_item` guesses `question`/`problem`/`answer`; verify against the actual schema and adjust. |
| Canary/obfuscation scheme matches the real release | 🧭 / 📦 | Our scheme mirrors simple-evals; confirm byte-for-byte on real rows. |

## Splits & leakage
| item | status | notes |
| --- | --- | --- |
| Deterministic OPTIMIZE/CONFIRM split | ✅ | `eval/split.py`; disjointness tested. |
| Time-disjoint split (recommended for LiveBrowseComp) | ✅ | `mode="time"`; needs `released_at` on real rows. |
| Entity-disjoint split | 🧭 | Not implemented; decide whether entity disjointness is required and how to extract entities. |
| No-answer-leakage guard on policy memory | ✅ | `assert_no_answer_leakage`; tested. |
| No-answer-leakage check on **real** traces | 🟡 | Guard is generic, but should be re-run/inspected on real traces. |
| Exact-question-overlap guard across splits | 🧭 | `norm_hash` exists; wire an explicit cross-split overlap assertion for real data. |

## Agent, policy, learning
| item | status | notes |
| --- | --- | --- |
| Level 1 routing bandit | ✅ | `policy/router.py`, `policy/contextual_bandit.py`. |
| Level 2 query / verify / stop policies | ✅ | `policy/{query,verification,stopping}_policy.py`. |
| Answer-free policy memory + frozen snapshot | ✅ | `policy/memory.py`. |
| Signatures generalize on real questions | 🟡 / 🧭 | Heuristic epistemic features may need tuning on real phrasing. |
| Deterministic hash embedder | ✅ | Fine for tests; for real runs consider OpenAI embeddings (🔑). |
| Real answerer (OpenAI Responses `gpt-5.5`) | 🟡 / 🔑 | Adapter shape exists for search; an answerer wrapper must be wired to the loop. |

## Tools / providers
| item | status | notes |
| --- | --- | --- |
| Provider adapter interface (router stays provider-agnostic) | ✅ | `tools/base.py`. |
| OpenAI Responses `web_search` (+ low/unlimited context) | 🟡 / 🔑 | `tools/openai_web_search.py`; env-gated, never called in tests. |
| Page fetch | ✅ (fixture) / 🟡 (live) | `tools/page_fetch.py` (stdlib). |
| Independent adapters (Brave/Tavily/Exa/Serper, news, official-domain) | 🟡 / 🔑 | Implemented, env-gated. |
| Recorded-tool replay (responses read from log) | ✅ | `RecordingInvoker`/`ReplayInvoker`. |
| Live tool calls wired into the recorded run | 🧭 | `RecordingInvoker` already records any provider; just pass live providers. Decide caching policy for replay. |

## Evaluation, metrics, reporting
| item | status | notes |
| --- | --- | --- |
| `correct_per_tool_call` + secondary metrics | ✅ | `eval/metrics.py`. |
| McNemar + bootstrap CI | ✅ | `eval/significance.py`. |
| Budget curves | ✅ | `scripts/run_budget_curve.py`, report. |
| Full report artifacts + replay check | ✅ | `eval/report.py`, `replay_check`. |
| Closed-book (no-search) baseline | 🧭 | Required for real runs; see Methodology risks. Add as a condition. |
| LLM-judge grading (for free-form answers) | 🟡 / 🔑 | Grader hook exists (`grade(..., judge=)`); judge fn must be wired + logged. |
| Cost / latency accounting from real providers | 🟡 | Fields exist on `SearchResponse`; populated by live adapters. |

## Regimes improvement loop
| item | status | notes |
| --- | --- | --- |
| Failure-regime detectors | ✅ | `regimes/detectors.py`. |
| Bounded mutations + OPTIMIZE/CONFIRM gates | ✅ | `regimes/{action_space,gates,runner}.py`. |
| Promotion/rejection events | ✅ | Tested. |
| Loop usefulness on real data | 🟡 / 🧭 | On synthetic data the bandit is already strong, so the loop often (correctly) rejects. Real data will exercise it more. |

## Methodological decisions still open (🧭)
1. **Closed-book + no-search baselines** to bound intrinsic knowledge.
2. **Same model / tools / prompt across conditions** — fix exact versions.
3. **Split policy** — time-disjoint vs entity-disjoint for LiveBrowseComp.
4. **Judge** — exact-match only, or an LLM judge (and which model)?
5. **Caching / replay policy** for non-deterministic live providers.
6. **Stopping criteria for a "real" run** — how many items, how many seeds.

## Not needed for v0 (⏸️)
- Arbitrary prompt/code mutation; answer/judge-prompt optimization.
- BrowseComp-Plus fixed-corpus retrieval.
- Natural-language "lesson" (layer-3) generation.
- Multi-provider cost optimization beyond the per-call accounting already present.
