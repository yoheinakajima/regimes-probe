# QUERY_POLICY.md

## Purpose (Level 2: query formulation)

Once Level 1 routing (`docs/ROUTING_POLICY.md`) selects a tool, the query policy decides
**how to query** it. A `query_plan` selects a **query policy arm** (a transformation from
question to query string), scored by the contextual bandit
(`docs/CONTEXTUAL_BANDIT.md`) using trace-derived `query_template_rewards`
(`docs/POLICY_MEMORY.md`).

There are **no hard-coded topical categories**. Arms are generic epistemic
transformations; their priors are conditioned on latent regimes and generic query
signatures.

---

## Query policy arms

| Arm | Definition |
|---|---|
| `direct_question` | Pass the (normalized) natural-language question through unchanged. |
| `keyword_compressed` | Strip stopwords/function words; keep salient content tokens and entities. Improves recall for `query-fragile` signatures. |
| `quoted_entities` | Wrap detected named entities / exact phrases in quotes to force exact-match retrieval. |
| `source_constrained` | Append source-quality constraints (e.g., authoritative-source operators) for `source-authority-sensitive` signatures. |
| `freshness_terms` | Inject recency tokens / date hints for `freshness-sensitive` / `staleness-prone` signatures. |
| `exact_answer_shape` | Add tokens cueing the required answer shape (date, number, name, enum) for `answer-shape-constrained` signatures. |
| `site_or_domain_constrained` | When a domain is known, constrain to that site/domain (e.g., `site:` operator). Only valid when a domain is available. |

Arms are not mutually exclusive in principle, but v0 selects **one arm per `query_plan`**
(composition is left to learned, tool-specific templates).

---

## Arm examples

Example question: **"Who won the 2026 Eurovision Song Contest?"**
(detected entities: `Eurovision Song Contest`; signature includes `freshness-sensitive`,
`answer-shape-constrained`; domain `eurovision.tv` known.)

| Arm | Resulting query string |
|---|---|
| `direct_question` | `Who won the 2026 Eurovision Song Contest?` |
| `keyword_compressed` | `2026 Eurovision Song Contest winner` |
| `quoted_entities` | `"Eurovision Song Contest" 2026 winner` |
| `source_constrained` | `2026 Eurovision Song Contest winner official results` |
| `freshness_terms` | `2026 Eurovision Song Contest winner latest results` |
| `exact_answer_shape` | `2026 Eurovision Song Contest winner name` |
| `site_or_domain_constrained` | `Eurovision Song Contest 2026 winner site:eurovision.tv` |

The exact transformation logic per arm is deterministic and **template-driven** so runs
are replayable.

---

## Learned and tool-specific templates

- **Query templates can be learned and scored.** Each arm maps to one or more parametric
  templates; their `query_template_rewards` are tracked per `(regime_label, signature)`
  and per tool, so the bandit learns which arm/template wins where.
- **Tool-specific templates are allowed.** The same arm may render differently for
  different tools (e.g., a web search engine vs. a news API vs. a fetch tool), because
  operator syntax differs. Templates are keyed by `(arm, tool)`.

---

## LLM use in query generation (v0 only, gated)

An LLM may generate a query string in v0 **only if** all of the following hold:

1. **Logged** — the prompt, model, and output are recorded in the raw trace archive.
2. **Cached** — identical inputs return the cached output.
3. **Deterministic enough for replay** — fixed model + decoding params + cache, so a
   replay reproduces the same query string.
4. **Replaceable by templates later** — the generated query must be expressible as, and
   later swapped for, a learned template, so the LLM is not a permanent dependency.

The default and target system uses learned template selection, not an LLM.

---

## Benchmark must distinguish three query modes

The benchmark explicitly separates and labels these conditions so results are comparable:

| Mode | Description |
|---|---|
| **generic LLM query** | An LLM formulates the query (gated as above). Upper-bound / reference condition. |
| **learned template selection** | The contextual bandit selects among the query policy arms / learned templates. This is the system under test. |
| **fixed query baseline** | Always use a single fixed arm (e.g., `direct_question`) with no learning. Lower-bound baseline. |

Each `query_plan` records which mode produced it, so `correct_per_tool_call` can be
attributed and compared across modes.

---

## Selection (reuses the contextual bandit)

```
function choose_query_plan(ctx, tool, snapshot, mode, now_tick, domain_known):
    arms = [direct_question, keyword_compressed, quoted_entities,
            source_constrained, freshness_terms, exact_answer_shape]
    if domain_known:
        arms.append(site_or_domain_constrained)     # only valid when a domain exists

    # score_tool / choose_tool_plan operate over the query_template arm family
    plan = choose_tool_plan(ctx, snapshot, mode, now_tick, available_arms = arms)

    query_string = render_template(plan.chosen_arm, tool, ctx)   # deterministic
    return QueryPlan(
        arm = plan.chosen_arm,
        tool = tool,
        query_string = query_string,
        ranked = plan.ranked,
        query_mode = "learned_template_selection",   # or generic_llm_query / fixed_query_baseline
        explanation = plan.explanation
    )
```

---

## Failure regime addressed

| Regime | How query policy relates |
|---|---|
| `query_miss` | Poorly-formed query returns no usable evidence; corrected by learning `query_template_rewards` per signature (`query-fragile`, `evidence-sparse`). |

---

## Level 2: query decomposition (multi-query clue extraction)

**Motivation (from the first clean BrowseComp run).** Level 1 routing alone
produced **no accuracy gain** on BrowseComp: the dominant failure seam was
`exact_answer_missing`, and searching the *whole* long clue-dense prompt as one
query surfaced spam / benchmark-mirroring pages rather than the evidence page that
carries the answer. The bottleneck was **query formulation**, not tool choice.

**v0 → v1.** v0 fixed the whole-prompt problem and reduced contamination, but
extracted single capitalized tokens, yielding weak/broad queries (`African One`,
`Mexican`, `Name December 2023`, `Between Asia 1945-1955`). Accuracy stayed 0.
**v1 extracts clue SPANS and scores query quality.**

`policy/query_decomposition.py` (deterministic v1; no model, no network) splits the
question into clauses, extracts **clue spans** per clause, and composes 3–6
targeted, length-capped candidate queries, one per arm:

| arm | what it sends (v1: spans, not single tokens) |
|---|---|
| `exact_phrase_clue` | quoted phrases, else the most distinctive multiword span, quoted |
| `entity_clue` | multiword entities / top phrase spans (never a single generic word) + an answer-shape hint |
| `relation_clue` | core spans from **different clauses** (predicate + object) + a title word |
| `rare_terms_clue` | 4–8 rare terms from across clauses + the top span |
| `date_range_clue` | a date/range **attached to a nearby noun phrase** (never a bare date) |
| `location_constraint_clue` | a location entity + the object-type span (e.g. `New Mexico` + `Mexican restaurant`) |
| `source_type_query` / `quoted_anchor_terms` | entity/span + source hint; or top entities quoted |
| `negative_noise_removed` / `full_question_compressed` | question minus noise — **fallback only** |

Clue-span extraction keeps quoted phrases, multiword entities, noun-phrase spans
around distinctive predicates (`road accident`, `private university`,
`food festival`), date+noun phrases (`19th century monument`), institution/source
names, and rare terms (`ironworks`, `manga`). It **avoids** single generic
capitalized words (`African`/`Mexican`/`Name`/`Early`/`One`), pure-date queries,
boilerplate, and <3-meaningful-token queries unless quoted/named-entity. Each
candidate is scored (`score_query`: meaningful/rare token counts,
generic-token penalty, span-length, specificity → `expected_search_quality`);
candidates below `QUALITY_THRESHOLD` / with no noun-like head / too short are
**dropped** (with a reason). Kept arms are chosen by an arm-priority order so
concise distinctive forms survive the cap; the fallback arms are always **last**,
so the long arm is never the cold-start default.

### Query forms as bandit arms (tool × query_arm)

When decomposition is enabled, the learned query bandit chooses the **query
form**, conditioned on the tool: the context key is `cluster|tool`, so the policy
learns `tool × query_arm → reward`. Each tool call records `query_arm`,
`query_text_hash`, `query_text_preview`, and the `clue_ids` it used; the reward
attributes back to the chosen `(tool, query_arm)` pair.

### Flag

Off by default (preserves prior behavior). Enable with
`--enable-query-decomposition` (or `policy.enable_query_decomposition: true`). The
dry-run print, `run_manifest.json`, and `report.json` all record
`query_decomposition_enabled`, and `debug_questions.jsonl` includes
`query_text_preview` + `query_arm` per call.

### Optional LLM decomposition (future)

A model-based decomposer may be layered on later, but it MUST go through the
existing answerer/model cache, be logged/replayable, carry a prompt-version hash,
default to the cheap `answer_model`, and stay behind the same flag. v0 is heuristic
and always available (no key, no spend).

## Benchmark-contamination penalty

`eval/contamination.py` flags results that mirror the benchmark rather than carry
evidence: known eval/dataset hosts (HuggingFace / GitHub / arXiv / simple-evals /
paperswithcode), `BrowseComp`-naming pages, or snippets that reproduce a long span
of the question. Contaminated results earn a reward penalty
(`RewardWeights.contamination_penalty`) so the query/tool form that surfaces them
learns to avoid them. `report.json.contamination` reports
`benchmark_contaminated_result_count`, `contamination_rate`, and per-provider /
per-domain breakdowns.

## Level 2b: iterative clue resolution (staged candidate-entity search)

**Motivation (v1/v2 runs).** Clue-span decomposition (above) reduced contamination
and produced sane queries, but accuracy stayed 0 with seam `exact_answer_missing`:
the agent issued *parallel* clue queries and hoped the answer was in a snippet.
BrowseComp usually needs **iterative resolution** — identify an intermediate
entity from clue 1's results, then search that entity together with clue 2, etc.

`agent/clue_resolution.py` (deterministic v0; gold-free, no network) implements
staged search, enabled by `--enable-iterative-clue-resolution`:

- **Stage 1** issues the best decomposed clue query.
- **Candidate extraction**: from result titles/snippets/URLs, pull proper-noun
  phrases (multiword preferred over single tokens), score them by **frequency**
  across results, **source authority**, and **proximity** to clue terms, and
  **filter generic hubs** (Facebook/Wikipedia/YouTube/Reddit/… and noise domain
  labels like `example`/`news`).
- **Stage 2+**: compose a follow-up query = `"{top candidate entity}"` + the next
  unused clue span (arm `candidate_entity_followup`), or + an answer-shape hint
  (arm `answer_shape_followup`). Continue until the budget is exhausted or the
  evidence carries a likely answer. **Every follow-up counts against the budget.**

Each call records its `stage`, `parent_query_id`, the extracted `candidate_entities`,
the `selected_candidate`, the selection `reason`, and whether `evidence_improved`.
`debug_questions.jsonl` carries the full **stage chain**, and
`scripts/debug_run_failures.py` prints it (Stage 1 query → results → entities →
Stage 2 query → …). Metrics: `mean_candidate_entity_count`,
`mean_followup_query_count`, `evidence_improved_after_followup_rate`,
`mean_stage_depth_used`, `answer_found_after_stage_mean`.

### Strategy as a policy arm

The follow-up query forms are real query-bandit arms (`candidate_entity_followup`,
`answer_shape_followup`), distinct from the **direct clue** arms (`entity_clue`,
`exact_phrase_clue`, …). Their reward is attributed per `(tool, query_arm)` and per
signature cluster, so policy memory learns **whether iterative resolution helps a
given cluster** — exactly as it learns tool and query-form choice.

**Sequencing:** evaluate Firecrawl scrape/fetch only **after** candidate-entity
targeting works — fetching a page the staged search would not have found just adds
cost. Iterative clue resolution comes before scrape.

## Level 2c: typed candidate hypotheses (role + evidence-progress gating)

**Motivation (iterative v0 run).** Staged search worked structurally but latched
onto the *wrong* intermediate candidate and re-exploited it: source/publisher names
(`Brittle Paper`), broad orgs (`World Health Organization`), broad locations
(`Tennessee`), generic concepts (`Art Deco`, `Antagonist`), program/source names
(`National Sheriffs`). That is a **hypothesis-selection** failure, not a provider
or contamination failure — and it is generic to multi-step search, not BrowseComp-
specific.

`agent/clue_resolution.py` now treats each extracted candidate as a **typed
hypothesis** (`CandidateHypothesis`) and carries it forward only if it is
role-compatible with the unresolved target and follow-up evidence improves.

- **Generic roles**: `person`, `organization`, `location`, `title_or_work`,
  `publication_or_source`, `event`, `concept`, `date_or_time`, `unknown`
  (`classify_entity_role`, deterministic, gold-free).
- **Target-role inference** (`infer_target_roles`): maps question cues to a small
  set of `target_roles` + `intermediate_roles` (who/founder/author → person;
  which TV series/book/manga → title_or_work; restaurant/museum → organization;
  town/monument/where → location; year/when → date_or_time as answer shape).
- **Role-compatible scoring** (`adjusted_score`): boost candidates whose role is in
  `target_roles` (+) or plausible `intermediate_roles`; apply
  `source_entity_penalty` (publisher/platform unless a source is wanted),
  `location_penalty` (broad locations unless a place is wanted, via a generic
  continents/states/countries lexicon), `genericity_penalty` (concepts; broad
  `World/National/…` orgs; domain-only candidates; obvious clue terms that don't
  narrow; contaminated-only candidates). Cross-domain corroboration boosts.
- **Evidence-progress gate**: a candidate is preferred for the next hop only if its
  follow-up improved evidence; a follow-up that adds nothing is marked
  `no_progress` and downweighted.
- **Anti-sticky beam** (`HypothesisBeam`, beam_size 3): apply `sticky_penalty` to a
  candidate that made no progress, **force exploration** of the next-best after a
  candidate fails twice, avoid repeating equivalent follow-up queries, and record
  rejected candidates + reasons. This prevents loops like
  `Art Deco → Art Deco → Art Deco`.
- **Composition**: follow-up = `"{candidate}"` + one *unresolved* distinctive clue
  span (preferred) or an answer-shape hint — never `candidate + generic filler`,
  never a repeated query, never an already-failed clue.

Debug (`debug_questions.jsonl`, `scripts/debug_run_failures.py`) shows
`target_roles`, the candidate hypotheses with role + raw/adjusted scores + each
penalty, the selected candidate + role + reason, the rejected candidates +
`rejection_reason`, and the sticky / no-progress flags per stage. Metrics:
`selected_candidate_role_counts`, `candidate_role_match_rate`,
`candidate_rejection_counts`, `sticky_candidate_count`,
`no_progress_followup_count`, `evidence_improved_after_candidate_rate`,
`mean_beam_size`, `candidate_switch_count`, `repeated_candidate_query_count`.

**Sequencing:** scrape/fetch still comes *after* candidate targeting improves —
scraping the wrong page only yields richer wrong evidence.

## Level 3: evidence reading (page_fetch vs firecrawl_scrape)

After search (Level 2) and candidate targeting (Level 2c) select a URL, **Level 3
reads the page**. Reading tools are URL-only follow-ups — never first-hop search
arms (`tools/metadata.py`: `page_fetch`=fetch, `firecrawl_scrape`=scrape; both in
`FOLLOWUP_FAMILIES`, so `first_hop_tools` excludes them). Reads count against the
tool budget, are recorded/replayable/cached like any tool call, and surface
provider failures (incl. Firecrawl **402/quota**) without crashing the run.

**Tool selection** (`agent/reading_policy.py:select_reading_tool`, deterministic):
- `page_fetch` — cheap basic-HTML fetch; the default and the fallback.
- `firecrawl_scrape` — richer (and **paid / quota-limited**) extraction, preferred
  for PDFs, pages with likely hidden/structured content (records/database/
  directory/census/…), or authoritative pages whose search snippet is too short to
  contain the answer / corroborated across providers. Gated behind
  `--enable-scrape-tools` + `FIRECRAWL_API_KEY`.

**When to read** a URL: it is from an authoritative / likely-evidence domain, the
title/snippet matches an unresolved clue or answer-shape hint, or the same
URL/domain appears across independent search providers — and it has not already
been read. **Do not read**: contaminated (benchmark-mirroring) URLs, social-media
pages (unless `allow_social_scrape`), generic pages with no target-relevant clue
match, or domains/URLs that already produced no progress; and never scrape every
top result blindly.

**Fail closed + fallback**: if `firecrawl_scrape` errors (402/HTTP/quota), the call
becomes a recorded failed observation (no crash) and — if
`policy.scrape_fallback_to_page_fetch` (default true) — the **same URL is retried
with `page_fetch`** on the next step (counted against budget). The failing domain
is not re-scraped.

Debug (`debug_questions.jsonl` `scrape` block; `scripts/debug_run_failures.py`)
records per read: `read_tool`, `scrape_url`, `scrape_provider`,
`scrape_selected_reason`, `scrape_success`, `scrape_chars`,
`evidence_added_by_scrape`, `answer_shape_found_after_scrape`,
`unresolved_clues_supported_after_scrape`, `scrape_failure_type`,
`scrape_cost_estimate`, `fallback_to_page_fetch`. Metrics: `scrape_call_count`,
`scrape_success_rate`, `evidence_added_by_scrape_rate`,
`answer_shape_found_after_scrape_count`, `scrape_failure_counts`,
`scrape_fallback_count`, `scrape_to_answer_rate`.

**Sequencing:** scrape is **Level 3**, applied only after search and candidate
targeting — scraping the wrong page just yields richer wrong evidence.
