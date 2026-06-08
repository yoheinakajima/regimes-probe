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

**v0 → v1.** v0 fixed the whole-prompt problem and **reduced contamination**, but a
second clean run showed it then produced *weak* clue queries: it extracted single
capitalized tokens, yielding `African One`, `Mexican`, `Name December 2023`,
`Between Asia 1945-1955`, `Early One first decade…`. Accuracy stayed 0
(`exact_answer_missing`). **v1 extracts clue SPANS and scores query quality.**

`policy/query_decomposition.py` (deterministic v1; no model, no network) splits the
question into clauses, extracts **clue spans** per clause, and composes 3–6
targeted candidate queries, one per arm:

| arm | what it sends (v1: spans, not single tokens) |
|---|---|
| `exact_phrase_clue` | quoted phrases, else the most distinctive multiword span, quoted |
| `entity_clue` | multiword entities / top phrase spans (never a single generic word) + an inferred answer-shape term |
| `relation_clue` | core spans from **different clauses** (predicate + object) + a title word |
| `rare_terms_clue` | 4–8 rare terms drawn from across clauses + the top span |
| `date_range_clue` | a date/range **attached to a nearby noun phrase** (never a bare date) |
| `location_constraint_clue` | a location entity + the object-type span (e.g. `New Mexico` + `Mexican restaurant`) |
| `source_type_query` | entity/span + a source hint (paper/patent/documentary/…) when obvious |
| `quoted_anchor_terms` | the 2–3 strongest entities/quotes, quoted |
| `negative_noise_removed` / `full_question_compressed` | the question minus noise — **fallback only** |

Clue-span extraction keeps quoted phrases, multiword capitalized entities,
noun-phrase spans around distinctive predicates (e.g. `road accident`,
`private university`, `food festival`, `elementary school`), date+noun phrases
(`19th century monument`), institution/source names, and rare terms (`ironworks`,
`manga`). It **avoids** single generic capitalized words (`African`, `Mexican`,
`Name`, `Early`, `One`, `Between`), pure-date queries, question boilerplate, and
over-short (<3 meaningful-token) queries unless quoted or a named entity. Queries
are length-capped; the long fallback arms are always **last**.

### Query-quality scoring

Each candidate is scored (`score_query`): `meaningful_token_count`,
`rare_token_count`, `generic_token_penalty`, `span_length_score`,
`clue_specificity_score`, and a composite `expected_search_quality`. Candidates
below `QUALITY_THRESHOLD` — or with no noun-like head, or too short — are
**dropped** (with a recorded reason). The kept arms are chosen by an arm-priority
order so concise distinctive forms (exact phrase, entity, date+noun) survive the
cap rather than long bag-of-words queries.

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
`query_decomposition_enabled`. Per tool call, `debug_questions.jsonl` includes the
selected `query_text_preview` + `query_arm`, the **candidate query list before
selection** with each candidate's `expected_search_quality`, the selected
`query_quality`, and the **dropped-candidate count** (with reasons in the query
plan). `scripts/debug_run_failures.py` prints the candidate list and marks the
selected query.

**Sequencing:** evaluate Firecrawl **scrape/fetch only after query quality
improves** — scraping a page that better targeting would not have surfaced just
adds cost. Query decomposition (v1) comes first.

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
