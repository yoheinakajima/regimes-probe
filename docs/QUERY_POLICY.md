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

## Level 4: task-frame / constraint-graph search (constraint satisfaction over latent variables)

**Motivation.** Levels 1–3 treat a question as a bag of clues and high-frequency
entities. But a BrowseComp-style question is a **constraint-satisfaction problem
over latent variables**: it describes a *target answer* (one hidden variable of a
known type) plus several *intermediate* hidden variables, with *constraints*
linking them ("a restaurant **in New Mexico** founded **by a chef** born **in which
year**"). The bottleneck on hard items is not provider quality or scrape mechanics —
it is **task-state representation**. Without an explicit model of *which hidden
variable each search/read is meant to resolve*, the agent chases the most frequent
proper noun and never tests the discriminative constraints. Level 4 makes that
state first-class and uses it to drive search, candidate selection, reading, and
stopping. It is **generic** (no per-item or per-topic rules) and gated behind
`policy.enable_task_frame` (default **off**, so the synthetic demo is byte-identical).

**Task frame** (`agent/task_frame.py:parse_task_frame`, deterministic v0, gold-free):
- **Slots** — typed hidden variables. `target_answer_slots` (what the question asks
  for) vs `latent_slots` (intermediates). The target's role comes from the
  *interrogative head noun* ("which TV **series**" → `title_or_work`; "…in which
  **year**" → `date_or_time`; "**who**…" → `person`), so secondary entities ("an
  actor who…") become intermediates, not competing answers. Each `Slot` carries
  `slot_role`, `is_target_answer_slot`, `is_intermediate_slot`, `depends_on`,
  `expected_evidence_type`.
- **Constraints** — one per clause, typed (`identity`/`attribute`/`relation`/
  `temporal`/`location`/`distance`/`source`/`authorship`/`membership`/`title_work`/
  `answer_shape`), with `normalized_terms`, `applies_to` (slot ids),
  `specificity_score`/`discriminative_score`, and a `status`
  (`unresolved`/`resolved`/`contradicted`) tracking supporting/contradicting
  evidence.
- **Known context terms** — entities/quoted phrases/acronyms *given* in the question
  (e.g. **WHO** in "released by WHO"). These are constraints/context and are **never
  promoted into a target answer slot** — a guard against answering with a term the
  question handed you.
- `dependency_edges` chain intermediates → target; `parse_quality` scores whether we
  found a target plus at least one specific constraint.

**Hypothesis table + evidence ledger** (`agent/hypothesis_table.py`): a *hypothesis*
is a partial assignment of candidates to slots, scored by constraints
supported/contradicted and slot coverage (`confidence = support − 1.5·contradiction
+ 0.5·coverage`). Each result becomes an `EvidenceRecord` linked to the slots,
constraints, and candidates it touches, with an `evidence_progress_score`. Candidate
**role typing is strict and context-aware**: a bare multi-word proper noun cannot be
typed in isolation, so the surrounding evidence's role-trigger noun is trusted ("Casa
Verde **restaurant**" → organization; "**actor** Jane Doe" → person). Hypotheses are
**downweighted/deactivated after repeated no-progress** actions
(`repeated_no_progress`) and **rejected when contradicted** with no support.

**Action planner** (`agent/action_planner.py`): chooses the next *epistemic action*
over the frame — `answer_if_supported` (target bound and ≥2 supporting evidences),
`abstain_if_no_path`, `search_with_candidate_and_constraint`, `search_for_slot`
(targets the highest-specificity unresolved constraint), `read_url_for_constraint`,
`extract_candidate_from_evidence`. Queries are generated from slots + constraints and
de-duplicated by hash so the loop never repeats an equivalent query.

**Read-value gate over the frame** (`action_planner.frame_read_value`): a read fires
only with a frame-grounded reason — `supports_candidate_hypothesis`,
`contains_unresolved_clue`, `contains_answer_shape_hint`,
`cross_provider_same_url_or_domain`, or `structured_or_pdf_and_high_relevance`. It is
**rejected** for `benchmark_contaminated`, social, or — critically —
`no_unresolved_slot_or_constraint` (a generically authoritative page with an
insufficient snippet is *not* a sufficient reason on its own). The tested
constraint ids are recorded on the decision.

Debug (`debug_questions.jsonl`; `scripts/debug_run_failures.py`) records per attempt
the `task_frame` summary (targets/latent/constraints/known-context), `frame_coverage`
(`slot_resolution_rate`, `constraint_support_rate`, `target_slot_support_rate`,
`hypothesis_coverage_score`), the `hypothesis_summary` (`top_hypotheses` +
rejected), and per call the chosen `task_action` (kind) and `evidence_record`.
Metrics add `evidence_progress_per_action`, `read_value_precision`,
`no_progress_action_rate`, `repeated_equivalent_query_rate`,
`hypothesis_rejection_counts`, and `final_answer_supported_by_constraints_rate`.
Policy memory remains **answer-free**: only the route/query/verify/stop policy is
learned; no slot value, candidate, or answer text is ever written.

### Level 4b: optional LLM task-frame parser (cached, validated, fallback-safe)

The deterministic v0 parser proved the constraint-satisfaction architecture, but
**parse quality is the limiting factor**: on live BrowseComp it mis-types slots,
attaches constraints to the wrong slot, and promotes a source/context entity to a
target (e.g. searching "Infectious Diseases" university for the WHO report, or
binding Friends/Schwimmer too early on a TV-series item). The frame, planner, and
hypothesis machinery are sound — the *frame* is wrong.

`agent/llm_task_frame.py` adds an **optional** LLM parser (`--enable-llm-task-frame-parser`,
default off; requires `--enable-task-frame`). It produces task **state only** and
emits the **same `TaskFrame` schema** — there is **no new planner contract**; the
hypothesis table, evidence ledger, action planner, and read-value gate are
unchanged. Discipline mirrors the gated LLM query policy:

- **Answer-free / gold-free.** The parser receives only the question text. It must
  not receive or emit final-answer text; nothing it produces is written to policy
  memory. The pinned prompt (`prompts.py::task_frame_parser`) explicitly tells the
  model: *do not answer, do not solve, do not guess entities; only parse into slots
  and constraints; distinguish known context from unknown variables; the target is
  the thing asked for (not every entity); attach each clue to the slot it
  constrains; preserve multi-hop dependencies; do not promote a source/org to a
  target unless the question asks for it.* It carries generic, non-answering example
  shapes (TV-series/actor, restaurant/hotel/museum/founder/year, report/foreword/
  introduction author, paper/journal/census).
- **Cached + replayable.** Every call goes through a parser cache keyed by
  `prompt_fingerprint | model | question`. A replay (or any run with no injected
  `model_fn`) **never calls a model** — a cache miss falls back to the deterministic
  parser. The cache records `prompt_version`, `prompt_hash`, `model`, `input_hash`,
  `output_hash`, `parse_quality`, and `validation_errors`.
- **Schema-validated + fallback-safe.** `validate_payload` enforces: ≥1 target slot;
  the target role matches the interrogative head where detectable; constraints
  attach to existing slot ids; dependency edges reference existing slots; no
  unsupported role/constraint labels; no empty required fields; **known context
  terms are not also target slots**; and **no gold-like answer text** (a target slot
  naming a concrete entity absent from the question, or any forbidden `answer`/
  `final_answer`/`solution` key, is rejected). Any failure — invalid JSON, schema
  error, or aggregate `parse_quality` below the floor — records a `fallback_reason`
  and uses the deterministic frame. **The deterministic parser is always the
  default and the safety net.**

**Parse-quality components** (`score_parse_quality`, each in [0,1], aggregated):
`target_identification_score`, `constraint_attachment_score`, `dependency_score`,
`known_context_separation_score`, `slot_coverage_score`, `ambiguity_score`.

Debug (`task_frame_parse` block) and metrics add parser provenance:
`parser_used` (deterministic|llm), `prompt_version`/`prompt_hash`, `parse_quality`,
`fallback_reason`, `validation_errors`; aggregates `llm_task_frame_used_count`,
`llm_task_frame_fallback_count`, `task_frame_parse_quality_mean`,
`parser_validation_failure_counts`, `deterministic_fallback_rate`,
`target_slot_role_distribution`, `constraint_attachment_count`.

**Live model wiring + strict guardrail.** `--enable-llm-task-frame-parser` is a hard
error without `--enable-task-frame` (`run_live` exits non-zero *before* writing any
artifact; the same check lives in `build_agent`) — a user-requested flag is never
silently downgraded. When enabled, `run_live` injects a `model_fn`
(`providers.build_task_frame_model_fn`) that calls the parser model through the
**same `RecordingCache`** as the answerer: a dry-run (un-armed) raises `NotArmed`, a
replay miss raises `ReplayMiss`, and a successful call is recorded — so **dry-run and
replay never call the model**, and any failure is caught and falls back to the
deterministic parser (`fallback_reason=model_error:*`). The parser model defaults to
`--answer-model`; `--task-frame-parser-model` overrides it. The model parses the
question into task state only — it is **never** asked to answer. Call accounting
(answer-free) is reported: `parser_model_calls_estimated` (dry-run),
`parser_model_calls`, `parser_cache_hits`, `parser_cache_misses`,
`parser_fallback_count`, `task_frame_parser_model`.

**Answer-support gate.** A final answer counts as constraint-supported
(`final_answer_supported_by_constraints` / `answer_support_gate`) only via
`action_planner.evaluate_answer_support`, which requires ALL of: a target-slot
candidate exists; **non-contaminated** evidence *tied to the selected hypothesis*
supports the target slot or a found answer-shape hint; the target's
most-discriminative constraint(s) are resolved (not unresolved); no constraint on a
bound slot is contradicted; and the hypothesis carries ≥2 supported constraints.
This closes the earlier bug where a bound slot alone reported `answer_supported=True`
with no correct answer. When the gate fails, `missing_support_reasons` lists exactly
which checks failed. The blocking check now consumes the **affordance** model below:
a constraint with `can_block_answer` (or `blocks_answer_if_unresolved`) on a bound
slot must be resolved before answering — falling back to the most-discriminative
target constraint only when the frame declares no explicit blocking constraint.

### Level 4c: open-world semantic constraints + operational affordances

**Why.** The first wired LLM-parser run fell back constantly because the schema
required a narrow `constraint_type` enum, while the model produced rich, *defensible*
labels — `employment_relation`, `authorship + educational background`,
`distance_and_temporal_attribute`, `source_reference`. Rejecting those is both wrong
(the labels are useful) and a path to **overfitting to BrowseComp** (we would be
hand-maintaining an enum that chases one benchmark's phrasing). The real need is not
a fixed vocabulary; it is that every parsed object be *operationally usable*.

**The split (raw semantics vs. derived affordances).** Constraints are now
**open-world**: `semantic_label` is free-form, `semantic_facets` is an open list
(common ones — identity/attribute/relation/temporal/spatial/quantitative/authorship/
source/membership/biographical/title_or_work/comparison/distance/answer_shape — are
lightly standardized but a *new* facet is preserved, never rejected). What the
planner branches on is a small **closed** set of **operational affordances**
(`agent/affordances.py`): `can_search`, `can_verify`, `can_read`, `can_compare`,
`can_bind_slot`, `can_support_answer`, `can_block_answer`. Affordances are emitted by
the parser and/or derived by generic heuristics from facets + slot roles + fields —
e.g. `employment_relation` → search/verify/bind/support; `distance_and_temporal_*`
→ + compare; `source_reference` → + read. Each constraint also carries
`required`, `priority`, `testable_claim`, `evidence_needed`, `how_to_test`,
`suggested_query_templates`, `supports_answer_slot_ids`, `blocks_answer_if_unresolved`,
`parser_confidence`, plus `raw_parser_output` / `planner_interpretation` /
`validation_warnings` (raw and derived are kept side by side, never collapsed).

**Validation is now open-world.** `validate_payload` checks *usability + safety*,
not vocabulary: JSON well-formed; `applies_to` (scalar normalized to a list)
references real slots; `required` constraints have a `testable_claim` plus
`evidence_needed`/`how_to_test` and apply to a target/intermediate slot; no
gold/answer text. An unfamiliar `semantic_label`, facet, or slot role is **never** a
hard rejection — it becomes a `parser_warning` (unknown slot roles coerce to
`unknown`; unknown affordances are dropped; novel facets are flagged but kept). The
deterministic parser enriches its constraints through the **same** path, so both
parsers expose identical operational fields.

**The planner branches on affordances, not labels** (`ActionPlanner.plan`): it orders
unresolved constraints by `(blocks, supports, priority, specificity)` and picks an
action by affordance — `search_to_bind_slot` (`can_bind_slot`),
`search_to_test_constraint` (`can_verify` on a bound target), `compare_candidates`
(`can_compare` with competing candidates), `read_to_verify_constraint` (the
frame-grounded read gate), `answer_if_supported`, `abstain_if_blocked`. It never
switches on a free-form label, so a new label/facet needs **no planner change**.

### Level 4d: epistemic escalation controller (front door)

Running the constraint-graph machinery on every question is wasteful. A lightweight
controller (`agent/epistemic_mode.py`, deterministic, gold-free) reads cheap signals
— clause count (hops), entity density, multi-hop connectives, recency cues, budget —
and picks the cheapest sufficient mode: `direct_answer_possible` → `simple_lookup` →
`decomposed_search` → `iterative_research` → `task_frame_required`. Easy factual
questions ("capital of France") route to direct/simple; clue-dense multi-hop
questions (most BrowseComp items, *emergently* not by hard-coding) route to
`task_frame_required`. Flags: `--auto-epistemic-mode` (let it decide per question;
default off so the benchmark uses explicit flags), `--force-task-frame`,
`--disable-direct-answer`. The decision (`selected_epistemic_mode`,
`escalation_reason`, `skipped_heavy_parser_reason`, `estimated_effort`, `signals`) is
recorded on every attempt for audit.

### Level 4e: variables vs. constants vs. bindings (don't reject descriptors)

**Why.** Once the open-world parser worked, a *good* frame still fell back: the
TV-series question parsed to a target slot named "90s TV series"
(role `title_or_work`, with the actor clues as constraints) — and the validator
rejected it with `known_context_promoted_to_target` because the slot name overlapped
the question text. That is the wrong test. "90s TV series" is not a leaked answer; it
is a **descriptor of the unknown target variable**. Target slot names *normally* come
from the question. Rejecting overlap conflates a **variable descriptor** with a
**bound value**.

**The distinction (generic, no phrase allowlist).** Slots carry an explicit binding
lifecycle (`agent/task_frame.py::SLOT_STATUSES`): `unbound_variable` →
`candidate_binding` → `derived_value`, plus `known_constant` for values *given* in
the question. A slot also carries `descriptor_text`, `bound_value`,
`known_context_refs`, `evidence_required_to_bind`, and `parser_confidence`. At parse
time every target/intermediate slot is an `unbound_variable` (candidates only appear
later, from evidence). `infer_slot_status` fills the status when the parser omits it.

**The rewritten check** (`classify_target_slot`) replaces string-overlap with a
variable/constant test. A target slot **fails** only when it is a *premature binding*
(`bound_value` populated, or `slot_status` already `known_constant`/`candidate_binding`)
or a *concrete known constant*: a Title-Case proper-noun entity (`_is_named_entity`:
"World Health Organisation", "Tennessee", "Gracie Award", "New Mexico") whose role
does **not** match the interrogative head **and** that has **no binding constraints**
(nothing to find — it is already given). A slot **passes** (`unbound_variable_descriptor`)
when its role matches the question's head, OR it has binding constraints, OR it reads
as a descriptor (a generic type/role head or relational language — `_is_descriptor`,
which is False for a bare proper-noun entity). Genuinely ambiguous cases produce a
`validation_warning`, never a fallback. Outcomes: "90s TV series"/"founder full name"/
"person who wrote the introduction"/"global report released by WHO" → **pass**;
"World Health Organisation" (asked for a person) / "Tennessee" (asked for a series)
→ **fail**. Reusing question wording in a target slot name is **not leakage and not
overfitting** — it is how you describe an unknown.

**Answer support is unchanged** (Level 4b): a descriptor alone never implies support.
`answer_supported` still requires an evidence-backed candidate binding for the target
slot, non-contaminated evidence, required blocking constraints satisfied, and the full
answer → hypothesis → slot → evidence → constraint path.

## Level 5: candidate slates + frontier scheduling (possible mid-hop answers)

**Why.** A human solving a hard multi-hop question does not carry one global "best
candidate." They keep a **candidate slate per unresolved variable** — possible
hotels, possible museums, possible founders, possible birth years — test each
candidate against *that slot's* constraints, reject bad ones, expand promising ones
to dependent slots, and only answer from a supported hypothesis. `agent/candidate_frontier.py`
makes that pattern first-class. It is generic (no fixed slot names / constraint
labels / gold answers), **skipped** for direct/simple epistemic modes (Level 4d), and
**ActiveGraph-native** (see `ACTIVEGRAPH_DESIGN.md`): every candidate, status change,
merge, promotion, evidence link, and frontier decision is an event with a
deterministic id, projected to `graph_projection.json`, so the slates are replayable,
forkable, and learnable from traces — not a sidecar that only lives in the loop.

**Candidate slates** (`CandidateSlate`, one per slot that needs binding). A
`SlotCandidate` is a *per-slot assignment* (the same entity proposed for two slots is
two assignments with their own score/status), carrying provenance (evidence/action
ids, source domains), `constraints_supported`/`contradicted`/`unknown`, an
`evidence_score`, and a lifecycle status. A candidate is **not** globally selected
just because it appears often; it competes inside its slate. Known-context constants
keep their isolated role and are admitted only when evidence binds them to a
role-compatible slot (a given location never becomes a restaurant candidate).

**Status lifecycle** (`active → rejected | confirmed | merged | stale`, all evented).
**Reject** on `contradicted_constraint` (e.g. a constraint anchored "opened in 1955"
vs. evidence "opened in 1972"), `repeated_no_progress`, `known_context_not_candidate`,
`contaminated_source`, duplicate, etc. **Confirm/promote** when the slot's required
blocking constraints are supported by non-contaminated evidence above threshold and
nothing is contradicted. **Merge** duplicates (same normalized text / cross-provider
identity), preserving provenance from both.

**Hypotheses are combinations** of slot candidates (`FrontierHypothesis`: a partial
slot→candidate assignment) scored by slot coverage, required-constraint coverage,
**source diversity**, minus contradictions / no-progress / contaminated evidence,
preferring hypotheses that unlock target-answer slots.

**Frontier scheduler.** Each evidence update re-generates `FrontierAction`s
(`generate_candidates_for_slot`, `verify_candidate_constraint`,
`expand_candidate_to_dependent_slot`, `compare_candidates_for_slot`,
`read_candidate_source`, `promote_candidate_to_confirmed`,
`answer_from_confirmed_hypothesis`, `abstain_no_viable_hypothesis`, …) each with an
`expected_information_gain` and `estimated_cost`. `select_frontier_action` picks by
**expected information gain net of cost** — resolving high-priority blocking
constraints first, binding upstream slots before downstream target answers,
**cheap verification before expensive reads**, and distinguishing competing
candidates — and avoids repeating equivalent tests, expanding failed candidates, or
searching broad known-context terms. A **read** must name a `(candidate, slot,
constraint)` triple it would affect; otherwise it is rejected
(`no_candidate_slot_constraint_affected`), along with contaminated/no-progress/generic
URLs. The answer-support gate is unchanged — slates feed it, they do not weaken it.

### Level 5b: frontier controller (optionally drives tool selection)

Level 5 (above) is **shadow/projection** by default: the candidate-slate frontier
records its recommended next action every step, but the Level-4 `ActionPlanner` still
drives the actual tool calls. `--enable-frontier-controller` (requires
`--enable-task-frame`; a hard config error otherwise) **promotes** the frontier from
shadow to controller: `FrontierScheduler.propose_step_action` selects the next action
by expected information gain and **translates** it into the executed step —
`generate_candidates_for_slot`/`verify_candidate_constraint`/`compare_candidates_for_slot`/
`expand_candidate_to_dependent_slot` → a search query built from the slot's most
**discriminative** blocking constraint (never a bare repeat of the target descriptor —
e.g. it opens the TV-series frame on "actor born Tennessee", not `"90s TV series" 90s
TV series`); `read_candidate_source` → a page read **only** when a `(candidate, slot,
constraint)` URL can be resolved from the observations; `answer_from_confirmed_hypothesis`/
`abstain_no_viable_hypothesis` → terminal. **Every executed tool call links to a
`frontier_action` id** (`CallRecord.frontier_action_id`, `tool_call_from_frontier_action`
edge). If a selected action cannot be executed safely it records
`frontier_action_unexecutable` and **falls back to the old planner for that step**, so
the controller can never get stuck. Shadow mode (default) records, per step,
`frontier_recommended_action` / `planner_actual_action` / `action_agreement`; active
mode records `frontier_controller_used`, `old_planner_fallback_count`,
`frontier_action_execution_success/failure_count`, and `tool_calls_from_frontier_actions`.
Easy questions bypass the whole layer via the escalation controller (Level 4d).
