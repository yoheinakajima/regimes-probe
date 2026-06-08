# Methodology risks (read before believing any number)

This project is easy to fool yourself with. This document lists the ways a
result here could be *misleading* and the controls that defend against each. It
is deliberately skeptical. Pair it with `docs/STATUS.md` (the claim ledger) and
`docs/LEAKAGE_CONTROLS.md`.

## The single most important caveat

**The synthetic result is proof of the harness, not of the idea.**
`results/demo/` shows that *given a fixture engineered so the right tool/query/
stop choice changes the outcome*, the contextual bandit learns those choices and
`correct_per_tool_call` rises. That validates the **mechanism and plumbing**. It
says **nothing** about whether the same mechanism helps on real browsing
questions. No BrowseComp/LiveBrowseComp claim follows from it.

## Specific risks

### 1. The random_memory control sitting "in between" is ambiguous
On the synthetic set, ordering is `no_memory < random_memory < policy_memory`.
That is the *hoped-for* ordering, but a non-trivial `random_memory` score can
mean either of two very different things:
- **Benign:** random priors occasionally route to the right tool by luck, and the
  fixture is simply learnable — `policy_memory` ≫ `random_memory` is the real
  signal.
- **Concerning:** the *structure* of having any per-signature priors leaks
  exploitable information (e.g., the fixture is so easy that almost any bias
  helps), inflating both memory conditions.

**Defense:** require `policy_memory` to beat `random_memory` by a margin that
exceeds the bootstrap CI, on **held-out CONFIRM**, on real data. If
`random_memory ≈ policy_memory`, treat the signal as not established.

### 2. The fixture may be too easy
The synthetic corpus gives exactly one gold tool/arm per item and distinct
distractors, so retrieving the gold document anywhere yields the right answer.
Real retrieval is noisier, answers are contested, and freshness/authority matter
more. Easy fixtures over-state learnability.

**Defense:** the real benchmarks are the test; the fixture only gates the harness.

### 3. Intrinsic knowledge (the model already knows the answer)
If the answerer can answer without searching, "tool routing" is irrelevant and
any apparent gain is noise.

**Defense (required for real runs):** run a **closed-book** baseline (no tools)
and a **no-search** baseline. LiveBrowseComp is primary precisely because recent
facts reduce intrinsic-knowledge dependence. Report the closed-book number
alongside every result.

### 4. Conditions differing in more than the independent variable
If `no_memory` and `policy_memory` differ in model, prompt, tool config, budget,
or grading, the comparison is invalid.

**Defense:** identical model / tools / answer prompt / judge / budget across
conditions; the **only** difference is the frozen policy memory. Pin exact model
ids in `report.json` meta.

### 5. Single-budget cherry-picking
A method can win at one budget and lose elsewhere.

**Defense:** report the **budget curve** (`[1,3,5,10]`), not one cap.

### 6. Answer leakage into "procedural" memory
The whole premise collapses if benchmark answers leak into memory.

**Defense:** policy memory stores rewards/priors/embeddings/hashes only;
`assert_no_answer_leakage` runs on every object and the frozen snapshot;
`tests/test_policy_memory_no_answer_leakage.py` checks no gold string appears.
On real data, **re-run this check on the actual traces** and grep the committed
snapshot.

### 7. Non-determinism / cache effects masquerading as learning
Live providers are non-deterministic; repeated runs and provider caches can move
numbers independent of the policy.

**Defense:** record every tool response in the event log; verify the replay
check passes; report repeated-run variance; note cache risks in the report.

### 7b. Provider/API failures during tool calls
Real providers return errors (HTTP 400/429/5xx, timeouts). These must not crash a
run, but they also must not be swept under the rug.

**Defense:** a tool-call error becomes a *recorded* failed observation (sanitized,
no secrets), earns no evidence gain, and penalizes that tool/query arm so the
bandit can learn the tool is unreliable in that context. Per-tool failure counts
go to report.json / summary.md / tool_rewards.csv, and a high overall failure rate
(> headline.max_provider_failure_rate, default 0.2) or an entirely-failed required
condition makes the run not headline-eligible — even though it stays structurally
valid. If a provider is failing often, the comparison is degraded; fix the adapter
or drop that arm before claiming anything.

### 8. OPTIMIZE leaking into the headline
In-sample improvement is nearly free and meaningless on its own.

**Defense:** OPTIMIZE never appears as a headline; **CONFIRM is the only
headline**, evaluated against a frozen snapshot that was frozen *before* CONFIRM.

### 9. Grading artifacts
Normalized exact-match can both over- and under-count; an LLM judge introduces
its own bias and non-determinism.

**Defense:** log the grading method per item; if using an LLM judge, cache and
log it; report exact-match and judged numbers separately.

### 10. Multiple comparisons / seed mining
Trying many configs/seeds and reporting the best is p-hacking.

**Defense:** pre-register the conditions; report all seeds with mean ± CI;
promotions in the regimes loop go through OPTIMIZE→CONFIRM gating and are all
recorded (accepted *and* rejected).

## Minimum bar for a credible real result

A learned-policy improvement is reportable only if **all** hold:
1. Same model / tools / prompt / budget across conditions.
2. Closed-book and no-search baselines reported.
3. `policy_memory` beats `random_memory` and `no_memory` on **CONFIRM** beyond
   the bootstrap CI.
4. The improvement holds across the budget curve.
5. No-answer-leakage check passes on the real traces, and the committed snapshot
   contains no answer text.
6. The replay check passes.
7. Only CONFIRM is used as the headline; OPTIMIZE is disclosed but not claimed.
8. Limitations and "what is not claimed" are stated in `summary.md`.

Until then, the honest statement is: **the scaffold demonstrates the intended
mechanism on a synthetic fixture.**

## 11. Tool-family scope creep (Monid / Firecrawl / Wokelo / browser-use)

Adding agentic, scrape, specialized-research, and browser tools widens the arm
set — and the risk surface. Keep the experiment honest:

- **browser-use is deferred on purpose.** A browser-control agent turns BrowseComp
  search-routing into open-ended browser automation, and introduces
  **prompt-injection and state-mutation** risks (a malicious page can steer the
  agent). It is NOT implemented as a live adapter in v0; it is registry/docs
  scaffold only.
- **Stateful/paid tools (`monid_run`) and browser-like tools (`firecrawl_interact`)
  are off by default** and require explicit allow flags. They can take
  side-effecting/paid actions, which breaks the "read-only evidence acquisition"
  framing of the benchmark — exclude them from headline runs.
- **Agentic discovery (Monid) is a *different* capability** from search. Treat it as
  a distinct arm/ablation ("does a learned tool-discovery policy help?"), not as a
  search provider; don't let it quietly inflate or deflate the search-routing result.
- **Scrape (Firecrawl) changes evidence quality**, not just routing. Full-page
  markdown can raise correctness independent of the *policy* — so compare like with
  like (same tool set across conditions; the same-conditions validator enforces
  this) and report scrape on/off as an explicit ablation.
- **Wokelo is unverified here** (JS-rendered docs); the adapter fails closed until
  the official endpoint shape/OpenAPI is supplied. Do not guess endpoints.
- **First BrowseComp runs stay cheap search-only.** Add families one at a time,
  after a baseline works, so any change in `correct_per_tool_call` is attributable.
- **Debug artifacts contain gold-answer previews — policy memory does not.**
  `debug_questions.jsonl` is an audit/debug artifact and deliberately carries
  bounded gold/prediction previews for the operator's eyes; this is **not**
  policy-memory leakage. Only `memory_snapshot_leakage_pass` (frozen policy
  memory) gates eligibility; the raw audit trace may contain gold by design. Do
  not conflate the two layers (`LEAKAGE_CONTROLS.md`).
- **Fetch/scrape tools are follow-up, not first-hop.** `page_fetch` (and
  `firecrawl_scrape`) operate on a URL from evidence; they must never be routed as
  a first-hop *search* arm. A real BrowseComp debug run that listed `page_fetch`
  as a search arm called it with the question text and failed 78/78
  (`provider_failure_rate=0.30`, blocking eligibility). The router now routes only
  over `FIRST_HOP_FAMILIES` (search/specialized/discovery) and uses follow-up
  tools only after a search returns URLs; a non-URL `page_fetch` call fails
  gracefully with `requires_url`. When auditing a run, check that
  `provider_failure_rate` is not dominated by one tool family.
- **Searching the whole BrowseComp prompt is a trap.** The first clean BrowseComp
  run was structurally headline-eligible but scored **0 accuracy**: search worked,
  but feeding the entire long clue-dense question as one query surfaced
  spam/benchmark-mirroring pages, so the exact answer was never in the snippets
  (dominant seam: `exact_answer_missing`). Level 1 routing alone cannot fix this —
  **Level 2 query decomposition** (multiple targeted clue queries, query form as a
  bandit arm) is required first. Evaluate Firecrawl **scrape** only *after* search
  targeting improves: scraping the wrong (contaminated) page just adds cost.
- **Benchmark contamination inflates apparent relevance.** Pages that mirror the
  benchmark (HF/GitHub/arXiv/simple-evals, or snippets reproducing the question)
  look relevant but carry no independent evidence. They are detected
  (`eval/contamination.py`), penalized in reward, and reported
  (`contamination_rate`, per-domain/provider). Watch `contamination_rate` on any
  real run — a high rate means the query form is fetching the benchmark, not
  evidence.
- **Debug artifacts contain a gold-in-result audit flag.** `debug_questions.jsonl`
  marks whether a result contained the exact gold answer — an audit aid in the
  debug layer only; policy memory remains answer-free.
- **Offline forks are ablations, not measurements.** A forked run reuses a
  parent's cached outcomes and re-weights/re-routes over already-observed data;
  it can reveal which variant *would* have done better on the same evidence, but
  it is `offline_fork=true` and never headline-eligible. A fork that would change
  the realized query distribution refuses (cache miss) rather than spending — so
  do not read a "successful" fork as evidence the new policy generalizes.
- **Sticky wrong-candidate exploitation in staged search.** Iterative clue
  resolution can latch onto a high-frequency but low-progress intermediate
  (a publisher like `Brittle Paper`, a broad org like `World Health Organization`,
  a broad location like `Tennessee`, a concept like `Art Deco`) and re-query it.
  This is a hypothesis-selection failure, not a provider or contamination failure.
  The candidate-hypothesis policy types each candidate by role, requires role
  compatibility with the inferred target and **evidence progress** before carrying
  it forward, and uses an anti-sticky beam (force exploration after two no-progress
  follow-ups). When auditing a run, watch `candidate_role_match_rate` (should be
  high), `sticky_candidate_count` / `no_progress_followup_count` (should be low),
  and `repeated_candidate_query_count` (should be ~0). This is a generic multi-step
  search rule, not BrowseComp tuning.
- **Scrape reads wrong pages → richer wrong evidence.** Firecrawl scrape produces
  much more text than a snippet, which can *look* authoritative while being the
  wrong page. It is therefore Level 3 (after search + candidate targeting), URL-only
  (never a first-hop arm), and gated: contaminated/social/no-progress/no-clue-match
  URLs are not scraped, and every read counts against budget. Firecrawl is also
  **paid / quota-limited**: 402/quota/HTTP errors are captured as provider failures
  (not crashes) and fall back to the cheap `page_fetch`. Watch `scrape_to_answer_rate`
  and `evidence_added_by_scrape_rate` — if scraping does not raise answer rate, it is
  just adding cost. Compare scrape on/off as an explicit ablation (same tool set
  otherwise), since richer extraction can change correctness independent of policy.
- **Task-state representation is the real bottleneck (not providers/contamination/
  scrape).** A BrowseComp-style question is constraint satisfaction over latent
  variables — a typed target answer plus intermediate hidden variables linked by
  constraints. Without an explicit task frame, an agent cannot say *which hidden
  variable a given search/read is meant to resolve*, so it chases the most frequent
  entity and never tests the discriminative constraints. The Level-4 task-frame
  layer (`agent/task_frame.py` + `hypothesis_table.py` + `action_planner.py`,
  `QUERY_POLICY.md`) is the **generic** fix: typed slots/constraints, a hypothesis
  table scored by constraints supported/contradicted, and a frame-grounded
  read-value gate. Risks to watch: (a) the v0 parser is heuristic and deterministic —
  it can mis-type a slot or mis-attach a constraint, so `parse_quality` and
  `slot_resolution_rate` are audit signals, not guarantees, and an LLM parser
  (cached/prompt-versioned) is the planned upgrade; (b) **do not encode per-item or
  per-topic rules** into the parser — the slot roles, constraint types, and the
  interrogative-head-noun target rule must stay generic, or you are overfitting the
  benchmark; (c) the layer must stay **answer-free in policy memory** — slot values,
  candidates, and answer text are never written (only the route/query/verify/stop
  policy is learned), enforced by `assert_no_answer_leakage`; (d) known-context terms
  given in the question (e.g. an org named in the prompt) must **never** be promoted
  into a target answer slot. Watch `read_value_precision`, `no_progress_action_rate`,
  `repeated_equivalent_query_rate`, and `final_answer_supported_by_constraints_rate`.
- **The LLM task-frame parser must produce task STATE, not answers.** Letting a model
  parse the frame (`--enable-llm-task-frame-parser`) is the natural fix for the v0
  parser's quality ceiling, but it reintroduces every LLM risk and one new one:
  *the parser could quietly answer the question* (put the gold entity into a target
  slot) and the agent would then "find" its own guess. Controls: (a) the prompt
  forbids answering/solving/guessing; (b) `validate_payload` rejects a target slot
  naming a concrete entity absent from the question and any `answer`/`final_answer`/
  `solution` key, and rejects known-context-as-target — on any failure the
  **deterministic parser is used instead** (`fallback_reason` recorded); (c) the
  parser is **cached + replayable** (keyed by prompt/model/question) so a replay
  never calls a model and numbers are reproducible — but note the new
  non-determinism risk at parse time, so log `prompt_version`/`prompt_hash`/`model`
  and report `deterministic_fallback_rate`; (d) it produces only task state — the
  answerer/grader/judge prompts and the answer path are unchanged, and nothing the
  parser emits is written to policy memory. A "win" from the LLM parser is only
  credible if `task_frame_parse_quality_mean` and frame-grounded action rates rise
  **and** the closed-book baseline did not already know the answer (the parser must
  not be smuggling intrinsic knowledge into the frame). The parser model is wired
  through the **same `RecordingCache`** as the answerer (dry-run/replay never call
  it), defaults to `--answer-model`, and its spend is reported separately
  (`parser_model_calls`, `parser_cache_hits/misses`, `parser_fallback_count`) so a
  "win" cannot hide extra model calls. Flags fail closed:
  `--enable-llm-task-frame-parser` without `--enable-task-frame` is a hard error (no
  silent downgrade), so a run never quietly does something other than what was asked.
- **"answer_supported" must mean supported, not "a slot is filled."** An earlier
  task-frame run reported `target_sup=1.0` / `answer_supported=True` while producing
  no correct answer — the flag was satisfied by a *bound slot*, not by evidence. That
  is exactly the kind of self-deception this doc exists to catch: a green internal
  signal that does not track correctness. The fix is a strict gate
  (`evaluate_answer_support`) requiring non-contaminated evidence tied to the
  hypothesis to support the target, the target's discriminative constraint resolved,
  nothing contradicted, and ≥2 supported constraints; failures are itemized in
  `missing_support_reasons`. Watch for the inverse failure too: if `answer_support_gate`
  is frequently True but accuracy stays low, the gate is being met by the *wrong*
  hypothesis (a parse-quality / constraint-resolution problem), and the gate's
  `True` rate must not be read as an accuracy proxy.
- **Open-world labels must not become a loophole.** Moving from a fixed
  `constraint_type` enum to free-form `semantic_label`/`semantic_facets` (Level 4c)
  fixed brittle fallbacks and avoids overfitting an enum to BrowseComp — but
  "accept any label" must not mean "accept anything." The guard is that validation
  shifted from *vocabulary* to *operational usability*: a `required` constraint
  still must carry a `testable_claim` and an `evidence_needed`/`how_to_test`, still
  must attach to a real slot, and the gold-text / known-context-as-target checks are
  unchanged. The planner branches only on the **closed** affordance set, so a novel
  label cannot smuggle in new control flow. Risks to watch: (a) a parser could
  emit an affordance (e.g. `can_support_answer`) that the evidence does not justify —
  affordances are *capabilities*, not *outcomes*; the answer-support gate still
  requires real evidence, so an over-claimed affordance cannot by itself produce a
  supported answer; (b) `parser_confidence` and self-declared `priority` are model
  outputs and must not be trusted as ground truth — treat them as hints and audit
  `validation_warnings`; (c) facet sprawl — if every question invents new facets,
  the ontology is not converging; track novel-facet rate over a run and fold the
  recurring ones into the standardized list deliberately, from traces, not ad hoc.
- **Escalation can skip work that was actually needed.** The epistemic controller
  (`--auto-epistemic-mode`) routes easy questions away from the task frame to save
  budget. The failure mode is a *mis-routed* hard question answered shallowly. It is
  OFF by default (the benchmark uses explicit flags so all items get the same
  machinery and conditions stay comparable); when on, audit
  `selected_epistemic_mode` vs. correctness and watch for items that were routed to
  direct/simple but should have escalated. Never read "cheaper" as "better" without
  checking the easy/hard split held.
