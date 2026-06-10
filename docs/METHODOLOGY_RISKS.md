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
- **Ingestion bugs masquerade as model failure.** The first open-world live preview
  fell back to the deterministic parser with `validation_failed`, which *looked* like
  the model produced a bad frame — but the frame was semantically good; the harness
  was rejecting the model's own slot-id namespace (`T1`/`I1`) and dict-form
  `dependency_edges`. The lesson: when a parse falls back, read the raw output and the
  `validation_errors`/`id_mapping` before blaming the model (the preview now surfaces
  all of these). The remap (raw ids → internal `s*`) must stay **total and
  consistent** across every reference — a partial remap would silently drop a
  constraint's slot link and let the answer-support gate pass on an incomplete frame.
  Guard: `id_mapping` is recorded and `raw_slot_id` is kept on every slot so a
  reviewer can re-derive the mapping from the trace; references that do not survive
  remap (genuinely unknown ids) still fall back rather than being quietly dropped.
- **A validator can be wrong in BOTH directions; don't only guard leakage.** The
  known-context-as-target check originally rejected any target slot whose name
  overlapped the question — which *looks* like a prudent anti-leakage guard but
  silently discarded good frames (a target named "90s TV series" is a descriptor of
  the unknown, not a leaked answer). Over-rejection is as harmful as under-rejection:
  it hides real parser quality behind deterministic fallbacks and makes the LLM
  parser look useless. The fix is a **variable/constant/binding** distinction
  (`classify_target_slot`): reject only a *premature binding* (`bound_value` set, or a
  Title-Case constant whose role mismatches the question and that has no binding
  constraints), accept *descriptor variables* (type/relational language, or role
  matches the interrogative head, or it has constraints), and *warn* (never fall back)
  on the ambiguous middle. Two failure modes to keep watching: (a) the parser
  smuggling an answer into a target slot as a `bound_value` or a specific named
  entity — that must still fail (it is real leakage); (b) the validator drifting back
  toward string matching — track the `known_context_promoted_to_target` reason
  distribution and the `ambiguous_target_descriptor` warning rate, and confirm the
  reasons are `premature_bound_value`/`concrete_known_constant`, not descriptor
  overlap. The answer-support gate is unchanged and still evidence-based, so a
  descriptor that passes validation cannot by itself produce a supported answer.
- **Candidate slates can manufacture false confidence.** Maintaining possible
  mid-hop answers per slot (Level 5) is the right multi-hop pattern, but it adds
  surface area for self-deception: (a) a candidate could be *confirmed* on weak
  evidence — confirmation requires the slot's REQUIRED blocking constraints supported
  by **non-contaminated** evidence above threshold and nothing contradicted, and the
  final answer-support gate is unchanged, so a confirmed slate candidate still cannot
  by itself produce a supported answer; (b) the frontier scheduler optimizes
  *expected information gain*, which is a heuristic, not truth — watch
  `read_on_candidate_rate` and `frontier_expected_gain_mean` vs. actual evidence
  progress, and treat a high gain estimate with low realized progress as a mis-scored
  frontier, not success; (c) over-merging distinct entities into one candidate hides
  alternatives — merges require same normalized text / cross-provider identity and
  preserve both provenances, and `candidate_merge_rate` should stay low; (d) the layer
  is heavy and must be **skipped** for easy questions (the escalation controller gates
  it) — if `skipped_candidate_slate_count` is ~0 on a mixed set, the agent is forcing
  every question into research machinery. Because every candidate/status/merge/
  promotion/frontier decision is an event in the projection, these are auditable from
  traces rather than asserted; contamination still poisons confirmation, and nothing
  from the slate layer is written to answer-free policy memory.
- **Handing control to the frontier can quietly change the comparison.** Promoting the
  frontier from shadow to **controller** (`--enable-frontier-controller`) changes which
  queries/reads run, so a controller-vs-shadow A/B is only fair if everything else is
  pinned (same frame, parser, tools, budget, answerer) — the controller is OFF by
  default and shadow stays available precisely so the two are comparable. Risks: (a) a
  controller that *looks* better because it issues more (or cheaper) calls, not better
  ones — read `frontier_action_execution_success_rate` (success requires real evidence
  progress, not merely that a tool ran) and watch `tool_calls_from_frontier_actions`
  vs. accuracy; (b) the EIG heuristic is a *guess at value*, not truth — a high
  `first_action_discriminative_constraint_rate` is necessary but not sufficient, and a
  query that is discriminative-by-label can still retrieve nothing, so check realized
  evidence progress per selected action; (c) silent stalling — an action the frontier
  cannot execute records `frontier_action_unexecutable` and **falls back to the old
  planner** (counted by `frontier_fallback_to_old_planner_count`); a high fallback rate
  means the controller is not actually in control and the result is really the old
  planner's. The answer-support gate is unchanged, so the controller cannot make an
  unsupported answer "supported"; and every controller decision is event/trace-backed,
  so none of this is taken on faith.

- **Letting an LLM propose queries is the highest-leverage *and* highest-risk knob.** The
  Level-5c proposer (`--enable-llm-frontier-repair` / `--enable-llm-frontier-planner`)
  exists because deterministic query synthesis emits generic descriptors that retrieve
  junk; an LLM composes constraint-grounded queries far better. But an LLM in the loop is
  exactly where leakage, non-reproducibility, and silent condition drift creep in, so the
  design is **LLM-proposes / code-disposes** and the risks are pinned down concretely:
  (a) **answer leakage** — the model sees only a bounded `ResearchStateCard` (no gold, no
  secrets) and emits *actions*, never answers; an `answer_from_confirmed_hypothesis`
  proposal is rejected unless the deterministic answer-support gate already holds, and
  nothing the proposer sees or emits reaches policy memory (the existing no-answer-leakage
  test covers a run with the proposer on); (b) **the LLM smuggling in a generic or
  off-frame query** — every proposal is deterministically validated (real
  slot/constraint/candidate/hypothesis ids, non-generic, anchored to a constraint or
  known-context term, non-duplicate, allowed/affordable tool, connected to an *unresolved*
  slot/constraint) and rejected-with-reason otherwise, so the LLM cannot widen the action
  space, only propose within it — watch `proposal_rejection_counts` and
  `selected_query_constraint_anchor_rate`; (c) **non-reproducibility / hidden spend** — every
  call is cached by `prompt_fingerprint | model | card_hash`, a dry-run/replay makes **zero**
  model calls (a cache miss refuses rather than spends and falls back to the deterministic
  query), and temperature is 0, so a run replays byte-identically and an offline fork
  re-scores cached proposals for free; (d) **silently changing the comparison** — the
  proposer's `mode|model|prompt-fingerprint` is stamped into
  `ConditionSpec.llm_frontier_settings`, which the same-conditions validator requires to be
  **identical** across the `no_memory` / `policy_memory` arms, so "the LLM composed the
  queries differently between arms" is caught as an unexpected diff rather than mistaken for
  a memory effect. The honest claim this supports is mechanistic — "fewer generic queries,
  more anchored evidence-advancing calls" — not a headline accuracy delta; all of it is
  reconstructable from `graph_projection.json` (`llm_frontier_proposal` /
  `llm_frontier_validation` / `tool_call_from_llm_frontier_proposal`), so none of it is
  taken on faith. The whole layer is OFF by default; the synthetic demo never constructs the
  proposer, so the committed demo artifacts are unaffected.

- **A good proposal is worthless if it is mistranslated — and that failure is silent.** The
  first repair run made this concrete: the LLM proposed well, but the selected proposal's
  slot/constraints were dropped on the way to execution (repair kept the *deterministic*
  action and swapped only the query), so evidence linked to the wrong slot, support stayed
  at zero, and a correct candidate (`Cristina Ortiz`) was never promoted — a result that
  *looks* like "the LLM didn't help" but is really a plumbing bug. The lesson: an
  LLM-proposes/code-disposes design must prove the **disposal** is faithful, not just the
  proposal. The guardrails now do: the executed action is built from the proposal's own
  fields and materialized as `lfp_<proposal_id>`; a deterministic integrity gate refuses
  (and falls back) on any slot/constraint/query/evidence-link mismatch
  (`frontier_action_integrity_error`); and execution "success" is redefined to require
  progress on the *selected* slot/constraint, so an unrelated candidate can no longer make a
  wasted action look productive. Watch `llm_proposal_to_action_integrity_rate` (must be 1.0)
  and `candidate_promoted_from_llm_frontier_count`; a sub-1.0 integrity rate means the
  comparison is measuring a translation bug, not the LLM. Repair triggering was also too
  narrow (only one-word generics) — it now fires on repeated zero-progress, stale/noise
  candidates, and missing high-priority anchors, with the reason recorded, so "the
  deterministic query was bad but not *generic*" no longer slips through unrepaired.

- **Retrieval is not understanding — and raw n-grams quietly corrupt the state.** Before the
  interpreter, slates filled with page chrome ("Datasets", "Hugging Face", "Translate",
  "Login", "Merriam", "FOUNDER Definition") and constraint support rose from arbitrary term
  overlap, so a slate could *look* busy while containing nothing real — and a correct entity
  (`Cristina Ortiz`) could be present in results yet never promoted. The risk is twofold: (a)
  **false progress** — junk candidates and overlap-based support inflate evidence/hypothesis
  scores without any real finding; (b) **a BrowseComp-specific stoplist would be cheating** —
  hardcoding "reject huggingface.co" overfits the benchmark and hides the general problem.
  The defense is the generic interpreter (`agent/evidence_interpreter.py`): slates are
  populated **only** from accepted candidate assertions, constraint support changes **only**
  via a constraint assertion from a non-noise, role-compatible source, and rejection is by
  generic **source role + page intent** (a definition page defines terms; UI/nav text is not
  evidence; a contaminated page cannot support an answer) — not a domain list. Every
  acceptance/rejection is recorded with a reason and a quote and projected, so a reviewer can
  see *why* each slate entry exists. Watch `source_role_noise_rate`,
  `noise_candidate_rejection_rate`, `constraint_support_from_interpretation_rate`, and the
  debug-only `has_gold_but_not_promoted_count`; if support is rising without accepted
  assertions, or chrome is entering slates, the interpretation layer is the thing to fix, not
  the score. The interpreter is deterministic by default (no model dependence); the optional
  LLM source-role hook is cached/replayable and answer-free, so it cannot leak or
  de-reproduce a run.

- **Two candidate namespaces silently desync the system.** The first interpreter run made
  this concrete: the evidence interpreter extracted `Cristina Ortiz`, but the verifier — which
  knew candidates only by their `cand{N}` id — rejected a proposal naming `Cristina Ortiz` as
  `nonexistent_candidate`. A sidecar "assertion" layer that does not *materialize* the
  canonical candidate is worse than nothing: it looks productive while the rest of the system
  can't see its candidates, and evidence reaches slots (`ev→slot`) but never constraints
  (`ev→cons`) because support was computed by a different, overlap-only path. The defenses:
  an accepted assertion materializes/updates the **one** canonical `SlotCandidate` (with an
  `assertion_id → candidate_id` link and a per-slot text index), the verifier resolves a
  candidate by text *or* id across slots before declaring it nonexistent, and constraint
  support is attached from the interpreter's recognizers (acceptable source + role-compatible
  candidate + a real anchor) rather than recomputed from term overlap. Watch
  `canonical_candidate_resolution_rate` (≈1.0), `verifier_nonexistent_but_extracted_candidate_count`
  (0), and `evidence_to_constraint_link_rate`; a regression here means the layers have drifted
  apart again. And because permissive role typing (`unknown` → every slot) quietly poisons
  every slate, an unknown-role observation is now a recorded `weak_observation`, not a
  candidate — fan-out into all slates is itself treated as a bug.

- **Reads and a future LLM evidence judge can both inflate things — guard each.** Scheduling
  reads aggressively (forced after a no-support streak) closes body-only constraints but
  **inflates tool cost**, so read value must be judged by support yield *per read and per
  dollar*, not by read count; `read_starvation_count` / `support_from_read_rate` are
  diagnostics, not goals. The next layer — a full LLM **evidence judge** — is exactly where
  over-crediting creeps in: an LLM will happily call a directory heading "support". The
  defenses are non-negotiable and already enforced for the deterministic recognizers, and
  must hold for any judge: support stays **quote-backed, source-clean
  (non-contaminated/non-noise), candidate-specific and predicate-specific, replayable
  (cached, zero model calls on replay), and answer-free** — partial support never resolves a
  blocking constraint, it only moves EIG/scheduling. A **relaxed Answer Support Contract**,
  if ever enabled, is **not** equivalent to strict all-constraint support: it must be behind
  an explicit flag, record every downgraded constraint with a reason, report unsupported
  constraints on the answer, and be distinguished from strict in any headline claim. Default
  stays strict. Finally, **provider routing** learned from one run may not generalize —
  empty/contamination/support-yield rates are observations to score over many traces, never
  a hard-coded "provider X is best" rule.

- **An LLM evidence judge over-credits evidence unless every guard holds.** The Level-5f
  judge (`--enable-llm-evidence-judge`) decides whether a source excerpt supports a
  candidate/slot/constraint — exactly the place an LLM will happily call a directory heading
  or a title match "support". So the judge's output is **not trusted raw**: hard rules are
  enforced *after* the model (a contaminated/noise source can never be full/partial support;
  `full_support` requires a quote tying the candidate to the predicate; title overlap alone
  is downgraded; chrome never reaches it), and those rules are unit-tested against a stub
  that *tries* to over-credit. **Full support must stay quote-backed, source-clean,
  candidate-specific, and constraint-specific.** **Partial support is not answer support** —
  it moves EIG/scheduling only and can never resolve a blocking constraint, so the strict
  answer gate is unchanged and a partial-only candidate abstains. The judge is **cached and
  replayable** (keyed by `prompt_fingerprint | model | triple_hash`; replay makes zero model
  calls) — mandatory for auditability and to keep runs reproducible. And the judge **improves
  interpretation, not benchmark-claim eligibility**: it never answers, never sees gold, never
  writes to policy memory; the metrics `full_support_from_contaminated_source_count` and
  `partial_support_from_contaminated_source_count` are invariants pinned at 0. A run with the
  judge on is still gated by the same same-conditions + headline-eligibility checks as any
  other.

- **A reliable read path and a strict confirm gate are themselves correctness risks if
  fudged.** Level 5f hardens both. Reads must be URL-backed (a query is not a read), a
  zero-char fetch is a *failure* that falls back once to scrape (never counted as a success),
  and `read_requires_url_violation_count` is pinned at 0 — otherwise "we read the page" can
  silently mean "we fetched nothing". The confirm gate is tightened so a candidate is
  `confirmed` only with **all** blocking constraints on **full** (not partial) support, a
  **discriminative** blocking constraint among them, and a non-junk slot value — "plausible",
  "has some biographical matches", or "is a founder/designer of *some* entity" is explicitly
  not confirmation, and relational support needs the **exact** dependent object anchored.
  These are enforced *after* the LLM judge by deterministic rules and pinned-0 invariants, so
  an over-eager judge cannot promote a plausible-but-unproven candidate. The known limitation:
  these hardenings raise *precision* of support, which can lower apparent accuracy on hard
  items by abstaining more — that is the intended, honest trade, not a regression to fix by
  loosening the gate.

- **Level 5g — read-intent honesty and "prompt text is not evidence".** Two consistency
  hazards are closed in this layer. First, a loop that *says* it needs a read but only ever
  runs searches is dishonest: when a read is desired and the candidate has a clean,
  non-contaminated URL the read now executes against that URL, and a read whose candidate
  carries only a query string is recorded as `read_blocked_no_url` instead of silently
  becoming a search (`selected_read_action_translated_to_search_count` = 0). Second, and more
  dangerous for accuracy claims: the question itself states the constraints to be verified, so
  resolving a blocking constraint from the prompt's own wording would manufacture support out
  of thin air. A blocking constraint may therefore resolve only from a clean source carrying a
  distinctive anchor (a year, or a proper-cased term taken from the constraint's `text_span`) —
  a clean snippet that merely echoes generic question terms ("driving distance in miles") does
  not resolve it, and `initial_blocking_constraint_resolved_without_evidence_count` is pinned to
  0. Support is also made to materialize end to end
  (`full_support_judgment_without_materialized_constraint_support_count` = 0) so the answer gate
  and the judgment can never disagree. These are precision hardenings: like the 5f gates they
  can lower apparent accuracy by abstaining more, which is the intended trade, not a regression.
  Deferred (and explicitly not yet claimed): a generic `bind_target_answer_slot` action, a
  parser-fallback variable/constant classifier, exa/firecrawl alternate-URL read fallback, and
  projecting read-intent states as first-class graph objects.

- **Level 5h — closing the read→judge loop without manufacturing support.** 5g made reads
  execute; the danger 5h addresses is a loop that *looks* busy but never closes — reads that
  add no evidence, judges re-reading the same truncated snippet, and a target answer slot left
  unbound while the frontier re-verifies an already-supported subject. The honest hazards and
  their controls: (1) a read→judge **obligation** (`PendingReadJudgment`) is closed only by
  re-judging the *fetched page body* on targeted passages — never the snippet that created it
  and never only the document head (`judge_reused_truncated_excerpt_after_full_read_count` =
  `read_head_only_judgment_count` = 0); an unresolved read stays open with an explicit reason,
  it is not laundered into a vague new candidate. (2) **Target binding does not fabricate an
  answer**: `bind_target_answer_slot` only retrieves/reads/binds answer-shaped candidates from
  the supported subject; the strict answer-support gate is unchanged, and a target slot is never
  filled with the subject entity (`target_answer_slot_filled_with_subject_count` = 0). (3)
  **Anti-pollution is precision, not cleverness**: blocking a generic single-token seed,
  refusing to read dictionary pages for entity tasks, and staging dependent slots behind their
  dependencies all *narrow* what the loop does — like the 5f/5g gates they can lower apparent
  accuracy by abstaining more, which is the intended trade. The candidate-local "confirmed"
  label is renamed `slot_candidate_supported` precisely so a reader never mistakes a local slot
  decision for global answer readiness. Detector counts (`read_loop_open_count`,
  `proposal_gate_starvation_count`, …) are **debug labels for mechanism comparison only** — an
  `n=2` smoke supports **no** benchmark or generalization claim, and the recommended fixed
  dev/debug slice is for debugging deltas, never a headline. Deferred and explicitly not
  claimed: source-subject extraction, coreference collapse, and a batch-judge / explicit-location
  hard filter.
