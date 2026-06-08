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
- **Offline forks are ablations, not measurements.** A forked run reuses a
  parent's cached outcomes and re-weights/re-routes over already-observed data;
  it can reveal which variant *would* have done better on the same evidence, but
  it is `offline_fork=true` and never headline-eligible. A fork that would change
  the realized query distribution refuses (cache miss) rather than spending — so
  do not read a "successful" fork as evidence the new policy generalizes.
