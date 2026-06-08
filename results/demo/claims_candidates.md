# Claim candidates — `demo` (dataset: synthetic_browse)

> structurally_valid = **True**; headline_eligible_memory_claim = **False**.
> conditions present: ['closed_book', 'no_memory_search', 'policy_memory', 'random_memory'].

Memory-performance claims are REFUSED below.

## Verified (structural)
- OPTIMIZE and CONFIRM splits are disjoint.
- the graph is a deterministic projection of the event log (replay).
- policy memory contains no answer text (leakage check passed).
- tool-call budgets were enforced.
- the requested runs completed.
- CONFIRM used a frozen policy-memory snapshot.
- no live memory updates during CONFIRM.
- no_memory_search and policy_memory differ only in memory access.
- provider failure rate within the headline threshold.

## Partially verified
- A closed-book baseline ran (0 tool calls); intrinsic-knowledge accuracy estimate = 0.036 on this dataset.
- Mechanism (routing/query/stop learning) executes end to end and is recorded/replayable; generalization is not established by this run.

## Memory-performance claims
- **REFUSED**: not headline-eligible; no memory-performance claim may be made.
  - reason: dataset is a synthetic/placeholder fixture — not a benchmark headline

## Not supported / not headline eligible
- No memory-learning claim about `synthetic_browse` is supported by this run.
- This is a synthetic/placeholder fixture: it demonstrates the mechanism only, not benchmark performance.
- reason: dataset is a synthetic/placeholder fixture — not a benchmark headline
## Limitations
- Dataset is `synthetic_browse` — a synthetic/placeholder harness, NOT a real benchmark score.
- The synthetic answerer models a competent extractor; it isolates retrieval/epistemic policy.
- No live providers were called; only fixture-backed, deterministic tools.
- not claimed: No claim of BrowseComp or LiveBrowseComp performance.
- not claimed: No claim base model weights changed (they do not).
- not claimed: No claim benchmark answers are stored in policy memory (they are not).
