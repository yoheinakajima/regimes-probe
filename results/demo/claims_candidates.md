# Claim candidates — `demo` (dataset: synthetic_browse)

> headline_eligible = **False**. Performance claims are REFUSED below.

## Verified (structural)
- OPTIMIZE and CONFIRM splits are disjoint.
- CONFIRM used a frozen policy-memory snapshot.
- policy memory contains no answer text (leakage check passed).
- baseline and policy differ only in memory access.
- the graph is a deterministic projection of the event log (replay).
- baseline and policy runs both completed.
- tool-call budgets were enforced.
- no live memory updates during CONFIRM.

## Partially verified
- A closed-book baseline ran (0 tool calls); intrinsic-knowledge accuracy estimate = 0.036 on this dataset.
- Mechanism (routing/query/stop learning) executes end to end and is recorded/replayable; generalization to real data is not established here.

## Performance claims
- **REFUSED**: headline_eligible is false; no performance/benchmark claim may be generated from this run.
  - reason: dataset is a synthetic/placeholder fixture — not a benchmark headline

## Not supported / not headline eligible
- No claim about `synthetic_browse` performance is supported by this run.
- This is a synthetic/placeholder fixture: it demonstrates the mechanism only, not benchmark performance.

## Limitations
- Dataset is `synthetic_browse` — a synthetic/placeholder harness, NOT a real benchmark score.
- The synthetic answerer models a competent extractor; it isolates retrieval/epistemic policy.
- No live providers were called; only fixture-backed, deterministic tools.
- not claimed: No claim of BrowseComp or LiveBrowseComp performance.
- not claimed: No claim base model weights changed (they do not).
- not claimed: No claim benchmark answers are stored in policy memory (they are not).
