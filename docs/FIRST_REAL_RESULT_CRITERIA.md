# What would count as a real result

Tiers for a *real* (non-synthetic) outcome on LiveBrowseComp / BrowseComp. None
of these is met yet — today's evidence is synthetic-harness only (see
`docs/STATUS.md`). Each tier's checks should be reflected by `headline_eligible`
in the report plus the metrics, and grounded in committed `results/{run_id}/`
artifacts. Read `docs/METHODOLOGY_RISKS.md` alongside this.

## Minimum credible result
All of the following:
- Run on a **LiveBrowseComp or BrowseComp subset** (real data, not the fixture).
- **`closed_book` baseline** included (intrinsic-knowledge estimate, 0 tools).
- **`no_memory_search` baseline** included (search, no memory).
- **`policy_memory`** run included (frozen snapshot).
- **`random_memory`** control included.
- **Same-conditions validator passes** — baseline and policy differ only in
  memory access (`report.json.same_conditions.ok == true`).
- **Frozen memory on CONFIRM** (`confirm_uses_frozen_snapshot`, no live updates).
- **No-answer-leakage check passes** on the real traces and the committed
  snapshot.
- **Replay/check artifacts generated** and `replay_check.projection_matches`.
- **Improvement in `correct_per_tool_call` at budget 3 or 5** on CONFIRM.
- **No unacceptable accuracy regression** vs `no_memory_search`.
- **Confidence intervals reported** (bootstrap CI for `correct_per_tool_call`;
  McNemar for accuracy flips), and `headline_eligible == true`.

## Strong result
The minimum, plus:
- Improvement **across multiple budget caps** (e.g. 1, 3, 5), not one cap.
- Improved **`first_tool_hit_rate`**.
- Improved **`correct_per_tool_call`** with the CI excluding the baseline.
- **Reduced stale-source errors** (`stale_source_error_rate` down) on the
  freshness-sensitive subset.
- **`random_memory` does NOT match `policy_memory`** — the gain is from the
  learned policy, not from merely having priors.
- Result **repeats across at least two seeds/splits** (mean ± CI).

## Very strong result
The strong tier, plus:
- **Held-out improvement on LiveBrowseComp** (the primary, recent-fact target).
- **BrowseComp secondary corroboration** (the gain is not dataset-specific).
- **Consistent gains across at least two model/tool configurations** (e.g.
  `gpt-5.5` and `gpt-5.4-mini`, or two search providers).
- A **regimes-style policy update that promotes on CONFIRM**, not just OPTIMIZE
  (`promotion.accepted` with a held-out improvement).
- **Report artifacts are sufficient to reproduce every claim** — split ids,
  model/tool versions, prompts, page hashes, snapshots, and the replay check are
  all committed, and re-running reproduces the numbers.

## How the harness enforces these
- `report.json.headline_eligible` gates the minimum structural checks
  (`eval/eligibility.py`) and refuses to call a synthetic run a benchmark result.
- `eval/conditions.py` enforces same-conditions.
- `scripts/validate_live_readiness.py` checks configuration before a run.
- `scripts/compare_runs.py` compares two runs (budget curves, deltas, McNemar).
- `scripts/inspect_memory_snapshot.py` verifies the snapshot is answer-free.

Until a run clears the **Minimum credible** bar with `headline_eligible == true`,
the honest claim remains: *the scaffold demonstrates the intended mechanism on a
synthetic fixture.*

## Two eligibility verdicts (don't conflate them)

Reports carry **two** distinct flags (see `eval/eligibility.py`):

- **`structurally_valid`** — split disjoint, replay passes, no answer leakage,
  budgets enforced, requested runs completed. A *plumbing* run (e.g.
  `closed_book,no_memory_search`) can be structurally valid.
- **`headline_eligible_memory_claim`** — the run can support the **main
  memory-learning claim**. This requires, in addition to structural validity:
  a **real** dataset; **all four** conditions present (`closed_book`,
  `no_memory_search`, `random_memory`, `policy_memory`); the same-conditions check
  on `no_memory_search` vs `policy_memory`; frozen CONFIRM memory with no live
  updates; and **CONFIRM ≥ `headline.min_confirm_size`** (default 20).

**A run without `policy_memory` is never headline-eligible** — there is no memory
comparison to make. `scripts/generate_claims.py` refuses memory-performance claims
in that case and labels the run a "real … plumbing run completed". The minimum
credible result above therefore requires `headline_eligible_memory_claim == true`,
not merely `structurally_valid`.
