# Public-review checklist

What a skeptical reviewer should inspect before believing any claim from a run.
Each item names the artifact/command that settles it. For a synthetic run,
`headline_eligible` is false and no performance claim is permitted — review still
applies to the *mechanism* claims.

## Integrity of the comparison
- [ ] **Same conditions.** `report.json.same_conditions.ok == true` and
  `unexpected_diffs == []`. Inspect `report.json.condition_specs` — baseline and
  policy must match on model, prompt fingerprint, tools, budget, split id,
  grader, provider config, and query/verify/stop policy.
- [ ] **Prompt versions pinned.** `run_manifest.json.prompts[*].fingerprint`
  (name@version#hash) is recorded; the answerer/judge prompts are
  `allowed_to_vary: false`.
- [ ] **Provider configs.** `run_manifest.json.provider_config` lists provider
  names and env-var **names** only — no secret values anywhere.

## Splits & leakage
- [ ] **Exact split ids.** `run_manifest.json.split.optimize_ids` /
  `confirm_ids` are committed; verify OPTIMIZE ∩ CONFIRM = ∅.
- [ ] **Frozen memory on CONFIRM.** `run_manifest.json.memory.freezes_during_confirm == true`
  and `eligibility.checks.confirm_memory_frozen == true`.
- [ ] **No answer leakage.** `eligibility.checks.no_answer_leakage == true`; run
  `python scripts/inspect_memory_snapshot.py results/{run_id}/memory_snapshot.json`
  and confirm the leakage scan PASSes and no gold string appears.

## Baselines & controls
- [ ] **Closed-book baseline** present (`closed_book`, 0 tool calls) — bounds
  intrinsic knowledge.
- [ ] **No-memory baseline** present (`no_memory_search`).
- [ ] **Random-memory control** present (`random_memory`) and clearly **below**
  `policy_memory` (else the signal is suspect).

## Evidence of effect
- [ ] **Replay check.** `replay_check.md` → `projection_matches: true`.
- [ ] **Per-question flips.** `per_question.csv` + `scripts/compare_runs.py`
  McNemar: wrong→correct flips with few/no reverse flips.
- [ ] **Budget curves.** `budget_curve.csv` — improvement holds across budgets,
  not one cherry-picked cap.
- [ ] **Confidence intervals.** `report.json.significance` — bootstrap CI for
  `correct_per_tool_call` excludes the baseline; McNemar p reported.

## Provenance & reproducibility
- [ ] **Artifact hashes.** `python scripts/hash_artifacts.py results/{run_id}`
  → `artifact_hashes.json`; recompute and compare.
- [ ] **Git provenance.** `run_manifest.json.git` (branch/commit/dirty).
- [ ] **Dataset identity.** `run_manifest.json.dataset` (name/version/checksum/path).
- [ ] **Cost footprint.** `run_manifest.json.cost_estimate` matches the run scale.

## Claims discipline
- [ ] **Unsupported claims avoided.** `summary.md` "What is NOT claimed" present;
  `scripts/generate_claims.py` REFUSES performance claims when
  `headline_eligible == false`.
- [ ] **CONFIRM-only headline.** OPTIMIZE numbers are disclosed but never the
  headline.
- [ ] **No synthetic-as-benchmark.** A synthetic/placeholder dataset yields
  `headline_eligible: false` with the reason recorded.

## One-command sanity
```bash
python scripts/run_synthetic_full.py --run-id review     # full no-key pipeline
python scripts/inspect_memory_snapshot.py results/review/memory_snapshot.json
python scripts/hash_artifacts.py results/review
python scripts/generate_claims.py results/review/report.json
```
