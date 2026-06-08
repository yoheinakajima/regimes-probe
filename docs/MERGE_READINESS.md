# Merge readiness

What must be true before this branch is merged. None of this requires API keys or
network. (Opening the PR / merging is a human decision — this doc is the gate.)

## Hard gates (must all hold)
- [ ] **Tests pass:** `make test` green (60+ tests, no keys/network).
- [ ] **No-key full pipeline passes:** `make synthetic-full` writes
  `results/{run_id}/` with `report.json`, `summary.md`, budget curve, snapshot,
  `run_manifest.json`, and a passing replay check.
- [ ] **Real-shaped smoke passes:** `make real-shaped-smoke`
  (`tests/test_real_data_shape.py`) — the real adapter path works on placeholder
  data with no keys.
- [ ] **Preflight passes (non-strict):** `make preflight-live` reports 0 blockers
  and writes a dry-run manifest, making **no provider calls**.
- [ ] **Docs check passes:** `make docs-check` — required docs present and
  README/STATUS make no unguarded benchmark claim.
- [ ] **No overclaims:** README and `STATUS.md` state the current claim as
  *"the scaffold demonstrates the intended mechanism on a synthetic fixture"*;
  `headline_eligible` is false on every committed synthetic run.
- [ ] **No secrets committed:** no API keys or `.env`; `config/tools.yaml` is
  git-ignored; only env-var **names** appear in configs/manifests.
- [ ] **Artifacts are intentional:** generated outputs are either git-ignored or
  deliberately committed as grounding (`results/demo/`, `results/regimes-demo/`);
  scratch run dirs are not committed.
- [ ] **Live providers opt-in:** no module performs network I/O at import; every
  provider adapter is env-gated; unit tests never touch the network.
- [ ] **First live-run plan documented:** `docs/NEXT_LIVE_RUN.md`,
  `docs/REAL_BENCHMARK_READINESS.md`, `docs/FIRST_REAL_RESULT_CRITERIA.md`,
  `docs/METHODOLOGY_RISKS.md`, and `docs/PUBLIC_REVIEW_CHECKLIST.md` exist and
  are linked from the README.

## Recommended (not blocking)
- [ ] `make ablations` runs and the Level1→Level2 progression is sane.
- [ ] `make inspect-memory` shows an answer-free snapshot (leakage scan PASS).
- [ ] `scripts/hash_artifacts.py` + `scripts/generate_claims.py` run on the demo;
  the claim generator REFUSES performance claims (synthetic).

## One-shot verification
```bash
make check            # test + docs-check + synthetic-full + real-shaped-smoke
make preflight-live   # 0 blockers, no provider calls
```

## Explicitly out of scope for this merge
- Any live provider call or API-key-requiring step.
- Any BrowseComp/LiveBrowseComp performance number.
- Arbitrary prompt/code mutation; answer/judge-prompt optimization.

When every hard gate holds, the branch is mergeable as a **credible, auditable,
no-key scaffold** whose first live run is fully planned. It is still not a
benchmark result, and the docs say so.
