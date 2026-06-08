# Offline forked ablations

After a paid provider run, we want to compare policy/reward/router variants
**without spending again**. The recording cache (`docs/LIVE_LADDER.md`) already
stores every provider/model/tool outcome keyed by its sanitized request. An
*offline fork* reuses that cache in `replay` mode to rerun the CONFIRM phase
under different parameters — and refuses (never calls out) on any cache miss.

## Command

```
python scripts/fork_offline_ablation.py results/live/<run_id> \
    --out <new_run_id> --policy-config overrides.yaml
```

- `--policy-config` — a YAML of overrides merged onto the parent's
  `config_snapshot.yaml` (e.g. `reward:`, `bandit:`, `router:`, `stopping:`).
- `--cache` — recording cache path (default: the parent manifest's `cache.path`).
- `--conditions` — default `policy_memory` (see *Scope* below).
- `--budgets` — default: the parent's budgets.
- `--answerer {deterministic,live}` — `live` replays the parent's cached model
  answers (use for a live-origin fork); `deterministic` for the synthetic/fixture
  answerer.

## What v0 does

1. Loads the parent `run_manifest.json`, `config_snapshot.yaml`, and
   `memory_snapshot.json` (resumed as the frozen snapshot — **the experience
   phase is skipped**, so no experience-time provider calls are needed).
2. Opens the recording cache in **replay** mode and builds replay-only providers
   (`live/fork.py:build_replay_providers`): a cache hit is served; a miss raises
   `ReplayMiss`/`NotArmed`. No provider is ever called.
3. Applies the policy overrides and reruns the requested conditions through the
   normal `run_live_pipeline`.
4. Writes a full artifact set (`report.json`, `graph_projection.json`,
   `debug_questions.jsonl`, …) with `offline_fork=true` and `parent_run_id` in the
   manifest. **An offline fork is never headline-eligible** (it reused cached
   outcomes; it is an ablation, not a fresh measurement).

## Scope (why the default is `policy_memory` only)

The CONFIRM behavior of the memory-variant conditions (`policy_memory`,
`random_memory`) is fully determined by the frozen snapshot, so they replay from
the cache **exactly**. The `no_memory_search` baseline is fixed across policy
variants and its realized queries depend on agent warm-state from the parent's
experience phase, which a fresh fork cannot reproduce without re-running
experience (i.e. spending). So:

- Fork the **memory variant** (`policy_memory`) under your new parameters.
- Read the **baseline** (`no_memory_search`) from the *parent* report — it does
  not change with the policy variant.

If a fork would need a request that is not in the cache (e.g. an override that
changes the realized query distribution), it **refuses with a clear message and
spends nothing** — it does not silently fall back to a live call.

## Safety invariants

- No provider/model call is ever made by a fork (replay-only, un-armed).
- No secrets are read or written (the cache is already sanitized).
- The benchmark methodology is unchanged; a fork only re-weights/re-routes over
  already-observed outcomes.
- Forks are clearly marked (`offline_fork=true`, `parent_run_id`) and cannot be
  promoted to a headline memory claim.

## Extension points (TODO for a v1)

- Record the parent's answerer kind in the manifest so a live fork auto-selects
  the live replay answerer.
- Re-key the cache on the *realized* query so query-distribution-preserving
  router changes can also be forked.
- A `compare_forks.py` that diffs two forks' reports/metrics directly.
