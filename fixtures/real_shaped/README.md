# Real-data-SHAPED placeholder fixtures

These files mimic the **structure** of the real benchmarks so the adapter path,
split builder, report generator, leakage checks, budget curves, and memory
snapshot can be exercised **with no API keys and no network**:

- `browsecomp_sample.csv` — BrowseComp shape: `problem,answer,canary` with the
  problem/answer XOR-obfuscated by the row canary (same scheme as
  `datasets/browsecomp.py`).
- `livebrowsecomp_sample.jsonl` — LiveBrowseComp shape: one JSON object per line
  with `id`, `question`, `answer`, `released_at`.
- `real_shaped_corpus.json` — a fake search corpus keyed to these item ids so
  the agent loop produces non-trivial outputs.

**These are NOT BrowseComp or LiveBrowseComp content.** Every question is an
obviously-fictional placeholder (Zembla, Greyhold Keep, Skyforge, …). Results on
these fixtures must **never** be reported as BrowseComp/LiveBrowseComp
performance. See `docs/METHODOLOGY_RISKS.md` and `docs/REAL_BENCHMARK_READINESS.md`.

Regenerate with: `python scripts/build_real_shaped_fixtures.py`
Exercised by: `tests/test_real_data_shape.py`
