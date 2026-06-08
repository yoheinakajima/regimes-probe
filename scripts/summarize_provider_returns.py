#!/usr/bin/env python
"""Per-provider tool-return summary for a run (no network).

    python scripts/summarize_provider_returns.py results/live/<run_id>

Aggregates the per-call records in debug_questions.jsonl (and cache info from
run_manifest.json) to show, per provider: calls, success/failure counts, failure
rate, status/error codes, empty-result counts, mean results per call, top
returned domains, snippet availability, and cache hits vs live calls.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse
import sys


def _load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python scripts/summarize_provider_returns.py results/{run_id}")
        return 2
    run_dir = Path(sys.argv[1])
    rows = _load_jsonl(run_dir / "debug_questions.jsonl")
    if not rows:
        print(f"no debug_questions.jsonl in {run_dir}")
        return 2

    calls = defaultdict(int)
    failures = defaultdict(int)
    empty = defaultdict(int)
    results_total = defaultdict(int)
    status_codes = defaultdict(Counter)
    error_types = defaultdict(Counter)
    domains = defaultdict(Counter)
    snippet_have = defaultdict(int)
    snippet_total = defaultdict(int)

    for r in rows:
        for c in r.get("calls", []):
            t = c.get("tool", "?")
            calls[t] += 1
            results_total[t] += c.get("n_results", 0)
            if c.get("failed"):
                failures[t] += 1
                if c.get("status_code") is not None:
                    status_codes[t][c["status_code"]] += 1
                if c.get("error_type"):
                    error_types[t][c["error_type"]] += 1
            elif c.get("n_results", 0) == 0:
                empty[t] += 1
        for ev in r.get("evidence", []):
            # attribute evidence domains/snippets best-effort (provider not on the
            # evidence row; use the run's tools as a proxy is unreliable, so we
            # only aggregate domains/snippets globally here).
            host = urlparse(ev.get("url", "")).hostname or ""
            if host:
                domains["_all"][host] += 1
            snippet_total["_all"] += 1
            if ev.get("snippet_preview"):
                snippet_have["_all"] += 1

    manifest = {}
    mp = run_dir / "run_manifest.json"
    if mp.exists():
        manifest = json.loads(mp.read_text(encoding="utf-8"))
    cache = manifest.get("cache", {})

    print(f"=== provider returns — {run_dir} ===\n")
    print(f"{'provider':<24} {'calls':>6} {'fail':>5} {'fail%':>6} {'empty':>6} {'mean_res':>9}")
    print("-" * 62)
    for t in sorted(calls, key=lambda x: -calls[x]):
        n = calls[t]
        fr = failures[t] / n if n else 0.0
        mean_res = results_total[t] / n if n else 0.0
        print(f"{t:<24} {n:>6} {failures[t]:>5} {fr:>6.2f} {empty[t]:>6} {mean_res:>9.2f}")
    print()
    for t in sorted(failures, key=lambda x: -failures[x]):
        if failures[t]:
            print(f"  {t}: status_codes={dict(status_codes[t])} error_types={dict(error_types[t])}")
    print()
    st, sh = snippet_total.get("_all", 0), snippet_have.get("_all", 0)
    print(f"evidence snippet availability: {sh}/{st} rows have a snippet preview")
    top = domains.get("_all", Counter()).most_common(10)
    print(f"top returned domains: {top}")
    if cache:
        print(f"\ncache: live_calls={cache.get('live_calls')} hits={cache.get('cache_hits')} "
              f"entries={cache.get('n_entries')} mode={cache.get('mode')}")
    else:
        print("\ncache: (no manifest cache info — synthetic/dry run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
