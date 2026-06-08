#!/usr/bin/env python
"""No-key docs lint: required docs exist + README/STATUS make no benchmark claim.

    python scripts/docs_check.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

REQUIRED = [
    "ARCHITECTURE", "RESEARCH_PLAN", "BENCHMARK_TARGETS", "MODEL_AND_TOOL_CHOICES",
    "ACTIVEGRAPH_DESIGN", "EVENT_SCHEMA", "POLICY_MEMORY", "CONTEXTUAL_BANDIT",
    "REGIMES_IMPROVEMENT_LOOP", "ROUTING_POLICY", "QUERY_POLICY",
    "VERIFICATION_AND_STOPPING", "GRADING_AND_REWARD", "EVALUATION_PROTOCOL",
    "LEAKAGE_CONTROLS", "REPORTING", "IMPLEMENTATION_PLAN", "STATUS",
    "REAL_BENCHMARK_READINESS", "NEXT_LIVE_RUN", "METHODOLOGY_RISKS",
    "FIRST_REAL_RESULT_CRITERIA", "PUBLIC_REVIEW_CHECKLIST", "MERGE_READINESS",
    "LIVE_LADDER", "TOOL_ABSTRACTIONS", "OFFLINE_FORK_ABLATIONS",
]

# An affirmative claim of a real benchmark result (excluding negations/hypotheticals).
_CLAIM = re.compile(r"\b(achiev\w*|improv\w*|beat\w*|outperform\w*)\b.*\b(browsecomp|livebrowsecomp)\b",
                    re.IGNORECASE)
_NEGATION = re.compile(r"\b(not|no|never|would|if|cannot|can't|refus\w*|claim no|"
                       r"does not|isn't|aren't|hypothe\w*)\b", re.IGNORECASE)


def main() -> int:
    problems: list[str] = []

    for name in REQUIRED:
        if not (DOCS / f"{name}.md").exists():
            problems.append(f"missing doc: docs/{name}.md")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "no real benchmark performance is claimed" not in readme.lower() \
            and "not been run on browsecomp" not in readme.lower():
        problems.append("README.md lacks the explicit 'no real benchmark performance' disclaimer")

    status = (DOCS / "STATUS.md").read_text(encoding="utf-8")
    if "not claimed" not in status.lower() and "unsupported claims" not in status.lower():
        problems.append("STATUS.md lacks an 'unsupported claims / not claimed' section")

    for fname in ("README.md", "docs/STATUS.md"):
        text = (ROOT / fname).read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if _CLAIM.search(line) and not _NEGATION.search(line):
                problems.append(f"{fname}:{i} possible unguarded benchmark claim: {line.strip()[:90]}")

    if problems:
        print("docs-check FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"docs-check OK: {len(REQUIRED)} required docs present; "
          "README/STATUS make no unguarded benchmark claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
