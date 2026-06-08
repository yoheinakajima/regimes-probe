#!/usr/bin/env python
"""Hash a run's artifacts into an audit ledger (no network).

    python scripts/hash_artifacts.py results/demo

Writes results/{run_id}/artifact_hashes.json mapping each artifact to its
sha256, so a reviewer can verify the committed artifacts were not altered.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# The audit set (others are ignored if absent; artifact_hashes.json itself is
# excluded so the ledger never hashes itself).
_ARTIFACTS = [
    "report.json", "summary.md", "per_question.csv", "budget_curve.csv",
    "tool_rewards.csv", "query_rewards.csv", "stop_verify_rewards.csv",
    "memory_snapshot.json", "policy_updates.json", "run_manifest.json",
    "replay_check.md", "config_snapshot.yaml", "debug_questions.jsonl",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python scripts/hash_artifacts.py results/{run_id}")
        return 2
    run_dir = Path(sys.argv[1])
    if not run_dir.is_dir():
        print(f"not a directory: {run_dir}")
        return 2

    names = list(_ARTIFACTS)
    # include any extra config_*.yaml copied into the run dir
    names += sorted(p.name for p in run_dir.glob("config_*.yaml") if p.name not in names)

    ledger: dict[str, str] = {}
    missing: list[str] = []
    for name in names:
        p = run_dir / name
        if p.exists():
            ledger[name] = _sha256(p)
        else:
            missing.append(name)

    out = run_dir / "artifact_hashes.json"
    out.write_text(json.dumps({"run_dir": str(run_dir), "sha256": ledger,
                               "missing": missing}, indent=2), encoding="utf-8")
    print(f"hashed {len(ledger)} artifacts -> {out}")
    for name, digest in ledger.items():
        print(f"  {digest[:16]}  {name}")
    if missing:
        print(f"  (absent: {', '.join(missing)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
