#!/usr/bin/env python
"""(Re)generate the committed real-data-shaped placeholder fixtures.

No keys, no network. Writes fixtures/real_shaped/{browsecomp_sample.csv,
livebrowsecomp_sample.jsonl,real_shaped_corpus.json}.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regimes_probe.datasets.real_shaped import write_real_shaped_fixtures


def main() -> int:
    out = write_real_shaped_fixtures(ROOT / "fixtures" / "real_shaped")
    print(f"wrote real-shaped placeholder fixtures -> {out}")
    for p in sorted(out.glob("*")):
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
