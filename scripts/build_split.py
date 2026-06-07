#!/usr/bin/env python
"""Build and persist a deterministic OPTIMIZE/CONFIRM split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import build_synthetic, base_argparser, load_config

from regimes_probe.eval.split import build_split


def main() -> int:
    args = base_argparser("Build a deterministic OPTIMIZE/CONFIRM split.").parse_args()
    cfg = load_config(args.config)
    items, *_ = build_synthetic(cfg)
    sp = cfg.get("split", {})
    split = build_split(items, confirm_fraction=sp.get("confirm_fraction", 0.4),
                        salt=sp.get("salt", "regimes-probe-v0"), mode=sp.get("mode", "hash"))
    run_id = args.run_id or "split"
    out = Path(args.results_root) / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "split.json").write_text(json.dumps(split.to_dict(), indent=2), encoding="utf-8")
    print(f"OPTIMIZE={len(split.optimize_ids)}  CONFIRM={len(split.confirm_ids)}  -> {out/'split.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
