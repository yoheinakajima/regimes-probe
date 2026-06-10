#!/usr/bin/env python
"""Create a stable dev/debug slice manifest (no network, no dataset mutation).

    python scripts/make_debug_slice.py --dataset synthetic_browse --n 12 --seed 0 \
        --slice-id mech-debug --out results/debug_slice_manifest.json

The slice is a small, STABLE set of item ids preserved across iterations so
``compare_runs`` can show per-item flips and mechanism-detector deltas on the exact same
items. It is debug/dev ONLY — never headline eligible — and never mutates the dataset.

You can pin a slice across runs with ``--items id1,id2,...`` (or ``--items-file path``),
which records those exact ids (``--method explicit_ids``). Otherwise selection is a
deterministic ``seeded_sample`` (or ``first_n``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regimes_probe.eval.debug_slice import select_debug_slice   # noqa: E402


def _load_item_ids(dataset: str, dataset_path: str) -> list[str]:
    """Read item ids WITHOUT mutating the dataset (ids/questions only)."""
    from regimes_probe.datasets import (
        BrowseCompAdapter, LiveBrowseCompAdapter, SyntheticBrowseAdapter)
    if dataset in ("synthetic_browse", "synthetic"):
        adapter = (SyntheticBrowseAdapter(Path(dataset_path)) if dataset_path
                   else SyntheticBrowseAdapter())
    elif dataset == "browsecomp":
        adapter = BrowseCompAdapter(Path(dataset_path)) if dataset_path else BrowseCompAdapter()
    elif dataset in ("livebrowsecomp", "live_browsecomp"):
        adapter = (LiveBrowseCompAdapter(Path(dataset_path)) if dataset_path
                   else LiveBrowseCompAdapter())
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    return [it.id for it in adapter.load()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Create a stable dev/debug slice manifest.")
    ap.add_argument("--dataset", default="synthetic_browse")
    ap.add_argument("--dataset-path", default="")
    ap.add_argument("--slice-id", default="debug")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", default="seeded_sample",
                    choices=("seeded_sample", "first_n", "explicit_ids"))
    ap.add_argument("--items", default="", help="comma-separated item ids to pin")
    ap.add_argument("--items-file", default="", help="file with one item id per line to pin")
    ap.add_argument("--out", default="results/debug_slice_manifest.json")
    args = ap.parse_args()

    explicit = [s.strip() for s in args.items.split(",") if s.strip()]
    if args.items_file:
        explicit += [ln.strip() for ln in Path(args.items_file).read_text().splitlines()
                     if ln.strip()]
    method = "explicit_ids" if explicit else args.method
    try:
        pool = _load_item_ids(args.dataset, args.dataset_path)
    except Exception as exc:                                   # dataset unavailable / no path
        print(f"could not load dataset {args.dataset!r}: {exc}", file=sys.stderr)
        return 2

    manifest = select_debug_slice(
        pool, n=args.n, seed=args.seed, slice_id=args.slice_id, method=method,
        dataset=args.dataset, dataset_path=args.dataset_path, explicit_ids=explicit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(f"wrote {out} — slice_id={manifest.slice_id} method={manifest.selection_method} "
          f"n={len(manifest.item_ids)} headline_eligible={manifest.headline_eligible}")
    print("DEBUG/DEV slice only — NOT a benchmark or generalization claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
