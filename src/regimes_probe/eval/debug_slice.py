"""Level 5i-N: a stable dev/debug slice selector for *mechanism* debugging.

A debug slice is a small (10-20 item), **stable** set of item ids preserved across
iterations, so ``compare_runs`` can show per-item flips and mechanism-detector deltas on the
exact same items. It is **debug/dev only** — never headline eligible — and it does not mutate
the dataset. Selection is deterministic given ``(item ids, n, seed, method)`` so the slice
replays identically.

This module is pure and import-light (no dataset/provider imports), so it is trivially
offline-testable; the CLI wrapper (``scripts/make_debug_slice.py``) does the dataset loading.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass, field
from typing import Any

SLICE_KIND = "debug_dev_slice"
SELECTION_METHODS = ("first_n", "seeded_sample", "explicit_ids")


def _checksum(item_ids: list[str]) -> str:
    h = hashlib.sha256()
    for i in item_ids:
        h.update(i.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


@dataclass
class DebugSliceManifest:
    """Replayable, deterministic record of a debug slice — **never headline eligible**."""
    slice_id: str
    dataset: str
    selection_method: str
    seed: int
    n: int
    item_ids: list[str]
    dataset_path: str = ""
    source_item_count: int = 0
    source_checksum: str = ""
    slice_checksum: str = ""
    kind: str = SLICE_KIND
    headline_eligible: bool = False           # invariant: a debug slice is never a headline
    note: str = ("debug/dev mechanism-comparison slice only; NOT a benchmark or "
                 "generalization claim")
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_debug_slice(item_ids, *, n: int = 12, seed: int = 0, slice_id: str = "debug",
                       method: str = "seeded_sample", dataset: str = "",
                       dataset_path: str = "", explicit_ids=None) -> DebugSliceManifest:
    """Deterministically select up to ``n`` item ids for a stable debug slice.

    ``method``:
    - ``first_n``        — the first ``n`` ids in sorted order (fully deterministic).
    - ``seeded_sample``  — a seeded sample of ``n`` ids from the sorted pool (deterministic
                           given ``seed``).
    - ``explicit_ids``   — use ``explicit_ids`` verbatim (intersected with the available
                           pool, order preserved) — the way to *pin* a slice across runs.

    The dataset is never mutated; only ids are read."""
    if method not in SELECTION_METHODS:
        raise ValueError(f"unknown selection method: {method!r}")
    pool = [str(i) for i in item_ids]
    pool_sorted = sorted(set(pool))
    if method == "explicit_ids":
        wanted = [str(i) for i in (explicit_ids or [])]
        avail = set(pool)
        chosen = [i for i in wanted if i in avail]          # pinned + order-preserving
    elif method == "first_n":
        chosen = pool_sorted[: max(0, n)]
    else:                                                    # seeded_sample
        k = min(max(0, n), len(pool_sorted))
        chosen = sorted(random.Random(seed).sample(pool_sorted, k)) if k else []
    return DebugSliceManifest(
        slice_id=slice_id, dataset=dataset, dataset_path=dataset_path, selection_method=method,
        seed=int(seed), n=int(n), item_ids=chosen, source_item_count=len(pool_sorted),
        source_checksum=_checksum(pool_sorted), slice_checksum=_checksum(chosen))


#: Mechanism-detector keys surfaced in debug-slice comparisons (5i-N). These are DEBUG labels
#: for mechanism deltas only — never benchmark/performance claims.
DEBUG_SLICE_DETECTOR_KEYS = (
    "read_loop_open_count",
    "requires_read_resolved_by_read_count",
    "read_success_no_evidence_added_count",
    "proposal_gate_starvation_count",
    "seed_query_pollution_count",
    "target_slot_starved_count",
    "bind_target_answer_slot_success_count",
    "generic_single_token_seed_executed_count",
    "generic_definition_source_read_count",
    "premature_founder_search_before_place_supported_count",
    "judge_calls_saved_by_prejudge_triage",
    "judge_calls_saved_by_batching",
)


def is_debug_slice_report(report: dict) -> bool:
    """Whether a report.json was produced over a debug/dev slice (never headline eligible)."""
    meta = report.get("meta", {}) if isinstance(report, dict) else {}
    ds = meta.get("debug_slice") or report.get("debug_slice")
    return bool(ds and (ds.get("kind") == SLICE_KIND or ds.get("headline_eligible") is False))


def detector_deltas(metrics_a: dict, metrics_b: dict) -> dict[str, dict[str, Any]]:
    """Per-detector (a, b, delta) for the debug-slice comparison surface."""
    out: dict[str, dict[str, Any]] = {}
    for k in DEBUG_SLICE_DETECTOR_KEYS:
        a = metrics_a.get(k)
        b = metrics_b.get(k)
        if a is None and b is None:
            continue
        av, bv = float(a or 0), float(b or 0)
        out[k] = {"a": a, "b": b, "delta": round(bv - av, 4)}
    return out
