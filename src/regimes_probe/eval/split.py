"""Deterministic OPTIMIZE / CONFIRM splits.

OPTIMIZE is the in-sample experience set; CONFIRM is held out. The split is a
pure function of item ids (and optionally release dates) so it is reproducible
and auditable. CONFIRM must never leak into OPTIMIZE (tested in
``tests/test_split.py``). See ``docs/EVALUATION_PROTOCOL.md`` and
``docs/LEAKAGE_CONTROLS.md``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Optional

from regimes_probe.datasets.base import Item


@dataclass
class Split:
    optimize_ids: list[str]
    confirm_ids: list[str]
    mode: str
    salt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "salt": self.salt,
            "n_optimize": len(self.optimize_ids),
            "n_confirm": len(self.confirm_ids),
            "optimize_ids": self.optimize_ids,
            "confirm_ids": self.confirm_ids,
        }

    def assert_disjoint(self) -> None:
        overlap = set(self.optimize_ids) & set(self.confirm_ids)
        if overlap:
            raise AssertionError(f"OPTIMIZE/CONFIRM overlap: {sorted(overlap)}")


def _bucket(item_id: str, salt: str) -> float:
    h = hashlib.sha256(f"{salt}:{item_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


def build_split(
    items: list[Item],
    *,
    confirm_fraction: float = 0.5,
    salt: str = "regimes-probe-v0",
    mode: str = "hash",
) -> Split:
    """Build a deterministic split.

    ``mode``:
      * ``hash`` — hash item id into [0,1); CONFIRM if below ``confirm_fraction``.
        Entity overlap is avoided to the extent ids are unique.
      * ``time`` — sort by ``released_at`` and put the most recent
        ``confirm_fraction`` into CONFIRM (time-disjoint, recommended for
        LiveBrowseComp).
    """
    if mode == "time" and all(i.released_at for i in items):
        ordered = sorted(items, key=lambda i: (i.released_at, i.id))
        n_confirm = round(len(ordered) * confirm_fraction)
        confirm = ordered[len(ordered) - n_confirm :]
        optimize = ordered[: len(ordered) - n_confirm]
        split = Split(
            optimize_ids=[i.id for i in optimize],
            confirm_ids=[i.id for i in confirm],
            mode="time",
            salt=salt,
        )
    else:
        optimize_ids, confirm_ids = [], []
        for it in items:
            (confirm_ids if _bucket(it.id, salt) < confirm_fraction else optimize_ids).append(it.id)
        split = Split(
            optimize_ids=sorted(optimize_ids),
            confirm_ids=sorted(confirm_ids),
            mode="hash",
            salt=salt,
        )
    split.assert_disjoint()
    return split


def partition(items: list[Item], split: Split) -> tuple[list[Item], list[Item]]:
    """Return (optimize_items, confirm_items) in deterministic id order."""
    by_id = {i.id: i for i in items}
    opt = [by_id[i] for i in split.optimize_ids if i in by_id]
    con = [by_id[i] for i in split.confirm_ids if i in by_id]
    return opt, con
