"""Dataset adapters: synthetic (tests) + BrowseComp / LiveBrowseComp (real)."""

from __future__ import annotations

from regimes_probe.datasets.base import (
    DatasetAdapter,
    DatasetUnavailable,
    Item,
    items_to_jsonl,
)
from regimes_probe.datasets.browsecomp import BrowseCompAdapter, decrypt, derive_key, encrypt
from regimes_probe.datasets.livebrowsecomp import LiveBrowseCompAdapter
from regimes_probe.datasets.synthetic import SyntheticBrowseAdapter

__all__ = [
    "DatasetAdapter",
    "DatasetUnavailable",
    "Item",
    "items_to_jsonl",
    "BrowseCompAdapter",
    "decrypt",
    "derive_key",
    "encrypt",
    "LiveBrowseCompAdapter",
    "SyntheticBrowseAdapter",
]
