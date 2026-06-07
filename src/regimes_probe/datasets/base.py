"""Dataset adapter interface + the benchmark :class:`Item`.

An ``Item`` carries the question and (for grading only) the gold answer. The
gold answer is consumed exclusively by the grader and is NEVER written into
policy memory (see ``docs/LEAKAGE_CONTROLS.md``). Adapters capture a version
string and content checksum so every run is auditable.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Item:
    """One benchmark question.

    ``meta`` holds harness/ground-truth hints (e.g. which fake tool carries the
    answer, freshness sensitivity). For real datasets ``meta`` may be sparse.
    """

    id: str
    question: str
    answer: str = ""
    answer_aliases: list[str] = field(default_factory=list)
    released_at: Optional[str] = None        # ISO date (for time-disjoint splits)
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def gold_answers(self) -> list[str]:
        out = [self.answer] if self.answer else []
        return out + list(self.answer_aliases)

    def public_dict(self) -> dict[str, Any]:
        """Answer-free projection — safe to log/store anywhere."""
        return {
            "id": self.id,
            "question": self.question,
            "released_at": self.released_at,
            "source": self.source,
            "meta": {k: v for k, v in self.meta.items() if k not in ("answer", "gold")},
        }


class DatasetAdapter(ABC):
    """Common interface for synthetic and real benchmark datasets."""

    name: str = "abstract_dataset"

    @abstractmethod
    def load(self) -> list[Item]:
        """Return the list of items. May raise a clear error if data is missing."""

    def version(self) -> str:
        """Human-readable dataset version/identifier."""
        return f"{self.name}@unknown"

    def checksum(self, items: Optional[list[Item]] = None) -> str:
        """Stable content checksum over the loaded items (questions + ids)."""
        items = items if items is not None else self.load()
        h = hashlib.sha256()
        for it in sorted(items, key=lambda i: i.id):
            h.update(it.id.encode())
            h.update(b"\x00")
            h.update(it.question.encode())
        return h.hexdigest()[:16]


class DatasetUnavailable(RuntimeError):
    """Raised when a real dataset's files/credentials are absent."""


def items_to_jsonl(items: list[Item]) -> str:
    return "\n".join(json.dumps(_item_full(i), sort_keys=True) for i in items)


def _item_full(i: Item) -> dict[str, Any]:
    return {
        "id": i.id,
        "question": i.question,
        "answer": i.answer,
        "answer_aliases": i.answer_aliases,
        "released_at": i.released_at,
        "source": i.source,
        "meta": i.meta,
    }
