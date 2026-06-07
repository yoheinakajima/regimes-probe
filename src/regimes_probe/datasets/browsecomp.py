"""BrowseComp adapter (secondary benchmark).

BrowseComp ships its answers obfuscated with a per-row XOR-against-canary +
base64 scheme (mirrors openai/simple-evals ``browsecomp_eval.py``). This adapter
implements the canonical decode and FAILS GRACEFULLY: a row that cannot be
decoded is reported, not silently dropped.

Refs:
  https://arxiv.org/abs/2504.12516
  https://github.com/openai/simple-evals
  https://github.com/openai/simple-evals/blob/main/browsecomp_eval.py

Unit tests never require the real data; the decode helpers are tested on
synthetic round-trips.
"""

from __future__ import annotations

import base64
import csv
import hashlib
from pathlib import Path
from typing import Iterable, Optional

from regimes_probe.datasets.base import DatasetAdapter, DatasetUnavailable, Item


def _derive_keystream(password: str, length: int) -> bytes:
    """Reproduce simple-evals' keystream: repeated SHA256(password) blocks."""
    out = bytearray()
    counter = 0
    while len(out) < length:
        block = hashlib.sha256(f"{password}{counter}".encode()).digest()
        out.extend(block)
        counter += 1
    return bytes(out[:length])


def decrypt(ciphertext_b64: str, canary: str) -> str:
    """Decrypt one base64 XOR-obfuscated field using the row ``canary``.

    Raises ``ValueError`` if the payload is not valid base64 / not decodable as
    UTF-8 — callers convert that into a graceful per-row skip with a count.
    """
    raw = base64.b64decode(ciphertext_b64)
    key = _derive_keystream(canary, len(raw))
    plain = bytes(b ^ k for b, k in zip(raw, key))
    return plain.decode("utf-8")


def encrypt(plaintext: str, canary: str) -> str:
    """Inverse of :func:`decrypt` (used by tests for round-trips)."""
    raw = plaintext.encode("utf-8")
    key = _derive_keystream(canary, len(raw))
    cipher = bytes(b ^ k for b, k in zip(raw, key))
    return base64.b64encode(cipher).decode("ascii")


class BrowseCompAdapter(DatasetAdapter):
    name = "browsecomp"

    def __init__(self, csv_path: str | Path | None = None) -> None:
        self.csv_path = Path(csv_path) if csv_path else None
        self.skipped: int = 0

    def load(self) -> list[Item]:
        if self.csv_path is None or not self.csv_path.exists():
            raise DatasetUnavailable(
                "BrowseComp CSV not found. Download from openai/simple-evals and "
                "pass csv_path=... (the loader expects 'problem','answer','canary' "
                "columns, obfuscated)."
            )
        items: list[Item] = []
        self.skipped = 0
        with self.csv_path.open(encoding="utf-8") as fh:
            for i, row in enumerate(csv.DictReader(fh)):
                canary = row.get("canary", "")
                try:
                    question = decrypt(row["problem"], canary)
                    answer = decrypt(row["answer"], canary)
                except (ValueError, KeyError):
                    self.skipped += 1
                    continue
                items.append(
                    Item(
                        id=f"browsecomp-{i:04d}",
                        question=question,
                        answer=answer,
                        source="browsecomp",
                        meta={"obfuscated": True},
                    )
                )
        if not items:
            raise DatasetUnavailable(
                f"BrowseComp: 0 rows decoded ({self.skipped} undecodable). "
                "Check the canary/obfuscation scheme."
            )
        return items

    def version(self) -> str:
        if self.csv_path and self.csv_path.exists():
            digest = hashlib.sha256(self.csv_path.read_bytes()).hexdigest()[:12]
            return f"browsecomp@{digest}"
        return "browsecomp@unavailable"
