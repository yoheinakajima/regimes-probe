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


def derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length key from ``password`` — EXACTLY as openai/simple-evals.

    key = sha256(password).digest() (32 bytes), repeated/truncated to ``length``::

        key = sha256(password.encode()).digest()
        return key * (length // len(key)) + key[: length % len(key)]

    (The earlier implementation used per-block ``sha256(f"{password}{counter}")``,
    which does NOT match the official BrowseComp CSV and would fail to decode it.)
    """
    key = hashlib.sha256(password.encode()).digest()
    return key * (length // len(key)) + key[: length % len(key)]


#: Backwards-compatible alias (kept for any external callers).
_derive_keystream = derive_key


def decrypt(ciphertext_b64: str, canary: str) -> str:
    """Decrypt one base64 XOR-obfuscated field using the row ``canary`` (password).

    Matches simple-evals: base64-decode, derive the repeated-SHA256 key, XOR, then
    UTF-8 decode. Raises ``ValueError``/``UnicodeDecodeError`` if the payload is not
    valid base64 / not decodable — callers convert that into a graceful per-row
    skip with a count.
    """
    raw = base64.b64decode(ciphertext_b64)
    key = derive_key(canary, len(raw))
    plain = bytes(b ^ k for b, k in zip(raw, key))
    return plain.decode("utf-8")


def encrypt(plaintext: str, canary: str) -> str:
    """Inverse of :func:`decrypt` (XOR is symmetric). Used by tests/fixtures."""
    raw = plaintext.encode("utf-8")
    key = derive_key(canary, len(raw))
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
