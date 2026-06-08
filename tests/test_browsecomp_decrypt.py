"""BrowseComp decode must match openai/simple-evals EXACTLY (no network/keys).

The official scheme:
    key = sha256(password).digest()                       # 32 bytes
    keystream = key * (length // 32) + key[: length % 32]  # repeat/truncate
    plaintext = base64_decode(ciphertext) XOR keystream

The known-answer test below cross-checks our :func:`decrypt` against a ciphertext
computed by an INDEPENDENT inline implementation of that formula (not our module),
so it would catch any drift from the official method.
"""

from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path

import pytest

from regimes_probe.datasets.browsecomp import (
    BrowseCompAdapter, decrypt, derive_key, encrypt)
from regimes_probe.datasets.base import DatasetUnavailable
from regimes_probe.datasets.livebrowsecomp import looks_plaintext


# ---- independent inline reference (the official formula, not our module) ----
def _ref_key(password: str, length: int) -> bytes:
    key = hashlib.sha256(password.encode()).digest()
    return key * (length // len(key)) + key[: length % len(key)]


def _ref_encrypt(plaintext: str, password: str) -> str:
    raw = plaintext.encode()
    k = _ref_key(password, len(raw))
    return base64.b64encode(bytes(a ^ b for a, b in zip(raw, k))).decode()


def test_derive_key_matches_official_formula():
    for pw, length in [("BROWSECOMP", 1), ("x", 31), ("x", 32), ("x", 33),
                       ("canary-2026", 100), ("p", 0)]:
        assert derive_key(pw, length) == _ref_key(pw, length)


def test_known_plaintext_decrypts_with_official_method():
    # ciphertext pinned from the INDEPENDENT official-formula implementation.
    canary = "BROWSECOMP_CANARY_2026"
    cipher = "yByF96nz77YrKnOnLscyyiFumHCfYtIlVQnsileovab0EcC7ufz/9SRldvs="
    assert decrypt(cipher, canary) == "The quick brown fox jumps over the lazy dog."
    # short (answer-like) value
    assert decrypt("zBWSvqs=", canary) == "Paris"


def test_decrypt_agrees_with_independent_reference():
    canary = "row-canary-abc"
    for pt in ["Paris", "What is the capital of France?",
               "A longer answer spanning more than thirty-two bytes for keystream wrap."]:
        ct = _ref_encrypt(pt, canary)         # built by the official formula
        assert decrypt(ct, canary) == pt      # our decrypt must reproduce it


def test_round_trip_encrypt_decrypt():
    for pt in ["x", "hello world", "unicode: café résumé 日本語"]:
        assert decrypt(encrypt(pt, "c"), "c") == pt


def test_keystream_wraps_past_32_bytes():
    # >32-byte plaintext exercises the repeat/truncate path
    pt = "0123456789" * 5  # 50 chars > 32
    assert decrypt(_ref_encrypt(pt, "k"), "k") == pt


# ---- integration-safe: use the official CSV only if present locally, else skip ----
def _find_official_csv() -> Path | None:
    cand = []
    if os.environ.get("BROWSECOMP_CSV"):
        cand.append(Path(os.environ["BROWSECOMP_CSV"]))
    root = Path(__file__).resolve().parents[1]
    cand += [root / "data" / "browse_comp_test_set.csv",
             root / "data" / "browsecomp.csv"]
    for p in cand:
        if p.exists():
            return p
    return None


def test_official_csv_decodes_if_present():
    csv = _find_official_csv()
    if csv is None:
        pytest.skip("official BrowseComp CSV not present locally (set BROWSECOMP_CSV "
                    "or place at data/browse_comp_test_set.csv); no download in tests")
    adapter = BrowseCompAdapter(csv)
    items = adapter.load()                     # raises DatasetUnavailable if 0 decode
    assert len(items) > 0 and adapter.skipped == 0
    # decoded questions look like plaintext, not base64 blobs
    assert looks_plaintext(items[0].question)
    assert items[0].question.strip() != ""
