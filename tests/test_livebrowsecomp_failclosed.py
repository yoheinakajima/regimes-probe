"""LiveBrowseComp adapter fails closed on obfuscated rows (no network, no keys).

The HF release ships encoded ``problem``/``answer`` fields; the adapter must
refuse rather than pass an encrypted blob through as ``Item.question`` (which
would spend API calls on garbage).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from regimes_probe.datasets.base import DatasetUnavailable
from regimes_probe.datasets.browsecomp import encrypt
from regimes_probe.datasets.livebrowsecomp import (
    LiveBrowseCompAdapter, looks_obfuscated, looks_plaintext)

ROOT = Path(__file__).resolve().parents[1]
_CANARY = "LIVEBROWSECOMP-CANARY"


def _write(tmp_path, rows) -> Path:
    p = tmp_path / "rows.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def _encrypted_row():
    # shaped like the Hugging Face encrypted rows: problem/answer encoded, NO question.
    return {"id": "lbc-0001",
            "problem": encrypt("Who first announced the 2026 result?", _CANARY),
            "answer": encrypt("Dr. Jane Doe", _CANARY),
            "released_at": "2026-05-01"}


# ---------------------------------------------------------------- the core guard
def test_encrypted_row_fails_closed(tmp_path):
    path = _write(tmp_path, [_encrypted_row()])
    with pytest.raises(DatasetUnavailable) as exc:
        LiveBrowseCompAdapter(local_jsonl=path).load()
    assert "obfuscated" in str(exc.value).lower() or "fail-closed" in str(exc.value).lower()


def test_encrypted_problem_is_never_returned_as_question(tmp_path):
    row = _encrypted_row()
    path = _write(tmp_path, [row])
    try:
        items = LiveBrowseCompAdapter(local_jsonl=path).load()
    except DatasetUnavailable:
        items = []
    # the encrypted blob must NOT appear as any item's question
    assert all(it.question != row["problem"] for it in items)
    assert items == []   # fail-closed => no items at all


def test_plaintext_question_loads(tmp_path):
    path = _write(tmp_path, [{"id": "p1", "question": "What is the capital of France?",
                              "answer": "Paris", "released_at": "2026-01-01"}])
    items = LiveBrowseCompAdapter(local_jsonl=path).load()
    assert len(items) == 1 and items[0].question.endswith("?") and items[0].answer == "Paris"


def test_valid_canary_decodes_question_and_answer(tmp_path):
    path = _write(tmp_path, [_encrypted_row()])
    items = LiveBrowseCompAdapter(local_jsonl=path, canary=_CANARY).load()
    assert items[0].question == "Who first announced the 2026 result?"
    assert items[0].answer == "Dr. Jane Doe"      # short answer decoded, not passed through


def test_wrong_canary_fails_closed(tmp_path):
    path = _write(tmp_path, [_encrypted_row()])
    with pytest.raises(DatasetUnavailable):
        LiveBrowseCompAdapter(local_jsonl=path, canary="WRONG-CANARY").load()


def test_question_field_that_is_encoded_fails_closed(tmp_path):
    # even a 'question' field is refused if it doesn't look like plaintext
    path = _write(tmp_path, [{"id": "q1", "question": encrypt("hello there friend", _CANARY),
                              "answer": "x"}])
    with pytest.raises(DatasetUnavailable):
        LiveBrowseCompAdapter(local_jsonl=path).load()


def test_mixed_rows_fail_closed_if_any_obfuscated(tmp_path):
    path = _write(tmp_path, [
        {"id": "ok", "question": "What year is it?", "answer": "2026"},
        _encrypted_row()])
    with pytest.raises(DatasetUnavailable):
        LiveBrowseCompAdapter(local_jsonl=path).load()


# ---------------------------------------------------------------- helpers
def test_looks_plaintext_vs_obfuscated():
    assert looks_plaintext("Who is the CEO of Acme Robotics?")
    assert looks_plaintext("Paris")                       # short plaintext token
    assert not looks_plaintext(encrypt("a long obfuscated question here", _CANARY))
    assert looks_obfuscated(encrypt("a long obfuscated question here", _CANARY))
    assert not looks_obfuscated("")                        # empty is not "obfuscated"


# ---------------------------------------------------------------- run_live refusal
def test_run_live_refuses_encrypted_real_path(tmp_path):
    path = _write(tmp_path, [_encrypted_row()])
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_live.py"),
         "--dataset", "livebrowsecomp", "--dataset-path", str(path),
         "--optimize", "5", "--confirm", "10", "--budgets", "1",
         "--results-root", str(tmp_path / "out")],
        capture_output=True, text=True, timeout=120)
    assert out.returncode == 2, out.stdout + out.stderr
    assert "REFUSING" in out.stdout and "obfuscated" in out.stdout.lower()
    # and it did NOT fall back to the placeholder
    assert "real_shaped_placeholder" not in out.stdout
