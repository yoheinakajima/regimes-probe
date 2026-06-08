"""preview_task_frames.py diagnostics (no live model/provider calls).

Every case is dry-run: the LLM parser replays from a pre-seeded parser cache (or
misses), so no model is ever called.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regimes_probe.agent.llm_task_frame import LLMTaskFrameParser  # noqa: E402

SCRIPT = ROOT / "scripts" / "preview_task_frames.py"
MODEL = "gpt-5.4-mini"
Q = "Which 90s TV series starred an actor born in Tennessee?"

_BAD = {  # a required constraint that references an unknown slot + is not operational
    "target_answer_slots": [{"slot_id": "s0", "slot_name": "series",
                             "slot_role": "title_or_work", "is_target_answer_slot": True,
                             "is_intermediate_slot": False}],
    "latent_slots": [{"slot_id": "s1", "slot_name": "actor", "slot_role": "person",
                      "is_target_answer_slot": False, "is_intermediate_slot": True}],
    "constraints": [{"constraint_id": "c0", "text_span": "father in law enforcement",
                     "semantic_label": "biographical_identity",
                     "semantic_facets": ["biographical", "identity"],
                     "applies_to": ["s9"], "required": True, "priority": "high",
                     "affordances": ["can_search", "can_verify"]}],
    "known_context_terms": ["Tennessee"]}

_GOOD = {
    "target_answer_slots": [{"slot_id": "s0", "slot_name": "series",
                             "slot_role": "title_or_work", "is_target_answer_slot": True,
                             "is_intermediate_slot": False}],
    "latent_slots": [{"slot_id": "s1", "slot_name": "actor", "slot_role": "person",
                      "is_target_answer_slot": False, "is_intermediate_slot": True}],
    "constraints": [{"constraint_id": "c0", "text_span": "actor born in Tennessee",
                     "semantic_label": "biographical_birthplace",
                     "semantic_facets": ["biographical", "spatial"], "applies_to": ["s1"],
                     "required": True, "priority": "high",
                     "testable_claim": "actor born in Tennessee", "how_to_test": "read bio"}],
    "known_context_terms": ["Tennessee"], "dependency_edges": [["s1", "s0"]]}


def _seed(tmp_path, payload) -> str:
    key = LLMTaskFrameParser(model=MODEL)._input_hash(Q)
    p = tmp_path / "pc.json"
    p.write_text(json.dumps({key: json.dumps(payload)}))
    return str(p)


def _run(*args) -> str:
    out = subprocess.run([sys.executable, str(SCRIPT), *args],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_fallback_preview_prints_validation_errors_and_classification(tmp_path):
    out = _run("--use-llm", "--parser-cache", _seed(tmp_path, _BAD),
               "--task-frame-parser-model", MODEL, Q)
    assert "parser_used: deterministic" in out
    assert "fallback_reason: validation_failed" in out
    assert "open_world_validation: FAILED" in out
    assert "validation_errors:" in out and "constraint_refs_unknown_slot:s9" in out
    assert "bad_slot_references" in out                       # classification surfaced
    # raw LLM semantics shown even on fallback
    assert "biographical_identity" in out


def test_show_raw_includes_bounded_raw_output(tmp_path):
    out = _run("--use-llm", "--parser-cache", _seed(tmp_path, _BAD), "--show-raw",
               "--task-frame-parser-model", MODEL, Q)
    assert "RAW PARSER OUTPUT (bounded)" in out
    assert '"semantic_label": "biographical_identity"' in out


def test_cache_hit_displayed(tmp_path):
    out = _run("--use-llm", "--parser-cache", _seed(tmp_path, _GOOD),
               "--task-frame-parser-model", MODEL, Q)
    assert "parser_cache_hit: true" in out
    assert "model_called: false" in out
    assert "parser_used: llm" in out and "open_world_validation: passed" in out


def test_cache_miss_displayed(tmp_path):
    # empty cache file -> the question's key is absent -> miss.
    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    out = _run("--use-llm", "--parser-cache", str(empty),
               "--task-frame-parser-model", MODEL, Q)
    assert "parser_cache_hit: false" in out
    assert "fallback_reason: cache_miss_in_replay" in out


def test_fresh_parser_bypasses_cache(tmp_path):
    # cache HAS the key, but --fresh-parser must ignore it -> miss.
    out = _run("--use-llm", "--fresh-parser", "--parser-cache", _seed(tmp_path, _GOOD),
               "--task-frame-parser-model", MODEL, Q)
    assert "parser_cache_hit: false" in out
    assert "ignoring cache file" in out


def test_aliases_work(tmp_path):
    # --enable-llm-task-frame-parser == --use-llm ; --answer-model == --task-frame-parser-model
    out = _run("--enable-llm-task-frame-parser", "--answer-model", MODEL,
               "--parser-cache", _seed(tmp_path, _GOOD), Q)
    assert f"model: {MODEL}" in out
    assert "parser_used: llm" in out                          # alias engaged the LLM parser


def test_dry_run_makes_no_model_call(tmp_path):
    out = _run("--use-llm", "--parser-cache", _seed(tmp_path, _GOOD),
               "--task-frame-parser-model", MODEL, Q)
    assert "DRY-RUN: no model is called" in out
    assert "model_called: false" in out
    assert "model_called: true" not in out
