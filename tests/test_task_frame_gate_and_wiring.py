"""Flag guardrail, parser model_fn wiring, and answer-support gate (no model calls)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))   # make `_common` importable like run_live does

from regimes_probe.agent.action_planner import evaluate_answer_support
from regimes_probe.agent.hypothesis_table import (
    Candidate, EvidenceRecord, Hypothesis, HypothesisTable)
from regimes_probe.agent.llm_task_frame import LLMTaskFrameParser, build_task_frame
from regimes_probe.agent.task_frame import Constraint, Slot, TaskFrame
from regimes_probe.live.cache import RecordingCache, ReplayMiss
from regimes_probe.live.providers import NotArmed, build_task_frame_model_fn


# ------------------------------------------------------------ A. flag guardrail
def test_validate_flags_unit_raises():
    from _common import validate_task_frame_flags, LLM_PARSER_REQUIRES_TASK_FRAME
    with pytest.raises(ValueError) as exc:
        validate_task_frame_flags(task_frame=False, llm_parser=True)
    assert "requires --enable-task-frame" in str(exc.value)
    assert LLM_PARSER_REQUIRES_TASK_FRAME in str(exc.value)
    # valid combinations do not raise
    validate_task_frame_flags(task_frame=True, llm_parser=True)
    validate_task_frame_flags(task_frame=True, llm_parser=False)
    validate_task_frame_flags(task_frame=False, llm_parser=False)


def test_build_agent_raises_on_llm_parser_without_task_frame():
    from _common import build_agent
    cfg = {"policy": {"enable_task_frame": False, "enable_llm_task_frame_parser": True}}
    with pytest.raises(ValueError):
        build_agent(cfg, ["generic_web_search"])


def test_cli_llm_parser_without_task_frame_exits_nonzero(tmp_path):
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_live.py"),
         "--dataset", "real-shaped", "--optimize", "4", "--confirm", "6",
         "--budgets", "1", "--conditions", "closed_book,no_memory_search",
         "--enable-llm-task-frame-parser",          # WITHOUT --enable-task-frame
         "--results-root", str(tmp_path)],
        capture_output=True, text=True, timeout=120)
    assert out.returncode != 0, out.stdout
    assert "--enable-llm-task-frame-parser requires --enable-task-frame" in out.stdout
    # fails fast: no run artifacts were written.
    assert not list(tmp_path.glob("live-*"))


def test_cli_dry_run_with_both_flags_makes_no_calls(tmp_path):
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_live.py"),
         "--dataset", "real-shaped", "--optimize", "4", "--confirm", "6",
         "--budgets", "1", "--conditions", "closed_book,no_memory_search",
         "--enable-task-frame", "--enable-llm-task-frame-parser",
         "--results-root", str(tmp_path)],
        capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "DRY-RUN" in out.stdout and "NO providers were called" in out.stdout
    assert "task-frame parser:" in out.stdout      # estimate is printed
    run_dirs = list(tmp_path.glob("live-*"))
    assert run_dirs and not (run_dirs[0] / "report.json").exists()


# ------------------------------------------------------------ B. model_fn wiring
def test_model_fn_dry_run_raises_not_armed():
    fn = build_task_frame_model_fn("m", RecordingCache(mode="off"), armed=False)
    with pytest.raises(NotArmed):
        fn("some prompt")


def test_model_fn_replay_miss_raises():
    fn = build_task_frame_model_fn("m", RecordingCache(mode="replay"), armed=True)
    with pytest.raises(ReplayMiss):
        fn("some prompt")


def test_parser_falls_back_when_model_fn_raises_in_dry_run():
    # the parser wraps a model_fn that refuses (dry-run) -> deterministic fallback.
    fn = build_task_frame_model_fn("m", RecordingCache(mode="off"), armed=False)
    parser = LLMTaskFrameParser(model_fn=fn, model="m")
    frame, meta = build_task_frame("x", "Which museum opened in 1920?",
                                   use_llm=True, parser=parser)
    assert meta.parser_used == "deterministic"
    assert meta.fallback_reason.startswith("model_error")
    assert parser.model_calls == 1 and parser.fallback_count == 1


def test_model_fn_records_and_replays_without_second_network_call():
    # record once (armed) then replay (no network): same text, no model call.
    rec = RecordingCache(mode="record")

    class _FakeOpenAI:
        def __init__(self): self.responses = self
        def create(self, **kw): return type("R", (), {"output_text": '{"ok": 1}'})()

    import regimes_probe.live.providers as P
    # patch the OpenAI import target by injecting via sys.modules shim
    import types
    fake_mod = types.SimpleNamespace(OpenAI=_FakeOpenAI)
    sys.modules["openai"] = fake_mod
    try:
        fn = build_task_frame_model_fn("m", rec, armed=True)
        assert fn("prompt-A") == '{"ok": 1}'
        assert rec.calls == 1
    finally:
        del sys.modules["openai"]
    # now replay from the same cache: no network, served from store.
    rec.mode = "replay"
    fn2 = build_task_frame_model_fn("m", rec, armed=False)   # un-armed: must NOT call
    assert fn2("prompt-A") == '{"ok": 1}'
    assert rec.calls == 1                                    # unchanged: replay hit


# ------------------------------------------------------------ D. answer-support gate
def _gate_frame():
    f = TaskFrame(item_id="x")
    f.target_answer_slots = [Slot("s0", "restaurant", "organization",
                                  is_target_answer_slot=True)]
    f.constraints = [
        Constraint("c0", "won a James Beard award", ["james", "beard", "award"],
                   "relation", applies_to=["s0"], specificity_score=2.0,
                   discriminative_score=2.0, status="unresolved"),
        Constraint("c1", "in New Mexico", ["new", "mexico"], "location",
                   applies_to=["s0"], specificity_score=1.0,
                   discriminative_score=1.0, status="unresolved")]
    return f


def _register(t, cid, text, role="organization"):
    c = Candidate(candidate_id=cid, candidate_text=text, normalized_text=text.lower(),
                  inferred_role=role)
    t.candidates[text.lower()] = c
    t._by_id[cid] = c


def test_answer_supported_false_when_only_slots_populated_no_evidence():
    f = _gate_frame()
    t = HypothesisTable(f)
    _register(t, "cand1", "Casa Verde")
    # slot bound, but NO supporting evidence, constraints unresolved.
    h = Hypothesis(hypothesis_id="h1", slot_assignments={"s0": "cand1"},
                   evidence_ids=[], support_score=0.0)
    t.hypotheses = {"h1": h}
    t.evidence = []
    res = evaluate_answer_support(f, t)
    assert res.supported is False
    assert "no_clean_evidence_supports_target_or_answer_shape" in res.missing_support_reasons
    assert any(r.startswith("required_blocking_constraint_unresolved") for r in res.missing_support_reasons)
    assert "insufficient_constraint_support" in res.missing_support_reasons


def test_answer_supported_true_only_with_evidence_and_resolved_constraints():
    f = _gate_frame()
    f.constraints[0].status = "resolved"        # the high-priority target constraint
    f.constraints[1].status = "resolved"
    t = HypothesisTable(f)
    _register(t, "cand1", "Casa Verde")
    ev = EvidenceRecord(evidence_id="e1", source_tool="generic_web_search",
                        supports_slot_ids=["s0"], contamination_score=0.0)
    t.evidence = [ev]
    h = Hypothesis(hypothesis_id="h1", slot_assignments={"s0": "cand1"},
                   evidence_ids=["e1"], support_score=2.0,
                   constraint_status={"c0": "supported", "c1": "supported"})
    t.hypotheses = {"h1": h}
    res = evaluate_answer_support(f, t)
    assert res.supported is True, res.missing_support_reasons
    assert res.target_candidate == "Casa Verde"


def test_answer_support_false_when_supporting_evidence_is_contaminated():
    f = _gate_frame()
    f.constraints[0].status = "resolved"
    f.constraints[1].status = "resolved"
    t = HypothesisTable(f)
    _register(t, "cand1", "Casa Verde")
    ev = EvidenceRecord(evidence_id="e1", source_tool="generic_web_search",
                        supports_slot_ids=["s0"], contamination_score=1.0)  # contaminated
    t.evidence = [ev]
    h = Hypothesis(hypothesis_id="h1", slot_assignments={"s0": "cand1"},
                   evidence_ids=["e1"], support_score=2.0,
                   constraint_status={"c0": "supported", "c1": "supported"})
    t.hypotheses = {"h1": h}
    res = evaluate_answer_support(f, t)
    assert res.supported is False
    assert "no_clean_evidence_supports_target_or_answer_shape" in res.missing_support_reasons


def test_answer_support_false_when_constraint_contradicted():
    f = _gate_frame()
    f.constraints[0].status = "resolved"
    f.constraints[1].status = "contradicted"
    t = HypothesisTable(f)
    _register(t, "cand1", "Casa Verde")
    ev = EvidenceRecord(evidence_id="e1", source_tool="generic_web_search",
                        supports_slot_ids=["s0"], contamination_score=0.0)
    t.evidence = [ev]
    h = Hypothesis(hypothesis_id="h1", slot_assignments={"s0": "cand1"},
                   evidence_ids=["e1"], support_score=2.0,
                   constraint_status={"c0": "supported"})
    t.hypotheses = {"h1": h}
    res = evaluate_answer_support(f, t)
    assert res.supported is False
    assert any(r.startswith("constraints_contradicted") for r in res.missing_support_reasons)
