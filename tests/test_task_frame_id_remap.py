"""LLM-local slot-id validation + consistent remapping (no model/provider calls).

The first open-world live preview produced a semantically good frame but fell back
because raw slot ids (T1/T2/I1/...) and dict-form dependency_edges were mishandled.
These tests pin the ingestion fix.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regimes_probe.agent.llm_task_frame import (  # noqa: E402
    LLMTaskFrameParser, build_task_frame, parser_warnings, validate_payload)

Q = ("There is a Mexican restaurant in NM 2.5 to 3.5 miles from a hotel opened in "
     "1955 and 21 to 29 miles from a museum founded in 2005. What is the founder's "
     "name and the year they were born?")


def _slot(sid, name, role, *, target, depends_on=None):
    return {"slot_id": sid, "slot_name": name, "slot_role": role,
            "is_target_answer_slot": target, "is_intermediate_slot": not target,
            "depends_on": depends_on or []}


def _user_payload(*, edges="dict"):
    """The frame shape the live LLM emitted: raw T*/I* ids, dict-form edges."""
    if edges == "dict":
        dep = [{"from": "I1", "to": "T1"}, {"from": "I2", "to": "T1"},
               {"from": "I3", "to": "T1"}, {"from": "T1", "to": "T2"}]
    elif edges == "omitted":
        dep = []
    elif edges == "inconsistent":              # explicit edges disagree with depends_on
        dep = [["I1", "T1"]]
    else:
        dep = edges
    return {
        "target_answer_slots": [
            _slot("T1", "restaurant_founder_identity", "person", target=True,
                  depends_on=["I1", "I2", "I3"]),
            _slot("T2", "restaurant_founder_birth_year", "date_or_time", target=True,
                  depends_on=["I1", "I2", "I3", "T1"])],
        "latent_slots": [
            _slot("I1", "mexican_restaurant_in_new_mexico", "organization", target=False),
            _slot("I2", "hotel_opened_1955", "organization", target=False),
            _slot("I3", "museum_founded_2005", "organization", target=False)],
        "constraints": [
            {"constraint_id": "C1", "text_span": "Mexican restaurant in NM",
             "semantic_label": "identity", "semantic_facets": ["identity"],
             "applies_to": ["I1"]},
            {"constraint_id": "C4", "text_span": "founder name and birth year",
             "semantic_label": "biographical", "semantic_facets": ["biographical"],
             "applies_to": ["T1", "T2", "I1"], "required": True, "priority": "high",
             "testable_claim": "founder X born in Y", "how_to_test": "read history",
             "supports_answer_slot_ids": ["T1", "T2"]}],
        "dependency_edges": dep,
        "known_context_terms": ["New Mexico", "1955", "2005"]}


def _parser(payload):
    return LLMTaskFrameParser(model_fn=lambda _p: json.dumps(payload), model="stub")


# ----------------------------------------------------- raw ids validate + accept
def test_raw_ids_validate_and_frame_is_accepted_as_llm():
    p = _user_payload(edges="dict")
    assert validate_payload(p, Q) == []                  # raw T*/I* ids are valid
    frame, meta = build_task_frame("x", Q, use_llm=True, parser=_parser(p))
    assert meta.parser_used == "llm", f"fell back: {meta.fallback_reason}"
    # remapped to a clean internal namespace, raw ids preserved.
    assert {s.raw_slot_id for s in frame.target_answer_slots} == {"T1", "T2"}
    assert meta.id_mapping == {"T1": "s0", "T2": "s1", "I1": "s2", "I2": "s3", "I3": "s4"}


def test_depends_on_does_not_trigger_unknown_slot_error():
    p = _user_payload(edges="dict")
    errs = validate_payload(p, Q)
    assert not any("dependency_edge_refs_unknown_slot" in e for e in errs)
    assert not any("depends_on" in e for e in errs)      # depends_on is advisory


def test_constraint_applies_to_raw_ids_are_remapped():
    frame, meta = build_task_frame("x", Q, use_llm=True, parser=_parser(_user_payload()))
    c4 = next(c for c in frame.constraints if c.constraint_id == "C4")
    # T1/T2/I1 -> s0/s1/s2 consistently.
    assert c4.applies_to == ["s0", "s1", "s2"]


def test_supports_answer_slot_ids_raw_ids_are_remapped():
    frame, _ = build_task_frame("x", Q, use_llm=True, parser=_parser(_user_payload()))
    c4 = next(c for c in frame.constraints if c.constraint_id == "C4")
    assert c4.supports_answer_slot_ids == ["s0", "s1"]


def test_omitted_dependency_edges_derived_from_depends_on():
    frame, meta = build_task_frame("x", Q, use_llm=True, parser=_parser(_user_payload(edges="omitted")))
    assert meta.parser_used == "llm"
    # derived from depends_on: (I1->T1),(I2->T1),(I3->T1),(I1->T2)... remapped.
    assert ("s2", "s0") in frame.dependency_edges          # I1 -> T1
    assert ("s0", "s1") in frame.dependency_edges          # T1 -> T2


def test_inconsistent_but_valid_edges_warn_not_fallback():
    p = _user_payload(edges="inconsistent")
    frame, meta = build_task_frame("x", Q, use_llm=True, parser=_parser(p))
    assert meta.parser_used == "llm"                       # accepted, not a fallback
    assert any("dependency_edges_inconsistent_with_depends_on" in w
               for w in meta.validation_warnings)
    # depends_on is preferred (richer than the single explicit edge).
    assert len(frame.dependency_edges) > 1


def test_genuinely_unknown_slot_ref_still_falls_back():
    p = _user_payload(edges="dict")
    p["constraints"][0]["applies_to"] = ["I9"]             # I9 does not exist
    frame, meta = build_task_frame("x", Q, use_llm=True, parser=_parser(p))
    assert meta.parser_used == "deterministic"
    assert any("constraint_refs_unknown_slot:I9" in e for e in meta.validation_errors)


def test_dict_form_edge_with_unknown_endpoint_falls_back():
    p = _user_payload(edges=[{"from": "I1", "to": "ZZ"}])  # ZZ unknown
    _, meta = build_task_frame("x", Q, use_llm=True, parser=_parser(p))
    assert meta.parser_used == "deterministic"
    assert any("dependency_edge_refs_unknown_slot" in e for e in meta.validation_errors)


# ----------------------------------------------------- preview shows id_mapping
def test_preview_shows_id_mapping(tmp_path):
    key = LLMTaskFrameParser(model="gpt-5.4-mini")._input_hash(Q)
    cache = tmp_path / "pc.json"
    cache.write_text(json.dumps({key: json.dumps(_user_payload(edges="dict"))}))
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "preview_task_frames.py"),
         "--use-llm", "--parser-cache", str(cache),
         "--task-frame-parser-model", "gpt-5.4-mini", Q],
        capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "raw_slot_ids:" in out.stdout and "'T1'" in out.stdout
    assert "id_mapping (raw->internal):" in out.stdout and "'T1': 's0'" in out.stdout
    assert "dependency_edges_remapped (frame):" in out.stdout
    assert "fallback_stage:" in out.stdout
    assert "parser_used: llm" in out.stdout
