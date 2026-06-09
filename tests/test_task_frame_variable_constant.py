"""Variable / constant / binding distinction in target-slot validation.

A target slot is an UNBOUND VARIABLE described by question language — not a leaked
answer. The validator must reject only a concrete known constant or a premature
binding, never a descriptor that reuses question text. No model/provider calls.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regimes_probe.agent.clue_resolution import _norm  # noqa: E402
from regimes_probe.agent.llm_task_frame import (  # noqa: E402
    LLMTaskFrameParser, build_task_frame, classify_target_slot, validate_payload)


def _kn(terms):
    return {_norm(t) for t in terms if _norm(t)}


def _decide(slot, question, *, kct=(), payload=None):
    return classify_target_slot(slot, payload or {}, question, _kn(kct))


def _slot(name, role, **kw):
    return {"slot_id": "s0", "slot_name": name, "slot_role": role, **kw}


# ---------------------------------------------------- descriptors PASS
def test_90s_tv_series_descriptor_passes_despite_overlap():
    q = "Which 90s TV series starred an actor born in Tennessee?"
    d, r = _decide(_slot("90s TV series", "title_or_work"), q,
                   kct=["90s TV series", "Tennessee"])
    assert d == "pass" and r == "unbound_variable_descriptor"


def test_founder_full_name_descriptor_passes():
    q = "What is the founder's full name and the year they were born?"
    d, _ = _decide(_slot("founder full name", "person"), q, kct=["New Mexico"])
    assert d == "pass"


def test_person_who_wrote_the_introduction_passes():
    q = "Who wrote the introduction of the report released by WHO?"
    d, _ = _decide(_slot("person who wrote the introduction", "person"), q, kct=["WHO"])
    assert d == "pass"


def test_report_descriptor_passes_when_question_asks_for_the_report():
    q = "Which global report was released by WHO in 2021?"
    d, _ = _decide(_slot("global report released by WHO", "title_or_work"), q, kct=["WHO"])
    assert d == "pass"


# ---------------------------------------------------- constants FAIL
def test_world_health_organisation_fails_when_question_asks_person():
    q = "Who wrote the introduction of the report released by WHO?"
    d, r = _decide(_slot("World Health Organisation", "organization"), q,
                   kct=["WHO", "World Health Organisation"])
    assert d == "fail" and r in ("concrete_known_constant", "exact_context_promoted_to_target")


def test_tennessee_fails_when_question_asks_series():
    q = "Which 90s TV series starred an actor born in Tennessee?"
    d, r = _decide(_slot("Tennessee", "location"), q, kct=["Tennessee"])
    assert d == "fail"


def test_premature_bound_value_fails():
    q = "Who founded the restaurant?"
    d, r = _decide(_slot("founder", "person", bound_value="Maria Lopez"), q)
    assert d == "fail" and r == "premature_bound_value"


def test_target_equal_to_known_constant_with_no_binding_constraints_fails():
    q = "Which founder is associated with New Mexico landmarks?"
    # "New Mexico" is a given location constant; no constraints bind it; role mismatch.
    d, r = _decide(_slot("New Mexico", "location"), "Who is the founder?",
                   kct=["New Mexico"])
    assert d == "fail"


# ---------------------------------------------------- ambiguous WARNS
def test_ambiguous_overlap_warns_not_fails():
    # lowercase non-type token, role-mismatched, no constraints, not in context.
    d, r = _decide(_slot("subject", "person"), "Which series is it?")
    assert d == "warn" and r == "ambiguous_descriptor"


# ---------------------------------------------------- end-to-end accept / fallback
def _payload(target_slot, *, kct, constraints=None):
    return {
        "target_answer_slots": [dict(target_slot, is_target_answer_slot=True,
                                     is_intermediate_slot=False)],
        "latent_slots": [], "constraints": constraints or [],
        "dependency_edges": [], "known_context_terms": list(kct)}


def test_end_to_end_descriptor_target_accepted_as_llm():
    q = "Which 90s TV series starred an actor born in Tennessee?"
    p = _payload(_slot("90s TV series", "title_or_work"),
                 kct=["90s TV series", "Tennessee"])
    _, meta = build_task_frame("x", q, use_llm=True,
                               parser=LLMTaskFrameParser(model_fn=lambda _p: json.dumps(p),
                                                         model="stub"))
    assert meta.parser_used == "llm", f"fell back: {meta.fallback_reason}"


def test_end_to_end_constant_target_falls_back():
    q = "Who wrote the introduction of the report released by WHO?"
    p = _payload(_slot("World Health Organisation", "organization"),
                 kct=["WHO", "World Health Organisation"])
    _, meta = build_task_frame("x", q, use_llm=True,
                               parser=LLMTaskFrameParser(model_fn=lambda _p: json.dumps(p),
                                                         model="stub"))
    assert meta.parser_used == "deterministic"
    assert any("known_context_promoted_to_target" in e for e in meta.validation_errors)


def test_frame_slot_carries_variable_semantics():
    q = "Which 90s TV series starred an actor born in Tennessee?"
    p = _payload(_slot("90s TV series", "title_or_work"),
                 kct=["90s TV series", "Tennessee"])
    frame, _ = build_task_frame("x", q, use_llm=True,
                                parser=LLMTaskFrameParser(model_fn=lambda _p: json.dumps(p),
                                                          model="stub"))
    s = frame.target_answer_slots[0]
    assert s.slot_status == "unbound_variable"
    assert s.bound_value == ""
    assert s.to_dict()["descriptor_text"] == "90s TV series"


# ---------------------------------------------------- graph projection vocabulary
def test_graph_projection_includes_variable_vocabulary():
    from decimal import Decimal
    from regimes_probe.agent.planner import AgentConfig, EpistemicAgent
    from regimes_probe.datasets.base import Item
    from regimes_probe.eval.debug import build_debug_record
    from regimes_probe.eval.grader import grade
    from regimes_probe.eval.projection import build_graph_projection
    from regimes_probe.eval.reward import RewardWeights, compute_rewards
    from regimes_probe.policy.contextual_bandit import BanditParams
    from regimes_probe.policy.memory import PolicyMemory
    from regimes_probe.policy.verification_policy import VerificationConfig
    from regimes_probe.tools.base import SearchProvider, SearchResponse, SearchResult
    from regimes_probe.tools.fake import FakePageFetch

    q = "Which 90s TV series starred an actor born in Tennessee?"
    p = _payload(_slot("90s TV series", "title_or_work"),
                 kct=["90s TV series", "Tennessee"],
                 constraints=[{"constraint_id": "C1", "text_span": "born in Tennessee",
                               "semantic_label": "biographical", "applies_to": ["s0"],
                               "required": True, "priority": "high",
                               "testable_claim": "actor born TN", "how_to_test": "read bio"}])

    class _P(SearchProvider):
        name = "generic_web_search"
        cost_per_call = Decimal("0.002")
        deterministic = True

        def available(self): return True

        def search(self, qq, *, limit=5, **o):
            return SearchResponse(provider=self.name, query=qq, results=(
                SearchResult(title="A 90s series", url="https://gov.example/s",
                             snippet="A 90s TV series with an actor born in Tennessee.",
                             source_authority=0.7, rank=0, extra={"item_id": "x"}),),
                cost=self.cost_per_call)

    item = Item(id="x", answer="Some Series", question=q)
    agent = EpistemicAgent(
        AgentConfig(available_tools=["generic_web_search", "page_fetch"],
                    enable_task_frame=True, enable_llm_task_frame_parser=True,
                    verification=VerificationConfig(min_support=2)),
        task_frame_parser=LLMTaskFrameParser(model_fn=lambda _p: json.dumps(p), model="stub"))
    tr = agent.attempt(item, PolicyMemory(BanditParams()),
                       {"generic_web_search": _P(), "page_fetch": FakePageFetch([])},
                       budget=3, explore=False, attempt_id="t")
    rec = build_debug_record(item=item, trace=tr, grade=grade(item, tr.final_answer),
                             reward=compute_rewards(tr, correct=False, gold_norms=[],
                                                    weights=RewardWeights.full(),
                                                    freshness_sensitive=False),
                             condition="no_memory_search", budget=3)
    proj = build_graph_projection("rid", runs=[], debug_records=[rec], eligibility_verdict={})
    types = {o["type"] for o in proj["objects"]}
    rels = {r["type"] for r in proj["relations"]}
    assert "known_context_term" in types and "slot_variable" in types
    assert "slot_descriptor" in types and "binding_status" in types
    assert "slot_has_descriptor" in rels and "known_context_not_answer" in rels


def test_policy_memory_stays_answer_free_variable_frames():
    from regimes_probe.agent.planner import AgentConfig, EpistemicAgent
    from regimes_probe.datasets.synthetic import SyntheticBrowseAdapter
    from regimes_probe.eval.harness import experience_phase
    from regimes_probe.policy.contextual_bandit import BanditParams
    from regimes_probe.policy.memory import PolicyMemory
    from regimes_probe.policy.policy_fragment import assert_no_answer_leakage
    from regimes_probe.tools.fake import build_fake_providers, load_corpus
    items = SyntheticBrowseAdapter().load()[:6]
    providers = build_fake_providers(load_corpus(ROOT / "fixtures" / "fake_search_corpus.json"),
                                     ["generic_web_search", "news_search"])
    agent = EpistemicAgent(AgentConfig(
        available_tools=["generic_web_search", "news_search", "page_fetch"],
        enable_task_frame=True))
    mem = PolicyMemory(BanditParams())
    experience_phase(items, agent, providers, mem, budget=3, passes=2)
    assert_no_answer_leakage(mem.snapshot().to_dict(), "snapshot")


# ---------------------------------------------------- preview prints slot_status + check
def test_preview_prints_slot_status_and_known_context_target_check(tmp_path):
    q = "Which 90s TV series starred an actor born in Tennessee?"
    p = _payload(_slot("90s TV series", "title_or_work"), kct=["90s TV series", "Tennessee"])
    key = LLMTaskFrameParser(model="gpt-5.4-mini")._input_hash(q)
    cache = tmp_path / "pc.json"
    cache.write_text(json.dumps({key: json.dumps(p)}))
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "preview_task_frames.py"),
         "--use-llm", "--parser-cache", str(cache),
         "--task-frame-parser-model", "gpt-5.4-mini", q],
        capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "known_context_target_check: passed" in out.stdout
    assert "target_overlap_allowed: true" in out.stdout
    assert "unbound_variable_descriptor" in out.stdout
    assert "/unbound_variable" in out.stdout            # frame slot shows status
