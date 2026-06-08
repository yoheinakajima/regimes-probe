"""Optional cached/replayable LLM task-frame parser (no provider/model calls).

Every "LLM" call here is a stub ``model_fn`` returning canned JSON; replay/cache
paths never invoke a model. Validates acceptance, schema/quality fallback, cache
behavior, provenance recording, answer-freeness, and the end-to-end wiring.
"""

from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace

from regimes_probe.agent import prompts
from regimes_probe.agent.llm_task_frame import (
    LLMTaskFrameParser, ParserCache, build_task_frame, validate_payload)
from regimes_probe.agent.planner import AgentConfig, EpistemicAgent
from regimes_probe.datasets.base import Item
from regimes_probe.policy.contextual_bandit import BanditParams
from regimes_probe.policy.memory import PolicyMemory
from regimes_probe.policy.policy_fragment import assert_no_answer_leakage
from regimes_probe.policy.verification_policy import VerificationConfig
from regimes_probe.tools.base import SearchProvider, SearchResponse, SearchResult
from regimes_probe.tools.fake import FakePageFetch


# --------------------------------------------------------------------- helpers
def _slot(sid, name, role, *, target=False, depends_on=None):
    return {"slot_id": sid, "slot_name": name, "slot_role": role,
            "is_target_answer_slot": target, "is_intermediate_slot": not target,
            "depends_on": depends_on or [], "expected_evidence_type": role}


def _con(cid, span, terms, ctype, applies):
    return {"constraint_id": cid, "text_span": span, "normalized_terms": terms,
            "constraint_type": ctype, "applies_to": applies,
            "specificity_score": 1.0, "discriminative_score": 1.0, "status": "unresolved"}


def _payload(targets, latents, constraints, *, known=None, edges=None):
    return {"target_answer_slots": targets, "latent_slots": latents,
            "constraints": constraints, "dependency_edges": edges or [],
            "known_context_terms": known or [], "unresolved_slots": [],
            "answer_shape_hints": [], "source_requirements": [], "parse_quality": 0.9}


def _parser(payload, *, model="stub-v1", counter=None):
    def model_fn(_prompt):
        if counter is not None:
            counter["n"] += 1
        return json.dumps(payload)
    return LLMTaskFrameParser(model_fn=model_fn, model=model)


# ---- representative valid frames -------------------------------------------
TV_Q = "Which TV series featured an actor who immigrated from the Caribbean and won an award in 2019?"
TV_PAYLOAD = _payload(
    [_slot("s0", "series", "title_or_work", target=True, depends_on=["s1"])],
    [_slot("s1", "actor", "person")],
    [_con("c0", "won an award in 2019", ["won", "award", "2019"], "temporal", ["s1"])],
    known=["Caribbean"], edges=[["s1", "s0"]])

RESTAURANT_Q = ("A Mexican restaurant in New Mexico, near a museum and a hotel, was "
                "founded by a chef born in which year?")
RESTAURANT_PAYLOAD = _payload(
    [_slot("s0", "year", "date_or_time", target=True, depends_on=["s3"])],
    [_slot("s1", "restaurant", "organization"), _slot("s2", "hotel", "organization"),
     _slot("s3", "chef", "person")],
    [_con("c0", "near a museum and a hotel", ["near", "museum", "hotel"], "distance", ["s1"]),
     _con("c1", "founded by a chef", ["founded", "chef"], "relation", ["s1", "s3"])],
    known=["New Mexico"], edges=[["s1", "s0"], ["s3", "s0"]])

WHO_Q = ("A report released by WHO had a foreword by one person and an introduction by "
         "another person. Who wrote the introduction?")
WHO_PAYLOAD = _payload(
    [_slot("s0", "introduction author", "person", target=True, depends_on=["s1"])],
    [_slot("s1", "report", "title_or_work"), _slot("s2", "foreword author", "person")],
    [_con("c0", "report released by WHO", ["report", "released"], "source", ["s1"])],
    known=["WHO"], edges=[["s1", "s0"]])

MANGA_Q = ("What is the title of this manga set in an elementary school that retells a "
           "classic story?")
MANGA_PAYLOAD = _payload(
    [_slot("s0", "manga", "title_or_work", target=True)],
    [],
    [_con("c0", "set in an elementary school", ["set", "elementary", "school"],
          "attribute", ["s0"])],
    known=[])

PAPER_Q = "A paper using a census sample — which journal published it?"
PAPER_PAYLOAD = _payload(
    [_slot("s0", "journal", "publication_or_source", target=True, depends_on=["s1"])],
    [_slot("s1", "paper", "title_or_work")],
    [_con("c0", "using a census sample", ["census", "sample"], "attribute", ["s1"])],
    known=[], edges=[["s1", "s0"]])

FIXTURES = [(TV_Q, TV_PAYLOAD, "title_or_work"),
            (RESTAURANT_Q, RESTAURANT_PAYLOAD, "date_or_time"),
            (WHO_Q, WHO_PAYLOAD, "person"),
            (MANGA_Q, MANGA_PAYLOAD, "title_or_work"),
            (PAPER_Q, PAPER_PAYLOAD, "publication_or_source")]


# --------------------------------------------------------------------- accept
def test_valid_llm_frame_is_accepted():
    frame, meta = build_task_frame("x", TV_Q, use_llm=True, parser=_parser(TV_PAYLOAD))
    assert meta.parser_used == "llm" and not meta.validation_errors
    assert meta.parse_quality >= 0.5
    assert [s.slot_role for s in frame.target_answer_slots] == ["title_or_work"]
    assert any(s.slot_role == "person" for s in frame.latent_slots)


def test_fixture_question_types_parse_with_correct_target_role():
    for q, payload, target_role in FIXTURES:
        frame, meta = build_task_frame("x", q, use_llm=True, parser=_parser(payload))
        assert meta.parser_used == "llm", f"{q!r} fell back: {meta.fallback_reason}"
        assert frame.target_answer_slots[0].slot_role == target_role
        # the parsed frame is answer-free (nothing answer-shaped in policy terms).
        assert_no_answer_leakage(frame.to_dict(), "llm_frame")


# --------------------------------------------------------------------- fallback
def test_invalid_json_falls_back_to_deterministic():
    p = LLMTaskFrameParser(model_fn=lambda _p: "sorry, here is the answer: 42", model="stub")
    frame, meta = build_task_frame("x", TV_Q, use_llm=True, parser=p)
    assert meta.parser_used == "deterministic" and meta.fallback_reason == "invalid_json"
    assert frame.target_answer_slots                      # deterministic still yields a frame


def test_missing_target_slot_falls_back():
    bad = _payload([], [_slot("s1", "actor", "person")], [])
    frame, meta = build_task_frame("x", TV_Q, use_llm=True, parser=_parser(bad))
    assert meta.parser_used == "deterministic"
    assert "no_target_slot" in meta.validation_errors


def test_constraint_referencing_unknown_slot_falls_back():
    bad = _payload([_slot("s0", "series", "title_or_work", target=True)], [],
                   [_con("c0", "x", ["x"], "temporal", ["s9"])])
    frame, meta = build_task_frame("x", TV_Q, use_llm=True, parser=_parser(bad))
    assert meta.parser_used == "deterministic"
    assert any("constraint_refs_unknown_slot" in e for e in meta.validation_errors)


def test_known_context_promoted_to_target_is_flagged():
    bad = _payload([_slot("s0", "WHO", "organization", target=True)], [], [], known=["WHO"])
    errors = validate_payload(bad, WHO_Q)
    assert any("known_context_promoted_to_target" in e for e in errors)
    frame, meta = build_task_frame("x", WHO_Q, use_llm=True, parser=_parser(bad))
    assert meta.parser_used == "deterministic"


def test_gold_like_answer_text_is_rejected():
    # a target slot naming a concrete entity absent from the question = guessed answer.
    bad = _payload([_slot("s0", "Casa Verde", "organization", target=True)], [], [])
    errors = validate_payload(bad, "Which restaurant won the prize?")
    assert "possible_answer_text_in_target_slot" in errors
    # and any forbidden answer key is rejected.
    assert any("forbidden_answer_key" in e
               for e in validate_payload({"final_answer": "x", **TV_PAYLOAD}, TV_Q))


def test_low_quality_frame_falls_back():
    # Passes schema validation (one target matches the head role) but scores below
    # the quality floor: too many targets, mostly-unknown slots, unattached
    # constraints, and intermediates with no dependency edges.
    weak = _payload(
        [_slot("s0", "series", "title_or_work", target=True),
         _slot("s4", "thing", "unknown", target=True),
         _slot("s5", "thing2", "unknown", target=True)],
        [_slot("s1", "a", "unknown"), _slot("s2", "b", "unknown")],
        [_con("c0", "x", ["x"], "attribute", []), _con("c1", "y", ["y"], "attribute", [])])
    frame, meta = build_task_frame("x", TV_Q, use_llm=True, parser=_parser(weak))
    assert meta.parser_used == "deterministic" and meta.fallback_reason == "low_quality"


# --------------------------------------------------------------------- cache / replay
def test_parser_cache_avoids_second_model_call():
    counter = {"n": 0}
    cache = ParserCache()
    p = LLMTaskFrameParser(model_fn=lambda _p: json.dumps(TV_PAYLOAD), cache=cache, model="m")
    p.model_fn = (lambda _p: (counter.__setitem__("n", counter["n"] + 1)
                              or json.dumps(TV_PAYLOAD)))
    build_task_frame("x", TV_Q, use_llm=True, parser=p)
    build_task_frame("x", TV_Q, use_llm=True, parser=p)
    assert counter["n"] == 1                              # second parse served from cache


def test_replay_only_with_empty_cache_falls_back_without_model():
    p = LLMTaskFrameParser(model_fn=None, replay_only=True)
    frame, meta = build_task_frame("x", TV_Q, use_llm=True, parser=p)
    assert meta.parser_used == "deterministic"
    assert meta.fallback_reason == "cache_miss_in_replay"


def test_file_backed_cache_replays_without_model(tmp_path):
    path = str(tmp_path / "cache.json")
    p1 = _parser(TV_PAYLOAD)
    p1.cache = ParserCache(path)
    build_task_frame("x", TV_Q, use_llm=True, parser=p1)   # writes cache file
    # fresh parser, NO model_fn, replay-only: must still accept from the cache file.
    p2 = LLMTaskFrameParser(model_fn=None, cache=ParserCache(path),
                            model=p1.model, replay_only=True)
    frame, meta = build_task_frame("x", TV_Q, use_llm=True, parser=p2)
    assert meta.parser_used == "llm" and meta.cache_hit


def test_prompt_version_and_hash_recorded():
    _, meta = build_task_frame("x", TV_Q, use_llm=True, parser=_parser(TV_PAYLOAD))
    pr = prompts.get("task_frame_parser")
    assert meta.prompt_version == pr.version
    assert meta.prompt_hash == pr.content_hash
    assert meta.input_hash and meta.output_hash


def test_deterministic_is_default_when_flag_off():
    frame, meta = build_task_frame("x", TV_Q, use_llm=False, parser=_parser(TV_PAYLOAD))
    assert meta.parser_used == "deterministic"
    assert meta.parse_quality_components                  # components scored either way


# --------------------------------------------------------------------- end-to-end
class _FrameProvider(SearchProvider):
    name = "generic_web_search"
    cost_per_call = Decimal("0.002")
    deterministic = True

    def available(self): return True

    def search(self, q, *, limit=5, **o):
        return SearchResponse(provider=self.name, query=q, results=(
            SearchResult(title="Some series — TV", url="https://gov.example/s",
                         snippet="A TV series featuring an actor.",
                         source_authority=0.7, rank=0, extra={"item_id": "x"}),), cost=self.cost_per_call)


def test_llm_parser_drives_loop_and_debug_records_provenance():
    from regimes_probe.eval.debug import build_debug_record
    from regimes_probe.eval.grader import grade
    from regimes_probe.eval.reward import RewardWeights, compute_rewards
    item = Item(id="x", answer="Some Series", question=TV_Q)
    providers = {"generic_web_search": _FrameProvider(), "page_fetch": FakePageFetch([])}
    agent = EpistemicAgent(
        AgentConfig(available_tools=["generic_web_search", "page_fetch"],
                    stop_mode="always_full", enable_task_frame=True,
                    enable_llm_task_frame_parser=True,
                    verification=VerificationConfig(min_support=2)),
        task_frame_parser=_parser(TV_PAYLOAD))
    tr = agent.attempt(item, PolicyMemory(BanditParams()), providers,
                       budget=3, explore=False, attempt_id="t")
    assert tr.task_frame_parse.get("parser_used") == "llm"
    assert tr.task_frame_parse.get("parse_quality", 0) >= 0.5
    rec = build_debug_record(item=item, trace=tr, grade=grade(item, tr.final_answer),
                             reward=compute_rewards(tr, correct=False, gold_norms=[],
                                                    weights=RewardWeights.full(),
                                                    freshness_sensitive=False),
                             condition="no_memory_search", budget=3).to_dict()
    assert rec["task_frame_parse"]["parser_used"] == "llm"
    assert "parse_quality" in rec["task_frame_parse"]
    assert rec["task_frame_parse"]["prompt_hash"]


def test_loop_falls_back_to_deterministic_when_parser_invalid():
    item = Item(id="x", answer="Some Series", question=TV_Q)
    providers = {"generic_web_search": _FrameProvider(), "page_fetch": FakePageFetch([])}
    bad_parser = LLMTaskFrameParser(model_fn=lambda _p: "not json", model="stub")
    agent = EpistemicAgent(
        AgentConfig(available_tools=["generic_web_search", "page_fetch"],
                    enable_task_frame=True, enable_llm_task_frame_parser=True),
        task_frame_parser=bad_parser)
    tr = agent.attempt(item, PolicyMemory(BanditParams()), providers,
                       budget=2, explore=False, attempt_id="t")
    assert tr.task_frame_parse.get("parser_used") == "deterministic"
    assert tr.task_frame and tr.task_frame["target_answer_slots"]
