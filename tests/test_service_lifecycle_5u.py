"""Level 5u offline tests: deterministic service-lifecycle completion + rejudgment
persistence accounting. Every pending obligation terminates in exactly one concrete,
event-derived service status (URL terminal states propagate to the whole URL group;
``pending_service_not_attempted_policy_error`` survives only as an invariant-violation
name); every attempted targeted rejudgment lands in exactly one explicit lifecycle bucket
(model error / cache-write failure / invalid response fail-closed — never silently
unaccounted); pinned safety metrics are explicit integers and ``consistency_violations``
is always a list. Synthetic fixtures only — zero live calls, no gold/benchmark content.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from regimes_probe.agent.candidate_frontier import CandidateFrontier, SlotCandidate, _hash
from regimes_probe.agent.evidence_interpreter import EvidenceInterpreter
from regimes_probe.agent.llm_task_frame import LLMTaskFrameParser, ParserCache, build_task_frame
from regimes_probe.agent.read_judgment import strict_rejudgment_key
from regimes_probe.eval.replay_validation import (
    consistency_violations, deterministic_replay_judge, load_legacy_run,
    validate_artifacts_dir)

ROOT = Path(__file__).resolve().parents[1]
_URL_A = "https://profiles.example/people/alpha-profile-page-with-the-relevant-biography"
_URL_B = "https://records.example/archive/beta-archive-page-with-historic-listings"
_URL_SOCIAL = "https://twitter.com/someone/status/1234567890123456789"
_BODY_SUPPORT = (("page boilerplate filler text. " * 40)
                 + " Avery Quinn studied at the Northvale Institute in 1987. ")
#: predicate-relevant (relation verb "studied") but WITHOUT the constraint's year, so the
#: deterministic stub judge honestly keeps the obligation open (requires_read).
_BODY_PARTIAL = (("page boilerplate filler text. " * 40)
                 + " Avery Quinn studied at the Northvale Institute for several terms. ")


def _frame_payload():
    return {
        "target_answer_slots": [{"slot_id": "D", "slot_name": "engineer",
                                 "slot_role": "person", "is_target_answer_slot": True,
                                 "is_intermediate_slot": False}],
        "latent_slots": [],
        "constraints": [{"constraint_id": "c1", "text_span": "studied at Northvale in 1987",
                         "normalized_terms": ["northvale", "1987"], "applies_to": ["D"],
                         "discriminative_score": 2.0, "specificity_score": 2.0,
                         "required": True, "priority": "high",
                         "blocks_answer_if_unresolved": True, "testable_claim": "x",
                         "how_to_test": "read", "semantic_label": "education"},
                        {"constraint_id": "c2", "text_span": "later worked in Valdorra",
                         "normalized_terms": ["valdorra", "worked"], "applies_to": ["D"],
                         "discriminative_score": 2.0, "specificity_score": 2.0,
                         "required": True, "priority": "high",
                         "blocks_answer_if_unresolved": True, "testable_claim": "y",
                         "how_to_test": "read", "semantic_label": "career"}],
        "dependency_edges": [], "known_context_terms": []}


def _frontier(judge=None):
    parser = LLMTaskFrameParser(model_fn=lambda _p: json.dumps(_frame_payload()),
                                model="stub")
    frame, _ = build_task_frame("x", "Who is the engineer who studied at Northvale in 1987?",
                                use_llm=True, parser=parser)
    judge = judge or deterministic_replay_judge()
    fr = CandidateFrontier(frame, interpreter=EvidenceInterpreter(judge=judge))
    return fr, judge


def _sid(fr) -> str:
    return next(iter(fr.slates))


def _add_candidate(fr, text="Avery Quinn", cid="cand1"):
    sid = _sid(fr)
    cand = SlotCandidate(candidate_id=cid, candidate_text=text,
                         normalized_text_hash=_hash(text.lower()), inferred_role="person",
                         slot_id=sid, constraints_unknown=["c1"], aliases=[text],
                         source_urls=[_URL_A])
    fr.slates[sid].candidates[cid] = cand
    fr.candidates_by_id[cid] = cand
    return cand


def _add_pending(fr, *, url=_URL_A, candidate_id="cand1", constraint_id="c1",
                 source_role="professional_profile", contaminated=False):
    fr._register_pending_read_judgment(
        candidate_id=candidate_id, slot_id=_sid(fr), constraint_id=constraint_id,
        source_url=url, source_role=source_role, contaminated=contaminated)
    return next(p for p in fr.pending_read_judgments.values()
                if p.candidate_id == candidate_id and p.constraint_id == constraint_id)


# -------------------------------------------- 1: blocked URL propagates to the whole group
def test_blocked_url_status_propagates_to_all_obligations_in_group():
    fr, _ = _frontier()
    p1 = _add_pending(fr, url=_URL_SOCIAL, candidate_id="cand1", constraint_id="c1")
    p2 = _add_pending(fr, url=_URL_SOCIAL, candidate_id="cand2", constraint_id="c2")
    assert fr.propose_pending_service_read(budget_remaining=4,
                                           page_fetch_available=True) is None
    fr.finalize_pending_service(budget_remaining=2, reading_tools_enabled=True)
    assert p1.service_status == "service_blocked_source_not_readable"
    assert p2.service_status == "service_blocked_source_not_readable"
    assert p1.service_block_reason == "social_media"
    assert p1.service_group_id and p1.service_group_id == p2.service_group_id
    # one of the two group members is marked as deduped (group state propagated).
    assert {p1.service_stage_reason, p2.service_stage_reason} >= {
        "service_deduped_to_url_group"}
    m = fr.metrics()
    assert m["pending_service_url_blocked_count"] == 1
    assert m["service_block_reason_counts"] == {"social_media": 1}
    assert m["service_invariant_violation_count"] == 0
    assert m["pending_obligation_without_service_reason_count"] == 0


# -------------------------------------------- 2: budget exhaustion propagates to the group
def test_budget_exhaustion_propagates_to_all_obligations_in_group():
    fr, _ = _frontier()
    p1 = _add_pending(fr, url=_URL_A, candidate_id="cand1", constraint_id="c1")
    p2 = _add_pending(fr, url=_URL_A, candidate_id="cand2", constraint_id="c2")
    fr.finalize_pending_service(budget_remaining=0, reading_tools_enabled=True)
    assert p1.service_status == "service_budget_exhausted"
    assert p2.service_status == "service_budget_exhausted"
    assert p1.service_budget_remaining_when_decided == 0
    rec = fr._service_rec(_URL_A)
    assert rec["status"] == "service_budget_exhausted"
    assert fr.metrics()["pending_service_url_budget_exhausted_count"] == 1


# -------------------------------------------- 3: success serves the whole group
def test_successful_read_marks_all_group_obligations_success_or_satisfied():
    fr, judge = _frontier()
    _add_candidate(fr)
    p1 = _add_pending(fr, url=_URL_A, candidate_id="cand1", constraint_id="c1")
    p2 = _add_pending(fr, url=_URL_A, candidate_id="cand2", constraint_id="c2")
    sp = fr.propose_pending_service_read(budget_remaining=4, page_fetch_available=True)
    assert sp is not None
    fr.record_pending_service_outcome(url=_URL_A, tool=sp.read_decision.tool,
                                      body_chars=len(_BODY_SUPPORT),
                                      page_fetch_available=True)
    fr.route_read_into_pending_judgments(candidate_id="cand1", source_url=_URL_A,
                                         read_text=_BODY_SUPPORT, judge=judge)
    fr.finalize_pending_service(budget_remaining=2, reading_tools_enabled=True)
    allowed = {"service_attempted_success", "service_already_satisfied_by_same_url_read"}
    assert p1.service_status in allowed and p2.service_status in allowed
    assert p1.service_read_success is True and p2.service_read_success is True
    assert fr.metrics()["service_invariant_violation_count"] == 0


# -------------------------------------------- 4: policy_error never ordinary control flow
def _record(pendings, *, service_urls=(), events=(), slot_id="s0"):
    return {
        "item_id": "u-001",
        "question_preview": "Who is the engineer who studied at the Northvale Institute in 1987?",
        "task_frame": {
            "target_answer_slots": [{"slot_id": slot_id, "slot_name": "engineer",
                                     "descriptor": "engineer"}],
            "constraints": [{"constraint_id": "c1",
                             "text_span": "studied at the Northvale Institute in 1987",
                             "normalized_terms": ["northvale", "institute", "1987"],
                             "applies_to": [slot_id]}]},
        "candidate_frontier": {
            "events": list(events), "events_truncated": 0, "interpretations": [],
            "pending_read_judgments": list(pendings),
            "pending_service_urls": list(service_urls),
            "metrics": {"unrelated_read_executed_while_pending_count": 0},
            "slates": [{"slot_role": "person",
                        "top_candidates": [{"candidate_text_preview": "Avery Quinn"}]}]},
        "calls": [], "evidence": []}


def _pending_dict(pid, url, *, status="", suppressed_reason="", **extra):
    d = {"pending_read_judgment_id": pid, "candidate_id": f"cand_{pid}", "slot_id": "s0",
         "constraint_id": "c1", "source_url": url, "source_url_host": "",
         "source_subject": "Avery Quinn", "missing_anchors": [], "target_terms": [],
         "created_step": 1, "resolved_step": None, "resolution": "open",
         "read_selected": False, "passage_preview": "", "source_role": "article",
         "suppressed_reason": suppressed_reason, "selected_read_url": "", "read_tool": "",
         "service_status": status}
    d.update(extra)
    return d


def test_old_5t_statuses_reclassified_without_policy_error(tmp_path):
    # a 5t-era record: blocked / budget / suppressed obligations with OLD status names
    # plus an OLD-named URL registry — must reclassify into concrete reasons.
    pendings = [
        _pending_dict("prj1", _URL_SOCIAL, status="pending_service_url_disallowed"),
        _pending_dict("prj2", _URL_SOCIAL, status="pending_service_url_disallowed"),
        _pending_dict("prj3", _URL_B, status="pending_service_budget_exhausted"),
        _pending_dict("prj4", _URL_A, status="pending_service_suppressed_source",
                      suppressed_reason="noise_source_role:generic_definition_page"),
    ]
    service_urls = [
        {"url": _URL_SOCIAL, "url_host": "twitter.com", "obligation_ids": ["prj1", "prj2"],
         "attempted": False, "blocked_reason": "url_disallowed",
         "status": "pending_service_url_disallowed"},
        {"url": _URL_B, "url_host": "records.example", "obligation_ids": ["prj3"],
         "attempted": False, "blocked_reason": "",
         "status": "pending_service_budget_exhausted"},
    ]
    run = tmp_path / "run"
    run.mkdir()
    (run / "debug_questions.jsonl").write_text(
        json.dumps(_record(pendings, service_urls=service_urls)) + "\n")
    (run / "llm_evidence_judge_cache.json").write_text("{}")
    res = load_legacy_run(run)
    pm = res.metrics
    assert pm["stage_reason_counts"].get("pending_service_not_attempted_policy_error", 0) == 0
    assert pm["stage_reason_counts"].get("service_blocked_disallowed_tool", 0) == 2
    assert pm["stage_reason_counts"].get(
        "pending_service_not_attempted_budget_exhausted", 0) == 1
    assert pm["pending_service_not_attempted_policy_error_count"] == 0
    assert pm["service_invariant_violation_count"] == 0
    # URL terminal categories partition the registry (blocked=1, budget=1).
    assert pm["pending_service_url_count"] == 2
    assert pm["pending_service_url_blocked_count"] == 1
    assert pm["pending_service_url_budget_exhausted_count"] == 1
    assert pm["blocked_service_urls_sample"]
    assert pm["consistency_violations"] == []


# -------------------------------------------- 5: still-open verdict recorded + replayed
def _persist_run(tmp_path, fr, judge_unused, body, slot_id) -> Path:
    run = tmp_path / "run"
    (run / "cache").mkdir(parents=True, exist_ok=True)
    cache = {"mode": "auto", "store_raw": False, "entries": [
        {"request_hash": "h1", "provider": "page_fetch", "name": "page_fetch",
         "request_meta": {"query": _URL_A},
         "response": {"provider": "page_fetch", "query": _URL_A,
                      "results": [{"title": "Avery Quinn", "url": _URL_A,
                                   "snippet": body, "rank": 0}], "error": None}}]}
    (run / "cache" / "provider_cache.json").write_text(json.dumps(cache))
    dbg = fr.to_debug()
    record = _record(dbg["pending_read_judgments"],
                     service_urls=dbg["pending_service_urls"], events=dbg["events"],
                     slot_id=slot_id)
    record["candidate_frontier"]["metrics"] = {
        "unrelated_read_executed_while_pending_count":
            dbg["metrics"]["unrelated_read_executed_while_pending_count"]}
    record["task_frame"]["target_answer_slots"][0]["slot_id"] = slot_id
    record["task_frame"]["constraints"][0]["applies_to"] = [slot_id]
    (run / "debug_questions.jsonl").write_text(json.dumps(record) + "\n")
    return run


def test_still_open_rejudgment_recorded_and_replays_judged_unclosed(tmp_path):
    run = tmp_path / "run"
    (run / "cache").mkdir(parents=True)
    judge = deterministic_replay_judge()
    judge.cache = ParserCache(str(run / "llm_evidence_judge_cache.json"))
    fr, _ = _frontier(judge=judge)
    _add_candidate(fr)
    p = _add_pending(fr)
    # candidate + constraint term present but NOT the year -> stub judge keeps it open.
    fr.route_read_into_pending_judgments(candidate_id="cand1", source_url=_URL_A,
                                         read_text=_BODY_PARTIAL, judge=judge)
    assert p.rejudgment_status == "rejudgment_attempted_recorded"
    assert isinstance(judge.cache.get(strict_rejudgment_key(p.pending_read_judgment_id)),
                      dict)
    fr.finalize_pending_service(budget_remaining=0, reading_tools_enabled=True)
    run = _persist_run(tmp_path, fr, judge, _BODY_PARTIAL, p.slot_id)
    res = load_legacy_run(run)
    o = res.obligations[0]
    assert o.pipeline_status == "judged_unclosed"
    assert o.closure_code == "requires_read_still_open"
    assert o.rejudgment_status == "rejudgment_attempted_recorded"
    pm = res.metrics
    assert pm["closed_count"] == 0
    assert pm["requires_read_still_open_counted_closed_count"] == 0
    assert pm["rejudgment_invariant_violation_count"] == 0
    assert pm["rejudgment_status_counts"] == {"rejudgment_attempted_recorded": 1}
    assert pm["consistency_violations"] == []


# -------------------------------------------- 6: invalid model response fails closed
class _WeirdJudge:
    """A judge stub returning an out-of-vocabulary verdict (simulates a malformed model)."""
    enabled = True

    def __init__(self):
        self.cache = ParserCache()

    def judge(self, **kw):
        return SimpleNamespace(judgment="banana_unparseable_verdict")


def test_invalid_model_response_recorded_fail_closed_not_missing():
    fr, _ = _frontier()
    _add_candidate(fr)
    p = _add_pending(fr)
    judge = _WeirdJudge()
    fr.route_read_into_pending_judgments(candidate_id="cand1", source_url=_URL_A,
                                         read_text=_BODY_SUPPORT, judge=judge)
    assert p.rejudgment_status == "rejudgment_attempted_invalid_response_recorded_fail_closed"
    rec = judge.cache.get(strict_rejudgment_key(p.pending_read_judgment_id))
    assert isinstance(rec, dict) and rec["verdict"] == "requires_read"   # fail closed
    assert p.closure_state == "requires_read_still_open"
    m = fr.metrics()
    assert m["pending_read_targeted_rejudgment_closed_count"] == 0
    assert m["rejudgment_status_counts"] == {
        "rejudgment_attempted_invalid_response_recorded_fail_closed": 1}
    assert m["rejudgment_invariant_violation_count"] == 0


class _RaisingJudge:
    enabled = True

    def __init__(self):
        self.cache = ParserCache()

    def judge(self, **kw):
        raise RuntimeError("model exploded")


def test_model_error_fails_closed_with_explicit_bucket():
    fr, _ = _frontier()
    _add_candidate(fr)
    p = _add_pending(fr)
    fr.route_read_into_pending_judgments(candidate_id="cand1", source_url=_URL_A,
                                         read_text=_BODY_SUPPORT, judge=_RaisingJudge())
    assert p.rejudgment_status == "rejudgment_attempted_model_error"
    assert p.closure_state == "requires_read_still_open"
    assert fr.metrics()["pending_read_targeted_rejudgment_closed_count"] == 0


# -------------------------------------------- 7: cache write failure surfaced + sampled
class _BrokenCacheJudge:
    enabled = True

    class _BrokenCache:
        def get(self, k):
            return None

        def put(self, k, v):
            raise IOError("disk full")

    def __init__(self):
        self.cache = self._BrokenCache()

    def judge(self, **kw):
        return SimpleNamespace(judgment="requires_read")


def test_cache_write_failure_surfaced_with_bounded_sample(tmp_path):
    fr, _ = _frontier()
    _add_candidate(fr)
    p = _add_pending(fr)
    fr.route_read_into_pending_judgments(candidate_id="cand1", source_url=_URL_A,
                                         read_text=_BODY_SUPPORT,
                                         judge=_BrokenCacheJudge())
    assert p.rejudgment_status == "rejudgment_attempted_cache_write_failed"
    assert p.rejudgment_attempted and not p.rejudgment_recorded
    fr.finalize_pending_service(budget_remaining=0, reading_tools_enabled=True)
    run = _persist_run(tmp_path, fr, None, _BODY_SUPPORT, p.slot_id)
    (run / "llm_evidence_judge_cache.json").write_text("{}")
    res = load_legacy_run(run)
    pm = res.metrics
    assert pm["rejudgment_status_counts"] == {"rejudgment_attempted_cache_write_failed": 1}
    assert pm["rejudgment_cache_write_failed_samples"]
    assert len(pm["rejudgment_cache_write_failed_samples"]) <= 5
    assert pm["rejudgment_invariant_violation_count"] == 0
    # the lifecycle equation reconciles: 1 attempted == 1 cache_write_failed.
    assert pm["pending_read_targeted_rejudgment_attempted_count"] == 1
    assert pm["consistency_violations"] == []


# -------------------------------------------- 3b: claimed-recorded but absent = violation
def test_claimed_recorded_verdict_with_absent_entry_is_invariant_violation(tmp_path):
    fr, judge = _frontier()
    _add_candidate(fr)
    p = _add_pending(fr)
    fr.route_read_into_pending_judgments(candidate_id="cand1", source_url=_URL_A,
                                         read_text=_BODY_SUPPORT, judge=judge)
    assert p.rejudgment_status == "rejudgment_attempted_recorded"
    fr.finalize_pending_service(budget_remaining=0, reading_tools_enabled=True)
    run = _persist_run(tmp_path, fr, judge, _BODY_SUPPORT, p.slot_id)
    # the judge cache was NOT persisted to disk: the claimed-recorded entry is absent.
    (run / "llm_evidence_judge_cache.json").write_text("{}")
    res = load_legacy_run(run)
    pm = res.metrics
    assert pm["rejudgment_invariant_violation_count"] == 1
    assert pm["pending_read_targeted_rejudgment_missing_count"] == 1
    assert pm["rejudgment_invariant_violation_samples"]
    # the equation still reconciles (1 attempted == 1 missing_invariant_violation).
    assert pm["consistency_violations"] == []


# -------------------------------------------- 8/9: pinned metrics explicit + list type
def test_unrelated_read_metric_always_integer_and_consistency_always_list(tmp_path):
    # (a) a run dir with records.
    pendings = [_pending_dict("prj1", _URL_A, status="service_budget_exhausted")]
    service_urls = [{"url": _URL_A, "url_host": "profiles.example",
                     "obligation_ids": ["prj1"], "attempted": False,
                     "blocked_reason": "", "status": "service_budget_exhausted"}]
    run = tmp_path / "run"
    run.mkdir()
    (run / "debug_questions.jsonl").write_text(
        json.dumps(_record(pendings, service_urls=service_urls)) + "\n")
    (run / "llm_evidence_judge_cache.json").write_text("{}")
    res = load_legacy_run(run)
    assert isinstance(res.metrics["unrelated_read_executed_while_pending_count"], int)
    assert res.metrics["unrelated_read_executed_while_pending_count"] == 0
    assert isinstance(res.metrics["consistency_violations"], list)
    assert res.metrics["consistency_violations"] == []
    assert isinstance(res.metrics["debug_snippet_judged_count"], int)
    assert isinstance(res.metrics["requires_read_still_open_counted_closed_count"], int)
    # (b) nothing inspectable: pinned metrics still explicit, never None/absent.
    res2 = validate_artifacts_dir(tmp_path / "does-not-exist")
    assert res2.metrics["unrelated_read_executed_while_pending_count"] == 0
    assert res2.metrics["consistency_violations"] == []


# -------------------------------------------- accounting reconciliation end-to-end
def test_full_lifecycle_reconciles_url_and_obligation_and_rejudgment_accounting(tmp_path):
    run = tmp_path / "run"
    (run / "cache").mkdir(parents=True)
    judge = deterministic_replay_judge()
    judge.cache = ParserCache(str(run / "llm_evidence_judge_cache.json"))
    fr, _ = _frontier(judge=judge)
    _add_candidate(fr)
    served = _add_pending(fr, url=_URL_A, candidate_id="cand1", constraint_id="c1")
    blocked = _add_pending(fr, url=_URL_SOCIAL, candidate_id="cand2", constraint_id="c2")
    suppressed = _add_pending(fr, url=_URL_B, candidate_id="cand3", constraint_id="c1",
                              source_role="generic_definition_page")
    sp = fr.propose_pending_service_read(budget_remaining=4, page_fetch_available=True)
    assert sp is not None and getattr(sp.read_obs, "url", "") == _URL_A
    fr.record_pending_service_outcome(url=_URL_A, tool=sp.read_decision.tool,
                                      body_chars=len(_BODY_SUPPORT),
                                      page_fetch_available=True)
    fr.route_read_into_pending_judgments(candidate_id="cand1", source_url=_URL_A,
                                         read_text=_BODY_SUPPORT, judge=judge)
    assert fr.propose_pending_service_read(budget_remaining=3,
                                           page_fetch_available=True) is None
    fr.finalize_pending_service(budget_remaining=3, reading_tools_enabled=True)
    assert served.service_status == "service_attempted_success"
    assert blocked.service_status == "service_blocked_source_not_readable"
    assert suppressed.service_status == "service_suppressed_non_executable"
    run = _persist_run(tmp_path, fr, judge, _BODY_SUPPORT, served.slot_id)
    res = load_legacy_run(run)
    pm = res.metrics
    # URL partition: 1 attempted + 1 blocked == 2 registered (suppressed never registers).
    assert pm["pending_service_url_count"] == 2
    assert pm["pending_service_url_attempted_count"] == 1
    assert pm["pending_service_url_blocked_count"] == 1
    assert pm["pending_service_url_invariant_violation_count"] == 0
    # obligation terminal statuses reconcile with the registry obligation count.
    assert pm["pending_service_obligation_count"] == \
        pm["pending_service_obligation_terminal_status_total"]
    assert pm["service_invariant_violation_count"] == 0
    # rejudgment lifecycle: 1 attempted == 1 recorded; closed via strict verdict.
    assert pm["pending_read_targeted_rejudgment_attempted_count"] == 1
    assert pm["rejudgment_status_counts"] == {"rejudgment_attempted_recorded": 1}
    assert pm["rejudgment_invariant_violation_count"] == 0
    assert pm["closed_count"] == 1
    assert pm["consistency_violations"] == []
    assert pm["live_provider_calls"] == 0 and pm["live_model_calls"] == 0
