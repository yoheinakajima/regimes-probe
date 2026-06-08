"""ActiveGraph-native artifacts: graph projection, verdict/regime objects,
fragment lineage, and offline forked ablations (no providers, no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from regimes_probe.agent.answerer import build_closed_book_knowledge
from regimes_probe.agent.planner import (
    AgentConfig, EpistemicAgent, build_closed_book_agent)
from regimes_probe.datasets.livebrowsecomp import LiveBrowseCompAdapter
from regimes_probe.eval.reward import RewardWeights
from regimes_probe.live.cache import RecordingCache, ReplayMiss
from regimes_probe.live.providers import CachedProvider, NotArmed
from regimes_probe.live.runner import run_live_pipeline
from regimes_probe.policy.contextual_bandit import BanditParams
from regimes_probe.tools.fake import build_fake_providers, load_corpus

ROOT = Path(__file__).resolve().parents[1]
RS = ROOT / "fixtures" / "real_shaped"
RS_JSONL = RS / "livebrowsecomp_sample.jsonl"


def _cfg(**headline):
    c = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
    if headline:
        c.setdefault("headline", {}).update(headline)
    return c


def _items():
    return LiveBrowseCompAdapter(local_jsonl=RS_JSONL).load()


def _agents(cfg, tools, items):
    # Mirror exactly how live/fork.py rebuilds the agent so a fork's queries match
    # the parent's cached queries (as_of feeds freshness-sensitive reformulation).
    ac = AgentConfig(available_tools=tools + ["page_fetch"],
                     query_mode=cfg["policy"]["query_mode"],
                     stop_mode=cfg["policy"]["stop_mode"],
                     as_of=cfg.get("run", {}).get("as_of", "2026-06-01"))
    return (EpistemicAgent(ac),
            build_closed_book_agent(ac, build_closed_book_knowledge(items)))


def _run(tmp_path, *, conditions, budgets, run_id, is_real=False, cfg=None,
         providers=None, tools=("generic_web_search", "news_search"),
         cache=None, dataset_path=None, optimize=4, confirm=4):
    cfg = cfg or _cfg()
    tools = list(tools)
    items = _items()
    if providers is None:
        providers = build_fake_providers(load_corpus(RS / "real_shaped_corpus.json"), tools)
    ag, cb = _agents(cfg, tools, items)
    return run_live_pipeline(
        cfg, items, providers=providers, search_agent=ag, cb_agent=cb,
        cache=cache or RecordingCache(mode="off"), conditions=conditions, budgets=budgets,
        optimize=optimize, confirm=confirm, split_seed="t", run_id=run_id,
        results_root=str(tmp_path), dataset_label=("LiveBrowseComp" if is_real
                                                   else "real_shaped_placeholder"),
        dataset_version="v", dataset_path=dataset_path, is_real=is_real,
        search_tools=tools, weights=RewardWeights.full(), params=BanditParams(),
        live_settings={"tools": tools + ["page_fetch"]})


# --------------------------------------------------------------- graph projection
def test_graph_projection_has_expected_object_and_relation_types(tmp_path):
    out = _run(tmp_path, conditions=["closed_book", "no_memory_search", "policy_memory"],
               budgets=[1, 3], run_id="proj")
    proj = json.loads((Path(out["run_dir"]) / "graph_projection.json").read_text())
    oc, rc = proj["object_counts"], proj["relation_counts"]
    for t in ("benchmark_run", "benchmark_item", "question_attempt", "answer_attempt",
              "grade_result", "reward_assignment", "tool_call", "tool_response",
              "evidence_observation", "routing_plan", "query_plan", "memory_snapshot",
              "policy_fragment", "eligibility_verdict", "report", "claim_candidate"):
        assert oc.get(t, 0) >= 1, f"missing object type {t}"
    for r in ("attempt_for_item", "plan_for_attempt", "query_for_tool_call",
              "response_for_tool_call", "evidence_from_tool_response", "grade_for_answer",
              "reward_for_attempt", "memory_snapshot_contains_fragment",
              "policy_fragment_from_traces", "eligibility_for_run",
              "claim_supported_by_artifact"):
        assert rc.get(r, 0) >= 1, f"missing relation type {r}"
    # the projection is provenance-linked to the event-log replay
    assert proj["event_log"]["projection_matches"] is True
    # secret-free + answer-free (gold lives only in debug_questions.jsonl)
    blob = json.dumps(proj).lower()
    assert "api_key" not in blob and "authorization" not in blob


def test_projection_is_bounded_to_detail_sample(tmp_path):
    out = _run(tmp_path, conditions=["no_memory_search"], budgets=[3], run_id="bound")
    proj = json.loads((Path(out["run_dir"]) / "graph_projection.json").read_text())
    # every attempt is a node, but heavy sub-nodes are capped at detail_limit.
    n_attempts = proj["object_counts"]["question_attempt"]
    detailed = proj["object_counts"].get("routing_plan", 0)
    assert detailed <= proj["detail_limit"]
    assert detailed <= n_attempts


# --------------------------------------------------------------- eligibility verdict
def test_eligibility_verdict_object_in_report_and_projection(tmp_path):
    out = _run(tmp_path, conditions=["closed_book", "no_memory_search", "policy_memory"],
               budgets=[1], run_id="verdict")
    rd = Path(out["run_dir"])
    report = json.loads((rd / "report.json").read_text())
    v = report["eligibility_verdict"]
    for k in ("structurally_valid", "headline_eligible_memory_claim", "dataset_is_real",
              "conditions_present", "confirm_size", "min_confirm_size", "split_disjoint",
              "replay_pass", "memory_snapshot_leakage_pass", "raw_trace_archive_leakage_pass",
              "report_artifacts_leakage_pass", "same_conditions_pass", "frozen_confirm_memory",
              "no_live_updates_during_confirm", "budget_enforced", "provider_failure_rate",
              "max_provider_failure_rate", "reasons", "supporting_artifact_paths"):
        assert k in v, f"verdict missing {k}"
    proj = json.loads((rd / "graph_projection.json").read_text())
    ev = [o for o in proj["objects"] if o["type"] == "eligibility_verdict"]
    assert ev and ev[0]["data"]["structurally_valid"] == v["structurally_valid"]


def test_plumbing_run_structurally_valid_but_not_headline(tmp_path):
    out = _run(tmp_path, conditions=["closed_book", "no_memory_search"], budgets=[1],
               run_id="plumb", is_real=True)
    v = json.loads((Path(out["run_dir"]) / "report.json").read_text())["eligibility_verdict"]
    assert v["structurally_valid"] is True
    assert v["headline_eligible_memory_claim"] is False   # no policy_memory -> never headline


def test_all_four_real_can_be_headline_eligible(tmp_path):
    out = _run(tmp_path, conditions=["closed_book", "no_memory_search", "random_memory",
                                     "policy_memory"], budgets=[1], run_id="elig4",
               is_real=True, cfg=_cfg(min_confirm_size=3))
    v = json.loads((Path(out["run_dir"]) / "report.json").read_text())["eligibility_verdict"]
    assert v["structurally_valid"] is True
    assert v["headline_eligible_memory_claim"] is True, v["reasons"]


def test_synthetic_cannot_be_headline_eligible(tmp_path):
    out = _run(tmp_path, conditions=["closed_book", "no_memory_search", "random_memory",
                                     "policy_memory"], budgets=[1], run_id="syn4",
               is_real=False, cfg=_cfg(min_confirm_size=3))
    v = json.loads((Path(out["run_dir"]) / "report.json").read_text())["eligibility_verdict"]
    assert v["headline_eligible_memory_claim"] is False


# --------------------------------------------------------------- failure regimes
def test_failure_regime_objects_attach_to_attempts(tmp_path):
    out = _run(tmp_path, conditions=["closed_book", "no_memory_search"], budgets=[3],
               run_id="regimes")
    rd = Path(out["run_dir"])
    report = json.loads((rd / "report.json").read_text())
    assert report["failure_regime_summary"], "expected some failure regimes on a partial run"
    proj = json.loads((rd / "graph_projection.json").read_text())
    frs = [o for o in proj["objects"] if o["type"] == "failure_regime"]
    assert frs
    fr = frs[0]["data"]
    for k in ("regime", "item_id", "attempt_id", "condition", "budget", "triggers"):
        assert k in fr
    from regimes_probe.eval.failure_regime import FAILURE_REGIMES
    assert fr["regime"] in FAILURE_REGIMES
    # the regime node is connected to its attempt via regime_for_attempt
    rels = [r for r in proj["relations"] if r["type"] == "regime_for_attempt"]
    assert any(r["source"] == frs[0]["id"] for r in rels)


# --------------------------------------------------------------- fragment lineage
def test_policy_fragments_link_to_source_traces_without_answers(tmp_path):
    out = _run(tmp_path, conditions=["no_memory_search", "policy_memory"], budgets=[1],
               run_id="lineage")
    rd = Path(out["run_dir"])
    snap = json.loads((rd / "memory_snapshot.json").read_text())
    assert snap["fragments"], "policy_memory run should consolidate fragments"
    frag = next(iter(snap["fragments"].values()))
    assert frag["source_trace_ids"] and frag["source_attempt_ids"]
    assert frag["fragment_id"]
    # no answer-bearing keys in the fragment
    for forbidden in ("answer", "final_answer", "gold", "gold_answer", "solution"):
        assert forbidden not in frag
    # leakage scan still passes on the whole snapshot
    from regimes_probe.policy.policy_fragment import assert_no_answer_leakage
    assert_no_answer_leakage(snap, "snapshot")  # does not raise


# --------------------------------------------------------------- metrics keys
def test_report_metrics_have_new_keys(tmp_path):
    out = _run(tmp_path, conditions=["no_memory_search"], budgets=[3], run_id="metrics")
    report = json.loads((Path(out["run_dir"]) / "report.json").read_text())
    cell = report["metrics"]["no_memory_search@3"]
    for k in ("accuracy", "correct_per_tool_call", "mean_tool_calls", "first_tool_hit_rate",
              "provider_failure_rate", "failed_tool_call_count", "stale_source_error_rate",
              "false_stop_rate", "over_search_rate", "abstention_rate", "support_found_rate",
              "evidence_score_mean"):
        assert k in cell, f"metrics missing {k}"


# --------------------------------------------------------------- offline fork
def test_offline_fork_refuses_live_calls():
    """A replay-only provider over an empty cache must raise, never call out."""
    from regimes_probe.live.fork import build_replay_providers
    providers = build_replay_providers(["generic_web_search"], RecordingCache(mode="replay"))
    with pytest.raises((ReplayMiss, NotArmed)):
        providers["generic_web_search"].search("anything", limit=3)


def test_offline_fork_missing_cache_refuses(tmp_path):
    """Forking a run with no recording cache refuses gracefully (no spend)."""
    parent = _run(tmp_path, conditions=["no_memory_search", "policy_memory"], budgets=[1],
                  run_id="nocache", dataset_path=str(RS_JSONL))
    from regimes_probe.live.fork import fork_offline
    with pytest.raises(FileNotFoundError):
        fork_offline(parent["run_dir"], out_run_id="forkx", results_root=str(tmp_path / "f"))


def test_offline_fork_end_to_end_not_headline_no_spend(tmp_path):
    """Record a parent (cache populated), fork it offline, change a reward weight.
    The fork must reuse the cache (no live calls) and never be headline-eligible."""
    cache_path = tmp_path / "cache.json"
    cfg = _cfg()
    tools = ["generic_web_search", "news_search"]
    record = RecordingCache(cache_path, mode="record")
    inner = build_fake_providers(load_corpus(RS / "real_shaped_corpus.json"), tools)
    providers = {n: CachedProvider(p, record, armed=True) for n, p in inner.items()}
    _run(tmp_path, conditions=["no_memory_search", "policy_memory"], budgets=[1],
         run_id="parent", providers=providers, cache=record, dataset_path=str(RS_JSONL),
         cfg=cfg, tools=tools)
    assert cache_path.exists() and record.stores > 0

    from regimes_probe.live.fork import fork_offline
    # Fork the memory-variant condition (replays exactly from the frozen snapshot);
    # change only a reward param. No new spend.
    result = fork_offline(
        str(tmp_path / "parent"), out_run_id="fork", results_root=str(tmp_path),
        cache_path=str(cache_path),
        policy_overrides={"reward": {"preset": "full"}},
        conditions=["policy_memory"], budgets=[1])
    rd = Path(result["run_dir"])
    manifest = json.loads((rd / "run_manifest.json").read_text())
    assert manifest["offline_fork"] is True
    assert manifest["parent_run_id"] == "parent"
    assert result["eligibility"]["headline_eligible_memory_claim"] is False
    # no money spent: the fork served everything from cache (zero live calls).
    assert result["cache"]["live_calls"] == 0
