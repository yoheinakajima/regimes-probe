"""Debug artifacts, populated metrics, and layer-separated leakage (no providers)."""

from __future__ import annotations

import io
import json
import urllib.error
from decimal import Decimal
from pathlib import Path

import yaml

from regimes_probe.agent.answerer import build_closed_book_knowledge
from regimes_probe.agent.planner import AgentConfig, EpistemicAgent, build_closed_book_agent
from regimes_probe.datasets.base import Item
from regimes_probe.datasets.livebrowsecomp import LiveBrowseCompAdapter
from regimes_probe.eval.leakage import (
    leakage_check_details, scan_gold_in_payload, snapshot_leakage)
from regimes_probe.eval.reward import RewardWeights
from regimes_probe.live.cache import RecordingCache
from regimes_probe.live.runner import run_live_pipeline
from regimes_probe.policy.contextual_bandit import BanditParams
from regimes_probe.policy.memory import PolicyMemory
from regimes_probe.policy.policy_fragment import assert_no_answer_leakage
from regimes_probe.tools.base import SearchProvider
from regimes_probe.tools.fake import build_fake_providers, load_corpus

ROOT = Path(__file__).resolve().parents[1]
RS = ROOT / "fixtures" / "real_shaped"


class Boom(SearchProvider):
    name = "tavily_search"
    cost_per_call = Decimal("0.004")
    def available(self): return True
    def search(self, q, *, limit=5, **o):
        raise urllib.error.HTTPError("https://api.tavily.com/search", 400, "Bad Request", {},
                                     io.BytesIO(b'{"detail":"bad schema"}'))


def _cfg():
    return yaml.safe_load((ROOT / "config" / "default.yaml").read_text())


def _items():
    return LiveBrowseCompAdapter(local_jsonl=RS / "livebrowsecomp_sample.jsonl").load()


def _run(tmp_path, *, tools, providers, conditions, budgets, run_id):
    items = _items()
    ag = EpistemicAgent(AgentConfig(available_tools=tools + ["page_fetch"]))
    cb = build_closed_book_agent(AgentConfig(available_tools=tools + ["page_fetch"]),
                                 build_closed_book_knowledge(items))
    return run_live_pipeline(
        _cfg(), items, providers=providers, search_agent=ag, cb_agent=cb,
        cache=RecordingCache(mode="off"), conditions=conditions, budgets=budgets,
        optimize=4, confirm=4, split_seed="t", run_id=run_id, results_root=str(tmp_path),
        dataset_label="real_shaped_placeholder", dataset_version="v", dataset_path=None,
        is_real=False, search_tools=tools, weights=RewardWeights.full(), params=BanditParams(),
        live_settings={"tools": tools + ["page_fetch"]})


# --------------------------------------------------- leakage: false positive fixed
def test_short_gold_substring_no_longer_false_positives():
    # gold token appears as a SUBSTRING of a hash leaf, but not as a whole word.
    snap = {"traces": [{"norm_hash": "deadvaelbeef", "embedding": [0.1, 0.2]}]}
    # the OLD brittle check would have flagged this:
    assert "vael" in json.dumps(snap)
    clean, leaked = scan_gold_in_payload(snap, {"vael"})
    assert clean is True and leaked is None            # word-boundary scan: not a leak


def test_real_gold_text_in_snapshot_is_detected_with_specific_path():
    snap = {"fragments": {"k": {"note": "the steward is edrin vael per source"}}}
    res = snapshot_leakage(snap, [Item(id="x", question="q", answer="Edrin Vael")])
    assert res["pass"] is False
    assert res["failed_path"] and res["message"]       # specific, not opaque


def test_report_leakage_matches_inspector_for_clean_snapshot():
    # build a real snapshot via experience; both checks must agree (PASS).
    from regimes_probe.datasets.synthetic import SyntheticBrowseAdapter
    from regimes_probe.eval.harness import experience_phase
    items = SyntheticBrowseAdapter().load()
    providers = build_fake_providers(load_corpus(ROOT / "fixtures" / "fake_search_corpus.json"),
                                     ["generic_web_search", "news_search", "official_domain_search",
                                      "brave_search"])
    ag = EpistemicAgent(AgentConfig(available_tools=["generic_web_search", "news_search",
                                                     "official_domain_search", "brave_search",
                                                     "page_fetch"]))
    mem = PolicyMemory(BanditParams())
    experience_phase(items[:20], ag, providers, mem, budget=4, passes=2)
    snap = mem.snapshot().to_dict()
    # inspector-style check
    assert_no_answer_leakage(snap, "policy_memory_snapshot")        # does not raise -> PASS
    # report-time check uses the SAME function/payload
    assert snapshot_leakage(snap, items)["pass"] is True


def test_raw_archive_gold_does_not_fail_policy_memory_leakage():
    items = _items()                                    # items DO have gold answers
    clean_snap = {"traces": [{"norm_hash": "abc123", "embedding": [0.0]}], "fragments": {}}
    d = leakage_check_details(clean_snap, items)
    assert d["memory_snapshot_leakage_pass"] is True    # snapshot is clean
    assert d["raw_trace_archive_leakage_pass"] is True  # raw archive gold is by design
    assert d["raw_trace_archive_may_contain_answers"] is True
    assert d["gates_memory_claim"] == "memory_snapshot_leakage_pass"


# --------------------------------------------------- report.json metrics populated
def test_report_metrics_populated(tmp_path):
    providers = build_fake_providers(load_corpus(RS / "real_shaped_corpus.json"),
                                     ["generic_web_search", "news_search"])
    out = _run(tmp_path, tools=["generic_web_search", "news_search"], providers=providers,
               conditions=["closed_book", "no_memory_search"], budgets=[1, 3], run_id="m")
    report = json.loads((Path(out["run_dir"]) / "report.json").read_text())
    assert report["metrics"], "report.json.metrics must not be empty"
    # keyed by condition@budget, with the expected metric fields
    cell = report["metrics"]["no_memory_search@3"]
    for k in ("accuracy", "correct_per_tool_call", "provider_failure_rate", "failed_tool_calls"):
        assert k in cell


# --------------------------------------------------- provider failures in report + summary
def test_provider_failures_in_report_and_summary(tmp_path):
    providers = build_fake_providers(load_corpus(RS / "real_shaped_corpus.json"), [])
    providers["tavily_search"] = Boom()
    out = _run(tmp_path, tools=["tavily_search"], providers=providers,
               conditions=["no_memory_search"], budgets=[2], run_id="pf")
    rd = Path(out["run_dir"])
    report = json.loads((rd / "report.json").read_text())
    assert report["tool_failures"].get("tavily_search", 0) >= 1
    assert report["provider_failure_rate"] > 0
    summary = (rd / "summary.md").read_text()
    assert "Provider failures" in summary and "tavily_search" in summary
    # tool_rewards.csv has the failed_calls column
    assert "failed_calls" in (rd / "tool_rewards.csv").read_text().splitlines()[0]


# --------------------------------------------------- debug artifact: bounded + complete
def test_debug_jsonl_bounded_and_has_previews(tmp_path):
    providers = build_fake_providers(load_corpus(RS / "real_shaped_corpus.json"),
                                     ["generic_web_search", "news_search"])
    out = _run(tmp_path, tools=["generic_web_search", "news_search"], providers=providers,
               conditions=["no_memory_search"], budgets=[3], run_id="dbg")
    rows = [json.loads(l) for l in
            (Path(out["run_dir"]) / "debug_questions.jsonl").read_text().splitlines() if l.strip()]
    assert rows
    r = rows[0]
    for k in ("question_preview", "gold_preview", "prediction_preview", "tool_sequence",
              "provider_names", "evidence", "failure_seam", "regime", "calls"):
        assert k in r
    # previews are BOUNDED (not full unbounded content)
    assert len(r["question_preview"]) <= 201
    for ev in r["evidence"]:
        assert len(ev["snippet_preview"]) <= 241
        assert len(ev["title_preview"]) <= 201


def test_debug_record_captures_failed_provider_and_seam(tmp_path):
    providers = build_fake_providers(load_corpus(RS / "real_shaped_corpus.json"), [])
    providers["tavily_search"] = Boom()
    out = _run(tmp_path, tools=["tavily_search"], providers=providers,
               conditions=["no_memory_search"], budgets=[2], run_id="dbgfail")
    rows = [json.loads(l) for l in
            (Path(out["run_dir"]) / "debug_questions.jsonl").read_text().splitlines() if l.strip()]
    # at least one record records the failed tool call with sanitized error
    failed = [r for r in rows if r["failed_tool_calls"] > 0]
    assert failed
    err = failed[0]["failed_tool_errors"][0]
    assert err["tool"] == "tavily_search" and err["status_code"] == 400
    assert failed[0]["failure_seam"] in ("provider_error", "provider_returned_no_results")
    # no-evidence/abstained case yields an evidence-absent-style seam somewhere
    assert any(r["failure_seam"] in ("evidence_absent", "provider_error", "exact_answer_missing")
               for r in rows)


def test_debug_scripts_run(tmp_path):
    import subprocess, sys
    providers = build_fake_providers(load_corpus(RS / "real_shaped_corpus.json"), [])
    providers["tavily_search"] = Boom()
    out = _run(tmp_path, tools=["tavily_search"], providers=providers,
               conditions=["no_memory_search"], budgets=[2], run_id="scripts")
    rd = out["run_dir"]
    a = subprocess.run([sys.executable, str(ROOT / "scripts" / "debug_run_failures.py"), rd,
                        "--limit", "3"], capture_output=True, text=True, timeout=60)
    assert a.returncode == 0 and "failure seams" in a.stdout
    b = subprocess.run([sys.executable, str(ROOT / "scripts" / "summarize_provider_returns.py"), rd],
                       capture_output=True, text=True, timeout=60)
    assert b.returncode == 0 and "provider returns" in b.stdout and "tavily_search" in b.stdout
