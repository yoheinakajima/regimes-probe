"""Headline eligibility distinguishes plumbing runs from memory-claim runs.

No providers are called (mock providers / crafted reports).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from regimes_probe.agent.answerer import build_closed_book_knowledge
from regimes_probe.agent.planner import AgentConfig, EpistemicAgent, build_closed_book_agent
from regimes_probe.datasets.livebrowsecomp import LiveBrowseCompAdapter
from regimes_probe.eval.eligibility import compute_eligibility, REQUIRED_CHECKS
from regimes_probe.eval.reward import RewardWeights
from regimes_probe.live.cache import RecordingCache
from regimes_probe.live.runner import run_live_pipeline
from regimes_probe.policy.contextual_bandit import BanditParams
from regimes_probe.tools.fake import build_fake_providers, load_corpus

ROOT = Path(__file__).resolve().parents[1]
RS = ROOT / "fixtures" / "real_shaped"
TOOLS = ["generic_web_search", "news_search", "official_domain_search", "brave_search"]
_FOUR = ["closed_book", "no_memory_search", "random_memory", "policy_memory"]


def _setup():
    cfg = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
    items = LiveBrowseCompAdapter(local_jsonl=RS / "livebrowsecomp_sample.jsonl").load()
    providers = build_fake_providers(load_corpus(RS / "real_shaped_corpus.json"), TOOLS)
    agent = EpistemicAgent(AgentConfig(available_tools=TOOLS + ["page_fetch"]))
    cb = build_closed_book_agent(AgentConfig(available_tools=TOOLS + ["page_fetch"]),
                                 build_closed_book_knowledge(items))
    return cfg, items, providers, agent, cb


def _run(conditions, *, is_real, run_id, tmp_path):
    cfg, items, providers, agent, cb = _setup()
    return run_live_pipeline(
        cfg, items, providers=providers, search_agent=agent, cb_agent=cb,
        cache=RecordingCache(mode="off"), conditions=conditions, budgets=[1],
        optimize=4, confirm=4, split_seed="t", run_id=run_id, results_root=str(tmp_path),
        dataset_label=("LiveBrowseComp" if is_real else "real_shaped_placeholder"),
        dataset_version="v", dataset_path=None, is_real=is_real, search_tools=TOOLS,
        weights=RewardWeights.full(), params=BanditParams())


# --------------------------------------------------- plumbing run (no policy_memory)
def test_plumbing_run_structurally_valid_not_headline(tmp_path):
    out = _run(["closed_book", "no_memory_search"], is_real=True, run_id="plumb",
               tmp_path=tmp_path)
    report = json.loads((Path(out["run_dir"]) / "report.json").read_text())
    assert report["structurally_valid"] is True
    assert report["headline_eligible_memory_claim"] is False
    assert report["headline_eligible"] is False                  # back-compat alias agrees
    assert "policy_memory" not in report["conditions_present"]
    assert any("policy_memory" in r for r in report["headline_eligibility_reasons"])
    # the misleading same-conditions phrase must NOT appear
    summary = (Path(out["run_dir"]) / "summary.md").read_text()
    assert "differ only in memory access" not in summary
    assert "N/A — policy_memory was not run" in summary
    # same_conditions in report.json is marked not applicable
    assert report["same_conditions"].get("not_applicable") is True


def test_claims_generator_refuses_when_policy_memory_missing(tmp_path):
    out = _run(["closed_book", "no_memory_search"], is_real=True, run_id="plumb2",
               tmp_path=tmp_path)
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_claims.py"),
         str(Path(out["run_dir"]) / "report.json")],
        capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    o = res.stdout
    assert "PLUMBING run completed" in o
    assert "REFUSED" in o and "policy_memory` was NOT run" in o
    assert "differ only in memory access" not in o               # no misleading claim


# --------------------------------------------------- synthetic four-condition run
def test_synthetic_all_four_not_headline_because_not_real(tmp_path):
    out = _run(_FOUR, is_real=False, run_id="syn4", tmp_path=tmp_path)
    report = json.loads((Path(out["run_dir"]) / "report.json").read_text())
    assert report["structurally_valid"] is True
    assert report["headline_eligible_memory_claim"] is False
    assert any("synthetic" in r or "placeholder" in r
               for r in report["headline_eligibility_reasons"])


# --------------------------------------------------- crafted real eligible report
def _write_report(tmp_path, *, conditions_present, dataset_is_real, confirm_size):
    checks = {c: True for c in REQUIRED_CHECKS}
    elig = compute_eligibility(checks, dataset_is_real=dataset_is_real,
                               conditions_present=conditions_present, confirm_size=confirm_size,
                               min_confirm=20)
    cells = [{"condition": c, "budget": 3,
              "metrics": {"correct_per_tool_call": 0.5 if c == "policy_memory" else 0.2,
                          "accuracy": 0.3}} for c in conditions_present]
    report = {
        "run_id": "crafted", "meta": {"dataset": "BrowseComp", "limitations": [], "not_claimed": []},
        "structurally_valid": elig.structurally_valid,
        "headline_eligible_memory_claim": elig.headline_eligible_memory_claim,
        "headline_eligible": elig.headline_eligible,
        "headline_eligibility_reasons": elig.headline_eligibility_reasons,
        "conditions_present": conditions_present, "eligibility": elig.to_dict(),
        "same_conditions": {"ok": True}, "conditions": cells,
        "significance": {"budget": 3, "mcnemar": {"c_only_treatment_correct": 5,
                                                  "b_only_baseline_correct": 0, "p_value": 0.02}},
    }
    p = tmp_path / "report.json"
    p.write_text(json.dumps(report))
    return p


def test_claims_permits_candidates_when_headline_eligible(tmp_path):
    p = _write_report(tmp_path, conditions_present=_FOUR, dataset_is_real=True, confirm_size=24)
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_claims.py"), str(p)],
                         capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    assert "CANDIDATE" in res.stdout and "correct_per_tool_call" in res.stdout
    assert "policy_memory` was NOT run" not in res.stdout
