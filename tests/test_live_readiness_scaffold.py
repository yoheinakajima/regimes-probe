"""Prompt registry, cost estimator, and run-manifest (no keys, no network)."""

from __future__ import annotations

import json

from regimes_probe.agent import prompts
from regimes_probe.eval.cost import estimate_calls
from regimes_probe.eval.manifest import build_manifest, git_provenance
from regimes_probe.eval.split import build_split


# ---------------------------------------------------------------- prompts
def test_prompt_fingerprint_stable_and_hashed():
    fp = prompts.fingerprint("answerer")
    assert fp.startswith("answerer@v1#")
    assert fp == prompts.fingerprint("answerer")            # deterministic
    assert prompts.get("answerer").content_hash != prompts.get("judge").content_hash


def test_prompt_vary_flags():
    assert prompts.get("answerer").allowed_to_vary is False
    assert prompts.get("judge").allowed_to_vary is False
    assert prompts.get("query_llm").allowed_to_vary is True


def test_registry_dict_has_no_raw_content():
    reg = prompts.registry_dict()
    blob = json.dumps(reg)
    # the registry exposes hashes/fingerprints, not the raw prompt bytes
    assert "content_hash" in blob and "fingerprint" in blob
    assert prompts.get("answerer").content not in blob


# ---------------------------------------------------------------- cost
def test_cost_scales_with_items_and_budgets():
    small = estimate_calls(n_optimize=5, n_confirm=10, budgets=[1, 3])
    big = estimate_calls(n_optimize=10, n_confirm=20, budgets=[1, 3, 5, 10])
    assert big.answerer_calls > small.answerer_calls
    assert big.worst_case_tool_calls > small.worst_case_tool_calls


def test_cost_judge_calls_only_when_llm():
    assert estimate_calls(n_optimize=5, n_confirm=10, budgets=[1], judge="exact").judge_calls == 0
    assert estimate_calls(n_optimize=5, n_confirm=10, budgets=[1], judge="llm").judge_calls > 0


def test_cost_unknown_without_prices_and_computed_with():
    e0 = estimate_calls(n_optimize=5, n_confirm=10, budgets=[1])
    assert e0.estimated_cost_usd == "unknown"
    e1 = estimate_calls(n_optimize=5, n_confirm=10, budgets=[1],
                        prices={"answerer_per_call": 0.01, "tool_per_call": 0.001})
    assert isinstance(e1.estimated_cost_usd, float) and e1.estimated_cost_usd > 0


# ---------------------------------------------------------------- manifest
def test_manifest_has_required_keys_and_no_secrets(items):
    split = build_split(items[:20], confirm_fraction=0.5)
    tools_cfg = {"adapters": {"openai_web_search": {
        "enabled": True, "api_key_env": "OPENAI_API_KEY", "secret": "SHOULD_NOT_APPEAR"}}}
    m = build_manifest(
        run_id="t", cfg={"live": {"answer_model": "gpt-5.5"}, "budgets": [1, 3]},
        dataset_label="synthetic_browse", dataset_version="v", dataset_checksum="abc",
        dataset_path=None, split=split, search_tools=["news_search"], tools_cfg=tools_cfg,
        memory_cfg={"confirm_uses_frozen_snapshot": True, "confirm_updates_memory": False},
        eligibility_preflight={"headline_eligible": False}, cost_estimate={"answerer_calls": 1},
    )
    for key in ("git", "dataset", "split", "models", "prompts", "provider_config",
                "budgets", "memory", "headline_eligibility_preflight", "cost_estimate"):
        assert key in m
    assert m["memory"]["freezes_during_confirm"] is True
    assert m["split"]["optimize_ids"] and m["split"]["confirm_ids"]
    # secrets/values never leak — only env-var NAMES
    blob = json.dumps(m)
    assert "SHOULD_NOT_APPEAR" not in blob
    assert m["provider_config"]["openai_web_search"]["api_key_env"] == "OPENAI_API_KEY"


def test_git_provenance_shape():
    g = git_provenance()
    assert set(g) == {"branch", "commit", "dirty"}
