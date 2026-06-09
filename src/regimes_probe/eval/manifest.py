"""Dry-run manifest: everything needed to audit/reproduce a run, no secrets.

The manifest is written *before* any live call so a run is fully described up
front. It records git provenance, dataset identity, the exact split, model/prompt
versions, tool/provider names (never secret values), budgets, memory policy, the
preflight eligibility verdict, and the call/cost estimate.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Optional

from regimes_probe.agent import prompts


def _git(args: list[str]) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def git_provenance() -> dict[str, Any]:
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    commit = _git(["rev-parse", "HEAD"])
    status = _git(["status", "--porcelain"])
    return {
        "branch": branch,
        "commit": commit,
        "dirty": (bool(status) if status is not None else None),
    }


def build_manifest(
    *,
    run_id: str,
    cfg: dict[str, Any],
    dataset_label: str,
    dataset_version: str,
    dataset_checksum: Optional[str],
    dataset_path: Optional[str],
    split,
    search_tools: list[str],
    tools_cfg: Optional[dict[str, Any]],
    memory_cfg: dict[str, Any],
    eligibility_preflight: dict[str, Any],
    cost_estimate: dict[str, Any],
    live: bool = False,
    live_settings: Optional[dict[str, Any]] = None,
    parent_run_id: Optional[str] = None,
    offline_fork: bool = False,
) -> dict[str, Any]:
    """Assemble the manifest dict (pure given inputs; git is read locally)."""
    from regimes_probe.tools.metadata import first_hop_tools as _first_hop
    from regimes_probe.tools.metadata import followup_tools as _followup

    live_cfg = cfg.get("live", {})
    ls = live_settings or {}
    # All enabled tools and their first-hop/follow-up split. Prefer the resolved
    # live settings; otherwise derive from search_tools (page_fetch is appended at
    # runtime as a follow-up tool, so derive both subsets from metadata families).
    all_enabled_tools = ls.get("tools") or list(search_tools)
    first_hop_tools_list = ls.get("first_hop_tools") or _first_hop(all_enabled_tools)
    followup_tools_list = ls.get("followup_tools") or _followup(all_enabled_tools)
    adapters = (tools_cfg or {}).get("adapters", {}) if tools_cfg else {}
    provider_names = sorted(adapters.keys()) or list(search_tools)
    provider_config = {
        name: {"enabled": bool(a.get("enabled", False)),
               "api_key_env": a.get("api_key_env", "")}   # NAME only, never value
        for name, a in adapters.items()
    }

    return {
        "run_id": run_id,
        "live": live,
        # Offline-fork lineage: a forked ablation reuses a parent's cached
        # provider/model/tool outcomes and changes only policy params (no spend).
        "offline_fork": bool(offline_fork),
        "parent_run_id": parent_run_id,
        "git": git_provenance(),
        "dataset": {
            "name": dataset_label,
            "version": dataset_version,
            "checksum": dataset_checksum,
            "path": dataset_path,
        },
        "split": {
            "seed": split.salt,
            "mode": split.mode,
            "n_optimize": len(split.optimize_ids),
            "n_confirm": len(split.confirm_ids),
            "optimize_ids": list(split.optimize_ids),
            "confirm_ids": list(split.confirm_ids),
        },
        "models": {
            "answer_model": ls.get("answer_model", live_cfg.get("answer_model")),
            "web_search_model": ls.get("web_search_model"),
            "web_search_context_size": ls.get("web_search_context_size"),
            "embedder": live_cfg.get("embedder"),
            "search_baseline": live_cfg.get("search_baseline"),
        },
        "provider_mode": ls.get("provider_mode"),
        "provider_classes": ls.get("provider_classes", []),
        "tools_meta": ls.get("tools_meta", {}),       # family/cost/stateful/safe per arm
        "openai_web_search_enabled": ls.get("openai_web_search_enabled"),
        "agentic_tool_discovery_enabled": ls.get("agentic_tool_discovery_enabled"),
        "scrape_tools_enabled": ls.get("scrape_tools_enabled"),
        "browserish_tools_enabled": ls.get("browserish_tools_enabled"),
        "stateful_or_paid_tools_allowed": ls.get("stateful_or_paid_tools_allowed"),
        "query_decomposition_enabled": ls.get(
            "query_decomposition_enabled",
            bool(cfg.get("policy", {}).get("enable_query_decomposition", False))),
        "iterative_clue_resolution_enabled": ls.get(
            "iterative_clue_resolution_enabled",
            bool(cfg.get("policy", {}).get("enable_iterative_clue_resolution", False))),
        "task_frame_enabled": ls.get(
            "task_frame_enabled",
            bool(cfg.get("policy", {}).get("enable_task_frame", False))),
        "llm_task_frame_parser_enabled": ls.get(
            "llm_task_frame_parser_enabled",
            bool(cfg.get("policy", {}).get("enable_llm_task_frame_parser", False))),
        "frontier_controller_enabled": ls.get(
            "frontier_controller_enabled",
            bool(cfg.get("policy", {}).get("enable_frontier_controller", False))),
        "task_frame_parser_model": ls.get(
            "task_frame_parser_model",
            cfg.get("policy", {}).get("task_frame_parser_model")),
        "live_settings": ls,
        "prompts": prompts.registry_dict(),
        # Tool inventory, split by routing role:
        #   all_enabled_tools — every tool the agent may use
        #   first_hop_tools   — search-family bandit arms (routed/learned)
        #   followup_tools    — URL-only tools (page_fetch/scrape); NOT bandit arms
        "all_enabled_tools": list(all_enabled_tools),
        "first_hop_tools": list(first_hop_tools_list),
        "followup_tools": list(followup_tools_list),
        # back-compat: == first-hop search arms (kept for older readers).
        "tools_enabled": list(search_tools),
        "provider_names": provider_names,
        "provider_config": provider_config,
        "budgets": cfg.get("budgets", []),
        "memory": {
            "condition": "frozen_policy_memory",
            "confirm_uses_frozen_snapshot": memory_cfg.get("confirm_uses_frozen_snapshot"),
            "confirm_updates_memory": memory_cfg.get("confirm_updates_memory"),
            "freezes_during_confirm": bool(
                memory_cfg.get("confirm_uses_frozen_snapshot")
                and not memory_cfg.get("confirm_updates_memory")
            ),
        },
        "grader": "normalized_match" + (
            "+llm_judge" if cfg.get("grading", {}).get("judge") == "llm" else ""
        ),
        "headline_eligibility_preflight": eligibility_preflight,
        "cost_estimate": cost_estimate,
    }


def write_manifest(run_dir: str | Path, manifest: dict[str, Any]) -> Path:
    import json
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
