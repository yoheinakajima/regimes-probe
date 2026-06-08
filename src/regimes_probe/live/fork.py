"""Offline forked ablations: rerun policy variants on cached outcomes, no spend.

After a paid provider run, the recording cache holds every provider/model/tool
outcome keyed by its sanitized request. An *offline fork* reuses that cache in
``replay`` mode to rerun the CONFIRM conditions under different policy/reward/
router parameters — **without ever calling a provider**. Any request not in the
cache raises :class:`~regimes_probe.live.cache.ReplayMiss` (so changing the query
distribution is surfaced, not silently re-spent).

A forked run is recorded with ``offline_fork=true`` and ``parent_run_id`` in its
manifest and can never be headline-eligible. See ``docs/OFFLINE_FORK_ABLATIONS.md``.

Extension points (TODO): a v1 could (a) record the parent answerer kind so a live
fork auto-selects the live replay answerer; (b) support query-distribution-
preserving router changes by re-keying the cache on the realized query rather
than the policy; (c) diff two forks' reports automatically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from regimes_probe.live.cache import RecordingCache
from regimes_probe.live.providers import CachedProvider, NotArmed
from regimes_probe.tools.base import SearchProvider, SearchResponse

_REAL_DATASETS = {"BrowseComp", "LiveBrowseComp", "browsecomp", "livebrowsecomp"}


class RefusingProvider(SearchProvider):
    """Inner provider that refuses to do any I/O — used behind a replay cache."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.deterministic = False

    def available(self) -> bool:  # present so it can be wrapped
        return True

    def search(self, query: str, *, limit: int = 5, **opts: Any) -> SearchResponse:
        raise NotArmed(f"{self.name}: offline fork refuses to call providers "
                       "(replay cache miss — would re-spend)")


def build_replay_providers(tool_names: list[str],
                           cache: RecordingCache) -> dict[str, SearchProvider]:
    """Replay-only providers: cache hit serves; miss raises (never any network)."""
    return {name: CachedProvider(RefusingProvider(name), cache, armed=False)
            for name in tool_names}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_items(dataset_name: str, dataset_path: Optional[str]):
    from regimes_probe.datasets.livebrowsecomp import LiveBrowseCompAdapter
    if dataset_path and dataset_path.endswith(".jsonl"):
        return LiveBrowseCompAdapter(local_jsonl=dataset_path).load()
    if dataset_path and dataset_path.endswith(".csv"):
        from regimes_probe.datasets.browsecomp import BrowseCompAdapter
        return BrowseCompAdapter(dataset_path).load()
    from regimes_probe.datasets.synthetic import SyntheticBrowseAdapter
    return SyntheticBrowseAdapter().load()


def fork_offline(
    parent_run_dir: str | Path,
    *,
    out_run_id: str,
    results_root: str | Path,
    cache_path: Optional[str] = None,
    policy_overrides: Optional[dict[str, Any]] = None,
    conditions: Optional[list[str]] = None,
    budgets: Optional[list[int]] = None,
    answerer: str = "deterministic",
) -> dict[str, Any]:
    """Run an offline forked ablation from a completed parent run. No spend.

    Raises ``FileNotFoundError`` if the parent run or its recording cache is
    missing (graceful refusal: the caller reports it; nothing is spent).
    """
    import yaml

    from regimes_probe.agent.answerer import build_closed_book_knowledge
    from regimes_probe.agent.planner import (
        AgentConfig, EpistemicAgent, build_closed_book_agent)
    from regimes_probe.live.runner import run_live_pipeline
    from regimes_probe.policy.contextual_bandit import BanditParams
    from regimes_probe.eval.reward import RewardWeights

    parent_run_dir = Path(parent_run_dir)
    manifest_path = parent_run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no run_manifest.json in {parent_run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parent_run_id = manifest.get("run_id")

    cfg_path = parent_run_dir / "config_snapshot.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    cfg = _merge(cfg, policy_overrides or {})

    # Resolve the recording cache (the source of cached outcomes).
    cp = cache_path or (manifest.get("cache") or {}).get("path")
    if not cp or not Path(cp).exists():
        raise FileNotFoundError(
            "offline fork requires the parent run's recording cache "
            f"(looked for {cp!r}). Re-run the parent with --recording-cache, or pass "
            "--cache. No providers were called.")
    cache = RecordingCache(cp, mode="replay")

    ds = manifest.get("dataset", {})
    dataset_name = ds.get("name", "synthetic")
    dataset_path = ds.get("path")
    is_real = dataset_name in _REAL_DATASETS
    items = _load_items(dataset_name, dataset_path)

    tools = list(manifest.get("tools_enabled", [])) + (
        ["page_fetch"] if "page_fetch" not in manifest.get("tools_enabled", []) else [])
    search_tools = list(manifest.get("tools_enabled", []))
    providers = build_replay_providers(tools, cache)

    agent_cfg = AgentConfig(available_tools=tools,
                            query_mode=cfg.get("policy", {}).get("query_mode", "learned"),
                            stop_mode=cfg.get("policy", {}).get("stop_mode", "learned"),
                            as_of=cfg.get("run", {}).get("as_of", "2026-06-01"))
    if answerer == "live":
        # Live-origin fork: replay the parent's cached model answers (no spend).
        from regimes_probe.live.providers import build_live_answerer
        model = (manifest.get("models", {}) or {}).get("answer_model") or "gpt-5.4-mini"
        search_agent = EpistemicAgent(
            agent_cfg, answerer=build_live_answerer("search", model=model, cache=cache, armed=False))
        cb_agent = EpistemicAgent(
            agent_cfg, answerer=build_live_answerer("closed_book", model=model, cache=cache, armed=False))
    else:
        search_agent = EpistemicAgent(agent_cfg)
        cb_agent = build_closed_book_agent(agent_cfg, build_closed_book_knowledge(items))

    snap_path = parent_run_dir / "memory_snapshot.json"
    resume = json.loads(snap_path.read_text(encoding="utf-8")) if snap_path.exists() else None

    split = manifest.get("split", {})
    optimize = int(split.get("n_optimize", 0))
    confirm = int(split.get("n_confirm", len(items)))
    split_seed = split.get("seed", "regimes-probe-v0")
    # Default to the memory-variant condition: its CONFIRM behavior is fully
    # determined by the frozen snapshot, so it replays from cache exactly. The
    # no_memory_search baseline is fixed across policy variants — read it from the
    # parent report rather than re-deriving it (its queries depend on agent
    # warm-state that a fresh fork cannot reproduce without re-spending).
    conditions = conditions or ["policy_memory"]
    budgets = budgets or (manifest.get("budgets") or [1])

    return run_live_pipeline(
        cfg, items, providers=providers, search_agent=search_agent, cb_agent=cb_agent,
        cache=cache, conditions=conditions, budgets=budgets, optimize=optimize,
        confirm=confirm, split_seed=split_seed, run_id=out_run_id,
        results_root=str(results_root), dataset_label=dataset_name,
        dataset_version=ds.get("version", "fork"), dataset_path=dataset_path,
        is_real=is_real, search_tools=search_tools,
        weights=RewardWeights.full(), params=BanditParams.from_dict(cfg.get("bandit", {})),
        resume_snapshot=resume, parent_run_id=parent_run_id, offline_fork=True)
