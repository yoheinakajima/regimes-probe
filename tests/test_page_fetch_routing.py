"""page_fetch is a follow-up (URL) tool, never a first-hop search arm (no network).

Regression for the BrowseComp debug run where page_fetch was routed first-hop
with a search query and failed 78/78 times (provider_failure_rate=0.2955).
"""

from __future__ import annotations

from regimes_probe.agent.planner import AgentConfig, EpistemicAgent
from regimes_probe.eval.harness import run_condition
from regimes_probe.eval.metrics import compute_metrics
from regimes_probe.eval.reward import RewardWeights
from regimes_probe.live.settings import resolve_live_settings
from regimes_probe.policy.contextual_bandit import BanditParams
from regimes_probe.policy.memory import PolicyMemory
from regimes_probe.tools.fake import FakePageFetch, build_fake_providers
from regimes_probe.tools.metadata import is_first_hop, is_followup
from regimes_probe.tools.page_fetch import PageFetch

TOOLS = ["generic_web_search", "news_search", "official_domain_search", "brave_search"]


# ---------------------------------------------------- metadata / cheap-mode arms
def test_page_fetch_is_followup_family_not_first_hop():
    assert is_followup("page_fetch") and not is_first_hop("page_fetch")
    assert is_first_hop("tavily_search") and not is_followup("tavily_search")


def test_cheap_mode_does_not_put_page_fetch_in_first_hop_arms():
    s = resolve_live_settings(mode="cheap", env={"TAVILY_API_KEY": "x", "EXA_API_KEY": "y"})
    # page_fetch is present (as a follow-up tool) but NEVER a first-hop arm.
    assert "page_fetch" in s.tools
    assert "page_fetch" not in s.first_hop_tools
    assert "page_fetch" in s.followup_tools
    assert "page_fetch" not in s.search_tools
    assert set(s.first_hop_tools) == {"tavily_search", "exa_search"}


# ---------------------------------------------------- page_fetch requires a URL
def test_page_fetch_requires_a_url_live_adapter_no_network():
    resp = PageFetch().search("who is the CEO of Acme Robotics?")  # a query, not a URL
    assert resp.failed
    assert resp.error_meta["error_type"] == "requires_url"
    assert resp.results == ()


def test_fake_page_fetch_requires_a_url():
    pf = FakePageFetch([{"url": "https://example.com/a", "snippet": "x"}])
    bad = pf.search("not a url")
    assert bad.failed and bad.error_meta["error_type"] == "requires_url"
    ok = pf.search("https://example.com/a")
    assert not ok.failed and ok.results and ok.results[0].url == "https://example.com/a"


# ---------------------------------------------------- follow-up on a URL works
def test_page_fetch_used_as_followup_on_url_from_search(items, corpus):
    """On a multihop item the answer is hidden in the snippet and revealed only
    by fetching the URL search returned — the loop must page_fetch that URL."""
    providers = build_fake_providers(corpus, TOOLS)
    item = next(i for i in items if i.meta.get("regime") == "multihop")
    agent = EpistemicAgent(AgentConfig(available_tools=TOOLS + ["page_fetch"]))
    mem = PolicyMemory(BanditParams())
    sig = agent.signature(item)
    # Seed the stop bandit toward fetch_page so the follow-up fires deterministically.
    for _ in range(6):
        mem.bandits["stop"].update_reward(sig.cluster_key, "fetch_page", 1.5)
    trace = agent.attempt(item, mem, providers, budget=3, explore=False, attempt_id="t")
    fetches = [c for c in trace.calls if c.tool == "page_fetch"]
    assert fetches, "expected a page_fetch follow-up call"
    fc = fetches[0]
    assert fc.query.startswith("http"), "page_fetch must be called with a URL"
    assert not fc.failed, "follow-up page_fetch on a real URL must not fail"


# ------------------------------------ provider failure rate falls (fake run)
def _run_metrics(available_tools, providers, items, *, budget=3):
    agent = EpistemicAgent(AgentConfig(available_tools=available_tools))
    r = run_condition(items, agent, providers, PolicyMemory(BanditParams()),
                      condition="no_memory_search", budget=budget,
                      weights=RewardWeights.full())
    return compute_metrics(r.outcomes)


# ------------------------------------ reporting/manifest excludes page_fetch from arms
def test_resolve_explicit_tools_splits_first_hop_and_followup():
    s = resolve_live_settings(
        cli_tools=["serper_search", "exa_search", "firecrawl_search"],
        env={"SERPER_API_KEY": "a", "EXA_API_KEY": "b", "FIRECRAWL_API_KEY": "c"})
    assert s.first_hop_tools == ["serper_search", "exa_search", "firecrawl_search"]
    assert s.followup_tools == ["page_fetch"]          # appended as follow-up only
    assert "page_fetch" in s.tools and "page_fetch" not in s.first_hop_tools
    d = s.to_dict()
    assert d["first_hop_tools"] == ["serper_search", "exa_search", "firecrawl_search"]
    assert d["followup_tools"] == ["page_fetch"]


def test_manifest_separates_first_hop_from_followup():
    from types import SimpleNamespace
    from regimes_probe.eval.manifest import build_manifest
    s = resolve_live_settings(
        cli_tools=["serper_search", "exa_search", "firecrawl_search"],
        env={"SERPER_API_KEY": "a", "EXA_API_KEY": "b", "FIRECRAWL_API_KEY": "c"})
    split = SimpleNamespace(salt="t", mode="hash", optimize_ids=["a"], confirm_ids=["b", "c"])
    m = build_manifest(
        run_id="r", cfg={"budgets": [1, 3]}, dataset_label="d", dataset_version="v",
        dataset_checksum="x", dataset_path=None, split=split,
        search_tools=s.first_hop_tools, tools_cfg=None, memory_cfg={},
        eligibility_preflight={}, cost_estimate={}, live=True, live_settings=s.to_dict())
    assert "page_fetch" not in m["first_hop_tools"]
    assert m["first_hop_tools"] == ["serper_search", "exa_search", "firecrawl_search"]
    assert m["followup_tools"] == ["page_fetch"]
    assert "page_fetch" in m["all_enabled_tools"]
    assert "page_fetch" not in m["tools_enabled"]      # back-compat == first-hop arms


def test_cost_estimate_does_not_give_page_fetch_full_first_hop_budget():
    from regimes_probe.live.runner import estimate_live
    s = resolve_live_settings(
        cli_tools=["serper_search", "exa_search", "firecrawl_search"],
        env={"SERPER_API_KEY": "a", "EXA_API_KEY": "b", "FIRECRAWL_API_KEY": "c"})
    est = estimate_live(["no_memory_search", "policy_memory"], [1, 3], n_opt=5, n_con=10,
                        passes=4, exp_budget=5, judge="exact", settings=s.to_dict())
    assert "page_fetch" not in est["first_hop_tools"]
    assert est["followup_tools"] == ["page_fetch"]
    # page_fetch must NOT be shown as receiving the full first-hop search budget.
    assert est["max_calls_by_tool"]["serper_search"] == est["worst_case_first_hop_calls"]
    assert est["max_calls_by_tool"]["page_fetch"] == est["worst_case_followup_calls"]
    assert est["worst_case_followup_calls"] < est["worst_case_first_hop_calls"]


def test_provider_failure_rate_falls_when_page_fetch_excluded_from_first_hop(items, corpus):
    sample = items[:12]
    # FORCED first-hop page_fetch (degraded: only page_fetch available) -> every
    # call is page_fetch with a search query -> requires_url failures.
    forced = _run_metrics(["page_fetch"], build_fake_providers(corpus, []), sample)
    # NORMAL: search tools first-hop, page_fetch reserved as a follow-up.
    normal = _run_metrics(TOOLS + ["page_fetch"], build_fake_providers(corpus, TOOLS), sample)
    assert forced["provider_failure_rate"] > 0.5      # page_fetch fails first-hop
    assert normal["provider_failure_rate"] == 0.0     # no first-hop fetch misuse
    assert normal["provider_failure_rate"] < forced["provider_failure_rate"]
