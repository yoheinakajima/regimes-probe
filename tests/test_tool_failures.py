"""Tool-call failures become recorded failed observations, not crashes.

No real providers/network: the failing provider is a fake that raises HTTPError,
and Tavily is exercised with a mocked transport.
"""

from __future__ import annotations

import io
import json
import urllib.error
from decimal import Decimal

import pytest

from regimes_probe.activegraph_pack import EventLog, RecordingInvoker, ReplayInvoker
from regimes_probe.agent.planner import AgentConfig, EpistemicAgent
from regimes_probe.datasets.synthetic import SyntheticBrowseAdapter
from regimes_probe.eval.eligibility import compute_eligibility, REQUIRED_CHECKS
from regimes_probe.eval.grader import grade, normalize_answer
from regimes_probe.eval.harness import run_condition
from regimes_probe.eval.metrics import compute_metrics
from regimes_probe.eval.reward import RewardWeights, compute_rewards
from regimes_probe.live.cache import RecordingCache
from regimes_probe.live.providers import CachedProvider
from regimes_probe.policy.contextual_bandit import BanditParams
from regimes_probe.policy.memory import PolicyMemory
from regimes_probe.tools.base import SearchProvider, safe_search
from regimes_probe.tools.fake import build_fake_providers, load_corpus

import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _http_error(body=b'{"detail":"bad request; api_key=tvly-SECRET999"}'):
    return urllib.error.HTTPError("https://api.tavily.com/search", 400, "Bad Request",
                                  {}, io.BytesIO(body))


class BoomProvider(SearchProvider):
    name = "tavily_search"
    cost_per_call = Decimal("0.004")

    def available(self):
        return True

    def search(self, query, *, limit=5, **opts):
        raise _http_error()


def _providers_with_boom():
    corpus = load_corpus(ROOT / "fixtures" / "fake_search_corpus.json")
    p = build_fake_providers(corpus, ["generic_web_search", "news_search", "official_domain_search"])
    p["tavily_search"] = BoomProvider()
    return p


# --------------------------------------------------- safe_search normalizes + redacts
def test_safe_search_converts_httperror_and_redacts(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-SECRET999")
    r = safe_search(BoomProvider(), "q")
    assert r.failed and r.error_meta["status_code"] == 400
    assert r.error_meta["error_type"] == "HTTPError"
    assert "tvly-SECRET999" not in json.dumps(r.error_meta)        # secret redacted
    assert "<redacted>" in r.error_meta["message"]


def test_config_errors_still_raise():
    from regimes_probe.tools.base import ProviderUnavailable

    class NeedsKey(SearchProvider):
        name = "x"
        def available(self): return False
        def search(self, q, *, limit=5, **o):
            raise ProviderUnavailable("needs KEY")
    with pytest.raises(ProviderUnavailable):
        safe_search(NeedsKey(), "q")                              # preflight error propagates


# --------------------------------------------------- end-to-end: no crash, recorded
def test_run_condition_does_not_crash_on_provider_error():
    items = SyntheticBrowseAdapter().load()[:6]
    providers = _providers_with_boom()
    agent = EpistemicAgent(AgentConfig(
        available_tools=["tavily_search", "generic_web_search", "news_search", "page_fetch"]))
    res = run_condition(items, agent, providers, PolicyMemory(BanditParams()),
                        condition="t", budget=3, explore=True, update_memory=True)
    m = compute_metrics(res.outcomes)
    assert m["failed_tool_calls"] >= 1
    assert m["failed_tool_calls_by_tool"].get("tavily_search", 0) >= 1
    assert 0.0 < m["provider_failure_rate"] <= 1.0
    assert any(o.failed_tool_calls > 0 for o in res.outcomes)      # recorded, not crashed


def test_failed_call_recorded_in_trace_and_penalized():
    items = SyntheticBrowseAdapter().load()
    providers = _providers_with_boom()
    agent = EpistemicAgent(AgentConfig(available_tools=["tavily_search", "page_fetch"]))
    # seed routing to the failing tool so it is chosen
    mem = PolicyMemory(BanditParams())
    sig = agent.signature(items[0])
    for _ in range(6):
        mem.bandits["tool"].update_reward(sig.cluster_key, "tavily_search", 1.0)
    trace = agent.attempt(items[0], mem, providers, budget=1, explore=False, attempt_id="a")
    assert trace.calls[0].failed is True and trace.calls[0].error_type == "HTTPError"
    rr = compute_rewards(trace, correct=False, gold_norms=["x"], weights=RewardWeights.full(),
                         freshness_sensitive=False)
    assert rr.flags["had_tool_failure"] is True
    # the failed tool arm got a clearly negative reward (failure penalty applied)
    assert rr.arm_rewards["tool"]["tavily_search"] < 0


# --------------------------------------------------- cache stores + replays failure
def test_cache_stores_and_replays_failed_call(tmp_path, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-SECRET999")
    cache = RecordingCache(tmp_path / "c.json", mode="auto")
    prov = CachedProvider(BoomProvider(), cache, armed=True)
    r1 = prov.search("q")
    assert r1.failed and cache.calls == 1
    r2 = prov.search("q")                                          # served from cache
    assert r2.failed and cache.calls == 1 and cache.hits == 1     # no re-spend on the failure
    blob = (tmp_path / "c.json").read_text()
    assert "tvly-SECRET999" not in blob                           # no secret in cache
    # replay mode replays the failure (no provider call)
    rcache = RecordingCache(tmp_path / "c.json", mode="replay")
    rprov = CachedProvider(BoomProvider(), rcache, armed=True)
    r3 = rprov.search("q")
    assert r3.failed and rcache.calls == 0


def test_recording_then_replay_invoker_reproduces_failure():
    providers = _providers_with_boom()
    log = EventLog(run_id="f")
    rinv = RecordingInvoker(providers, log, attempt_id="a")
    r1 = rinv.call("tavily_search", "q")
    assert r1.failed
    replay = ReplayInvoker(rinv.recorded)
    r2 = replay.call("tavily_search", "q")
    assert r2.failed and r2.error_meta["status_code"] == 400


# --------------------------------------------------- Tavily adapter (mocked 400)
def test_tavily_uses_bearer_header_and_no_apikey_in_body(monkeypatch):
    from regimes_probe.tools.tavily_search import TavilySearch
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-SECRET999")
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["req"] = req
        class _R:
            def read(self): return json.dumps({"results": [
                {"title": "t", "url": "u", "content": "c", "score": 0.5}]}).encode()
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return _R()
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    resp = TavilySearch().search("q", limit=2)
    body = json.loads(captured["req"].data)
    assert "api_key" not in body                                  # not in body (caused the 400)
    assert captured["req"].headers.get("Authorization") == "Bearer tvly-SECRET999"
    assert resp.results and resp.results[0].url == "u"


def test_tavily_400_becomes_failed_response(monkeypatch):
    from regimes_probe.tools.tavily_search import TavilySearch
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-SECRET999")

    def fake_urlopen(req, timeout=None):
        raise _http_error()
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    r = safe_search(TavilySearch(), "q")
    assert r.failed and r.error_meta["status_code"] == 400
    assert "tvly-SECRET999" not in json.dumps(r.error_meta)


# --------------------------------------------------- eligibility tolerates few failures
def test_few_failures_still_structurally_valid_high_rate_blocks_headline():
    checks = {c: True for c in REQUIRED_CHECKS}
    four = ["closed_book", "no_memory_search", "random_memory", "policy_memory"]
    ok = compute_eligibility(checks, dataset_is_real=True, conditions_present=four,
                             confirm_size=24, provider_failure_rate=0.05)
    assert ok.structurally_valid and ok.headline_eligible_memory_claim
    bad = compute_eligibility(checks, dataset_is_real=True, conditions_present=four,
                              confirm_size=24, provider_failure_rate=0.5,
                              max_provider_failure_rate=0.2)
    assert bad.structurally_valid is True                         # still valid
    assert bad.headline_eligible_memory_claim is False
    assert any("failure rate" in r for r in bad.headline_eligibility_reasons)


def test_entire_required_condition_failure_blocks_headline():
    checks = {c: True for c in REQUIRED_CHECKS}
    four = ["closed_book", "no_memory_search", "random_memory", "policy_memory"]
    e = compute_eligibility(checks, dataset_is_real=True, conditions_present=four,
                            confirm_size=24, provider_failure_rate=0.1,
                            failed_conditions=["no_memory_search"])
    assert e.headline_eligible_memory_claim is False
    assert any("failed entirely" in r for r in e.headline_eligibility_reasons)
