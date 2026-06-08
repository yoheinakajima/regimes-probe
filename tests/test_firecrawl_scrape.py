"""Level 3 evidence reading: firecrawl_scrape as a URL-only follow-up tool with a
page_fetch fallback (no providers, no network)."""

from __future__ import annotations

import io
import urllib.error
from decimal import Decimal

from regimes_probe.agent.planner import AgentConfig, EpistemicAgent
from regimes_probe.agent.reading_policy import select_reading_tool
from regimes_probe.datasets.base import Item
from regimes_probe.policy.contextual_bandit import BanditParams
from regimes_probe.policy.memory import PolicyMemory
from regimes_probe.policy.verification_policy import VerificationConfig
from regimes_probe.tools.base import SearchProvider, SearchResponse, SearchResult
from regimes_probe.tools.metadata import first_hop_tools, followup_tools, is_first_hop


# --------------------------------------------------- never a first-hop arm
def test_firecrawl_scrape_is_followup_not_first_hop():
    assert not is_first_hop("firecrawl_scrape")
    tools = ["serper_search", "exa_search", "firecrawl_search", "firecrawl_scrape", "page_fetch"]
    assert "firecrawl_scrape" not in first_hop_tools(tools)
    assert first_hop_tools(tools) == ["serper_search", "exa_search", "firecrawl_search"]
    assert set(followup_tools(tools)) == {"firecrawl_scrape", "page_fetch"}


# --------------------------------------------------- reading-tool selection policy
_KW = dict(unresolved_clue_terms=["accident"], answer_shape=[], cross_provider_domains=set(),
           page_fetch_available=True, scrape_available=True, scraped_urls=set(),
           no_progress_domains=set())


def test_scrape_preferred_for_pdf_and_structured_pages():
    d = select_reading_tool(url="https://gov.example/report.pdf", title="Report",
                            snippet="accident", source_authority=0.7, contaminated=False, **_KW)
    assert d.tool == "firecrawl_scrape" and d.is_scrape
    d2 = select_reading_tool(url="https://gov.example/records/42", title="Public records",
                             snippet="accident", source_authority=0.7, contaminated=False, **_KW)
    assert d2.tool == "firecrawl_scrape"


def test_page_fetch_for_simple_static_page():
    d = select_reading_tool(
        url="https://blog.example/post", title="A blog post",
        snippet="accident " * 60, source_authority=0.5, contaminated=False, **_KW)
    assert d.tool == "page_fetch" and not d.is_scrape


def test_reading_skips_contaminated_social_already_read_and_no_progress():
    assert select_reading_tool(url="https://x.example/a", title="t", snippet="accident",
                               source_authority=0.7, contaminated=True, **_KW).tool is None
    soc = dict(_KW)
    assert select_reading_tool(url="https://facebook.com/p", title="t", snippet="accident",
                               source_authority=0.7, contaminated=False, **soc).reason == "social_media"
    rd = dict(_KW); rd["scraped_urls"] = {"gov.example/x"}
    assert select_reading_tool(url="https://gov.example/x", title="t", snippet="accident",
                               source_authority=0.7, contaminated=False, **rd).reason == "already_read"
    npd = dict(_KW); npd["no_progress_domains"] = {"gov.example"}
    assert select_reading_tool(url="https://gov.example/y", title="t", snippet="accident",
                               source_authority=0.7, contaminated=False, **npd).reason == "no_progress_domain"


def test_reading_skips_generic_page_with_no_clue_match():
    nk = dict(_KW); nk["unresolved_clue_terms"] = ["zzzznotpresent"]
    d = select_reading_tool(url="https://blog.example/p", title="random", snippet="nothing here",
                            source_authority=0.2, contaminated=False, **nk)
    assert d.tool is None and d.reason == "no_clue_match"


def test_fallback_prefers_page_fetch():
    fk = dict(_KW); fk["prefer_page_fetch"] = True
    d = select_reading_tool(url="https://gov.example/report.pdf", title="Report",
                            snippet="accident", source_authority=0.9, contaminated=False, **fk)
    assert d.tool == "page_fetch" and d.reason == "fallback_after_scrape_failure"


def test_scrape_not_chosen_when_unavailable():
    nk = dict(_KW); nk["scrape_available"] = False
    d = select_reading_tool(url="https://gov.example/report.pdf", title="Report",
                            snippet="accident", source_authority=0.9, contaminated=False, **nk)
    assert d.tool == "page_fetch"


# --------------------------------------------------- adapter fail-closed (402 etc.)
def test_firecrawl_402_becomes_failed_response_not_crash():
    from regimes_probe.tools.base import safe_search

    class _Scrape402(SearchProvider):
        name = "firecrawl_scrape"
        is_fetch = True
        cost_per_call = Decimal("0.003")
        def available(self): return True
        def search(self, q, *, limit=1, **o):
            raise urllib.error.HTTPError("https://api.firecrawl.dev/v1/scrape", 402,
                                         "Payment Required", {}, io.BytesIO(b'{"error":"quota"}'))

    resp = safe_search(_Scrape402(), "https://gov.example/x")
    assert resp.failed and resp.error_meta["status_code"] == 402
    assert resp.error_meta["error_type"] == "HTTPError"
    assert resp.results == ()


# --------------------------------------------------- end-to-end loop integration
class _AuthSearch(SearchProvider):
    """Returns one supporting result on an authoritative structured page (short
    snippet) so the always_full stop policy chooses fetch_page."""
    name = "generic_web_search"
    cost_per_call = Decimal("0.002")
    deterministic = True

    def __init__(self, url="https://gov.example/records/edrin-vael", contaminated=False):
        self.url = url
        self.contaminated = contaminated

    def available(self): return True

    def search(self, q, *, limit=5, **o):
        return SearchResponse(provider=self.name, query=q, results=(
            SearchResult(title="Public records: Edrin Vael", url=self.url,
                         snippet="Edrin Vael, road accident.", source_authority=0.8, rank=0,
                         extra={"item_id": "x", "asserts": "Edrin Vael"}),), cost=self.cost_per_call)


class _RecordingScrape(SearchProvider):
    name = "firecrawl_scrape"
    is_fetch = True
    cost_per_call = Decimal("0.003")

    def __init__(self, *, fail=False):
        self.fail = fail
        self.queries: list[str] = []

    def available(self): return True

    def search(self, q, *, limit=1, **o):
        self.queries.append(q)
        if self.fail:
            raise urllib.error.HTTPError("https://api.firecrawl.dev/v1/scrape", 402,
                                         "Payment Required", {}, io.BytesIO(b"{}"))
        return SearchResponse(provider=self.name, query=q, results=(
            SearchResult(title="Edrin Vael full record", url=q,
                         snippet="Edrin Vael was born in 1951; road accident in 1970.",
                         source_authority=0.8, rank=0,
                         extra={"item_id": "x", "asserts": "Edrin Vael"}),), cost=self.cost_per_call)


class _RecordingFetch(SearchProvider):
    name = "page_fetch"
    is_fetch = True
    cost_per_call = Decimal("0.001")

    def __init__(self):
        self.queries: list[str] = []

    def available(self): return True

    def search(self, q, *, limit=1, **o):
        self.queries.append(q)
        return SearchResponse(provider=self.name, query=q, results=(
            SearchResult(title="basic", url=q, snippet="Edrin Vael born 1951.",
                         source_authority=0.6, rank=0, extra={"item_id": "x"}),),
            cost=self.cost_per_call)


def _agent():
    return EpistemicAgent(AgentConfig(
        available_tools=["generic_web_search", "firecrawl_scrape", "page_fetch"],
        stop_mode="always_full", enable_iterative_clue_resolution=True,
        verification=VerificationConfig(min_support=2)))


def _item():
    return Item(id="x", answer="Edrin Vael",
                question="Who is the author who survived a road accident?")


def test_scrape_only_receives_urls_and_counts_against_budget():
    scrape = _RecordingScrape()
    providers = {"generic_web_search": _AuthSearch(), "firecrawl_scrape": scrape,
                 "page_fetch": _RecordingFetch()}
    tr = _agent().attempt(_item(), PolicyMemory(BanditParams()), providers,
                          budget=3, explore=False, attempt_id="t")
    assert len(tr.calls) <= 3                               # reads count against budget
    scrape_calls = [c for c in tr.calls if c.scrape.get("read_tool") == "firecrawl_scrape"]
    assert scrape_calls, "expected a firecrawl_scrape read"
    assert all(q.startswith("http") for q in scrape.queries)   # URLs only, never a question
    assert scrape_calls[0].scrape["scrape_url"].startswith("http")
    assert scrape_calls[0].scrape["is_scrape"] is True


def test_contaminated_url_is_not_scraped():
    scrape = _RecordingScrape()
    providers = {"generic_web_search": _AuthSearch(url="https://huggingface.co/datasets/x"),
                 "firecrawl_scrape": scrape, "page_fetch": _RecordingFetch()}
    tr = _agent().attempt(_item(), PolicyMemory(BanditParams()), providers,
                          budget=3, explore=False, attempt_id="t")
    assert scrape.queries == []                             # contaminated host never scraped


def test_firecrawl_failure_falls_back_to_page_fetch_without_crashing():
    scrape = _RecordingScrape(fail=True)
    fetch = _RecordingFetch()
    providers = {"generic_web_search": _AuthSearch(), "firecrawl_scrape": scrape,
                 "page_fetch": fetch}
    tr = _agent().attempt(_item(), PolicyMemory(BanditParams()), providers,
                          budget=4, explore=False, attempt_id="t")
    # the scrape was attempted and failed (captured, no crash)...
    failed = [c for c in tr.calls if c.scrape.get("is_scrape") and not c.scrape.get("scrape_success")]
    assert failed and failed[0].scrape["scrape_failure_type"] == "HTTPError"
    assert failed[0].scrape.get("fallback_to_page_fetch") is True
    # ...and a page_fetch retried the SAME url.
    assert fetch.queries and fetch.queries[0] == scrape.queries[0]
    assert len(tr.calls) <= 4
