"""Firecrawl adapters (optional): search, scrape, and (scaffold) interact.

  * ``firecrawl_search`` — web search returning results + optional page content.
  * ``firecrawl_scrape``  — fetch a URL as markdown/html + metadata (richer
    evidence than a search snippet).
  * ``firecrawl_interact``— browser-like interaction; SCAFFOLD, disabled by
    default (gated behind ``--enable-browserish-tools``).

Requires ``FIRECRAWL_API_KEY``. All calls go through the recording cache via the
live invoker. Docs: https://docs.firecrawl.dev/
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from decimal import Decimal
from typing import Any, Optional

from regimes_probe.tools.base import (
    ProviderUnavailable, SearchProvider, SearchResponse, SearchResult, ToolDisabled)

_BASE = os.environ.get("FIRECRAWL_BASE_URL", "https://api.firecrawl.dev/v1")


def _post(path: str, body: dict[str, Any]) -> dict[str, Any]:
    key = os.environ["FIRECRAWL_API_KEY"]
    req = urllib.request.Request(
        f"{_BASE}/{path.lstrip('/')}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        return json.loads(resp.read().decode())


class _FirecrawlBase(SearchProvider):
    deterministic = False

    def available(self) -> bool:
        return bool(os.environ.get("FIRECRAWL_API_KEY"))


class FirecrawlSearch(_FirecrawlBase):
    name = "firecrawl_search"
    cost_per_call = Decimal("0.004")

    def search(self, query: str, *, limit: int = 5, **opts: Any) -> SearchResponse:
        if not self.available():
            raise ProviderUnavailable("firecrawl_search requires FIRECRAWL_API_KEY")
        t0 = time.monotonic()
        data = _post("search", {"query": query, "limit": limit})
        rows = data.get("data") or data.get("results") or data.get("web") or []
        results = [_web_to_result(r, i) for i, r in enumerate(rows[:limit])]
        return SearchResponse(provider=self.name, query=query, results=tuple(results),
                              cost=self.cost_per_call, latency_s=time.monotonic() - t0)


class FirecrawlScrape(_FirecrawlBase):
    name = "firecrawl_scrape"
    is_fetch = True
    cost_per_call = Decimal("0.003")

    def __init__(self, *, enabled: bool = False, max_chars: int = 4000,
                 raw_chars: int = 0) -> None:
        # 5l-8: the stored-snippet cap is now CONFIGURABLE (was a hardcoded [:4000]). A run can
        # set a larger cap for read-judgment-backed reads (judge input stays bounded by passage
        # windows regardless). Default stays conservative for ordinary reads.
        self.max_chars = max_chars
        #: 5m-7: when > 0, expose a BOUNDED raw body via fetch_meta["raw_text"] so the
        #: recording cache (store_raw) can persist it for offline replay validation.
        self.raw_chars = raw_chars

    def search(self, query: str, *, limit: int = 1, **opts: Any) -> SearchResponse:
        # ``query`` is the URL to scrape.
        if not self.available():
            raise ProviderUnavailable("firecrawl_scrape requires FIRECRAWL_API_KEY")
        t0 = time.monotonic()
        formats = list(opts.get("formats", ["markdown"]))
        cap = int(opts.get("max_chars", self.max_chars))
        data = _post("scrape", {"url": query, "formats": formats})
        doc = data.get("data") or data
        md = str(doc.get("markdown") or doc.get("html") or doc.get("content") or "")
        meta = doc.get("metadata") or {}
        result = SearchResult(
            title=meta.get("title", query), url=query, snippet=md[:cap],
            published_at=meta.get("publishedTime") or meta.get("date"),
            source_authority=0.6, rank=0,
            extra={"provider": "firecrawl", "tool_family": "scrape", "metadata": meta})
        fetch_meta = {"fetched_chars": len(md), "stored_body_chars": len(md[:cap]),
                      "adapter_max_chars": cap, "body_truncated_for_storage": len(md) > cap,
                      "truncation_origin": "firecrawl_scrape_adapter_max_chars"}
        if self.raw_chars > 0:
            fetch_meta["raw_text"] = md[: self.raw_chars]
        return SearchResponse(provider=self.name, query=query, results=(result,),
                              cost=self.cost_per_call, latency_s=time.monotonic() - t0,
                              fetch_meta=fetch_meta)


class FirecrawlInteract(_FirecrawlBase):
    """Browser-like interaction. SCAFFOLD — disabled by default."""

    name = "firecrawl_interact"
    cost_per_call = Decimal("0")

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = enabled

    def search(self, query: str, *, limit: int = 1, **opts: Any) -> SearchResponse:
        if not self.enabled:
            raise ToolDisabled(
                "firecrawl_interact is a browser-like tool, disabled by default. Enable "
                "explicitly with --enable-browserish-tools (introduces state / "
                "prompt-injection risk; not part of the default search-routing experiment).")
        if not self.available():
            raise ProviderUnavailable("firecrawl_interact requires FIRECRAWL_API_KEY")
        t0 = time.monotonic()
        data = _post("interact", {"url": query, "actions": opts.get("actions", [])})
        doc = data.get("data") or data
        result = SearchResult(title=query, url=query,
                              snippet=str(doc.get("markdown") or doc.get("content") or "")[:4000],
                              rank=0, extra={"provider": "firecrawl", "tool_family": "browserish"})
        return SearchResponse(provider=self.name, query=query, results=(result,),
                              cost=self.cost_per_call, latency_s=time.monotonic() - t0)


def _web_to_result(r: dict[str, Any], i: int) -> SearchResult:
    return SearchResult(
        title=r.get("title", ""), url=r.get("url") or r.get("link", ""),
        snippet=(r.get("markdown") or r.get("description") or r.get("snippet") or "")[:1500],
        published_at=r.get("date") or (r.get("metadata") or {}).get("date"),
        source_authority=0.5, rank=i,
        extra={"provider": "firecrawl", "tool_family": "search"})


def firecrawl_search() -> FirecrawlSearch:
    return FirecrawlSearch()


def firecrawl_scrape() -> FirecrawlScrape:
    return FirecrawlScrape()


def firecrawl_interact(*, enabled: bool = False) -> FirecrawlInteract:
    return FirecrawlInteract(enabled=enabled)
