"""Tavily Search adapter (optional). Requires ``TAVILY_API_KEY``."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from decimal import Decimal
from typing import Any

from regimes_probe.tools.base import (
    ProviderUnavailable,
    SearchProvider,
    SearchResponse,
    SearchResult,
)

_ENDPOINT = "https://api.tavily.com/search"


class TavilySearch(SearchProvider):
    name = "tavily_search"
    deterministic = False
    cost_per_call = Decimal("0.004")

    def available(self) -> bool:
        return bool(os.environ.get("TAVILY_API_KEY"))

    def search(self, query: str, *, limit: int = 5, **opts: Any) -> SearchResponse:
        if not self.available():
            raise ProviderUnavailable("tavily_search requires TAVILY_API_KEY")
        # Current Tavily API: Bearer-token AUTH HEADER (not api_key in the body).
        # The old body-key form returns HTTP 400 on the current API.
        body = {
            "query": query,
            "max_results": limit,
            "search_depth": opts.get("search_depth", "basic"),
        }
        if opts.get("allowed_domains"):
            body["include_domains"] = list(opts["allowed_domains"])
        req = urllib.request.Request(
            _ENDPOINT,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"},
        )
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            data = json.loads(resp.read().decode())
        rows = data.get("results", [])
        results = [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
                source_authority=float(r.get("score", 0.5)),
                rank=i,
            )
            for i, r in enumerate(rows[:limit])
        ]
        return SearchResponse(
            provider=self.name,
            query=query,
            results=tuple(results),
            cost=self.cost_per_call,
            latency_s=time.monotonic() - t0,
        )
