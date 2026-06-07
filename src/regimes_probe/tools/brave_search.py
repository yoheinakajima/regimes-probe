"""Brave Search adapter (optional). Requires ``BRAVE_SEARCH_API_KEY``."""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from decimal import Decimal
from typing import Any

from regimes_probe.tools.base import (
    ProviderUnavailable,
    SearchProvider,
    SearchResponse,
    SearchResult,
)

_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


class BraveSearch(SearchProvider):
    name = "brave_search"
    deterministic = False
    cost_per_call = Decimal("0.003")

    def available(self) -> bool:
        return bool(os.environ.get("BRAVE_SEARCH_API_KEY"))

    def search(self, query: str, *, limit: int = 5, **opts: Any) -> SearchResponse:
        if not self.available():
            raise ProviderUnavailable("brave_search requires BRAVE_SEARCH_API_KEY")
        url = _ENDPOINT + "?" + urllib.parse.urlencode({"q": query, "count": limit})
        req = urllib.request.Request(
            url,
            headers={
                "X-Subscription-Token": os.environ["BRAVE_SEARCH_API_KEY"],
                "Accept": "application/json",
            },
        )
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            data = json.loads(resp.read().decode())
        rows = (data.get("web") or {}).get("results", [])
        results = [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("description", ""),
                published_at=r.get("age"),
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
