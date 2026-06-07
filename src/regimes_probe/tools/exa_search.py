"""Exa (neural) Search adapter (optional). Requires ``EXA_API_KEY``."""

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

_ENDPOINT = "https://api.exa.ai/search"


class ExaSearch(SearchProvider):
    name = "exa_search"
    deterministic = False
    cost_per_call = Decimal("0.005")

    def available(self) -> bool:
        return bool(os.environ.get("EXA_API_KEY"))

    def search(self, query: str, *, limit: int = 5, **opts: Any) -> SearchResponse:
        if not self.available():
            raise ProviderUnavailable("exa_search requires EXA_API_KEY")
        body = {"query": query, "numResults": limit, "contents": {"text": True}}
        if "allowed_domains" in opts:
            body["includeDomains"] = list(opts["allowed_domains"])
        req = urllib.request.Request(
            _ENDPOINT,
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "x-api-key": os.environ["EXA_API_KEY"],
            },
        )
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            data = json.loads(resp.read().decode())
        rows = data.get("results", [])
        results = [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=(r.get("text", "") or "")[:500],
                published_at=r.get("publishedDate"),
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
