"""OpenAI Responses API ``web_search`` adapters.

This is the *benchmark-matching hosted-search baseline*: the same hosted
browsing tool the benchmark assumes, wrapped behind the common
:class:`SearchProvider` interface so the router stays provider-agnostic.

Three variants share one implementation, differing only in the context budget
passed to the model:

  * ``openai_web_search``                  — default
  * ``openai_web_search_low_context``      — small context, cheaper
  * ``openai_web_search_unlimited_context``— optional, expensive

All require ``OPENAI_API_KEY`` and the ``openai`` package. ``available()`` is
False otherwise, so unit tests never touch them.

Docs:
  https://developers.openai.com/api/docs/models
  https://developers.openai.com/api/docs/guides/tools-web-search
"""

from __future__ import annotations

import os
import time
from decimal import Decimal
from typing import Any

from regimes_probe.tools.base import (
    ProviderUnavailable,
    SearchProvider,
    SearchResponse,
    SearchResult,
)


class OpenAIWebSearch(SearchProvider):
    deterministic = False

    def __init__(
        self,
        *,
        name: str = "openai_web_search",
        model: str = "gpt-5.5",
        context_size: str = "medium",   # "low" | "medium" | "high"
        cost_per_call: Decimal | str = "0.03",
    ) -> None:
        self.name = name
        self.model = model
        self.context_size = context_size
        self.cost_per_call = (
            cost_per_call if isinstance(cost_per_call, Decimal) else Decimal(str(cost_per_call))
        )

    def available(self) -> bool:
        if not os.environ.get("OPENAI_API_KEY"):
            return False
        try:
            import openai  # noqa: F401
        except Exception:
            return False
        return True

    def search(self, query: str, *, limit: int = 5, **opts: Any) -> SearchResponse:
        if not self.available():
            raise ProviderUnavailable(
                f"{self.name} requires OPENAI_API_KEY and the 'openai' package"
            )
        from openai import OpenAI  # local import: never at module load

        client = OpenAI()
        t0 = time.monotonic()
        tool: dict[str, Any] = {"type": "web_search"}
        # Constrained search (official_domain_search) passes allowed_domains.
        allowed = opts.get("allowed_domains")
        if allowed:
            tool["filters"] = {"allowed_domains": list(allowed)}
        if self.context_size != "unlimited":
            tool["search_context_size"] = self.context_size
        resp = client.responses.create(
            model=self.model,
            input=query,
            tools=[tool],
        )
        latency = time.monotonic() - t0
        results = _extract_results(resp, limit)
        return SearchResponse(
            provider=self.name,
            query=query,
            results=tuple(results),
            cost=self.cost_per_call,
            latency_s=latency,
        )


def _extract_results(resp: Any, limit: int) -> list[SearchResult]:
    """Pull URL citations out of a Responses API result, best-effort."""
    results: list[SearchResult] = []
    try:
        for item in getattr(resp, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                for ann in getattr(content, "annotations", []) or []:
                    if getattr(ann, "type", "") == "url_citation":
                        results.append(
                            SearchResult(
                                title=getattr(ann, "title", "") or "",
                                url=getattr(ann, "url", "") or "",
                                snippet=getattr(content, "text", "")[:500],
                                rank=len(results),
                            )
                        )
                        if len(results) >= limit:
                            return results
    except Exception:
        pass
    return results


def openai_web_search() -> OpenAIWebSearch:
    return OpenAIWebSearch(name="openai_web_search", context_size="medium")


def openai_web_search_low_context() -> OpenAIWebSearch:
    return OpenAIWebSearch(
        name="openai_web_search_low_context", context_size="low", cost_per_call="0.015"
    )


def openai_web_search_unlimited_context() -> OpenAIWebSearch:
    return OpenAIWebSearch(
        name="openai_web_search_unlimited_context",
        context_size="unlimited",
        cost_per_call="0.12",
    )
