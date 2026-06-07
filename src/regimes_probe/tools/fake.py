"""Deterministic, fixture-backed search/fetch providers.

These power Study 0 (harness validity) and every unit test: no network, no
keys, fully deterministic. The corpus is engineered so that *which tool* and
*which query arm* you choose changes whether the answer-bearing document is
surfaced — that is the learnable signal the contextual bandit exploits.

Corpus document schema (``fixtures/fake_search_corpus.json``)::

    {
      "doc_id": "d_syn-001_news",
      "item_id": "syn-001",
      "title": "...",
      "url": "https://news.example.com/acme",
      "snippet": "... Acme Robotics ...",   # answer text iff contains_answer
      "published_at": "2026-05-02",
      "source_authority": 0.9,
      "contains_answer": true,               # snippet reveals the answer
      "answer_on_fetch_only": false,         # multihop: answer only after fetch
      "tools": ["news_search", "generic_web_search"],
      "match_tokens": ["acme", "robotics", "ceo"],
      "requires_tokens": ["2026"],           # query must contain ALL of these
      "base_rank": 0
    }

Retrieval is pure token overlap with arm/tool gating — intentionally simple so
the harness behaviour is auditable, not a black box.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from regimes_probe.tools.base import SearchProvider, SearchResponse, SearchResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. Deterministic; shared with retrieval."""
    return _TOKEN_RE.findall(text.lower())


def load_corpus(path: str | Path) -> list[dict[str, Any]]:
    """Load the fake search corpus JSON (a list of document dicts)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "documents" in data:
        data = data["documents"]
    if not isinstance(data, list):
        raise ValueError("fake corpus must be a list of documents")
    return data


class FakeSearchProvider(SearchProvider):
    """A search tool backed by the fixture corpus.

    Each instance represents one *tool family* (e.g. ``news_search``). A
    document is surfaced by this provider only if the provider's name is in the
    document's ``tools`` list, all of the document's ``requires_tokens`` appear
    in the query, and at least one ``match_token`` overlaps the query.
    """

    deterministic = True

    def __init__(
        self,
        name: str,
        corpus: list[dict[str, Any]],
        *,
        cost_per_call: Decimal | str | float = "0.002",
        noise_docs: Optional[list[dict[str, Any]]] = None,
        latency_s: float = 0.0,
    ) -> None:
        self.name = name
        self._corpus = corpus
        self.cost_per_call = (
            cost_per_call if isinstance(cost_per_call, Decimal) else Decimal(str(cost_per_call))
        )
        self._noise = noise_docs or []
        self._latency = latency_s

    def available(self) -> bool:
        return True

    def search(self, query: str, *, limit: int = 5, **opts: Any) -> SearchResponse:
        q_tokens = set(tokenize(query))
        allowed_domains = opts.get("allowed_domains")
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for doc in self._corpus:
            if self.name not in doc.get("tools", []):
                continue
            requires = doc.get("requires_tokens", [])
            if any(tok not in q_tokens for tok in requires):
                continue
            match_tokens = set(doc.get("match_tokens", []))
            overlap = len(q_tokens & match_tokens)
            if overlap == 0:
                continue
            if allowed_domains is not None:
                host = urlparse(doc.get("url", "")).hostname or ""
                if not any(host == d or host.endswith("." + d) for d in allowed_domains):
                    continue
            scored.append((doc.get("base_rank", 99), -overlap, doc))

        scored.sort(key=lambda t: (t[0], t[1], t[2].get("doc_id", "")))
        results: list[SearchResult] = []
        for rank, (_, _, doc) in enumerate(scored[:limit]):
            # A doc *asserts* an answer (gold or distractor). Multihop docs
            # ("answer_on_fetch_only") hide that assertion from the snippet, so
            # the stop/verify policy must choose to fetch the page to read it.
            reveal = not doc.get("answer_on_fetch_only", False)
            snippet = doc["snippet"] if reveal else _redact_answer(doc)
            results.append(
                SearchResult(
                    title=doc.get("title", ""),
                    url=doc.get("url", ""),
                    snippet=snippet,
                    published_at=doc.get("published_at"),
                    source_authority=float(doc.get("source_authority", 0.5)),
                    rank=rank,
                    extra={
                        "doc_id": doc.get("doc_id"),
                        "item_id": doc.get("item_id"),
                        "answer_on_fetch_only": bool(doc.get("answer_on_fetch_only")),
                        "asserts": doc.get("asserts") if reveal else None,
                        "is_distractor": bool(doc.get("is_distractor")),
                    },
                )
            )
        return SearchResponse(
            provider=self.name,
            query=query,
            results=tuple(results),
            cost=self.cost_per_call,
            latency_s=self._latency,
        )


class FakePageFetch(SearchProvider):
    """A page-fetch tool: given a URL, returns the full document.

    For multihop items (``answer_on_fetch_only``) the search snippet hides the
    answer but the fetched page reveals it — so the stop/verify policy must
    decide to ``fetch_page`` rather than stop on the snippet.
    """

    name = "page_fetch"
    deterministic = True
    is_fetch = True

    def __init__(
        self,
        corpus: list[dict[str, Any]],
        *,
        cost_per_call: Decimal | str | float = "0.001",
    ) -> None:
        self._by_url = {d.get("url"): d for d in corpus}
        self.cost_per_call = (
            cost_per_call if isinstance(cost_per_call, Decimal) else Decimal(str(cost_per_call))
        )

    def search(self, query: str, *, limit: int = 5, **opts: Any) -> SearchResponse:
        # For a fetch tool, ``query`` is the URL to retrieve.
        doc = self._by_url.get(query)
        results: list[SearchResult] = []
        if doc is not None:
            results.append(
                SearchResult(
                    title=doc.get("title", ""),
                    url=doc.get("url", ""),
                    snippet=doc.get("snippet", ""),  # full page reveals answer
                    published_at=doc.get("published_at"),
                    source_authority=float(doc.get("source_authority", 0.5)),
                    rank=0,
                    extra={
                        "doc_id": doc.get("doc_id"),
                        "item_id": doc.get("item_id"),
                        "fetched": True,
                        "asserts": doc.get("asserts"),  # fetch always reveals
                        "is_distractor": bool(doc.get("is_distractor")),
                    },
                )
            )
        return SearchResponse(
            provider=self.name,
            query=query,
            results=tuple(results),
            cost=self.cost_per_call,
        )


def _redact_answer(doc: dict[str, Any]) -> str:
    """Hide the asserted answer from a snippet (multihop: fetch to reveal)."""
    ans = doc.get("asserts")
    snip = doc.get("snippet", "")
    if ans:
        snip = re.sub(re.escape(ans), "…", snip, flags=re.IGNORECASE)
    return snip


def build_fake_providers(
    corpus: list[dict[str, Any]],
    tool_names: list[str],
    *,
    costs: Optional[dict[str, str]] = None,
) -> dict[str, SearchProvider]:
    """Construct one :class:`FakeSearchProvider` per tool name plus page_fetch."""
    costs = costs or {}
    providers: dict[str, SearchProvider] = {}
    for name in tool_names:
        providers[name] = FakeSearchProvider(
            name, corpus, cost_per_call=costs.get(name, "0.002")
        )
    providers["page_fetch"] = FakePageFetch(corpus, cost_per_call=costs.get("page_fetch", "0.001"))
    return providers
