"""Page-fetch adapter: retrieve a URL and return its text as a single result.

Used by the stop/verify policy to corroborate a candidate answer or to reach
multihop answers hidden from search snippets. Live fetching uses the stdlib
only (no extra deps); the fixture-backed :class:`~regimes_probe.tools.fake.FakePageFetch`
covers tests.
"""

from __future__ import annotations

import time
import urllib.request
from decimal import Decimal
from html.parser import HTMLParser
from typing import Any

from regimes_probe.tools.base import SearchProvider, SearchResponse, SearchResult


def _looks_like_url(s: str) -> bool:
    return isinstance(s, str) and s.strip().lower().startswith(("http://", "https://"))


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip and data.strip():
            self._chunks.append(data.strip())

    @property
    def text(self) -> str:
        return " ".join(self._chunks)


class PageFetch(SearchProvider):
    name = "page_fetch"
    deterministic = False
    is_fetch = True

    def __init__(self, *, cost_per_call: Decimal | str = "0.001", max_chars: int = 4000,
                 raw_chars: int = 0) -> None:
        self.cost_per_call = (
            cost_per_call if isinstance(cost_per_call, Decimal) else Decimal(str(cost_per_call))
        )
        self.max_chars = max_chars
        #: 5m-7: when > 0, expose a BOUNDED raw body via fetch_meta["raw_text"] so the
        #: recording cache (store_raw) can persist it for offline replay validation.
        self.raw_chars = raw_chars

    def available(self) -> bool:
        return True

    def search(self, query: str, *, limit: int = 1, **opts: Any) -> SearchResponse:
        # ``query`` MUST be a URL — page_fetch is a follow-up tool, not a search
        # arm. If routed first-hop with a search query, fail gracefully with a
        # clear ``requires_url`` error instead of an opaque urllib failure.
        if not _looks_like_url(query):
            meta = {
                "tool": self.name, "provider": self.name, "error_type": "requires_url",
                "status_code": None, "success": False,
                "message": ("page_fetch requires a URL (http/https), not a search query; "
                            "it is a follow-up tool, not a first-hop search arm"),
            }
            return SearchResponse(
                provider=self.name, query=query, results=(), cost=Decimal("0"),
                latency_s=0.0, error=meta["message"], error_meta=meta)
        t0 = time.monotonic()
        # 5t-6: a per-call cap override (bounded predicate re-read) — mirrors the existing
        # firecrawl_scrape opt; the default stays the conservative adapter cap.
        cap = int(opts.get("max_chars", self.max_chars))
        try:
            req = urllib.request.Request(query, headers={"User-Agent": "regimes-probe/0.1"})
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                raw = resp.read().decode("utf-8", errors="replace")
                final_url = resp.geturl()
            parser = _TextExtractor()
            parser.feed(raw)
            full = parser.text
            text = full[:cap]
            results = (
                SearchResult(title=query, url=query, snippet=text, rank=0),
            )
            err = None
            # 5j-B: the 4000-char cap is an ADAPTER cap (this tool's ``max_chars``), applied to
            # the parsed page text BEFORE it becomes the stored/judged snippet. Record both the
            # fetched (parsed) length and the stored length so truncation is auditable, not a
            # silent display artifact.
            fetch_meta = {
                "fetched_chars": len(full),
                "stored_body_chars": len(text),
                "adapter_max_chars": cap,
                "body_truncated_for_storage": len(full) > cap,
                "truncation_origin": "page_fetch_adapter_max_chars",
            }
            if self.raw_chars > 0:
                fetch_meta["raw_text"] = full[: self.raw_chars]
            if final_url and final_url != query:
                fetch_meta["final_url"] = final_url      # 5r-5: redirect provenance
        except Exception as exc:  # network failure surfaces as an error response
            results = ()
            err = str(exc)
            fetch_meta = None
        return SearchResponse(
            provider=self.name,
            query=query,
            results=results,
            cost=self.cost_per_call,
            latency_s=time.monotonic() - t0,
            error=err,
            fetch_meta=fetch_meta,
        )
