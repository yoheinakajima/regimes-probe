"""Level 3 evidence reading: choose a URL-reading tool (page_fetch vs
firecrawl_scrape) and gate WHEN to read (deterministic, gold-free).

Reading tools run on a URL produced by earlier search/candidate selection — they
are never first-hop search arms. ``page_fetch`` is a cheap static-HTML fetch;
``firecrawl_scrape`` is a richer (and paid / quota-limited) extractor for complex,
JS-heavy, or PDF pages, or pages with likely hidden structured content. We only
read a page that is plausibly on the evidence path, and we never re-read an
equivalent URL or a domain that already produced no progress.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

#: Social/platform hosts we do not read unless explicitly allowed.
SOCIAL_DOMAINS = (
    "facebook.com", "twitter.com", "x.com", "instagram.com", "tiktok.com",
    "reddit.com", "linkedin.com", "pinterest.com", "youtube.com", "threads.net",
    "t.me", "snapchat.com", "tumblr.com", "weibo.com", "vk.com", "mastodon.social",
)
#: Title/snippet markers that suggest hidden / structured content worth richer
#: extraction (a basic snippet/fetch is likely insufficient).
_STRUCTURED_MARKERS = (
    "profile", "records", "record", "database", "directory", "dataset", "census",
    "statistics", "table", "registry", "archive", "filing", "obituary", "catalog",
    "catalogue", "index of", "full text", "annual report", "proceedings",
)
#: Below this snippet length we treat the search snippet as insufficient.
_INSUFFICIENT_SNIPPET = 200


def _host(url: str) -> str:
    return (urlparse(url or "").hostname or "").lower()


def normalize_url(url: str) -> str:
    """Equivalence key: host + path without trailing slash / fragment / query."""
    p = urlparse(url or "")
    host = (p.hostname or "").lower()
    path = (p.path or "").rstrip("/").lower()
    return f"{host}{path}"


def _is_social(host: str) -> bool:
    return any(host == d or host.endswith("." + d) for d in SOCIAL_DOMAINS)


@dataclass
class ReadingDecision:
    tool: Optional[str]          # "firecrawl_scrape" | "page_fetch" | None (skip)
    query_arm: str               # "scrape" | "fetch" | "skip"
    reason: str                  # scrape_selected_reason / skip reason
    is_scrape: bool = False

    def to_dict(self):
        return {"read_tool": self.tool, "query_arm": self.query_arm,
                "reason": self.reason, "is_scrape": self.is_scrape}


def select_reading_tool(
    *, url: str, title: str, snippet: str, source_authority: float,
    contaminated: bool, unresolved_clue_terms, answer_shape,
    cross_provider_domains, page_fetch_available: bool, scrape_available: bool,
    scraped_urls, no_progress_domains, allow_social: bool = False,
    prefer_page_fetch: bool = False, force_read: bool = False) -> ReadingDecision:
    """Decide whether and how to read ``url``. Returns ``tool=None`` to skip."""
    host = _host(url)
    nurl = normalize_url(url)
    text = f"{title or ''} {snippet or ''}".lower()
    clue_terms = [t.lower() for t in (unresolved_clue_terms or []) if len(t) >= 3]
    shapes = [s.lower() for s in (answer_shape or [])]

    if not (page_fetch_available or scrape_available):
        return ReadingDecision(None, "skip", "no_reading_tool")
    if nurl in scraped_urls:
        return ReadingDecision(None, "skip", "already_read")
    if contaminated:
        return ReadingDecision(None, "skip", "contaminated_url")
    if _is_social(host) and not allow_social:
        return ReadingDecision(None, "skip", "social_media")
    if host and host in no_progress_domains:
        return ReadingDecision(None, "skip", "no_progress_domain")

    clue_match = any(t in text for t in clue_terms)
    shape_match = any(s in text for s in shapes)
    authoritative = source_authority >= 0.5
    cross = bool(host and host in (cross_provider_domains or set()))
    # Don't read a generic source page with no target-relevant signal (unless this
    # is a known multihop target the loop explicitly asked to read).
    if not (force_read or clue_match or shape_match or cross or authoritative):
        return ReadingDecision(None, "skip", "no_clue_match")

    insufficient = len((snippet or "").strip()) < _INSUFFICIENT_SNIPPET
    is_pdf = nurl.endswith(".pdf")
    structured = any(m in text for m in _STRUCTURED_MARKERS)

    # Tool choice: page_fetch is the cheap default + fallback; firecrawl_scrape is
    # reserved for pages where a basic fetch/snippet is likely insufficient.
    if prefer_page_fetch and page_fetch_available:
        return ReadingDecision("page_fetch", "fetch", "fallback_after_scrape_failure")
    if not scrape_available:
        if page_fetch_available:
            return ReadingDecision("page_fetch", "fetch", "basic_fetch_scrape_unavailable")
        return ReadingDecision(None, "skip", "no_reading_tool")
    if is_pdf:
        return ReadingDecision("firecrawl_scrape", "scrape", "pdf_needs_rich_extraction", True)
    if structured or (authoritative and insufficient) or (cross and insufficient):
        why = ("structured_content" if structured else
               "authoritative_insufficient_snippet" if authoritative else
               "cross_provider_insufficient_snippet")
        return ReadingDecision("firecrawl_scrape", "scrape", why, True)
    if page_fetch_available:
        return ReadingDecision("page_fetch", "fetch", "basic_static_fetch")
    return ReadingDecision("firecrawl_scrape", "scrape", "only_reading_tool_available", True)
