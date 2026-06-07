"""Provider adapters (the only place network I/O is allowed).

Adapters are wrapped as ActiveGraph tools in ``activegraph_pack.tools`` for
replay. The fixture-backed :mod:`fake` adapters power tests and Study 0; live
adapters are each gated behind an env var and are unavailable without it.
"""

from __future__ import annotations

from regimes_probe.tools.base import (
    ProviderUnavailable,
    SearchProvider,
    SearchResponse,
    SearchResult,
)
from regimes_probe.tools.fake import (
    FakePageFetch,
    FakeSearchProvider,
    build_fake_providers,
    load_corpus,
)

__all__ = [
    "ProviderUnavailable",
    "SearchProvider",
    "SearchResponse",
    "SearchResult",
    "FakePageFetch",
    "FakeSearchProvider",
    "build_fake_providers",
    "load_corpus",
]


#: Names of every adapter the project ships, with whether it is optional.
ADAPTER_REGISTRY: dict[str, dict[str, str]] = {
    "openai_web_search": {"module": "openai_web_search", "key": "OPENAI_API_KEY", "optional": "no"},
    "openai_web_search_low_context": {"module": "openai_web_search", "key": "OPENAI_API_KEY", "optional": "yes"},
    "openai_web_search_unlimited_context": {"module": "openai_web_search", "key": "OPENAI_API_KEY", "optional": "yes"},
    "generic_web_search": {"module": "generic_web_search", "key": "", "optional": "yes"},
    "news_search": {"module": "generic_web_search", "key": "NEWS_API_KEY", "optional": "yes"},
    "official_domain_search": {"module": "generic_web_search", "key": "OFFICIAL_SEARCH_API_KEY", "optional": "yes"},
    "page_fetch": {"module": "page_fetch", "key": "", "optional": "no"},
    "brave_search": {"module": "brave_search", "key": "BRAVE_SEARCH_API_KEY", "optional": "yes"},
    "tavily_search": {"module": "tavily_search", "key": "TAVILY_API_KEY", "optional": "yes"},
    "exa_search": {"module": "exa_search", "key": "EXA_API_KEY", "optional": "yes"},
    "serper_search": {"module": "serper_search", "key": "SERPER_API_KEY", "optional": "yes"},
    "academic_search": {"module": "generic_web_search", "key": "ACADEMIC_API_KEY", "optional": "yes"},
    "code_search": {"module": "generic_web_search", "key": "CODE_SEARCH_API_KEY", "optional": "yes"},
}
