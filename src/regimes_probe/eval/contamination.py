"""Benchmark-contamination detector for search results (no network).

A clean BrowseComp run can score 0 not because retrieval is broken but because
search surfaces *the benchmark itself*: pages that mirror the question text
(leaderboards, dataset cards, eval repos, arXiv writeups) rather than an evidence
source that carries the answer. Those results are worse than useless — they look
relevant (they contain the question) but contain no independent evidence, and
rewarding the query/tool that fetched them teaches the wrong lesson.

Heuristic (all answer-free; uses only the question + the result text):
  * the result title/snippet mirrors a long span of the original question, or
  * the result domain is a known benchmark/eval host (HF / GitHub / arXiv /
    simple-evals / paperswithcode …), or
  * the title/snippet names the benchmark itself (``BrowseComp`` etc.).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

_TOKEN = re.compile(r"[a-z0-9]+")

#: Domains that host benchmarks / eval code / dataset cards, not primary evidence.
CONTAMINATION_DOMAINS: tuple[str, ...] = (
    "huggingface.co", "github.com", "raw.githubusercontent.com", "gist.github.com",
    "arxiv.org", "openreview.net", "paperswithcode.com", "kaggle.com",
    "github.io", "openai.com/index/browsecomp",
)
#: URL/text substrings that signal a benchmark page regardless of host.
_BENCHMARK_MARKERS = ("simple-evals", "simple_evals", "browsecomp", "browse_comp",
                      "browse-comp", "browsecomp.csv", "leaderboard")
#: Names of the benchmarks themselves appearing in title/snippet.
_BENCHMARK_NAMES = ("browsecomp", "browse comp", "livebrowsecomp", "live browsecomp",
                    "simple-evals", "simpleqa", "gpqa benchmark")

#: Min run of consecutive question tokens mirrored in a result to flag it.
MIN_MIRROR_RUN = 8
#: Min fraction of (long) question tokens present in the result to flag it.
MIN_MIRROR_COVERAGE = 0.6
_LONG_Q_TOKENS = 14


def _toks(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def longest_token_run(question: str, text: str) -> int:
    """Longest run of consecutive question tokens appearing (in order) in text."""
    q = _toks(question)
    t = _toks(text)
    if not q or not t:
        return 0
    tset_index: dict[str, list[int]] = {}
    for i, w in enumerate(t):
        tset_index.setdefault(w, []).append(i)
    best = 0
    # DP over starting positions in the question (bounded; questions are short).
    for qi in range(len(q)):
        for ti in tset_index.get(q[qi], []):
            run = 0
            while (qi + run < len(q) and ti + run < len(t)
                   and q[qi + run] == t[ti + run]):
                run += 1
            best = max(best, run)
    return best


def _coverage(question: str, text: str) -> float:
    q = set(_toks(question))
    if not q:
        return 0.0
    t = set(_toks(text))
    return len(q & t) / len(q)


@dataclass
class ContaminationResult:
    contaminated: bool
    reason: Optional[str]
    domain: str

    def to_dict(self):
        return {"contaminated": self.contaminated, "reason": self.reason, "domain": self.domain}


def detect_contamination(*, url: str, title: str, snippet: str,
                         question: str) -> ContaminationResult:
    """Flag a result that mirrors the benchmark/question rather than evidence."""
    host = (urlparse(url or "").hostname or "").lower()
    text = f"{title or ''} {snippet or ''}"
    text_l = text.lower()
    url_l = (url or "").lower()

    if any(host == d or host.endswith("." + d) or d in url_l for d in CONTAMINATION_DOMAINS):
        return ContaminationResult(True, "benchmark/eval host domain", host)
    if any(m in url_l or m in text_l for m in _BENCHMARK_MARKERS):
        return ContaminationResult(True, "benchmark marker in url/text", host)
    if any(n in text_l for n in _BENCHMARK_NAMES):
        return ContaminationResult(True, "names the benchmark", host)

    run = longest_token_run(question, text)
    if run >= MIN_MIRROR_RUN:
        return ContaminationResult(True, f"mirrors {run}-token span of the question", host)
    qtoks = _toks(question)
    if len(qtoks) >= _LONG_Q_TOKENS and _coverage(question, text) >= MIN_MIRROR_COVERAGE:
        return ContaminationResult(True, "mirrors most of the question text", host)
    return ContaminationResult(False, None, host)
