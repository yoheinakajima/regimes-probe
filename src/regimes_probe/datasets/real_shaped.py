"""Generate the *real-data-shaped* placeholder fixtures.

These fixtures mimic the **structure** of BrowseComp (an obfuscated CSV with
``problem`` / ``answer`` / ``canary`` columns) and LiveBrowseComp (a JSONL of
``question`` / ``answer`` / ``released_at`` rows), but every question is an
obviously-fictional, public-domain-style placeholder. They exist ONLY to prove
the real-benchmark *adapter path* (decode, split, report, leakage checks, budget
curves, memory snapshot) works on real-data-shaped inputs.

They are NOT BrowseComp/LiveBrowseComp content and must never be reported as
such. See ``docs/REAL_BENCHMARK_READINESS.md`` and ``docs/METHODOLOGY_RISKS.md``.

Run via ``scripts/build_real_shaped_fixtures.py``; outputs are committed under
``fixtures/real_shaped/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from regimes_probe.datasets.browsecomp import encrypt
from regimes_probe.tools.fake import tokenize

# (id, question, answer, distractor, gold_tool, released_at, canary)
_PLACEHOLDERS = [
    ("rsq-001", "In this placeholder corpus, what is the capital of the fictional country Zembla?",
     "Novograd", "Old Vexen", "generic_web_search", "2025-11-02", "PLACEHOLDER-CANARY-1"),
    ("rsq-002", "In the synthetic record, who is the current steward of the fictional Greyhold Keep?",
     "Edrin Vael", "Mira Sloan", "news_search", "2026-04-12", "PLACEHOLDER-CANARY-2"),
    ("rsq-003", "Per the official charter in this placeholder set, what license code governs the fictional Aurora Guild?",
     "AUR-7781", "AUR-0000", "official_domain_search", "2025-06-01", "PLACEHOLDER-CANARY-3"),
    ("rsq-004", "In which fictional city was the placeholder company Tindle and Co founded?",
     "Marrowend", "Caldis", "brave_search", "2024-02-09", "PLACEHOLDER-CANARY-4"),
    ("rsq-005", "What is the middle name of the founder of the fictional placeholder order Lumen Circle?",
     "Aurelia", "Bartholomew", "generic_web_search", "2024-09-21", "PLACEHOLDER-CANARY-5"),
    ("rsq-006", "In the synthetic record, what is the latest airship model from the fictional Skyforge works?",
     "SF-12 Cirrus", "SF-01 Drift", "news_search", "2026-05-03", "PLACEHOLDER-CANARY-6"),
    ("rsq-007", "What color is the banner of the fictional placeholder house Calderwood?",
     "indigo", "amber", "generic_web_search", "2023-08-15", "PLACEHOLDER-CANARY-7"),
    ("rsq-008", "Under the official placeholder registry, what district number is assigned to fictional Port Aelis?",
     "District 14", "District 99", "official_domain_search", "2025-01-30", "PLACEHOLDER-CANARY-8"),
]

_ALL_SEARCH_TOOLS = ["generic_web_search", "news_search", "official_domain_search", "brave_search"]


def _match_tokens(question: str, answer: str) -> list[str]:
    stop = {"in", "the", "this", "of", "what", "is", "a", "an", "to", "and", "for",
            "who", "which", "was", "founded", "fictional", "placeholder", "synthetic",
            "record", "per", "set", "under", "from", "color", "code", "number"}
    toks = [t for t in tokenize(question) if t not in stop and len(t) > 3]
    # keep the two most distinctive (longest) tokens for stable matching
    toks = sorted(set(toks), key=lambda t: (-len(t), t))[:2]
    return toks or tokenize(question)[:2]


def generate_real_shaped_fixtures() -> dict[str, Any]:
    """Return dict with browsecomp_csv, livebrowsecomp_jsonl, corpus, items."""
    csv_lines = ["problem,answer,canary"]
    jsonl_rows: list[dict[str, Any]] = []
    docs: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []

    for (iid, q, ans, distractor, gold_tool, released, canary) in _PLACEHOLDERS:
        # BrowseComp-shaped obfuscated CSV row (problem + answer XOR'd by canary).
        enc_q = encrypt(q, canary)
        enc_a = encrypt(ans, canary)
        csv_lines.append(f'"{enc_q}","{enc_a}","{canary}"')

        # LiveBrowseComp-shaped JSONL row (plaintext, with a release date).
        jsonl_rows.append({"id": iid, "question": q, "answer": ans,
                           "released_at": released, "task": "placeholder"})

        # Items carry NO harness gold_tool hints in meta — mimicking real data,
        # where the adapter has only question/answer/date.
        items.append({"id": iid, "question": q, "answer": ans, "answer_aliases": [],
                      "released_at": released, "source": "real_shaped_placeholder",
                      "meta": {"topic": "placeholder"}})

        mtok = _match_tokens(q, ans)
        docs.append({
            "doc_id": f"d_{iid}_gold", "item_id": iid,
            "title": f"{iid} primary source", "url": f"https://placeholder.example/{iid}",
            "snippet": f"Placeholder record states: {ans}.", "published_at": released,
            "source_authority": 0.9, "asserts": ans, "answer_on_fetch_only": False,
            "tools": [gold_tool], "match_tokens": mtok, "requires_tokens": [], "base_rank": 0,
        })
        for t in _ALL_SEARCH_TOOLS:
            if t == gold_tool:
                continue
            docs.append({
                "doc_id": f"d_{iid}_{t}", "item_id": iid,
                "title": f"{iid} secondary ({t})", "url": f"https://blog.example-{t}.test/{iid}",
                "snippet": f"Unverified placeholder claim: {distractor} {t}.",
                "published_at": "2023-01-01", "source_authority": 0.35,
                "asserts": f"{distractor} {t}", "answer_on_fetch_only": False,
                "is_distractor": True, "tools": [t], "match_tokens": mtok,
                "requires_tokens": [], "base_rank": 1,
            })

    return {
        "browsecomp_csv": "\n".join(csv_lines) + "\n",
        "livebrowsecomp_jsonl": "\n".join(json.dumps(r) for r in jsonl_rows) + "\n",
        "corpus": {"documents": docs},
        "items": items,
    }


def write_real_shaped_fixtures(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gen = generate_real_shaped_fixtures()
    (out / "browsecomp_sample.csv").write_text(gen["browsecomp_csv"], encoding="utf-8")
    (out / "livebrowsecomp_sample.jsonl").write_text(gen["livebrowsecomp_jsonl"], encoding="utf-8")
    (out / "real_shaped_corpus.json").write_text(json.dumps(gen["corpus"], indent=2), encoding="utf-8")
    return out
