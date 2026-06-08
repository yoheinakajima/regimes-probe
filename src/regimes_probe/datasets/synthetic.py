"""Synthetic browse dataset adapter (Study 0 — harness validity).

Loads ``fixtures/synthetic_browse.json`` — a small, fully deterministic set of
browse-style questions engineered so that tool choice and query/stop choices
change the outcome. No network, no keys.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from regimes_probe.datasets.base import DatasetAdapter, Item

_DEFAULT = Path(__file__).resolve().parents[3] / "fixtures" / "synthetic_browse.json"


class SyntheticBrowseAdapter(DatasetAdapter):
    name = "synthetic_browse"

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else _DEFAULT

    def load(self) -> list[Item]:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        rows = data["items"] if isinstance(data, dict) else data
        return [
            Item(
                id=r["id"],
                question=r["question"],
                answer=r.get("answer", ""),
                answer_aliases=r.get("answer_aliases", []),
                released_at=r.get("released_at"),
                source=r.get("source", "synthetic"),
                meta=r.get("meta", {}),
            )
            for r in rows
        ]

    def version(self) -> str:
        raw = self.path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()[:12]
        return f"synthetic_browse@{digest}"


# ---------------------------------------------------------------------------
# Deterministic fixture generator.
#
# Produces items across latent *epistemic regimes* (freshness, authority,
# generic, multihop). For each item exactly one (tool, query-arm) combination
# surfaces the gold answer; every other search tool returns a confident
# *distractor* asserting a wrong answer. So both routing (Level 1) and query
# formulation / stopping (Level 2) materially change correctness — which is the
# learnable signal the contextual bandit exploits. Nothing here is random.
# ---------------------------------------------------------------------------

# (regime, question template, gold answer, distractor answer, gold tool,
#  gold query arm, requires token, known domain, freshness/multihop flags)
_REGIMES = [
    # freshness: "currently/latest" (no literal year) → news_search + freshness_terms
    ("freshness", "Who is currently the CEO of {e}?", "{a}", "{d}",
     "news_search", "freshness_terms", "2026", None, True, False),
    ("freshness", "What is the latest flagship product released by {e}?", "{a}", "{d}",
     "news_search", "freshness_terms", "2026", None, True, False),
    # authority: official/regulatory → official_domain_search + source_constrained
    ("authority", "What is the official registered headquarters address of {e}?", "{a}", "{d}",
     "official_domain_search", "source_constrained", "official", "{dom}", False, False),
    ("authority", "Under official regulation, what license number governs {e}?", "{a}", "{d}",
     "official_domain_search", "source_constrained", "official", "{dom}", False, False),
    # generic: plain fact → generic_web_search + direct_question
    ("generic", "In which city was the company {e} founded?", "{a}", "{d}",
     "generic_web_search", "direct_question", None, None, False, False),
    ("generic", "Which university is most associated with the researcher {e}?", "{a}", "{d}",
     "generic_web_search", "direct_question", None, None, False, False),
    # multihop: answer hidden in the page → generic_web_search + keyword_compressed + fetch
    ("multihop", "What is the middle name of the founder of {e}?", "{a}", "{d}",
     "generic_web_search", "keyword_compressed", None, None, False, True),
]

_ENTITIES = [
    ("Acme Robotics", "acme", "robotics", "example-acme.gov"),
    ("Borealis Foods", "borealis", "foods", "example-borealis.gov"),
    ("Cobalt Systems", "cobalt", "systems", "example-cobalt.gov"),
    ("Delphi Mining", "delphi", "mining", "example-delphi.gov"),
    ("Everest Labs", "everest", "labs", "example-everest.gov"),
    ("Fairwind Energy", "fairwind", "energy", "example-fairwind.gov"),
    ("Granite Capital", "granite", "capital", "example-granite.gov"),
    ("Helios Pharma", "helios", "pharma", "example-helios.gov"),
]

_ANSWERS = {
    "freshness": ("Mara Olsen", "Jon Pike"),
    "authority": ("HQ-4821 Birch Lane", "HQ-0000 Old Mill"),
    "generic": ("Trenton", "Oldcastle"),
    "multihop": ("Ekaterina", "Bartholomew"),
}


def generate_synthetic_fixtures() -> tuple[dict, dict]:
    """Return (items_payload, corpus_payload) — deterministic, fully offline."""
    items: list[dict] = []
    docs: list[dict] = []
    all_search_tools = ["generic_web_search", "news_search", "official_domain_search", "brave_search"]

    idx = 0
    for ri, (regime, qt, at, dt, gold_tool, gold_arm, req, dom, fresh, multihop) in enumerate(_REGIMES):
        for ei, (ename, tok1, tok2, domain) in enumerate(_ENTITIES):
            idx += 1
            item_id = f"syn-{idx:03d}"
            gold_ans, distractor_ans = _ANSWERS[regime]
            question = qt.format(e=ename)
            known_domain = (dom.format(dom=domain) if dom else None)
            match_tokens = [tok1, tok2]
            # fresh items: gold published recent, stale distractor old.
            gold_pub = "2026-05-10" if fresh else "2024-02-01"
            stale_pub = "2022-03-01"
            requires = [req] if req else []

            # gold doc — only on the gold tool, behind the gold query arm.
            docs.append({
                "doc_id": f"d_{item_id}_gold",
                "item_id": item_id,
                "title": f"{ename}: primary source",
                "url": f"https://{(known_domain or 'www.example.com')}/{tok1}",
                "snippet": f"{ename} — verified record: {gold_ans}.",
                "published_at": gold_pub,
                "source_authority": 0.92 if regime == "authority" else 0.7,
                "asserts": gold_ans,
                "answer_on_fetch_only": multihop,
                "tools": [gold_tool],
                "match_tokens": match_tokens,
                "requires_tokens": requires,
                "base_rank": 0,
            })
            # distractor docs — on every other search tool, confident wrong answer.
            for t in all_search_tools:
                if t == gold_tool:
                    continue
                docs.append({
                    "doc_id": f"d_{item_id}_{t}",
                    "item_id": item_id,
                    "title": f"{ename}: secondary mention ({t})",
                    "url": f"https://blog.example-{t}.com/{tok1}",
                    "snippet": f"Some sources claim {ename} is linked to {distractor_ans} {t}.",
                    "published_at": stale_pub if fresh else "2023-06-01",
                    "source_authority": 0.4,
                    # Distinct per tool so distractors split the vote rather than
                    # ganging up; a single retrieved gold doc (higher authority)
                    # remains the answerer's argmax.
                    "asserts": f"{distractor_ans} {t}",
                    "answer_on_fetch_only": False,
                    "is_distractor": True,
                    "tools": [t],
                    "match_tokens": match_tokens,
                    "requires_tokens": [],
                    "base_rank": 1,
                })

            items.append({
                "id": item_id,
                "question": question,
                "answer": gold_ans,
                "answer_aliases": [],
                "released_at": gold_pub,
                "source": "synthetic",
                "meta": {
                    "regime": regime,
                    "gold_tool": gold_tool,
                    "gold_query_arm": gold_arm,
                    "known_domain": known_domain,
                    "freshness_sensitive": fresh,
                    "answer_on_fetch_only": multihop,
                    # a few generic facts are "intrinsically known" by the
                    # closed-book model (simulated): the first 2 generic items.
                    "intrinsic_knowable": (regime == "generic" and ei < 2),
                },
            })

    # ---- hard items that deliberately trigger generic failure regimes, so the
    # regime diagnostics are meaningful and the synthetic result is NOT perfect.
    _append_hard_items(items, docs, idx)
    return {"items": items}, {"documents": docs}


def _append_hard_items(items: list[dict], docs: list[dict], start_idx: int) -> None:
    i = start_idx

    def add_item(regime, q, ans, **meta):
        nonlocal i
        i += 1
        iid = f"syn-{i:03d}"
        meta.setdefault("regime", regime)
        items.append({"id": iid, "question": q, "answer": ans, "answer_aliases": [],
                      "released_at": meta.get("released_at", "2025-01-01"),
                      "source": "synthetic", "meta": meta})
        return iid

    # (A) obscure / evidence-sparse: NO gold doc anywhere — only a single
    # low-authority distractor on the alphabetically-first tool, so the agent
    # answers it (wrong, weak support) -> route_miss / query_miss / evidence_sparse
    # / verification_miss.
    for name, tok, wrong in [("Quillon Reach", "quillon", "Mistgrave"),
                             ("Vexley Hollow", "vexley", "Dunmoor"),
                             ("Thornfield Span", "thornfield", "Ashby")]:
        iid = add_item("obscure", f"What is the obscure charter seat of the placeholder enclave {name}?",
                       f"Unknown-{tok}", freshness_sensitive=False, answer_on_fetch_only=False)
        docs.append({"doc_id": f"d_{iid}_brave", "item_id": iid, "title": f"{name} rumor",
                     "url": f"https://blog.example-brave.test/{tok}",
                     "snippet": f"Rumor links {name} to {wrong}.", "published_at": "2023-01-01",
                     "source_authority": 0.3, "asserts": wrong, "answer_on_fetch_only": False,
                     "is_distractor": True, "tools": ["brave_search"], "match_tokens": [tok, "enclave"],
                     "requires_tokens": [], "base_rank": 0})

    # (B) stale-gold freshness: the only supporting doc is the gold answer but it
    # is OLD on a freshness-sensitive question -> stale_evidence (correct but stale).
    for name, tok, ans in [("Halcyon Drift", "halcyon", "Captain Reyes"),
                           ("Nimbus Lattice", "nimbus", "Captain Orr")]:
        iid = add_item("stale_fresh", f"Who is currently the lead pilot of the placeholder fleet {name}?",
                       ans, freshness_sensitive=True, answer_on_fetch_only=False)
        docs.append({"doc_id": f"d_{iid}_gold", "item_id": iid, "title": f"{name} archive",
                     "url": f"https://news.example.com/{tok}", "snippet": f"Archive: {ans}.",
                     "published_at": "2021-05-01", "source_authority": 0.7, "asserts": ans,
                     "answer_on_fetch_only": False, "tools": ["news_search", "generic_web_search"],
                     "match_tokens": [tok, "pilot"], "requires_tokens": [], "base_rank": 0})

    # (C) high-authority distractor: a 0.95-authority WRONG source on news plus a
    # 0.7 gold on generic. At budget>=2 the agent gathers both, the high-auth
    # distractor wins the vote -> contradiction_unresolved (wrong + conflict).
    for name, tok, ans, wrong in [("Ferrous Pact", "ferrous", "Article 9", "Article 1"),
                                  ("Verdant Accord", "verdant", "Clause 7", "Clause 99")]:
        iid = add_item("authority_trap", f"Which placeholder clause governs the treaty body {name}?",
                       ans, freshness_sensitive=False, answer_on_fetch_only=False)
        docs.append({"doc_id": f"d_{iid}_gold", "item_id": iid, "title": f"{name} text",
                     "url": f"https://www.example.com/{tok}", "snippet": f"Text: {ans}.",
                     "published_at": "2024-01-01", "source_authority": 0.7, "asserts": ans,
                     "answer_on_fetch_only": False, "tools": ["generic_web_search"],
                     "match_tokens": [tok, "clause"], "requires_tokens": [], "base_rank": 0})
        docs.append({"doc_id": f"d_{iid}_news", "item_id": iid, "title": f"{name} bulletin",
                     "url": f"https://news.example.com/{tok}", "snippet": f"Bulletin asserts {wrong}.",
                     "published_at": "2024-02-01", "source_authority": 0.95, "asserts": wrong,
                     "answer_on_fetch_only": False, "is_distractor": True, "tools": ["news_search"],
                     "match_tokens": [tok, "clause"], "requires_tokens": [], "base_rank": 0})
