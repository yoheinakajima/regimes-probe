"""Answer grading.

Default grader: normalized exact/alias match (lowercase, strip articles and
punctuation). An optional LLM judge can be plugged in for free-form answers; it
must be logged and replayable (cached by prompt hash) — never used by unit
tests. See ``docs/GRADING_AND_REWARD.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from regimes_probe.datasets.base import Item

_ARTICLES = {"a", "an", "the"}
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    """Lowercase, drop punctuation and leading articles, collapse whitespace."""
    t = _PUNCT.sub(" ", (text or "").lower())
    toks = [w for w in _WS.sub(" ", t).strip().split(" ") if w]
    while toks and toks[0] in _ARTICLES:
        toks = toks[1:]
    return " ".join(toks)


@dataclass
class GradeResult:
    item_id: str
    correct: bool
    abstained: bool
    method: str
    prediction_norm: str
    gold_norm: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "correct": self.correct,
            "abstained": self.abstained,
            "method": self.method,
            # gold_norm retained here for audit of the *grade*, not stored in
            # policy memory.
            "gold_norm": self.gold_norm,
        }


def grade(
    item: Item,
    prediction: Optional[str],
    *,
    judge: Optional[Callable[[Item, str], bool]] = None,
) -> GradeResult:
    """Grade a prediction against an item's gold answer(s)."""
    golds = [normalize_answer(g) for g in item.gold_answers() if g]
    if prediction is None or not prediction.strip():
        return GradeResult(item.id, False, True, "abstain", "", golds)
    pred = normalize_answer(prediction)
    # Normalized exact / alias / containment match.
    correct = any(pred == g or (g and (g in pred or pred in g)) for g in golds)
    method = "normalized_match"
    if not correct and judge is not None:
        correct = bool(judge(item, prediction))
        method = "llm_judge"
    return GradeResult(item.id, correct, False, method, pred, golds)
