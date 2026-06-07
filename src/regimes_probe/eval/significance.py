"""Statistical tests: McNemar for accuracy flips, bootstrap CI for the headline.

Pure Python (no SciPy). Deterministic bootstrap (seeded) so reports are
reproducible. See ``docs/EVALUATION_PROTOCOL.md``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class McNemarResult:
    b: int          # baseline correct, treatment wrong
    c: int          # baseline wrong, treatment correct
    statistic: float
    p_value: float
    n_discordant: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "b_only_baseline_correct": self.b,
            "c_only_treatment_correct": self.c,
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 4),
            "n_discordant": self.n_discordant,
        }


def _norm_sf(z: float) -> float:
    """Survival function of standard normal via erfc."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def mcnemar(
    baseline_correct: list[bool], treatment_correct: list[bool]
) -> McNemarResult:
    """Paired McNemar test on aligned per-question correctness (continuity-corrected)."""
    if len(baseline_correct) != len(treatment_correct):
        raise ValueError("paired inputs must have equal length")
    b = sum(1 for x, y in zip(baseline_correct, treatment_correct) if x and not y)
    c = sum(1 for x, y in zip(baseline_correct, treatment_correct) if not x and y)
    n = b + c
    if n == 0:
        return McNemarResult(b, c, 0.0, 1.0, 0)
    stat = (abs(b - c) - 1) ** 2 / n  # chi-square with continuity correction, df=1
    p = _norm_sf(math.sqrt(stat))  # two-sided via |Z|; sf of sqrt(chi2)
    return McNemarResult(b, c, stat, min(1.0, 2 * p), n)


@dataclass
class BootstrapCI:
    point: float
    lo: float
    hi: float
    level: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "point": round(self.point, 6),
            "ci_lo": round(self.lo, 6),
            "ci_hi": round(self.hi, 6),
            "level": self.level,
        }


def bootstrap_correct_per_tool_call(
    correct: list[int],
    tool_calls: list[int],
    *,
    iters: int = 2000,
    level: float = 0.95,
    seed: int = 12345,
) -> BootstrapCI:
    """Bootstrap CI for sum(correct)/sum(tool_calls) (a ratio metric)."""
    n = len(correct)
    if n == 0 or sum(tool_calls) == 0:
        return BootstrapCI(0.0, 0.0, 0.0, level)
    point = sum(correct) / sum(tool_calls)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(iters):
        sc = st = 0
        for _ in range(n):
            j = rng.randrange(n)
            sc += correct[j]
            st += tool_calls[j]
        estimates.append(sc / st if st else 0.0)
    estimates.sort()
    alpha = (1 - level) / 2
    lo = estimates[int(alpha * iters)]
    hi = estimates[min(iters - 1, int((1 - alpha) * iters))]
    return BootstrapCI(point, lo, hi, level)


def paired_diff_ci(
    a: list[float], b: list[float], *, iters: int = 2000, level: float = 0.95, seed: int = 999
) -> BootstrapCI:
    """Bootstrap CI for the mean paired difference a-b (e.g. per-question reward)."""
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    if n == 0:
        return BootstrapCI(0.0, 0.0, 0.0, level)
    point = sum(diffs) / n
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(iters):
        s = sum(diffs[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    alpha = (1 - level) / 2
    return BootstrapCI(point, means[int(alpha * iters)], means[min(iters - 1, int((1 - alpha) * iters))], level)
