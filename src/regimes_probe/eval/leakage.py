"""Leakage checks, separated by layer.

The project has three layers (see ``docs/ACTIVEGRAPH_DESIGN.md``):
  1. raw event-trace archive — the audit log; MAY contain answer text by design.
  2. frozen policy memory — answer-free; the thing the hot path reads.
  3. optional natural-language lessons.

The **main memory claim** must depend on layer-2 (frozen policy memory) leakage
ONLY — not on whether the raw benchmark archive happens to contain gold answers.

The authoritative frozen-memory check is the SAME one
``scripts/inspect_memory_snapshot.py`` uses: :func:`assert_no_answer_leakage`
(key-based) over the serialized snapshot. A secondary, robust gold-token scan is
added — but it is deliberately conservative (word-boundary, length≥4, alphabetic)
so a short/common BrowseComp answer (e.g. ``"1"``, ``"a"``) coincidentally
appearing inside a hash or a float does NOT raise a false positive.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from regimes_probe.eval.grader import normalize_answer
from regimes_probe.policy.policy_fragment import assert_no_answer_leakage


def gold_scan_tokens(items) -> set[str]:
    """Normalized gold strings worth scanning for: length≥4 with an alpha char.

    Shorter/numeric answers are too ambiguous to substring-scan reliably; the
    key-based :func:`assert_no_answer_leakage` is the guard for those.
    """
    out: set[str] = set()
    for it in items:
        for g in getattr(it, "gold_answers", lambda: [])():
            n = normalize_answer(g)
            if len(n) >= 4 and any(c.isalpha() for c in n):
                out.add(n)
    return out


def _string_leaves(obj: Any):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _string_leaves(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _string_leaves(v)
    elif isinstance(obj, str):
        yield obj


def scan_gold_in_payload(payload: Any, tokens: set[str]) -> tuple[bool, Optional[str]]:
    """Word-boundary scan for any gold token across the payload's STRING leaves.

    Returns (clean, leaked_token). Numeric/hash leaves rarely create whole-word
    matches, so this avoids the substring false positives of ``g in json.dumps``.
    """
    if not tokens:
        return True, None
    text = " ".join(s.lower() for s in _string_leaves(payload))
    for g in sorted(tokens, key=len, reverse=True):
        if re.search(r"\b" + re.escape(g) + r"\b", text):
            return False, g
    return True, None


def snapshot_leakage(snap_dict: dict[str, Any], items) -> dict[str, Any]:
    """Authoritative frozen-memory leakage check (matches the inspector)."""
    passed_keys, failed_path, message = True, None, ""
    try:
        assert_no_answer_leakage(snap_dict, "policy_memory_snapshot")
    except Exception as exc:                       # forbidden key present
        passed_keys = False
        message = str(exc)[:300]
        failed_path = message
    gold_ok, leaked = scan_gold_in_payload(snap_dict, gold_scan_tokens(items))
    if not gold_ok:
        failed_path = failed_path or "policy_memory_snapshot (gold answer token present)"
        message = message or f"gold answer text appears in frozen policy memory: '{(leaked or '')[:6]}…'"
    return {
        "pass": bool(passed_keys and gold_ok),
        "method": "assert_no_answer_leakage + word-boundary gold scan",
        "failed_path": failed_path,
        "message": message,
    }


def leakage_check_details(snap_dict: dict[str, Any], items) -> dict[str, Any]:
    """Structured, layer-separated leakage details for report.json.

    ``raw_trace_archive`` (layer 1) is the audit log and is permitted to contain
    gold answer text by design, so it does NOT gate the memory claim. Only
    ``memory_snapshot_leakage_pass`` gates eligibility.
    """
    snap = snapshot_leakage(snap_dict or {}, items)
    return {
        "memory_snapshot_leakage_pass": snap["pass"],
        # The committed snapshot file is exactly snap_dict, so this is the same check.
        "report_artifacts_leakage_pass": snap["pass"],
        # Raw audit trace may contain gold by design (layer 1) — NOT a policy-memory leak.
        "raw_trace_archive_leakage_pass": True,
        "raw_trace_archive_may_contain_answers": True,
        "failed_path": snap["failed_path"],
        "message": snap["message"],
        "method": snap["method"],
        "gates_memory_claim": "memory_snapshot_leakage_pass",
    }
