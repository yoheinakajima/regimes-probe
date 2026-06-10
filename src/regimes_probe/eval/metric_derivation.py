"""Level 5j-D: event-derived metric derivation + parity check.

Business logic emits structured events; several mechanism metrics should therefore be
*projections* of the event log, not only inline mutable counters. This module recomputes the
event-backed mechanism metrics from a frontier's event log and verifies they match the inline
counters surfaced in ``frontier.metrics()``. A test runs the parity check; a mismatch is a bug
(an inline counter drifting from the events it claims to summarise).

Not every counter is event-derivable today (some summarise interpreter-internal triage that is
not individually event-emitted); those are documented in ``INLINE_ONLY`` with the reason, and a
parity check is added wherever an event stream exists.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

#: metric_name -> (event_type, predicate over the event) for event-derived metrics.
_RESOLVED = {"full_support", "partial_support", "contradiction"}


def derive_metrics_from_events(events: list[dict]) -> dict[str, int]:
    """Recompute event-backed mechanism metrics purely from the event log (5j-D)."""
    et = Counter(e.get("event_type") for e in events)

    def _data(e):
        return e.get("data", {}) or {}

    read_resolved = sum(1 for e in events
                        if e.get("event_type") == "read_judgment_resolved"
                        and _data(e).get("resolution") in _RESOLVED)
    read_still_open = sum(1 for e in events
                          if e.get("event_type") == "read_judgment_still_unresolved")
    read_resolved_pending = sum(1 for e in events
                                if e.get("event_type") == "read_interpreted"
                                and _data(e).get("read_resolved_pending_judgment"))
    read_added_candidates = sum(1 for e in events
                                if e.get("event_type") == "read_interpreted"
                                and _data(e).get("read_added_candidates"))
    read_added_support = sum(1 for e in events
                             if e.get("event_type") == "read_interpreted"
                             and _data(e).get("read_added_constraint_support"))
    return {
        "requires_read_resolved_by_read_count": read_resolved,
        "requires_read_unresolved_after_successful_read_count": read_still_open,
        "read_resolved_pending_judgment_count": read_resolved_pending,
        "read_added_candidates_count": read_added_candidates,
        "read_added_constraint_support_count": read_added_support,
        "read_required_by_judge_count": et.get("read_required_by_judge", 0),
        "read_judged_after_read_count": et.get("read_judged_after_read", 0),
        "read_desired_count": et.get("read_desired", 0),
        "read_selected_count": et.get("read_selected", 0),
        "read_blocked_no_url_count": et.get("read_blocked_no_url", 0),
        "bind_target_answer_slot_selected_count": et.get("bind_target_answer_slot_selected", 0),
        "source_subject_extracted_count": et.get("source_subject_extracted", 0),
        "explicit_location_mismatch_rejected_count":
            et.get("explicit_location_mismatch_rejected", 0),
    }


#: frontier metric name -> derived metric name (the event projection that should equal it).
_FRONTIER_PARITY = {
    "requires_read_resolved_by_read_count": "requires_read_resolved_by_read_count",
    "requires_read_unresolved_after_successful_read_count":
        "requires_read_unresolved_after_successful_read_count",
    "read_resolved_pending_judgment_count": "read_resolved_pending_judgment_count",
    "read_added_candidates_count": "read_added_candidates_count",
    "read_added_constraint_support_count": "read_added_constraint_support_count",
    "read_desired_count": "read_desired_count",
    "read_selected_count": "read_selected_count",
    "read_blocked_no_url_count": "read_blocked_no_url_count",
    "bind_target_answer_slot_selected_count": "bind_target_answer_slot_selected_count",
}

#: interpreter stat name -> derived metric name.
_INTERP_PARITY = {
    "source_subject_extracted_count": "source_subject_extracted_count",
    "explicit_location_mismatch_rejected_count": "explicit_location_mismatch_rejected_count",
}

#: counters that legitimately remain inline (no per-instance event), with the reason.
INLINE_ONLY = {
    "judge_calls_saved_by_prejudge_triage":
        "summarises constraints NOT judged (a non-event); parity is via the C fixture proxy",
    "read_loop_open_count":
        "derived from a read with chars but no evidence added AND pending obligations; the "
        "read_interpreted event carries the inputs but the join is computed inline",
}


def verify_metric_derivation(frontier, *, interpreter_stats: dict | None = None) -> dict[str, Any]:
    """Recompute event-backed metrics and compare to the inline counters (5j-D).

    Returns ``{ok, mismatches, derived, checked}``. ``ok`` is True when every event-derivable
    metric matches its inline counter."""
    derived = derive_metrics_from_events(list(frontier.events))
    fm = frontier.metrics()
    mismatches = []
    for inline_name, derived_name in _FRONTIER_PARITY.items():
        if inline_name in fm and derived_name in derived and fm[inline_name] != derived[derived_name]:
            mismatches.append({"metric": inline_name, "inline": fm[inline_name],
                               "event_derived": derived[derived_name], "source": "frontier"})
    istats = interpreter_stats
    if istats is None and getattr(frontier, "interpreter", None) is not None:
        istats = frontier.interpreter.stats()
    if istats:
        for inline_name, derived_name in _INTERP_PARITY.items():
            if inline_name in istats and derived[derived_name] != istats[inline_name]:
                mismatches.append({"metric": inline_name, "inline": istats[inline_name],
                                   "event_derived": derived[derived_name],
                                   "source": "interpreter"})
    return {"ok": not mismatches, "mismatches": mismatches, "derived": derived,
            "checked": list(_FRONTIER_PARITY) + list(_INTERP_PARITY),
            "inline_only": dict(INLINE_ONLY)}
