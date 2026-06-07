"""ActiveGraph pack for regimes-probe.

Maps the benchmark onto ActiveGraph: object types (:mod:`objects`), event types
(:mod:`events`), relation types (:mod:`relations`), recording behaviors and the
event log (:mod:`behaviors`), and the recorded-tool boundary (:mod:`tools`).

The event log is the source of truth; the graph is a deterministic projection;
behaviors are deterministic; the only I/O in a recorded run is the provider call
inside the recording tool invoker. See ``docs/ACTIVEGRAPH_DESIGN.md``.
"""

from __future__ import annotations

from regimes_probe.activegraph_pack.behaviors import (
    EventLog,
    ReplayReport,
    record_attempt,
    record_grade_and_reward,
    record_policy_update,
    record_run_start,
    replay_check,
)
from regimes_probe.activegraph_pack.events import ALL_EVENTS, Events
from regimes_probe.activegraph_pack.objects import ALL_OBJECTS, Objects
from regimes_probe.activegraph_pack.relations import ALL_RELATIONS, Relations
from regimes_probe.activegraph_pack.tools import RecordingInvoker, ReplayInvoker

__all__ = [
    "EventLog", "ReplayReport", "replay_check",
    "record_run_start", "record_attempt", "record_grade_and_reward", "record_policy_update",
    "ALL_EVENTS", "Events", "ALL_OBJECTS", "Objects", "ALL_RELATIONS", "Relations",
    "RecordingInvoker", "ReplayInvoker",
]
