"""Live execution layer (opt-in, env-gated).

Nothing here is imported by the no-key path or the test suite's offline parts.
Live providers and the live answerer are constructed only by ``scripts/run_live.py``
when ``--execute`` is passed, and every call flows through a
:class:`~regimes_probe.live.cache.RecordingCache` so reruns/replay never re-spend.

Credentials come from environment variables only and are never written into
manifests, configs, logs, or artifacts.
"""

from __future__ import annotations

from regimes_probe.live.cache import RecordingCache, ReplayMiss, sanitize

__all__ = ["RecordingCache", "ReplayMiss", "sanitize"]
