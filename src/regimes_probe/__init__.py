"""regimes-probe — an ActiveGraph-native benchmark and improvement loop for
learning epistemic *search policy* from traces.

The package learns a procedural policy for *where to look, how to search, when
to verify, and when to stop* — not factual answers. See ``docs/ARCHITECTURE.md``.

Design rules (enforced throughout):
  * The event log is the source of truth; the graph is a deterministic
    projection of the log (ActiveGraph).
  * Behavior bodies are deterministic: no ``random``, ``datetime.now``,
    ``uuid.uuid4`` or direct network I/O inside behavior bodies.
  * All network calls go through ActiveGraph tools (``activegraph_pack.tools``).
  * Policy memory stores traces, rewards, priors and policy fragments — never
    final answer text (see ``docs/LEAKAGE_CONTROLS.md``).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
