"""Recording / replay cache for live provider + answerer calls.

Every live call is keyed by a hash of its *sanitized* request. Modes:

  * ``off``    — no caching (always call; only valid under --execute).
  * ``auto``   — hit → return cached (no call); miss → call + store (default;
                 prevents re-spend on rerun).
  * ``replay`` — hit → return cached; miss → raise (NEVER calls a provider).
  * ``record`` — always call + store (refresh the cache).

A cache entry records: request hash, provider name, tool/model name, sanitized
request metadata, response hash, timestamp, and the structured response payload
(needed to replay). The raw provider payload is stored only when ``store_raw`` is
set. Secrets are never stored — :func:`sanitize` strips key-like fields, and the
request metadata passed in already contains no credentials.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

_SECRET_RE_KEYS = ("key", "token", "secret", "authorization", "auth", "password", "bearer")


def sanitize(obj: Any) -> Any:
    """Recursively drop key-like fields so no secret can enter the cache/log."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if any(s in str(k).lower() for s in _SECRET_RE_KEYS):
                out[k] = "<redacted>"
            else:
                out[k] = sanitize(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj


def _hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


class ReplayMiss(RuntimeError):
    """Raised in replay mode when a request is not in the cache (would re-spend)."""


class RecordingCache:
    def __init__(self, path: str | Path | None = None, *, mode: str = "off",
                 store_raw: bool = False) -> None:
        assert mode in ("off", "auto", "replay", "record"), mode
        self.path = Path(path) if path else None
        self.mode = mode
        self.store_raw = store_raw
        self.entries: dict[str, dict[str, Any]] = {}
        self.calls = 0          # live provider/answerer calls actually made
        self.hits = 0           # cache hits (no spend)
        self.stores = 0
        if self.path and self.path.exists():
            self.load()

    # ---- persistence ----
    def load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.entries = {e["request_hash"]: e for e in data.get("entries", [])}

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"mode": self.mode, "store_raw": self.store_raw,
                   "entries": list(self.entries.values())}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ---- keys ----
    def request_hash(self, provider: str, name: str, request_meta: dict[str, Any]) -> str:
        return _hash({"provider": provider, "name": name,
                      "request": sanitize(request_meta)})

    def get(self, request_hash: str) -> Optional[dict[str, Any]]:
        return self.entries.get(request_hash)

    def store(self, request_hash: str, *, provider: str, name: str,
              request_meta: dict[str, Any], response_payload: dict[str, Any],
              raw: Any = None) -> None:
        entry = {
            "request_hash": request_hash,
            "provider": provider,
            "name": name,
            "request_meta": sanitize(request_meta),
            "response_hash": _hash(response_payload),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "response": response_payload,
        }
        if self.store_raw and raw is not None:
            entry["raw"] = sanitize(raw)
        self.entries[request_hash] = entry
        self.stores += 1
        self.save()

    def summary(self) -> dict[str, Any]:
        return {"mode": self.mode, "n_entries": len(self.entries),
                "live_calls": self.calls, "cache_hits": self.hits,
                "stores": self.stores, "path": str(self.path) if self.path else None}
