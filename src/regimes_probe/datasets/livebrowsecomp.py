"""LiveBrowseComp adapter (primary benchmark).

LiveBrowseComp targets *recent* facts, reducing intrinsic-knowledge dependence,
which is why it is the primary target.

**Fail-closed loading (important).** The Hugging Face release
``Forival/LiveBrowseComp`` ships ``problem`` / ``answer`` fields that are
*obfuscated* (encoded, BrowseComp-style) to prevent training contamination — they
are NOT plaintext. This adapter therefore refuses to treat ``problem`` as a
plaintext question. A row is usable only if a plaintext ``question`` (or
``query``) is present, OR a configured ``canary`` decodes ``problem``/``answer``
into plausible plaintext. Otherwise :class:`DatasetUnavailable` is raised, so a
live run never spends API calls on encrypted blobs.

The official decode mechanism could not be verified from this environment
(network restricted). Until it is resolved, run LiveBrowseComp only via a
plaintext export or an explicitly configured + validated decode path. See
``docs/BENCHMARK_TARGETS.md`` and ``docs/NEXT_LIVE_RUN.md``.

Refs:
  https://arxiv.org/abs/2605.28721
  https://huggingface.co/datasets/Forival/LiveBrowseComp
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional

from regimes_probe.datasets.base import DatasetAdapter, DatasetUnavailable, Item

#: Field names that carry PLAINTEXT questions (``problem`` is the obfuscated one).
_PLAINTEXT_Q_FIELDS = ("question", "query")
_PLAINTEXT_A_FIELDS = ("answer", "final_answer")
_OBFUSCATED_Q_FIELDS = ("problem",)

_BASE64ISH = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")


def _is_base64ish(s: str) -> bool:
    s = s.strip()
    return len(s) >= 24 and bool(_BASE64ISH.match(s))


def looks_plaintext(s: Optional[str]) -> bool:
    """Heuristic: does ``s`` look like a natural-language question/answer?

    Plaintext questions contain whitespace and printable characters; obfuscated
    fields are long, space-free, base64-charset blobs. Short single tokens
    (e.g. a one-word answer) are treated as plaintext only when they are not a
    long base64 blob.
    """
    if not s:
        return False
    t = s.strip()
    if not t:
        return False
    printable = sum(ch.isprintable() for ch in t) / len(t)
    if printable < 0.95:
        return False
    if " " in t:
        return True
    return not _is_base64ish(t)


def looks_obfuscated(s: Optional[str]) -> bool:
    """True when ``s`` is non-empty but does not look like plaintext."""
    return bool(s) and not looks_plaintext(s)


class LiveBrowseCompAdapter(DatasetAdapter):
    name = "livebrowsecomp"

    def __init__(
        self,
        *,
        hf_name: str = "Forival/LiveBrowseComp",
        split: str = "test",
        local_jsonl: str | Path | None = None,
        canary: Optional[str] = None,
    ) -> None:
        self.hf_name = hf_name
        self.split = split
        self.local_jsonl = Path(local_jsonl) if local_jsonl else None
        self.canary = canary
        self._loaded_version = "livebrowsecomp@unloaded"

    def load(self) -> list[Item]:
        if self.local_jsonl is not None:
            rows = self._read_local(self.local_jsonl)
        else:
            rows = self._read_hf()
        return self._rows_to_items(rows)

    # -- local JSONL (offline cache / plaintext export) --
    def _read_local(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise DatasetUnavailable(f"LiveBrowseComp local cache not found: {path}")
        self._loaded_version = (
            f"livebrowsecomp@local:{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}"
        )
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    # -- Hugging Face --
    def _read_hf(self) -> list[dict[str, Any]]:
        try:
            from datasets import load_dataset
        except Exception as exc:  # pragma: no cover - optional dep
            raise DatasetUnavailable(
                "LiveBrowseComp needs the 'datasets' package: pip install "
                "'regimes-probe[datasets]'. Or pass local_jsonl=..."
            ) from exc
        try:
            ds = load_dataset(self.hf_name, split=self.split)
        except Exception as exc:  # pragma: no cover - network/credentials
            raise DatasetUnavailable(
                f"Could not load {self.hf_name}:{self.split} from Hugging Face: {exc}"
            ) from exc
        self._loaded_version = f"livebrowsecomp@hf:{self.hf_name}:{self.split}"
        return [dict(r) for r in ds]

    # -- fail-closed row decoding --
    def _try_decode(self, value: str) -> Optional[str]:
        from regimes_probe.datasets.browsecomp import decrypt
        try:
            return decrypt(value, self.canary)
        except Exception:
            return None

    def _decode_row(self, r: dict[str, Any]) -> tuple[Optional[str], str, Optional[str]]:
        """Return (question, answer, reason). question is None iff row is unusable.

        Two schemas:
          * plaintext: a ``question``/``query`` field that actually looks like
            plaintext (answers expected plaintext; an encoded answer is decoded
            only if a canary is configured, else fail closed).
          * obfuscated: only ``problem`` is present — require a configured canary
            and decode BOTH ``problem`` and ``answer`` with it (the heuristic is
            not trusted for short encrypted answers).
        """
        plain_q = next((r[f] for f in _PLAINTEXT_Q_FIELDS if r.get(f)), None)
        a_raw = next((r[f] for f in _PLAINTEXT_A_FIELDS if r.get(f)), "")

        if plain_q is not None:
            if not looks_plaintext(plain_q):
                return None, "", "'question' field present but not plaintext (looks encoded)"
            if a_raw and looks_obfuscated(a_raw):
                dec = self._try_decode(a_raw) if self.canary else None
                if not (dec and looks_plaintext(dec)):
                    return None, "", "answer present but encoded; cannot grade (no valid decode)"
                return plain_q.strip(), dec, None
            return plain_q.strip(), (str(a_raw).strip() if a_raw else ""), None

        problem = next((r[f] for f in _OBFUSCATED_Q_FIELDS if r.get(f)), None)
        if problem is not None:
            if not self.canary:
                return None, "", ("obfuscated 'problem' with no plaintext 'question' "
                                  "and no canary configured")
            qd = self._try_decode(problem)
            if not (qd and looks_plaintext(qd)):
                return None, "", "'problem' did not decode to plaintext with the given canary"
            ad = ""
            if a_raw:
                dec = self._try_decode(a_raw)
                if not (dec and looks_plaintext(dec)):
                    return None, "", "'answer' did not decode to plaintext with the given canary"
                ad = dec
            return qd.strip(), ad, None

        return None, "", "no question/query/problem field"

    def _rows_to_items(self, rows: list[dict[str, Any]]) -> list[Item]:
        items: list[Item] = []
        reasons: list[str] = []
        for i, r in enumerate(rows):
            q, a, reason = self._decode_row(r)
            if q is None:
                if len(reasons) < 5:
                    reasons.append(f"row {i}: {reason}")
                continue
            items.append(Item(
                id=r.get("id") or f"livebrowsecomp-{i:04d}",
                question=q, answer=a,
                released_at=r.get("released_at") or r.get("date"),
                source="livebrowsecomp",
                meta={k: v for k, v in r.items()
                      if k not in ("question", "query", "problem", "answer", "final_answer")},
            ))
        if reasons or not items:
            n_bad = len(rows) - len(items)
            raise DatasetUnavailable(
                f"LiveBrowseComp: {n_bad}/{len(rows)} rows are obfuscated/undecodable and "
                f"were refused (fail-closed). Examples: {reasons}. "
                "LiveBrowseComp's HF 'problem'/'answer' fields are encoded; this adapter "
                "will NOT pass encrypted text through as a question. Provide a PLAINTEXT "
                "export (with 'question'/'answer' fields) or configure a validated decode "
                "path (canary=...). See docs/BENCHMARK_TARGETS.md."
            )
        return items

    def version(self) -> str:
        return self._loaded_version
