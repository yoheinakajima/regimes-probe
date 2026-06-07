"""LiveBrowseComp adapter (primary benchmark).

LiveBrowseComp targets *recent* facts, reducing intrinsic-knowledge dependence,
which is why it is the primary target. Data is loaded from the Hugging Face
dataset ``Forival/LiveBrowseComp`` when the ``datasets`` package and network are
available; otherwise a clear :class:`DatasetUnavailable` is raised. Unit tests
never require it.

Refs:
  https://arxiv.org/abs/2605.28721
  https://huggingface.co/datasets/Forival/LiveBrowseComp
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from regimes_probe.datasets.base import DatasetAdapter, DatasetUnavailable, Item


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
            return self._load_local(self.local_jsonl)
        return self._load_hf()

    # -- local JSONL (offline cache) --
    def _load_local(self, path: Path) -> list[Item]:
        if not path.exists():
            raise DatasetUnavailable(f"LiveBrowseComp local cache not found: {path}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self._loaded_version = (
            f"livebrowsecomp@local:{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}"
        )
        return [self._row_to_item(r, i) for i, r in enumerate(rows)]

    # -- Hugging Face --
    def _load_hf(self) -> list[Item]:
        try:
            from datasets import load_dataset
        except Exception as exc:  # pragma: no cover - depends on optional dep
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
        return [self._row_to_item(dict(r), i) for i, r in enumerate(ds)]

    def _row_to_item(self, r: dict[str, Any], i: int) -> Item:
        question = r.get("question") or r.get("problem") or r.get("query") or ""
        answer = r.get("answer") or r.get("final_answer") or ""
        # Some releases obfuscate the answer with a canary; decode if asked.
        if self.canary and answer:
            from regimes_probe.datasets.browsecomp import decrypt

            try:
                answer = decrypt(answer, self.canary)
            except Exception:
                pass  # leave as-is; grader will simply not match
        return Item(
            id=r.get("id") or f"livebrowsecomp-{i:04d}",
            question=question,
            answer=answer,
            released_at=r.get("released_at") or r.get("date"),
            source="livebrowsecomp",
            meta={k: v for k, v in r.items() if k not in ("question", "answer", "problem")},
        )

    def version(self) -> str:
        return self._loaded_version
