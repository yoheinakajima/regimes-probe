"""Deterministic embeddings for tests + optional live OpenAI embeddings.

The :class:`HashEmbedder` is a fixed feature-hashing embedder: pure, no
network, identical output across processes and runs. It is good enough for
nearest-neighbour trace retrieval over a small synthetic corpus and keeps the
whole hot path replayable without keys.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HashEmbedder:
    """Feature-hashing embedder. Deterministic and dependency-free.

    Each token is hashed into ``dim`` buckets (with a sign hash) and the vector
    is L2-normalised. Cosine similarity over these vectors gives a stable,
    replayable notion of "similar questions" for trace retrieval.
    """

    name = "hash_embedder"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _tokens(text):
            h = hashlib.sha1(tok.encode("utf-8")).digest()
            bucket = int.from_bytes(h[:4], "big") % self.dim
            sign = 1.0 if (h[4] & 1) else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class OpenAIEmbedder:
    """Optional live embeddings via OpenAI. Requires ``OPENAI_API_KEY``.

    Not deterministic across model versions, so live runs that use it must
    record the model id for the audit trail. Never used by unit tests.
    """

    name = "openai_embedder"

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self.model = model

    def available(self) -> bool:
        if not os.environ.get("OPENAI_API_KEY"):
            return False
        try:
            import openai  # noqa: F401
        except Exception:
            return False
        return True

    def embed(self, text: str) -> list[float]:
        from openai import OpenAI

        client = OpenAI()
        resp = client.embeddings.create(model=self.model, input=text)
        return list(resp.data[0].embedding)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Inputs are assumed finite; returns 0 for zero norms."""
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
