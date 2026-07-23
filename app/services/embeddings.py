"""Embedding abstraction + an offline, deterministic default implementation.

The :class:`Embedder` Protocol is the seam: swap :class:`DeterministicEmbedder`
for a real model (fastembed, sentence-transformers, an API embedder) without
touching the router or vector store.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

_WORD_RE = re.compile(r"[a-z0-9]+")

# Function/question words carry little routing or retrieval signal and, with a
# hashing embedder, mostly add collision noise that blurs relevant vs irrelevant
# similarity. Removing them makes the deterministic embedder behave like a clean
# keyword retriever. (If a text is *all* stopwords we keep the original tokens so
# short greetings like "how are you" still produce a non-zero vector.)
_STOPWORDS = frozenset(
    """
    a an and are as at be been being but by can could did do does for from had has
    have how i if in into is it its me my of on or our so that the their them then
    there they this to up us was we were what when where which who why will with
    would you your am about get got the
    """.split()
)


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into a fixed-dimension vector."""

    dim: int

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]: ...


class DeterministicEmbedder:
    """Hashing bag-of-words (+ bigram) embedder.

    Fully offline and deterministic across processes (uses md5, not the salted
    builtin ``hash``). Vectors are L2-normalized so a dot product equals cosine
    similarity. Good enough for keyword-overlap routing and RAG retrieval in the
    demo; *not* a semantic model — that is the point of the Protocol seam.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("embedding dim must be positive")
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        words = _WORD_RE.findall(text.lower())
        content = [w for w in words if w not in _STOPWORDS]
        return content or words  # fall back to raw tokens if all were stopwords

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % self.dim
        sign = 1.0 if digest[4] & 1 else -1.0
        return index, sign

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = self._tokens(text)
        for tok in tokens:
            idx, sign = self._bucket(tok)
            vec[idx] += sign
        # Bigrams add a little word-order signal, down-weighted.
        for a, b in zip(tokens, tokens[1:]):
            idx, sign = self._bucket(f"{a}_{b}")
            vec[idx] += sign * 0.5

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]
