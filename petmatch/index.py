"""A tiny in-memory vector index (cosine similarity).

For the benchmark this is all we need, and it avoids running Postgres on day 1.
It exposes the same idea pgvector will later: add vectors with ids, search top-k.
Swap it for a pgvector-backed class when the API is built (Days 7-8).
"""
from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


class VectorIndex:
    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None
        self._ids: list[str] = []

    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        if len(vectors) != len(ids):
            raise ValueError("vectors and ids must have the same length")
        v = l2_normalize(np.asarray(vectors, dtype=np.float32))
        self._vectors = v if self._vectors is None else np.vstack([self._vectors, v])
        self._ids.extend(ids)

    def __len__(self) -> int:
        return len(self._ids)

    def search(self, queries: np.ndarray, k: int) -> tuple[list[list[str]], np.ndarray]:
        """Return ids and cosine scores of the k most similar items for each query."""
        if self._vectors is None:
            raise RuntimeError("index is empty")
        q = l2_normalize(np.asarray(queries, dtype=np.float32))
        sims = q @ self._vectors.T
        k = min(k, len(self._ids))
        top = np.argsort(-sims, axis=1)[:, :k]
        ids = [[self._ids[j] for j in row] for row in top]
        scores = np.take_along_axis(sims, top, axis=1)
        return ids, scores
