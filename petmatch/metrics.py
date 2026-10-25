"""Retrieval metrics for "did the right pet show up near the top?"."""
from __future__ import annotations

import numpy as np


def first_hit_ranks(ranked_ids: list[list[str]], true_ids: list[str]) -> np.ndarray:
    """1-based rank of the first correct pet per query; 0 if it never appears.

    A pet usually has several gallery photos, so we de-duplicate by pet before
    ranking: the user sees a list of *pets*, not of photos.
    """
    ranks = []
    for ids, truth in zip(ranked_ids, true_ids):
        seen, rank = set(), 0
        for pid in ids:
            if pid in seen:
                continue
            seen.add(pid)
            if pid == truth:
                rank = len(seen)
                break
        ranks.append(rank)
    return np.array(ranks, dtype=int)


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean((ranks > 0) & (ranks <= k))) if len(ranks) else 0.0


def mean_reciprocal_rank(ranks: np.ndarray) -> float:
    if not len(ranks):
        return 0.0
    rr = np.where(ranks > 0, 1.0 / np.maximum(ranks, 1), 0.0)
    return float(rr.mean())
