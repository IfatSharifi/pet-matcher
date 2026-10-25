"""Combined ranking of lost reports for a found report (and vice versa).

score = w_img   * image similarity        (DINOv2 cosine, best photo pair)
      + w_color * color agreement          (soft: tan ~ brown counts half)
      + w_dist  * -log(1 + km apart)
      + w_date  * -log(1 + days between lost and found)
Hard rules: different species, or found more than a day BEFORE the pet was lost -> excluded.

Everything is a numpy matrix of shape (found reports, lost reports), so scoring all
pairs for ~1,400 x 1,400 reports takes milliseconds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Colors people (and models) confuse: half credit instead of none.
NEAR_COLORS = {frozenset(p) for p in [
    ("tan", "brown"), ("tan", "golden"), ("golden", "cream"), ("cream", "white"), ("brown", "red"),
    ("red", "orange"), ("golden", "orange"), ("brindle", "brown"), ("merle", "gray"), ("gray", "white"),
    ("gray", "black"), ("spotted", "white"),
]}

HE_COLORS = {"black": "שחור", "white": "לבן", "brown": "חום", "tan": "חום בהיר", "golden": "זהוב",
             "cream": "קרם", "gray": "אפור", "orange": "ג'ינג'י", "red": "אדמדם", "brindle": "מנומר",
             "merle": "מרל", "spotted": "מנוקד"}

EXCLUDED = -1e9


def species_compatible(a: str, b: str) -> bool:
    """Exclude only a clear dog-vs-cat conflict.

    'other' and 'unknown' match anything: a single odd label (Haiku called a wolf-like
    dog 'other') must not knock the true match out of every result list.
    """
    return {a, b} != {"dog", "cat"}


def color_pair(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return 0.5 if frozenset((a, b)) in NEAR_COLORS else 0.0


def color_similarity(a: list[str], b: list[str]) -> float:
    """Symmetric soft overlap in [0, 1]; 0.5 (neutral) when either side has no colors."""
    if not a or not b:
        return 0.5
    fwd = sum(max(color_pair(x, y) for y in b) for x in a)
    back = sum(max(color_pair(y, x) for x in a) for y in b)
    return (fwd + back) / (len(a) + len(b))


def color_matrix(found_colors: list[list[str]], lost_colors: list[list[str]]) -> np.ndarray:
    m = np.empty((len(found_colors), len(lost_colors)), dtype=np.float32)
    cache: dict[tuple, float] = {}
    for i, fc in enumerate(found_colors):
        for j, lc in enumerate(lost_colors):
            key = (tuple(fc), tuple(lc))
            if key not in cache:
                cache[key] = color_similarity(fc, lc)
            m[i, j] = cache[key]
    return m


def haversine_matrix(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=np.float64)) for x in (lat1, lon1, lat2, lon2))
    dlat = lat2[None, :] - lat1[:, None]
    dlon = lon2[None, :] - lon1[:, None]
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1)[:, None] * np.cos(lat2)[None, :] * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


def best_photo_similarity(query_vecs: np.ndarray, photo_vecs: np.ndarray, photo_report: np.ndarray,
                          n_reports: int) -> np.ndarray:
    """(queries, reports): cosine of the best-matching photo in each lost report."""
    q = query_vecs / np.linalg.norm(query_vecs, axis=1, keepdims=True)
    p = photo_vecs / np.linalg.norm(photo_vecs, axis=1, keepdims=True)
    sims = q @ p.T
    out = np.full((len(q), n_reports), -1.0, dtype=np.float32)
    for r in range(n_reports):
        cols = np.flatnonzero(photo_report == r)
        if len(cols):
            out[:, r] = sims[:, cols].max(axis=1)
    return out


@dataclass
class Features:
    image: np.ndarray        # cosine
    color: np.ndarray        # 0..1
    km: np.ndarray           # distance
    days: np.ndarray         # found date - lost date
    allowed: np.ndarray      # bool: passes the hard rules

    def subset(self, rows: np.ndarray) -> "Features":
        return Features(self.image[rows], self.color[rows], self.km[rows], self.days[rows], self.allowed[rows])


@dataclass(frozen=True)
class Weights:
    image: float = 1.0
    color: float = 0.0
    dist: float = 0.0
    date: float = 0.0
    use_rules: bool = False


def score(f: Features, w: Weights) -> np.ndarray:
    s = (w.image * f.image + w.color * f.color
         - w.dist * np.log1p(f.km) - w.date * np.log1p(np.clip(f.days, 0, None)))
    if w.use_rules:
        s = np.where(f.allowed, s, EXCLUDED)
    return s


def true_ranks(scores: np.ndarray, truth_cols: np.ndarray) -> np.ndarray:
    """1-based rank of the true lost report per found report (ties count against us)."""
    true = scores[np.arange(len(scores)), truth_cols]
    return (scores >= true[:, None]).sum(axis=1)  # includes itself -> rank 1 if nothing beats it


def mrr(ranks: np.ndarray) -> float:
    return float(np.mean(1.0 / ranks))


def tune(f: Features, truth: np.ndarray, base: Weights, grid: dict[str, list[float]],
         passes: int = 3) -> Weights:
    """Coordinate search: improve one weight at a time, keep what raises MRR on this data."""
    best = base
    best_mrr = mrr(true_ranks(score(f, best), truth))
    for _ in range(passes):
        improved = False
        for name, values in grid.items():
            for v in values:
                cand = Weights(**{**best.__dict__, name: v})
                m = mrr(true_ranks(score(f, cand), truth))
                if m > best_mrr + 1e-9:
                    best, best_mrr, improved = cand, m, True
        if not improved:
            break
    return best


def reason_he(colors_found: list[str], colors_lost: list[str], km: float, days: int,
              image_sim: float, found_is_query: bool = True) -> str:
    """One Hebrew line explaining a match, e.g. 'אותו צבע: שחור, לבן · 1.8 ק"מ מהמקום שבו אבד · נמצא יומיים אחרי'."""
    parts = []
    same = [c for c in colors_found if c in colors_lost]
    near = [c for c in colors_found if c not in same and any(color_pair(c, x) == 0.5 for x in colors_lost)]
    if same:
        parts.append("אותו צבע: " + ", ".join(HE_COLORS.get(c, c) for c in same))
    elif near:
        parts.append("צבע דומה")
    if km < 0.15:
        parts.append("פחות מ-100 מטר מהמקום שבו אבד")
    elif km < 0.95:
        parts.append(f"{int(round(km * 1000, -2))} מטר מהמקום שבו אבד")
    else:
        parts.append(f"{km:.1f} ק\"מ מהמקום שבו אבד")
    if days <= 0:
        parts.append("נמצא באותו יום")
    elif days == 1:
        parts.append("נמצא יום אחרי")
    elif days == 2:
        parts.append("נמצא יומיים אחרי")
    else:
        parts.append(f"נמצא {days} ימים אחרי")
    if image_sim >= 0.75:
        parts.insert(0, "פנים דומות מאוד")
    return " · ".join(parts)
