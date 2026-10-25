"""Tests that need no model downloads. Run: python -m unittest discover tests"""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from petmatch.data import Photo, load_from_csv, load_from_folders, make_split
from petmatch.embedders import ColorHistogramEmbedder
from petmatch.index import VectorIndex
from petmatch.metrics import first_hit_ranks, mean_reciprocal_rank, recall_at_k


def make_fake_pets(root: Path, n_pets=12, photos_per_pet=3, singles=4, seed=0):
    """Each pet = a distinct base color; each photo = that color + noise."""
    rng = np.random.default_rng(seed)
    for i in range(n_pets + singles):
        d = root / f"pet{i:02d}"
        d.mkdir(parents=True)
        base = rng.integers(0, 256, size=3)
        for j in range(photos_per_pet if i < n_pets else 1):
            noise = rng.integers(-20, 21, size=(48, 48, 3))
            arr = np.clip(base + noise, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(d / f"{j}.png")


class TestMetrics(unittest.TestCase):
    def test_ranks_dedupe_by_pet(self):
        # gallery returns photos; 'a' twice should count as one pet
        ranked = [["a", "a", "b"], ["c", "b", "a"], ["x", "y", "z"]]
        ranks = first_hit_ranks(ranked, ["b", "a", "q"])
        self.assertEqual(ranks.tolist(), [2, 3, 0])

    def test_recall_and_mrr(self):
        ranks = np.array([1, 2, 6, 0])
        self.assertEqual(recall_at_k(ranks, 1), 0.25)
        self.assertEqual(recall_at_k(ranks, 5), 0.5)
        self.assertAlmostEqual(mean_reciprocal_rank(ranks), (1 + 0.5 + 1 / 6) / 4)


class TestIndex(unittest.TestCase):
    def test_nearest_is_itself(self):
        idx = VectorIndex()
        idx.add(np.eye(3), ["x", "y", "z"])
        ids, scores = idx.search(np.array([[0, 2.0, 0]]), k=2)
        self.assertEqual(ids[0][0], "y")
        self.assertAlmostEqual(float(scores[0][0]), 1.0, places=5)

    def test_k_larger_than_index(self):
        idx = VectorIndex()
        idx.add(np.eye(2), ["x", "y"])
        ids, _ = idx.search(np.eye(2), k=10)
        self.assertEqual(len(ids[0]), 2)


class TestSplit(unittest.TestCase):
    def test_one_query_per_multi_photo_pet_and_singles_are_distractors(self):
        photos = [Photo(Path(f"{pid}{n}.jpg"), pid)
                  for pid, count in [("a", 3), ("b", 2), ("c", 1)] for n in range(count)]
        s = make_split(photos, seed=1)
        self.assertEqual(sorted(q.pet_id for q in s.queries), ["a", "b"])
        self.assertEqual(len(s.gallery), 4)  # 2 of a, 1 of b, 1 distractor c
        self.assertFalse(set(map(id, s.queries)) & set(map(id, s.gallery)))

    def test_no_matchable_pets_raises(self):
        with self.assertRaises(ValueError):
            make_split([Photo(Path("x.jpg"), "a"), Photo(Path("y.jpg"), "b")])

    def test_same_seed_same_split(self):
        photos = [Photo(Path(f"{p}{n}.jpg"), p) for p in "abcdef" for n in range(3)]
        a = make_split(photos, max_pets=4, seed=7)
        b = make_split(photos, max_pets=4, seed=7)
        self.assertEqual([q.path for q in a.queries], [q.path for q in b.queries])


class TestEmbeddingCache(unittest.TestCase):
    def test_cached_vectors_match_and_skip_recompute(self):
        import petmatch.embedders as E
        with tempfile.TemporaryDirectory() as tmp:
            old = E.CACHE_DIR
            E.CACHE_DIR = Path(tmp) / "cache"
            try:
                make_fake_pets(Path(tmp) / "pets", n_pets=3, photos_per_pet=2, singles=0)
                paths = [p.path for p in load_from_folders(Path(tmp) / "pets")]
                emb = ColorHistogramEmbedder()
                first = emb.embed(paths)
                calls = []
                emb._embed_batch = lambda imgs: calls.append(len(imgs))  # must not be called
                again = emb.embed(list(reversed(paths)))
                self.assertEqual(calls, [])
                np.testing.assert_allclose(again, first[::-1])
            finally:
                E.CACHE_DIR = old


class TestEndToEnd(unittest.TestCase):
    def test_folder_and_csv_loaders_agree_and_baseline_finds_pets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pets"
            make_fake_pets(root)
            photos = load_from_folders(root)

            csv_path = Path(tmp) / "pets.csv"
            with csv_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["path", "pet_id"])
                for p in photos:
                    w.writerow([p.path.relative_to(tmp), p.pet_id])
            self.assertEqual(sorted((str(p.path), p.pet_id) for p in load_from_csv(csv_path)),
                             sorted((str(p.path), p.pet_id) for p in photos))

            split = make_split(photos, seed=0)
            emb = ColorHistogramEmbedder()
            idx = VectorIndex()
            idx.add(emb.embed([p.path for p in split.gallery]), [p.pet_id for p in split.gallery])
            ranked, _ = idx.search(emb.embed([p.path for p in split.queries]), k=20)
            ranks = first_hit_ranks(ranked, [q.pet_id for q in split.queries])
            # distinct solid colors are easy: the baseline should nail them
            self.assertGreaterEqual(recall_at_k(ranks, 5), 0.9)


if __name__ == "__main__":
    unittest.main()
