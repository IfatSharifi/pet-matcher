"""Ranking tests - no downloads, no API. Run: python -m unittest discover tests"""
import unittest

import numpy as np

from petmatch.ranking import (EXCLUDED, Features, Weights, best_photo_similarity, color_similarity,
                              haversine_matrix, mrr, reason_he, score, true_ranks, tune)
from petmatch.simulate import SCENARIOS, Place, haversine_km, simulate


class TestSpecies(unittest.TestCase):
    def test_only_dog_vs_cat_excludes(self):
        from petmatch.ranking import species_compatible
        self.assertFalse(species_compatible("dog", "cat"))
        self.assertTrue(species_compatible("dog", "dog"))
        self.assertTrue(species_compatible("other", "dog"))    # the bug: a wolf-like dog labeled 'other'
        self.assertTrue(species_compatible("unknown", "cat"))


class TestColors(unittest.TestCase):
    def test_exact_near_and_unknown(self):
        self.assertEqual(color_similarity(["black", "white"], ["white", "black"]), 1.0)
        self.assertEqual(color_similarity(["tan"], ["brown"]), 0.5)      # near colors: half credit
        self.assertEqual(color_similarity(["black"], ["golden"]), 0.0)
        self.assertEqual(color_similarity([], ["black"]), 0.5)           # unknown: neutral
        self.assertEqual(color_similarity(["black"], ["black", "white"]),
                         color_similarity(["black", "white"], ["black"]))  # symmetric


class TestGeo(unittest.TestCase):
    def test_tel_aviv_jerusalem(self):
        km = haversine_matrix([32.080], [34.781], [31.778], [35.214])[0, 0]
        self.assertAlmostEqual(km, 54, delta=2)
        self.assertAlmostEqual(km, haversine_km(Place(32.080, 34.781, ""), Place(31.778, 35.214, "")), places=3)

    def test_simulation_is_repeatable_and_found_after_lost(self):
        ids = [f"d{i}" for i in range(300)]
        a = simulate(ids, SCENARIOS["base"], seed=1)
        b = simulate(ids, SCENARIOS["base"], seed=1)
        self.assertEqual(a, b)
        lost, found = a
        kms = [haversine_km(lost[i][0], found[i][0]) for i in ids]
        self.assertTrue(all(found[i][1] >= lost[i][1] for i in ids))
        self.assertLess(np.median(kms), 4)  # mostly found nearby under 'base'
        wander = simulate(ids, SCENARIOS["wander"], seed=1)
        self.assertGreater(np.median([haversine_km(wander[0][i][0], wander[1][i][0]) for i in ids]),
                           np.median(kms))


class TestScoring(unittest.TestCase):
    def test_best_photo_per_report(self):
        q = np.array([[1.0, 0.0]])
        photos = np.array([[0.0, 1.0], [1.0, 0.1], [0.6, 0.8]])
        report = np.array([0, 0, 1])
        sim = best_photo_similarity(q, photos, report, 2)
        self.assertAlmostEqual(float(sim[0, 0]), 1 / np.sqrt(1.01), places=5)  # best of report 0
        self.assertAlmostEqual(float(sim[0, 1]), 0.6, places=5)

    def test_ranks_ties_count_against_truth(self):
        s = np.array([[0.9, 0.5, 0.9], [0.1, 0.8, 0.2]])
        self.assertEqual(true_ranks(s, np.array([0, 1])).tolist(), [2, 1])

    def test_hard_rules_exclude(self):
        f = Features(np.ones((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)), np.array([[0, -5]]),
                     np.array([[True, False]]))
        s = score(f, Weights(use_rules=True))
        self.assertEqual(s[0, 1], EXCLUDED)

    def test_tune_finds_useful_signal(self):
        # image is useless noise; distance identifies the truth -> tuning must turn distance on
        rng = np.random.default_rng(0)
        n = 60
        image = rng.uniform(0.4, 0.6, (n, n)).astype(np.float32)
        km = rng.uniform(5, 50, (n, n))
        np.fill_diagonal(km, 0.5)
        f = Features(image, np.full((n, n), 0.5), km, np.zeros((n, n), int), np.ones((n, n), bool))
        truth = np.arange(n)
        w = tune(f, truth, Weights(), {"dist": [0, 0.01, 0.05, 0.1, 0.5]})
        self.assertGreater(w.dist, 0)
        self.assertGreater(mrr(true_ranks(score(f, w), truth)), 0.95)


class TestReasons(unittest.TestCase):
    def test_hebrew_reason(self):
        r = reason_he(["black", "white"], ["black", "white"], 1.84, 2, 0.8)
        self.assertEqual(r, 'פנים דומות מאוד · אותו צבע: שחור, לבן · 1.8 ק"מ מהמקום שבו אבד · נמצא יומיים אחרי')
        self.assertIn("600 מטר", reason_he(["tan"], ["brown"], 0.58, 0, 0.5))
        self.assertIn("פחות מ-100 מטר", reason_he(["tan"], ["tan"], 0.01, 0, 0.5))
        self.assertIn("צבע דומה", reason_he(["tan"], ["brown"], 0.58, 0, 0.5))
        self.assertIn("נמצא באותו יום", reason_he(["tan"], ["brown"], 0.58, 0, 0.5))


if __name__ == "__main__":
    unittest.main()
