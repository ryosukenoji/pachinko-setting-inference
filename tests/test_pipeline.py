"""CSV取込・ホール事前学習・外部事前のテスト（stdlib unittest）。"""

import os
import random
import unittest

from src import dataio as io_mod
from src import learn as learn_mod
from src import posterior as post_mod
from src import validate as val_mod

HERE = os.path.dirname(__file__)
MACHINES_DIR = os.path.join(HERE, "..", "data", "machines")
SAMPLE_CSV = os.path.join(HERE, "fixtures", "observations_sample.csv")


class TestDataIO(unittest.TestCase):
    def test_read_observations(self):
        recs = io_mod.read_observations_csv(SAMPLE_CSV)
        self.assertEqual(len(recs), 3)
        a = recs[0]
        self.assertEqual(a["machine"], "my_juggler_v")
        self.assertEqual(a["label"], "A台")
        self.assertEqual(a["N"], 6000)
        self.assertEqual(a["counts"], {"BIG": 24, "REG": 21, "budou": 1010})

    def test_load_machine_by_id(self):
        m = post_mod.load_machine_by_id(MACHINES_DIR, "im_juggler_ex")
        self.assertEqual(m["machine"], "アイムジャグラーEX")

    def test_load_machine_by_id_missing(self):
        with self.assertRaises(FileNotFoundError):
            post_mod.load_machine_by_id(MACHINES_DIR, "no_such_machine")


class TestPriorParsing(unittest.TestCase):
    def setUp(self):
        self.settings = [1, 2, 3, 4, 5, 6]

    def test_parse_inline_json_normalizes(self):
        prior = post_mod.parse_prior('{"6": 2, "1": 2}', self.settings)
        self.assertAlmostEqual(sum(prior.values()), 1.0, places=9)
        self.assertAlmostEqual(prior[6], 0.5, places=9)
        self.assertEqual(prior[3], 0.0)

    def test_parse_none(self):
        self.assertIsNone(post_mod.parse_prior(None, self.settings))

    def test_parse_rejects_zero_sum(self):
        with self.assertRaises(ValueError):
            post_mod.parse_prior('{"1":0}', self.settings)


class TestHallPriorLearning(unittest.TestCase):
    def setUp(self):
        self.model = post_mod.load_machine_by_id(MACHINES_DIR, "my_juggler_v")

    def test_em_recovers_skewed_hall_distribution(self):
        # 真のホール分布: 設定1が多く設定6が少ない → EM がその偏りを再現するはず
        rng = random.Random(7)
        true_dist = {1: 0.5, 2: 0.2, 3: 0.1, 4: 0.1, 5: 0.05, 6: 0.05}
        settings = list(true_dist.keys())
        weights = [true_dist[k] for k in settings]
        obs = []
        for _ in range(400):
            k = rng.choices(settings, weights=weights, k=1)[0]
            obs.append(val_mod.simulate_session(self.model, k, N=6000, rng=rng))

        res = learn_mod.estimate_hall_prior(self.model, obs, iters=200)
        prior = res["prior"]
        self.assertAlmostEqual(sum(prior.values()), 1.0, places=6)
        # 低設定(1)の質量が高設定(6)を明確に上回る
        self.assertGreater(prior[1], prior[6])
        # 大まかに真分布へ寄る（設定1は推定でも最大質量）
        self.assertEqual(max(prior, key=prior.get), 1)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            learn_mod.estimate_hall_prior(self.model, [])


if __name__ == "__main__":
    unittest.main()
