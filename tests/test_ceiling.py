"""軸B 天井EV計算のテスト。"""

import unittest

from src import ceiling as ceil


class TestCeilingEV(unittest.TestCase):
    def test_conservative_hand_calc(self):
        # 天井1200 / 現在900 → 残り300G。コイン持ち33G/1000円 → 1Gあたり 30.3円
        # 恩恵2000枚×20円 = 40,000円。投資 = 300×(1000/33) ≈ 9,091円
        res = ceil.ceiling_ev(1200, 900, benefit_yen=40000, yen_per_game=1000 / 33)
        self.assertEqual(res["model"], "conservative")
        self.assertEqual(res["remaining_games"], 300)
        self.assertAlmostEqual(res["invest_yen"], 300 * 1000 / 33, places=2)
        self.assertAlmostEqual(res["expected_yen"], 40000 - 300 * 1000 / 33, places=2)
        self.assertTrue(res["play"])

    def test_past_ceiling(self):
        res = ceil.ceiling_ev(1200, 1300, benefit_yen=40000, yen_per_game=30)
        self.assertEqual(res["model"], "past-ceiling")
        self.assertFalse(res["play"])

    def test_early_hit_reduces_invest_vs_conservative(self):
        # 早い当たりを織り込むと期待消化Gが残りより小さく、投資が減る
        cons = ceil.ceiling_ev(1200, 600, 40000, 30)
        early = ceil.ceiling_ev(1200, 600, 40000, 30, hit_rate=1 / 400, early_hit_yen=10000)
        self.assertEqual(early["model"], "early-hit")
        self.assertLess(early["expected_games"], cons["remaining_games"])
        self.assertLess(early["prob_reach_ceiling"], 1.0)

    def test_breakeven_current_monotone_sense(self):
        # 恩恵が大きいほど狙い目ボーダーは手前（小さいG）になる
        be_small = ceil.breakeven_current(1200, 20000, 1000 / 33)
        be_big = ceil.breakeven_current(1200, 60000, 1000 / 33)
        self.assertIsNotNone(be_small)
        self.assertIsNotNone(be_big)
        self.assertLess(be_big, be_small)

    def test_breakeven_none_when_unprofitable(self):
        # 1Gの消化コストが恩恵を上回るなら、天井1G手前でも-EV→狙い目なし
        be = ceil.breakeven_current(1200, benefit_yen=1000, yen_per_game=2000)
        self.assertIsNone(be)


if __name__ == "__main__":
    unittest.main()
