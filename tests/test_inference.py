"""推定エンジン + 検証パイプラインの単体テスト（stdlib unittest, 依存ゼロ）。

    python3 -m unittest discover -s tests -v
"""

import math
import os
import random
import unittest

from src import ev as ev_mod
from src import likelihood as lik
from src import posterior as post_mod
from src import validate as val_mod

MACHINE = os.path.join(
    os.path.dirname(__file__), "..", "data", "machines", "my_juggler_v.json"
)


class TestLikelihood(unittest.TestCase):
    def setUp(self):
        self.model = post_mod.load_machine(MACHINE)

    def test_event_probs_monotone_budou(self):
        # ぶどう確率は設定が上がるほど高い（1/one_in 増加）= 設定差の主信号
        ps = [lik.event_probs_for_setting(self.model, k)["budou"] for k in range(1, 7)]
        self.assertEqual(ps, sorted(ps))

    def test_loglik_rejects_overcount(self):
        with self.assertRaises(ValueError):
            lik.log_likelihood(100, {"BIG": 200}, {"BIG": 0.004})

    def test_loglik_higher_for_matching_setting(self):
        # 設定6相当の観測は設定6の尤度が設定1より高いはず
        probs6 = lik.event_probs_for_setting(self.model, 6)
        N = 6000
        counts = {
            "BIG": round(N * probs6["BIG"]),
            "REG": round(N * probs6["REG"]),
            "budou": round(N * probs6["budou"]),
        }
        ll6 = lik.log_likelihood(N, counts, probs6)
        ll1 = lik.log_likelihood(N, counts, lik.event_probs_for_setting(self.model, 1))
        self.assertGreater(ll6, ll1)


class TestPosterior(unittest.TestCase):
    def setUp(self):
        self.model = post_mod.load_machine(MACHINE)

    def test_posterior_normalized(self):
        obs = {"N": 3000, "counts": {"BIG": 11, "REG": 9, "budou": 480}}
        post = post_mod.compute_posterior(self.model, obs)
        self.assertAlmostEqual(sum(post.values()), 1.0, places=9)

    def test_zero_data_returns_prior(self):
        # N=0・観測なし → 一様事前のまま
        post = post_mod.compute_posterior(self.model, {"N": 0, "counts": {}})
        for k, v in post.items():
            self.assertAlmostEqual(v, 1.0 / 6, places=9)

    def test_prior_zero_excludes_setting(self):
        prior = {1: 0.0, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2, 6: 0.2}
        obs = {"N": 1000, "counts": {"budou": 165}}
        post = post_mod.compute_posterior(self.model, obs, prior)
        self.assertEqual(post[1], 0.0)

    def test_recovers_true_setting_with_large_N(self):
        rng = random.Random(42)
        obs = val_mod.simulate_session(self.model, true_setting=6, N=12000, rng=rng)
        post = post_mod.compute_posterior(self.model, obs)
        self.assertEqual(post_mod.map_setting(post), 6)

    def test_events_filter_excludes_budou(self):
        # ぶどうを除外した事後は、ぶどう込みと異なるが正規化は保たれる
        rng = random.Random(99)
        obs = val_mod.simulate_session(self.model, true_setting=6, N=10000, rng=rng)
        full = post_mod.compute_posterior(self.model, obs)
        no_budou = post_mod.compute_posterior(self.model, obs, events={"BIG", "REG"})
        self.assertAlmostEqual(sum(no_budou.values()), 1.0, places=9)
        self.assertNotAlmostEqual(full[6], no_budou[6], places=3)
        # ぶどう抜きだと隣接設定に化けうる（MAPは5/6で揺れる）が、
        # 高設定群(4-6)の判別は BIG/REG だけでも保たれる（意思決定に直結する量）
        self.assertGreater(sum(no_budou[k] for k in (4, 5, 6)), 0.5)

    def test_partial_observation_must_restrict_events(self):
        # 回帰: 店カウンターの BIG/REG だけ（ぶどう未観測）を、events 限定せず
        # 全イベントのまま渡すと、ぶどう分が "other" に化けて事後が壊れる
        # （設定1へ誤収束）。events を観測量に限定すれば妥当な分布になる。
        rng = random.Random(7)
        obs = val_mod.simulate_session(self.model, true_setting=5, N=4000, rng=rng)
        obs["counts"].pop("budou")  # ぶどうは観測できなかった想定

        buggy = post_mod.compute_posterior(self.model, obs)  # events 限定なし=誤り
        fixed = post_mod.compute_posterior(self.model, obs, events=set(obs["counts"]))
        # 誤った経路は低設定に偏る / 正しい経路は高設定群に質量が乗る
        self.assertGreater(fixed[5] + fixed[6] + fixed[4], buggy[5] + buggy[6] + buggy[4])


class TestEV(unittest.TestCase):
    def setUp(self):
        self.model = post_mod.load_machine(MACHINE)

    def test_expected_payout_bounds(self):
        # 確率質量が全部設定6なら期待機械割=設定6の機械割（マイジャグラーV=109.4%）
        post = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1.0}
        self.assertAlmostEqual(
            ev_mod.expected_payout(post, self.model["payout"]),
            self.model["payout"]["6"], places=6
        )

    def test_slot_decision_play_when_high(self):
        post = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1.0}
        dec = ev_mod.slot_decision(post, self.model["payout"])
        self.assertTrue(dec["play"])

    def test_slot_decision_quit_when_low(self):
        post = {1: 1.0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        dec = ev_mod.slot_decision(post, self.model["payout"])
        self.assertFalse(dec["play"])

    def test_border(self):
        self.assertTrue(ev_mod.pachinko_border_decision(19.5, 18.2)["play"])
        self.assertFalse(ev_mod.pachinko_border_decision(17.0, 18.2)["play"])

    def test_pachinko_yen_ev_hand_calc(self):
        # R=20, B=18, 4000回転: 投資=4000/20*1000=200,000円
        # 期待収支 = 200000*(20-18)/18 = +22,222円
        res = ev_mod.pachinko_yen_ev(20.0, 18.0, 4000)
        self.assertAlmostEqual(res["invest_yen"], 200000.0, places=2)
        self.assertAlmostEqual(res["expected_yen"], 200000.0 * 2 / 18, places=2)
        self.assertTrue(res["play"])

    def test_pachinko_yen_ev_at_border_is_zero(self):
        res = ev_mod.pachinko_yen_ev(18.0, 18.0, 4000)
        self.assertAlmostEqual(res["expected_yen"], 0.0, places=6)
        self.assertFalse(res["play"])

    def test_pachinko_yen_ev_below_border_negative(self):
        res = ev_mod.pachinko_yen_ev(16.0, 18.0, 4000)
        self.assertLess(res["expected_yen"], 0)

    def test_yen_ev_setting6_matches_hand_calc(self):
        # 全質量が設定6(109.4%)、5000G・3枚・20円/枚:
        # net = 3*5000*(1.094-1) = 1410枚 → 28,200円
        post = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1.0}
        res = ev_mod.slot_yen_ev(post, self.model["payout"], games=5000,
                                 bet_per_game=3, yen_per_coin=20.0)
        self.assertAlmostEqual(res["expected_yen"], 28200.0, places=2)
        self.assertAlmostEqual(res["prob_plus"], 1.0, places=9)

    def test_yen_ev_low_setting_negative(self):
        post = {1: 1.0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}  # 設定1=97.0%
        res = ev_mod.slot_yen_ev(post, self.model["payout"], games=5000)
        self.assertLess(res["expected_yen"], 0)
        self.assertAlmostEqual(res["prob_plus"], 0.0, places=9)

    def test_yen_ev_expectation_over_posterior(self):
        # 期待収支は per_setting の事後加重和に一致
        post = {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.1, 6: 0.1}
        res = ev_mod.slot_yen_ev(post, self.model["payout"], games=3000)
        manual = sum(p["prob"] * p["yen"] for p in res["per_setting"])
        self.assertAlmostEqual(res["expected_yen"], manual, places=6)
        self.assertGreaterEqual(res["best"]["yen"], res["worst"]["yen"])

    def test_pnl_distribution_mean_matches_analytic_ev(self):
        # モンテカルロ分布の平均は解析的な期待収支(slot_yen_ev)に一致するはず
        post = {1: 0.1, 2: 0.1, 3: 0.2, 4: 0.25, 5: 0.2, 6: 0.15}
        analytic = ev_mod.slot_yen_ev(post, self.model["payout"], games=5000)["expected_yen"]
        dist = ev_mod.session_pnl_distribution(post, self.model, games=5000,
                                               trials=30000, seed=1)
        # 相対誤差 2% 以内（モンテカルロ誤差）
        self.assertLess(abs(dist["mean_yen"] - analytic), abs(analytic) * 0.02 + 2000)

    def test_pnl_percentiles_ordered_and_band_positive_width(self):
        post = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1.0}
        dist = ev_mod.session_pnl_distribution(post, self.model, games=5000,
                                               trials=20000, seed=2)
        p = dist["percentiles"]
        self.assertLessEqual(p["p5"], p["p25"])
        self.assertLessEqual(p["p25"], p["p50"])
        self.assertLessEqual(p["p50"], p["p75"])
        self.assertLessEqual(p["p75"], p["p95"])
        # 短期分散があるので帯には幅がある（p95 > p5）
        self.assertGreater(p["p95"], p["p5"])
        self.assertTrue(0.0 <= dist["prob_plus"] <= 1.0)

    def test_pnl_requires_payout_coins(self):
        post = {1: 1.0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        with self.assertRaises(ValueError):
            ev_mod.session_pnl_distribution(post, {"payout": {}, "events": {}}, games=100)

    def test_spins_to_recover_positive_ev(self):
        # 設定6(109.4%) 確定、2万円回収: 1Gあたり = 3*(1.094-1)*20 = 5.64円
        post = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1.0}
        rec = ev_mod.spins_to_recover(post, self.model["payout"], 20000)
        self.assertTrue(rec["recoverable"])
        self.assertAlmostEqual(rec["per_game_yen"], 3 * 0.094 * 20, places=6)
        self.assertAlmostEqual(rec["required_games"], 20000 / (3 * 0.094 * 20), places=2)

    def test_spins_to_recover_negative_ev_unrecoverable(self):
        post = {1: 1.0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}  # 97.0% < 100
        rec = ev_mod.spins_to_recover(post, self.model["payout"], 20000)
        self.assertFalse(rec["recoverable"])
        self.assertIsNone(rec["required_games"])

    def test_daily_plus_grows_for_high_setting(self):
        res = ev_mod.simulate_fixed_setting_daily(self.model, 6, 8000, [1, 365])
        ms = {r["day"]: r for r in res["milestones"]}
        # 高設定はプラス確率が日数とともに増える＆95%到達日数が有限
        self.assertGreater(ms[365]["prob_plus"], ms[1]["prob_plus"])
        self.assertIsNotNone(res["days_to_95pct_profitable"])

    def test_daily_low_setting_never_95(self):
        res = ev_mod.simulate_fixed_setting_daily(self.model, 1, 8000, [1, 365])
        ms = {r["day"]: r for r in res["milestones"]}
        # 低設定はプラス確率が日数とともに減る＆95%到達なし
        self.assertLess(ms[365]["prob_plus"], ms[1]["prob_plus"])
        self.assertIsNone(res["days_to_95pct_profitable"])


class TestValidationDiscipline(unittest.TestCase):
    """検証規律そのものが機能することを（軽量設定で）確認する。"""

    def setUp(self):
        self.model = post_mod.load_machine(MACHINE)

    def test_negative_control_beats_chance(self):
        nc = val_mod.negative_control(self.model, N=8000, trials_per_setting=40, seed=1)
        self.assertGreater(nc["overall_top1_accuracy"], nc["chance_level"])

    def test_required_samples_monotone_accuracy(self):
        rs = val_mod.required_samples(
            self.model, 4, 5, target_accuracy=0.9,
            candidates=(2000, 20000), trials=60, seed=2,
        )
        accs = [r["accuracy"] for r in rs["curve"]]
        # より多い N の方が分離精度が高い（厳密単調でなくても末尾が先頭以上）
        self.assertGreaterEqual(accs[-1], accs[0])

    def test_ev_backtest_played_beats_baseline(self):
        bt = val_mod.ev_backtest(self.model, N=8000, trials=400, seed=3)
        # 『打つ判定』台の真の平均機械割は、全台無差別ベースラインを上回るべき
        if bt["played_mean_true_payout"] is not None:
            self.assertGreater(
                bt["played_mean_true_payout"], bt["baseline_play_all_mean_payout"]
            )


if __name__ == "__main__":
    unittest.main()
