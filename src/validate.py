"""検証規律（Loto6 evaluation.py の移植）。

設定看破は『それっぽい数字』を返すので、当たっているかを必ず検証する。
ここでは信号が既知（真の設定が分かっている合成データ）の条件で推定器を較正する。

提供する検証:
  - negative_control : 既知設定で生成 → 事後が真の設定へ収束するか
  - calibration      : 「P(設定=k)=p」と言った台が長期で本当に頻度 p か
  - required_samples : 設定 a と b を所定精度で分離するのに必要な N（検出力/MDE）
  - ev_backtest      : 『打つ判定』の台の合成実収支が期待EVと整合するか
"""

from __future__ import annotations

import random
from typing import Dict, List, Mapping, Optional, Sequence

from .ev import expected_payout, slot_decision
from .likelihood import event_probs_for_setting
from .posterior import compute_posterior, map_setting


# --------------------------------------------------------------------------- #
# 合成データ生成
# --------------------------------------------------------------------------- #
def simulate_session(model: Mapping, true_setting: int, N: int, rng: random.Random) -> Dict:
    """真の設定から N ゲーム分の観測（各イベント回数）を生成する。

    1ゲームを {events..., other} の multinomial 1試行とみなしてサンプリング。
    """
    probs = event_probs_for_setting(model, true_setting)
    names = list(probs.keys())
    weights = [probs[n] for n in names]
    p_other = 1.0 - sum(weights)
    if p_other < 0:
        raise ValueError(f"event probabilities exceed 1 for setting {true_setting}")
    categories = names + ["other"]
    weights = weights + [p_other]

    counts = {n: 0 for n in names}
    # random.choices を N 回ではなく1コールで（高速・等価）
    draws = rng.choices(categories, weights=weights, k=N)
    for d in draws:
        if d != "other":
            counts[d] += 1
    return {"N": N, "counts": counts, "true_setting": true_setting}


# --------------------------------------------------------------------------- #
# 1. ネガティブコントロール
# --------------------------------------------------------------------------- #
def negative_control(
    model: Mapping,
    N: int,
    trials_per_setting: int = 200,
    seed: int = 0,
    prior: Optional[Mapping[int, float]] = None,
    events=None,
) -> Dict:
    """各既知設定で合成 → 事後の真設定への質量と top-1 的中率を測る。

    合格目安: 真設定の平均事後質量・top-1 的中率が一様事前(1/6≈0.167)を明確に上回る。
    N を増やすほど 1.0 に近づくべき（収束）。
    """
    settings: List[int] = model["settings"]
    rng = random.Random(seed)
    per_setting = {}
    overall_correct = 0
    overall_total = 0
    for true_k in settings:
        mass_sum = 0.0
        correct = 0
        for _ in range(trials_per_setting):
            obs = simulate_session(model, true_k, N, rng)
            post = compute_posterior(model, obs, prior, events=events)
            mass_sum += post[true_k]
            if map_setting(post) == true_k:
                correct += 1
        per_setting[true_k] = {
            "mean_true_mass": mass_sum / trials_per_setting,
            "top1_accuracy": correct / trials_per_setting,
        }
        overall_correct += correct
        overall_total += trials_per_setting
    return {
        "N": N,
        "trials_per_setting": trials_per_setting,
        "per_setting": per_setting,
        "overall_top1_accuracy": overall_correct / overall_total,
        "chance_level": 1.0 / len(settings),
    }


# --------------------------------------------------------------------------- #
# 2. キャリブレーション
# --------------------------------------------------------------------------- #
def calibration(
    model: Mapping,
    N: int,
    target_setting: Optional[int] = None,
    trials: int = 2000,
    n_bins: int = 10,
    seed: int = 0,
    prior: Optional[Mapping[int, float]] = None,
) -> Dict:
    """「設定=k の確率 p」と言った台が、長期で本当に頻度 p で設定 k かを検証。

    真の設定を全設定から一様にサンプル → 推定 → 予測確率をビン化し、
    各ビンの平均予測確率 vs 実現頻度を比較する。良い較正なら対角線に乗る。
    """
    settings: List[int] = model["settings"]
    if target_setting is None:
        target_setting = settings[-1]  # 既定: 最高設定（実戦で一番知りたい）
    rng = random.Random(seed)

    bins = [{"pred_sum": 0.0, "hit": 0, "count": 0} for _ in range(n_bins)]
    for _ in range(trials):
        true_k = rng.choice(settings)
        obs = simulate_session(model, true_k, N, rng)
        post = compute_posterior(model, obs, prior)
        p = post[target_setting]
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx]["pred_sum"] += p
        bins[idx]["count"] += 1
        if true_k == target_setting:
            bins[idx]["hit"] += 1

    rows = []
    ece = 0.0  # Expected Calibration Error
    for b in bins:
        if b["count"] == 0:
            continue
        mean_pred = b["pred_sum"] / b["count"]
        freq = b["hit"] / b["count"]
        rows.append(
            {
                "mean_predicted": mean_pred,
                "observed_freq": freq,
                "count": b["count"],
            }
        )
        ece += (b["count"] / trials) * abs(mean_pred - freq)

    return {
        "N": N,
        "target_setting": target_setting,
        "trials": trials,
        "bins": rows,
        "ece": ece,  # 0 に近いほど良い
    }


# --------------------------------------------------------------------------- #
# 3. 検出力 / 必要サンプル（MDE）
# --------------------------------------------------------------------------- #
def required_samples(
    model: Mapping,
    setting_a: int,
    setting_b: int,
    target_accuracy: float = 0.95,
    candidates: Sequence[int] = (1000, 2000, 3000, 5000, 8000, 10000, 15000, 20000),
    trials: int = 300,
    seed: int = 0,
    events=None,
) -> Dict:
    """設定 a と b の2択分離が target_accuracy に達する最小 N を探す。

    事前は {a, b} の2点一様。真の設定を a/b 半々で生成し、MAP が当たる率を測る。
    朝一など少サンプルでの過信を防ぐための定量化。
    """
    rng = random.Random(seed)
    two_prior = {setting_a: 0.5, setting_b: 0.5}
    per_N = []
    answer = None
    for N in candidates:
        correct = 0
        for i in range(trials):
            true_k = setting_a if i % 2 == 0 else setting_b
            obs = simulate_session(model, true_k, N, rng)
            post = compute_posterior(model, obs, prior=two_prior, events=events)
            if map_setting(post) == true_k:
                correct += 1
        acc = correct / trials
        per_N.append({"N": N, "accuracy": acc})
        if answer is None and acc >= target_accuracy:
            answer = N
    return {
        "setting_a": setting_a,
        "setting_b": setting_b,
        "target_accuracy": target_accuracy,
        "required_N": answer,  # None なら候補内では未達
        "curve": per_N,
    }


# --------------------------------------------------------------------------- #
# 4. EV バックテスト
# --------------------------------------------------------------------------- #
def ev_backtest(
    model: Mapping,
    N: int,
    setting_distribution: Optional[Mapping[int, float]] = None,
    trials: int = 2000,
    threshold_pct: float = 100.0,
    seed: int = 0,
    prior: Optional[Mapping[int, float]] = None,
) -> Dict:
    """ホールの真の設定分布から台を生成 → 『打つ判定』台の実機械割平均を測る。

    合格目安: play=True と判定した台の真の平均機械割 > 100%（判定が収益的）。
    setting_distribution 既定はやや弱め（高設定が少ない現実的ホール）。
    """
    settings: List[int] = model["settings"]
    payout = model["payout"]
    if setting_distribution is None:
        # 現実的な弱め分布: 低設定多め・高設定少なめ
        base = {1: 0.35, 2: 0.25, 3: 0.18, 4: 0.12, 5: 0.06, 6: 0.04}
        setting_distribution = {k: base.get(k, 0.0) for k in settings}
    dist_settings = list(setting_distribution.keys())
    dist_weights = [setting_distribution[k] for k in dist_settings]

    rng = random.Random(seed)
    played_true_payout = []
    skipped_true_payout = []
    for _ in range(trials):
        true_k = rng.choices(dist_settings, weights=dist_weights, k=1)[0]
        obs = simulate_session(model, true_k, N, rng)
        post = compute_posterior(model, obs, prior)
        dec = slot_decision(post, payout, threshold_pct)
        true_pay = float(payout[str(true_k)])
        if dec["play"]:
            played_true_payout.append(true_pay)
        else:
            skipped_true_payout.append(true_pay)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    # 全台無差別に打った場合の期待機械割（ベースライン）
    baseline = sum(setting_distribution[k] * float(payout[str(k)]) for k in dist_settings) \
        / sum(dist_weights)

    return {
        "N": N,
        "trials": trials,
        "threshold_pct": threshold_pct,
        "n_played": len(played_true_payout),
        "n_skipped": len(skipped_true_payout),
        "played_mean_true_payout": _mean(played_true_payout),
        "skipped_mean_true_payout": _mean(skipped_true_payout),
        "baseline_play_all_mean_payout": baseline,  # 全台無差別に打った場合
    }
