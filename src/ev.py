"""事後分布 → EV判定（スロット）と、回転率 → ボーダー判定（パチンコ）。

design_proposal.md §3 の A（設定看破→EV）と B（ボーダー理論）に対応。
本モジュールは『予測』ではなく、推定済み分布／観測済み回転率からの決定論的な判定のみ。
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Mapping, Optional


def expected_payout(posterior: Mapping[int, float], payout: Mapping) -> float:
    """事後 × 公表機械割 → 期待機械割（%）。

    payout は JSON 由来でキーが str の場合があるため両対応。
    """
    ev = 0.0
    for k, p in posterior.items():
        val = payout.get(str(k), payout.get(k))
        if val is None:
            raise KeyError(f"payout missing for setting {k}")
        ev += p * float(val)
    return ev


def slot_decision(
    posterior: Mapping[int, float],
    payout: Mapping,
    threshold_pct: float = 100.0,
) -> Dict:
    """期待機械割が閾値（既定100%）超なら『打つ』。

    Returns:
        {expected_payout, threshold, play, edge_pct}
        edge_pct は期待機械割 - 閾値（プラスで優位）。
    """
    ev = expected_payout(posterior, payout)
    return {
        "expected_payout": ev,
        "threshold": threshold_pct,
        "play": ev > threshold_pct,
        "edge_pct": ev - threshold_pct,
    }


def slot_yen_ev(
    posterior: Mapping[int, float],
    payout: Mapping,
    games: int,
    bet_per_game: int = 3,
    yen_per_coin: float = 20.0,
) -> Dict:
    """事後 × 機械割 → 想定遊技数での円建て期待収支と、設定不確実性による分布。

    機械割 M%(設定k) = 払出コイン / 投入コイン × 100 なので、
        投入コイン = bet_per_game × games
        純増コイン(k) = 投入コイン × (M_k/100 - 1)
        円(k) = 純増コイン(k) × yen_per_coin
        期待収支 = Σ post_k × 円(k)

    Args:
        games: これから回す予定のゲーム数。
        bet_per_game: 1ゲームの掛けコイン（ジャグラー等は3枚）。
        yen_per_coin: 換金レート（円/枚）。等価=20。

    返り値の per_setting で「設定不確実性に由来する収支の幅」を、prob_plus で
    「プラス収支になる確率」を示す。**1セッション内の短期分散（ヒキ）は含まない**
    （これは設定が確定していても残る別物で、短期では支配的。caveat 参照）。
    """
    total_bet = bet_per_game * games
    per = []
    for k in sorted(posterior):
        m = float(payout.get(str(k), payout.get(k)))
        net_coins = total_bet * (m / 100.0 - 1.0)
        per.append({
            "setting": k,
            "prob": posterior[k],
            "payout": m,
            "yen": net_coins * yen_per_coin,
        })
    exp_yen = sum(p["prob"] * p["yen"] for p in per)
    return {
        "games": games,
        "bet_per_game": bet_per_game,
        "yen_per_coin": yen_per_coin,
        "total_bet_coins": total_bet,
        "total_bet_yen": total_bet * yen_per_coin,
        "expected_yen": exp_yen,
        "prob_plus": sum(p["prob"] for p in per if p["yen"] > 0),
        "per_setting": per,
        "best": max(per, key=lambda x: x["yen"]),
        "worst": min(per, key=lambda x: x["yen"]),
        "caveat": "設定不確実性のみ反映。1セッションの短期分散（ヒキ）は別途で、短期では支配的。",
    }


def spins_to_recover(
    posterior: Mapping[int, float],
    payout: Mapping,
    target_yen: float,
    bet_per_game: int = 3,
    yen_per_coin: float = 20.0,
) -> Dict:
    """目標額(円)を期待ベースで回収するのに必要なゲーム数（必要回転数の逆算）。

    事後加重の期待機械割 E[M] から 1Gあたり期待純益 = bet×(E[M]/100-1)×yen_per_coin。
    必要G数 = target_yen / 1Gあたり期待純益。E[M]<=100（-EV）なら回収不能（None）。

    注意: これは**期待値ベースの目安**。短期は分散支配で、この回転数を回しても
    実収支は大きく振れる（session_pnl_distribution の帯を併用すること）。
    """
    exp_m = expected_payout(posterior, payout)
    per_game_yen = bet_per_game * (exp_m / 100.0 - 1.0) * yen_per_coin
    recoverable = per_game_yen > 0
    return {
        "target_yen": target_yen,
        "expected_payout": exp_m,
        "per_game_yen": per_game_yen,
        "recoverable": recoverable,
        "required_games": (target_yen / per_game_yen) if recoverable else None,
        "caveat": "期待値ベースの目安。短期は分散支配で実収支は大きく振れる。-EV設定では回収不能。",
    }


def _poisson(rng: random.Random, lam: float) -> int:
    """Poisson 乱数（標準ライブラリのみ）。BIG/REG は希少イベントなので二項の良近似。

    λ が大きい領域は正規近似に切り替え（Knuth 法の反復回数を抑える）。
    """
    if lam <= 0:
        return 0
    if lam > 30:
        return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))
    target = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= target:
            return k - 1


def session_pnl_distribution(
    posterior: Mapping[int, float],
    model: Mapping,
    games: int,
    trials: int = 20000,
    yen_per_coin: float = 20.0,
    seed: int = 0,
) -> Dict:
    """設定不確実性（事後）＋ 短期分散（ヒキ）を合成した、1セッション収支の分布。

    各試行: 設定 k を事後からサンプル → その設定で BIG/REG 回数を Poisson サンプル
    → ボーナス払出＋（機械割に整合させた）残余の決定値 − 投入、で純増枚数を得る。

    期待値は構成上 slot_yen_ev と一致（残余を機械割に合わせて補完するため）。
    分散は BIG/REG 回数のゆらぎ × 純増枚数から出る（payout_coins は分散の大きさにのみ影響）。

    Returns:
        mean_yen / prob_plus / percentiles{p5,p25,p50,p75,p95} / worst・best（サンプル端） など。
    """
    pc = model.get("payout_coins")
    if not pc:
        raise ValueError("model に payout_coins が無いため短期分散を計算できません")
    bet = pc.get("bet_per_game", 3)
    big_pay = pc["BIG"]
    reg_pay = pc["REG"]
    payout = model["payout"]

    settings: List[int] = sorted(posterior)
    weights = [posterior[k] for k in settings]

    # 設定ごとに p_big/p_reg と残余（機械割に整合する決定値）を前計算
    p_big = {k: 1.0 / model["events"]["BIG"]["one_in"][str(k)] for k in settings}
    p_reg = {k: 1.0 / model["events"]["REG"]["one_in"][str(k)] for k in settings}
    total_bet = bet * games
    residual = {}
    for k in settings:
        m = float(payout.get(str(k), payout.get(k)))
        expected_total_payout = (m / 100.0) * total_bet
        expected_bonus = games * (p_big[k] * big_pay + p_reg[k] * reg_pay)
        residual[k] = expected_total_payout - expected_bonus  # ぶどう・小役等の決定値

    rng = random.Random(seed)
    yens = []
    plus = 0
    for _ in range(trials):
        k = rng.choices(settings, weights=weights, k=1)[0]
        n_big = _poisson(rng, games * p_big[k])
        n_reg = _poisson(rng, games * p_reg[k])
        total_payout = n_big * big_pay + n_reg * reg_pay + residual[k]
        net_coins = total_payout - total_bet
        y = net_coins * yen_per_coin
        yens.append(y)
        if y > 0:
            plus += 1

    yens.sort()

    def pct(q):
        idx = min(int(q * trials), trials - 1)
        return yens[idx]

    return {
        "games": games,
        "trials": trials,
        "yen_per_coin": yen_per_coin,
        "mean_yen": sum(yens) / trials,
        "prob_plus": plus / trials,
        "percentiles": {
            "p5": pct(0.05),
            "p25": pct(0.25),
            "p50": pct(0.50),
            "p75": pct(0.75),
            "p95": pct(0.95),
        },
        "worst": yens[0],
        "best": yens[-1],
        "caveat": "設定不確実性＋ボーナス分散を反映。payout_coins は代表値[unverified]のため帯の幅は近似。",
    }


def _normal_cdf(x: float) -> float:
    """標準正規分布の累積分布 Φ(x)（math.erf のみ）。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def simulate_fixed_setting_daily(
    model: Mapping,
    setting: int,
    games_per_day: int,
    day_milestones,
    yen_per_coin: float = 20.0,
) -> Dict:
    """設定を固定して毎日打ち続けた場合の、累積収支の推移（解析・正規近似）。

    1日の収支は独立同分布。BIG/REG 回数 ~ Poisson(λ) なので
        1日の期待コイン = bet×games×(機械割/100 - 1)
        1日の分散コイン = λ_BIG×BIG払出² + λ_REG×REG払出²   （残余は決定値→分散0）
    D日後の累積は平均 D×μ、標準偏差 √D×σ の正規分布。
    → +EV設定は平均が線形に伸び、標準偏差は√Dなので、いつか帯が0を上回り勝ち確定に近づく。
       -EV設定は逆に沈む。これが「長期EVは効く／短期は分散支配」の数値的な姿。

    Returns:
        per_day（μ・σ）と、各マイルストン日数での 平均・90%帯・プラス確率、
        および「95%の確率でプラスになる日数」。
    """
    pc = model.get("payout_coins")
    if not pc:
        raise ValueError("model に payout_coins が無いため計算できません")
    bet = pc.get("bet_per_game", 3)
    big_pay, reg_pay = pc["BIG"], pc["REG"]
    s = str(setting)
    p_big = 1.0 / model["events"]["BIG"]["one_in"][s]
    p_reg = 1.0 / model["events"]["REG"]["one_in"][s]
    m = float(model["payout"].get(s, model["payout"].get(setting)))

    lam_big = games_per_day * p_big
    lam_reg = games_per_day * p_reg
    daily_mu = bet * games_per_day * (m / 100.0 - 1.0) * yen_per_coin
    daily_var = (lam_big * big_pay ** 2 + lam_reg * reg_pay ** 2) * (yen_per_coin ** 2)
    daily_sigma = math.sqrt(daily_var)

    rows = []
    for d in day_milestones:
        mean = d * daily_mu
        sigma = math.sqrt(d) * daily_sigma
        if sigma > 0:
            p_plus = _normal_cdf(mean / sigma)
        else:
            p_plus = 1.0 if mean > 0 else 0.0
        rows.append({
            "day": d,
            "mean_yen": mean,
            "p5": mean - 1.645 * sigma,
            "p95": mean + 1.645 * sigma,
            "prob_plus": p_plus,
        })

    # 95%の確率でプラスになる日数: √d·μ/σ ≥ 1.645  →  d ≥ (1.645·σ/μ)²
    days_to_95 = None
    if daily_mu > 0 and daily_sigma > 0:
        days_to_95 = (1.645 * daily_sigma / daily_mu) ** 2

    return {
        "setting": setting,
        "machine_rate": m,
        "games_per_day": games_per_day,
        "yen_per_coin": yen_per_coin,
        "daily_mu_yen": daily_mu,
        "daily_sigma_yen": daily_sigma,
        "milestones": rows,
        "days_to_95pct_profitable": days_to_95,
        "caveat": "設定を毎日固定で打てる理想化（実際はホールが日々変える）。payout_coinsの分散は代表値。正規近似。",
    }


def pachinko_border_decision(
    spins_per_1k: float,
    borderline: float,
) -> Dict:
    """パチンコ: 実測回転率（¥1000あたり回転数）がボーダー超なら +EV。

        EV > 0  ⇔  実測回転率 > ボーダーライン

    Args:
        spins_per_1k: 観測した¥1000あたり回転数（区間推定の点推定値）。
        borderline: その台・交換率の損益分岐回転率（公表値）。
    """
    return {
        "spins_per_1k": spins_per_1k,
        "borderline": borderline,
        "play": spins_per_1k > borderline,
        "margin": spins_per_1k - borderline,
    }


def pachinko_yen_ev(
    spins_per_1k: float,
    borderline: float,
    total_spins: int,
    yen_per_unit: float = 1000.0,
) -> Dict:
    """パチンコ: 回転率とボーダーから、予定総回転数での円建て期待収支。

        投資(円)   = yen_per_unit × total_spins / R        （R回転で yen_per_unit 円消費）
        期待収支(円) = 投資 × (R − B) / B = yen_per_unit × total_spins × (R−B)/(R×B)

    R=B で期待収支0、R>B で +EV。スロットの機械割EVと違い設定の事後は無く、
    観測した回転率（区間推定の点推定）から決定論的に算出する。

    前提（簡略化）: 等価・現金投資ベース。持ち玉比率・交換ギャップ・出玉変動は未考慮
    （非等価では実効ボーダーが上がるぶん楽観側に出るので、borderline 側で吸収すること）。
    """
    R, B = spins_per_1k, borderline
    if R <= 0 or B <= 0:
        raise ValueError("回転率・ボーダーは正の値である必要があります")
    invest = yen_per_unit * total_spins / R
    ev = invest * (R - B) / B
    return {
        "spins_per_1k": R,
        "borderline": B,
        "total_spins": total_spins,
        "invest_yen": invest,
        "expected_yen": ev,
        "ev_ratio": (R - B) / B,             # 期待収支 / 投資
        "ev_per_1k_invest": yen_per_unit * (R - B) / B,
        "play": R > B,
    }
