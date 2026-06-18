"""事後分布 → EV判定（スロット）と、回転率 → ボーダー判定（パチンコ）。

design_proposal.md §3 の A（設定看破→EV）と B（ボーダー理論）に対応。
本モジュールは『予測』ではなく、推定済み分布／観測済み回転率からの決定論的な判定のみ。
"""

from __future__ import annotations

from typing import Dict, Mapping


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
