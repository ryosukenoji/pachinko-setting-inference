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
