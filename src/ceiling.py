"""軸B: 天井・ゾーン狙いの期待値計算（決定論的 +EV / design_proposal §3 C）。

設定看破（軸A）と違い隠れ変数を推定しない。公表の天井スペックと、台に表示された
現在ゲーム数から、残りEVを引き算と期待値で出す。

  残りゲーム数 = 天井G − 現在G
  期待収支(円) = 期待リワード − 期待投資

2モデル:
  - conservative（hit_rate なし）: 天井までブン回す前提。早い当たりを無視＝EVの下限。
  - early-hit（hit_rate あり）: 通常時初当りを幾何分布で織り込む。最初の当たり or 天井で止める想定。
    E[消化G] = (1 − (1−p)^remaining) / p,  P(天井到達) = (1−p)^remaining
"""

from __future__ import annotations

from typing import Dict, Optional


def ceiling_ev(
    ceiling_games: int,
    current_games: int,
    benefit_yen: float,
    yen_per_game: float,
    hit_rate: Optional[float] = None,
    early_hit_yen: float = 0.0,
) -> Dict:
    """現在ゲーム数から天井狙いの期待収支を返す。

    Args:
        ceiling_games: 天井（保証される規定ゲーム数）。
        current_games: 現在のゲーム数（台表示）。
        benefit_yen: 天井到達時の恩恵の円価値（恩恵枚数 × 換金レート）。
        yen_per_game: 通常時1Gあたりの投資（= 1000 / コイン持ちG）。
        hit_rate: 通常時初当り確率（1ゲームあたり）。None なら conservative。
        early_hit_yen: 早い当たり（通常当たり）の円価値。early-hit モデルで使用。
    """
    remaining = ceiling_games - current_games
    if remaining <= 0:
        return {
            "model": "past-ceiling",
            "remaining_games": remaining,
            "note": "現在Gが天井以上。スペック/表示を確認（即当たり圏 or 別カウンタ）。",
            "expected_yen": 0.0,
            "play": False,
        }

    if hit_rate is None or hit_rate <= 0:
        invest = remaining * yen_per_game
        ev = benefit_yen - invest
        return {
            "model": "conservative",
            "remaining_games": remaining,
            "expected_games": remaining,
            "invest_yen": invest,
            "reward_yen": benefit_yen,
            "prob_reach_ceiling": 1.0,
            "expected_yen": ev,
            "play": ev > 0,
        }

    p = hit_rate
    p_ceiling = (1.0 - p) ** remaining
    expected_games = (1.0 - p_ceiling) / p
    invest = expected_games * yen_per_game
    p_early = 1.0 - p_ceiling
    reward = early_hit_yen * p_early + benefit_yen * p_ceiling
    ev = reward - invest
    return {
        "model": "early-hit",
        "remaining_games": remaining,
        "expected_games": expected_games,
        "invest_yen": invest,
        "reward_yen": reward,
        "prob_reach_ceiling": p_ceiling,
        "expected_yen": ev,
        "play": ev > 0,
    }


def breakeven_current(
    ceiling_games: int,
    benefit_yen: float,
    yen_per_game: float,
    hit_rate: Optional[float] = None,
    early_hit_yen: float = 0.0,
) -> Optional[int]:
    """+EVで拾える『現在ゲーム数の下限（狙い目ボーダー）』を返す。

    これ以上回っている捨て台なら打つ価値がある、という実戦の境界。
    EV>0 になる最小の current_games を 1G 刻みで探す。無ければ None。
    """
    for cur in range(0, ceiling_games):
        res = ceiling_ev(ceiling_games, cur, benefit_yen, yen_per_game,
                         hit_rate=hit_rate, early_hit_yen=early_hit_yen)
        if res.get("play"):
            return cur
    return None
