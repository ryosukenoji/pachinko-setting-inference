"""設定別 multinomial 対数尤度。

各遊技（1ゲーム）の結果を互いに排他なカテゴリ {BIG, REG, ぶどう, ..., other} の
1つとみなす multinomial モデル。設定 k に対する尤度は

    L(data | k) = multinomial係数 × Π_c p_c(k)^{x_c}

multinomial係数は設定 k に依存しないため事後の正規化で打ち消える。よって
対数尤度は Σ_c x_c · log p_c(k) のみを計算すれば事後比較には十分（定数項は省略）。

注: BIG/REG/ぶどうを「独立 binomial」と扱う流儀もあるが、1ゲームの結果は排他なので
multinomial が正しい。設定差のある観測量が1つ（binomial）の機種でも、other を補えば
本関数の特殊形として扱える。
"""

from __future__ import annotations

import math
from typing import Dict, Mapping

# log(0) を避けるためのフロア。p_other が極小・負になる退化ケースで使用。
_EPS = 1e-12


def event_probs_for_setting(model: Mapping, setting: int, events=None) -> Dict[str, float]:
    """機種モデルから、ある設定の各イベント発生確率 p = 1/one_in を取り出す。

    events に名前の集合/リストを渡すと、その観測量だけで尤度を組む。除外した
    イベントは自動的に 'other' バケツに吸収される（＝未観測カテゴリの正しい
    周辺化）。例: events={"BIG","REG"} で『ぶどうを観測しない』推定を再現できる。
    """
    s = str(setting)
    probs: Dict[str, float] = {}
    for name, spec in model["events"].items():
        if events is not None and name not in events:
            continue
        one_in = spec["one_in"][s]
        if one_in <= 0:
            raise ValueError(f"one_in must be > 0 for event {name!r}, setting {setting}")
        probs[name] = 1.0 / one_in
    if events is not None and not probs:
        raise ValueError(f"no matching events for filter {events!r}")
    return probs


def log_likelihood(N: int, counts: Mapping[str, int], event_probs: Mapping[str, float]) -> float:
    """設定固定下での multinomial 対数尤度（設定非依存の定数項は省略）。

    Args:
        N: 総ゲーム数（試行数）。
        counts: イベント名 -> 観測回数。event_probs に無いキーは無視。
        event_probs: イベント名 -> 1ゲームあたり発生確率。

    other カテゴリ（どのイベントにも該当しないゲーム）の確率と回数を
    p_other = 1 - Σ p_event, x_other = N - Σ x_event として補い、尤度に含める。
    """
    if N < 0:
        raise ValueError("N must be >= 0")
    obs_sum = 0
    p_sum = 0.0
    ll = 0.0
    for name, p in event_probs.items():
        x = counts.get(name, 0)
        if x < 0:
            raise ValueError(f"count for {name!r} must be >= 0")
        obs_sum += x
        p_sum += p
        if x > 0:
            ll += x * math.log(max(p, _EPS))

    x_other = N - obs_sum
    if x_other < 0:
        raise ValueError(
            f"observed event counts ({obs_sum}) exceed total games N ({N})"
        )
    p_other = 1.0 - p_sum
    if x_other > 0:
        ll += x_other * math.log(max(p_other, _EPS))
    return ll
