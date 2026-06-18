"""ホール設定分布の推定（M6 学習）。

各台の観測を「設定の混合分布からの multinomial サンプル」とみなし、
混合重み π（＝ホールがその機種に入れている設定の分布）を EM で推定する。
各設定の成分パラメータ（設定別確率）は機種JSONで固定なので、未知は π のみ。

  E-step: 台 i・設定 k の責任度 r_ik ∝ π_k · L(obs_i | k)
  M-step: π_k = mean_i r_ik

得られた π は、その機種・その期間のホール傾向。次回推定の情報事前として infer --prior に使える。
これは『ベースライン比較』の事前（データ無視・事前のみ）を経験的に求める手段でもある。
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence

from .likelihood import event_probs_for_setting, log_likelihood


def estimate_hall_prior(
    model: Mapping,
    observations: Sequence[Mapping],
    iters: int = 100,
    tol: float = 1e-9,
    init: Optional[Mapping[int, float]] = None,
) -> Dict:
    """観測セッション群からホールの設定分布 π を EM 推定する。

    Args:
        model: load_machine() の戻り値。
        observations: [{"N":..., "counts":{...}}, ...]。machine/label は無視。
        iters: 最大反復数。
        tol: 収束判定（π の L1 変化量）。
        init: π の初期値。None なら一様。

    Returns:
        {"prior": {setting: prob}, "iterations": int, "log_likelihood": float,
         "n_sessions": int}
    """
    settings: List[int] = list(model["settings"])
    n = len(observations)
    if n == 0:
        raise ValueError("observations is empty")

    # 各台・各設定の対数尤度を前計算（成分パラメータは固定なので一度だけ）。
    # 各台は観測されたイベントだけで尤度を組む（未観測は "other" に吸収）。
    loglik = []  # loglik[i][k]
    for obs in observations:
        N = obs["N"]
        counts = obs.get("counts", {})
        events = set(counts.keys()) if counts else None
        row = {k: log_likelihood(N, counts, event_probs_for_setting(model, k, events))
               for k in settings}
        loglik.append(row)

    pi = dict(init) if init else {k: 1.0 / len(settings) for k in settings}

    total_ll = -math.inf
    used_iters = 0
    for it in range(1, iters + 1):
        used_iters = it
        # E-step: 責任度（log 領域で log-sum-exp 正規化）
        resp = []
        total_ll = 0.0
        for row in loglik:
            log_unnorm = {
                k: (math.log(pi[k]) + row[k]) if pi[k] > 0 else -math.inf
                for k in settings
            }
            m = max(log_unnorm.values())
            exp = {k: math.exp(v - m) for k, v in log_unnorm.items()}
            z = sum(exp.values())
            total_ll += m + math.log(z)  # この台の周辺対数尤度を加算
            resp.append({k: exp[k] / z for k in settings})

        # M-step
        new_pi = {k: sum(r[k] for r in resp) / n for k in settings}

        delta = sum(abs(new_pi[k] - pi[k]) for k in settings)
        pi = new_pi
        if delta < tol:
            break

    return {
        "prior": pi,
        "iterations": used_iters,
        "log_likelihood": total_ll,
        "n_sessions": n,
    }
