"""事前 × 尤度 → 設定の事後分布。

    事後(設定=k | データ) ∝ 事前(k) × 尤度(データ | k)

対数領域で計算し、log-sum-exp で安定に正規化する（リークなし・単一経路）。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from .likelihood import event_probs_for_setting, log_likelihood


def load_machine(path) -> dict:
    """機種設定差テーブル（JSON）を読み込む。"""
    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)
    if "settings" not in model or "events" not in model:
        raise ValueError(f"invalid machine file: {path}")
    return model


def load_machine_by_id(machines_dir, machine_id: str) -> dict:
    """data/machines/<machine_id>.json を読み込む（CSV の machine 列から解決）。"""
    p = Path(machines_dir) / f"{machine_id}.json"
    if not p.exists():
        raise FileNotFoundError(f"machine '{machine_id}' not found under {machines_dir}")
    return load_machine(p)


def parse_prior(spec: Optional[str], settings: List[int]) -> Optional[Dict[int, float]]:
    """事前指定をパースする。

    spec は (a) JSON 文字列 '{"1":0.3,...}'、(b) JSON ファイルパス（{"prior":{...}} か
    マッピング直書き）のいずれか。None なら None（呼び出し側で一様事前）。
    キーは設定番号(int)へ正規化し、合計1に再正規化する。
    """
    if spec is None:
        return None
    text = spec
    p = Path(spec)
    if p.exists():
        text = p.read_text(encoding="utf-8")
    obj = json.loads(text)
    if isinstance(obj, dict) and "prior" in obj:
        obj = obj["prior"]
    prior = {int(k): float(v) for k, v in obj.items()}
    for k in settings:
        prior.setdefault(k, 0.0)
    total = sum(prior.values())
    if total <= 0:
        raise ValueError("prior sums to <= 0")
    return {k: prior[k] / total for k in settings}


def uniform_prior(settings: List[int]) -> Dict[int, float]:
    n = len(settings)
    return {k: 1.0 / n for k in settings}


def _normalize_log(log_unnorm: Mapping[int, float]) -> Dict[int, float]:
    """log-sum-exp 正規化。log の非正規化スコア -> 確率。"""
    m = max(log_unnorm.values())
    exp = {k: math.exp(v - m) for k, v in log_unnorm.items()}
    z = sum(exp.values())
    return {k: v / z for k, v in exp.items()}


def compute_posterior(
    model: Mapping,
    observation: Mapping,
    prior: Optional[Mapping[int, float]] = None,
    events=None,
) -> Dict[int, float]:
    """観測（N と各イベント回数）から設定事後分布を返す。

    Args:
        model: load_machine() の戻り値。
        observation: {"N": int, "counts": {event_name: int, ...}}。
        prior: 設定 -> 事前確率。None なら一様事前。
        events: 尤度に使う観測量の限定（None なら機種の全イベント）。
            例: {"BIG","REG"} で『ぶどう未観測』を再現。

    Returns:
        設定(int) -> 事後確率。合計1。
    """
    settings: List[int] = model["settings"]
    if prior is None:
        prior = uniform_prior(settings)

    N = observation["N"]
    counts = observation.get("counts", {})

    log_unnorm: Dict[int, float] = {}
    for k in settings:
        p_k = prior.get(k, 0.0)
        if p_k <= 0.0:
            log_unnorm[k] = -math.inf
            continue
        ll = log_likelihood(N, counts, event_probs_for_setting(model, k, events))
        log_unnorm[k] = math.log(p_k) + ll

    return _normalize_log(log_unnorm)


def map_setting(posterior: Mapping[int, float]) -> int:
    """最大事後確率（MAP）の設定。"""
    return max(posterior, key=posterior.get)


def summarize(model: Mapping, observation: Mapping, prior=None) -> dict:
    """事後分布 + MAP + 観測サマリをまとめて返す（CLI/レポート用）。"""
    post = compute_posterior(model, observation, prior)
    return {
        "machine": model.get("machine"),
        "N": observation["N"],
        "counts": dict(observation.get("counts", {})),
        "posterior": post,
        "map_setting": map_setting(post),
    }
