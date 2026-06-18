"""観測データの入出力（手入力 CSV → 推定入力）。

CSV フォーマット（ヘッダ必須）:
    machine,label,games,BIG,REG,budou
    my_juggler_v,A台,6000,24,21,1010
    my_juggler_v,B台,5200,12,9,820

  - machine : data/machines/<machine>.json のファイル名 stem（または machine フィールド一致）
  - label   : 台の識別名（任意・空可）
  - games   : 総ゲーム数 N（必須）
  - それ以外の列 : イベント観測回数。空セルは「未観測（0扱いしない＝counts に入れない）」。

機種ごとに観測量の列名は異なってよい（機種JSONの events キーと突き合わせる）。
"""

from __future__ import annotations

import csv
from typing import Dict, List

_META_COLS = {"machine", "label", "games", "n", "total"}


def read_observations_csv(path) -> List[Dict]:
    """観測CSVを読み、推定入力レコードのリストに変換する。

    Returns:
        [{"machine": str, "label": str, "N": int, "counts": {event: int}}, ...]
    """
    records: List[Dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return records
        event_cols = [c for c in reader.fieldnames if c and c.strip().lower() not in _META_COLS]
        for lineno, row in enumerate(reader, start=2):
            machine = (row.get("machine") or "").strip()
            if not machine:
                raise ValueError(f"{path}:{lineno}: 'machine' 列が空です")
            games_raw = row.get("games") or row.get("N") or row.get("total")
            if games_raw is None or str(games_raw).strip() == "":
                raise ValueError(f"{path}:{lineno}: 'games' 列が空です")
            N = int(float(games_raw))
            counts: Dict[str, int] = {}
            for col in event_cols:
                val = row.get(col)
                if val is not None and str(val).strip() != "":
                    counts[col] = int(float(val))
            records.append(
                {
                    "machine": machine,
                    "label": (row.get("label") or "").strip(),
                    "N": N,
                    "counts": counts,
                }
            )
    return records
