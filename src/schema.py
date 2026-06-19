"""機種設定差テーブルの整合性検証。

公式参照で横展開する際、版（世代）取り違え・設定抜け・確率の単調性崩れを
機械的に弾くためのバリデータ。CLI(`validate-tables`)とテストから使う。

スキーマ v2 の必須フィールド:
    machine, generation, schema, settings, events, payout
各 event は settings 全てに one_in を持つこと。payout も settings 全てを持つこと。
"""

from __future__ import annotations

from typing import Dict, List, Mapping

REQUIRED_TOP = ("machine", "generation", "schema", "settings", "events", "payout")
KNOWN_GENERATIONS = ("4号機", "5号機", "6号機", "スマスロ", "6.5号機")


def validate_machine(model: Mapping, *, strict: bool = False) -> List[str]:
    """機種テーブルの問題点リストを返す（空なら整合）。

    strict=True で、警告（単調性の崩れ等、誤りとは限らないもの）も含める。
    """
    issues: List[str] = []

    for key in REQUIRED_TOP:
        if key not in model:
            issues.append(f"必須フィールド欠落: {key}")
    if issues:
        return issues  # 構造が壊れていればこれ以上見ない

    gen = model["generation"]
    if gen not in KNOWN_GENERATIONS:
        issues.append(f"未知の generation: {gen!r}（{KNOWN_GENERATIONS} のいずれか想定）")

    settings = model["settings"]
    if not settings:
        issues.append("settings が空")
        return issues
    sset = [str(s) for s in settings]

    # events: 各イベントが全設定の one_in を持つか・正値か
    for name, spec in model["events"].items():
        one_in = spec.get("one_in", {})
        for s in sset:
            if s not in one_in:
                issues.append(f"event {name!r}: 設定{s} の one_in 欠落")
            elif one_in[s] <= 0:
                issues.append(f"event {name!r}: 設定{s} の one_in が非正値 ({one_in[s]})")

    # payout: 全設定を持つか
    payout = model["payout"]
    for s in sset:
        if s not in payout and int(s) not in payout:
            issues.append(f"payout: 設定{s} 欠落")

    # payout_coins があれば bet/BIG/REG を確認
    pc = model.get("payout_coins")
    if pc is not None:
        for k in ("BIG", "REG"):
            if k not in pc:
                issues.append(f"payout_coins: {k} 欠落")

    # provenance.status が cross-checked なら sources を要求
    prov = model.get("provenance", {})
    if prov.get("status") == "cross-checked" and not prov.get("sources"):
        issues.append("provenance.status=cross-checked だが sources が無い")

    if strict:
        # 機械割は設定が上がるほど高いのが通例（崩れていれば版/転記ミスを疑う）
        pays = [float(payout.get(s, payout.get(int(s)))) for s in sset]
        if pays != sorted(pays):
            issues.append(f"[warn] payout が設定順で単調増加でない: {pays}")

    return issues


def assert_valid(model: Mapping, name: str = "") -> None:
    issues = validate_machine(model)
    if issues:
        head = f"machine table invalid ({name}): " if name else "machine table invalid: "
        raise ValueError(head + "; ".join(issues))
