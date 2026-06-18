"""実験: ぶどうカウントの有無で、設定推定がどれだけ変わるか。

クロール可能な実戦データサイト（rx7038型）は BIG/REG/総回転は出すが『ぶどう回数』が
無い。本実験は「ぶどう込み（手入力）」vs「ぶどう抜き（クロール相当=BIG/REGのみ）」を
3つの指標で比較し、『どの機種ならクロールデータだけで戦えるか』を数値で出す。

  指標1: 高設定(4-6) vs 低設定(1-3) の判別精度 vs 回転数  ← 打つ/やめる に直結
  指標2: 上記を90%精度にするのに必要な回転数
  指標3: 固定N=8000 での全6設定 top-1 的中率（ぶどうの限界的価値）

参考: 設定4 vs 5（隣接中間設定の厳密ID）も併記。

    python3 scripts/exp_budou_value.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import posterior as post_mod  # noqa: E402
from src import validate as val_mod  # noqa: E402

MACHINES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "machines")

MACHINES = ["my_juggler_v", "im_juggler_ex"]
CONDITIONS = [
    ("ぶどう込み", None),
    ("ぶどう抜き", {"BIG", "REG"}),
]
CANDIDATES = (2000, 4000, 6000, 8000, 10000, 15000, 25000, 40000)
TRIALS = 400
FIXED_N = 8000
SEED = 0
HIGH = {4, 5, 6}  # 機械割>100 の高設定群


def pct(x):
    return f"{x * 100:.1f}%"


def high_low_accuracy(model, N, events, trials, seed):
    """真設定を1〜6から一様抽出→推定→『高設定群か』の2択判別精度。

    pred_high = P(設定∈{4,5,6}|data) > 0.5。true_high = 真設定>=4。
    これは『打つ/やめる』判定（高設定なら+EV傾向）に直結する量。
    """
    settings = list(model["settings"])
    rng = random.Random(seed)
    correct = 0
    for _ in range(trials):
        true_k = rng.choice(settings)
        obs = val_mod.simulate_session(model, true_k, N, rng)
        post = post_mod.compute_posterior(model, obs, events=events)
        p_high = sum(post[k] for k in settings if k in HIGH)
        if (p_high > 0.5) == (true_k in HIGH):
            correct += 1
    return correct / trials


def main():
    print(f"候補N={CANDIDATES}  試行={TRIALS}  N(top1)={FIXED_N}  seed={SEED}\n")

    data = {}  # (machine, cond_label) -> dict
    for mid in MACHINES:
        model = post_mod.load_machine_by_id(MACHINES_DIR, mid)
        name = model.get("machine", mid)
        for label, events in CONDITIONS:
            curve = [high_low_accuracy(model, N, events, TRIALS, SEED) for N in CANDIDATES]
            req = next((CANDIDATES[i] for i, a in enumerate(curve) if a >= 0.90), None)
            nc = val_mod.negative_control(model, N=FIXED_N, trials_per_setting=TRIALS // 6,
                                          seed=SEED, events=events)
            rs45 = val_mod.required_samples(model, 4, 5, target_accuracy=0.90,
                                            candidates=CANDIDATES, trials=TRIALS // 2,
                                            seed=SEED, events=events)
            data[(name, label)] = {
                "curve": curve, "req90": req,
                "top1": nc["overall_top1_accuracy"],
                "acc45": rs45["curve"][-1]["accuracy"],
            }

    print("=== サマリ ===")
    h = f"{'機種':<16}{'条件':<12}{'高/低90%必要N':>16}{'高/低@8000':>12}{'top1@8000':>12}{'4v5@40k':>10}"
    print(h)
    print("-" * len(h))
    for mid in MACHINES:
        name = post_mod.load_machine_by_id(MACHINES_DIR, mid).get("machine", mid)
        for label, _ in CONDITIONS:
            d = data[(name, label)]
            req_s = f"{d['req90']:,}" if d["req90"] is not None else f">{CANDIDATES[-1]:,}"
            n8 = d["curve"][CANDIDATES.index(FIXED_N)] if FIXED_N in CANDIDATES else None
            n8_s = pct(n8) if n8 is not None else "-"
            print(f"{name:<16}{label:<12}{req_s:>16}{n8_s:>12}{pct(d['top1']):>12}{pct(d['acc45']):>10}")

    print("\n=== 高設定(4-6) vs 低設定(1-3) 判別精度カーブ ===")
    for mid in MACHINES:
        name = post_mod.load_machine_by_id(MACHINES_DIR, mid).get("machine", mid)
        print(f"\n[{name}]")
        print(f"  {'N':>8}" + "".join(f"{lab:>14}" for lab, _ in CONDITIONS) + f"{'差(pt)':>10}")
        for i, N in enumerate(CANDIDATES):
            a_with = data[(name, CONDITIONS[0][0])]["curve"][i]
            a_without = data[(name, CONDITIONS[1][0])]["curve"][i]
            line = f"  {N:>8,}{pct(a_with):>14}{pct(a_without):>14}{(a_with - a_without) * 100:>+9.1f}"
            print(line)


if __name__ == "__main__":
    main()
