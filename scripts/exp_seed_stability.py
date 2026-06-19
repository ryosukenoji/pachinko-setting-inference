"""実験: 検証結果がseedに対して安定か（モンテカルロ誤差の確認）。

これまでの実験は seed=0 単一だった。結論（高/低判別精度・ぶどうの寄与）が
乱数seedで揺れていないかを、複数seedで negative_control の top-1 的中率を回して
平均±標準偏差で確認する。標準偏差が小さければ単一seedの結論を信用してよい。

    python3 scripts/exp_seed_stability.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import posterior as post_mod  # noqa: E402
from src import validate as val_mod  # noqa: E402

MACHINES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "machines")
MACHINES = ["my_juggler_v", "im_juggler_ex"]
CONDITIONS = [("ぶどう込み", None), ("ぶどう抜き", {"BIG", "REG"})]
SEEDS = list(range(10))
N = 8000
TRIALS = 60  # /設定/seed


def mean_std(xs):
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, var ** 0.5


def main():
    print(f"N={N}  試行={TRIALS}/設定  seeds={SEEDS}\n")
    h = f"{'機種':<16}{'条件':<12}{'top1平均':>10}{'±標準偏差':>12}{'最小〜最大':>16}"
    print(h)
    print("-" * len(h))
    for mid in MACHINES:
        model = post_mod.load_machine_by_id(MACHINES_DIR, mid)
        name = model.get("machine", mid)
        for label, events in CONDITIONS:
            accs = [
                val_mod.negative_control(model, N=N, trials_per_setting=TRIALS,
                                         seed=s, events=events)["overall_top1_accuracy"]
                for s in SEEDS
            ]
            m, sd = mean_std(accs)
            print(f"{name:<16}{label:<12}{m * 100:>9.1f}%{sd * 100:>11.2f}pt"
                  f"{min(accs) * 100:>8.1f}〜{max(accs) * 100:.1f}%")


if __name__ == "__main__":
    main()
