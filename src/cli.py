"""CLI: 設定推定・EV判定・検証を実行する。

例:
  # 観測から設定事後 + EV判定
  python -m src.cli infer --machine data/machines/my_juggler_v.json \\
      --games 5000 --BIG 18 --REG 14 --budou 800

  # パチンコ ボーダー判定
  python -m src.cli border --rate 19.5 --border 18.2

  # 検証スイート（ネガコン / 必要サンプル / キャリブレーション / EVバックテスト）
  python -m src.cli validate --machine data/machines/my_juggler_v.json
"""

from __future__ import annotations

import argparse
import json
import sys

from . import dataio as io_mod
from . import ev as ev_mod
from . import learn as learn_mod
from . import posterior as post_mod
from . import validate as val_mod


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def cmd_infer(args) -> int:
    model = post_mod.load_machine(args.machine)
    prior = post_mod.parse_prior(getattr(args, "prior", None), model["settings"])
    counts = {}
    for name in model["events"]:
        v = getattr(args, name, None)
        if v is not None:
            counts[name] = v
    observation = {"N": args.games, "counts": counts}

    # 観測したイベントだけで尤度を組む（未入力のイベントは "other" に吸収＝正しい周辺化）。
    # 例: 店カウンターの N/BIG/REG だけなら ぶどうは推定に使わない。
    events = set(counts.keys()) if counts else None

    post = post_mod.compute_posterior(model, observation, prior, events=events)
    summary = {
        "machine": model.get("machine"),
        "N": observation["N"],
        "counts": dict(counts),
        "posterior": post,
        "map_setting": post_mod.map_setting(post),
        "events_used": sorted(events) if events else "all",
    }
    if prior is not None:
        print(f"（情報事前を適用: {{ {', '.join(f'{k}:{_fmt_pct(v)}' for k, v in prior.items())} }}）")

    print(f"機種: {summary['machine']}")
    print(f"総ゲーム数 N={summary['N']}  観測={summary['counts']}")
    print("--- 設定事後分布 ---")
    for k in sorted(post):
        bar = "#" * int(round(post[k] * 40))
        print(f"  設定{k}: {_fmt_pct(post[k]):>6}  {bar}")
    print(f"MAP設定: {summary['map_setting']}")

    dec = ev_mod.slot_decision(post, model["payout"], args.threshold)
    print("--- EV判定 ---")
    print(f"  期待機械割: {dec['expected_payout']:.2f}%  (閾値 {dec['threshold']:.1f}%)")
    print(f"  判定: {'打つ (+EV)' if dec['play'] else 'やめ (-EV)'}  edge={dec['edge_pct']:+.2f}pt")

    yen = None
    if args.play_games:
        yen = ev_mod.slot_yen_ev(post, model["payout"], args.play_games,
                                 bet_per_game=args.bet, yen_per_coin=args.coin_yen)
        print(f"--- 円建て期待収支（あと {args.play_games:,}G 回す想定 / "
              f"{args.bet}枚掛け / {args.coin_yen:g}円per枚） ---")
        print(f"  期待収支: {yen['expected_yen']:+,.0f}円   "
              f"プラス収支確率: {_fmt_pct(yen['prob_plus'])}")
        b, w = yen["best"], yen["worst"]
        print(f"  設定不確実性による幅: 最悪 設定{w['setting']} {w['yen']:+,.0f}円 "
              f"〜 最良 設定{b['setting']} {b['yen']:+,.0f}円")
        print("  設定別内訳（事後 × 機械割 → 収支）:")
        for p in yen["per_setting"]:
            print(f"    設定{p['setting']}: {_fmt_pct(p['prob']):>6} × {p['payout']:.1f}% "
                  f"→ {p['yen']:+,.0f}円")
        print(f"  ※ {yen['caveat']}")

        if model.get("payout_coins"):
            dist = ev_mod.session_pnl_distribution(
                post, model, args.play_games, trials=args.trials,
                yen_per_coin=args.coin_yen, seed=args.seed)
            pp = dist["percentiles"]
            print(f"  --- 収支のブレ幅（設定不確実性＋短期分散/ヒキ, {dist['trials']:,}試行） ---")
            print(f"    中央値(p50): {pp['p50']:+,.0f}円")
            print(f"    50%予想帯(p25〜p75): {pp['p25']:+,.0f} 〜 {pp['p75']:+,.0f}円")
            print(f"    90%予想帯(p5〜p95): {pp['p5']:+,.0f} 〜 {pp['p95']:+,.0f}円")
            print(f"    プラス収支確率: {_fmt_pct(dist['prob_plus'])}")
            print(f"    ※ {dist['caveat']}")
            yen["distribution"] = dist

    if args.json:
        out = {"summary": summary, "decision": dec}
        if yen is not None:
            out["yen_ev"] = yen
        print(json.dumps(out, ensure_ascii=False, default=str))
    return 0


def cmd_border(args) -> int:
    dec = ev_mod.pachinko_border_decision(args.rate, args.border)
    print(f"実測回転率: {dec['spins_per_1k']:.2f} 回/¥1000   ボーダー: {dec['borderline']:.2f}")
    print(f"判定: {'打つ (+EV)' if dec['play'] else 'やめ (-EV)'}  margin={dec['margin']:+.2f}")
    if args.json:
        print(json.dumps(dec, ensure_ascii=False))
    return 0


def cmd_batch(args) -> int:
    """観測CSV → ホール内の全台を一括推定し、EVエッジ順にランキング。"""
    records = io_mod.read_observations_csv(args.csv)
    cache = {}
    rows = []
    for rec in records:
        mid = rec["machine"]
        if mid not in cache:
            cache[mid] = post_mod.load_machine_by_id(args.machines_dir, mid)
        model = cache[mid]
        prior = post_mod.parse_prior(args.prior, model["settings"])
        events = set(rec["counts"].keys()) if rec["counts"] else None
        post = post_mod.compute_posterior(model, rec, prior, events=events)
        dec = ev_mod.slot_decision(post, model["payout"], args.threshold)
        rows.append({
            "label": rec["label"] or "-",
            "machine": model.get("machine", mid),
            "N": rec["N"],
            "map": post_mod.map_setting(post),
            "ev": dec["expected_payout"],
            "edge": dec["edge_pct"],
            "play": dec["play"],
        })

    rows.sort(key=lambda r: r["edge"], reverse=True)
    print(f"観測 {len(rows)} 台  (閾値 {args.threshold:.1f}%)  ※エッジ降順")
    print(f"{'判定':<6}{'ラベル':<10}{'機種':<18}{'N':>7}{'MAP':>5}{'期待割':>9}{'edge':>9}")
    for r in rows:
        mark = "打つ" if r["play"] else "やめ"
        print(f"{mark:<6}{r['label']:<10}{r['machine']:<18}{r['N']:>7}{r['map']:>5}"
              f"{r['ev']:>8.2f}%{r['edge']:>+8.2f}")
    n_play = sum(1 for r in rows if r["play"])
    print(f"--- 打つ判定: {n_play} 台 / {len(rows)} 台 ---")
    return 0


def cmd_learn(args) -> int:
    """観測CSV（同一機種）→ ホール設定分布を EM 推定。infer/batch の --prior に流用可。"""
    records = io_mod.read_observations_csv(args.csv)
    machines = {r["machine"] for r in records}
    if len(machines) != 1:
        print(f"learn は単一機種のCSVのみ対応（検出: {sorted(machines)}）", file=sys.stderr)
        return 2
    mid = machines.pop()
    model = post_mod.load_machine_by_id(args.machines_dir, mid)
    result = learn_mod.estimate_hall_prior(model, records, iters=args.iters)
    prior = result["prior"]
    print(f"機種: {model.get('machine', mid)}   セッション数: {result['n_sessions']}   "
          f"反復: {result['iterations']}")
    print("--- 推定ホール設定分布 π ---")
    for k in sorted(prior):
        bar = "#" * int(round(prior[k] * 40))
        print(f"  設定{k}: {_fmt_pct(prior[k]):>6}  {bar}")
    payout = model["payout"]
    exp_pay = sum(prior[k] * float(payout[str(k)]) for k in prior)
    print(f"このπでの期待機械割（無作為に1台）: {exp_pay:.2f}%")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"machine": mid, "prior": {str(k): v for k, v in prior.items()}},
                      f, ensure_ascii=False, indent=2)
        print(f"事前を書き出し: {args.out}")
    return 0


def cmd_validate(args) -> int:
    model = post_mod.load_machine(args.machine)

    print("=== ネガティブコントロール（既知設定で生成→真設定へ収束するか） ===")
    nc = val_mod.negative_control(model, N=args.games, trials_per_setting=args.trials, seed=args.seed)
    print(f"  N={nc['N']}  chance={_fmt_pct(nc['chance_level'])}")
    for k, row in nc["per_setting"].items():
        print(f"  設定{k}: 真設定平均事後={_fmt_pct(row['mean_true_mass'])}  "
              f"top1的中={_fmt_pct(row['top1_accuracy'])}")
    print(f"  総合 top1的中率: {_fmt_pct(nc['overall_top1_accuracy'])}")

    print("\n=== 必要サンプル（設定4 vs 5 の分離, 目標95%） ===")
    rs = val_mod.required_samples(model, setting_a=4, setting_b=5,
                                  target_accuracy=0.95, trials=args.trials, seed=args.seed)
    print(f"  必要N: {rs['required_N']}")
    for row in rs["curve"]:
        print(f"    N={row['N']:>6}: 分離精度={_fmt_pct(row['accuracy'])}")

    print("\n=== キャリブレーション（最高設定の予測確率 ≈ 実現頻度, ECE→0が良い） ===")
    cal = val_mod.calibration(model, N=args.games, trials=args.cal_trials, seed=args.seed)
    print(f"  対象=設定{cal['target_setting']}  ECE={cal['ece']:.4f}")
    for row in cal["bins"]:
        print(f"    予測{_fmt_pct(row['mean_predicted']):>6} ≈ 実測{_fmt_pct(row['observed_freq']):>6}  "
              f"(n={row['count']})")

    print("\n=== EVバックテスト（『打つ判定』台の真の平均機械割 > 100% か） ===")
    bt = val_mod.ev_backtest(model, N=args.games, trials=args.cal_trials, seed=args.seed)
    print(f"  打った台数={bt['n_played']}  見送り={bt['n_skipped']}")
    pm = bt["played_mean_true_payout"]
    print(f"  打った台の真の平均機械割: {pm:.2f}%" if pm is not None else "  打った台なし")
    print(f"  ベースライン(全台無差別): {bt['baseline_play_all_mean_payout']:.2f}%")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="setting-inference", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("infer", help="観測から設定事後 + EV判定")
    pi.add_argument("--machine", required=True)
    pi.add_argument("--games", type=int, required=True, help="総ゲーム数 N")
    pi.add_argument("--BIG", type=int, default=None)
    pi.add_argument("--REG", type=int, default=None)
    pi.add_argument("--budou", type=int, default=None, help="ぶどう回数")
    pi.add_argument("--threshold", type=float, default=100.0, help="EV判定閾値(機械割%)")
    pi.add_argument("--prior", default=None,
                    help="情報事前: JSON文字列 or ファイルパス（learn の出力を流用可）")
    pi.add_argument("--play-games", type=int, default=None, dest="play_games",
                    help="これから回す予定G数。指定で円建て期待収支を出力")
    pi.add_argument("--bet", type=int, default=3, help="1G掛けコイン（既定3枚）")
    pi.add_argument("--coin-yen", type=float, default=20.0, dest="coin_yen",
                    help="換金レート 円/枚（既定20=等価）")
    pi.add_argument("--trials", type=int, default=20000,
                    help="収支ブレ幅モンテカルロの試行数（既定20000）")
    pi.add_argument("--seed", type=int, default=0, help="モンテカルロ乱数seed")
    pi.add_argument("--json", action="store_true")
    pi.set_defaults(func=cmd_infer)

    pb = sub.add_parser("border", help="パチンコ回転率→ボーダー判定")
    pb.add_argument("--rate", type=float, required=True, help="実測 ¥1000あたり回転数")
    pb.add_argument("--border", type=float, required=True, help="ボーダーライン")
    pb.add_argument("--json", action="store_true")
    pb.set_defaults(func=cmd_border)

    pba = sub.add_parser("batch", help="観測CSV → 複数台を一括推定しEV順にランキング")
    pba.add_argument("--csv", required=True)
    pba.add_argument("--machines-dir", default="data/machines", dest="machines_dir")
    pba.add_argument("--threshold", type=float, default=100.0)
    pba.add_argument("--prior", default=None, help="全台に適用する情報事前(JSON/パス)")
    pba.set_defaults(func=cmd_batch)

    pl = sub.add_parser("learn", help="観測CSV(単一機種) → ホール設定分布をEM推定")
    pl.add_argument("--csv", required=True)
    pl.add_argument("--machines-dir", default="data/machines", dest="machines_dir")
    pl.add_argument("--iters", type=int, default=100)
    pl.add_argument("--out", default=None, help="推定事前をJSONで書き出す先")
    pl.set_defaults(func=cmd_learn)

    pv = sub.add_parser("validate", help="検証スイートを実行")
    pv.add_argument("--machine", required=True)
    pv.add_argument("--games", type=int, default=5000, help="検証で使う N")
    pv.add_argument("--trials", type=int, default=200, help="ネガコン/必要サンプルの試行数/設定")
    pv.add_argument("--cal-trials", type=int, default=2000, dest="cal_trials",
                    help="キャリブレーション/バックテストの試行数")
    pv.add_argument("--seed", type=int, default=0)
    pv.set_defaults(func=cmd_validate)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
