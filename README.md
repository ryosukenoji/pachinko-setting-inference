# Pachinko / Pachislot Setting Inference

パチンコ・パチスロの**隠れた設定（設定1〜6 / 釘）を、観測データ（回転数・大当り・小役）から推定**し、長期EVで「打つ/やめる」を判断するツールの構想・検証プロジェクト。

- 構造案: [`docs/design_proposal.md`](docs/design_proposal.md)
- 出自: [Loto6 Particle Drift Model](https://github.com/ryosukenoji/loto6-particle-drift) の検証規律の転用先

> ロトは「既知の固定分布（信号なし）」で予測不能だった。パチンコ/スロットは「**未知だが固定の分布を、公表メニューから推定する**」問題で、ベイズ推論で正当に解ける。ドリフトの比喩は捨て、**検証パイプラインだけを引き継ぐ**。

---

## 1. この検証パイプラインを使う

Loto6プロジェクトで作った「**本物の勝ちか、データを拷問した自白か**」を見分ける規律をそのまま適用する。設定看破ツールは"それっぽい数字"を返すため、これが無いと自己欺瞞になる。

| 規律 | 本プロジェクトでの適用 |
|---|---|
| 事前宣言 | 主指標（例: EV判定の的中率）と成功閾値を実装前に固定 |
| ベースライン比較 | 「データ無視・事前のみ」と比較し、データ込みが上回るか |
| ネガティブコントロール | **既知設定でシミュレートしたデータ**を投入し、事後が真の設定に収束するか |
| キャリブレーション | 「設定6が80%」の台が長期で本当に80%か（予測確率≈実現頻度） |
| リークフリー | 推定は観測時点までのデータのみ。未来の結果を混ぜない |
| 検出力 / MDE | 設定5と4の分離に必要な回転数を算出し、少サンプルの過信を防ぐ |

→ 詳細は [`docs/design_proposal.md`](docs/design_proposal.md) §4。

---

## 2. ツールとしての完成までの TODO

| Milestone | 内容 | 状態 |
|---|---|---|
| M0 | 構造案・本README | done |
| M1 | **尤度テーブル整備** — ジャグラー系3機種（[マイジャグラーV](data/machines/my_juggler_v.json)・[アイムジャグラーEX](data/machines/im_juggler_ex.json)・[ファンキージャグラー2](data/machines/funky_juggler2.json)）を2ソース照合 `cross-checked`。版(`generation`)必須・整合性検証器あり。横展開リストは [`docs/machine_catalog.md`](docs/machine_catalog.md) | done |
| M2 | **ベイズ事後エンジン** — multinomial 尤度 × 事前 → 設定事後分布（[`src/likelihood.py`](src/likelihood.py) / [`src/posterior.py`](src/posterior.py)） | done |
| M3 | **EV判定** — 事後 × 公表機械割 → 打つ/やめる。パチンコは回転率→ボーダー判定（[`src/ev.py`](src/ev.py)） | done |
| M4 | **検証** — ネガコン・キャリブレーション・必要サンプル・EVバックテスト（[`src/validate.py`](src/validate.py)、Loto6の `evaluation.py` 移植） | done |
| M5 | **データ取得** — 手入力CLI / 観測CSV取込（[`src/dataio.py`](src/dataio.py)・`batch`コマンド） | done（手入力＋CSV取込。自動収集は規約確認後・未着手） |
| M6 | **複数機種対応・ホール事前分布の学習** — 第2機種 [`im_juggler_ex.json`](data/machines/im_juggler_ex.json)、EM学習（[`src/learn.py`](src/learn.py)）、外部事前 `--prior` | done（自動収集除く） |

最小で価値が出る **M1〜M4**（1機種で「推定が正しいことを検証済み」の状態）に加え、M5・M6（CSV取込・複数機種・ホール事前学習）まで実装済み。依存ライブラリなし（Python標準ライブラリのみ）。

### 使い方

```bash
# 観測から設定事後 + EV判定（手入力）
python3 -m src.cli infer --machine data/machines/my_juggler_v.json \
    --games 6000 --BIG 24 --REG 21 --budou 1010

# 円建て期待収支 + 収支ブレ幅（あと5000G回す想定 / 3枚掛け / 等価20円）
python3 -m src.cli infer --machine data/machines/my_juggler_v.json \
    --games 4000 --BIG 17 --REG 16 --budou 690 \
    --play-games 5000 --bet 3 --coin-yen 20
#   期待収支(長期平均) +12,034円 / 設定別 -9,000(設定1)〜+28,200円(設定6)
#   収支ブレ幅(設定不確実性＋短期分散/ヒキ, モンテカルロ):
#     中央値 +11,368円 / 50%帯 -5,673〜+29,001 / 90%帯 -28,713〜+55,881円 / プラス確率67%
#   ※ 短期はヒキの分散が支配的。EVは長期平均で、1セッションの実収支は帯の中で大きく振れる

# パチンコ: 回転率 → ボーダー判定（+ 予定回転数で円建て期待収支）
python3 -m src.cli border --rate 19.5 --border 18.2 --spins 4000
#   → 投資 205,128円 / 期待収支 +14,652円 (+7.1%)  ※等価・現金投資前提

# ホール内の複数台を観測CSVから一括推定し、EVエッジ順にランキング
python3 -m src.cli batch --csv tests/fixtures/observations_sample.csv

# 観測CSV(単一機種) → ホール設定分布 π を EM 推定（次回の情報事前に流用可）
python3 -m src.cli learn --csv <観測.csv> --out data/priors/hall.json
python3 -m src.cli infer --machine data/machines/my_juggler_v.json \
    --games 5200 --BIG 12 --REG 9 --budou 820 --prior data/priors/hall.json

# 回収に必要なG数（期待ベースの逆算）: 2万円を取り返すには？
python3 -m src.cli infer --machine data/machines/my_juggler_v.json \
    --games 4000 --BIG 17 --REG 16 --budou 690 --recover-yen 20000
#   → 期待機械割104% / 1Gあたり+2.41円 / 必要 約8,310G  ※期待ベース。短期は分散支配

# 設定を固定して毎日打ち続けたら（長期EV vs 分散の可視化）
python3 -m src.cli daily --machine data/machines/my_juggler_v.json --setting 6
#   → 設定6: 1日+45,120円, 95%プラス到達 約1日, 365日で+1,646万円(90%帯つき)
python3 -m src.cli daily --machine data/machines/my_juggler_v.json --setting 3
#   → 設定3(99.9%): 1日の標準偏差±2.8万円に隠れるが、プラス確率は1年で49%→37%へ沈む

# 登録機種の一覧 + テーブル整合性チェック
python3 -m src.cli tables

# 検証スイート（ネガコン / 必要サンプル / キャリブレーション / EVバックテスト）
python3 -m src.cli validate --machine data/machines/my_juggler_v.json

# テスト（39件・全機種テーブルの整合性チェック含む）
python3 -m unittest discover -s tests -v
```

観測CSVの形式（ヘッダ必須・機種ごとに列名可変）:

```csv
machine,label,games,BIG,REG,budou
my_juggler_v,A台,6000,24,21,1010
im_juggler_ex,C台,6000,23,23,930
```

> `learn` のEM学習は多数セッションで初めて意味を持つ（少数だと退化解になる）。`--prior` は「データ無視・事前のみ」のベースライン較正にも使える。

### 実装で確認された事実（検証規律の出力）

- **推定は機能する** `[confirmed: ネガコン]`: 大量回転（≳8000G）では真設定へ収束。設定6は top-1 的中が最も高い。
- **隣接設定の分離は重い** `[confirmed: 必要サンプル分析]`: 設定4 vs 5 は2万回転でも95%分離に届かない。ぶどう以外の設定差が小さいため。**朝一・少サンプルでの確信は過信** — これを定量化するのが M4 の役割。
- **判定は収益的** `[confirmed: EVバックテスト]`: 「打つ」と判定した台の真の平均機械割 ≈ 100.2% で、全台無差別（≈99%）を上回る。
- **テーブル誤差は結論を反転させる** `[confirmed: 2026-06-18 裏取り]`: マイジャグラーVの初期テーブル（REG列が1設定ずれ）では「ぶどう手入力の価値 +6〜8pt」だったが、2ソース照合で修正したらその価値はほぼ消失（→ クロールのBIG/REGで代替可）。固定パラメータ＝尤度テーブルの正確性が全推定の土台。詳細は [`docs/experiments/2026-06-18-budou-value.md`](docs/experiments/2026-06-18-budou-value.md)。
- 尤度テーブルの裏取り状況: **ジャグラー系3機種を2ソース照合済み**（`cross-checked`）。BIG/REG/機械割は公式で固まるが、**ぶどうは全機種とも公式非公表の実測推定**（サイト間でばらつく）。アイムEXのぶどうは「設定1-5ほぼ無差・設定6のみ優遇」=設定6フラグ専用。横展開は公式(BIG/REG)主信号のジャグラー系が最優先（[`docs/machine_catalog.md`](docs/machine_catalog.md)）。版の取り違え防止に `generation` を必須化し、`python3 -m src.cli tables` / `tests/test_tables.py` で整合性を機械チェック。

---

## 3. 不確定要素（リスク）

実測・調査で確認したものを含む。

- **機種依存で「効かない」** `[confirmed: 調査]`: ツールの効きは機種次第。設定差が小さい/設定差のある観測量が少ない機種では事後がほとんど動かない。→ 効く機種に限定して始める。
- **推定分散**: 設定の分離には大量回転が必要。朝一など少サンプルでは事後が広く、確信できない（M4の検出力分析で定量化）。
- **尤度テーブルの正確性** `[unverified]`: 設定別確率は攻略サイトの解析値依存。一次ソース（メーカー公表/大量実測）との突き合わせが必要。
- **データ取得の合法性/ToS** `[unverified]`: 後述。スクレイピングは各サイト規約に抵触し得る。手入力・公式アプリ利用が安全側。
- **ホールの対抗**: 高設定を入れない / 釘を毎日締める / 設定差の小さい機種選定。推定が正しくても**母数に高設定が無ければEVは出ない**。
- **規制で設定差圧縮** `[unverified: 最新仕様未確認]`: 近年（6号機・スマスロ）は設定差・天井が以前より薄い可能性。要確認。
- **運用コスト**: 時間集約・競争（他のアドバンテージプレイヤー）。ツールが正しくても時給化される。
- **本質的限界**: これは「次の1回転を当てる」ものではない。**長期EVの推定**であり、短期は分散に支配される。

---

## 4. データをどこから取得するか（調査済み）

### 4.1 観測データ（台ごとの N回転・大当り・履歴）

| ソース | 種別 | 取得性 | 備考 |
|---|---|---|---|
| 台の**データカウンター**（店内表示） | 一次 | 手入力 | 回転数・大当り回数・履歴。確実だが手動 |
| **データロボ サイトセブン**（site777.jp） | 集約・有料 | 月額〜¥324 | 過去7日の大当り履歴・グラフ・差枚。ホールコード単位。ツール連携でCSV書き出し可 `[confirmed]` |
| 台データオンライン / みんレポ(min-repo.com) / みんパチ / アナスロ等 | 集約・無料含む | サイト/アプリ | 差枚・履歴。カバー店舗はサイト依存 `[confirmed]` |
| 各ホール公式データサイト | 一次寄り | 公開 | 店舗による |

→ 小役（ぶどう等）の**回数は店データに出ないことが多い**ため、設定看破の主入力は**自分でカウント（手入力/カウンターアプリ）**が基本。店データは大当り確率・差枚の補助。

### 4.2 尤度（設定差の理論確率）

- 機種ごとの設定別 BIG/REG/小役確率は攻略サイト（DMM P-TOWN等）やメーカー情報に掲載 `[confirmed: 存在]`。
- 既存の設定判別アプリ（セブンノア、Aメソッド、App Store各種）は**まさにこのベイズ推定を実装済み** `[confirmed]`。→ 車輪の再発明を避け、本プロジェクトの差別化は「**検証規律の同梱**（ネガコン・キャリブレーション）」に置く。

### 4.3 法務・規約の注意

- スクレイピングは各データサイトの利用規約に抵触し得る `[unverified: 要規約確認]`。実装前に対象サイトのToSを確認すること。
- 安全側の初期方針: **手入力＋公式アプリ＋自分の許諾済みデータ**でMVPを作り、自動収集は規約確認後に限定導入。

---

## 出典（データソース調査, 2026-06-18 取得）

- [データロボ サイトセブン](http://m.site777.jp/f/A0100.do)
- [無料で見れるデータサイト・アプリまとめ](https://minimum-wage-slot.com/online-data-site-and-apps/)
- [データ確認できるサイトまとめ](https://sin-surobi.com/patisurotool/18348/)
- [みんレポ](https://min-repo.com/)
- [スクレイピングでデータ収集（手法解説）](https://diy-programming.site/gambling/aotagari-2/)
- [設定判別ツールはベイズ推定（効きは機種依存）](https://pachi778.com/pachislot-tool-effectiveness.html)
- [ジャグラー設定推測（ぶどう確率の設定差）](https://p-town.dmm.com/specials/2278)
- [設定判別ツール セブンノア](https://seven.noor.jp/)

> 注意: 本プロジェクトは長期EVの推定であり、短期の勝敗を保証しない。ギャンブルは自己責任・余剰資金の範囲で。
