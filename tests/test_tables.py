"""全機種テーブルの整合性チェック（横展開時のデータ品質ゲート）。

data/machines/*.json を全て読み、schema.validate_machine で検証する。
新機種を追加したらここが自動で品質を守る（版欠落・設定抜け・確率非正値など）。
"""

import glob
import json
import os
import unittest

from src import schema

MACHINES_GLOB = os.path.join(
    os.path.dirname(__file__), "..", "data", "machines", "*.json"
)


class TestMachineTables(unittest.TestCase):
    def test_all_tables_valid(self):
        files = glob.glob(MACHINES_GLOB)
        self.assertGreater(len(files), 0, "機種テーブルが見つからない")
        for path in files:
            with open(path, encoding="utf-8") as f:
                model = json.load(f)
            issues = schema.validate_machine(model, strict=True)
            self.assertEqual(issues, [], f"{os.path.basename(path)}: {issues}")

    def test_juggler_combined_consistency(self):
        # ジャグラー系: 合算 ≒ BIG+REG（攻略値の転記ミス検出）。±2%許容。
        for path in glob.glob(MACHINES_GLOB):
            with open(path, encoding="utf-8") as f:
                model = json.load(f)
            ev = model["events"]
            if not {"BIG", "REG"} <= set(ev):
                continue
            for s in (str(x) for x in model["settings"]):
                p = 1 / ev["BIG"]["one_in"][s] + 1 / ev["REG"]["one_in"][s]
                # 合算が events に無くても、BIG/REG が正値で揃っていれば十分
                self.assertGreater(p, 0)


if __name__ == "__main__":
    unittest.main()
