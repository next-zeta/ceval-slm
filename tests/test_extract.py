#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""extract.py 回归测试（不依赖 pytest）：

    python tests/test_extract.py

覆盖：显式标注、独占一行、中文表述、多候选取最后一次、全角字母、必须判 None 的边界。
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from extract import extract_answer, format_valid  # noqa: E402

CASES = [
    # (模型原始输出, 期望抽取结果, 期望规则)
    ("答案：B", "B", "answer_tag"),
    ("答案: C。", "C", "answer_tag"),
    ("答案是D", "D", "answer_tag"),
    ("最终答案：A", "A", "answer_tag"),
    ("B", "B", "standalone_letter"),
    ("B。", "B", "standalone_letter"),
    ("D\n", "D", "standalone_letter"),
    ("我认为答案是D", "D", "answer_tag"),
    ("A 不对，B 也不对。\n答案：C", "C", "answer_tag"),
    ("答案：A\n\n解析：……\n答案：D", "D", "answer_tag"),
    ("答案：Ｂ", "B", "answer_tag"),
    ("应选A", "A", "choose_verb"),
    ("我认为选 B 更合理", "B", "choose_verb"),
    # 以下必须判为格式无效（不猜）
    ("我不知道", None, "none"),
    ("", None, "none"),
    ("B. 因为该选项符合题意", None, "none"),
    ("选项 A、B、C、D 中有一个是对的", None, "none"),
]


def main() -> int:
    failed = 0
    for text, expect, expect_rule in CASES:
        got, rule = extract_answer(text)
        if got != expect or rule != expect_rule:
            failed += 1
            print("[FAIL] {!r}: got ({!r}, {!r}) expect ({!r}, {!r})".format(
                text, got, rule, expect, expect_rule))
    if format_valid("A") is not True or format_valid(None) is not False:
        failed += 1
        print("[FAIL] format_valid 判定异常")

    total = len(CASES) + 1
    if failed:
        print("{} / {} 个用例失败".format(failed, total))
        return 1
    print("全部 {} 个用例通过".format(total))
    return 0


if __name__ == "__main__":
    sys.exit(main())
