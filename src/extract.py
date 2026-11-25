#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""答案抽取器：全组唯一实现（冻结文件）。

设计依据：《方案.md》§4.4。任何修改都必须全实验重新评分并在文档说明原因。

优先级（自上而下，命中即停；同一规则命中多次时取**最后一次**）：

1. ``answer_tag``      —— ``答案：X`` / ``答案: X`` / ``答案是X`` 等显式标注；
2. ``standalone_letter`` —— 独占一行的单字母（可带句号），应对模型只吐 "B" 的情况；
3. ``choose_verb``     —— ``选 X`` / ``应选 X`` 等中文表述。

都不命中时返回 ``None``：该题计为**错误**，同时计入**格式无效**。

刻意不做的事：
- 不从推理过程里"猜"答案（例如 ``A 不对，B 才对`` 不会因为出现 A 而判 A）；
- 不把选项内容（"B. 人民法院"）当成答案字母；
- 不因为输出里提到多个字母就把全部当成候选——只认最终答案位置。
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

#: 合法的答案字母集合
LETTERS = "ABCD"

#: 全角字母 -> 半角（模型偶尔输出 "Ｂ"）
_FULLWIDTH = str.maketrans("ＡＢＣＤ", "ABCD")

RULE_ANSWER_TAG = "answer_tag"
RULE_STANDALONE = "standalone_letter"
RULE_CHOOSE_VERB = "choose_verb"
RULE_NONE = "none"

#: 抽取规则（顺序即优先级）。注意第 2 条要求整行只有字母，避免把 "B. 因为…" 误判成最终答案。
_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    (RULE_ANSWER_TAG, re.compile(r"答案\s*(?:是|为)?\s*[:：]?\s*([ABCD])")),
    (RULE_STANDALONE, re.compile(r"(?:^|\n)[ \t]*([ABCD])[ \t]*[。.]?[ \t]*(?=\n|$)", re.M)),
    (RULE_CHOOSE_VERB, re.compile(r"(?:应选|选)\s*([ABCD])")),
)


def normalize_text(text: str) -> str:
    """把全角字母换成半角，并统一换行符；其余字符原样保留。"""
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return text.translate(_FULLWIDTH)


def extract_answer(text: str) -> Tuple[Optional[str], str]:
    """从模型原始输出中抽取最终选项。

    Args:
        text: 模型原始输出（未清洗，可含解释）。

    Returns:
        ``(answer, rule)``：answer 为 ``"A"``–``"D"`` 或 ``None``；rule 为命中的规则名，
        未命中时为 ``"none"``。
    """
    norm = normalize_text(text)
    for rule, pattern in _PATTERNS:
        last = None
        for match in pattern.finditer(norm):
            last = match.group(1)
        if last is not None:
            return last, rule
    return None, RULE_NONE


def format_valid(answer: Optional[str]) -> bool:
    """抽取成功即视为"答案格式有效"。"""
    return answer in tuple(LETTERS)
