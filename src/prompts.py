#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""提示模板：全组唯一实现（冻结文件）。

依据《方案.md》§4：

- ``base``           基线零样本模板（§4.1）：只声明"给出字母"，不禁止解释；
- ``strict_fmt``     成员 A 的改进（§4.2）：仅加强输出契约，唯一变量；
- ``strict_fmt_role`` 可选第三段（§4.2 加分项）：在 strict_fmt 上再加一句角色设定；
- ``fewshot``        成员 B 的改进（§4.3）：拼入 dev 集示例，示例数量由调用方给定。

题目块、选项块、结尾 "答案：" 四者保持一致，保证换模板时只有头部指令在变。
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, Mapping, Sequence

TEMPLATE_NAMES = ("base", "strict_fmt", "strict_fmt_role", "fewshot")

BASE_HEADER = "请回答下面的单项选择题，给出正确选项的字母。"

STRICT_FMT_HEADER = (
    "请回答下面的单项选择题。\n\n"
    "输出必须严格遵守以下格式，除此之外不要输出任何内容：\n"
    "答案：X\n"
    "其中 X 是 A、B、C 或 D 中的一个字母。不要复述题目，不要给出分析或解释。"
)

STRICT_FMT_ROLE_HEADER = "你是一名严谨的中文学科专家。\n\n" + STRICT_FMT_HEADER

FEWSHOT_HEADER = "你是参加中文考试的考生。请根据示例作答，只输出一个选项字母（A、B、C 或 D）。"

#: 模板头部文本，用于在日志/汇总里做"同一模板"校验
HEADERS: Dict[str, str] = {
    "base": BASE_HEADER,
    "strict_fmt": STRICT_FMT_HEADER,
    "strict_fmt_role": STRICT_FMT_ROLE_HEADER,
    "fewshot": FEWSHOT_HEADER,
}

#: 可被 run_eval.py 直接引用的别名
TEMPLATES = HEADERS


def format_options(options: Mapping[str, str]) -> str:
    """把选项字典渲染成 ``A. xxx`` 四行。"""
    return "\n".join(
        "{}. {}".format(letter, str(options[letter]).strip())
        for letter in ("A", "B", "C", "D")
    )


def format_question(item: Mapping[str, Any]) -> str:
    """题目 + 选项（不含头部指令，不含结尾"答案："）。"""
    return "题目：{}\n\n{}".format(str(item["question"]).strip(), format_options(item["options"]))


def format_example(item: Mapping[str, Any], index: int) -> str:
    """少样本示例：题干 + 选项 + 答案字母，**不解释**（防止把解题过程喂给模型）。"""
    return "示例 {}：\n{}\n答案：{}".format(index, format_question(item), item["answer"])


def build_prompt(
    item: Mapping[str, Any],
    template: str = "base",
    shots: Sequence[Mapping[str, Any]] = (),
) -> str:
    """组装完整输入文本。

    Args:
        item: 评测题，需含 ``question`` / ``options``。
        template: ``TEMPLATE_NAMES`` 之一。
        shots: few-shot 示例（``template="fewshot"`` 时使用；顺序即拼入顺序）。

    Returns:
        交给 ``tokenizer.apply_chat_template`` 的用户消息文本。
    """
    if template not in HEADERS:
        raise ValueError("未知模板: {}（可选 {}）".format(template, ", ".join(TEMPLATE_NAMES)))

    head = HEADERS[template]
    body = format_question(item)

    if template == "fewshot":
        blocks = [head]
        for i, shot in enumerate(shots, start=1):
            blocks.append(format_example(shot, i))
        blocks.append(body)
    else:
        blocks = [head, body]

    return "\n\n".join(blocks) + "\n\n答案："


def template_signature(template: str) -> str:
    """模板指纹：用于在汇总里确认"两组结果用的是同一套提示词"。"""
    return hashlib.sha1(HEADERS[template].encode("utf-8")).hexdigest()[:12]


def shot_texts(shots: Iterable[Mapping[str, Any]]) -> str:
    """把示例拼成一段可读文本，便于写日志/复盘（不参与推理）。"""
    return "\n\n".join(format_example(s, i) for i, s in enumerate(shots, start=1))
