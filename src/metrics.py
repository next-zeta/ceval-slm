#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""指标计算：只从逐题记录反算，避免"表格数字和日志对不上"。

口径（《方案.md》§2、§6.2）：

- ``accuracy``           = 答对题数 / 全部评测题数；
- ``format_valid_rate``  = 抽取到唯一选项的题数 / 全部评测题数（抽取失败 = 计错且格式无效）；
- ``truncation_rate``    = 输出达到 ``max_new_tokens`` 上限仍未结束的题数 / 全部题数；
- ``total_elapsed_s``    = 逐题 ``elapsed_s`` 之和（不含模型加载与数据读取）；
- ``avg_elapsed_ms``     = ``total_elapsed_s`` / 题数 × 1000。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def _rate(num: int, den: int) -> float:
    return (num / den) if den else 0.0


def build_summary(records: Iterable[Mapping[str, Any]], meta: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """从逐题记录生成汇总。``meta`` 中的字段（模型、配置、环境）会被并入顶层。"""
    rows: List[Mapping[str, Any]] = list(records)
    n = len(rows)
    correct = sum(1 for r in rows if r.get("correct"))
    fmt = sum(1 for r in rows if r.get("format_valid"))
    trunc = sum(1 for r in rows if r.get("truncated"))
    total_elapsed = sum(float(r.get("elapsed_s") or 0.0) for r in rows)
    n_calls = sum(int(r.get("n_calls") or 0) for r in rows)
    out_tokens = sum(int(r.get("output_tokens") or 0) for r in rows)

    per_subject: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        sub = str(r["subject"])
        bucket = per_subject.setdefault(sub, {"n": 0, "correct": 0, "format_valid": 0})
        bucket["n"] += 1
        bucket["correct"] += int(bool(r.get("correct")))
        bucket["format_valid"] += int(bool(r.get("format_valid")))

    for sub, bucket in per_subject.items():
        bucket["accuracy"] = _rate(bucket["correct"], bucket["n"])
        bucket["format_valid_rate"] = _rate(bucket["format_valid"], bucket["n"])

    summary: Dict[str, Any] = {
        "n_samples": n,
        "accuracy": _rate(correct, n),
        "correct": correct,
        "format_valid_rate": _rate(fmt, n),
        "truncation_rate": _rate(trunc, n),
        "total_elapsed_s": round(total_elapsed, 3),
        "avg_elapsed_s": round(total_elapsed / n, 4) if n else 0.0,
        "avg_elapsed_ms": round(total_elapsed / n * 1000, 1) if n else 0.0,
        "total_n_calls": n_calls,
        "avg_n_calls": round(n_calls / n, 3) if n else 0.0,
        "avg_output_tokens": round(out_tokens / n, 1) if n else 0.0,
        "per_subject": per_subject,
    }
    if rows and all("batch_index" in row for row in rows):
        calls_by_batch = {row["batch_index"]: int(row["n_calls"]) for row in rows}
        summary["total_generate_api_calls"] = sum(calls_by_batch.values())
    if meta:
        summary.update(dict(meta))
    return summary


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """写 UTF-8 JSON（中文不转义，末尾带换行，便于 git diff）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_name(p.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        temporary.replace(p)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """写逐题记录 JSONL（一行一题）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
