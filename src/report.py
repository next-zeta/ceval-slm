#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""读取各次运行的 summary.json，生成 Markdown 实验记录表。

用法::

    python src/report.py --logs logs
    python src/report.py --logs logs --out docs/实验记录总表_自动生成.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

HEADER = (
    "| 实验编号 | 负责人 | Commit | 设备 | 模板 / shots / 每题生成 / batch | 单题生成总数 "
    "| generate API 调用数 | 准确率 | 格式有效率 | 截断率 | 总耗时(s) | 分摊每题(ms) |"
)
SEP = "| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"


def _fmt_pct(value: Any) -> str:
    try:
        return "{:.2f}%".format(float(value) * 100)
    except (TypeError, ValueError):
        return "-"


def _fmt_num(value: Any, digits: int = 1) -> str:
    try:
        return ("{:." + str(digits) + "f}").format(float(value))
    except (TypeError, ValueError):
        return "-"


def collect(logs_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(logs_root.glob("*/summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("[警告] 跳过无法解析的 {}".format(path), file=sys.stderr)
            continue
        data["_path"] = str(path)
        rows.append(data)
    rows.sort(key=lambda r: str(r.get("exp_id", "")))
    return rows


def render(rows: List[Dict[str, Any]]) -> str:
    lines = [HEADER, SEP]
    for r in rows:
        change = "{}/{}/{}/{}".format(
            r.get("template", "?"),
            r.get("fewshot_k", 0),
            r.get("generations_per_question", 1),
            r.get("batch_size", 1),
        )
        lines.append("| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r.get("exp_id", "?"),
            r.get("owner") or "-",
            r.get("git_commit", "?"),
            (r.get("env") or {}).get("device", "-"),
            change,
            r.get("total_n_calls", "-"),
            r.get("total_generate_api_calls", "-"),
            _fmt_pct(r.get("accuracy")),
            _fmt_pct(r.get("format_valid_rate")),
            _fmt_pct(r.get("truncation_rate")),
            _fmt_num(r.get("total_elapsed_s"), 1),
            _fmt_num(r.get("avg_elapsed_ms"), 0),
        ))
    return "\n".join(lines) + "\n"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从 logs/*/summary.json 生成实验总表")
    parser.add_argument("--logs", default="logs", help="日志根目录（默认 logs）")
    parser.add_argument("--out", help="写入 Markdown 文件（默认只打印）")
    args = parser.parse_args(argv)

    root = Path(args.logs)
    if not root.exists():
        raise SystemExit("[错误] 找不到日志目录 {}（先在仓库根目录运行实验）".format(root.resolve()))

    rows = collect(root)
    if not rows:
        raise SystemExit("[错误] {} 下没有 summary.json".format(root.resolve()))

    table = render(rows)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table, encoding="utf-8", newline="\n")
        print("已写入 {}".format(out))
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
