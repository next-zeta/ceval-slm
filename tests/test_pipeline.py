#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""端到端冒烟测试：不联网、不加载模型，验证"抽样 → 自检 → 渲染提示词"整条链路。

    python tests/test_pipeline.py

做法：在临时目录里造一份 5 学科 × 50 题的假 C-Eval（val/dev 各一套），
跑 prepare_data.py 的 local 后端，再跑 run_eval.py --dry-run 检查模板拼接。

之所以需要它：真正的数据下载和模型推理依赖网络与 torch，队友在装环境阶段就能先跑这个，
确认抽样规则、可复算性、模板拼接和非法配置拦截都是好的。
"""
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
SUBJECTS = ["college_economics", "accountant", "physician", "civil_servant", "urban_and_rural_planner"]


def run(*args) -> subprocess.CompletedProcess:
    return subprocess.run([PY, *args], cwd=str(REPO), capture_output=True, text=True, encoding="utf-8")


def build_fixture(root: Path) -> Path:
    raw = root / "ceval"
    for split, n in (("val", 50), ("dev", 5)):
        (raw / split).mkdir(parents=True, exist_ok=True)
        for subject in SUBJECTS:
            path = raw / split / "{}_{}.csv".format(subject, split)
            with path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["id", "question", "A", "B", "C", "D", "answer", "explanation"])
                for i in range(n):
                    writer.writerow([
                        i,
                        "{} {} 第 {} 题：以下说法正确的是____。".format(subject, split, i),
                        "甲选项{}".format(i), "乙选项{}".format(i),
                        "丙选项{}".format(i), "丁选项{}".format(i),
                        "ABCD"[i % 4], "",
                    ])
    return raw


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="ceval_pipeline_"))
    failures = []
    try:
        raw = build_fixture(tmp)
        out1, out2 = tmp / "out1", tmp / "out2"

        for out in (out1, out2):
            proc = run("src/prepare_data.py", "--source", "local", "--raw-dir", str(raw), "--out-dir", str(out))
            if proc.returncode != 0:
                failures.append("prepare_data 失败:\n" + proc.stdout + proc.stderr)
                print("\n".join(failures))
                return 1

        subset = json.loads((out1 / "ceval_subset_200.json").read_text(encoding="utf-8"))
        pool = json.loads((out1 / "ceval_fewshot_pool.json").read_text(encoding="utf-8"))
        if len(subset["items"]) != 200:
            failures.append("子集题数 {} != 200".format(len(subset["items"])))
        if len(pool["items"]) != 15:
            failures.append("少样本池题数 {} != 15".format(len(pool["items"])))

        per = {}
        for item in subset["items"]:
            per[item["subject"]] = per.get(item["subject"], 0) + 1
        if len(per) != 5 or any(n != 40 for n in per.values()):
            failures.append("每科题数异常: {}".format(per))

        if (out1 / "ceval_subset_200.json").read_bytes() != (out2 / "ceval_subset_200.json").read_bytes():
            failures.append("两次抽样字节不一致（不可复算）")

        base_cfg, few_cfg, bad_cfg = tmp / "base.json", tmp / "few.json", tmp / "bad.json"
        base_cfg.write_text(json.dumps({
            "subset": str(out1 / "ceval_subset_200.json"),
            "fewshot_pool": str(out1 / "ceval_fewshot_pool.json"),
        }, ensure_ascii=False), encoding="utf-8")
        few_cfg.write_text(json.dumps({
            "template": "fewshot", "fewshot_k": 2,
            "subset": str(out1 / "ceval_subset_200.json"),
            "fewshot_pool": str(out1 / "ceval_fewshot_pool.json"),
        }, ensure_ascii=False), encoding="utf-8")
        bad_cfg.write_text(json.dumps({"template": "base", "fewshot_k": 2}), encoding="utf-8")

        proc = run("src/run_eval.py", "--config", str(base_cfg), "--dry-run")
        if proc.returncode != 0 or "答案：" not in proc.stdout:
            failures.append("base dry-run 失败:\n" + proc.stdout + proc.stderr)

        proc = run("src/run_eval.py", "--config", str(few_cfg), "--dry-run")
        if proc.returncode != 0 or "示例 1：" not in proc.stdout or "示例 2：" not in proc.stdout:
            failures.append("fewshot dry-run 失败:\n" + proc.stdout + proc.stderr)
        if "示例 3：" in proc.stdout:
            failures.append("fewshot_k=2 却拼入了第 3 个示例")

        proc = run("src/run_eval.py", "--config", str(bad_cfg), "--dry-run")
        if proc.returncode == 0:
            failures.append("fewshot_k>0 + base 模板没有被拒绝")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        print("\n".join("[FAIL] " + f for f in failures))
        return 1
    print("PIPELINE OK: 抽样可复算 / 200 题 / 每科 40 / base 与 fewshot 模板正确 / 非法配置被拒")
    return 0


if __name__ == "__main__":
    sys.exit(main())
