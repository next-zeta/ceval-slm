#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""构造固定的 200 题评测子集与少样本池（可复算，同输入必得同 md5）。

设计依据《方案.md》§3，以及与组内确认后的 v0.3 调整：

- 只用 C-Eval **val** 划分（答案公开、不碰 test）；
- 5 个学科 × 40 题 = 200 题，学科按 val 题量 ≥40 选取：
  ``college_economics`` / ``accountant`` / ``physician`` / ``civil_servant`` /
  ``urban_and_rural_planner``（原因见 docs/数据集说明.md）；
- 抽样：学科按固定顺序，题号升序后 ``random.Random(SEED).sample(qids, 40)``；
- 少样本池：每科从 **dev** 取 3 题，顺序固定，与评测集零交集。

用法::

    python src/prepare_data.py --source hf
    HF_ENDPOINT=https://hf-mirror.com python src/prepare_data.py --source hf
    python src/prepare_data.py --source local --raw-dir data/raw/ceval

产物（写入 ``data/``）：

- ``ceval_subset_200.json``
- ``ceval_fewshot_pool.json``
- ``subset_manifest.json``（md5、种子、题量、来源、生成时间）
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "data"

#: 全组冻结（v0.3）：5 个学科，均满足 C-Eval val 题量 >= 40
SUBJECTS: Tuple[str, ...] = (
    "college_economics",          # 经济学：val 55
    "accountant",                 # 会计：val 49
    "physician",                  # 医师：val 49
    "civil_servant",              # 公务员：val 47
    "urban_and_rural_planner",    # 城乡规划：val 46
)
PER_SUBJECT = 40
SEED = 42
FEWSHOT_PER_SUBJECT = 3

DATASET_NAME = "ceval/ceval-exam"

SUBSET_FILE = "ceval_subset_200.json"
POOL_FILE = "ceval_fewshot_pool.json"
MANIFEST_FILE = "subset_manifest.json"


# --------------------------------------------------------------------------- #
# 数据读取
# --------------------------------------------------------------------------- #
def _norm_row(row: Mapping[str, Any], subject: str, split: str) -> Dict[str, Any]:
    """统一行结构，并在入口处做字段完整性校验（不合法直接报错，不静默丢弃）。"""
    try:
        qid = int(row["id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit("[错误] {} {} 行缺少合法 id: {!r}".format(subject, split, row)) from exc

    question = str(row.get("question") or "").strip()
    options = {letter: str(row.get(letter) or "").strip() for letter in "ABCD"}
    answer = str(row.get("answer") or "").strip().upper()

    if not question:
        raise SystemExit("[错误] {} {}#{} 题干为空".format(subject, split, qid))
    if any(not v for v in options.values()):
        raise SystemExit("[错误] {} {}#{} 选项不完整".format(subject, split, qid))
    if answer not in ("A", "B", "C", "D"):
        raise SystemExit("[错误] {} {}#{} 答案非法: {!r}".format(subject, split, qid, answer))

    return {
        "subject": subject,
        "qid": qid,
        "question": question,
        "options": options,
        "answer": answer,
        "source_split": split,
    }


def load_local(raw_dir: Path, subject: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """读取 C-Eval 官方 CSV 目录：``<raw_dir>/val/<subject>_val.csv``。"""
    result: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for split in ("val", "dev"):
        csv_path = raw_dir / split / "{}_{}.csv".format(subject, split)
        if not csv_path.exists():
            raise SystemExit(
                "[错误] 找不到 {}；请确认 --raw-dir 指向 C-Eval 官方数据根目录"
                "（应包含 val/ 与 dev/ 子目录）".format(csv_path)
            )
        rows: Dict[int, Dict[str, Any]] = {}
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            for raw in csv.DictReader(fh):
                item = _norm_row(raw, subject, split)
                rows[item["qid"]] = item
        result[split] = rows
    return result


def load_hf(subject: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """读取 HuggingFace ``ceval/ceval-exam``（国内可设 HF_ENDPOINT=https://hf-mirror.com）。"""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "[错误] 未安装 datasets。请先 `pip install -r requirements.txt`，"
            "或改用 --source local --raw-dir <C-Eval 官方数据目录>"
        )
    try:
        ds = load_dataset(DATASET_NAME, subject)
    except Exception as exc:  # noqa: BLE001 - 需要把网络/镜像问题原样告诉组员
        raise SystemExit(
            "[错误] 下载 {} / {} 失败: {}\n"
            "可尝试: 设置 HF_ENDPOINT=https://hf-mirror.com 后重试；"
            "或手动下载 C-Eval 官方数据后用 --source local".format(DATASET_NAME, subject, exc)
        ) from exc

    result: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for split in ("val", "dev"):
        rows: Dict[int, Dict[str, Any]] = {}
        for raw in ds[split]:
            item = _norm_row(dict(raw), subject, split)
            rows[item["qid"]] = item
        result[split] = rows
    return result


# --------------------------------------------------------------------------- #
# 抽样
# --------------------------------------------------------------------------- #
def sample_subset(
    loader,
    subjects: Sequence[str],
    per_subject: int,
    seed: int,
    fewshot_per_subject: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """按固定规则抽样，返回 (评测集, 少样本池, 每科题量统计)。"""
    rng = random.Random(seed)  # 全流程共用一个 rng，保证学科顺序固定时可复算
    subset: List[Dict[str, Any]] = []
    pool: List[Dict[str, Any]] = []
    counts: Dict[str, Dict[str, int]] = {}

    for subject in subjects:
        data = loader(subject)
        val_ids = sorted(data["val"].keys())
        dev_ids = sorted(data["dev"].keys())

        if len(val_ids) < per_subject:
            raise SystemExit(
                "[错误] {} 的 val 划分只有 {} 题，不足 {} 题；"
                "请换学科或调小 --per-subject（C-Eval val 题量见 docs/数据集说明.md）".format(
                    subject, len(val_ids), per_subject
                )
            )
        if len(dev_ids) < fewshot_per_subject:
            raise SystemExit(
                "[错误] {} 的 dev 划分只有 {} 题，不足 {} 题".format(
                    subject, len(dev_ids), fewshot_per_subject
                )
            )

        picked = sorted(rng.sample(val_ids, per_subject))
        for qid in picked:
            subset.append(data["val"][qid])

        for qid in dev_ids[:fewshot_per_subject]:
            shot = dict(data["dev"][qid])
            shot["dev_order"] = dev_ids.index(qid)
            pool.append(shot)

        counts[subject] = {
            "val_available": len(val_ids),
            "dev_available": len(dev_ids),
            "sampled": len(picked),
        }

    subset.sort(key=lambda it: (it["subject"], it["qid"]))
    pool.sort(key=lambda it: (it["subject"], it["qid"]))
    return subset, pool, counts


# --------------------------------------------------------------------------- #
# 防泄露与完整性自检
# --------------------------------------------------------------------------- #
def validate(subset: Sequence[Mapping[str, Any]], pool: Sequence[Mapping[str, Any]], per_subject: int) -> None:
    """子集/池自检；任何一条不通过都直接失败，避免带病数据进仓库。

    注意：C-Eval 的 val 与 dev **各自从 0 编号**，所以 ``(subject, qid)`` 跨划分必然重复，
    去重键必须带上 ``source_split``；真正的泄露检查靠题干文本比对。
    """
    keys_sub = [(it["source_split"], it["subject"], it["qid"]) for it in subset]
    if len(set(keys_sub)) != len(keys_sub):
        raise SystemExit("[自检失败] 评测集中存在重复题")

    texts_sub = {it["question"] for it in subset}
    texts_pool = {it["question"] for it in pool}
    if texts_sub & texts_pool:
        raise SystemExit("[自检失败] 少样本池与评测集存在相同题干（疑似泄露）")
    if len(texts_sub) != len(subset):
        raise SystemExit("[自检失败] 评测集中存在题干完全相同的题")

    per: Dict[str, int] = {}
    for it in subset:
        per[it["subject"]] = per.get(it["subject"], 0) + 1
    if len(per) < 5:
        raise SystemExit("[自检失败] 学科数 {} < 5".format(len(per)))
    for subject, n in per.items():
        if n != per_subject:
            raise SystemExit("[自检失败] {} 抽到 {} 题，期望 {}".format(subject, n, per_subject))
        if n < 20:
            raise SystemExit("[自检失败] {} 只有 {} 题，低于每科 20 题".format(subject, n))
    if len(subset) < 200:
        raise SystemExit("[自检失败] 总题数 {} < 200".format(len(subset)))


# --------------------------------------------------------------------------- #
# 写盘
# --------------------------------------------------------------------------- #
def _dump(path: Path, payload: Mapping[str, Any]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path.read_bytes()


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 C-Eval 固定 200 题子集与少样本池")
    parser.add_argument("--source", choices=("hf", "local"), default="hf", help="数据来源（默认 hf）")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "ceval",
                        help="--source local 时的 C-Eval 官方数据根目录")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="输出目录（默认 data/）")
    parser.add_argument("--subjects", nargs="+", default=list(SUBJECTS), help="学科列表（默认冻结的 5 科）")
    parser.add_argument("--per-subject", type=int, default=PER_SUBJECT, help="每科题数（默认 40）")
    parser.add_argument("--fewshot-per-subject", type=int, default=FEWSHOT_PER_SUBJECT,
                        help="每科 dev 示例数（默认 3）")
    parser.add_argument("--seed", type=int, default=SEED, help="随机种子（默认 42，冻结）")
    args = parser.parse_args(argv)

    loader = (lambda s: load_local(args.raw_dir, s)) if args.source == "local" else load_hf

    print("[1/4] 读取并抽样：{} 个学科 × {} 题，seed={}".format(len(args.subjects), args.per_subject, args.seed))
    subset, pool, counts = sample_subset(
        loader, args.subjects, args.per_subject, args.seed, args.fewshot_per_subject
    )

    print("[2/4] 自检：题量 / 学科数 / 与少样本池零交集")
    validate(subset, pool, args.per_subject)

    meta_subset = {
        "dataset": DATASET_NAME,
        "source": args.source,
        "split": "val",
        "seed": args.seed,
        "per_subject": args.per_subject,
        "subjects": list(args.subjects),
    }
    meta_pool = {
        "dataset": DATASET_NAME,
        "source": args.source,
        "split": "dev",
        "seed": args.seed,
        "fewshot_per_subject": args.fewshot_per_subject,
        "subjects": list(args.subjects),
        "note": "示例仅含题干+选项+答案字母，不含解题过程；与评测集零交集",
    }

    subset_bytes = _dump(args.out_dir / SUBSET_FILE, {"meta": meta_subset, "items": subset})
    pool_bytes = _dump(args.out_dir / POOL_FILE, {"meta": meta_pool, "items": pool})

    manifest = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "dataset": DATASET_NAME,
        "source": args.source,
        "seed": args.seed,
        "per_subject": args.per_subject,
        "n_total": len(subset),
        "n_fewshot": len(pool),
        "subjects": counts,
        "files": {
            SUBSET_FILE: {"md5": _md5(subset_bytes), "bytes": len(subset_bytes), "n_items": len(subset)},
            POOL_FILE: {"md5": _md5(pool_bytes), "bytes": len(pool_bytes), "n_items": len(pool)},
        },
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "note": "md5 可用于核对全组是否使用同一份题目；数据文件本身不含时间戳以保证字节稳定",
    }
    _dump(args.out_dir / MANIFEST_FILE, manifest)

    print("[3/4] 写入 {}：{} 题".format(args.out_dir / SUBSET_FILE, len(subset)))
    print("       写入 {}：{} 题".format(args.out_dir / POOL_FILE, len(pool)))
    print("[4/4] md5 已写入 {}".format(args.out_dir / MANIFEST_FILE))
    for subject, info in counts.items():
        print("  - {:<26} val {:>3} 题 → 抽 {:>2} 题".format(subject, info["val_available"], info["sampled"]))
    print("  子集 md5: {}".format(manifest["files"][SUBSET_FILE]["md5"]))
    print("  池   md5: {}".format(manifest["files"][POOL_FILE]["md5"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
