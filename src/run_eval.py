#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一推理与计时脚本：基线 / 四项改进 / 消融共用一个入口。

三个必须保持一致的东西（改一处就要全组同步）：

1. **提示词**：由 ``src/prompts.py`` 生成，模板名写在 config 里；
2. **抽取规则**：由 ``src/extract.py`` 统一实现，禁止在实验分支里另写正则；
3. **计时口径**：只包住 ``model.generate()``，不含模型加载、数据读取与解码。

用法::

    # Day 1 试跑（5 题）
    python src/run_eval.py --config configs/base.json --limit 5 --exp_id try_base_A

    # 正式基线
    python src/run_eval.py --config configs/base.json --exp_id exp00_base_A

    # 改进实验
    python src/run_eval.py --config configs/prompt.json   --exp_id exp01_prompt_A
    python src/run_eval.py --config configs/fewshot.json  --exp_id exp02_fewshot_B
    python src/run_eval.py --config configs/vote.json     --exp_id exp03_voting_C
    python src/run_eval.py --config configs/batch.json    --batch_size 8 --exp_id exp04_batch_D

    # 只检查模板与数据，不加载模型
    python src/run_eval.py --config configs/base.json --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

try:  # 作为包导入时（python -m src.run_eval）
    from .extract import extract_answer, format_valid
    from .metrics import build_summary, write_json
    from .prompts import TEMPLATE_NAMES, build_prompt, template_signature
    from .provenance import git_revision
except ImportError:  # 作为脚本运行时（python src/run_eval.py）
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from extract import extract_answer, format_valid
    from metrics import build_summary, write_json
    from prompts import TEMPLATE_NAMES, build_prompt, template_signature
    from provenance import git_revision

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"

DEFAULT_CONFIG: Dict[str, Any] = {
    "exp_id": "exp00_base",
    "owner": "",
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "model_revision": None,
    "template": "base",
    "fewshot_k": 0,
    "n_samples": 1,
    "do_sample": False,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 16,
    "batch_size": 1,
    "seed": 42,
    "device": "auto",
    "dtype": "float32",
    "limit": None,
    "subset": "data/ceval_subset_200.json",
    "fewshot_pool": "data/ceval_fewshot_pool.json",
    "log_dir": "logs",
}

_DTYPE_NAMES = ("float32", "float16", "bfloat16")


# --------------------------------------------------------------------------- #
# 配置与工具
# --------------------------------------------------------------------------- #
def load_config(path: Optional[str], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if path:
        cfg_path = Path(path)
        if not cfg_path.is_absolute():
            cfg_path = (Path.cwd() / cfg_path).resolve()
        if not cfg_path.exists():
            raise SystemExit("[错误] 找不到配置文件: {}".format(cfg_path))
        with cfg_path.open("r", encoding="utf-8") as fh:
            user_cfg = json.load(fh)
        if not isinstance(user_cfg, dict):
            raise SystemExit("[错误] 配置文件必须是 JSON 对象: {}".format(cfg_path))
        unknown = sorted(set(user_cfg) - set(DEFAULT_CONFIG))
        if unknown:
            raise SystemExit("[错误] 未知配置项 {}（可用项: {}）".format(unknown, sorted(DEFAULT_CONFIG)))
        cfg.update(user_cfg)
        cfg["_config_path"] = str(cfg_path)
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    if cfg["model"] == DEFAULT_CONFIG["model"] and cfg["model_revision"] is None:
        cfg["model_revision"] = DEFAULT_MODEL_REVISION

    if cfg["template"] not in TEMPLATE_NAMES:
        raise SystemExit("[错误] 未知模板 {!r}，可选 {}".format(cfg["template"], TEMPLATE_NAMES))
    if cfg["template"] == "fewshot" and int(cfg["fewshot_k"]) <= 0:
        raise SystemExit("[错误] fewshot 模板必须设置 fewshot_k > 0")
    if int(cfg["fewshot_k"]) > 0 and cfg["template"] != "fewshot":
        raise SystemExit(
            "[错误] 只有 fewshot 模板会拼入示例，当前 template={!r} 但 fewshot_k={}；"
            "要么改模板，要么把 fewshot_k 设回 0".format(cfg["template"], cfg["fewshot_k"])
        )
    if cfg["dtype"] not in _DTYPE_NAMES:
        raise SystemExit("[错误] dtype 只能是 {}".format(_DTYPE_NAMES))
    if int(cfg["batch_size"]) < 1 or int(cfg["n_samples"]) < 1:
        raise SystemExit("[错误] batch_size 与 n_samples 必须 >= 1")
    return cfg


def _rel(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


def _file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def resolve_device(spec: str, torch) -> "torch.device":
    if spec and spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and getattr(mps, "is_available", lambda: False)():
        return torch.device("mps")
    return torch.device("cpu")


def torch_dtype(name: str, device: str, torch):
    if device.startswith("cuda"):
        mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        return mapping[name]
    if name != "float32":
        print("[警告] {} 上请求的精度为 {}，本次使用 float32".format(device, name))
    return torch.float32


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def load_subset(path: Path, limit: Optional[int]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(
            "[错误] 找不到评测集 {}。先运行 `python src/prepare_data.py --source hf`".format(path)
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = list(payload["items"])
    items.sort(key=lambda it: (it["subject"], int(it["qid"])))
    meta = dict(payload.get("meta", {}))
    if limit:
        items = items[: int(limit)]
    return items, meta


def load_pool(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in payload.get("items", []):
        grouped.setdefault(item["subject"], []).append(item)
    for shots in grouped.values():
        shots.sort(key=lambda it: int(it["qid"]))
    return grouped


def shots_for(pool: Mapping[str, List[Dict[str, Any]]], subject: str, k: int) -> List[Dict[str, Any]]:
    if k <= 0:
        return []
    shots = pool.get(subject, [])
    if len(shots) < k:
        raise SystemExit("[错误] {} 的少样本池只有 {} 题，不足 fewshot_k={}".format(subject, len(shots), k))
    return shots[:k]


# --------------------------------------------------------------------------- #
# 推理
# --------------------------------------------------------------------------- #
def render_chat(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def synchronize_device(device, torch) -> None:
    """等待 GPU 完成当前设备上的工作，CPU 推理本身是同步的。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def generate_samples(model, tokenizer, texts: Sequence[str], cfg: Mapping[str, Any],
                     device, torch) -> Tuple[List[Dict[str, Any]], float]:
    """对一批文本各生成一次；返回 (每行结果, 本批 wall-clock 秒)。"""
    enc = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    in_len = int(enc["input_ids"].shape[1])
    prompt_tokens = enc["attention_mask"].sum(dim=1).tolist()

    kwargs: Dict[str, Any] = {
        "max_new_tokens": int(cfg["max_new_tokens"]),
        "pad_token_id": tokenizer.pad_token_id,
    }
    if bool(cfg["do_sample"]) or int(cfg["n_samples"]) > 1:
        kwargs.update(do_sample=True, temperature=float(cfg["temperature"]), top_p=float(cfg["top_p"]))
    else:
        kwargs.update(do_sample=False)

    with torch.inference_mode():
        synchronize_device(device, torch)
        t0 = time.perf_counter()
        out = model.generate(**enc, **kwargs)
        synchronize_device(device, torch)
        elapsed = time.perf_counter() - t0

    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or ())
    rows: List[Dict[str, Any]] = []
    for i in range(int(out.shape[0])):
        generated_ids = out[i, in_len:].tolist()
        end = next((index for index, token in enumerate(generated_ids) if token in eos_ids), None)
        ids = generated_ids[:end]
        text = tokenizer.decode(ids, skip_special_tokens=True).strip()
        rows.append({
            "text_for_parsing": text,
            "generated_token_ids": generated_ids,
            "output_tokens": len(ids),
            "prompt_tokens": int(prompt_tokens[i]),
            "truncated": end is None and len(ids) >= int(cfg["max_new_tokens"]),
        })
    return rows, elapsed


def vote(answers: Sequence[Optional[str]]) -> Optional[str]:
    """多数投票；三票各不同时取第一次采样（《方案.md》§5.1）。"""
    valid = [a for a in answers if a]
    if not valid:
        return None
    counts = Counter(valid)
    top = max(counts.values())
    winners = [a for a, c in counts.items() if c == top]
    if len(winners) == 1:
        return winners[0]
    first = answers[0]
    return first if first in winners else winners[0]


def _env_snapshot(torch, transformers, model, device, cfg) -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": getattr(torch, "__version__", "unknown"),
        "transformers": getattr(transformers, "__version__", "unknown"),
        "device": str(device),
        "platform": platform.platform(),
        "dtype": str(model.dtype).removeprefix("torch."),
        "dtype_requested": cfg["dtype"],
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def _load_model(cfg: Mapping[str, Any], device, torch):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    revision = cfg.get("model_revision") or None
    common = {"revision": revision, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], **common)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch_dtype(str(cfg["dtype"]), str(device), torch)
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=dtype, **common)
    except TypeError:  # 老版本 transformers 只认 torch_dtype
        model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=dtype, **common)
    model.to(device)
    model.eval()
    return tokenizer, model


def evaluate_rows(items, pool, cfg, tokenizer, model, device, torch) -> Iterator[Dict[str, Any]]:
    """逐题产出日志行，使已完成的批次可随时落盘。"""
    batch_size = int(cfg["batch_size"])
    n_samples = int(cfg["n_samples"])
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompts = [
            render_chat(tokenizer, build_prompt(item, cfg["template"],
                                                shots_for(pool, item["subject"], int(cfg["fewshot_k"]))))
            for item in batch
        ]
        samples = [generate_samples(model, tokenizer, prompts, cfg, device, torch)
                   for _ in range(n_samples)]
        elapsed_item = sum(elapsed for _, elapsed in samples) / len(batch)

        for offset, item in enumerate(batch):
            outs = [outputs[offset] for outputs, _ in samples]
            parsed = [extract_answer(out["text_for_parsing"]) for out in outs]
            answers = [answer for answer, _ in parsed]
            final = answers[0] if n_samples == 1 else vote(answers)
            gold = str(item["answer"]).strip().upper()
            yield {
                "exp_id": str(cfg["exp_id"]),
                "split": item.get("source_split", "val"),
                "subject": item["subject"],
                "qid": int(item["qid"]),
                "prompt": prompts[offset],
                "raw_output": "\n===SAMPLE===\n".join(out["text_for_parsing"] for out in outs),
                "generations": [dict(out, extracted=answer, rule=rule)
                                for out, (answer, rule) in zip(outs, parsed)],
                "sample_answers": answers if n_samples > 1 else None,
                "extracted": final,
                "rule": parsed[0][1] if n_samples == 1 else None,
                "gold": gold,
                "correct": final == gold,
                "format_valid": bool(format_valid(final)),
                "truncated": any(out["truncated"] for out in outs),
                "elapsed_s": round(elapsed_item, 4),
                "n_calls": n_samples,
                "prompt_tokens": outs[0]["prompt_tokens"],
                "output_tokens": sum(out["output_tokens"] for out in outs),
                "batch_size": len(batch),
                "batch_index": start // batch_size + 1,
                "template": cfg["template"],
            }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="C-Eval 统一推理与评测")
    parser.add_argument("--config", help="实验配置 JSON（configs/*.json）")
    parser.add_argument("--exp_id", help="实验编号，如 exp00_base_A（覆盖配置里的 exp_id）")
    parser.add_argument("--limit", type=int, help="只跑前 N 题（试跑用）")
    parser.add_argument("--batch_size", type=int, help="批量大小（覆盖配置）")
    parser.add_argument("--subset", help="评测集路径（默认 data/ceval_subset_200.json）")
    parser.add_argument("--note", default="", help="写进 summary 的自由备注（设备名等）")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖同名实验日志")
    parser.add_argument("--dry-run", action="store_true", help="只渲染第 1 条提示词并校验数据，不加载模型")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, {
        "exp_id": args.exp_id,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "subset": args.subset,
    })

    subset_path = _rel(cfg["subset"])
    items, subset_meta = load_subset(subset_path, args.limit)
    pool = load_pool(_rel(cfg["fewshot_pool"]))
    if not items:
        raise SystemExit("[错误] 评测集为空")

    first_prompt = build_prompt(items[0], cfg["template"],
                                shots_for(pool, items[0]["subject"], int(cfg["fewshot_k"])))
    if args.dry_run:
        print("=== 数据检查 ===")
        print("评测集: {}（{} 题，{} 个学科）".format(subset_path, len(items), len({i['subject'] for i in items})))
        print("模板  : {} | fewshot_k={} | 指纹 {}".format(
            cfg["template"], cfg["fewshot_k"], template_signature(cfg["template"])))
        print("=== 第 1 条完整输入 ===")
        print(first_prompt)
        return 0

    try:
        import torch
        import transformers
    except ImportError:
        raise SystemExit("[错误] 未安装 torch/transformers。请先 `pip install -r requirements.txt`")

    device = resolve_device(str(cfg["device"]), torch)
    exp_id = str(cfg["exp_id"])
    log_dir = _rel(cfg["log_dir"]) / exp_id
    jsonl_path = log_dir / "{}.jsonl".format(exp_id)
    summary_path = log_dir / "summary.json"
    failure_path = log_dir / "failure.json"
    if any(path.exists() for path in (jsonl_path, summary_path, failure_path)) and not args.overwrite:
        raise SystemExit(
            "[错误] {} 已有日志。换一个 --exp_id，或确认后加 --overwrite（会覆盖旧结果，正式实验不建议）".format(log_dir)
        )

    seed = int(cfg["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("=== 环境 ===")
    print("experiment : {}".format(exp_id))
    print("model      : {} (revision={})".format(cfg["model"], cfg.get("model_revision") or "default"))
    print("template   : {} | fewshot_k={} | n_samples={}".format(cfg["template"], cfg["fewshot_k"], cfg["n_samples"]))
    print("decoding   : do_sample={} max_new_tokens={} batch_size={}".format(
        bool(cfg["do_sample"]) or int(cfg["n_samples"]) > 1, cfg["max_new_tokens"], cfg["batch_size"]))
    print("device     : {} | requested dtype={}".format(device, cfg["dtype"]))
    print("subset     : {} 题（md5 {}）".format(len(items), _file_md5(subset_path)[:12]))

    batch_size = int(cfg["batch_size"])
    n_samples = int(cfg["n_samples"])
    git_commit = git_revision(ROOT)
    log_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    n_correct = 0
    log_rows = jsonl_path.open("w" if args.overwrite else "x", encoding="utf-8", newline="\n")
    stage = "load_model"
    try:
        with log_rows as fh:
            if args.overwrite:
                summary_path.unlink(missing_ok=True)
                failure_path.unlink(missing_ok=True)
            tokenizer, model = _load_model(cfg, device, torch)
            started = datetime.now(timezone.utc).astimezone()
            t_start = time.perf_counter()
            stage = "evaluate"
            for record in evaluate_rows(items, pool, cfg, tokenizer, model, device, torch):
                record["git_commit"] = git_commit
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                records.append(record)
                n_correct += int(record["correct"])
                done = len(records)
                if done == len(items) or (done % batch_size == 0 and done % 20 < batch_size):
                    spent = time.perf_counter() - t_start
                    eta = spent / done * (len(items) - done)
                    print("  [{:>3}/{}] acc={:.3f} 已用 {:.1f}s 预计剩余 {:.1f}s".format(
                        done, len(items), n_correct / done, spent, eta))

        stage = "write_summary"
        finished = datetime.now(timezone.utc).astimezone()
        meta = {
            "exp_id": exp_id,
            "owner": cfg.get("owner", ""),
            "model": cfg["model"],
            "model_revision": cfg.get("model_revision"),
            "model_resolved_revision": getattr(model.config, "_commit_hash", None),
            "generation_eos_token_id": model.generation_config.eos_token_id,
            "template": cfg["template"],
            "template_sha1": template_signature(cfg["template"]),
            "fewshot_k": int(cfg["fewshot_k"]),
            "generations_per_question": n_samples,
            "do_sample": bool(cfg["do_sample"]) or n_samples > 1,
            "temperature": float(cfg["temperature"]),
            "top_p": float(cfg["top_p"]),
            "max_new_tokens": int(cfg["max_new_tokens"]),
            "batch_size": batch_size,
            "seed": seed,
            "limit": args.limit,
            "subset": subset_path.name,
            "subset_md5": _file_md5(subset_path),
            "subset_subjects": subset_meta.get("subjects"),
            "git_commit": git_commit,
            "config_path": cfg.get("_config_path"),
            "note": args.note,
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": finished.isoformat(timespec="seconds"),
            "wall_clock_s": round(time.perf_counter() - t_start, 3),
            "env": _env_snapshot(torch, transformers, model, device, cfg),
        }
        summary = build_summary(records, meta)
        write_json(summary_path, summary)
    except (Exception, KeyboardInterrupt) as error:
        batch_start = len(records) // batch_size * batch_size
        failed_batch = items[batch_start:batch_start + batch_size] if stage == "evaluate" else []
        try:
            write_json(failure_path, {
                "exp_id": exp_id,
                "stage": stage,
                "completed_questions": len(records),
                "planned_questions": len(items),
                "batch": [{"subject": item["subject"], "qid": int(item["qid"])} for item in failed_batch],
                "git_commit": git_commit,
                "error_type": type(error).__name__,
                "error_message": str(error),
            })
        except OSError as log_error:
            print("[警告] 无法保存故障记录: {}".format(log_error), file=sys.stderr)
        raise

    print("=== 结果 ===")
    print("accuracy          : {:.4f} ({}/{})".format(summary["accuracy"], summary["correct"], summary["n_samples"]))
    print("format_valid_rate : {:.4f}".format(summary["format_valid_rate"]))
    print("truncation_rate   : {:.4f}".format(summary["truncation_rate"]))
    print("total_elapsed_s   : {:.1f}".format(summary["total_elapsed_s"]))
    print("avg_elapsed_s     : {:.3f}".format(summary["avg_elapsed_s"]))
    print("per_subject       :")
    for subject, stats in sorted(summary["per_subject"].items()):
        print("  - {:<26} {}/{} = {:.3f}  (格式有效 {:.3f})".format(
            subject, stats["correct"], stats["n"], stats["accuracy"], stats["format_valid_rate"]))
    print("logs: {}".format(log_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
