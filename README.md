# C-Eval 小语言模型基准评测、改进与消融

> 远端仓库：<https://github.com/next-zeta/ceval-slm> ｜ 分支 `main`

课程大作业（4 人小组）。基线模型固定为 **`Qwen/Qwen2.5-0.5B-Instruct`**，评测集是从 C-Eval
**val** 划分按固定种子抽出的 **200 题（5 学科 × 40 题，SEED=42）**，全部实验在同一份题目、
同一套抽取规则、同一台设备上做基线对照。

> 本仓库不是"刷分"仓库。考核看的是：基线能否被队友按 README 一次跑通、每次改动是否单变量受控、
> 逐题日志能否反算出总表里的每个数字。

---

## 1. 现在能做什么

| 能力 | 入口 |
| :--- | :--- |
| 生成 200 题子集 + 少样本池（可复算 md5） | `src/prepare_data.py` |
| 基线 / 提示词 / few-shot / 投票 / 批量推理 | `src/run_eval.py` + `configs/*.json` |
| 逐题日志与汇总（含环境快照、commit） | 自动写入 `logs/<exp_id>/` |
| 从日志反算实验总表 | `src/report.py` |
| 答案抽取器回归测试 | `tests/test_extract.py` |

明确**不做**：不训练、不微调、不改网络结构、不手写算子、不调用付费 API。

---

## 2. 目录结构

```
ceval-slm/
├── README.md                  ← 你正在看的文件
├── requirements.txt
├── 方案.md                    ← 组内冻结的方案（v0.3）
├── configs/                   ← 每个实验一份配置，只改一个变量
│   ├── base.json  prompt.json  fewshot.json  vote.json  batch.json  final.json  ablation.json
├── data/
│   ├── ceval_subset_200.json  ← 评测集（抽出来后入库）
│   ├── ceval_fewshot_pool.json
│   └── subset_manifest.json   ← md5 / 种子 / 题量 / 来源
├── src/
│   ├── prepare_data.py        ← 抽样脚本（成员 A）
│   ├── run_eval.py            ← 统一推理与计时（成员 B 维护）
│   ├── prompts.py             ← 提示模板（冻结）
│   ├── extract.py             ← 答案抽取（冻结，成员 C 校验）
│   ├── metrics.py             ← 指标计算
│   ├── provenance.py          ← commit / 环境可追溯（git 不在 PATH 时的回退）
│   └── report.py              ← 总表生成
├── tests/test_extract.py
├── logs/<exp_id>/{<exp_id>.jsonl, summary.json}   ← 正式实验证据，全部入库
└── docs/
    ├── 数据集说明.md            ← 学科选择理由、val 题量、变更记录
    ├── 实验记录总表.md
    ├── 交叉复现记录.md
    ├── 结果分析.md
    ├── 成员贡献表.md
    └── AI辅助声明.md
```

---

## 3. 环境准备

### 3.1 Python

需要 **Python ≥ 3.10**（`prepare_data.py` 用到 `Path.write_text(newline=...)`）。

```powershell
python --version
```

### 3.2 安装依赖

先装 **CPU 版 torch**，避免默认拉到 2.5 GB 的 CUDA 版：

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

有 NVIDIA 显卡又想要速度的同学，可以把第一行换成自己的 CUDA wheel（`cu121`/`cu124`），
但**同机对照**时基线必须用同一个环境。

验证：

```powershell
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

### 3.3 git 不在 PATH 时

本机实测：git 2.55 已安装在 `C:\Program Files\Git\cmd\git.exe`，但没有加入 PATH。
两种处理方式（任选其一）：

```powershell
# 方式一：当前会话临时生效
$env:Path += ';C:\Program Files\Git\cmd'

# 方式二：永久加入用户 PATH（无需管理员）
[Environment]::SetEnvironmentVariable(
  'Path',
  [Environment]::GetEnvironmentVariable('Path','User') + ';C:\Program Files\Git\cmd',
  'User')
```

首次提交前还要配置身份（只配一次）：

```powershell
git config --global user.name  "你的名字"
git config --global user.email "你的邮箱"
```

### 3.4 模型下载

```powershell
# 国内镜像（推荐）
$env:HF_ENDPOINT = 'https://hf-mirror.com'
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')"
```

模型缓存默认在 `%USERPROFILE%\.cache\huggingface`，**不要**提交进仓库；四人只下载一份互相拷也行
（把 `HF_HOME` 指到同一个目录即可）。

### 3.5 硬件与时间预算（预估，以 Day 1 实测为准）

| 设备 | 精度 | 单题耗时（16 token） | 200 题基线 |
| :--- | :--- | ---: | ---: |
| CPU 8 核 | float32 | 1.5–3 s | 5–10 min |
| RTX 3060 级 | float16 | 0.1–0.3 s | < 1 min |

实测值请填进 `docs/实验记录总表.md`，并把 `summary.json` 里的 `env` 字段一起保留。

---

## 4. 数据准备

```powershell
# 走 HuggingFace（默认；国内先设 $env:HF_ENDPOINT='https://hf-mirror.com'）
python src/prepare_data.py --source hf

# 或：已手动下载 C-Eval 官方数据（含 val/ 与 dev/ 的 CSV 目录）
python src/prepare_data.py --source local --raw-dir data/raw/ceval
```

产物：

| 文件 | 内容 |
| :--- | :--- |
| `data/ceval_subset_200.json` | 200 题，字段 `subject/qid/question/options{ABCD}/answer/source_split` |
| `data/ceval_fewshot_pool.json` | 每科 3 题 dev 示例，评测时固定取前 2 题 |
| `data/subset_manifest.json` | 两个文件的 md5、种子、每科题量、来源、生成时间 |

脚本内置断言：总题数 200、每科恰好 40、学科数 5、少样本池与评测集**零交集**（含题干重复检查）。
任一条不通过直接报错退出，不会产出带病数据。

> 学科为什么是 `college_economics / accountant / physician / civil_servant /
> urban_and_rural_planner`，以及为什么不用 test 划分：见 [`docs/数据集说明.md`](docs/数据集说明.md)。

**核对全组题目一致**：

```powershell
(Get-FileHash data/ceval_subset_200.json -Algorithm MD5).Hash
```

---

## 5. 快速开始

### 5.1 自检（不加载模型、不联网，30 秒）

```powershell
python tests/test_extract.py     # 答案抽取器 18 条边界用例
python tests/test_pipeline.py    # 用合成数据跑通"抽样 → 自检 → 模板拼接"
python src/run_eval.py --config configs/base.json --dry-run   # 打印第 1 题完整输入
```

前两条不需要模型和数据，装完 Python 就能跑，适合在装环境的空档先确认代码没问题。

### 5.2 5 题试跑（Day 1 必做）

```powershell
python src/run_eval.py --config configs/base.json --limit 5 --exp_id try_base_A
```

看三件事：单题耗时、输出长度、有没有大面积截断。若单题 > 5 s，把 `max_new_tokens` 降到 12
并**全组同步**（写进 `docs/数据集说明.md` 的变更记录）。

### 5.3 200 题基线

```powershell
python src/run_eval.py --config configs/base.json --exp_id exp00_base_A --note "自己的设备名"
```

`--exp_id` 必须带成员后缀（`_A/_B/_C/_D`），因为**每人的基线只和自己的改进比**。

---

## 6. 实验矩阵与命令

| 编号 | 负责人 | 分支 | 唯一变量 | 命令 |
| :--- | :--- | :--- | :--- | :--- |
| `exp00_base_*` | 全员 | `main` | 无 | `--config configs/base.json` |
| `exp01_prompt` | A | `feat/prompt` | 输出契约强度 | `--config configs/prompt.json` |
| `exp02_fewshot` | B | `feat/fewshot` | 加 2-shot | `--config configs/fewshot.json` |
| `exp03_voting` | C | `feat/voting` | 3 采样多数投票（600 次调用） | `--config configs/vote.json` |
| `exp04_batch` | D | `feat/batch` | `batch_size=4/8` | `--config configs/batch.json --batch_size 8` |
| `exp05_final` | D | `feat/final` | 组合方案（Day 2 晚定） | `--config configs/final.json` |
| `exp06_ablation` | D | `feat/final` | 去掉一个模块 | `--config configs/ablation.json` |

`configs/final.json` 目前是占位：Day 2 晚确定组合后再改，改完在 `docs/实验记录总表.md` 里写明组合内容。

---

## 7. 指标与评分规则

### 7.1 计时口径（全组统一，别自己改）

> **从进入 `model.generate()` 前一行开始，到它返回后一行结束。**
> 不含模型加载、题目读取、tokenize、decode 和写日志。

这条写死在 `src/run_eval.py` 里；批量推理时按 `批耗时 / 批内题数` 分摊到每题，并在 JSONL 的
`batch_size` 字段里留痕。

### 7.2 答案抽取（冻结，见 `src/extract.py`）

按优先级命中即停，同一规则多次命中取**最后一次**：

1. `答案：X` / `答案: X` / `答案是X`；
2. 独占一行的单个字母（可带句号）；
3. `选 X` / `应选 X`。

都不命中 → 该题**计错**且计入**格式无效**。禁止在实验分支里另写正则；改解析器要全量重评并在文档说明。

### 7.3 指标

| 指标 | 定义 |
| :--- | :--- |
| `accuracy` | 答对题数 / 全部评测题数（总分母恒为 200，错题不剔除） |
| `format_valid_rate` | 抽取到唯一选项的题数 / 全部题数 |
| `truncation_rate` | 达到 `max_new_tokens` 仍未结束的题数 / 全部题数 |
| `total_elapsed_s` | 逐题耗时之和（只含 generate） |
| `avg_elapsed_ms` | 总耗时 / 题数 |
| `n_calls` | 调用次数（投票实验 = 600） |

---

## 8. 日志与实验记录

每次运行自动写入：

```
logs/<exp_id>/<exp_id>.jsonl   ← 一行一题，含完整输入、原始输出、抽取结果、耗时
logs/<exp_id>/summary.json     ← 汇总 + 环境快照 + commit + 子集 md5
```

`summary.json` 里有 `git_commit`、`subset_md5`、`env.torch/transformers`，用来证明"这组数字是在这个
commit、这份题目、这个环境上跑出来的"。

生成对照表并与手填表逐格核对：

```powershell
python src/report.py --logs logs
python src/report.py --logs logs --out docs/实验记录总表_自动生成.md
```

> `logs/` 下的文件是正式实验证据，**必须提交**，不要加进 `.gitignore`。

---

## 9. 分支与提交约定

- `main`：只接受已跑完对照实验的合并；
- `feat/prompt`、`feat/fewshot`、`feat/voting`、`feat/batch`：一人一分支；
- 提交信息建议：

```
feat(prompt): 输出契约模板 strict_fmt
exp(prompt): exp01_prompt_A 200题完成 acc=0.xx 耗时 xx s
data: 锁定 200 题子集 md5=xxxx
docs: 补充交叉复现记录
```

- 每条正式结果都必须对应一个 commit，并把这个 hash 填进实验总表。

### 9.1 推送到远端（成员 A）

先在 GitHub / Gitee 上建一个**空仓库**（不要勾选自动生成 README），然后：

```powershell
git remote add origin https://github.com/<你的账号>/ceval-slm.git
git push -u origin main
```

推完把仓库地址填回《分工.md》的「Git 仓库地址」，并拉三位队友进来：

```powershell
# 队友第一次拉取
git clone https://github.com/<你的账号>/ceval-slm.git
```

> `logs/` 里是真实运行记录、`data/` 里是 200 道公开题库，都可以公开；模型权重不入库。
> 若不想公开，把仓库设为私有并添加协作者即可。

---

## 10. 复现自检清单

- [ ] 干净环境按 §3 装完依赖，`python -c "import torch"` 通过
- [ ] `python tests/test_extract.py` 全绿
- [ ] `python src/prepare_data.py --source hf` 产出的子集 md5 与组内一致
- [ ] `--limit 5` 试跑能跑通，5 条 JSONL 字段齐全
- [ ] `--dry-run` 打印的输入里没有泄露答案
- [ ] `logs/` 里能查到 commit、环境版本、子集 md5

---

## 11. 常见问题

**HF 下载超时**：设 `$env:HF_ENDPOINT='https://hf-mirror.com'`；或在本机下载后把 `HF_HOME` 指过去。

**torch 装成了 CUDA 版**：`pip uninstall torch` 后按 §3.2 的 CPU index 重装。

**内存不足**：0.5B 模型 fp32 约 2 GB 内存 + 推理开销；降到 `batch_size=1`，或关掉其他程序。

**截断率高（>20%）**：先看 `raw_output` 是不是模型在解释；把 `max_new_tokens` 提到 16~24 并全组同步，
截断的题**照常计错**，不单独重跑。

**四个人基线差很多**：正常。精度、内存带宽、是否用显卡都不同；每人只和自己的改进比。

**报错"已有日志"**：换一个 `--exp_id`；确实要覆盖再加 `--overwrite`（正式实验不建议）。

---

## 12. 来源与许可

- 数据：[C-Eval](https://github.com/hkust-nlp/ceval)（`ceval/ceval-exam`，学术使用需遵守其许可与引用要求）；
- 模型：[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)（Apache-2.0）；
- 推理框架：PyTorch + HuggingFace Transformers；
- 评测代码：本组自写，`src/` 下四个模块为原创实现，未复制第三方评测脚本。

AI 辅助使用情况见 [`docs/AI辅助声明.md`](docs/AI辅助声明.md)。

---

## 13. 当前状态

- 2026-11-25：仓库建立，目录结构、配置、脚本、文档模板就位；
- 2026-11-25：`prepare_data.py` 已生成 200 题子集（md5 `496fefb6…`）与 15 题少样本池，
  数据冻结在 tag **`v0.1-data`**（详见 `docs/数据集说明.md`）；
- 待办：全员装环境 → `--limit 5` 试跑 → 各自跑 200 题基线（`exp00_base_A/B/C/D`）。
