#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""可追溯信息：当前代码版本（git commit）。

为什么需要它：课程要求"每条实验结果对应一个 commit"。但 Windows 上经常出现
"装了 git 却没进 PATH"（本机就是这种情况），此时 ``subprocess`` 调不到 git，
如果不做回退，日志里就会写成 ``unknown``，等于丢掉可追溯性。

回退顺序：PATH 里的 git → 常见安装路径 → 直接读 ``.git/HEAD`` 与 ``refs/``。
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

_GIT_CANDIDATES = (
    r"C:\Program Files\Git\cmd\git.exe",
    r"C:\Program Files\Git\bin\git.exe",
    r"C:\Program Files (x86)\Git\cmd\git.exe",
)


def _git_exe() -> Optional[str]:
    found = shutil.which("git")
    if found:
        return found
    for candidate in _GIT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def _read_file(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _head_from_git_dir(root: Path) -> Optional[str]:
    """不依赖 git 可执行文件，直接解析 ``.git`` 目录。"""
    git_dir = root / ".git"
    if git_dir.is_file():  # worktree 场景：.git 是个 "gitdir: xxx" 文本
        content = _read_file(git_dir) or ""
        if not content.startswith("gitdir:"):
            return None
        git_dir = (root / content.split("gitdir:", 1)[1].strip()).resolve()
    head = _read_file(git_dir / "HEAD")
    if not head:
        return None
    if head.startswith("ref:"):
        ref = head.split("ref:", 1)[1].strip()
        sha = _read_file(git_dir / ref)
        if sha:
            return sha[:7]
        packed = _read_file(git_dir / "packed-refs") or ""
        for line in packed.splitlines():
            if line.endswith(" " + ref) or line.endswith("\t" + ref):
                return line.split()[0][:7]
        return None
    return head[:7]  # detached HEAD：直接是 sha


def git_revision(root: Optional[Path] = None) -> str:
    """返回短 commit hash；工作区有未提交改动时追加 ``-dirty``。拿不到就返回 ``unknown``。"""
    root = Path(root) if root else Path.cwd()
    git = _git_exe()
    if git:
        # 必须显式指定 utf-8：Windows 默认用 GBK 解码子进程输出，
        # 仓库路径或文件名含中文时（本仓库就是）会直接解码失败。
        try:
            out = subprocess.run([git, "rev-parse", "--short", "HEAD"], cwd=str(root),
                                 capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=10)
            if out.returncode == 0 and out.stdout.strip():
                dirty = subprocess.run([git, "status", "--porcelain", "--untracked-files=no"],
                                       cwd=str(root), capture_output=True, text=True,
                                       encoding="utf-8", errors="replace", timeout=10)
                suffix = "-dirty" if (dirty.stdout or "").strip() else ""
                return out.stdout.strip() + suffix
        except (OSError, subprocess.SubprocessError):
            pass
    fallback = _head_from_git_dir(root)
    return fallback or "unknown"
