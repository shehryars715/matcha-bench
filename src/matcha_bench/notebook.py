"""Helpers for the reproduction notebooks (notebooks/0*.ipynb): run a pipeline step with its output
shown in the cell, and compare a result with the number reported for it."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .config import REPO_ROOT


def setup() -> Path:
    """Work from the repository root, wherever Jupyter was started."""
    os.chdir(REPO_ROOT)
    return REPO_ROOT


def step(*args, env: dict | None = None) -> None:
    """Run `python <args>` from the repository root, streaming its output; raise if it fails."""
    cmd = [sys.executable, *map(str, args)]
    print("$ python " + " ".join(map(str, args)), flush=True)
    full_env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", **(env or {})}
    with subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          encoding="utf-8", errors="replace", env=full_env) as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
    if proc.returncode:
        raise RuntimeError(f"step failed with exit code {proc.returncode}: {' '.join(map(str, args))}")


def check(name: str, got, expected, tol: float | None = None) -> bool:
    """Print whether a result matches the reported value (within `tol` for numbers)."""
    same = abs(got - expected) <= tol if tol is not None else got == expected
    print(f"{'ok' if same else 'DIFFERENT':9} {name}: {got} (reported {expected})")
    return same
