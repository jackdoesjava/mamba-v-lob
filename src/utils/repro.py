"""Seeding and provenance, so an exported certificate can be traced to the run that made it."""

from __future__ import annotations

import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]  # src/utils/repro.py -> repo root


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed the global RNGs.

    Does not reach explicitly constructed generators: np.random.default_rng in stats.py and
    the torch.Generator passed to the training DataLoader are seeded at their call sites.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CPython fixes hash randomisation at startup, so this only reaches child processes
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # cuDNN autotuning picks a different algorithm run to run, so bitwise repro needs it off
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        sha = out.stdout.strip()
        if not sha:
            return "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except Exception:
        return "unknown"


def provenance(seed: int | None = None) -> dict:
    """Everything needed to reproduce a run, embedded alongside its outputs."""
    return {
        "git_sha": git_sha(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "seed": seed,
    }
