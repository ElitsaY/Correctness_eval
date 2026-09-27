"""Seeding, device selection and run manifests."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

TRACKED_PACKAGES = (
    "cap-eval",
    "numpy",
    "pandas",
    "scipy",
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "sentencepiece",
    "scikit-learn",
    "sacrebleu",
    "rouge-score",
    "nltk",
    "bert-score",
    "unbabel-comet",
)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and (if installed) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    if deterministic:
        # Required by cuBLAS for deterministic matmuls.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def resolve_device(preference: str) -> str:
    import torch

    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    path: str | Path,
    *,
    command: str,
    config: dict[str, Any],
    inputs: dict[str, str | Path],
    extra: dict[str, Any] | None = None,
) -> None:
    """Record everything needed to reproduce an output next to the output."""
    manifest = {
        "command": command,
        "argv": sys.argv,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": _git_state(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _package_versions(),
        "hardware": _hardware(),
        "inputs": {name: {"path": str(p), "sha256": sha256_file(p)} for name, p in inputs.items()},
        "config": config,
        **(extra or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    logger.info("Wrote run manifest to %s", path)


def read_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _hardware() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"cuda": False}
    info: dict[str, Any] = {"cuda": torch.cuda.is_available()}
    if info["cuda"]:
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
    return info
