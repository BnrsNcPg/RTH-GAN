from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import random
import re
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch
import yaml


EXPERIMENT_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]*$")
FORBIDDEN_NAME_PARTS = {"test", "test2", "new", "final", "finalnew", "final_new"}


def read_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def deep_merge(*mappings: dict) -> dict:
    result: dict = {}
    for mapping in mappings:
        for key, value in mapping.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
    return result


def git_commit() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip())
        return f"{commit}-dirty" if dirty else commit
    except Exception:
        return "UNAVAILABLE"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_sha256_lf_normalized(path: str | Path) -> str:
    """Hash a text file after normalizing CRLF/CR line endings to LF."""
    payload = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def git_status_snapshot(repo_root: str | Path | None = None) -> dict:
    """Return an auditable Git snapshot without storing source contents."""
    cwd = Path(repo_root).resolve() if repo_root is not None else None

    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        root = Path(git("rev-parse", "--show-toplevel")).resolve()
        status_text = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).rstrip()
        status_lines = status_text.splitlines() if status_text else []
        tracked_dirty = [line[3:] for line in status_lines if not line.startswith("??")]
        untracked = [line[3:] for line in status_lines if line.startswith("??")]
        return {
            "available": True,
            "root": str(root),
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current") or "DETACHED",
            "clean": not status_lines,
            "tracked_clean": not tracked_dirty,
            "status_porcelain_v1": status_lines,
            "tracked_dirty_paths": tracked_dirty,
            "untracked_paths": untracked,
        }
    except Exception as exc:
        return {"available": False, "error": type(exc).__name__}


def configured_file_hashes(
    mapping: Mapping[str, object],
    keys: Iterable[str],
    *,
    base_dir: str | Path | None = None,
) -> dict[str, dict[str, str]]:
    """Hash configured input files while retaining declared and resolved paths."""
    root = Path(base_dir).resolve() if base_dir is not None else Path.cwd().resolve()
    hashes: dict[str, dict[str, str]] = {}
    for key in keys:
        declared = mapping.get(key)
        if not declared:
            continue
        path = Path(str(declared))
        resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Configured file for {key} does not exist: {resolved}")
        hashes[key] = {
            "configured_path": str(declared),
            "resolved_path": str(resolved),
            "sha256": file_sha256(resolved),
            "sha256_lf_normalized": file_sha256_lf_normalized(resolved),
        }
    return hashes


def canonical_config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def environment_lines(device: str | torch.device | None = None) -> list[str]:
    gpu = "unavailable"
    if torch.cuda.is_available():
        try:
            gpu = torch.cuda.get_device_name(torch.device(device or "cuda"))
        except Exception:
            gpu = torch.cuda.get_device_name(0)
    return [
        f"python={platform.python_version()}",
        f"platform={platform.platform()}",
        f"hostname={socket.gethostname()}",
        f"torch={torch.__version__}",
        f"torchvision={package_version('torchvision')}",
        f"torchmetrics={package_version('torchmetrics')}",
        f"torch-fidelity={package_version('torch-fidelity')}",
        "lpips_implementation=torchmetrics",
        f"cuda={torch.version.cuda or 'unavailable'}",
        f"cudnn={torch.backends.cudnn.version() or 'unavailable'}",
        f"gpu={gpu}",
        f"device={device or 'unspecified'}",
    ]


def write_environment(path: str | Path, device: str | torch.device | None = None) -> None:
    Path(path).write_text("\n".join(environment_lines(device)) + "\n", encoding="utf-8")


def seed_everything(seed: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return {
        "seed": seed,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_guarantee": False,
        "note": "Seeds and cuDNN flags are fixed; exact determinism still depends on hardware and operators.",
    }


def validate_experiment_name(name: str) -> None:
    if not EXPERIMENT_NAME_PATTERN.fullmatch(name):
        raise ValueError("Experiment names must contain only lowercase letters, digits, and underscores")
    parts = set(name.split("_"))
    if parts & FORBIDDEN_NAME_PARTS:
        raise ValueError(f"Non-descriptive experiment name is forbidden: {name}")


def prepare_output_dir(path: str | Path, *, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing experiment directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: dict | list) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
