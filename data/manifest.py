from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


MANIFEST_COLUMNS = [
    "pair_id", "dataset", "subset", "visible_path", "infrared_path",
    "visible_filename", "infrared_filename", "visible_width", "visible_height",
    "infrared_width", "infrared_height", "visible_channels", "infrared_channels",
    "infrared_is_replicated_gray", "original_split", "new_split", "official_split",
    "controlled_split", "pairing_method", "registration_status",
    "registration_score", "manual_registration_status", "notes",
]


def empty_manifest() -> pd.DataFrame:
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def load_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_csv(path, keep_default_na=False)
    missing = [column for column in MANIFEST_COLUMNS if column not in frame.columns]
    if missing == ["manual_registration_status"]:
        frame["manual_registration_status"] = ""
        missing = []
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    if frame["pair_id"].duplicated().any():
        duplicates = frame.loc[frame["pair_id"].duplicated(False), "pair_id"].tolist()
        raise ValueError(f"Duplicate pair_id values: {duplicates[:10]}")
    return frame


def save_manifest(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    for column in MANIFEST_COLUMNS:
        if column not in output:
            output[column] = ""
    output[MANIFEST_COLUMNS].to_csv(path, index=False)


def manifest_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_manifest_path(value: str, manifest_path: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    resolved_manifest = Path(manifest_path).resolve()
    project_root = next(
        (
            parent
            for parent in resolved_manifest.parents
            if (parent / "pyproject.toml").is_file() or (parent / ".git").exists()
        ),
        None,
    )
    if project_root is None:
        project_root = resolved_manifest.parents[2]
    return (project_root / candidate).resolve()
