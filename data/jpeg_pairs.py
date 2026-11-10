"""Native-resolution JPEG intensity data; never calibrated temperature."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.manifest import resolve_manifest_path
from experiment_utils import file_sha256


def _jpeg_path(value, manifest, *, prepared=False):
    if prepared:
        path = Path(value)
        return (path if path.is_absolute() else Path(manifest).resolve().parent / path).resolve()
    return resolve_manifest_path(value, manifest)


def load_jpeg_manifest(manifest):
    frame = pd.read_csv(manifest, keep_default_na=False, dtype={"sequence_id": str})
    required = {"pair_id", "split", "sequence_id", "visible_path", "infrared_path",
                "visible_sha256", "infrared_sha256"}
    if not required <= set(frame):
        raise ValueError(f"JPEG manifest is missing columns: {sorted(required - set(frame))}")
    if frame.empty or frame.pair_id.eq("").any() or frame.pair_id.duplicated().any():
        raise ValueError("JPEG pair IDs must be nonempty and unique")
    if set(frame.split) != {"train", "val", "test"} or frame.sequence_id.eq("").any():
        raise ValueError("JPEG manifest requires train/val/test and explicit sequence groups")
    if "preparation_schema" in frame:
        _validate_landsat_preparation(frame, manifest)
    else:
        review_columns = {"manual_registration_status", "sequence_registration_status"}
        if not review_columns <= set(frame):
            raise ValueError(f"JPEG manifest is missing columns: {sorted(review_columns - set(frame))}")
        accepted = {"PASS_MANUAL", "PASS_WITH_LOCAL_VARIATION"}
        if not frame.sequence_registration_status.isin(accepted).all():
            raise ValueError("JPEG sequences require an existing accepted manual review")
        if frame.manual_registration_status.isin({"FAIL", "REVIEW"}).any():
            raise ValueError("Resolve outstanding manual registration decisions before training")
    for column in ("sequence_id", "visible_sha256", "infrared_sha256"):
        if frame[column].eq("").any() or frame.groupby(column).split.nunique().gt(1).any():
            raise ValueError(f"Missing identity or cross-split leakage: {column}")
    for column in ("visible_path", "infrared_path"):
        paths = frame[column].map(lambda value: str(_jpeg_path(value, manifest, prepared="preparation_schema" in frame)))
        if frame[column].eq("").any() or frame.groupby(paths).split.nunique().gt(1).any():
            raise ValueError(f"Missing source or cross-split leakage: {column}")
    return frame


def _validate_landsat_preparation(frame, manifest):
    """Source-grid admission is confined to the derived Landsat schema."""
    import json
    import re

    bands = ("red", "green", "blue", "st", "qa_pixel", "qa_radsat", "st_qa")
    required = {"scene_id", "sensor", "registration_basis", "height", "width",
                "source_manifest_sha256", "normalization_sha256", "preparation_report"}
    required.update(f"source_{band}_{suffix}" for band in bands for suffix in ("path", "sha256"))
    if not required <= set(frame):
        raise ValueError(f"Prepared Landsat JPEG manifest is missing columns: {sorted(required-set(frame))}")
    if (not frame.preparation_schema.eq("landsat_jpeg_v1").all()
            or not frame.registration_basis.eq("shared_source_product_grid").all()
            or frame.sensor.nunique() != 1 or not frame.sensor.isin({"landsat5", "landsat7", "landsat8"}).all()
            or not frame.scene_id.eq(frame.sequence_id).all()
            or frame.scene_id.eq("").any()):
        raise ValueError("Invalid prepared Landsat schema, sensor, scene identity, or registration basis")
    if not (pd.to_numeric(frame.height, errors="raise").eq(256).all()
            and pd.to_numeric(frame.width, errors="raise").eq(256).all()):
        raise ValueError("Prepared Landsat JPEG windows must be 256x256 without resizing")
    for column in ("source_manifest_sha256", "normalization_sha256", *(f"source_{b}_sha256" for b in bands)):
        if not frame[column].map(lambda value: bool(re.fullmatch(r"[0-9a-f]{64}", str(value)))).all():
            raise ValueError(f"Invalid prepared Landsat hash: {column}")
    for band in bands:
        column = f"source_{band}_path"
        paths = frame[column].map(lambda value: str(_jpeg_path(value, manifest, prepared=True)))
        if frame[column].eq("").any() or frame.groupby(paths).split.nunique().gt(1).any():
            raise ValueError(f"Missing source or cross-split leakage: {column}")
        if band in {"red", "green", "blue", "st"}:
            if frame.groupby(f"source_{band}_sha256").split.nunique().gt(1).any():
                raise ValueError(f"Science content cross-split leakage: {band}")
    if frame.preparation_report.nunique() != 1 or frame.source_manifest_sha256.nunique() != 1 or frame.normalization_sha256.nunique() != 1:
        raise ValueError("Prepared JPEGs must share one preparation report and normalization")
    report_path = _jpeg_path(frame.preparation_report.iloc[0], manifest, prepared=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("status") != "complete" or report.get("manifest_sha256") != file_sha256(manifest)
            or report.get("source_manifest_sha256") != frame.source_manifest_sha256.iloc[0]
            or report.get("normalization_sha256") != frame.normalization_sha256.iloc[0]):
        raise ValueError("Prepared JPEG manifest does not match the completed preparation report")
    statistics_path = report_path.parent / "normalization.json"
    if file_sha256(statistics_path) != report["normalization_sha256"]:
        raise ValueError("Prepared JPEG normalization hash changed")


def read_jpeg_pair(row, manifest, *, verify_hashes=False):
    arrays = []
    for modality in ("visible", "infrared"):
        path = _jpeg_path(row[modality + "_path"], manifest,
                          prepared=row.get("preparation_schema") == "landsat_jpeg_v1")
        if verify_hashes and file_sha256(path) != row[modality + "_sha256"]:
            raise ValueError(f"Source hash changed: {path}")
        with Image.open(path) as image:
            if image.format != "JPEG" or image.mode not in {"RGB", "L"}:
                raise ValueError(f"Expected an 8-bit RGB/grayscale JPEG: {path}")
            array = np.asarray(image).copy()
        if modality == "visible":
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError("Visible JPEG must contain RGB channels")
        elif array.ndim == 3:
            if not (np.array_equal(array[..., 0], array[..., 1]) and
                    np.array_equal(array[..., 0], array[..., 2])):
                raise ValueError("Infrared RGB JPEG must contain replicated grayscale channels")
            array = array[..., 0]
        arrays.append(array)
    visible, infrared = arrays
    if visible.shape[:2] != infrared.shape or any(size % 32 for size in infrared.shape):
        raise ValueError("JPEG pairs must share native dimensions divisible by 32; no resampling")
    return (torch.from_numpy(visible).permute(2, 0, 1).float() / 255,
            torch.from_numpy(infrared.copy()).unsqueeze(0).float() / 127.5 - 1)


class JPEGIntensityDataset(Dataset):
    def __init__(self, manifest, *, split, training=False, verify_hashes=True):
        self.manifest = Path(manifest)
        frame = load_jpeg_manifest(manifest)
        self.frame = frame.loc[frame.split.eq(split)].reset_index(drop=True)
        if self.frame.empty or (training and split != "train"):
            raise ValueError("Require a nonempty split and train-only augmentation")
        self.training = training
        # Validate every admitted source before any optimizer step, including hashes.
        sizes = {tuple(read_jpeg_pair(row, self.manifest, verify_hashes=verify_hashes)[1].shape)
                 for _, row in self.frame.iterrows()}
        if len(sizes) != 1:
            raise ValueError("JPEG split must have one shared native size")
        self.image_shape = next(iter(sizes))
        if "preparation_schema" in self.frame and self.image_shape != (1, 256, 256):
            raise ValueError("Prepared Landsat JPEG pixels must retain the declared 256x256 source window")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        visible, infrared = read_jpeg_pair(row, self.manifest)
        # No cropping, resizing, registration, or source writes.
        if self.training and random.random() < .5:
            visible, infrared = visible.flip(-1), infrared.flip(-1)
        item = {"visible": visible, "infrared": infrared, "context": {},
                "pair_id": row.pair_id, "sequence_id": row.sequence_id}
        if "preparation_schema" in self.frame:
            item.update(scene_id=row.scene_id, registration_basis=row.registration_basis)
        else:
            item.update(manual_registration_status=row.manual_registration_status,
                        sequence_registration_status=row.sequence_registration_status)
        return item
