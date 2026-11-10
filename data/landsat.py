"""Read-only Landsat C2 L2 quantitative patches with scene-exclusive splits."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from rtd.thermal import TemperatureNormalization, ThermalRadiativeExtractor


PATH_COLUMNS = ("red_path", "green_path", "blue_path", "st_path", "qa_pixel_path", "qa_radsat_path", "st_qa_path")
REQUIRED_COLUMNS = ("pair_id", "scene_id", "sensor", "split", "top", "left", "height", "width", "context_path", *PATH_COLUMNS)


def load_landsat_manifest(path):
    frame = pd.read_csv(path, keep_default_na=False)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame))
    if missing:
        raise ValueError(f"Landsat manifest is missing columns: {missing}")
    if frame.empty or frame.pair_id.duplicated().any() or frame.pair_id.eq("").any() or frame.scene_id.eq("").any():
        raise ValueError("Landsat manifest needs nonempty unique pair IDs and scene IDs")
    if not set(frame.split) <= {"train", "val", "test"}:
        raise ValueError("Landsat splits must be train, val, or test")
    if frame.groupby("scene_id").split.nunique().gt(1).any():
        raise ValueError("Scene leakage: a scene occurs in multiple splits")
    for name in PATH_COLUMNS:
        sources = frame[name].map(lambda value: str(resolve_source(value, path).resolve()))
        if frame.groupby(sources).split.nunique().gt(1).any():
            raise ValueError("Source leakage: a source raster occurs in multiple splits")
    if frame.sensor.nunique() != 1 or not set(frame.sensor) <= {"landsat5", "landsat7", "landsat8"}:
        raise ValueError("Use one supported Landsat sensor per manifest/experiment")
    if (frame.sensor == "landsat7").any():
        if "slc_status" not in frame or not frame.slc_status.eq("SLC_ON").all():
            raise ValueError("Manuscript Landsat-7 data requires documented SLC_ON acquisitions")
    for name in ("top", "left", "height", "width"):
        numbers = pd.to_numeric(frame[name], errors="raise")
        if not np.isfinite(numbers).all() or (numbers != np.floor(numbers)).any():
            raise ValueError(f"{name} must contain finite integers")
        if (numbers < (0 if name in {"top", "left"} else 1)).any():
            raise ValueError(f"Invalid patch {name}")
        frame[name] = numbers.astype(int)
    if not frame.height.eq(frame.width).all():
        raise ValueError("Paired manuscript patches must be square for quarter-turn augmentation")
    if frame[list(PATH_COLUMNS)].eq("").any().any():
        raise ValueError("All science and QA source paths are required")
    return frame


def validate_quality_config(quality):
    bits = quality.get("qa_pixel_reject_bits")
    uncertainty = quality.get("maximum_st_uncertainty_kelvin")
    if type(bits) is not int or not 0 < bits < 65536 or bits & 27 != 27:
        raise ValueError("QA_PIXEL rejection mask must include fill, dilated cloud, cloud, and shadow bits")
    if not isinstance(uncertainty, (float, int)) or not math.isfinite(uncertainty) or uncertainty <= 0:
        raise ValueError("Specify a positive maximum_st_uncertainty_kelvin")


def resolve_source(value, manifest):
    path = Path(value)
    return path if path.is_absolute() else Path(manifest).resolve().parent / path


def _read_grid_patch(path, row, expected_grid=None):
    # Optional dependency: historical aerial/Pix2Pix users do not need GDAL.
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(path) as source:
        grid = (source.crs, source.transform, source.height, source.width)
        if source.count != 1 or source.crs is None:
            raise ValueError(f"Expected one georeferenced science/QA band: {path}")
        if expected_grid is not None and grid != expected_grid:
            raise ValueError(f"Source grids differ; no automatic resampling or registration: {path}")
        if row.top + row.height > source.height or row.left + row.width > source.width:
            raise ValueError(f"Patch lies outside source grid: {row.pair_id}")
        window = Window(row.left, row.top, row.width, row.height)
        values = source.read(1, window=window)
        valid = source.read_masks(1, window=window) != 0
    if not valid.all() or not np.isfinite(values).all():
        raise ValueError(f"Source patch contains nodata/nonfinite pixels: {path}")
    return values, grid


def read_landsat_patch(row, manifest, quality):
    validate_quality_config(quality)
    if row.sensor == "landsat8" and quality["qa_pixel_reject_bits"] & 4 == 0:
        raise ValueError("Landsat-8 quality mask must also reject high-confidence cirrus")
    arrays, grid = {}, None
    for name in PATH_COLUMNS:
        values, grid = _read_grid_patch(resolve_source(row[name], manifest), row, grid)
        if values.dtype != np.dtype("uint16"):
            raise ValueError(f"Landsat C2 L2 raw DN/QA input must be uint16: {name}")
        arrays[name] = values
    rgb_dn = np.stack([arrays[name] for name in PATH_COLUMNS[:3]])
    st_dn = arrays["st_path"]
    valid = ((arrays["qa_pixel_path"] & quality["qa_pixel_reject_bits"]) == 0)
    valid &= arrays["qa_radsat_path"] == 0
    valid &= arrays["st_qa_path"].astype(np.float64) * 0.01 <= quality["maximum_st_uncertainty_kelvin"]
    valid &= ((rgb_dn >= 7273) & (rgb_dn <= 43636)).all(axis=0)
    valid &= st_dn >= 293
    if not valid.all():
        raise ValueError(f"Patch fails declared QA/range policy: {row.pair_id} ({int((~valid).sum())} invalid pixels)")
    rgb = torch.from_numpy(rgb_dn.astype(np.float32) * 0.0000275 - 0.2)
    kelvin = torch.from_numpy(st_dn.astype(np.float32) * 0.00341802 + 149.0).unsqueeze(0)
    return rgb, kelvin, grid


def fit_landsat_normalization(manifest, quality):
    frame = load_landsat_manifest(manifest)
    training = frame.loc[frame.split.eq("train")]
    if training.empty:
        raise ValueError("Cannot fit temperature normalization without training data")
    minima, maxima = [], []
    temperature_min, temperature_max = float("inf"), float("-inf")
    for _, row in training.iterrows():
        rgb, kelvin, _ = read_landsat_patch(row, manifest, quality)
        minima.append(rgb.flatten(1).amin(1))
        maxima.append(rgb.flatten(1).amax(1))
        temperature_min = min(temperature_min, float(kelvin.min()))
        temperature_max = max(temperature_max, float(kelvin.max()))
    minimum = torch.stack(minima).amin(0)
    maximum = torch.stack(maxima).amax(0)
    TemperatureNormalization(temperature_min, temperature_max, source_split="train")
    if (maximum <= minimum).any():
        raise ValueError("Training-subset visible channel ranges must be nonzero")
    return {
        "schema_version": 1, "source_split": "train",
        "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        "sensor": str(frame.sensor.iloc[0]), "quality": dict(quality),
        "temperature": {"minimum_kelvin": temperature_min, "maximum_kelvin": temperature_max, "source_split": "train"},
        "visible_minimum": minimum.tolist(),
        "visible_maximum": maximum.tolist(),
    }


class LandsatPairedDataset(Dataset):
    def __init__(self, manifest, statistics, *, split, training=False, require_context=True):
        self.manifest = Path(manifest)
        frame = load_landsat_manifest(manifest)
        self.frame = frame.loc[frame.split.eq(split)].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"Empty Landsat {split} split")
        if training and split != "train":
            raise ValueError("Spatial training augmentation is only allowed for the train split")
        self.statistics = dict(statistics)
        if statistics.get("source_split") != "train" or statistics.get("manifest_sha256") != hashlib.sha256(self.manifest.read_bytes()).hexdigest():
            raise ValueError("Normalization must come from this manifest's train split")
        if statistics.get("sensor") != frame.sensor.iloc[0]:
            raise ValueError("Statistics sensor does not match the manifest")
        self.normalization = TemperatureNormalization(**statistics["temperature"])
        self.minimum = torch.tensor(statistics["visible_minimum"]).view(3, 1, 1)
        self.span = torch.tensor(statistics["visible_maximum"]).view(3, 1, 1) - self.minimum
        if not torch.isfinite(self.minimum).all() or not torch.isfinite(self.span).all() or (self.span <= 0).any():
            raise ValueError("Visible training-subset channel ranges must be finite and nonzero")
        validate_quality_config(statistics["quality"])
        self.training, self.require_context = training, require_context

    def __len__(self):
        return len(self.frame)

    def _context(self, row, grid):
        path = resolve_source(row.context_path, self.manifest)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("sensor") != row.sensor or payload.get("radiance_units") != "W m^-2 sr^-1 um^-1" or not payload.get("provenance"):
            raise ValueError("Thermal context requires matching sensor, spectral radiance units, and provenance")
        context = payload["context"]
        if set(context) != set(ThermalRadiativeExtractor.CONTEXT_FIELDS):
            raise ValueError("Thermal context fields are incomplete or unknown")
        result = {}
        for name, value in context.items():
            if isinstance(value, str):
                source = Path(value) if Path(value).is_absolute() else path.parent / value
                array, _ = _read_grid_patch(source, row, grid)
                result[name] = torch.from_numpy(array.astype(np.float32)).unsqueeze(0)
            else:
                # Expand scalars so batches may mix scalar and raster contexts.
                result[name] = torch.tensor(value, dtype=torch.float32).view(1, 1, 1).expand(1, row.height, row.width).clone()
        return result

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        rgb, kelvin, grid = read_landsat_patch(row, self.manifest, self.statistics["quality"])
        visible = ((rgb - self.minimum) / self.span).clamp(0, 1)
        infrared = self.normalization.normalize(kelvin)
        context = self._context(row, grid) if self.require_context else {}
        if self.training:
            rotations, flip = random.randrange(4), random.random() < 0.5
            def augment(value):
                if value.shape[-2:] == (1, 1):
                    return value
                value = torch.rot90(value, rotations, dims=(-2, -1))
                return value.flip(-1) if flip else value
            visible, infrared, kelvin = map(augment, (visible, infrared, kelvin))
            context = {name: augment(value) for name, value in context.items()}
        return {"visible": visible, "infrared": infrared, "temperature_kelvin": kelvin,
                "context": context, "pair_id": row.pair_id, "scene_id": row.scene_id}
