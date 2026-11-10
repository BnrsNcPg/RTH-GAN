import hashlib
import json
import random
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

rasterio = pytest.importorskip("rasterio")
from data.landsat import LandsatPairedDataset, fit_landsat_normalization, load_landsat_manifest


QUALITY = {"qa_pixel_reject_bits": 31, "maximum_st_uncertainty_kelvin": 2.0}


def write_band(path, values, *, shift=0):
    with rasterio.open(path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1],
                       count=1, dtype=values.dtype, crs="EPSG:32650", transform=rasterio.Affine(30, 0, 500000 + shift, 0, -30, 4000000)) as target:
        target.write(values, 1)


@pytest.fixture
def landsat_manifest(tmp_path):
    grid = np.arange(64, dtype=np.uint16).reshape(8, 8)
    values = {"red_path": 10000 + grid, "green_path": 12000 + grid,
              "blue_path": 15000 + grid, "st_path": 40000 + grid,
              "qa_pixel_path": np.zeros_like(grid), "qa_radsat_path": np.zeros_like(grid),
              "st_qa_path": np.full_like(grid, 100)}
    paths = {}
    for name, array in values.items():
        filename = name + ".tif"
        write_band(tmp_path / filename, array)
        paths[name] = filename
    kelvin = (40000 + grid).astype(np.float32) * 0.00341802 + 149
    write_band(tmp_path / "down.tif", kelvin)
    (tmp_path / "context.json").write_text(json.dumps({"sensor": "landsat8", "radiance_units": "W m^-2 sr^-1 um^-1",
        "provenance": "synthetic unit-test fixture, not measured metadata",
        "context": {"emissivity": .95, "transmittance": .8, "downwelling_radiance": "down.tif", "upwelling_radiance": 1}}))
    rows = []
    for split in ("train", "val", "test"):
        split_paths = {}
        for key, value in paths.items():
            target = f"{split}_{value}"
            shutil.copyfile(tmp_path / value, tmp_path / target)
            split_paths[key] = target
        rows.append({"pair_id": f"pair_{split}", "scene_id": f"scene_{split}", "sensor": "landsat8", "split": split,
             "top": 0, "left": 0, "height": 8, "width": 8, "context_path": "context.json", **split_paths})
    path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_landsat_calibration_and_train_only_statistics_preserve_raw_files(landsat_manifest):
    source_paths = list(landsat_manifest.parent.glob("*.tif"))
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    statistics = fit_landsat_normalization(landsat_manifest, QUALITY)
    assert statistics["temperature"]["minimum_kelvin"] == pytest.approx(40000 * .00341802 + 149, abs=2e-5)
    item = LandsatPairedDataset(landsat_manifest, statistics, split="train")[0]
    assert item["visible"].shape == (3, 8, 8)
    assert item["infrared"].min() == pytest.approx(-1)
    assert item["infrared"].max() == pytest.approx(1)
    for path in source_paths:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before[path]


def test_landsat_validation_extremes_do_not_affect_fitted_statistics(landsat_manifest):
    frame = pd.read_csv(landsat_manifest)
    write_band(landsat_manifest.parent / "validation_st.tif", np.full((8, 8), 60000, dtype=np.uint16))
    frame.loc[frame.split.eq("val"), "st_path"] = "validation_st.tif"
    frame.to_csv(landsat_manifest, index=False)
    statistics = fit_landsat_normalization(landsat_manifest, QUALITY)
    assert statistics["temperature"]["maximum_kelvin"] < 300
    item = LandsatPairedDataset(landsat_manifest, statistics, split="val", require_context=False)[0]
    assert item["infrared"].min() > 1  # Evaluation values are never silently clipped/refitted.


def test_landsat_spatial_context_gets_identical_augmentation(landsat_manifest):
    statistics = fit_landsat_normalization(landsat_manifest, QUALITY)
    dataset = LandsatPairedDataset(landsat_manifest, statistics, split="train", training=True)
    for seed in range(8):
        random.seed(seed)
        item = dataset[0]
        torch.testing.assert_close(item["temperature_kelvin"], item["context"]["downwelling_radiance"])


@pytest.mark.parametrize("column,value", [("qa_pixel_path", 8), ("qa_radsat_path", 1), ("st_qa_path", 201), ("st_path", 0), ("red_path", 0)])
def test_landsat_invalid_quality_is_rejected(landsat_manifest, column, value):
    path = landsat_manifest.parent / ("train_" + column + ".tif")
    with rasterio.open(path) as source:
        values = source.read(1)
    values[2, 3] = value
    write_band(path, values)
    with pytest.raises(ValueError, match="QA/range policy"):
        fit_landsat_normalization(landsat_manifest, QUALITY)


def test_landsat_rejects_scene_leakage(landsat_manifest):
    frame = pd.read_csv(landsat_manifest)
    frame["scene_id"] = "shared_scene"
    frame.to_csv(landsat_manifest, index=False)
    with pytest.raises(ValueError, match="Scene leakage"):
        load_landsat_manifest(landsat_manifest)


def test_landsat_rejects_grid_mismatch_without_resampling(landsat_manifest):
    path = landsat_manifest.parent / "train_st_path.tif"
    write_band(path, np.full((8, 8), 40000, dtype=np.uint16), shift=30)
    with pytest.raises(ValueError, match="Source grids differ"):
        fit_landsat_normalization(landsat_manifest, QUALITY)


def test_landsat_inference_does_not_load_reference_context(landsat_manifest):
    statistics = fit_landsat_normalization(landsat_manifest, QUALITY)
    (landsat_manifest.parent / "context.json").unlink()
    item = LandsatPairedDataset(landsat_manifest, statistics, split="test", require_context=False)[0]
    assert item["context"] == {}


def test_landsat_rejects_statistics_from_other_manifest(landsat_manifest):
    statistics = fit_landsat_normalization(landsat_manifest, QUALITY)
    statistics["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="this manifest's train split"):
        LandsatPairedDataset(landsat_manifest, statistics, split="train")


def test_landsat_rejects_source_reuse_with_different_scene_labels(landsat_manifest):
    frame = pd.read_csv(landsat_manifest)
    frame.loc[frame.split.eq("val"), "st_path"] = frame.loc[frame.split.eq("train"), "st_path"].iloc[0]
    frame.to_csv(landsat_manifest, index=False)
    with pytest.raises(ValueError, match="Source leakage"):
        load_landsat_manifest(landsat_manifest)


def test_context_scalars_and_spatial_maps_have_batch_compatible_shapes(landsat_manifest):
    statistics = fit_landsat_normalization(landsat_manifest, QUALITY)
    item = LandsatPairedDataset(landsat_manifest, statistics, split="train")[0]
    assert all(value.shape == (1, 8, 8) for value in item["context"].values())
