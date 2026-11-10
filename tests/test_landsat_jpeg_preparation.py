"""Synthetic source-to-JPEG checks; no claim about real manuscript data."""
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

rasterio = pytest.importorskip("rasterio")

from data.landsat import PATH_COLUMNS
from data.landsat_jpeg_preparation import (
    load_preparation_manifest, prepare_landsat_jpeg, quantize_intensity, sha256,
)


def write_band(path, values, *, shift=0, nodata=None):
    with rasterio.open(path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1],
                       count=1, dtype=values.dtype, crs="EPSG:32650", nodata=nodata,
                       transform=rasterio.Affine(30, 0, 500000 + shift, 0, -30, 4000000)) as target:
        target.write(values, 1)


@pytest.fixture
def scene_manifest(tmp_path):
    raw = tmp_path / "source_products"
    raw.mkdir()
    yy, xx = np.indices((513, 513))
    pattern = (yy + xx).astype(np.uint16)
    rows = []
    for split, delta in (("train", 0), ("val", 10000), ("test", -5000)):
        arrays = {"red_path": 10000 + pattern.astype(np.int32) + delta,
                  "green_path": 12000 + pattern.astype(np.int32) + delta,
                  "blue_path": 14000 + pattern.astype(np.int32) + delta,
                  "st_path": 40000 + pattern.astype(np.int32) + delta,
                  "qa_pixel_path": np.zeros_like(pattern), "qa_radsat_path": np.zeros_like(pattern),
                  "st_qa_path": np.full_like(pattern, 100)}
        paths = {}
        for name, values in arrays.items():
            path = raw / f"{split}_{name}.tif"
            write_band(path, values.astype(np.uint16))
            paths[name] = path.relative_to(tmp_path).as_posix()
        rows.append({"scene_id": f"synthetic_{split}", "sensor": "landsat8", "split": split, **paths})
    manifest = tmp_path / "scenes.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def prepare(scene_manifest, output_name="derived", **kwargs):
    return prepare_landsat_jpeg(scene_manifest, scene_manifest.parent / output_name, kwargs or None)


def mutate_band(manifest, split, column, transform):
    frame = pd.read_csv(manifest, keep_default_na=False)
    path = manifest.parent / frame.loc[frame.split.eq(split), column].iloc[0]
    with rasterio.open(path) as source:
        values = source.read(1)
    transform(values)
    write_band(path, values)


def test_preparation_uses_train_bounds_and_preserves_all_source_hashes(scene_manifest):
    raw_paths = list((scene_manifest.parent / "source_products").glob("*.tif")) + [scene_manifest]
    before = {path: sha256(path) for path in raw_paths}
    report = prepare(scene_manifest)
    assert {path: sha256(path) for path in raw_paths} == before
    assert report["status"] == "complete"
    assert report["candidate_counts"] == report["accepted_counts"] == dict(train=4, val=4, test=4)
    assert report["raw_files_modified"] is False
    assert report["normalization"]["visible_minimum"][0] == pytest.approx(10000 * .0000275 - .2)
    # Row/column 512 is an incomplete edge and cannot influence training extrema.
    maximum_dn = 10000 + 511 + 511
    assert report["normalization"]["visible_maximum"][0] == pytest.approx(maximum_dn * .0000275 - .2)
    assert report["normalization"]["temperature_maximum_kelvin"] == pytest.approx((maximum_dn + 30000) * .00341802 + 149)
    assert report["clipping"]["splits"]["train"]["above_fraction"] == [0, 0, 0, 0]
    assert report["clipping"]["splits"]["val"]["above_fraction"] == [1, 1, 1, 1]
    assert report["clipping"]["splits"]["test"]["below_fraction"] == [1, 1, 1, 1]
    assert report["acquisition_date_range"] is None
    assert report["scenes_missing_acquisition_date"] == ["synthetic_train", "synthetic_val", "synthetic_test"]
    assert report["region_names"] == []
    assert len(report["geographic_bounds_wgs84"]) == 4
    output = scene_manifest.parent / "derived"
    manifest = pd.read_csv(output / "manifest.csv")
    assert report["manifest_sha256"] == sha256(output / "manifest.csv")
    assert manifest.preparation_schema.eq("landsat_jpeg_v1").all()
    assert manifest.registration_basis.eq("shared_source_product_grid").all()
    assert manifest.scene_id.equals(manifest.sequence_id)
    assert not any("manual_registration_status" in column for column in manifest)
    assert manifest.normalization_sha256.eq(report["normalization_sha256"]).all()
    assert set(manifest.top) == set(manifest.left) == {0, 256}
    for _, row in manifest.iterrows():
        with Image.open(output / row.visible_path) as image:
            assert image.mode == "RGB" and image.size == (256, 256)
        with Image.open(output / row.infrared_path) as image:
            assert image.mode == "L" and image.size == (256, 256)
        assert row.visible_sha256 == sha256(output / row.visible_path)
    assert report["encoding"]["quality"] == 100
    assert report["encoding"]["rgb_subsampling"] == 0
    assert report["encoding"]["lossy"] is True
    assert report["encoding"]["jpeg_codec_version"]
    assert report["preparation_code_sha256"] == sha256("data/landsat_jpeg_preparation.py")
    # The prepared schema is consumed without inventing aerial registration labels.
    from data.jpeg_pairs import JPEGIntensityDataset
    dataset = JPEGIntensityDataset(output / "manifest.csv", split="train")
    assert dataset[0]["visible"].shape == (3, 256, 256)
    assert dataset[0]["infrared"].shape == (1, 256, 256)


@pytest.mark.parametrize("column,value,reason", [
    ("qa_pixel_path", 4, "qa_pixel"), ("qa_radsat_path", 1, "qa_radsat"),
    ("st_qa_path", 201, "st_uncertainty"), ("red_path", 0, "sr_dn_range"),
])
def test_preparation_rejects_the_entire_bad_window_and_records_why(scene_manifest, column, value, reason):
    mutate_band(scene_manifest, "train", column, lambda array: array.__setitem__((1, 1), value))
    report = prepare(scene_manifest)
    assert report["candidate_counts"]["train"] == 4
    assert report["accepted_counts"]["train"] == 3
    assert report["rejected_counts"]["train"] == 1
    log = pd.read_csv(scene_manifest.parent / "derived" / "candidates.csv")
    failed = log.loc[~log.accepted]
    assert len(failed) == 1
    assert json.loads(failed.iloc[0].failure_pixels_by_reason)[reason] == 1


def test_preparation_accepts_uncertainty_threshold_boundary(scene_manifest):
    mutate_band(scene_manifest, "train", "st_qa_path", lambda array: array.fill(200))
    assert prepare(scene_manifest)["accepted_counts"]["train"] == 4


def test_preparation_refuses_overwrite_and_source_directory_outputs(scene_manifest):
    prepare(scene_manifest)
    with pytest.raises(FileExistsError, match="overwrite"):
        prepare(scene_manifest)
    with pytest.raises(ValueError, match="outside source raster"):
        prepare_landsat_jpeg(scene_manifest, scene_manifest.parent / "source_products" / "new")


def test_preparation_rejects_grid_mismatch_before_creating_output(scene_manifest):
    frame = pd.read_csv(scene_manifest)
    source = scene_manifest.parent / frame.iloc[0].st_path
    with rasterio.open(source) as handle:
        values = handle.read(1)
    write_band(source, values, shift=30)
    with pytest.raises(ValueError, match="Source grids differ"):
        prepare(scene_manifest)
    assert not (scene_manifest.parent / "derived").exists()


def test_preparation_rejects_scene_or_source_leakage(scene_manifest):
    frame = pd.read_csv(scene_manifest)
    frame.loc[1, "scene_id"] = frame.loc[0, "scene_id"]
    frame.to_csv(scene_manifest, index=False)
    with pytest.raises(ValueError, match="Scene leakage"):
        load_preparation_manifest(scene_manifest)
    frame.loc[1, "scene_id"] = "val_scene"
    frame.loc[1, "red_path"] = frame.loc[0, "red_path"]
    frame.to_csv(scene_manifest, index=False)
    with pytest.raises(ValueError, match="Source leakage"):
        load_preparation_manifest(scene_manifest)


def test_preparation_rejects_relabelled_copies_of_science_sources(scene_manifest):
    frame = pd.read_csv(scene_manifest)
    train = scene_manifest.parent / frame.iloc[0].red_path
    validation = scene_manifest.parent / frame.iloc[1].red_path
    validation.write_bytes(train.read_bytes())
    with pytest.raises(ValueError, match="Identical science raster hashes"):
        prepare(scene_manifest)
    assert not (scene_manifest.parent / "derived").exists()


def test_explicit_window_manifest_and_metadata_are_preserved(scene_manifest):
    frame = pd.read_csv(scene_manifest)
    frame["top"], frame["left"], frame["height"], frame["width"] = 256, 0, 256, 256
    frame["region_name"] = "Synthetic test region"
    metadata = scene_manifest.parent / "synthetic_MTL.txt"
    metadata.write_text('DATE_ACQUIRED = 2020-06-01\nLANDSAT_PRODUCT_ID = "SYNTHETIC_TEST"\n')
    frame["metadata_path"] = metadata.name
    frame.to_csv(scene_manifest, index=False)
    report = prepare(scene_manifest)
    assert report["accepted_counts"] == dict(train=1, val=1, test=1)
    assert report["spatial_processing"]["input_window_manifest"] is True
    assert report["acquisition_date_range"] == ["2020-06-01", "2020-06-01"]
    assert report["region_names"] == ["Synthetic test region"]
    assert all(scene["date_provenance"] == "MTL" for scene in report["scene_metadata"])
    assert all(scene["metadata_sha256"] == sha256(metadata) for scene in report["scene_metadata"])


def test_missing_slc_metadata_does_not_admit_landsat7(scene_manifest):
    frame = pd.read_csv(scene_manifest)
    frame["sensor"], frame["slc_status"] = "landsat7", "SLC_ON"
    frame.to_csv(scene_manifest, index=False)
    with pytest.raises(ValueError, match="documented pre-2003"):
        prepare(scene_manifest)
    frame["acquisition_date"] = "2003-05-31"
    frame.to_csv(scene_manifest, index=False)
    with pytest.raises(ValueError, match="documented pre-2003"):
        prepare(scene_manifest)
    frame["acquisition_date"] = "2002-01-01"
    frame.to_csv(scene_manifest, index=False)
    assert prepare(scene_manifest)["accepted_counts"]["train"] == 4


def test_non_grid_windows_and_degenerate_training_ranges_are_rejected(scene_manifest):
    frame = pd.read_csv(scene_manifest)
    frame["top"], frame["left"], frame["height"], frame["width"] = 1, 0, 256, 256
    frame.to_csv(scene_manifest, index=False)
    with pytest.raises(ValueError, match="non-overlapping patch grid"):
        prepare(scene_manifest)
    frame.drop(columns=["top", "left", "height", "width"]).to_csv(scene_manifest, index=False)
    mutate_band(scene_manifest, "train", "red_path", lambda array: array.fill(10000))
    with pytest.raises(ValueError, match="nondegenerate"):
        prepare(scene_manifest)
    assert not (scene_manifest.parent / "derived").exists()


def test_quantization_rounds_to_even_and_clips_before_jpeg():
    values = np.array([-10, 0, .5, 1.5, 2.5, 254.5, 255, 300])
    np.testing.assert_array_equal(quantize_intensity(values, 0, 255), [0, 0, 0, 2, 2, 254, 255, 255])


def test_signed_st_uncertainty_excludes_negative_values(scene_manifest):
    frame = pd.read_csv(scene_manifest)
    source = scene_manifest.parent / frame.iloc[0].st_qa_path
    values = np.full((513, 513), 100, dtype=np.int16)
    values[1, 1] = -9999
    write_band(source, values, nodata=-9999)
    report = prepare(scene_manifest)
    assert report["accepted_counts"]["train"] == 3
    log = pd.read_csv(scene_manifest.parent / "derived" / "candidates.csv")
    reasons = json.loads(log.loc[~log.accepted].iloc[0].failure_pixels_by_reason)
    assert reasons["st_uncertainty"] == reasons["nodata_or_nonfinite"] == 1


def test_preparation_schema_rejects_non_256_patches(scene_manifest):
    with pytest.raises(ValueError, match="patch_size=256"):
        prepare(scene_manifest, patch_size=32)


@pytest.mark.parametrize("change,expected", [
    ("normalization", "normalization hash changed"),
    ("manifest", "does not match the completed preparation report"),
    ("schema", "Invalid prepared Landsat schema"),
])
def test_prepared_dataset_rejects_changed_binding_or_unknown_schema(scene_manifest, change, expected):
    from data.jpeg_pairs import JPEGIntensityDataset

    prepare(scene_manifest)
    output = scene_manifest.parent / "derived"
    if change == "normalization":
        statistics = output / "normalization.json"
        statistics.write_text(statistics.read_text() + "\n")
    else:
        path = output / "manifest.csv"
        frame = pd.read_csv(path)
        if change == "schema":
            frame["preparation_schema"] = "unknown_grid_schema"
        else:
            frame.loc[0, "pair_id"] = "altered_identity"
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=expected):
        JPEGIntensityDataset(output / "manifest.csv", split="train")


def test_derived_jpeg_duplicates_across_splits_leave_no_complete_report(scene_manifest):
    for column in PATH_COLUMNS[:4]:
        mutate_band(scene_manifest, "test", column, lambda array: array.__iadd__(22000))
    with pytest.raises(ValueError, match="Identical derived JPEG content crosses splits"):
        prepare(scene_manifest)
    assert not (scene_manifest.parent / "derived" / "preparation.json").exists()


def test_prepared_geotiffs_bind_training_and_inference_to_the_same_preparation(scene_manifest):
    import torch

    from experiment_utils import deep_merge, read_yaml
    from infer_rthgan import run_inference
    from train_rthgan import run_training

    preparation = prepare(scene_manifest)
    output = scene_manifest.parent / "derived"
    manifest = output / "manifest.csv"
    report_hash = sha256(output / "preparation.json")
    expected = {"schema": "landsat_jpeg_v1", "report_sha256": report_hash,
                "source_manifest_sha256": sha256(scene_manifest),
                "normalization_sha256": sha256(output / "normalization.json"),
                "bounds": preparation["normalization"], "configuration": preparation["configuration"]}
    settings = deep_merge(read_yaml("configs/model/rthgan_manuscript.yaml"),
        read_yaml("configs/train/rthgan_manuscript_smoke.yaml"),
        {"experiment_name": "synthetic_prepared_geotiffs", "device": "cpu",
         "dataset": {"input_mode": "jpeg_intensity", "name": "synthetic_prepared", "manifest": str(manifest)},
         "model": {"variant": "n", "baseline_generator": {"base_channels": 2}},
         "thermal": {**read_yaml("configs/rtd/intensity_proxy_hcw.yaml"), "control": "visual_only", "weighting": "uniform"},
         "training": {"pinn_epochs": 0, "warmup_epochs": 1, "adversarial_epochs": 1, "max_batches": 1}})
    run = scene_manifest.parent / "training"
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        run_training(settings, run)
        state = torch.load(run / "checkpoints/last.pth", weights_only=False)
        assert state["completed"] == {"pinn": 0, "warmup": 1, "adversarial": 1}
        assert state["statistics"]["manifest_sha256"] == sha256(manifest)
        assert state["statistics"]["preparation"] == expected
        assert sha256(run / "preparation_snapshot.json") == report_hash
        assert json.loads((run / "normalization.json").read_text())["preparation"] == expected
        results = scene_manifest.parent / "inference"
        metadata = run_inference(run / "checkpoints/best.pth", manifest, results,
                                 metrics=False, max_samples=1)
    finally:
        torch.set_num_threads(previous_threads)
    assert metadata["data_preparation"] == expected
    assert metadata["dataset_manifest_hash"] == sha256(manifest)
    assert metadata["physical_interpretation"] is False
    assert json.loads((results / "provenance.json").read_text())["data_preparation"] == expected
    assert len(list((results / "predictions_intensity").glob("*.png"))) == 1
    assert not (results / "predictions_kelvin").exists()
    assert all(sha256(path) == digest for path, digest in preparation["source_hashes"].items())
