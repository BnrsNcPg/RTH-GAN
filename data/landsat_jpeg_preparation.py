"""Prepare derived JPEG intensity pairs from read-only Landsat C2 L2 grids.

This module implements a new preparation protocol. Its outputs do not establish
the provenance of any pre-existing JPEG dataset or published result.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
from PIL import Image, features

from data.landsat import PATH_COLUMNS, resolve_source


SCHEMA = "landsat_jpeg_v1"
DEFAULT_CONFIG = {
    "patch_size": 256,
    "qa_pixel_reject_bits": {"landsat5": 27, "landsat7": 27, "landsat8": 31},
    "maximum_st_uncertainty_kelvin": 2.0,
    "sr_dn_valid_range": [1, 65455],
    "st_dn_valid_range": [1, 65535],
    "jpeg_quality": 100,
    "rgb_subsampling": 0,
}
SPLITS = ("train", "val", "test")
WINDOW_COLUMNS = ("top", "left", "height", "width")
METADATA_COLUMNS = ("acquisition_date", "region_name", "metadata_path", "product_id", "slc_status")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preparation_config(config=None):
    result = copy.deepcopy(DEFAULT_CONFIG)
    if config is not None:
        unknown = set(config) - set(result)
        if unknown:
            raise ValueError(f"Unknown preparation configuration: {sorted(unknown)}")
        result.update(copy.deepcopy(config))
    size = result["patch_size"]
    if type(size) is not int or size != 256:
        raise ValueError("landsat_jpeg_v1 requires patch_size=256")
    masks = result["qa_pixel_reject_bits"]
    if not isinstance(masks, dict) or masks != DEFAULT_CONFIG["qa_pixel_reject_bits"]:
        raise ValueError("This protocol requires QA_PIXEL masks 27/27/31 for Landsat-5/7/8")
    uncertainty = result["maximum_st_uncertainty_kelvin"]
    if type(uncertainty) not in (int, float) or not math.isfinite(uncertainty) or uncertainty <= 0:
        raise ValueError("maximum_st_uncertainty_kelvin must be finite and positive")
    for name in ("sr_dn_valid_range", "st_dn_valid_range"):
        bounds = result[name]
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                or any(type(value) is not int for value in bounds)
                or not 1 <= bounds[0] < bounds[1] <= 65535):
            raise ValueError(f"Invalid declared DN range: {name}")
    if result["jpeg_quality"] != 100 or result["rgb_subsampling"] != 0:
        raise ValueError("This protocol requires JPEG quality=100 and RGB subsampling=0")
    return result


def _metadata_values(path):
    """Read supplied MTL text/JSON; never infer dates from sensor lifetime."""
    text = Path(path).read_text(encoding="utf-8-sig")
    if Path(path).suffix.lower() == ".json":
        values = {}

        def visit(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in {"DATE_ACQUIRED", "LANDSAT_PRODUCT_ID"}:
                        values[key] = str(item)
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(json.loads(text))
        return values
    return {match.group(1): match.group(2).strip().strip('"')
            for match in re.finditer(r"^\s*(DATE_ACQUIRED|LANDSAT_PRODUCT_ID)\s*=\s*(.*?)\s*$", text, re.M)}


def load_preparation_manifest(manifest, config=None):
    """Accept one row per scene, or explicit non-overlapping grid windows."""
    config = preparation_config(config)
    manifest = Path(manifest).resolve()
    frame = pd.read_csv(manifest, keep_default_na=False, dtype=str)
    required = {"scene_id", "sensor", "split", *PATH_COLUMNS}
    missing = required - set(frame)
    if missing or frame.empty:
        raise ValueError(f"Require a nonempty scene manifest with columns: {sorted(missing)}")
    if frame[list(required)].eq("").any().any():
        raise ValueError("Scene, split, sensor, and all science/QA paths must be nonempty")
    if frame.sensor.nunique() != 1 or frame.sensor.iloc[0] not in {"landsat5", "landsat7", "landsat8"}:
        raise ValueError("Use one supported sensor per preparation dataset")
    if set(frame.split) != set(SPLITS):
        raise ValueError("Require explicit train, val, and test scene partitions")
    if frame.groupby("scene_id").split.nunique().gt(1).any():
        raise ValueError("Scene leakage across splits")
    window_mode = all(column in frame for column in WINDOW_COLUMNS)
    if any(column in frame for column in WINDOW_COLUMNS) and not window_mode:
        raise ValueError("Supply all four window columns or none")
    if not window_mode and frame.scene_id.duplicated().any():
        raise ValueError("Scene manifests require exactly one row per scene")
    for column in PATH_COLUMNS:
        frame[column] = frame[column].map(lambda value: str(resolve_source(value, manifest).resolve()))
        if frame.groupby(column).split.nunique().gt(1).any():
            raise ValueError(f"Source leakage across splits: {column}")
        if frame.groupby("scene_id")[column].nunique().gt(1).any():
            raise ValueError("Each scene must identify one consistent set of source rasters")
    # Also reject one raster being relabeled as different bands across splits.
    owners = {}
    for _, row in frame.iterrows():
        for column in PATH_COLUMNS:
            previous = owners.setdefault(row[column], row.split)
            if previous != row.split:
                raise ValueError("Source leakage across band labels and splits")
    for column in METADATA_COLUMNS:
        if column not in frame:
            frame[column] = ""
        if frame.groupby("scene_id")[column].nunique().gt(1).any():
            raise ValueError(f"Inconsistent scene metadata: {column}")
    size = config["patch_size"]
    if window_mode:
        for column in WINDOW_COLUMNS:
            numbers = pd.to_numeric(frame[column], errors="raise")
            if not np.isfinite(numbers).all() or (numbers != np.floor(numbers)).any():
                raise ValueError("Window coordinates must be finite integers")
            frame[column] = numbers.astype(int)
        if (not frame.height.eq(size).all() or not frame.width.eq(size).all()
                or frame.top.lt(0).any() or frame.left.lt(0).any()
                or (frame.top % size).ne(0).any() or (frame.left % size).ne(0).any()):
            raise ValueError("Windows must follow the declared non-overlapping patch grid")
        if frame.duplicated(["scene_id", "top", "left"]).any():
            raise ValueError("Duplicate scene windows")
    return frame, window_mode


def _inspect_scene(row, manifest):
    import rasterio
    from rasterio.warp import transform_bounds

    grid = None
    for column in PATH_COLUMNS:
        with rasterio.open(row[column]) as source:
            current = (source.crs, source.transform, source.height, source.width)
            allowed = {"uint16", "int16"} if column == "st_qa_path" else {"uint16"}
            if source.count != 1 or source.crs is None or source.dtypes[0] not in allowed:
                raise ValueError(f"Expected one georeferenced science/QA band with a supported DN dtype: {row[column]}")
            if grid is not None and current != grid:
                raise ValueError(f"Source grids differ; no resampling or registration: {row.scene_id}")
            grid = current
            geographic_bounds = list(transform_bounds(source.crs, "EPSG:4326", *source.bounds))
    acquired, product_id = row.acquisition_date, row.product_id
    mtl = {}
    metadata_path, metadata_hash = None, None
    if row.metadata_path:
        metadata_path = resolve_source(row.metadata_path, manifest).resolve()
        metadata_hash = sha256(metadata_path)
        mtl = _metadata_values(metadata_path)
        if acquired and mtl.get("DATE_ACQUIRED") and acquired != mtl["DATE_ACQUIRED"]:
            raise ValueError(f"Acquisition date conflicts with MTL: {row.scene_id}")
        if product_id and mtl.get("LANDSAT_PRODUCT_ID") and product_id != mtl["LANDSAT_PRODUCT_ID"]:
            raise ValueError(f"Product ID conflicts with MTL: {row.scene_id}")
        acquired = acquired or mtl.get("DATE_ACQUIRED", "")
        product_id = product_id or mtl.get("LANDSAT_PRODUCT_ID", "")
    if acquired:
        try:
            if date.fromisoformat(acquired).isoformat() != acquired:
                raise ValueError("Require YYYY-MM-DD")
        except ValueError as error:
            raise ValueError(f"Invalid acquisition_date for {row.scene_id}") from error
    if row.sensor == "landsat7":
        if row.slc_status != "SLC_ON" or not acquired or date.fromisoformat(acquired) >= date(2003, 5, 31):
            raise ValueError("Landsat-7 requires SLC_ON and a documented pre-2003-05-31 acquisition date")
    if not all(math.isfinite(value) for value in geographic_bounds):
        raise ValueError("Cannot derive finite geographic bounds from source grid")
    info = {"scene_id": row.scene_id, "sensor": row.sensor, "split": row.split,
            "acquisition_date": acquired or None, "region_name": row.region_name or None,
            "product_id": product_id or None,
            "metadata_path": str(metadata_path) if metadata_path else None,
            "metadata_sha256": metadata_hash,
            "date_provenance": "MTL" if mtl.get("DATE_ACQUIRED") else ("manifest" if acquired else None),
            "slc_status": row.slc_status or None,
            "geographic_bounds_wgs84": geographic_bounds,
            "geographic_bounds_basis": "source raster footprint transformed to EPSG:4326",
            "grid": {"crs": str(grid[0]), "transform": list(grid[1]), "height": grid[2], "width": grid[3]}}
    return info


def _read_candidate(row, config):
    import rasterio
    from rasterio.windows import Window

    arrays, invalid = {}, np.zeros((row.height, row.width), dtype=bool)
    for column in PATH_COLUMNS:
        with rasterio.open(row[column]) as source:
            values = source.read(1, window=Window(row.left, row.top, row.width, row.height))
            mask = source.read_masks(1, window=Window(row.left, row.top, row.width, row.height))
        if values.shape != invalid.shape:
            raise ValueError(f"Window exceeds source grid: {row.pair_id}")
        invalid |= (mask == 0) | ~np.isfinite(values)
        arrays[column] = values
    rgb = np.stack([arrays[column] for column in PATH_COLUMNS[:3]])
    thermal = arrays["st_path"]
    reasons = {"nodata_or_nonfinite": invalid,
               "qa_pixel": (arrays["qa_pixel_path"] & config["qa_pixel_reject_bits"][row.sensor]) != 0,
               "qa_radsat": arrays["qa_radsat_path"] != 0,
               "st_uncertainty": ((arrays["st_qa_path"] < 0) | (arrays["st_qa_path"].astype(np.float64) * .01 > config["maximum_st_uncertainty_kelvin"])),
               "sr_dn_range": ((rgb < config["sr_dn_valid_range"][0]) | (rgb > config["sr_dn_valid_range"][1])).any(0),
               "st_dn_range": (thermal < config["st_dn_valid_range"][0]) | (thermal > config["st_dn_valid_range"][1])}
    failed = {name: int(values.sum()) for name, values in reasons.items() if values.any()}
    return (rgb.astype(np.float64) * .0000275 - .2,
            thermal.astype(np.float64) * .00341802 + 149., failed)


def quantize_intensity(values, minimum, maximum):
    """Round to even before JPEG encoding; JPEG decoding is not lossless."""
    scaled = (values - minimum) / (maximum - minimum)
    return np.rint(255 * np.clip(scaled, 0, 1)).astype(np.uint8)


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def prepare_landsat_jpeg(manifest, output_dir, config=None, *, configuration_path=None):
    """Write a new derived dataset; refuse overwrites and preserve source hashes.

    Structural errors abort. QA-invalid candidates are excluded and recorded.
    A failure after writing begins leaves an incomplete directory for review;
    it is never marked complete and a retry needs a new output directory.
    """
    config = preparation_config(config)
    manifest, output = Path(manifest).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output}")
    frame, window_mode = load_preparation_manifest(manifest, config)
    sources = sorted({Path(value) for column in PATH_COLUMNS for value in frame[column]})
    metadata_paths = {resolve_source(value, manifest).resolve() for value in frame.metadata_path if value}
    module_path = Path(__file__).resolve()
    cli_path = module_path.parents[1] / "scripts" / "prepare_landsat_jpeg.py"
    protected = set(sources) | metadata_paths | {manifest, module_path, cli_path}
    configuration_path = Path(configuration_path).resolve() if configuration_path else None
    if configuration_path:
        protected.add(configuration_path)
    if any(output == path.parent or output.is_relative_to(path.parent) for path in sources):
        raise ValueError("Derived output must be outside source raster directories")
    source_hashes = {str(path): sha256(path) for path in sorted(protected)}
    # Exact source copies under a different path must not bypass split checks.
    hashes_by_split = {}
    for _, row in frame.iterrows():
        for column in PATH_COLUMNS[:4]:
            digest = source_hashes[row[column]]
            previous = hashes_by_split.setdefault(digest, row.split)
            if previous != row.split:
                raise ValueError("Identical science raster hashes cross scene splits")
    scene_records, windows = [], []
    size = config["patch_size"]
    for index, (scene_id, group) in enumerate(frame.groupby("scene_id", sort=False)):
        row = group.iloc[0]
        info = _inspect_scene(row, manifest)
        scene_records.append(info)
        if window_mode:
            candidates = group.to_dict("records")
        else:
            candidates = [{**row.to_dict(), "top": top, "left": left, "height": size, "width": size}
                          for top in range(0, info["grid"]["height"] - size + 1, size)
                          for left in range(0, info["grid"]["width"] - size + 1, size)]
        for candidate in candidates:
            if candidate["top"] + size > info["grid"]["height"] or candidate["left"] + size > info["grid"]["width"]:
                raise ValueError(f"Window exceeds source grid: {scene_id}")
            candidate["pair_id"] = f"scene{index:06d}_r{candidate['top']:06d}_c{candidate['left']:06d}"
            windows.append(candidate)
    if not windows:
        raise ValueError("No complete source-grid windows")
    candidates = pd.DataFrame(windows)
    minima, maxima = np.full(4, np.inf), np.full(4, -np.inf)
    accepted, candidate_log = [], []
    for _, row in candidates.iterrows():
        rgb, thermal, failures = _read_candidate(row, config)
        candidate_log.append({"pair_id": row.pair_id, "scene_id": row.scene_id, "split": row.split,
                              **{name: int(row[name]) for name in WINDOW_COLUMNS},
                              "accepted": not failures, "failure_pixels_by_reason": json.dumps(failures, sort_keys=True)})
        if failures:
            continue
        accepted.append(row.to_dict())
        if row.split == "train":
            values = np.concatenate([rgb.reshape(3, -1), thermal.reshape(1, -1)], axis=0)
            minima = np.minimum(minima, values.min(1))
            maxima = np.maximum(maxima, values.max(1))
    admitted = pd.DataFrame(accepted)
    if admitted.empty or set(admitted.split) != set(SPLITS):
        raise ValueError("Every split needs an accepted patch after QA screening")
    if not np.isfinite(minima).all() or not np.isfinite(maxima).all() or (maxima <= minima).any():
        raise ValueError("Accepted training patches require four nondegenerate value ranges")
    # Check the source snapshot before any output is created.
    if any(sha256(path) != digest for path, digest in source_hashes.items()):
        raise ValueError("A source changed during preparation")
    output.mkdir(parents=True, exist_ok=False)
    (output / "visible").mkdir()
    (output / "infrared").mkdir()
    normalization = {"schema_version": 1, "preparation_schema": SCHEMA, "source_split": "train",
                     "sensor": str(frame.sensor.iloc[0]), "manifest_sha256": source_hashes[str(manifest)],
                     "visible_minimum": minima[:3].tolist(), "visible_maximum": maxima[:3].tolist(),
                     "temperature_minimum_kelvin": float(minima[3]), "temperature_maximum_kelvin": float(maxima[3]),
                     "encoding": "round_to_even(255*clip((value-minimum)/(maximum-minimum),0,1))"}
    _write_json(output / "normalization.json", normalization)
    normalization_hash = sha256(output / "normalization.json")
    pd.DataFrame(candidate_log).to_csv(output / "candidates.csv", index=False, lineterminator="\n")
    clipping = {split: {"below": [0] * 4, "above": [0] * 4, "pixels_per_channel": 0} for split in SPLITS}
    output_rows = []
    for _, row in admitted.iterrows():
        rgb, thermal, failures = _read_candidate(row, config)
        if failures:
            raise ValueError("Source QA changed after the training-statistics pass")
        values = np.concatenate([rgb.reshape(3, -1), thermal.reshape(1, -1)], axis=0)
        count = clipping[row.split]
        count["below"] = (np.array(count["below"]) + (values < minima[:, None]).sum(1)).tolist()
        count["above"] = (np.array(count["above"]) + (values > maxima[:, None]).sum(1)).tolist()
        count["pixels_per_channel"] += values.shape[1]
        visible = quantize_intensity(rgb, minima[:3, None, None], maxima[:3, None, None]).transpose(1, 2, 0)
        infrared = quantize_intensity(thermal, minima[3], maxima[3])
        visible_path = Path("visible") / f"{row.pair_id}.jpg"
        infrared_path = Path("infrared") / f"{row.pair_id}.jpg"
        Image.fromarray(visible).save(output / visible_path, format="JPEG", quality=100, subsampling=0)
        Image.fromarray(infrared).save(output / infrared_path, format="JPEG", quality=100)
        output_rows.append({"preparation_schema": SCHEMA, "dataset": f"{row.sensor}_jpeg_prepared",
                            "pair_id": row.pair_id, "scene_id": row.scene_id, "sequence_id": row.scene_id,
                            "sensor": row.sensor, "split": row.split,
                            "visible_path": visible_path.as_posix(), "infrared_path": infrared_path.as_posix(),
                            "visible_sha256": sha256(output / visible_path), "infrared_sha256": sha256(output / infrared_path),
                            "registration_basis": "shared_source_product_grid",
                            "source_manifest_sha256": source_hashes[str(manifest)],
                            "normalization_sha256": normalization_hash,
                            "preparation_report": "preparation.json",
                            **{name: int(row[name]) for name in WINDOW_COLUMNS},
                            **{f"source_{column}": row[column] for column in PATH_COLUMNS},
                            **{f"source_{column.removesuffix('_path')}_sha256": source_hashes[row[column]] for column in PATH_COLUMNS}})
    for split in SPLITS:
        count = clipping[split]
        total = count["pixels_per_channel"]
        count["below_fraction"] = (np.array(count["below"]) / total).tolist()
        count["above_fraction"] = (np.array(count["above"]) / total).tolist()
    derived_manifest = output / "manifest.csv"
    output_frame = pd.DataFrame(output_rows)
    for column in ("visible_sha256", "infrared_sha256"):
        if output_frame.groupby(column).split.nunique().gt(1).any():
            raise ValueError("Identical derived JPEG content crosses splits; derived output is incomplete")
    output_frame.to_csv(derived_manifest, index=False, lineterminator="\n")
    if any(sha256(path) != digest for path, digest in source_hashes.items()):
        raise ValueError("A source changed during preparation; derived output is incomplete")
    dates = [scene["acquisition_date"] for scene in scene_records if scene["acquisition_date"]]
    bounds = np.array([scene["geographic_bounds_wgs84"] for scene in scene_records])
    report = {"schema_version": 1, "preparation_schema": SCHEMA, "status": "complete",
              "preparation_code": "data/landsat_jpeg_preparation.py", "preparation_code_sha256": source_hashes[str(module_path)],
              "cli_script": "scripts/prepare_landsat_jpeg.py", "cli_script_sha256": source_hashes[str(cli_path)],
              "configuration_path": str(configuration_path) if configuration_path else None,
              "configuration_sha256": source_hashes[str(configuration_path)] if configuration_path else None,
              "data_domain": "jpeg_intensity", "physical_interpretation": False,
              "source_manifest": str(manifest), "source_manifest_sha256": source_hashes[str(manifest)],
              "manifest_sha256": sha256(derived_manifest), "normalization_sha256": sha256(output / "normalization.json"),
              "candidates_sha256": sha256(output / "candidates.csv"), "source_hashes": source_hashes,
              "configuration": config, "normalization": normalization,
              "source_value_decoding": {"surface_reflectance": {"scale": .0000275, "offset": -.2},
                                        "surface_temperature_kelvin": {"scale": .00341802, "offset": 149.},
                                        "st_uncertainty_kelvin": {"scale": .01, "offset": 0}},
              "encoding": {"pillow_version": PIL.__version__, "jpeg_codec_version": features.version_codec("jpg"),
                           "rgb_mode": "RGB", "infrared_mode": "L", "quality": 100, "rgb_subsampling": 0,
                           "lossy": True, "rounding_before_jpeg": "round_to_even"},
              "candidate_counts": candidates.split.value_counts().to_dict(),
              "accepted_counts": admitted.split.value_counts().to_dict(),
              "rejected_counts": {split: int((candidates.split == split).sum() - (admitted.split == split).sum()) for split in SPLITS},
              "scene_counts": {split: sum(scene["split"] == split for scene in scene_records) for split in SPLITS},
              "accepted_scene_counts": {split: int(admitted.loc[admitted.split == split].scene_id.nunique()) for split in SPLITS},
              "clipping": {"channel_order": ["red", "green", "blue", "temperature"], "splits": clipping},
              "scene_metadata": scene_records,
              "region_names": sorted({scene["region_name"] for scene in scene_records if scene["region_name"]}),
              "acquisition_date_range": [min(dates), max(dates)] if dates else None,
              "acquisition_date_range_scope": "scenes with supplied acquisition metadata only",
              "scenes_missing_acquisition_date": [scene["scene_id"] for scene in scene_records if not scene["acquisition_date"]],
              "scenes_missing_region_name": [scene["scene_id"] for scene in scene_records if not scene["region_name"]],
              "geographic_bounds_wgs84": [float(bounds[:, 0].min()), float(bounds[:, 1].min()), float(bounds[:, 2].max()), float(bounds[:, 3].max())],
              "split_policy": "supplied scene-exclusive assignments before grid-window enumeration",
              "spatial_processing": {"patch_size": size, "stride": size, "resize": False, "registration": False,
                                     "incomplete_edge_windows": "discarded", "input_window_manifest": window_mode},
              "raw_files_modified": False}
    _write_json(output / "preparation.json", report)
    return report
