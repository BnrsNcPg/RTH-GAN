"""Synthetic-only end-to-end validation; never a manuscript experiment result."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.landsat import fit_landsat_normalization
from experiment_utils import deep_merge, read_yaml, write_json
from infer_rthgan import run_inference
from train_rthgan import run_training


def run_smoke(output_dir):
    import rasterio
    root = Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite smoke directory: {root}")
    data = root / "synthetic_data"
    data.mkdir(parents=True)
    grid = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
    arrays = {"red_path": grid + 10000, "green_path": grid + 12000, "blue_path": grid + 15000,
        "st_path": grid + 40000, "qa_pixel_path": np.zeros_like(grid),
        "qa_radsat_path": np.zeros_like(grid), "st_qa_path": np.full_like(grid, 100)}
    context_path = data / "context.json"
    write_json(context_path, {"sensor": "landsat8", "radiance_units": "W m^-2 sr^-1 um^-1",
        "provenance": "ASSUMED SYNTHETIC TEST CONTEXT; not measured Landsat parameters",
        "context": {"emissivity": .95, "transmittance": .8, "downwelling_radiance": 2., "upwelling_radiance": 1.}})
    rows = []
    for split in ("train", "val", "test"):
        paths = {}
        for name, values in arrays.items():
            path = data / f"{split}_{name}.tif"
            with rasterio.open(path, "w", driver="GTiff", height=32, width=32, count=1,
                dtype="uint16", crs="EPSG:32650", transform=rasterio.Affine(30, 0, 500000, 0, -30, 4000000)) as target:
                target.write(values, 1)
            paths[name] = path.name
        rows.append({"pair_id": f"synthetic_{split}", "scene_id": f"synthetic_scene_{split}", "sensor": "landsat8",
            "split": split, "top": 0, "left": 0, "height": 32, "width": 32, "context_path": context_path.name, **paths})
    manifest, statistics_path = data / "manifest.csv", data / "normalization.json"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    statistics = fit_landsat_normalization(manifest, {"qa_pixel_reject_bits": 31, "maximum_st_uncertainty_kelvin": 2.})
    write_json(statistics_path, statistics)
    config = deep_merge(read_yaml(ROOT / "configs/model/rthgan_manuscript.yaml"),
        read_yaml(ROOT / "configs/train/rthgan_manuscript_smoke.yaml"),
        {"dataset": {"manifest": str(manifest), "normalization": str(statistics_path)},
         "thermal": {"sensor": "landsat8", "normalization": statistics["temperature"],
             "sensor_response": {"wavelength_um": [10.6, 10.9, 11.19], "response": [.2, 1., .3]}, "weighting": "hcw"}})
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        checkpoint = run_training(config, root / "run")
        run_training(config, root / "run", resume=root / "run/checkpoints/last.pth")
        # Remove the reference context to prove inference cannot depend on it.
        context_path.unlink()
        result = run_inference(checkpoint, manifest, root / "results", max_samples=1,
            evaluation_config={"evaluation": {"metrics": {"lpips": False, "fid": False, "kid": False}}})
    finally:
        torch.set_num_threads(previous_threads)
    result["synthetic_only"] = True
    write_json(root / "smoke_report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_smoke(args.output_dir)
    print(json.dumps({"synthetic_only": True, "evaluation_scope": result["evaluation_scope"],
        "images": result["num_test_images"], "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
