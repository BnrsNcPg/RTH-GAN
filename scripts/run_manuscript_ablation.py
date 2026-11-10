"""Launch a resolved prospective JPEG ablation without altering historical runs."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_utils import deep_merge, read_yaml
from manuscript_protocol import experiment_identity
from train_rthgan import run_training


def build_config(dataset_config, variant, *, seed=42, smoke=False, device=None, protocol_config=None):
    registry = read_yaml(protocol_config or ROOT / "configs/experiment/manuscript_definitions_v1.yaml")
    if registry["protocol"] not in {"manuscript_definitions_v1", "manuscript_reproducibility_v2"}:
        raise ValueError("Unsupported manuscript protocol")
    if variant not in registry["variants"]:
        raise ValueError(f"Unknown manuscript variant: {variant}")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    spec = registry["variants"][variant]
    dataset = read_yaml(dataset_config)
    if dataset.get("dataset", {}).get("input_mode") != "jpeg_intensity":
        raise ValueError("This protocol requires a jpeg_intensity dataset configuration")
    config = deep_merge(dataset, read_yaml(ROOT / "configs/model/rthgan_manuscript.yaml"),
                        read_yaml(ROOT / "configs/train/manuscript_definitions_v1.yaml"))
    config = deep_merge(config, {"model": {"variant": spec["model_variant"],
        "baseline_generator": {"base_channels": 64, "guidance_scale": 0.1},
        "generator": {"fusion_variant": spec.get("fusion_variant", "full"), "attention_query_chunk_size": 64}}})
    if "rpa_variant" in spec:
        config["model"]["generator"]["rpa_variant"] = spec["rpa_variant"]
    if "stage_i_objective" in spec:
        config["training"]["stage_i_objective"] = spec["stage_i_objective"]
    if smoke:
        config = deep_merge(config, {"model": {"discriminator_channels": 4,
            "baseline_generator": {"base_channels": 4},
            "generator": {"widths": [4, 8, 16, 32], "depths": [1, 1, 1, 1],
                "feature_channels": 8, "masks": 3, "attention_heads": 2,
                "pinn": {"hidden_channels": 8, "hidden_layers": 1, "positional_frequencies": 1,
                         "num_interior_points": 16, "num_boundary_points": 8}}},
            "training": {"pinn_epochs": 1, "warmup_epochs": 1, "adversarial_epochs": 1,
                         "max_batches": 1, "log_every_batches": 1}})
    config["training"]["pinn_epochs"] = spec.get("pinn_epochs", config["training"]["pinn_epochs"])
    config["thermal"] = read_yaml(ROOT / f"configs/rtd/intensity_control_{spec['control']}.yaml")
    dataset_name = re.sub(r"[^A-Za-z0-9_-]+", "_", config["dataset"]["name"]).strip("_")
    if not dataset_name:
        raise ValueError("Dataset name must contain an ASCII letter or digit")
    prefix = "manuscript_v1" if registry["protocol"] == "manuscript_definitions_v1" else registry["protocol"]
    config["experiment_name"] = f"{prefix}_{dataset_name}_{variant}_seed{seed}" + ("_smoke" if smoke else "")
    config["seed"] = seed
    config["protocol"] = registry["protocol"]
    if device is not None:
        config["device"] = device
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--protocol-config", type=Path, help="Optional prospective registry; default preserves v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Reduced model, one batch per stage; never formal results")
    parser.add_argument("--device")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Print resolved settings without reading images or creating a run")
    args = parser.parse_args()
    config = build_config(args.dataset_config, args.variant, seed=args.seed, smoke=args.smoke,
                          device=args.device, protocol_config=args.protocol_config)
    print(json.dumps({"identity": experiment_identity(config), "config": config}, indent=2), flush=True)
    if not args.dry_run:
        checkpoint = run_training(config, args.output_dir, resume=args.resume)
        print(json.dumps({"checkpoint": str(checkpoint)}), flush=True)


if __name__ == "__main__":
    main()
