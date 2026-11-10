from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_utils import canonical_config_hash, file_sha256, git_commit, read_yaml, utc_timestamp, write_json
from scripts.model_profiling import build_profile_generator, contraction_counts, inference_scope, parameter_counts


def main():
    parser = argparse.ArgumentParser(description="Count actual generator parameters and estimate contraction FLOPs, including dense PINN priors.")
    parser.add_argument("--method-config", type=Path, default=Path("configs/methods/rthgan_manuscript.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model/rthgan_manuscript_profile_512.yaml"))
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", type=Path, help="Optional saved architecture/weights; required to describe a trained model")
    parser.add_argument("--output-root", type=Path, default=Path("results/complexity"))
    args = parser.parse_args()
    method = read_yaml(args.method_config)
    if args.input_size < 1:
        raise ValueError("input-size must be positive")
    model_config = read_yaml(args.model_config)
    generator = build_profile_generator(method, model_config, args.device, args.checkpoint)
    sample = torch.zeros(1, 3, args.input_size, args.input_size, device=args.device)
    payload = {
        "schema_version": 2,
        "method": method["method"],
        "display_name": method["display_name"],
        "input_shape": [1, 3, args.input_size, args.input_size],
        **parameter_counts(generator),
        **contraction_counts(generator, sample),
        **inference_scope(generator),
        "model_config_hash": canonical_config_hash(model_config) if not args.checkpoint else None,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "checkpoint_sha256": file_sha256(args.checkpoint) if args.checkpoint else None,
        "weights": "checkpoint" if args.checkpoint else "random_initialization",
        "git_commit": git_commit(),
        "timestamp": utc_timestamp(),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / f"{method['method']}.json"
    write_json(output, payload)
    print(output)


if __name__ == "__main__":
    main()
