"""Context-free generator inference and unified evaluation on paired Landsat data."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from data.landsat import LandsatPairedDataset
from data.jpeg_pairs import JPEGIntensityDataset
from experiment_utils import file_sha256, git_commit, read_yaml, write_json
from metrics.evaluator import UnifiedEvaluator
from models.manuscript_ablation import build_manuscript_generator
from manuscript_protocol import experiment_identity, checkpoint_metadata


def run_inference(checkpoint, manifest, output_dir, *, split="test", device="cpu",
                  evaluation_config=None, range_policy="error", max_samples=None, metrics=True):
    if range_policy not in {"error", "clip"}:
        raise ValueError("range_policy must be error or clip")
    if max_samples is not None and (type(max_samples) is not int or max_samples < 1):
        raise ValueError("max_samples must be positive or null")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if state.get("format") != "rthgan_manuscript_v1":
        raise ValueError("Expected a manuscript RTH-GAN checkpoint")
    generator = build_manuscript_generator(state["config"]["model"]).to(device).eval()
    generator.load_state_dict(state["trainer"]["generator"], strict=True)
    intensity = state["statistics"].get("data_domain") == "jpeg_intensity"
    if intensity:
        if file_sha256(manifest) != state["statistics"]["manifest_sha256"]:
            raise ValueError("JPEG evaluation manifest must match the training snapshot")
        dataset = JPEGIntensityDataset(manifest, split=split)
    else:
        dataset = LandsatPairedDataset(manifest, state["statistics"], split=split, require_context=False)
    dataset_name = state["statistics"]["dataset" if intensity else "sensor"]
    identity = experiment_identity(state["config"])
    method = identity["method"]
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) for value in dataset.frame.pair_id):
        raise ValueError("Pair IDs must be safe file basenames")
    selected = Subset(dataset, range(min(max_samples, len(dataset)))) if max_samples else dataset
    evaluator = UnifiedEvaluator(evaluation_config, device=device, num_samples=len(selected)) if metrics else None
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite result directory: {output}")
    for folder in ("predictions_normalized", "predictions_intensity" if intensity else "predictions_kelvin", "ground_truth_normalized"):
        (output / folder).mkdir(parents=True, exist_ok=False)
    clipped_pixels, total_pixels = 0, 0
    with torch.no_grad():
        for batch in DataLoader(selected, batch_size=1, shuffle=False, num_workers=0):
            prediction = generator(batch["visible"].to(device)).cpu()
            target = batch["infrared"]
            pair_id = batch["pair_id"][0]
            np.save(output / "predictions_normalized" / (pair_id + ".npy"), prediction[0, 0].numpy(), allow_pickle=False)
            if intensity:
                pixels = ((prediction[0, 0] + 1) * 127.5).clamp(0, 255).round().numpy().astype(np.uint8)
                Image.fromarray(pixels).save(output / "predictions_intensity" / (pair_id + ".png"))
            else:
                np.save(output / "predictions_kelvin" / (pair_id + ".npy"), dataset.normalization(prediction)[0, 0].numpy(), allow_pickle=False)
            np.save(output / "ground_truth_normalized" / (pair_id + ".npy"), target[0, 0].numpy(), allow_pickle=False)
            generated_unit, target_unit = (prediction + 1) / 2, (target + 1) / 2
            clipped_pixels += int(((target_unit < 0) | (target_unit > 1)).sum())
            total_pixels += target_unit.numel()
            if evaluator:
                if range_policy == "clip":
                    generated_unit, target_unit = generated_unit.clamp(0, 1), target_unit.clamp(0, 1)
                evaluator.add(pair_id, generated_unit, target_unit, dataset=dataset_name, subset=split, method=method)
    metadata = {**identity,
        "experiment_name": state["config"]["experiment_name"], "dataset": dataset_name, "split": split,
        "data_domain": "jpeg_intensity" if intensity else "quantitative_landsat", "physical_interpretation": not intensity,
        "seed": state["config"]["seed"], "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(checkpoint), "dataset_manifest_hash": file_sha256(manifest),
        "git_commit": git_commit(), "config_hash": state["config_hash"],
        "training_git_commit": state.get("training_git_commit"), "training_completed_epochs": state["completed"],
        **checkpoint_metadata(state),
        "lambda_l1": state["config"]["training"]["lambda_l1"],
        "lambda_hac": state["config"]["training"]["lambda_hac"],
        "auxiliary_definition": state.get("auxiliary_definition"),
        "training_objectives": state.get("training_objectives"),
        "data_preparation": state["statistics"].get("preparation"),
        "evaluation_scope": "smoke" if max_samples or state["config"]["training"].get("max_batches") else "formal",
        "range_policy": range_policy, "out_of_range_reference_pixels": clipped_pixels,
        "reference_pixel_count": total_pixels, "reference_context_used_by_generator": False}
    write_json(output / "provenance.json", metadata)
    return evaluator.write(output, metadata) if evaluator else metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--evaluation-config", default="configs/evaluation/default.yaml")
    parser.add_argument("--range-policy", default="error", choices=("error", "clip"))
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--no-metrics", action="store_true")
    args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    run_inference(args.checkpoint, args.manifest, args.output_dir, split=args.split, device=device,
        evaluation_config=read_yaml(args.evaluation_config), range_policy=args.range_policy,
        max_samples=args.max_samples, metrics=not args.no_metrics)


if __name__ == "__main__":
    main()
