"""Train RTH-GAN on quantitative Landsat or explicitly labeled JPEG intensities."""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.landsat import LandsatPairedDataset
from data.jpeg_pairs import JPEGIntensityDataset
from experiment_utils import (canonical_config_hash, deep_merge, file_sha256, file_sha256_lf_normalized, git_commit,
    git_status_snapshot, read_yaml, seed_everything, utc_timestamp, validate_experiment_name,
    write_environment, write_json)
from manuscript_training import ManuscriptTrainer
from manuscript_protocol import experiment_identity, uses_pinn


def capture_rng(loader_generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": loader_generator.get_state()}


def restore_rng(state, loader_generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    loader_generator.set_state(state["loader"].cpu())


def run_training(config, output_dir, *, resume=None):
    settings = config["training"]
    phases = ("pinn", "warmup", "adversarial")
    for phase in phases:
        count = settings.get(phase + "_epochs")
        if phase == "pinn" and not uses_pinn(config):
            if type(count) is not int or count != 0:
                raise ValueError("The N baseline requires pinn_epochs=0")
        elif type(count) is not int or count < 1:
            raise ValueError(f"{phase}_epochs must be a positive integer")
    if type(settings.get("batch_size")) is not int or settings["batch_size"] < 1:
        raise ValueError("batch_size must be a positive integer")
    for key in ("learning_rate_pinn", "learning_rate_g", "learning_rate_d", "lambda_l1", "lambda_hac"):
        value = settings.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (key != "lambda_l1" and value == 0):
            raise ValueError(f"Specify a finite {'nonnegative' if key == 'lambda_l1' else 'positive'} {key}")
    if settings.get("num_workers", 0) != 0:
        raise ValueError("Manuscript checkpoint RNG recovery currently requires num_workers=0")
    log_every = settings.get("log_every_batches", 0)
    if type(log_every) is not int or log_every < 0:
        raise ValueError("log_every_batches must be a nonnegative integer")
    cpu_threads = settings.get("cpu_threads")
    if cpu_threads is not None:
        if type(cpu_threads) is not int or cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")
        torch.set_num_threads(cpu_threads)
    max_batches = settings.get("max_batches")
    if max_batches is not None and (type(max_batches) is not int or max_batches < 1):
        raise ValueError("max_batches must be null or a positive smoke-test limit")
    validate_experiment_name(config["experiment_name"])
    seed_everything(int(config["seed"]))
    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = Path(config["dataset"]["manifest"])
    input_mode = config["dataset"].get("input_mode", "quantitative_landsat")
    preparation_report = None
    if input_mode == "jpeg_intensity":
        if config["thermal"].get("mode") != "intensity_proxy":
            raise ValueError("JPEG training requires an explicit intensity_proxy auxiliary")
        statistics = {"schema_version": 1, "data_domain": "jpeg_intensity", "physical_interpretation": False,
            "dataset": config["dataset"]["name"], "manifest_sha256": file_sha256(manifest),
            "visible_mapping": "uint8 / 255", "infrared_mapping": "uint8 / 127.5 - 1"}
        dataset = JPEGIntensityDataset(manifest, split="train", training=True)
        validation = JPEGIntensityDataset(manifest, split="val")
        if "preparation_schema" in dataset.frame:
            reference = Path(dataset.frame.preparation_report.iloc[0])
            preparation_report = reference if reference.is_absolute() else manifest.resolve().parent / reference
            preparation = json.loads(preparation_report.read_text(encoding="utf-8"))
            statistics["preparation"] = {"schema": preparation["preparation_schema"],
                "report_sha256": file_sha256(preparation_report),
                "source_manifest_sha256": preparation["source_manifest_sha256"],
                "normalization_sha256": preparation["normalization_sha256"],
                "bounds": preparation["normalization"], "configuration": preparation["configuration"]}
    elif input_mode == "quantitative_landsat":
        if config["thermal"].get("mode", "thermal") != "thermal":
            raise ValueError("Quantitative Landsat training requires a physical thermal auxiliary")
        statistics = json.loads(Path(config["dataset"]["normalization"]).read_text(encoding="utf-8"))
        if config["thermal"].get("sensor") != statistics["sensor"]:
            raise ValueError("Thermal sensor response must identify the dataset's sensor")
        config = deep_merge(config, {"thermal": {"normalization": statistics["temperature"]}})
        dataset = LandsatPairedDataset(manifest, statistics, split="train", training=True)
        validation = LandsatPairedDataset(manifest, statistics, split="val", require_context=False)
        if not dataset.frame.height.eq(dataset.frame.height.iloc[0]).all() or int(dataset.frame.height.iloc[0]) % 32:
            raise ValueError("Training patches must share one size divisible by 32")
    else:
        raise ValueError("dataset.input_mode must be quantitative_landsat or jpeg_intensity")
    loader_generator = torch.Generator().manual_seed(int(config["seed"]))
    loader = DataLoader(dataset, batch_size=int(settings["batch_size"]), shuffle=True, num_workers=0, generator=loader_generator)
    validation_loader = DataLoader(validation, batch_size=1, shuffle=False, num_workers=0)
    trainer = ManuscriptTrainer(config, device=device)
    identity = experiment_identity(config)
    auxiliary_definition = (trainer.auxiliary.control_definition()
                            if hasattr(trainer.auxiliary, "control_definition") else None)
    if auxiliary_definition is not None:
        auxiliary_definition["lambda_hac"] = trainer.lambda_hac
        for kind in ("l1", "weighted_l1"):
            coefficient = auxiliary_definition[f"additional_{kind}_coefficient_before_lambda_hac"]
            auxiliary_definition[f"effective_additional_{kind}_coefficient"] = (
                trainer.lambda_hac * coefficient if coefficient is not None else None)
    source_paths = ("train_rthgan.py", "manuscript_training.py", "data/landsat.py", "rtd/thermal.py", "rtd/hcw.py",
        "models/thermal_rtd.py", "models/rghsg.py", "models/discriminator.py", "models/modules/manuscript_retinex.py",
        "models/modules/manuscript_fusion.py", "models/modules/pinn_retinex.py", "models/modules/euclidean_hyperbolic_fusion_v2.py",
        "models/manuscript_ablation.py", "models/generator.py", "manuscript_protocol.py")
    source_hashes = {name: file_sha256_lf_normalized(Path(__file__).resolve().parent / name) for name in source_paths}
    if input_mode == "jpeg_intensity":
        for name in ("data/jpeg_pairs.py", "data/manifest.py", "models/intensity_rtd.py",
                     "rtd/context.py", "rtd/normalized_proxy.py"):
            source_hashes[name] = file_sha256_lf_normalized(Path(__file__).resolve().parent / name)
    fingerprint = canonical_config_hash({"config": config, "statistics": statistics, "source_hashes": source_hashes})
    completed = {phase: 0 for phase in phases}
    best = float("inf")
    selection = None
    output = Path(output_dir)
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        if state.get("format") != "rthgan_manuscript_v1" or state["config_hash"] != fingerprint:
            raise ValueError("Resume checkpoint format/config/data statistics/source code do not match")
        if not output.is_dir() or not (output / "provenance.json").is_file():
            raise ValueError("Resume requires the original manuscript run directory")
        trainer.load_state_dict(state["trainer"])
        completed, best = state["completed"], state["best_validation_l1"]
        selection = state.get("selection")
        restore_rng(state["rng"], loader_generator)
    else:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite run directory: {output}")
        output.mkdir(parents=True)
        (output / "checkpoints").mkdir()
        (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        shutil.copy2(manifest, output / "manifest_snapshot.csv")
        if preparation_report is not None:
            shutil.copy2(preparation_report, output / "preparation_snapshot.json")
        write_json(output / "normalization.json", statistics)
        write_environment(output / "environment.txt", device)
        write_json(output / "provenance.json", {**identity, "git_commit": git_commit(),
            "git_status": git_status_snapshot(), "manifest_sha256": file_sha256(manifest),
            "config_hash": fingerprint, "source_hashes": source_hashes, "seed": config["seed"], "started_at": utc_timestamp(),
            "evaluation_scope": "smoke" if max_batches else "formal",
            "data_domain": input_mode, "physical_interpretation": input_mode == "quantitative_landsat",
            "auxiliary_definition": auxiliary_definition,
            "training_objectives": trainer.training_objectives,
            "lambda_l1": trainer.lambda_l1, "lambda_hac": trainer.lambda_hac,
            "implementation_note": "Configured architecture/optimizer choices; not reproduced manuscript metrics"})
    training_commit = json.loads((output / "provenance.json").read_text(encoding="utf-8"))["git_commit"]
    started = time.monotonic()
    write_json(output / "status.json", {"status": "running", "completed": completed, "updated_at": utc_timestamp()})
    try:
        for phase in phases:
            for epoch in range(completed[phase], settings[phase + "_epochs"]):
                totals, batches = {}, 0
                for batch in loader:
                    values = trainer.pretrain_step(batch) if phase == "pinn" else trainer.train_step(batch, adversarial=phase == "adversarial")
                    for key, value in values.items():
                        totals[key] = totals.get(key, 0.) + value
                    batches += 1
                    if log_every and (batches == 1 or batches % log_every == 0):
                        progress = {"event": "batch", "phase": phase, "epoch": epoch + 1,
                            "batch": batches, "total_batches": min(len(loader), max_batches or len(loader)),
                            "elapsed_seconds": time.monotonic() - started, **values}
                        print(json.dumps(progress, allow_nan=False), flush=True)
                        write_json(output / "status.json", {"status": "running", "completed": completed,
                            "progress": progress, "updated_at": utc_timestamp()})
                    if max_batches and batches >= max_batches:
                        break
                completed[phase] = epoch + 1
                row = {"phase": phase, "epoch": epoch + 1, **{key: value / batches for key, value in totals.items()}}
                improved = False
                if phase != "pinn":
                    row["validation_l1"] = trainer.validation_l1(validation_loader)
                    improved = row["validation_l1"] < best
                    best = min(best, row["validation_l1"])
                    if improved:
                        selection = {"phase": phase, "epoch": epoch + 1, "validation_l1": best}
                checkpoint = {"format": "rthgan_manuscript_v1", "config": config, "statistics": statistics,
                    "auxiliary_definition": auxiliary_definition,
                    "training_objectives": trainer.training_objectives,
                    "model_state_position": {"phase": phase, "epoch": epoch + 1},
                    "training_git_commit": training_commit, "source_hashes": source_hashes,
                    "config_hash": fingerprint, "trainer": trainer.state_dict(), "completed": dict(completed),
                    "best_validation_l1": best, "selection": selection, "rng": capture_rng(loader_generator)}
                torch.save(checkpoint, output / "checkpoints" / "last.pth")
                if improved:
                    torch.save(checkpoint, output / "checkpoints" / "best.pth")
                with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(row, allow_nan=False), flush=True)
                write_json(output / "status.json", {"status": "running", "completed": completed,
                    "last_epoch": row, "updated_at": utc_timestamp()})
        write_json(output / "status.json", {"status": "complete", "completed": completed,
            "best_validation_l1": best, "selection": selection})
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error": str(error), "completed": completed})
        raise
    return output / "checkpoints" / "best.pth"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--model-config", default="configs/model/rthgan_manuscript.yaml")
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--thermal-config", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = deep_merge(read_yaml(args.dataset_config), read_yaml(args.model_config), read_yaml(args.train_config))
    config["thermal"] = read_yaml(args.thermal_config)
    run_training(config, args.output_dir, resume=args.resume)


if __name__ == "__main__":
    main()
