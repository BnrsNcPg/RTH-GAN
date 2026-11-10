import copy
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from experiment_utils import deep_merge, file_sha256, read_yaml
from infer_rthgan import run_inference
from manuscript_protocol import experiment_identity, checkpoint_metadata, training_objectives
from manuscript_training import ManuscriptTrainer
from scripts.run_manuscript_ablation import build_config
from train_rthgan import run_training


@pytest.fixture(autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def settings(variant="rgg", control="visual_only"):
    return deep_merge(read_yaml("configs/model/rthgan_manuscript.yaml"),
        read_yaml("configs/train/rthgan_manuscript_smoke.yaml"),
        {"model": {"variant": variant, "baseline_generator": {"base_channels": 2}},
         "dataset": {"input_mode": "jpeg_intensity", "name": "synthetic_control"},
         "training": {"pinn_epochs": 0 if variant == "n" else 1},
         "thermal": read_yaml(f"configs/rtd/intensity_control_{control}.yaml")})


def jpeg_manifest(tmp_path, size):
    rows = []
    for index, split in enumerate(("train", "val", "test")):
        rng = np.random.default_rng(index)
        row = {"pair_id": f"synthetic_{split}", "sequence_id": split, "split": split,
               "manual_registration_status": "PASS_MANUAL", "sequence_registration_status": "PASS_MANUAL"}
        for name, shape in (("visible", (size, size, 3)), ("infrared", (size, size))):
            path = tmp_path / f"{split}_{name}.jpg"
            Image.fromarray(rng.integers(20, 230, size=shape, dtype=np.uint8)).save(path, quality=100)
            row[name + "_path"] = str(path)
            row[name + "_sha256"] = file_sha256(path)
        rows.append(row)
    path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_registry_resolves_nine_distinct_architecture_and_loss_identities(tmp_path):
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text("dataset:\n  name: synthetic\n  input_mode: jpeg_intensity\n  manifest: nonexistent.csv\n")
    registry = read_yaml("configs/experiment/manuscript_definitions_v1.yaml")
    configs = [build_config(dataset, variant, seed=123) for variant in registry["variants"]]
    identities = [experiment_identity(config)["method"] for config in configs]
    assert len(set(identities)) == len(configs) == 9
    assert all(config["seed"] == 123 and config["training"]["max_batches"] is None for config in configs)
    assert configs[0]["training"]["pinn_epochs"] == 0
    assert all(config["model"]["baseline_generator"]["base_channels"] == 64 for config in configs)
    for variant in registry["variants"]:
        smoke = build_config(dataset, variant, smoke=True)
        assert smoke["training"]["max_batches"] == 1
        assert smoke["training"]["pinn_epochs"] == (0 if variant == "n" else 1)
    with pytest.raises(ValueError, match="Unknown manuscript variant"):
        build_config(dataset, "typo")


@pytest.mark.parametrize("control", ["visual_only", "uniform", "clipping_matched_l1", "hcw"])
def test_trainer_preserves_basic_l1_and_applies_auxiliary_coefficient_once(control):
    config = settings(control=control)
    config["training"]["lambda_hac"] = 2.5
    trainer = ManuscriptTrainer(config)
    prior = copy.deepcopy(trainer.generator.retinex.state_dict())
    values = trainer.train_step({"visible": torch.rand(1, 3, 32, 32),
        "infrared": torch.full((1, 1, 32, 32), .2), "context": {}}, adversarial=False)
    assert values["g_total"] == pytest.approx(100 * values["l1"] + 2.5 * values["hac"], rel=1e-6)
    assert (values["hac"] == 0) == (control == "visual_only")
    for key, value in prior.items():
        torch.testing.assert_close(value, trainer.generator.retinex.state_dict()[key], atol=0, rtol=0)


@pytest.mark.parametrize("variant", ["n", "n_pinn"])
def test_unet_training_resume_and_inference_use_the_saved_variant(tmp_path, variant):
    manifest = jpeg_manifest(tmp_path, 256)
    config = settings(variant)
    config["dataset"]["manifest"] = str(manifest)
    run = tmp_path / "run"
    checkpoint = run_training(config, run)
    state = torch.load(checkpoint, weights_only=False)
    assert state["completed"]["pinn"] == (0 if variant == "n" else 1)
    assert (state["trainer"]["optimizer_pinn"] is None) == (variant == "n")
    assert state["selection"]["phase"] in {"warmup", "adversarial"}
    assert state["selection"]["epoch"] == 1
    before_history = (run / "history.jsonl").read_bytes()
    run_training(config, run, resume=run / "checkpoints/last.pth")
    assert (run / "history.jsonl").read_bytes() == before_history
    report = run_inference(checkpoint, manifest, tmp_path / "results", split="test", metrics=False)
    assert report["generator_variant"] == variant
    assert report["auxiliary_control"] == "visual_only"
    assert report["selected_checkpoint"] == state["selection"]
    assert report["auxiliary_definition"]["effective_additional_l1_coefficient"] == 0
    assert report["experiment_name"] == config["experiment_name"]
    assert (tmp_path / "results/predictions_intensity/synthetic_test.png").is_file()


def test_matched_control_records_coefficient_config_and_selected_checkpoint(tmp_path):
    manifest = jpeg_manifest(tmp_path, 32)
    config = settings(control="clipping_matched_l1")
    config["dataset"]["manifest"] = str(manifest)
    config["training"]["lambda_hac"] = 2.5
    run = tmp_path / "matched"
    checkpoint = run_training(config, run)
    state = torch.load(checkpoint, weights_only=False)
    definition = state["auxiliary_definition"]
    assert definition["effective_additional_l1_coefficient"] == pytest.approx(2.5 * .32 / math.pi)
    provenance = json.loads((run / "provenance.json").read_text())
    assert provenance["auxiliary_definition"] == definition
    assert provenance["lambda_l1"] == 100
    changed = copy.deepcopy(config)
    changed["thermal"] = read_yaml("configs/rtd/intensity_control_uniform.yaml")
    with pytest.raises(ValueError, match="do not match"):
        run_training(changed, run, resume=run / "checkpoints/last.pth")


def test_n_cannot_silently_log_fake_pinn_pretraining(tmp_path):
    config = settings("n")
    config["training"]["pinn_epochs"] = 1
    with pytest.raises(ValueError, match="pinn_epochs=0"):
        run_training(config, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_last_checkpoint_position_is_not_mislabeled_as_the_best_selection():
    selection = {"phase": "warmup", "epoch": 2, "validation_l1": .1}
    state = {"selection": selection, "model_state_position": {"phase": "adversarial", "epoch": 7}}
    metadata = checkpoint_metadata(state)
    assert metadata["checkpoint_position"] == {"phase": "adversarial", "epoch": 7}
    assert metadata["best_validation_selection"] == selection
    assert metadata["selected_checkpoint"] is None
    old = {"completed": {"pinn": 50, "warmup": 20, "adversarial": 0}}
    assert checkpoint_metadata(old)["checkpoint_position"] == {"phase": "warmup", "epoch": 20}


def test_explicit_quantitative_controls_keep_distinct_result_identities():
    config = settings()
    config["dataset"]["input_mode"] = "quantitative_landsat"
    config["thermal"] = {"weighting": "uniform"}
    uniform = experiment_identity(config)
    config["thermal"]["weighting"] = "hcw"
    hcw = experiment_identity(config)
    assert uniform["auxiliary_control"] == "thermal_uniform"
    assert uniform["method"] != hcw["method"]


V2_REGISTRY = "configs/experiment/manuscript_reproducibility_v2.yaml"


def test_v2_registry_preserves_v1_and_declares_distinct_rpa_and_loss_identities(tmp_path):
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text("dataset:\n  name: synthetic_v2\n  input_mode: jpeg_intensity\n  manifest: missing.csv\n")
    old = build_config(dataset, "full")
    assert old["experiment_name"] == "manuscript_v1_synthetic_v2_full_seed42"
    assert experiment_identity(old)["method"] == "rthgan_jpeg_rgg_full_hcw"
    variants = read_yaml(V2_REGISTRY)["variants"]
    configs = [build_config(dataset, v, protocol_config=V2_REGISTRY) for v in variants]
    assert len({experiment_identity(c)["method"] for c in configs}) == 4
    assert all(experiment_identity(c)["method"] != experiment_identity(old)["method"] for c in configs)
    assert all(c["training"]["warmup_epochs"] == 20 for c in configs)
    assert configs[-1]["model"]["generator"]["rpa_variant"] == "plain_skip"
    for c in configs:
        assert training_objectives(c)["stage_i"]["objective"] == c["training"]["stage_i_objective"]
    broken = copy.deepcopy(configs[0])
    broken["training"]["stage_i_objective"] = "l1+hcw"
    with pytest.raises(ValueError, match="stage_i_objective"):
        ManuscriptTrainer(broken)


@pytest.mark.parametrize("control,term", [("visual_only", None), ("uniform", "uniform_proxy"), ("hcw", "hcw")])
def test_stage_i_and_ii_follow_the_declared_control_and_freeze_contract(control, term, monkeypatch):
    config = settings(control=control)
    config["training"]["stage_i_objective"] = "l1" if term is None else "l1+" + term
    trainer = ManuscriptTrainer(config)
    prior = copy.deepcopy(trainer.generator.retinex.state_dict())
    discriminator = copy.deepcopy(trainer.discriminator.state_dict())
    batch = {"visible": torch.rand(1, 3, 32, 32), "infrared": torch.full((1, 1, 32, 32), .2), "context": {}}
    with monkeypatch.context() as patch:
        patch.setattr(trainer.discriminator, "forward", lambda *a: pytest.fail("Stage I evaluated D"))
        warm = trainer.train_step(batch, adversarial=False)
    assert warm["g_total"] == pytest.approx(100 * warm["l1"] + warm["hac"], rel=1e-6)
    assert warm["adversarial"] == warm["d_total"] == 0
    assert (warm["hac"] == 0) == (control == "visual_only")
    assert not trainer.optimizer_d.state
    for key, value in discriminator.items():
        torch.testing.assert_close(value, trainer.discriminator.state_dict()[key], atol=0, rtol=0)
    joint = trainer.train_step(batch, adversarial=True)
    assert joint["g_total"] == pytest.approx(100 * joint["l1"] + joint["hac"] + joint["adversarial"], rel=1e-6)
    assert joint["d_total"] > 0 and joint["adversarial"] > 0
    for key, value in prior.items():
        torch.testing.assert_close(value, trainer.generator.retinex.state_dict()[key], atol=0, rtol=0)


@pytest.mark.parametrize("variant", ["visual_only", "uniform", "full", "without_rpa"])
def test_v2_synthetic_training_records_objectives_and_restores_saved_rpa(tmp_path, variant):
    manifest = jpeg_manifest(tmp_path, 32)
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(f"dataset:\n  name: synthetic_v2\n  input_mode: jpeg_intensity\n  manifest: {manifest.as_posix()}\n")
    config = build_config(dataset, variant, protocol_config=V2_REGISTRY, smoke=True, device="cpu")
    run = tmp_path / "run"
    checkpoint = run_training(config, run)
    state = torch.load(checkpoint, weights_only=False)
    expected = training_objectives(config)
    assert state["training_objectives"] == expected
    assert json.loads((run / "provenance.json").read_text())["training_objectives"] == expected
    history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
    warm = next(row for row in history if row["phase"] == "warmup")
    assert warm["adversarial"] == warm["d_total"] == 0
    assert warm["g_total"] == pytest.approx(100 * warm["l1"] + warm["hac"], rel=1e-6)
    before = (run / "history.jsonl").read_bytes()
    run_training(config, run, resume=checkpoint)
    assert (run / "history.jsonl").read_bytes() == before
    report = run_inference(checkpoint, manifest, tmp_path / "results", metrics=False)
    assert report["training_objectives"] == expected
    assert report["rpa_variant"] == ("plain_skip" if variant == "without_rpa" else "full")
    assert report["method"].endswith("_repro_v2")
