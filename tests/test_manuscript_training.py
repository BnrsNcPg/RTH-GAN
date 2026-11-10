import copy
import pytest
import torch

from experiment_utils import deep_merge, read_yaml
from manuscript_training import ManuscriptTrainer, frozen_discriminator


@pytest.fixture(autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def trainer():
    config = deep_merge(read_yaml("configs/model/rthgan_manuscript.yaml"), read_yaml("configs/train/rthgan_manuscript_smoke.yaml"),
        {"thermal": {"normalization": {"minimum_kelvin": 250., "maximum_kelvin": 350., "source_split": "train"},
         "sensor_response": {"wavelength_um": [10., 11., 12.], "response": [1., 1., 1.]}}})
    return ManuscriptTrainer(config)


def batch():
    return {"visible": torch.rand(1, 3, 32, 32), "infrared": torch.full((1, 1, 32, 32), .2),
        "context": {key: torch.tensor(value).view(1, 1, 1, 1) for key, value in
            dict(emissivity=.95, transmittance=.8, downwelling_radiance=2., upwelling_radiance=1.).items()}}


def assert_state_equal(before, after):
    assert before.keys() == after.keys()
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)


def test_pinn_pretraining_updates_only_prior_and_freezes_it_afterward(trainer):
    before_prior = copy.deepcopy(trainer.generator.retinex.state_dict())
    before_visible = copy.deepcopy(trainer.generator.visible_encoder.state_dict())
    values = trainer.pretrain_step(batch())
    assert values["pinn_total"] > 0
    assert any(not torch.equal(value, trainer.generator.retinex.state_dict()[key]) for key, value in before_prior.items())
    assert_state_equal(before_visible, trainer.generator.visible_encoder.state_dict())
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in trainer.generator.retinex.parameters())


def test_optimizer_hyperparameters_distinguish_pinn_from_gan(trainer):
    assert trainer.optimizer_pinn.param_groups[0]["lr"] == .001
    assert trainer.optimizer_pinn.param_groups[0]["betas"] == (.9, .999)
    for optimizer in (trainer.optimizer_g, trainer.optimizer_d):
        assert optimizer.param_groups[0]["lr"] == .0002
        assert optimizer.param_groups[0]["betas"] == (.5, .999)
    assert trainer.lambda_l1 == 100 and trainer.lambda_hac == 1


def test_warmup_has_no_discriminator_updates_or_buffer_drift(trainer):
    before_d = copy.deepcopy(trainer.discriminator.state_dict())
    before_prior = copy.deepcopy(trainer.generator.retinex.state_dict())
    before_g = copy.deepcopy(trainer.generator.decoder.state_dict())
    values = trainer.train_step(batch(), adversarial=False)
    assert values["adversarial"] == values["d_total"] == 0
    assert values["hac"] > 0
    assert_state_equal(before_d, trainer.discriminator.state_dict())
    assert_state_equal(before_prior, trainer.generator.retinex.state_dict())
    assert not trainer.optimizer_d.state
    assert any(not torch.equal(value, trainer.generator.decoder.state_dict()[key]) for key, value in before_g.items())


def test_joint_training_runs_nd_updates_and_preserves_frozen_prior(trainer):
    trainer.discriminator_steps = 2
    before_prior = copy.deepcopy(trainer.generator.retinex.state_dict())
    values = trainer.train_step(batch(), adversarial=True)
    assert values["adversarial"] > 0 and values["d_total"] > 0
    assert all(state["step"].item() == 2 for state in trainer.optimizer_d.state.values())
    assert all(state["step"].item() == 1 for state in trainer.optimizer_g.state.values())
    assert_state_equal(before_prior, trainer.generator.retinex.state_dict())
    assert all(parameter.grad is None for parameter in trainer.discriminator.parameters())


def test_frozen_visual_discriminator_keeps_input_derivative_and_buffers(trainer):
    visible, target = torch.rand(1, 3, 32, 32), torch.rand(1, 1, 32, 32, requires_grad=True)
    before = copy.deepcopy(trainer.discriminator.state_dict())
    with frozen_discriminator(trainer.discriminator):
        trainer.discriminator(visible, target).sum().backward()
    assert torch.isfinite(target.grad).all() and target.grad.abs().sum() > 0
    assert_state_equal(before, trainer.discriminator.state_dict())
    assert all(parameter.grad is None and parameter.requires_grad for parameter in trainer.discriminator.parameters())


def test_trainer_checkpoint_restores_models_and_optimizer_state(trainer):
    trainer.train_step(batch(), adversarial=True)
    restored = ManuscriptTrainer(trainer.config)
    restored.load_state_dict(trainer.state_dict())
    trainer.generator.eval()
    restored.generator.eval()
    visible = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(trainer.generator(visible), restored.generator(visible))
    assert len(trainer.optimizer_g.state) == len(restored.optimizer_g.state)


def test_synthetic_end_to_end_training_resume_and_context_free_evaluation(tmp_path):
    pytest.importorskip("rasterio")
    from scripts.smoke_rthgan_manuscript import run_smoke
    result = run_smoke(tmp_path / "smoke")
    assert result["synthetic_only"] and result["evaluation_scope"] == "smoke"
    assert result["num_test_images"] == 1 and result["reference_context_used_by_generator"] is False
    root = tmp_path / "smoke"
    state = torch.load(root / "run/checkpoints/last.pth", weights_only=False)
    assert state["completed"] == {"pinn": 1, "warmup": 1, "adversarial": 1}
    assert state["training_git_commit"]
    assert len(state["source_hashes"]["models/rghsg.py"]) == 64
    assert len((root / "run/history.jsonl").read_text().splitlines()) == 3
    assert (root / "results/predictions_kelvin/synthetic_test.npy").is_file()


def test_interrupted_training_resumes_to_same_weights_as_uninterrupted_run(tmp_path, monkeypatch):
    pytest.importorskip("rasterio")
    from scripts.smoke_rthgan_manuscript import run_smoke
    from train_rthgan import run_training
    run_smoke(tmp_path / "reference")
    original = ManuscriptTrainer.train_step
    def interrupted(self, batch, *, adversarial):
        if adversarial:
            raise RuntimeError("synthetic interruption before adversarial training")
        return original(self, batch, adversarial=adversarial)
    with monkeypatch.context() as patch:
        patch.setattr(ManuscriptTrainer, "train_step", interrupted)
        with pytest.raises(RuntimeError, match="synthetic interruption"):
            run_smoke(tmp_path / "interrupted")
    root = tmp_path / "interrupted/run"
    run_training(read_yaml(root / "config.yaml"), root, resume=root / "checkpoints/last.pth")
    uninterrupted = torch.load(tmp_path / "reference/run/checkpoints/last.pth", weights_only=False)
    resumed = torch.load(root / "checkpoints/last.pth", weights_only=False)
    assert_state_equal(uninterrupted["trainer"]["generator"], resumed["trainer"]["generator"])
    assert_state_equal(uninterrupted["trainer"]["discriminator"], resumed["trainer"]["discriminator"])
