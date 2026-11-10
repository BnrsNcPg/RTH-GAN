import pytest
import torch
from torch import nn
from torchvision.ops import DeformConv2d

from experiment_utils import read_yaml
from scripts.model_profiling import build_profile_generator, contraction_counts, inference_scope, parameter_counts


def test_core_counts_match_analytic_linear_and_grouped_convolution_costs():
    model = nn.Sequential(nn.Conv2d(4, 6, 3, groups=2), nn.Flatten(), nn.Linear(6 * 3 * 3, 5))
    report = contraction_counts(model, torch.rand(2, 4, 5, 5))
    assert report["macs"] == 2 * 6 * 3 * 3 * 2 * 3 * 3 + 2 * 5 * 54
    assert report["flops"] == 2 * report["macs"]
    assert report["flops_complete"] is False
    assert not list(model[0]._forward_hooks)


def test_transposed_convolution_uses_declared_input_site_convention():
    model = nn.ConvTranspose2d(3, 2, 4, stride=2, padding=1)
    report = contraction_counts(model, torch.zeros(1, 3, 4, 4))
    assert report["macs"] == 3 * 4 * 4 * 2 * 4 * 4


def test_deformable_weighted_sums_are_included():
    class Deform(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = DeformConv2d(2, 3, 3, padding=1)

        def forward(self, value):
            return self.conv(value, value.new_zeros(1, 18, 4, 4))
    report = contraction_counts(Deform(), torch.zeros(1, 2, 4, 4))
    assert report["macs_by_component"]["deformable_convolution"] == 3 * 4 * 4 * 2 * 3 * 3


def test_manuscript_default_counts_include_frozen_prior_and_reject_mismatched_method():
    method = read_yaml("configs/methods/rthgan_manuscript.yaml")
    config = read_yaml("configs/model/rthgan_manuscript.yaml")
    model = build_profile_generator(method, config, "cpu")
    counts = parameter_counts(model)
    assert counts["params"] == 15_767_750
    assert counts["pinn_params"] == 2_785
    assert counts["trainable_params"] == counts["params"] - counts["pinn_params"]
    assert inference_scope(model)["includes_pinn_prior"]
    assert not inference_scope(model)["includes_pinn_pde_loss"]
    with pytest.raises(ValueError, match="architecture"):
        build_profile_generator(method, {"model": {"name": "unsupported"}}, "cpu")


def test_manuscript_attention_products_and_dense_prior_are_counted():
    method = read_yaml("configs/methods/rthgan_manuscript.yaml")
    config = {"model": {"name": "rthgan_manuscript", "generator": {
        "widths": [4, 8, 16, 32], "depths": [1, 1, 1, 1], "feature_channels": 8, "attention_heads": 2}}}
    model = build_profile_generator(method, config, "cpu")
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        counts = contraction_counts(model, torch.rand(1, 3, 32, 32))["macs_by_component"]
    finally:
        torch.set_num_threads(previous)
    assert counts["attention_products"] == 2 * (4 * 64**2 + 8 * 16**2 + 16 * 4**2)
    assert counts["pinn_linear"] == 32 * 32 * (19 * 32 + 2 * 32 * 32 + 32)
    assert counts["deformable_convolution"] > 0


def test_checkpoint_builder_uses_saved_architecture_instead_of_cli_config(tmp_path):
    method = read_yaml("configs/methods/rthgan_manuscript.yaml")
    config = {"model": {"name": "rthgan_manuscript", "generator": {
        "widths": [4, 8, 16, 32], "depths": [1, 1, 1, 1], "feature_channels": 8, "attention_heads": 2}}}
    model = build_profile_generator(method, config, "cpu")
    path = tmp_path / "checkpoint.pth"
    torch.save({"format": "rthgan_manuscript_v1", "config": config,
                "trainer": {"generator": model.state_dict()}}, path)
    restored = build_profile_generator(method, read_yaml("configs/model/rthgan_manuscript.yaml"), "cpu", path)
    assert parameter_counts(restored) == parameter_counts(model)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)
