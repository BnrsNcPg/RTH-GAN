"""Meaningful loss/gradient checks for the proposed Table 9 control protocol."""
import copy
import json
import math

import pytest
import torch

from experiment_utils import read_yaml
from models.thermal_rtd import build_thermal_auxiliary


CONTROLS = ("hcw", "uniform", "clipping_matched_l1", "visual_only")


def settings(control):
    return read_yaml(f"configs/rtd/intensity_control_{control}.yaml")


def test_uniform_and_clipping_matched_l1_agree_in_value_and_gradient_at_boundaries():
    uniform = build_thermal_auxiliary(settings("uniform"))
    matched = build_thermal_auxiliary(settings("clipping_matched_l1"))
    threshold = 2 * uniform.extractor.eps - 1
    values = [-1., threshold - 5e-7, threshold, threshold + 5e-7, -.4, .3, 1.]
    generated = torch.tensor(values, dtype=torch.float64).reshape(1, 1, 1, -1).requires_grad_()
    reference = torch.full_like(generated, .6, requires_grad=True)
    proxy = uniform(generated, reference, {})
    control = matched(generated, reference, {})
    torch.testing.assert_close(control.loss, proxy.loss, rtol=1e-12, atol=1e-15)
    proxy_grad = torch.autograd.grad(proxy.loss, generated)[0]
    matched_grad = torch.autograd.grad(control.loss, generated)[0]
    torch.testing.assert_close(matched_grad, proxy_grad, rtol=1e-12, atol=1e-15)
    assert torch.equal(matched_grad[..., :2], torch.zeros_like(matched_grad[..., :2]))
    assert matched_grad[..., -1].abs().item() > 0
    # Raw L1 is deliberately different below the epsilon floor.
    raw_gradient = torch.autograd.grad((generated - reference.detach()).abs().mean(), generated)[0]
    assert (raw_gradient[..., :2].abs() > 0).all()
    both_clipped = torch.tensor([[[[-1.]]]], dtype=torch.float64, requires_grad=True)
    clipped_reference = torch.full_like(both_clipped, threshold - 5e-7)
    collapsed = matched(both_clipped, clipped_reference, {})
    assert collapsed.loss.item() == 0
    assert torch.autograd.grad(collapsed.loss, both_clipped)[0].item() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matched_l1_derives_scale_and_matches_raw_l1_only_interior(dtype):
    config = settings("clipping_matched_l1")
    config["proxy_context"]["tau"] = [.2]  # Detect a hard-coded 0.32/pi coefficient.
    auxiliary = build_thermal_auxiliary(config)
    generated = torch.tensor([[[[-.7, .2], [.4, .8]]]], dtype=dtype, requires_grad=True)
    reference = torch.full_like(generated, -.2)
    result = auxiliary(generated, reference, {})
    coefficient = .2 * (.6 + .2) / (2 * math.pi)
    expected = coefficient * (generated - reference).abs().mean()
    torch.testing.assert_close(result.loss, expected)
    torch.testing.assert_close(
        torch.autograd.grad(result.loss, generated)[0],
        torch.autograd.grad(expected, generated)[0],
    )
    assert float(auxiliary.matched_l1_coefficient()) == pytest.approx(coefficient)


@pytest.mark.parametrize("control", CONTROLS)
def test_controls_freeze_reference_and_keep_only_generated_gradient(control):
    auxiliary = build_thermal_auxiliary(settings(control))
    generated = torch.zeros(2, 1, 3, 3, requires_grad=True)
    reference = torch.full_like(generated, .4, requires_grad=True)
    result = auxiliary(generated, reference, {})
    assert not result.reference.requires_grad and not result.weights.requires_grad
    assert not list(auxiliary.parameters())
    result.loss.backward()
    assert reference.grad is None and torch.isfinite(generated.grad).all()
    if control == "visual_only":
        assert result.loss.item() == 0 and generated.grad.abs().sum().item() == 0
        assert result.weights.abs().sum().item() == 0
    else:
        assert generated.grad.abs().sum().item() > 0


def test_hcw_uses_pixel_count_and_per_image_weights_not_weight_sum():
    hcw = build_thermal_auxiliary(settings("hcw"))
    uniform = build_thermal_auxiliary(settings("uniform"))
    generated = torch.full((2, 1, 3, 4), .5, dtype=torch.float64)
    reference = torch.zeros_like(generated)
    result = hcw(generated, reference, {})
    torch.testing.assert_close(result.weights, torch.full_like(reference, .2))
    torch.testing.assert_close(result.loss, .2 * uniform(generated, reference, {}).loss)
    pixel_mean = (result.weights * (result.generated - result.reference).abs()).flatten(1).mean(1).mean()
    torch.testing.assert_close(result.loss, pixel_mean)
    normalized_by_weights = (result.weights * (result.generated - result.reference).abs()).sum() / result.weights.sum()
    assert not torch.isclose(result.loss, normalized_by_weights)
    varied_reference = torch.linspace(-.8, .8, reference.numel(), dtype=torch.float64).reshape_as(reference)
    batch_loss = hcw(generated, varied_reference, {}).loss
    single_losses = [hcw(generated[i:i+1], varied_reference[i:i+1], {}).loss for i in range(2)]
    torch.testing.assert_close(batch_loss, torch.stack(single_losses).mean())


@pytest.mark.parametrize("control", ("uniform", "clipping_matched_l1"))
@pytest.mark.parametrize("lambda_hac", [0., 1., 2.5])
def test_outer_lambda_hac_is_applied_once_without_reconstruction_multiplier(control, lambda_hac):
    auxiliary = build_thermal_auxiliary(settings(control))
    generated = torch.full((1, 1, 2, 2), .3, dtype=torch.float64, requires_grad=True)
    reference = torch.full_like(generated, -.2)
    result = auxiliary(generated, reference, {})
    raw_l1 = (generated - reference).abs().mean()
    coefficient = .32 / math.pi
    total = 100 * raw_l1 + lambda_hac * result.loss
    expected = (100 + lambda_hac * coefficient) * (generated - reference).abs().mean()
    torch.testing.assert_close(total, expected, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(
        torch.autograd.grad(total, generated)[0],
        torch.autograd.grad(expected, generated)[0],
        rtol=1e-12, atol=1e-14,
    )


def test_legacy_default_hcw_and_uniform_keep_checkpoint_state_keys_and_behavior():
    legacy = read_yaml("configs/rtd/intensity_proxy_hcw.yaml")
    generated = torch.full((1, 1, 3, 3), .3)
    reference = torch.zeros_like(generated)
    for control in ("hcw", "uniform"):
        old = build_thermal_auxiliary({**legacy, "weighting": control})
        explicit = build_thermal_auxiliary(settings(control))
        explicit.load_state_dict(old.state_dict(), strict=True)
        assert set(old.state_dict()) == set(explicit.state_dict())
        assert old.control == explicit.control == control
        torch.testing.assert_close(old(generated, reference, {}).loss, explicit(generated, reference, {}).loss)
    no_weighting = copy.deepcopy(legacy)
    no_weighting.pop("weighting")
    assert build_thermal_auxiliary(no_weighting).control == "hcw"


@pytest.mark.parametrize("updates,match", [
    ({"control": "matched_l1"}, "control must"),
    ({"control": None}, "control must"),
    ({"control": []}, "control must"),
    ({"weighting": []}, "weighting must"),
    ({"control": "hcw", "weighting": "uniform"}, "conflicts"),
    ({"control": "clipping_matched_l1", "weighting": "hcw"}, "conflicts"),
    ({"lambda_match": 10.}, "derived"),
    ({"matched_l1_coefficient": .1}, "derived"),
    ({"physical_interpretation": True}, "disable physical"),
])
def test_ambiguous_or_invalid_control_configurations_are_rejected(updates, match):
    with pytest.raises(ValueError, match=match):
        build_thermal_auxiliary({**settings("uniform"), **updates})


@pytest.mark.parametrize("control", CONTROLS)
def test_control_definitions_are_json_serializable_and_report_the_executed_coefficient(control):
    auxiliary = build_thermal_auxiliary(settings(control))
    definition = auxiliary.control_definition()
    assert json.loads(json.dumps(definition, allow_nan=False)) == definition
    assert definition["control"] == control
    assert definition["clip_epsilon"] == 1e-6
    assert definition["proxy_scale"] == pytest.approx(.64 / math.pi)
    assert definition["proxy_offset"] == pytest.approx(.1)
    assert definition["reduction"] == "mean_pixels_per_image_then_mean_batch"
    if control in {"uniform", "clipping_matched_l1"}:
        assert definition["additional_l1_coefficient_before_lambda_hac"] == pytest.approx(.32 / math.pi)
    elif control == "visual_only":
        assert definition["additional_l1_coefficient_before_lambda_hac"] == 0
    else:
        assert definition["additional_l1_coefficient_before_lambda_hac"] is None
        assert definition["additional_weighted_l1_coefficient_before_lambda_hac"] == pytest.approx(.32 / math.pi)


@pytest.mark.parametrize("control", CONTROLS)
def test_controls_reject_physical_context_and_out_of_range_intensities(control):
    auxiliary = build_thermal_auxiliary(settings(control))
    generated = torch.zeros(1, 1, 2, 2)
    with pytest.raises(ValueError, match="physical thermal context"):
        auxiliary(generated, generated, {"emissivity": .95})
    with pytest.raises(ValueError, match=r"\[-1,1\]"):
        auxiliary(generated - 2, generated, {})
