import pytest
import torch
from torch import nn

from models.modules.manuscript_retinex import ManuscriptRetinex


@pytest.fixture(scope="module", autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class ConstantField(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def forward(self, coordinates, intensity):
        return self.value.expand_as(intensity)


def constant_model(value, config=None):
    model = ManuscriptRetinex(config)
    model.illumination_network = ConstantField(value)
    return model


def visible_pixels():
    # Includes a black pixel and a dark pixel below reflectance saturation.
    return torch.tensor([0., .0001, .01, 1.]).reshape(1, 1, 2, 2).repeat(1, 3, 1, 1).requires_grad_()


@pytest.mark.parametrize("illumination", [-1., -.002, -.0010001, -.001, -.0009999, -.0001])
def test_negative_illumination_and_old_singularity_have_finite_values_and_gradients(illumination):
    model = constant_model(illumination)
    visible = visible_pixels()
    priors = model(visible)
    torch.testing.assert_close(priors.illumination, torch.full((1, 1, 2, 2), illumination))
    expected = torch.tensor([0., .025, 1., 1.]).reshape(1, 1, 2, 2).repeat(1, 3, 1, 1)
    torch.testing.assert_close(priors.reflectance, expected)
    priors.reflectance.sum().backward()
    expected_gradient = torch.tensor([250., 250., 0., 0.]).reshape(1, 1, 2, 2).repeat(1, 3, 1, 1)
    torch.testing.assert_close(visible.grad, expected_gradient)
    torch.testing.assert_close(model.illumination_network.value.grad, torch.tensor(0.))


@pytest.mark.parametrize("illumination", [0., .001, .2, 2.])
def test_nonnegative_illumination_preserves_original_values_and_gradients(illumination):
    model = constant_model(illumination)
    visible = visible_pixels()
    reference_visible = visible.detach().clone().requires_grad_()
    reference_illumination = torch.tensor(illumination, requires_grad=True)
    reference = (reference_visible / (reference_illumination + .001)).clamp(0., 4.) / 4.
    priors = model(visible)
    torch.testing.assert_close(priors.reflectance, reference)
    torch.testing.assert_close(priors.illumination, torch.full((1, 1, 2, 2), illumination))
    priors.reflectance.sum().backward()
    reference.sum().backward()
    assert torch.isfinite(visible.grad).all()
    assert torch.isfinite(model.illumination_network.value.grad)
    torch.testing.assert_close(visible.grad, reference_visible.grad)
    torch.testing.assert_close(model.illumination_network.value.grad, reference_illumination.grad)


def test_reflectance_uses_fixed_scale_independent_of_image_extrema():
    model = constant_model(.249, {"reflectance_max": 2.})
    visible = torch.tensor([[[[.1, .2], [.3, .4]]]]).repeat(2, 3, 1, 1)
    visible[1, :, 1, 1] = 1.
    priors = model(visible)
    torch.testing.assert_close(priors.reflectance[:, :, 0, 0], torch.full((2, 3), .2))
    torch.testing.assert_close(priors.reflectance[:, :, 0, 1], torch.full((2, 3), .4))
    torch.testing.assert_close(priors.reflectance[0, :, 1, 1], torch.full((3,), .8))
    torch.testing.assert_close(priors.reflectance[1, :, 1, 1], torch.ones(3))
    torch.testing.assert_close(priors.intensity, visible.amax(1, keepdim=True))


def test_prior_calculation_remains_fp32_under_ambient_autocast():
    model = ManuscriptRetinex({"hidden_channels": 8, "hidden_layers": 1})
    visible = torch.rand(1, 3, 4, 4, requires_grad=True)
    observed = []
    hook = model.illumination_network.network[-1].register_forward_hook(
        lambda module, inputs, output: observed.append(output.dtype)
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        priors = model(visible)
    hook.remove()
    assert observed == [torch.float32]
    assert all(tensor.dtype == torch.float32 and torch.isfinite(tensor).all() for tensor in priors)
    (priors.reflectance.sum() + priors.illumination.sum()).backward()
    assert torch.isfinite(visible.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_visible_input_is_rejected(value):
    visible = torch.zeros(1, 3, 2, 2)
    visible[0, 0, 0, 0] = value
    with pytest.raises(ValueError, match="finite|expects visible values"):
        ManuscriptRetinex()(visible)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_illumination_is_rejected_before_clipping(value):
    with pytest.raises(FloatingPointError, match="illumination must be finite"):
        constant_model(value)(torch.zeros(1, 3, 2, 2))


@pytest.mark.parametrize("name", ["epsilon_r", "reflectance_max"])
@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf"), 1e-46, 1e-40, 1e40])
def test_unsafe_fp32_scales_are_rejected(name, value):
    with pytest.raises(ValueError, match="positive|finite"):
        ManuscriptRetinex({name: value})


def test_combined_normalization_scales_cannot_overflow_input_gradient_gain():
    with pytest.raises(ValueError, match="reciprocal overflow"):
        ManuscriptRetinex({"epsilon_r": 1e-10, "reflectance_max": 1e-30})


@pytest.mark.parametrize("epsilon", [torch.finfo(torch.float32).tiny, 1e-30])
def test_epsilon_that_can_overflow_division_backward_is_rejected(epsilon):
    with pytest.raises(ValueError, match="too small for FP32 division backward"):
        ManuscriptRetinex({"epsilon_r": epsilon})


def test_smallest_supported_epsilon_has_finite_dark_and_saturated_gradients():
    epsilon = torch.finfo(torch.float32).tiny ** .5
    model = constant_model(0., {"epsilon_r": epsilon})
    visible = torch.tensor([0., epsilon, 1., .5]).reshape(1, 1, 2, 2).repeat(1, 3, 1, 1).requires_grad_()
    priors = model(visible)
    priors.reflectance.sum().backward()
    assert torch.isfinite(priors.reflectance).all()
    assert torch.isfinite(visible.grad).all()
    assert torch.isfinite(model.illumination_network.value.grad)


def test_finite_illumination_cannot_hide_overflow_in_denominator():
    largest = torch.finfo(torch.float32).max
    model = constant_model(largest, {"epsilon_r": largest})
    with pytest.raises(FloatingPointError, match="denominator must be finite and positive"):
        model(torch.ones(1, 3, 2, 2))


def test_existing_checkpoint_parameter_keys_are_unchanged():
    model = ManuscriptRetinex({"hidden_channels": 8, "hidden_layers": 1, "positional_frequencies": 1})
    assert set(model.state_dict()) == {
        "illumination_network.encoding.frequencies",
        "illumination_network.network.0.weight", "illumination_network.network.0.bias",
        "illumination_network.network.2.weight", "illumination_network.network.2.bias",
    }
    restored = ManuscriptRetinex({"hidden_channels": 8, "hidden_layers": 1, "positional_frequencies": 1})
    restored.load_state_dict(model.state_dict(), strict=True)
    visible = torch.rand(1, 3, 2, 2)
    for actual, expected in zip(restored(visible), model(visible)):
        torch.testing.assert_close(actual, expected)
