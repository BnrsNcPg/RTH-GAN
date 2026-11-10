import pytest
import torch
from torch import nn

from models.modules.manuscript_fusion import (
    GeodesicDualSpaceFusion, geodesic_aggregate, origin_exp, origin_log, poincare_distance,
)
from models.modules.manuscript_retinex import ManuscriptRetinex
from models.rghsg import RetinexGuidedHyperbolicGenerator


@pytest.fixture(autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def tiny_generator():
    return RetinexGuidedHyperbolicGenerator(widths=(4, 8, 16, 32), depths=(1, 1, 1, 1),
        feature_channels=8, masks=3, attention_heads=2,
        pinn={"hidden_channels": 8, "hidden_layers": 1, "positional_frequencies": 1,
              "num_interior_points": 8, "num_boundary_points": 4})


def test_geodesic_endpoints_distance_fraction_and_nontrivial_geometry():
    background = origin_exp(torch.tensor([[[[.2]], [[.1]]]], dtype=torch.float64), 1.)
    foreground = origin_exp(torch.tensor([[[[-.1]], [[.3]]]], dtype=torch.float64), 1.)
    torch.testing.assert_close(geodesic_aggregate(background, foreground, 0, 1), background)
    torch.testing.assert_close(geodesic_aggregate(background, foreground, 1, 1), foreground)
    halfway = geodesic_aggregate(background, foreground, .4, 1)
    torch.testing.assert_close(poincare_distance(background, halfway, 1), .4 * poincare_distance(background, foreground, 1))
    assert not torch.allclose(halfway, .6 * background + .4 * foreground, atol=1e-5)


def test_origin_maps_preserve_zero_limit_and_derivative():
    value = torch.zeros(1, 3, 1, 1, dtype=torch.float64, requires_grad=True)
    result = origin_log(origin_exp(value, 1.), 1.)
    result.sum().backward()
    torch.testing.assert_close(result, value)
    torch.testing.assert_close(value.grad, torch.ones_like(value))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("curvature", [.1, 1., 10000.])
def test_origin_maps_and_geodesics_remain_finite_near_ball_boundary(dtype, curvature):
    value = torch.tensor([[[[100.]], [[-50.]]]], dtype=dtype, requires_grad=True)
    point = origin_exp(value, curvature)
    assert (curvature * point.square().sum(1) < 1).all()
    result = origin_log(point, curvature)
    distance = poincare_distance(point, -point, curvature)
    middle = geodesic_aggregate(point, -point, .5, curvature)
    assert all(torch.isfinite(item).all() for item in (point, result, distance, middle))
    (result.square().mean() + distance.mean() + middle.square().mean()).backward()
    assert torch.isfinite(value.grad).all()


@pytest.mark.parametrize("curvature", [.2, 1., 3.])
def test_origin_roundtrip_and_jacobian_away_from_projection(curvature):
    value = (torch.randn(1, 3, 1, 1, dtype=torch.float64) * .05).requires_grad_(True)
    torch.testing.assert_close(origin_log(origin_exp(value, curvature), curvature), value)
    assert torch.autograd.gradcheck(lambda tensor: origin_log(origin_exp(tensor, curvature), curvature), (value,))


def test_fusion_remains_finite_with_large_inputs_and_uses_both_fields():
    module = GeodesicDualSpaceFusion(4)
    background = (torch.randn(1, 4, 3, 3) * 100).requires_grad_(True)
    foreground = (torch.randn_like(background) * 100).requires_grad_(True)
    result, details = module(background, foreground, return_details=True)
    result.square().mean().backward()
    assert torch.isfinite(background.grad).all() and torch.isfinite(foreground.grad).all()
    assert (details["hyperbolic_point"].square().sum(1) < 1).all()
    assert (details["alpha"] > 0).all() and (details["alpha"] < 1).all()
    assert details["gate"].shape == background.shape


def test_fusion_accepts_autocast_feature_dtype_with_float32_weights():
    module = GeodesicDualSpaceFusion(4)
    background = torch.randn(1, 4, 2, 2, dtype=torch.bfloat16, requires_grad=True)
    foreground = torch.randn_like(background, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = module(background, foreground)
        loss = result.float().square().mean()
    loss.backward()
    assert torch.isfinite(result).all()
    assert torch.isfinite(background.grad).all() and torch.isfinite(foreground.grad).all()


def test_rghsg_output_masks_and_all_guidance_paths_have_gradients():
    torch.manual_seed(8)
    model = tiny_generator()
    output, details = model(torch.rand(1, 3, 32, 32), return_details=True)
    assert output.shape == (1, 1, 32, 32)
    assert (output.abs() <= 1).all()
    torch.testing.assert_close(details["attention"].sum(1), torch.ones(1, 16, 16))
    assert details["masks"].shape == (1, 4, 16, 16)
    assert details["masks"].abs().max() <= 1
    output.square().mean().backward()
    for branch in (model.illumination_encoder, model.reflectance_encoder, model.visible_encoder,
                   model.rpa, model.semantic_logits, model.fusion.interpolation, model.fusion.gate):
        gradients = [parameter.grad for parameter in branch.parameters() if parameter.requires_grad]
        assert gradients and all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in model.retinex.parameters())


def test_rghsg_checkpoint_roundtrip_and_context_free_inference():
    model, restored = tiny_generator().eval(), tiny_generator().eval()
    restored.load_state_dict(model.state_dict())
    visible = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(model(visible), restored(visible))
    model.train()
    assert not model.retinex.training


def test_foreground_background_use_signed_masks_and_channel_attention():
    model = tiny_generator().eval()
    projected = []
    handle = model.visible_projection.register_forward_hook(lambda module, inputs, output: projected.append(output))
    with torch.no_grad():
        model.mask_head.weight.zero_()
        model.mask_head.bias.copy_(torch.tensor([-2., -1., 1., 2.]))
        _, details = model(torch.rand(1, 3, 32, 32), return_details=True)
    handle.remove()
    feature = torch.nn.functional.interpolate(projected[0], size=(16, 16), mode="bilinear", align_corners=False)
    weights = (details["masks"][:, :-1] * details["attention"]).sum(1, keepdim=True)
    torch.testing.assert_close(details["foreground"], feature * weights)
    torch.testing.assert_close(details["background"], feature * details["masks"][:, -1:])
    assert (details["masks"][:, :2] < 0).all()
    torch.testing.assert_close(details["attention"].sum(1), torch.ones(1, 16, 16))


def test_manuscript_pinn_loss_has_finite_second_derivative_gradients():
    model = tiny_generator().retinex
    model.requires_grad_(True)
    losses = model.loss_terms(torch.rand(1, 3, 32, 32))
    losses.total.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert model.config.epsilon_w == 1e-4
    priors = model(torch.rand(1, 3, 32, 32))
    assert priors.reflectance.min() >= 0 and priors.reflectance.max() <= 1


def test_manuscript_range_penalty_is_unsquared_and_not_removed_by_sigmoid():
    class NegativeField(nn.Module):
        def forward(self, coordinates, intensity):
            return -2 + .1 * coordinates[..., :1].square()
    model = ManuscriptRetinex({"num_interior_points": 8, "num_boundary_points": 4})
    model.illumination_network = NegativeField()
    value = model.loss_terms(torch.rand(1, 3, 32, 32)).value
    assert 1.9 <= value <= 2.0


@pytest.mark.parametrize("shape", [(1, 3, 31, 32), (1, 1, 32, 32)])
def test_rghsg_rejects_invalid_visible_shape(shape):
    with pytest.raises(ValueError, match="RG-HSG requires"):
        tiny_generator()(torch.rand(shape))
