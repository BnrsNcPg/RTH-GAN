import pytest
import torch
from torch.nn import functional as F

from models.manuscript_ablation import ManuscriptUNetAblation, build_manuscript_generator
from models.modules.manuscript_fusion import GeodesicDualSpaceFusion
from models.rghsg import PlainSkipFusion, ReflectancePriorAttention, RetinexGuidedHyperbolicGenerator


@pytest.fixture(autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def tiny_settings():
    return {"widths": [4, 8, 16, 32], "depths": [1, 1, 1, 1],
            "feature_channels": 8, "masks": 3, "attention_heads": 2,
            "pinn": {"hidden_channels": 8, "hidden_layers": 1, "positional_frequencies": 1,
                     "num_interior_points": 8, "num_boundary_points": 4}}


@pytest.mark.parametrize("variant", ["mean", "local_only", "fixed_mean", "full"])
def test_fusion_ablation_uses_defined_operator_and_only_retained_parameters(variant):
    module = GeodesicDualSpaceFusion(4, variant=variant)
    background = torch.randn(1, 4, 3, 3, requires_grad=True)
    foreground = torch.randn_like(background, requires_grad=True)
    output, details = module(background, foreground, return_details=True)
    if variant == "mean":
        torch.testing.assert_close(output, (background + foreground) / 2)
        assert not list(module.parameters())
    elif variant == "local_only":
        torch.testing.assert_close(output, module.euclidean(torch.cat([background, foreground], 1)))
        assert not hasattr(module, "interpolation") and not hasattr(module, "gate")
    elif variant == "fixed_mean":
        torch.testing.assert_close(output, (details["local"] + details["radial"]) / 2)
        assert not hasattr(module, "gate")
        assert (details["alpha"] > 0).all() and (details["alpha"] < 1).all()
    else:
        torch.testing.assert_close(output, details["gate"] * details["local"]
                                   + (1 - details["gate"]) * details["radial"])
    output.square().mean().backward()
    assert torch.isfinite(background.grad).all() and torch.isfinite(foreground.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    if variant == "fixed_mean":
        assert sum(float(p.grad.abs().sum()) for p in module.interpolation.parameters()) > 0


def test_default_factory_preserves_full_model_checkpoint_contract():
    settings = tiny_settings()
    original = RetinexGuidedHyperbolicGenerator(**settings).eval()
    restored = build_manuscript_generator({"name": "rthgan_manuscript", "generator": settings}).eval()
    assert original.state_dict().keys() == restored.state_dict().keys()
    restored.load_state_dict(original.state_dict(), strict=True)
    visible = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(original(visible), restored(visible))


@pytest.mark.parametrize("variant", ["mean", "local_only", "fixed_mean"])
def test_rgg_fusion_ablations_keep_decoder_shape_and_upstream_gradients(variant):
    model = build_manuscript_generator({"variant": "rgg", "generator": {
        **tiny_settings(), "fusion_variant": variant}})
    output, details = model(torch.rand(1, 3, 32, 32), return_details=True)
    assert output.shape == (1, 1, 32, 32) and output.abs().max() <= 1
    assert details["foreground"].shape == (1, 8, 16, 16)
    output.square().mean().backward()
    for branch in (model.visible_encoder, model.illumination_encoder, model.reflectance_encoder,
                   model.mask_head, model.decoder):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.retinex.parameters())


def test_n_pinn_zero_initialization_preserves_n_and_frozen_prior_training_contract():
    settings = tiny_settings()
    baseline = build_manuscript_generator({"variant": "n", "baseline_generator": {"base_channels": 2}}).eval()
    model = build_manuscript_generator({"variant": "n_pinn", "generator": settings,
                                       "baseline_generator": {"base_channels": 2}}).eval()
    incompatible = model.load_state_dict(baseline.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(key.startswith(("retinex.", "illumination_projection.", "reflectance_projection."))
               for key in incompatible.missing_keys)
    assert baseline.retinex is None
    visible = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        expected, baseline_details = baseline(visible, return_details=True)
        actual, details = model(visible, return_details=True)
    torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    assert baseline_details == {}
    assert details["reflectance"].min() >= 0 and details["reflectance"].max() <= 1
    assert model.retinex.config.epsilon_w == 1e-4
    model.train()
    assert not model.retinex.training
    model(visible).square().mean().backward()
    assert all(p.grad is None and not p.requires_grad for p in model.retinex.parameters())
    for projection in (model.illumination_projection, model.reflectance_projection):
        assert torch.isfinite(projection.weight.grad).all()
        assert projection.weight.grad.abs().sum() > 0
    model.retinex.requires_grad_(True).train()
    model.freeze_retinex()
    assert not model.retinex.training and not any(p.requires_grad for p in model.retinex.parameters())


def test_n_pinn_injects_unit_range_priors_at_the_first_encoder_feature():
    model = ManuscriptUNetAblation(with_pinn=True, base_channels=2, pinn=tiny_settings()["pinn"]).eval()
    with torch.no_grad():
        model.illumination_projection.weight.fill_(0.01)
        model.reflectance_projection.weight.fill_(0.02)
    first, guided, prior_input = [], [], []
    handles = [model.down1.register_forward_hook(lambda module, args, result: first.append(result)),
               model.down2.register_forward_pre_hook(lambda module, args: guided.append(args[0])),
               model.retinex.register_forward_pre_hook(lambda module, args: prior_input.append(args[0]))]
    visible = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        output, details = model(visible, return_details=True)
        size = first[0].shape[-2:]
        expected = first[0] + 0.1 * (
            model.illumination_projection(F.interpolate(details["illumination"], size=size,
                                                        mode="bilinear", align_corners=False))
            + model.reflectance_projection(F.interpolate(details["reflectance"], size=size,
                                                          mode="bilinear", align_corners=False)))
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(prior_input[0], visible, atol=0, rtol=0)
    torch.testing.assert_close(guided[0], expected)
    assert output.shape == (1, 1, 256, 256) and output.abs().max() <= 1


@pytest.mark.parametrize("variant", ["n", "n_pinn"])
def test_unet_ablation_checkpoint_roundtrip(variant):
    config = {"variant": variant, "generator": tiny_settings(), "baseline_generator": {"base_channels": 2}}
    model, restored = build_manuscript_generator(config).eval(), build_manuscript_generator(config).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    visible = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        torch.testing.assert_close(model(visible), restored(visible))


@pytest.mark.parametrize("shape", [(1, 3, 128, 256), (1, 1, 256, 256), (3, 256, 256)])
def test_unet_ablation_rejects_invalid_shape(shape):
    with pytest.raises(ValueError, match="Manuscript U-Net requires"):
        ManuscriptUNetAblation(base_channels=2)(torch.rand(shape))


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_unet_ablation_rejects_invalid_input_range(value):
    visible = torch.rand(1, 3, 256, 256)
    visible[0, 0, 0, 0] = value
    with pytest.raises(ValueError, match=r"finite in \[0, 1\]"):
        ManuscriptUNetAblation(base_channels=2)(visible)


def test_ablation_factory_rejects_unknown_variants():
    with pytest.raises(ValueError, match="model.variant"):
        build_manuscript_generator({"variant": "unknown"})
    with pytest.raises(ValueError, match="Fusion variant"):
        GeodesicDualSpaceFusion(4, variant="sum")


def test_plain_skip_replacement_ignores_reflectance_and_preserves_skip_gradients():
    module = PlainSkipFusion(8, 4)
    low = torch.randn(1, 8, 3, 3, requires_grad=True)
    high = torch.randn(1, 4, 6, 6, requires_grad=True)
    prior = torch.randn_like(high, requires_grad=True)
    actual = module(low, high, prior)
    expected = module.merge(torch.cat([module.upsample(low), high], dim=1))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual, module(low, high, prior + 10), atol=0, rtol=0)
    actual.square().mean().backward()
    assert prior.grad is None
    assert torch.isfinite(low.grad).all() and low.grad.abs().sum() > 0
    assert torch.isfinite(high.grad).all() and high.grad.abs().sum() > 0
    assert not any(isinstance(m, ReflectancePriorAttention) for m in module.modules())


def test_without_rpa_keeps_full_mask_and_guidance_paths_and_strict_checkpoint():
    from scripts.model_profiling import contraction_counts
    config = {"variant": "rgg", "generator": {**tiny_settings(), "rpa_variant": "plain_skip"}}
    model = build_manuscript_generator(config)
    assert all(isinstance(m, PlainSkipFusion) for m in model.rpa)
    assert len(model.mask_decoder) == 3 and hasattr(model, "semantic_logits")
    counts = contraction_counts(model, torch.rand(1, 3, 32, 32))["macs_by_component"]
    assert counts.get("attention_products", 0) == counts.get("deformable_convolution", 0) == 0
    output, details = model(torch.rand(1, 3, 32, 32), return_details=True)
    assert output.shape == (1, 1, 32, 32) and torch.isfinite(output).all()
    assert details["foreground"].shape == details["background"].shape == (1, 8, 16, 16)
    output.square().mean().backward()
    for branch in (model.visible_encoder, model.illumination_encoder, model.reflectance_encoder,
                   model.semantic_logits, model.mask_head, model.decoder, model.fusion):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.retinex.parameters())
    restored = build_manuscript_generator(config).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    visible = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(restored(visible), model(visible), atol=0, rtol=0)
    with pytest.raises(ValueError, match="RPA variant"):
        RetinexGuidedHyperbolicGenerator(**tiny_settings(), rpa_variant="unknown")
