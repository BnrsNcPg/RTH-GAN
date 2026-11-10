import math

import numpy as np
import pytest
import torch

from models.thermal_rtd import ThermalAuxiliaryConsistency, build_thermal_auxiliary
from rtd.hcw import HomogeneityAwareConsistencyWeighting
from rtd.thermal import TemperatureNormalization, ThermalRadiativeExtractor


def normalization():
    return TemperatureNormalization(250.0, 350.0, source_split="train")


def extractor():
    return ThermalRadiativeExtractor([8.0, 10.0, 13.0], [0.2, 1.0, 0.4])


def context(**updates):
    result = dict(emissivity=0.95, transmittance=0.8, downwelling_radiance=2.0, upwelling_radiance=1.0)
    return {**result, **updates}


def test_temperature_normalization_is_invertible_and_does_not_refit_or_clip():
    transform = normalization()
    kelvin = torch.tensor([240., 250., 300., 350., 360.], dtype=torch.float64)
    torch.testing.assert_close(transform.normalize(kelvin), torch.tensor([-1.2, -1., 0., 1., 1.2], dtype=torch.float64))
    torch.testing.assert_close(transform(transform.normalize(kelvin)), kelvin)
    with pytest.raises(ValueError, match="train split"):
        TemperatureNormalization(250, 350, source_split="test")


def test_sensor_band_planck_matches_independent_scalar_integration():
    # Direct scalar Planck calculation in SI units, with an irregular sensor grid.
    wavelength_um = np.array([8., 10., 13.])
    response = np.array([0.2, 1., 0.4])
    spectrum = []
    for wavelength in wavelength_um * 1e-6:
        spectrum.append(2 * 6.62607015e-34 * 299792458.**2 / wavelength**5 /
                        math.expm1(6.62607015e-34 * 299792458. / (wavelength * 1.380649e-23 * 300.)) * 1e-6)
    steps = np.diff(wavelength_um)
    weighted = np.array(spectrum) * response
    expected = sum((weighted[:-1] + weighted[1:]) * steps / 2) / sum((response[:-1] + response[1:]) * steps / 2)
    actual = extractor().blackbody_radiance(torch.full((1, 1, 1, 1), 300., dtype=torch.float64))
    assert actual.item() == pytest.approx(expected, rel=1e-12)
    assert 8 < actual.item() < 11  # Spectral radiance per micrometre, not per metre.


def test_planck_gradient_matches_numerical_differentiation():
    kelvin = torch.tensor([[[[280., 310.]]]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(extractor().blackbody_radiance, (kelvin,))
    gradient = torch.autograd.grad(extractor().blackbody_radiance(kelvin).sum(), kelvin)[0]
    assert (gradient > 0).all()


def test_shared_context_offsets_cancel_without_context_gradients():
    operator = extractor()
    temperature = torch.tensor([[[[290., 310.]]]], requires_grad=True)
    emission = torch.tensor(0.9, requires_grad=True)
    paired = context(emissivity=emission)
    first = operator(temperature, paired) - operator(temperature + 5, paired)
    second = operator(temperature, context(emissivity=emission, downwelling_radiance=7, upwelling_radiance=4))
    second = second - operator(temperature + 5, context(emissivity=emission, downwelling_radiance=7, upwelling_radiance=4))
    torch.testing.assert_close(first, second)
    first.sum().backward()
    assert temperature.grad is not None and torch.isfinite(temperature.grad).all()
    assert emission.grad is None
    assert not list(operator.parameters())


@pytest.mark.parametrize("wavelength,response", [([10.], [1.]), ([10., 9.], [1., 1.]), ([9., 10.], [0., 0.]), ([9., 10.], [-1., 1.])])
def test_invalid_sensor_response_is_rejected(wavelength, response):
    with pytest.raises(ValueError):
        ThermalRadiativeExtractor(wavelength, response)


@pytest.mark.parametrize("updates", [dict(emissivity=1.1), dict(transmittance=-0.1), dict(upwelling_radiance=float("nan")), dict(transmittance=torch.ones(2))])
def test_invalid_context_is_rejected(updates):
    with pytest.raises(ValueError):
        extractor()(torch.full((2, 1, 2, 2), 300.), context(**updates))


def test_hcw_matches_literal_appendix_a_windows_and_population_variance():
    values = np.random.default_rng(2).uniform(260, 340, size=(4, 5))
    weighting = HomogeneityAwareConsistencyWeighting(normalization(), window_size=3, protection_size=3)
    field = torch.tensor(values[None, None], dtype=torch.float64)
    weights, details = weighting(field, return_details=True)
    normalized = (values - 250) / 100
    background = np.zeros_like(values)
    for row in range(4):
        for column in range(5):
            stats = []
            for index, (dy, dx) in enumerate(weighting.DIRECTIONS):
                window = [normalized[np.clip(row + 3 * dy + y, 0, 3), np.clip(column + 3 * dx + x, 0, 4)]
                          for y in range(-1, 2) for x in range(-1, 2)]
                variance = 0.0 if min(window) == max(window) else np.var(window, ddof=0)
                stats.append((variance, index, np.mean(window)))
            background[row, column] = np.mean([value[2] for value in sorted(stats)[:3]])
    deviation = np.abs(normalized - background)
    expected = 0.2 + 0.8 * deviation / (deviation + deviation.mean() + 1e-6)
    np.testing.assert_allclose(details["background"][0, 0].numpy(), background, atol=1e-14)
    np.testing.assert_allclose(weights[0, 0].numpy(), expected, atol=1e-14)


def test_hcw_constant_field_ties_use_direction_order():
    weighting = HomogeneityAwareConsistencyWeighting(normalization())
    reference = torch.full((2, 1, 3, 4), 300., requires_grad=True)
    weights, details = weighting(reference, return_details=True)
    torch.testing.assert_close(weights, torch.full_like(weights, 0.2))
    assert torch.equal(details["directions"][0, :, 0, 0], torch.tensor([0, 1, 2]))
    assert not weights.requires_grad


def test_hcw_is_per_image_and_symmetric_for_warm_and_cool_structure():
    hot = torch.full((1, 1, 15, 15), 300.)
    hot[..., 7, 7] = 320
    cold = 600 - hot
    weighting = HomogeneityAwareConsistencyWeighting(normalization())
    weights = weighting(torch.cat([hot, cold]))
    torch.testing.assert_close(weights[0], weights[1])
    torch.testing.assert_close(weights[:1], weighting(hot))
    assert weights[0, 0, 7, 7] > weights[0, 0, 0, 0]
    assert (weights >= 0.2).all() and (weights < 1).all()


def test_hac_uses_pixel_count_and_only_generated_gradient_path():
    auxiliary = ThermalAuxiliaryConsistency(normalization(), extractor())
    generated = torch.full((2, 1, 5, 5), 0.2, requires_grad=True)
    reference = torch.zeros_like(generated, requires_grad=True)
    output = auxiliary(generated, reference, context())
    expected_uniform = (output.generated - output.reference).abs().mean()
    torch.testing.assert_close(output.loss, expected_uniform * 0.2)
    assert not output.reference.requires_grad and not output.weights.requires_grad
    output.loss.backward()
    assert generated.grad is not None and (generated.grad > 0).all()
    assert reference.grad is None
    assert not list(auxiliary.parameters())


def test_hac_lower_bound_and_uniform_control():
    generated = torch.rand(2, 1, 9, 9) * 0.4
    reference = torch.rand_like(generated) * 0.4
    weighted = ThermalAuxiliaryConsistency(normalization(), extractor())(generated, reference, context())
    uniform = ThermalAuxiliaryConsistency(normalization(), extractor(), weighting="uniform")(generated, reference, context())
    assert 0.2 * uniform.loss <= weighted.loss <= uniform.loss


def test_template_requires_real_normalization_and_sensor_response():
    import yaml
    from pathlib import Path
    template = yaml.safe_load(Path("configs/rtd/thermal_template.yaml").read_text())
    with pytest.raises(ValueError, match="train-split temperature"):
        build_thermal_auxiliary(template)
