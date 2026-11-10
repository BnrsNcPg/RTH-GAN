"""Manuscript thermal RTD, separate from historical normalized proxies/ICGD."""
from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import torch
from torch import nn

from rtd.hcw import HomogeneityAwareConsistencyWeighting
from rtd.thermal import TemperatureNormalization, ThermalRadiativeExtractor, validate_thermal_map

from .discriminator import PatchDiscriminator


class ThermalConsistencyResult(NamedTuple):
    loss: torch.Tensor
    generated: torch.Tensor
    reference: torch.Tensor
    weights: torch.Tensor


class ThermalAuxiliaryConsistency(nn.Module):
    def __init__(
        self, normalization: TemperatureNormalization, extractor: ThermalRadiativeExtractor,
        *, weighting: str = "hcw", hcw: Mapping | None = None,
    ):
        super().__init__()
        if weighting not in {"hcw", "uniform"}:
            raise ValueError("Thermal consistency weighting must be hcw or uniform")
        self.normalization, self.extractor, self.weighting = normalization, extractor, weighting
        self.hcw = HomogeneityAwareConsistencyWeighting(normalization, **dict(hcw or {}))
        if list(self.parameters()):
            raise ValueError("Thermal auxiliary branch must contain no trainable parameters")

    def forward(self, generated: torch.Tensor, reference: torch.Tensor, context: Mapping):
        validate_thermal_map(generated, "generated thermal image")
        validate_thermal_map(reference, "reference thermal image")
        if generated.shape != reference.shape:
            raise ValueError("Generated and reference thermal shapes must match")
        with torch.no_grad():
            reference_kelvin = self.normalization(reference)
            reference_representation = self.extractor(reference_kelvin, context)
            weights = self.hcw(reference_kelvin) if self.weighting == "hcw" else torch.ones_like(reference)
        generated_representation = self.extractor(self.normalization(generated), context)
        discrepancy = weights * (generated_representation - reference_representation).abs()
        loss = discrepancy.flatten(1).mean(dim=1).mean()
        return ThermalConsistencyResult(loss, generated_representation, reference_representation, weights)


class ThermalRadiativeTransferGuidedDiscriminator(nn.Module):
    def __init__(self, auxiliary: ThermalAuxiliaryConsistency, *, base_channels: int = 64):
        super().__init__()
        self.visual_discriminator = PatchDiscriminator(base_channels=base_channels)
        self.auxiliary = auxiliary

    def forward(self, visible: torch.Tensor, target: torch.Tensor):
        return self.visual_discriminator(visible, target)

    def radiative_consistency_loss(self, generated, reference, context):
        return self.auxiliary(generated, reference, context)


def build_thermal_auxiliary(config: Mapping) -> nn.Module:
    mode = config.get("mode", "thermal")
    if mode == "intensity_proxy":
        from .intensity_rtd import IntensityProxyConsistency
        return IntensityProxyConsistency(config)
    if mode != "thermal":
        raise ValueError("Auxiliary mode must be thermal or intensity_proxy")
    normalization = config.get("normalization", {})
    response = config.get("sensor_response", {})
    if any(normalization.get(name) is None for name in ("minimum_kelvin", "maximum_kelvin", "source_split")):
        raise ValueError("Supply fixed train-split temperature normalization statistics")
    if not response.get("wavelength_um") or not response.get("response"):
        raise ValueError("Supply the selected sensor's measured spectral response")
    return ThermalAuxiliaryConsistency(
        TemperatureNormalization(**normalization),
        ThermalRadiativeExtractor(**response),
        weighting=config.get("weighting", "hcw"), hcw=config.get("hcw"),
    )
