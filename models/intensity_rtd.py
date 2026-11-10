"""Explicit dimensionless RTD proxy + reference-only HCW for JPEG experiments."""
from __future__ import annotations

import torch
from torch import nn

from rtd.context import rtd_context_from_mapping
from rtd.hcw import HomogeneityAwareConsistencyWeighting
from rtd.normalized_proxy import NormalizedProxyRadiativeExtractor
from rtd.thermal import validate_thermal_map
from .thermal_rtd import ThermalConsistencyResult


class IntensityNormalization(nn.Module):
    """Map normalized JPEG intensities to [0,1], without a Kelvin interpretation."""

    def to_unit_interval(self, image):
        return (image + 1) / 2


class IntensityProxyConsistency(nn.Module):
    CONTROLS = {"hcw", "uniform", "clipping_matched_l1", "visual_only"}

    def __init__(self, config):
        super().__init__()
        if config.get("physical_interpretation") is not False:
            raise ValueError("JPEG intensity proxy must explicitly disable physical interpretation")
        # Preserve old HCW/uniform configurations and checkpoint buffer names.
        # An explicit control must not silently reinterpret a legacy weighting.
        legacy_weighting = config.get("weighting")
        if legacy_weighting is not None and (
            not isinstance(legacy_weighting, str) or legacy_weighting not in {"hcw", "uniform"}
        ):
            raise ValueError("Intensity proxy weighting must be hcw or uniform")
        self.control = config.get("control", legacy_weighting or "hcw")
        if not isinstance(self.control, str) or self.control not in self.CONTROLS:
            raise ValueError(f"Intensity proxy control must be one of {sorted(self.CONTROLS)}")
        expected_weighting = "hcw" if self.control == "hcw" else "uniform"
        if legacy_weighting is not None and legacy_weighting != expected_weighting:
            raise ValueError("Intensity proxy control conflicts with weighting")
        if any(name in config for name in ("lambda_match", "matched_l1_coefficient")):
            raise ValueError("The matched L1 coefficient is derived from the proxy; do not configure it separately")
        context = rtd_context_from_mapping(config["proxy_context"])
        self.extractor = NormalizedProxyRadiativeExtractor(context)
        self.hcw = HomogeneityAwareConsistencyWeighting(IntensityNormalization(), **config.get("hcw", {}))
        self.weighting = expected_weighting

    def matched_l1_coefficient(self):
        """Additional L1 scale; the trainer applies lambda_hac exactly once."""
        scale, _ = self.extractor.linear_coefficients()
        return scale.abs() / 2

    def control_definition(self):
        """JSON-ready fixed control semantics, without changing checkpoint keys."""
        scale, offset = self.extractor.linear_coefficients()
        coefficient = float(self.matched_l1_coefficient())
        return {
            "version": "intensity_controls_v1",
            "control": self.control,
            "physical_interpretation": False,
            "clip_epsilon": self.extractor.eps,
            "clipping_lower_bound": 2 * self.extractor.eps - 1,
            "clipping_upper_bound": 1.0,
            "proxy_scale": float(scale),
            "proxy_offset": float(offset),
            "additional_l1_coefficient_before_lambda_hac": (
                coefficient if self.control in {"uniform", "clipping_matched_l1"}
                else 0.0 if self.control == "visual_only" else None
            ),
            "additional_weighted_l1_coefficient_before_lambda_hac": (
                coefficient if self.control == "hcw" else None
            ),
            "coefficient_source": "abs(proxy_scale)/2",
            "reduction": "mean_pixels_per_image_then_mean_batch",
            "weight_normalization": "pixel_count",
            "reference_gradient": "stopped",
            "lambda_hac_application": "trainer_multiplies_once",
        }

    def forward(self, generated, reference, context):
        if context:
            raise ValueError("JPEG proxy does not accept physical thermal context")
        for name, value in (("generated", generated), ("reference", reference)):
            validate_thermal_map(value, name)
            if (value < -1).any() or (value > 1).any():
                raise ValueError("JPEG proxy requires intensities in [-1,1]")
        if generated.shape != reference.shape:
            raise ValueError("Generated and reference intensity shapes must match")
        with torch.no_grad():
            reference_representation = self.extractor(reference)
            weights = self.hcw(reference) if self.weighting == "hcw" else torch.ones_like(reference)
        representation = self.extractor(generated)
        if self.control == "visual_only":
            weights = torch.zeros_like(reference)
            # A differentiable zero makes standalone gradient checks possible.
            loss = generated.sum() * 0
        elif self.control == "clipping_matched_l1":
            q_generated = 2 * self.extractor.proxy_reflectance(generated) - 1
            with torch.no_grad():
                q_reference = 2 * self.extractor.proxy_reflectance(reference) - 1
            coefficient = self.matched_l1_coefficient().to(generated)
            loss = coefficient * (q_generated - q_reference).abs().flatten(1).mean(1).mean()
        else:
            loss = (weights * (representation - reference_representation).abs()).flatten(1).mean(1).mean()
        return ThermalConsistencyResult(loss, representation, reference_representation, weights)
