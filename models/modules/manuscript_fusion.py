"""Foreground/background geodesic TA-DSF, manuscript Eq. (12)-(18)."""
from __future__ import annotations

import math

import torch
from torch import nn

from .euclidean_hyperbolic_fusion_v2 import mobius_add
from .euclidean_hyperbolic_fusion import project_to_ball


def _project(value, curvature):
    # A representable open-ball margin also keeps opposite-point Mobius
    # denominators away from cancellation, including for float32 callers.
    return project_to_ball(value, curvature, eps=max(1e-5, 4 * torch.finfo(value.dtype).eps))


def _mobius(left, right, curvature):
    return _project(mobius_add(_project(left, curvature), _project(right, curvature),
                              curvature, eps=1e-12), curvature)


def _radial_ratio(value, inverse=False):
    tiny = value.abs() < 1e-5
    if inverse:
        margin = max(1e-5, 4 * torch.finfo(value.dtype).eps)
        regular = torch.atanh(value.clamp(max=1-margin)) / value.clamp_min(1e-12)
        series = 1 + value.square() / 3
    else:
        regular = torch.tanh(value) / value.clamp_min(1e-12)
        series = 1 - value.square() / 3
    return torch.where(tiny, series, regular)


def origin_exp(value, curvature):
    radius = math.sqrt(curvature) * torch.linalg.vector_norm(value, dim=1, keepdim=True)
    return _project(value * _radial_ratio(radius), curvature)


def origin_log(value, curvature):
    value = _project(value, curvature)
    radius = math.sqrt(curvature) * torch.linalg.vector_norm(value, dim=1, keepdim=True)
    return value * _radial_ratio(radius, inverse=True)


def poincare_distance(left, right, curvature):
    displacement = _mobius(-left, right, curvature)
    radius = math.sqrt(curvature) * torch.linalg.vector_norm(displacement, dim=1, keepdim=True)
    margin = max(1e-5, 4 * torch.finfo(radius.dtype).eps)
    return 2 / math.sqrt(curvature) * torch.atanh(radius.clamp(max=1-margin))


def geodesic_aggregate(background, foreground, coefficient, curvature):
    displacement = _mobius(-background, foreground, curvature)
    scaled = origin_exp(coefficient * origin_log(displacement, curvature), curvature)
    return _mobius(background, scaled, curvature)


class GeodesicDualSpaceFusion(nn.Module):
    def __init__(self, channels, *, curvature=1.0, tangent_bound=0.5, variant="full"):
        super().__init__()
        if not math.isfinite(curvature) or not curvature > 0 or not math.isfinite(tangent_bound) or not tangent_bound > 0:
            raise ValueError("Curvature and tangent bound must be finite and positive")
        if variant not in {"full", "mean", "local_only", "fixed_mean"}:
            raise ValueError("Fusion variant must be full, mean, local_only, or fixed_mean")
        self.variant = variant
        self.curvature, self.tangent_bound = float(curvature), float(tangent_bound)
        if variant != "mean":
            self.euclidean = nn.Sequential(nn.Conv2d(2 * channels, channels, 3, padding=1), nn.GELU())
        if variant in {"full", "fixed_mean"}:
            self.interpolation = nn.Sequential(nn.Conv2d(3, max(4, channels // 4), 1), nn.GELU(), nn.Conv2d(max(4, channels // 4), 1, 1))
        if variant == "full":
            self.gate = nn.Conv2d(2 * channels, channels, 1)

    def _stable_tangent(self, value):
        norm = torch.linalg.vector_norm(value, dim=1, keepdim=True)
        return value * _radial_ratio(norm / self.tangent_bound)

    def forward(self, background, foreground, *, return_details=False):
        if background.shape != foreground.shape or background.ndim != 4:
            raise ValueError("Foreground/background fields must have matching BCHW shapes")
        if self.variant == "mean":
            fused = (background + foreground) * 0.5
            return (fused, {}) if return_details else fused
        euclidean = self.euclidean(torch.cat([background, foreground], dim=1))
        if self.variant == "local_only":
            return (euclidean, {"local": euclidean}) if return_details else euclidean
        # Keep near-boundary Mobius arithmetic out of mixed-precision kernels.
        with torch.autocast(device_type=background.device.type, enabled=False):
            bg = origin_exp(self._stable_tangent(background.double()), self.curvature)
            fg = origin_exp(self._stable_tangent(foreground.double()), self.curvature)
            zero = torch.zeros_like(bg)
            descriptors = torch.cat([poincare_distance(bg, fg, self.curvature),
                poincare_distance(zero, bg, self.curvature), poincare_distance(zero, fg, self.curvature)], dim=1)
            # Autocast is disabled here, so use the convolution's parameter
            # dtype rather than a possibly half/bfloat16 feature dtype.
            coefficient = torch.sigmoid(self.interpolation(descriptors.to(self.interpolation[0].weight.dtype)))
            point = geodesic_aggregate(bg, fg, coefficient.double(), self.curvature)
            hyperbolic = origin_log(point, self.curvature).to(background.dtype)
        details = {"alpha": coefficient, "hyperbolic_point": point,
                   "local": euclidean, "radial": hyperbolic}
        if self.variant == "fixed_mean":
            fused = (euclidean + hyperbolic) * 0.5
        else:
            gate = torch.sigmoid(self.gate(torch.cat([euclidean, hyperbolic], dim=1)))
            fused = gate * euclidean + (1 - gate) * hyperbolic
            details["gate"] = gate
        if return_details:
            return fused, details
        return fused
