"""Reference-only HCW, Equations (24)-(28) and Appendix A."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .thermal import TemperatureNormalization, validate_thermal_map


class HomogeneityAwareConsistencyWeighting(nn.Module):
    DIRECTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))

    def __init__(
        self, normalization: TemperatureNormalization, *, k: int = 3,
        window_size: int = 5, protection_size: int = 3,
        minimum_weight: float = 0.2, power: float = 1.0, epsilon: float = 1e-6,
    ):
        super().__init__()
        for name, value in (("window_size", window_size), ("protection_size", protection_size)):
            if type(value) is not int or value < 1 or value % 2 != 1:
                raise ValueError(f"{name} must be a positive odd integer")
        if type(k) is not int or not 1 <= k <= 8:
            raise ValueError("k must be an integer in [1, 8]")
        if not 0 < minimum_weight < 1:
            raise ValueError("minimum_weight must be in (0, 1)")
        if not math.isfinite(power) or power <= 0 or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("power and epsilon must be finite and positive")
        self.normalization = normalization
        self.k, self.window_size, self.protection_size = k, window_size, protection_size
        self.minimum_weight, self.power, self.epsilon = minimum_weight, power, epsilon

    def directional_statistics(self, normalized: torch.Tensor):
        validate_thermal_map(normalized, "HCW thermal representation")
        # Double precision preserves low-variance comparisons and deterministic
        # ties even when inputs have a large common temperature offset.
        field = normalized.detach().double()
        height, width = field.shape[-2:]
        half = self.window_size // 2
        offset = (self.protection_size + self.window_size) // 2
        radius = offset + half
        padded = F.pad(field, (radius,) * 4, mode="replicate")
        means, variances = [], []
        for row_direction, column_direction in self.DIRECTIONS:
            top = radius + offset * row_direction - half
            left = radius + offset * column_direction - half
            region = padded[..., top:top + height + 2 * half, left:left + width + 2 * half]
            windows = region.unfold(2, self.window_size, 1).unfold(3, self.window_size, 1)
            variance, mean = torch.var_mean(windows, dim=(-1, -2), unbiased=False)
            variances.append(variance)
            means.append(mean)
        return torch.cat(means, dim=1), torch.cat(variances, dim=1)

    @torch.no_grad()
    def forward(self, reference_kelvin: torch.Tensor, *, return_details: bool = False):
        validate_thermal_map(reference_kelvin, "reference temperature")
        normalized = self.normalization.to_unit_interval(reference_kelvin.double())
        means, variances = self.directional_statistics(normalized)
        # stable sorting implements the Appendix A direction-order tie break.
        selected = torch.argsort(variances, dim=1, stable=True)[:, :self.k]
        background = torch.gather(means, 1, selected).mean(dim=1, keepdim=True)
        deviation = (normalized - background).abs()
        mean_deviation = deviation.mean(dim=(-2, -1), keepdim=True) + self.epsilon
        scaled = deviation / (deviation + mean_deviation)
        weights = self.minimum_weight + (1 - self.minimum_weight) * scaled.pow(self.power)
        weights = weights.to(dtype=reference_kelvin.dtype)
        if return_details:
            return weights, {"background": background, "directions": selected, "deviation": deviation}
        return weights
