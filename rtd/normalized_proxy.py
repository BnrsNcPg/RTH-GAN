from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from .context import RTDContext, load_rtd_context


class NormalizedProxyRadiativeExtractor(nn.Module):
    """Frozen dimensionless auxiliary map for the revised RTD graph.

    For fixed context ``C`` the implemented band form is::

        L_k = tau_k * rho_hat / pi * (E_sun_k + E_sky_k) + L_path_k
        Phi(I; C) = sum_k response_weight_k * L_k * delta_lambda_k

    It is therefore exactly ``A(C) * rho_hat(I) + B(C)``, where::

        A(C) = sum_k w_k * delta_k * tau_k * (E_sun_k + E_sky_k) / pi
        B(C) = sum_k w_k * delta_k * L_path_k

    The current repository target contract is one channel in ``[-1, 1]``.
    Coefficients are registered buffers, never trainable parameters.
    """

    def __init__(self, context: RTDContext):
        super().__init__()
        if context.mode != "normalized_proxy":
            raise ValueError("NormalizedProxyRadiativeExtractor requires normalized_proxy mode")
        self.context_id = context.context_id
        self.eps = float(context.eps)
        self.register_buffer("tau", torch.tensor(context.tau, dtype=torch.float64))
        self.register_buffer("E_sun", torch.tensor(context.E_sun, dtype=torch.float64))
        self.register_buffer("E_sky", torch.tensor(context.E_sky, dtype=torch.float64))
        self.register_buffer("L_path", torch.tensor(context.L_path, dtype=torch.float64))
        self.register_buffer(
            "response_weight", torch.tensor(context.response_weight, dtype=torch.float64)
        )
        self.register_buffer(
            "delta_lambda", torch.tensor(context.delta_lambda, dtype=torch.float64)
        )

    @classmethod
    def from_context_file(cls, path: str | Path) -> "NormalizedProxyRadiativeExtractor":
        return cls(load_rtd_context(path))

    def proxy_reflectance(self, target: torch.Tensor) -> torch.Tensor:
        if target.ndim != 4 or target.shape[1] != 1:
            raise ValueError("RTD target must have shape Bx1xHxW")
        if not target.is_floating_point():
            raise TypeError("RTD target must be a floating-point tensor")
        target_01 = (target + 1.0) / 2.0
        return target_01.clamp(min=self.eps, max=1.0)

    def linear_coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        scale = (
            self.response_weight
            * self.delta_lambda
            * self.tau
            * (self.E_sun + self.E_sky)
            / math.pi
        ).sum()
        offset = (self.response_weight * self.delta_lambda * self.L_path).sum()
        return scale, offset

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        rho_hat = self.proxy_reflectance(target)
        coefficient_dtype = target.dtype
        coefficient_device = target.device
        shape = (1, -1, 1, 1, 1)
        rho_k = rho_hat.unsqueeze(1)
        tau = self.tau.to(device=coefficient_device, dtype=coefficient_dtype).view(shape)
        illumination = (self.E_sun + self.E_sky).to(
            device=coefficient_device, dtype=coefficient_dtype
        ).view(shape)
        path_offset = self.L_path.to(
            device=coefficient_device, dtype=coefficient_dtype
        ).view(shape)
        band_representation = tau * rho_k / math.pi * illumination + path_offset
        integration = (self.response_weight * self.delta_lambda).to(
            device=coefficient_device, dtype=coefficient_dtype
        ).view(shape)
        return (integration * band_representation).sum(dim=1)
