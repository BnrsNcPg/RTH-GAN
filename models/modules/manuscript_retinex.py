"""Manuscript Eq. (3)-(7), preserving the historical PINN formulations."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .pinn_retinex import (
    PINNRetinex, PINNRetinexConfig, PINNLosses, PositionalEncoding, RetinexPriors,
    _dense_coordinates, _sample_map,
    build_intensity_map, compute_boundary_normal_derivative,
    compute_retinex_pde_residual, sample_boundary_coordinates, sample_interior_coordinates,
)


class ManuscriptRetinex(PINNRetinex):
    def __init__(self, config=None):
        settings = {"epsilon_w": 1e-4, "formulation_mode": "corrected", **dict(config or {})}
        if settings["formulation_mode"] != "corrected":
            raise ValueError("Manuscript PINN requires the pixel-coordinate total-derivative formulation")
        super().__init__(PINNRetinexConfig.from_mapping(settings))
        # A positive Python float can still underflow/overflow when used by the
        # FP32 prior path. Exclude subnormals, which accelerators may flush to 0.
        limits = torch.finfo(torch.float32)
        scales = {}
        for name in ("epsilon_r", "reflectance_max"):
            value = float(getattr(self.config, name))
            rounded = torch.tensor(value, dtype=torch.float32).item()
            if not math.isfinite(value) or not math.isfinite(rounded) or rounded < limits.tiny:
                raise ValueError(f"{name} must be finite and positive in the normal FP32 range")
            scales[name] = rounded
        # Division backward may form X / D**2 even when clipping supplies a
        # zero gradient. Keep that intermediate finite instead of risking 0*Inf.
        if scales["epsilon_r"] < math.sqrt(limits.tiny):
            raise ValueError("epsilon_r is too small for FP32 division backward")
        if 1.0 / (scales["epsilon_r"] * scales["reflectance_max"]) > limits.max:
            raise ValueError("epsilon_r and reflectance_max cause FP32 reciprocal overflow")
        encoding = PositionalEncoding(self.config.positional_frequencies)
        layers = []
        width = encoding.output_channels + 1
        for _ in range(self.config.hidden_layers):
            layers.extend([nn.Linear(width, self.config.hidden_channels), nn.Tanh()])
            width = self.config.hidden_channels
        output = nn.Linear(width, 1)
        nn.init.constant_(output.bias, self.config.l_max / 2)
        layers.append(output)
        self.illumination_network = _UnconstrainedIllumination(encoding, nn.Sequential(*layers))

    def forward(self, visible: torch.Tensor) -> RetinexPriors:
        """Build fixed-scale priors without constraining the raw linear L field."""
        self._validate_visible(visible)
        if not torch.isfinite(visible).all():
            raise ValueError("Visible input must be finite")
        with torch.autocast(device_type=visible.device.type, enabled=False):
            visible_fp32 = visible.float()
            batch_size, _, height, width = visible_fp32.shape
            intensity = build_intensity_map(visible_fp32)
            coordinates = _dense_coordinates(batch_size, height, width, visible_fp32.device)
            illumination = self.illumination_network(coordinates, _sample_map(intensity, coordinates))
            illumination = illumination.float().transpose(1, 2).reshape(batch_size, 1, height, width)
            if not torch.isfinite(illumination).all():
                raise FloatingPointError("Manuscript illumination must be finite before division")
            # Preserve raw L for its encoder and soft range penalty. Only the
            # reflectance denominator is protected, including every negative L.
            denominator = (illumination + self.config.epsilon_r).clamp_min(self.config.epsilon_r)
            if not torch.isfinite(denominator).all() or not (denominator > 0).all():
                raise FloatingPointError("Manuscript reflectance denominator must be finite and positive")
            reflectance = visible_fp32 / denominator
            # Clamping can hide Inf; check the quotient before saturation.
            if not torch.isfinite(reflectance).all():
                raise FloatingPointError("Manuscript reflectance quotient must be finite before clipping")
            reflectance = reflectance.clamp(0.0, self.config.reflectance_max) / self.config.reflectance_max
        return RetinexPriors(illumination, reflectance, intensity)

    def loss_terms(self, visible):
        self._validate_visible(visible)
        if not torch.isfinite(visible).all():
            raise ValueError("Visible input must be finite")
        with torch.autocast(device_type=visible.device.type, enabled=False):
            intensity = build_intensity_map(visible.float()).detach()
            coordinates = sample_interior_coordinates(
                len(visible), self.config.num_interior_points, device=visible.device
            ).requires_grad_(True)
            # The reusable residual takes (|grad S|^2)^beta. Eq. (2) uses
            # |grad S|^beta, hence the explicit exponent conversion here.
            terms = compute_retinex_pde_residual(
                self.illumination_network, intensity, coordinates,
                lambda_s=self.config.lambda_s, beta=self.config.beta / 2,
                epsilon_w=self.config.epsilon_w, formulation_mode="corrected",
            )
            points, normals = sample_boundary_coordinates(
                len(visible), self.config.num_boundary_points, device=visible.device
            )
            boundary = compute_boundary_normal_derivative(
                self.illumination_network, intensity, points.requires_grad_(True), normals,
                formulation_mode="corrected",
            )
            pde = terms.residual.square().mean() if self.config.use_pde else terms.illumination.new_zeros(())
            bc = boundary.normal_derivative.square().mean()
            tv = terms.spatial_gradient.abs().sum(-1).mean()
            # Eq. (6) uses unsquared ReLU penalties on the interior samples.
            value = (F.relu(-terms.illumination) + F.relu(terms.illumination - self.config.l_max)).mean()
            total = self.config.lambda_pde * pde + self.config.lambda_bc * bc + self.config.lambda_tv * tv + self.config.lambda_val * value
        return PINNLosses(total, pde, bc, tv, value)


class _UnconstrainedIllumination(nn.Module):
    def __init__(self, encoding, network):
        super().__init__()
        self.encoding, self.network = encoding, network

    def forward(self, coordinates, intensity):
        return self.network(torch.cat([self.encoding(coordinates), intensity], dim=-1))
