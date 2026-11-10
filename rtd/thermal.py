"""Thermal auxiliary operator from manuscript Equations (19)-(23).

Wavelengths are in micrometres; all radiances are W m^-2 sr^-1 um^-1.
These quantities must come from the selected sensor and paired context.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import nn


class TemperatureNormalization(nn.Module):
    """Fixed training-subset affine normalization between Kelvin and [-1, 1]."""

    def __init__(self, minimum_kelvin: float, maximum_kelvin: float, *, source_split: str):
        super().__init__()
        if source_split != "train":
            raise ValueError("Temperature normalization must be fitted on the train split")
        if not (math.isfinite(minimum_kelvin) and math.isfinite(maximum_kelvin)):
            raise ValueError("Temperature normalization bounds must be finite")
        if not 0 < minimum_kelvin < maximum_kelvin:
            raise ValueError("Require 0 < minimum_kelvin < maximum_kelvin")
        self.register_buffer("minimum_kelvin", torch.tensor(minimum_kelvin, dtype=torch.float64))
        self.register_buffer("span_kelvin", torch.tensor(maximum_kelvin - minimum_kelvin, dtype=torch.float64))

    def to_unit_interval(self, kelvin: torch.Tensor) -> torch.Tensor:
        # Do not clip evaluation values to training extrema.
        return (kelvin - self.minimum_kelvin.to(kelvin)) / self.span_kelvin.to(kelvin)

    def normalize(self, kelvin: torch.Tensor) -> torch.Tensor:
        return self.to_unit_interval(kelvin) * 2 - 1

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return (image + 1) * 0.5 * self.span_kelvin.to(image) + self.minimum_kelvin.to(image)


def validate_thermal_map(value: torch.Tensor, name: str) -> None:
    if value.ndim != 4 or value.shape[1] != 1 or min(value.shape) < 1:
        raise ValueError(f"{name} must have nonempty shape B x 1 x H x W")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


class ThermalRadiativeExtractor(nn.Module):
    """Frozen sensor-band Planck operator; gradients flow only through T.

    Context fields are scalar or broadcastable BCHW tensors. A per-sample
    scalar must have shape Bx1x1x1, not B, to avoid accidental axis alignment.
    """

    CONTEXT_FIELDS = (
        "emissivity", "transmittance", "downwelling_radiance", "upwelling_radiance"
    )

    def __init__(self, wavelength_um: Sequence[float], response: Sequence[float]):
        super().__init__()
        wavelength = torch.as_tensor(wavelength_um, dtype=torch.float64).detach().clone()
        weights = torch.as_tensor(response, dtype=torch.float64).detach().clone()
        if wavelength.ndim != 1 or wavelength.numel() < 2 or weights.shape != wavelength.shape:
            raise ValueError("Sensor response requires matching 1D arrays with at least two samples")
        if not torch.isfinite(wavelength).all() or not torch.isfinite(weights).all():
            raise ValueError("Sensor response must be finite")
        if torch.any(wavelength <= 0) or torch.any(wavelength.diff() <= 0):
            raise ValueError("Wavelengths must be positive and strictly increasing")
        if torch.any(weights < 0) or torch.trapezoid(weights, wavelength) <= 0:
            raise ValueError("Sensor response must be nonnegative with positive integral")
        self.register_buffer("wavelength_um", wavelength)
        self.register_buffer("response", weights)

    def blackbody_radiance(self, kelvin: torch.Tensor) -> torch.Tensor:
        validate_thermal_map(kelvin, "temperature")
        if torch.any(kelvin <= 0):
            raise ValueError("Temperature must be strictly positive Kelvin")
        # Float64 avoids exponent/power overflow in SI Planck evaluation and
        # remains differentiable when the network itself uses float32.
        temperature = kelvin.double().unsqueeze(-1)
        wavelength_um = self.wavelength_um.to(device=kelvin.device)
        wavelength_m = wavelength_um * 1e-6
        h, speed_of_light, boltzmann = 6.62607015e-34, 299792458.0, 1.380649e-23
        exponent = h * speed_of_light / (wavelength_m * boltzmann * temperature)
        # exp(-x)/(1-exp(-x)) equals 1/expm1(x) without overflow for cold T.
        inverse_expm1 = torch.exp(-exponent) / (-torch.expm1(-exponent))
        spectral = (2 * h * speed_of_light**2 / wavelength_m.pow(5)) * inverse_expm1 * 1e-6
        response = self.response.to(device=kelvin.device)
        result = torch.trapezoid(spectral * response, wavelength_um, dim=-1)
        result = result / torch.trapezoid(response, wavelength_um)
        return result.to(dtype=kelvin.dtype)

    @staticmethod
    def _context_value(value, name: str, temperature: torch.Tensor) -> torch.Tensor:
        fixed = torch.as_tensor(value, device=temperature.device, dtype=temperature.dtype).detach()
        if fixed.ndim not in (0, 4):
            raise ValueError(f"{name} must be scalar or broadcastable BCHW")
        try:
            if torch.broadcast_shapes(fixed.shape, temperature.shape) != temperature.shape:
                raise ValueError(f"{name} must broadcast to the temperature shape")
        except RuntimeError as exc:
            raise ValueError(f"{name} must broadcast to the temperature shape") from exc
        if not torch.isfinite(fixed).all() or torch.any(fixed < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
        if name in {"emissivity", "transmittance"} and torch.any(fixed > 1):
            raise ValueError(f"{name} must be within [0, 1]")
        return fixed

    def forward(self, kelvin: torch.Tensor, context: Mapping) -> torch.Tensor:
        if set(context) != set(self.CONTEXT_FIELDS):
            raise ValueError(f"Thermal context must contain exactly {self.CONTEXT_FIELDS}")
        blackbody = self.blackbody_radiance(kelvin)
        emissivity, tau, down, up = (
            self._context_value(context[name], name, kelvin) for name in self.CONTEXT_FIELDS
        )
        return tau * (emissivity * blackbody + (1 - emissivity) * down) + up
