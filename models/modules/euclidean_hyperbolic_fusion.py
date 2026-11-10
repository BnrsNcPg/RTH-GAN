from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


def _require_positive_curvature(curvature: torch.Tensor) -> None:
    if not torch.isfinite(curvature).all() or torch.any(curvature <= 0):
        raise FloatingPointError("Poincare curvature must be finite and positive")


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"non-finite Stage 3A tensor: {name}")


def project_to_ball(
    value: torch.Tensor,
    curvature: torch.Tensor | float,
    *,
    dim: int = 1,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    """Project channel vectors into the open Poincare ball of curvature ``-c``."""
    c = torch.as_tensor(curvature, dtype=value.dtype, device=value.device)
    _require_positive_curvature(c)
    if not 0.0 < eps < 1.0:
        raise ValueError("eps must be in (0, 1)")
    maximum_norm = (1.0 - eps) / torch.sqrt(c)
    norm = torch.linalg.vector_norm(value, dim=dim, keepdim=True)
    scale = torch.clamp(maximum_norm / norm.clamp_min(eps), max=1.0)
    projected = value * scale
    _require_finite("project_to_ball", projected)
    return projected


def expmap0(
    tangent: torch.Tensor,
    curvature: torch.Tensor | float,
    *,
    dim: int = 1,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    """Exponential map at the origin for the Poincare ball."""
    c = torch.as_tensor(curvature, dtype=tangent.dtype, device=tangent.device)
    _require_positive_curvature(c)
    sqrt_c = torch.sqrt(c)
    norm = torch.linalg.vector_norm(tangent, dim=dim, keepdim=True)
    mapped = torch.tanh(sqrt_c * norm) * tangent / (sqrt_c * norm).clamp_min(eps)
    mapped = project_to_ball(mapped, c, dim=dim, eps=eps)
    _require_finite("expmap0", mapped)
    return mapped


def logmap0(
    point: torch.Tensor,
    curvature: torch.Tensor | float,
    *,
    dim: int = 1,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    """Logarithmic map at the origin for the Poincare ball."""
    c = torch.as_tensor(curvature, dtype=point.dtype, device=point.device)
    _require_positive_curvature(c)
    safe_point = project_to_ball(point, c, dim=dim, eps=eps)
    sqrt_c = torch.sqrt(c)
    norm = torch.linalg.vector_norm(safe_point, dim=dim, keepdim=True)
    argument = (sqrt_c * norm).clamp(min=0.0, max=1.0 - eps)
    mapped = torch.atanh(argument) * safe_point / (sqrt_c * norm).clamp_min(eps)
    _require_finite("logmap0", mapped)
    return mapped


@dataclass(frozen=True)
class FusionConfig:
    latent_channels: int = 128
    curvature_init: float = 1.0
    learnable_curvature: bool = True
    curvature_min: float = 1.0e-4
    ball_eps: float = 1.0e-5
    residual_alpha_init: float = 0.1
    enabled: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping | None) -> "FusionConfig":
        values = dict(values or {})
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - known - {"variant", "smoke"})
        if unknown:
            raise ValueError(f"Unknown Stage 3A fusion configuration keys: {unknown}")
        config = cls(**{key: value for key, value in values.items() if key in known})
        config.validate()
        return config

    def validate(self) -> None:
        if self.latent_channels < 1:
            raise ValueError("latent_channels must be positive")
        if self.curvature_init <= self.curvature_min or self.curvature_min <= 0.0:
            raise ValueError("curvature_init must exceed a positive curvature_min")
        if not 0.0 < self.ball_eps < 1.0:
            raise ValueError("ball_eps must be in (0, 1)")
        if self.residual_alpha_init < 0.0:
            raise ValueError("residual_alpha_init must be non-negative")


class _TwoBranchFusion(nn.Module):
    variant = "base"

    def __init__(self, channels: int, config: FusionConfig | Mapping | None = None):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.channels = int(channels)
        self.config = config if isinstance(config, FusionConfig) else FusionConfig.from_mapping(config)
        width = self.config.latent_channels
        self.euclidean_projection = nn.Conv2d(channels, width, 1)
        self.geometry_projection = nn.Conv2d(channels, width, 1)
        self.euclidean_transform = nn.Conv2d(width, width, 1)
        self.geometry_transform = nn.Conv2d(width, width, 1)
        self.output_projection = nn.Conv2d(width * 2, channels, 1)
        self.gate_projection = nn.Conv2d(width * 2, channels, 1)
        self.activation = nn.SiLU(inplace=False)
        self.residual_alpha = nn.Parameter(torch.tensor(float(self.config.residual_alpha_init)))
        self.residual_enabled = bool(self.config.enabled)
        self.last_diagnostics: dict[str, float] = {}

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def set_residual_enabled(self, enabled: bool) -> None:
        self.residual_enabled = bool(enabled)

    @staticmethod
    def _mean_vector_norm(value: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(value.float(), dim=1).mean()

    def _geometry_branch(self, value: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raise NotImplementedError

    def _record_diagnostics(
        self,
        source: torch.Tensor,
        euclidean: torch.Tensor,
        geometry: torch.Tensor,
        gate: torch.Tensor,
        output: torch.Tensor,
        geometry_diagnostics: dict[str, torch.Tensor],
    ) -> None:
        residual_norm = torch.linalg.vector_norm((output - source).float().flatten(1), dim=1)
        source_norm = torch.linalg.vector_norm(source.float().flatten(1), dim=1)
        values = {
            "euclidean_feature_norm": self._mean_vector_norm(euclidean),
            "hyperbolic_feature_norm": self._mean_vector_norm(geometry),
            "fusion_gate_mean": gate.float().mean(),
            "fusion_gate_std": gate.float().std(unbiased=False),
            "fusion_gate_min": gate.float().min(),
            "fusion_gate_max": gate.float().max(),
            "fusion_alpha": self.residual_alpha.detach().float(),
            "output_residual_ratio": (residual_norm / (source_norm + 1.0e-12)).mean(),
            **geometry_diagnostics,
        }
        for name, item in values.items():
            _require_finite(name, item)
        self.last_diagnostics = {name: float(item.detach()) for name, item in values.items()}

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != self.channels:
            raise ValueError(f"Expected B x {self.channels} x H x W input")
        _require_finite("fusion_input", value)
        if not self.residual_enabled:
            return value
        euclidean = self.euclidean_transform(self.activation(self.euclidean_projection(value)))
        geometry, geometry_diagnostics = self._geometry_branch(value)
        _require_finite("euclidean_branch", euclidean)
        _require_finite("geometry_branch", geometry)
        joined = torch.cat((euclidean, geometry), dim=1)
        mixed = self.output_projection(joined)
        gate = torch.sigmoid(self.gate_projection(joined))
        output = value + self.residual_alpha * gate * mixed
        _require_finite("fusion_output", output)
        self._record_diagnostics(value, euclidean, geometry, gate, output, geometry_diagnostics)
        return output


class EuclideanControlFusion(_TwoBranchFusion):
    """Capacity-matched two-branch Euclidean control for Stage 3A."""

    variant = "euclidean_control"

    def _geometry_branch(self, value: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        tangent = self.activation(self.geometry_projection(value))
        transformed = self.geometry_transform(tangent)
        zero = transformed.new_zeros(())
        return transformed, {
            "curvature": zero,
            "mean_tangent_feature_norm": self._mean_vector_norm(tangent),
            "mean_poincare_feature_norm": zero,
            "max_poincare_feature_norm": zero,
            "poincare_boundary_percent": zero,
        }


class EuclideanHyperbolicFusion(_TwoBranchFusion):
    """Minimal Euclidean/Poincare two-branch residual fusion."""

    variant = "euclidean_hyperbolic"

    def __init__(self, channels: int, config: FusionConfig | Mapping | None = None):
        super().__init__(channels, config)
        initial = float(self.config.curvature_init - self.config.curvature_min)
        raw = math.log(math.expm1(initial))
        if self.config.learnable_curvature:
            self.curvature_raw = nn.Parameter(torch.tensor(raw))
        else:
            self.register_buffer("curvature_raw", torch.tensor(raw), persistent=True)

    @property
    def curvature(self) -> torch.Tensor:
        value = F.softplus(self.curvature_raw) + self.config.curvature_min
        _require_positive_curvature(value)
        return value

    def _geometry_branch(self, value: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        tangent = self.activation(self.geometry_projection(value))
        curvature = self.curvature.to(dtype=tangent.dtype, device=tangent.device)
        point = expmap0(tangent, curvature, eps=self.config.ball_eps)
        transformed_tangent = self.geometry_transform(
            logmap0(point, curvature, eps=self.config.ball_eps)
        )
        transformed_point = expmap0(transformed_tangent, curvature, eps=self.config.ball_eps)
        transformed_point = project_to_ball(
            transformed_point, curvature, eps=self.config.ball_eps
        )
        output = logmap0(transformed_point, curvature, eps=self.config.ball_eps)
        point_norm = torch.linalg.vector_norm(transformed_point.float(), dim=1)
        radius = (1.0 - self.config.ball_eps) / torch.sqrt(curvature.float())
        boundary = point_norm >= radius * (1.0 - 1.0e-3)
        return output, {
            "curvature": curvature.float(),
            "mean_tangent_feature_norm": self._mean_vector_norm(tangent),
            "mean_poincare_feature_norm": point_norm.mean(),
            "max_poincare_feature_norm": point_norm.max(),
            "poincare_boundary_percent": boundary.float().mean() * 100.0,
        }


def build_stage3a_fusion(
    channels: int,
    config: Mapping | FusionConfig | None,
) -> _TwoBranchFusion | None:
    values = dict(config or {}) if not isinstance(config, FusionConfig) else {}
    if isinstance(config, FusionConfig):
        variant = "euclidean_hyperbolic"
        fusion_config = config
    else:
        variant = str(values.get("variant", "none"))
        fusion_config = FusionConfig.from_mapping(values) if variant != "none" else None
    if variant == "none":
        return None
    if variant == "euclidean_control":
        return EuclideanControlFusion(channels, fusion_config)
    if variant == "euclidean_hyperbolic":
        return EuclideanHyperbolicFusion(channels, fusion_config)
    raise ValueError(f"Unknown Stage 3A fusion variant: {variant}")
