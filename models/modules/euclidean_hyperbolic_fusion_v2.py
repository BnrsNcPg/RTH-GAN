from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from .euclidean_hyperbolic_fusion import expmap0, logmap0, project_to_ball


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"non-finite Stage 3B tensor: {name}")


def _as_positive_curvature(
    curvature: torch.Tensor | float,
    reference: torch.Tensor,
) -> torch.Tensor:
    value = torch.as_tensor(curvature, dtype=reference.dtype, device=reference.device)
    if not torch.isfinite(value).all() or torch.any(value <= 0):
        raise FloatingPointError("Poincare curvature must be finite and positive")
    return value


def radial_tangent_squash(
    value: torch.Tensor,
    tau: float,
    *,
    dim: int = 1,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Smoothly bound each spatial latent vector while preserving direction."""
    if not torch.isfinite(value).all():
        raise FloatingPointError("radial squash input must be finite")
    if not tau > 0.0 or not eps > 0.0:
        raise ValueError("tau and eps must be positive")
    norm = torch.linalg.vector_norm(value, dim=dim, keepdim=True)
    scale = float(tau) * torch.tanh(norm / float(tau)) / (norm + float(eps))
    squashed = value * scale
    _require_finite("radial_tangent_squash", squashed)
    return squashed


def mobius_add(
    left: torch.Tensor,
    right: torch.Tensor,
    curvature: torch.Tensor | float,
    *,
    dim: int = 1,
    eps: float = 1.0e-8,
    return_denominator: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Möbius addition in the Poincare ball with curvature magnitude ``c``."""
    if left.shape != right.shape:
        raise ValueError("Mobius addition requires tensors with identical shapes")
    _require_finite("mobius_add_left", left)
    _require_finite("mobius_add_right", right)
    c = _as_positive_curvature(curvature, left)
    left_sq = left.square().sum(dim=dim, keepdim=True)
    right_sq = right.square().sum(dim=dim, keepdim=True)
    inner = (left * right).sum(dim=dim, keepdim=True)
    denominator = 1.0 + 2.0 * c * inner + c.square() * left_sq * right_sq
    _require_finite("mobius_add_denominator", denominator)
    if torch.any(denominator <= 0):
        raise FloatingPointError("Mobius addition encountered a non-positive denominator")
    numerator = (
        (1.0 + 2.0 * c * inner + c * right_sq) * left
        + (1.0 - c * left_sq) * right
    )
    result = numerator / denominator.clamp_min(float(eps))
    _require_finite("mobius_add", result)
    if return_denominator:
        return result, denominator
    return result


def mobius_matvec(
    transform: nn.Module,
    point: torch.Tensor,
    curvature: torch.Tensor | float,
    *,
    dim: int = 1,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    """Apply a Euclidean linear map in the origin tangent space."""
    c = _as_positive_curvature(curvature, point)
    tangent = transform(logmap0(point, c, dim=dim, eps=eps))
    _require_finite("mobius_matvec_tangent", tangent)
    result = expmap0(tangent, c, dim=dim, eps=eps)
    _require_finite("mobius_matvec", result)
    return result


def euclidean_residual(transform: nn.Module, tangent: torch.Tensor) -> torch.Tensor:
    result = tangent + transform(tangent)
    _require_finite("euclidean_residual", result)
    return result


def hyperbolic_residual(
    transform: nn.Module,
    tangent: torch.Tensor,
    curvature: torch.Tensor | float,
    *,
    dim: int = 1,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    """Non-canceling tangent -> ball residual -> tangent formulation."""
    c = _as_positive_curvature(curvature, tangent)
    point = expmap0(tangent, c, dim=dim, eps=eps)
    delta = mobius_matvec(transform, point, c, dim=dim, eps=eps)
    combined = mobius_add(point, delta, c, dim=dim, eps=eps)
    combined = project_to_ball(combined, c, dim=dim, eps=eps)
    result = logmap0(combined, c, dim=dim, eps=eps)
    _require_finite("hyperbolic_residual", result)
    return result


def _expmap0_with_projection_events(
    tangent: torch.Tensor,
    curvature: torch.Tensor,
    *,
    dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    sqrt_c = torch.sqrt(curvature)
    norm = torch.linalg.vector_norm(tangent, dim=dim, keepdim=True)
    raw = torch.tanh(sqrt_c * norm) * tangent / (sqrt_c * norm).clamp_min(eps)
    maximum_norm = (1.0 - eps) / sqrt_c
    raw_norm = torch.linalg.vector_norm(raw, dim=dim, keepdim=True)
    events = raw_norm > maximum_norm
    projected = project_to_ball(raw, curvature, dim=dim, eps=eps)
    return projected, events


@dataclass(frozen=True)
class FusionV2Config:
    latent_channels: int = 128
    tau: float = 0.5
    curvature: float = 1.0
    ball_eps: float = 1.0e-5
    norm_eps: float = 1.0e-8
    residual_alpha_init: float = 0.1
    enabled: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping | None) -> "FusionV2Config":
        values = dict(values or {})
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - known - {"variant", "smoke"})
        if unknown:
            raise ValueError(f"Unknown Stage 3B fusion configuration keys: {unknown}")
        config = cls(**{key: value for key, value in values.items() if key in known})
        config.validate()
        return config

    def validate(self) -> None:
        if self.latent_channels < 1:
            raise ValueError("latent_channels must be positive")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")
        if self.curvature <= 0.0:
            raise ValueError("curvature must be positive")
        if not 0.0 < self.ball_eps < 1.0:
            raise ValueError("ball_eps must be in (0, 1)")
        if self.norm_eps <= 0.0:
            raise ValueError("norm_eps must be positive")
        if self.residual_alpha_init < 0.0:
            raise ValueError("residual_alpha_init must be non-negative")


class _TwoBranchFusionV2(nn.Module):
    stage = "stage3b"
    variant = "base_v2"

    def __init__(self, channels: int, config: FusionV2Config | Mapping | None = None):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.channels = int(channels)
        self.config = config if isinstance(config, FusionV2Config) else FusionV2Config.from_mapping(config)
        width = self.config.latent_channels
        self.euclidean_projection = nn.Conv2d(channels, width, 1, bias=True)
        self.geometry_projection = nn.Conv2d(channels, width, 1, bias=True)
        self.euclidean_transform = nn.Conv2d(width, width, 1, bias=True)
        self.geometry_transform = nn.Conv2d(width, width, 1, bias=True)
        self.output_projection = nn.Conv2d(width * 2, channels, 1, bias=True)
        self.gate_projection = nn.Conv2d(width * 2, channels, 1, bias=True)
        self.activation = nn.SiLU(inplace=False)
        self.residual_alpha = nn.Parameter(torch.tensor(float(self.config.residual_alpha_init)))
        self.register_buffer(
            "fixed_curvature",
            torch.tensor(float(self.config.curvature)),
            persistent=True,
        )
        self.residual_enabled = bool(self.config.enabled)
        self.last_diagnostics: dict[str, float] = {}

    @property
    def curvature(self) -> torch.Tensor:
        value = self.fixed_curvature
        if not torch.isfinite(value).all() or torch.any(value <= 0):
            raise FloatingPointError("Stage 3B fixed curvature must remain finite and positive")
        return value

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def set_residual_enabled(self, enabled: bool) -> None:
        self.residual_enabled = bool(enabled)

    @staticmethod
    def _vector_norm(value: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(value.float(), dim=1)

    def _stabilized_latent(
        self,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw = self.activation(self.geometry_projection(value))
        stable = radial_tangent_squash(
            raw,
            self.config.tau,
            eps=self.config.norm_eps,
        )
        raw_norm = self._vector_norm(raw)
        stable_norm = self._vector_norm(stable)
        return stable, {
            "mean_pre_squash_latent_norm": raw_norm.mean(),
            "mean_post_squash_tangent_norm": stable_norm.mean(),
            "max_post_squash_tangent_norm": stable_norm.max(),
        }

    def _geometry_branch(self, value: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raise NotImplementedError

    def _record_diagnostics(
        self,
        source: torch.Tensor,
        common: torch.Tensor,
        geometry: torch.Tensor,
        gate: torch.Tensor,
        output: torch.Tensor,
        geometry_diagnostics: dict[str, torch.Tensor],
    ) -> None:
        residual_norm = torch.linalg.vector_norm((output - source).float().flatten(1), dim=1)
        source_norm = torch.linalg.vector_norm(source.float().flatten(1), dim=1)
        values = {
            "euclidean_common_branch_norm": self._vector_norm(common).mean(),
            "geometry_output_norm": self._vector_norm(geometry).mean(),
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
        common = self.euclidean_transform(self.activation(self.euclidean_projection(value)))
        geometry, geometry_diagnostics = self._geometry_branch(value)
        _require_finite("euclidean_common_branch", common)
        _require_finite("geometry_branch", geometry)
        joined = torch.cat((common, geometry), dim=1)
        mixed = self.output_projection(joined)
        gate = torch.sigmoid(self.gate_projection(joined))
        output = value + self.residual_alpha * gate * mixed
        _require_finite("fusion_output", output)
        self._record_diagnostics(value, common, geometry, gate, output, geometry_diagnostics)
        return output


class EuclideanControlFusionV2(_TwoBranchFusionV2):
    """Radially stabilized capacity control with an ordinary residual."""

    variant = "euclidean_control_v2"

    def _geometry_branch(self, value: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        tangent, diagnostics = self._stabilized_latent(value)
        output = euclidean_residual(self.geometry_transform, tangent)
        zero = output.new_zeros(())
        one = output.new_ones(())
        return output, {
            **diagnostics,
            "curvature": zero,
            "tau": output.new_tensor(float(self.config.tau)),
            "mean_poincare_norm": zero,
            "max_poincare_norm": zero,
            "mean_ball_radius_utilization": zero,
            "max_ball_radius_utilization": zero,
            "boundary_occupancy_percent": zero,
            "projection_event_percent": zero,
            "mobius_denominator_min": one,
        }


class EuclideanHyperbolicFusionV2(_TwoBranchFusionV2):
    """Radially stabilized, non-canceling Poincare residual formulation."""

    variant = "euclidean_hyperbolic_v2"

    def _geometry_branch(self, value: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        tangent, diagnostics = self._stabilized_latent(value)
        curvature = self.curvature.to(dtype=tangent.dtype, device=tangent.device)
        point, point_events = _expmap0_with_projection_events(
            tangent,
            curvature,
            dim=1,
            eps=self.config.ball_eps,
        )
        transformed_tangent = self.geometry_transform(
            logmap0(point, curvature, eps=self.config.ball_eps)
        )
        delta, delta_events = _expmap0_with_projection_events(
            transformed_tangent,
            curvature,
            dim=1,
            eps=self.config.ball_eps,
        )
        combined, denominator = mobius_add(
            point,
            delta,
            curvature,
            eps=self.config.norm_eps,
            return_denominator=True,
        )
        sqrt_c = torch.sqrt(curvature)
        maximum_norm = (1.0 - self.config.ball_eps) / sqrt_c
        combined_norm = torch.linalg.vector_norm(combined, dim=1, keepdim=True)
        combined_events = combined_norm > maximum_norm
        projected = project_to_ball(
            combined,
            curvature,
            eps=self.config.ball_eps,
        )
        output = logmap0(projected, curvature, eps=self.config.ball_eps)
        point_norm = self._vector_norm(projected)
        safety_radius = (1.0 - self.config.ball_eps) / torch.sqrt(curvature.float())
        boundary = point_norm >= safety_radius * (1.0 - 1.0e-3)
        projection_events = point_events | delta_events | combined_events
        return output, {
            **diagnostics,
            "curvature": curvature.float(),
            "tau": output.new_tensor(float(self.config.tau)),
            "mean_poincare_norm": point_norm.mean(),
            "max_poincare_norm": point_norm.max(),
            "mean_ball_radius_utilization": (torch.sqrt(curvature.float()) * point_norm).mean(),
            "max_ball_radius_utilization": (torch.sqrt(curvature.float()) * point_norm).max(),
            "boundary_occupancy_percent": boundary.float().mean() * 100.0,
            "projection_event_percent": projection_events.float().mean() * 100.0,
            "mobius_denominator_min": denominator.float().min(),
        }


def build_stage3b_fusion(
    channels: int,
    config: Mapping | FusionV2Config | None,
) -> _TwoBranchFusionV2 | None:
    values = dict(config or {}) if not isinstance(config, FusionV2Config) else {}
    if isinstance(config, FusionV2Config):
        variant = "euclidean_hyperbolic_v2"
        fusion_config = config
    else:
        variant = str(values.get("variant", "none"))
        fusion_config = FusionV2Config.from_mapping(values) if variant != "none" else None
    if variant == "none":
        return None
    if variant == "euclidean_control_v2":
        return EuclideanControlFusionV2(channels, fusion_config)
    if variant == "euclidean_hyperbolic_v2":
        return EuclideanHyperbolicFusionV2(channels, fusion_config)
    raise ValueError(f"Unknown Stage 3B fusion variant: {variant}")
