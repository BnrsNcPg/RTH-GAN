from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping, NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PINNRetinexConfig:
    """Initial engineering defaults for the small shared coordinate PINN."""

    positional_frequencies: int = 4
    hidden_channels: int = 32
    hidden_layers: int = 3
    l_max: float = 1.0
    lambda_s: float = 0.001
    beta: float = 0.5
    epsilon_w: float = 0.01
    epsilon_r: float = 0.001
    reflectance_max: float = 4.0
    num_interior_points: int = 1024
    num_boundary_points: int = 256
    lambda_pde: float = 1.0
    lambda_bc: float = 0.1
    lambda_tv: float = 0.01
    lambda_val: float = 1.0
    use_pde: bool = True
    formulation_mode: str = "legacy"

    @classmethod
    def from_mapping(cls, values: Mapping | None) -> "PINNRetinexConfig":
        values = dict(values or {})
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - known - {"enabled", "lambda_total"})
        if unknown:
            raise ValueError(f"Unknown PINN configuration keys: {unknown}")
        return cls(**{key: value for key, value in values.items() if key in known})

    def validate(self) -> None:
        if self.positional_frequencies < 1:
            raise ValueError("positional_frequencies must be positive")
        if self.hidden_channels < 1 or self.hidden_layers < 1:
            raise ValueError("PINN hidden size and depth must be positive")
        if not 0.0 < self.l_max <= 1.0:
            raise ValueError("l_max must be in (0, 1]")
        if self.lambda_s < 0.0 or self.beta <= 0.0:
            raise ValueError("lambda_s must be non-negative and beta must be positive")
        if self.epsilon_w <= 0.0 or self.epsilon_r <= 0.0:
            raise ValueError("epsilon_w and epsilon_r must be positive")
        if self.reflectance_max <= 0.0:
            raise ValueError("reflectance_max must be positive")
        if self.num_interior_points < 1 or self.num_boundary_points < 4:
            raise ValueError("PINN sampling requires interior points and at least four boundary points")
        for name in ("lambda_pde", "lambda_bc", "lambda_tv", "lambda_val"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.formulation_mode not in {"legacy", "corrected"}:
            raise ValueError("formulation_mode must be legacy or corrected")


class RetinexPriors(NamedTuple):
    illumination: torch.Tensor
    reflectance: torch.Tensor
    intensity: torch.Tensor


class PINNLosses(NamedTuple):
    total: torch.Tensor
    pde: torch.Tensor
    boundary: torch.Tensor
    tv: torch.Tensor
    value: torch.Tensor


class PDEResidualTerms(NamedTuple):
    """Sampled quantities produced by the authoritative PDE implementation."""

    illumination: torch.Tensor
    intensity: torch.Tensor
    spatial_gradient: torch.Tensor
    divergence: torch.Tensor
    residual: torch.Tensor


class BoundaryDerivativeTerms(NamedTuple):
    illumination: torch.Tensor
    intensity: torch.Tensor
    spatial_gradient: torch.Tensor
    normal_derivative: torch.Tensor


def build_intensity_map(visible: torch.Tensor) -> torch.Tensor:
    """Return S(p)=max_k X_k(p) for an RGB tensor in [0, 1]."""

    if visible.ndim != 4 or visible.shape[1] != 3:
        raise ValueError("visible must have shape B x 3 x H x W")
    if not torch.is_floating_point(visible):
        raise TypeError("visible must be a floating-point tensor")
    return visible.amax(dim=1, keepdim=True)


def sample_interior_coordinates(
    batch_size: int,
    num_points: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample coordinates from the fixed open domain (-1, 1)^2."""

    margin = 1.0e-4
    return torch.rand(batch_size, num_points, 2, device=device, dtype=dtype).mul_(2.0 - 2.0 * margin).add_(-1.0 + margin)


def sample_boundary_coordinates(
    batch_size: int,
    num_points: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the four square edges and return their outward unit normals."""

    side = torch.arange(num_points, device=device).remainder(4).view(1, -1).expand(batch_size, -1)
    tangent = torch.rand(batch_size, num_points, device=device, dtype=dtype).mul_(2.0).sub_(1.0)
    minus_one = torch.full_like(tangent, -1.0)
    plus_one = torch.ones_like(tangent)
    zero = torch.zeros_like(tangent)
    x = torch.where(side == 0, minus_one, torch.where(side == 1, plus_one, tangent))
    y = torch.where(side == 2, minus_one, torch.where(side == 3, plus_one, tangent))
    nx = torch.where(side == 0, minus_one, torch.where(side == 1, plus_one, zero))
    ny = torch.where(side == 2, minus_one, torch.where(side == 3, plus_one, zero))
    return torch.stack((x, y), dim=-1), torch.stack((nx, ny), dim=-1)


def _dense_coordinates(batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1).reshape(1, height * width, 2).expand(batch_size, -1, -1)


def _sample_map(image: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    grid = coordinates.unsqueeze(2)
    sampled = F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(-1).transpose(1, 2)


def differentiable_bilinear_sample(
    image: torch.Tensor,
    coordinates: torch.Tensor,
) -> torch.Tensor:
    """Sample at normalized coordinates with weights supporting double backward.

    The image is a fixed coefficient field. Coordinate-dependent interpolation
    weights intentionally remain in the graph so corrected mode includes the
    S(p) input chain when differentiating the illumination network.
    """

    if image.ndim != 4 or coordinates.ndim != 3 or coordinates.shape[-1] != 2:
        raise ValueError("Expected image BxCxHxW and coordinates BxNx2")
    batch_size, channels, height, width = image.shape
    if coordinates.shape[0] != batch_size:
        raise ValueError("Image and coordinate batch sizes differ")
    x = ((coordinates[..., 0] + 1.0) * 0.5 * (width - 1)).clamp(0.0, width - 1)
    y = ((coordinates[..., 1] + 1.0) * 0.5 * (height - 1)).clamp(0.0, height - 1)
    # At the upper edges use the last *interior* cell (weight == 1).
    # Collapsing x0/x1 or y0/y1 at the endpoint erases dS/dn on two
    # Neumann boundaries, although the sampled values still look correct.
    x0 = x.floor().long().clamp(max=max(width - 2, 0))
    y0 = y.floor().long().clamp(max=max(height - 2, 0))
    x1 = (x0 + 1).clamp(max=width - 1)
    y1 = (y0 + 1).clamp(max=height - 1)
    wx = (x - x0.to(x.dtype)).unsqueeze(-1)
    wy = (y - y0.to(y.dtype)).unsqueeze(-1)
    flat = image.flatten(2).transpose(1, 2)

    def gather(ix: torch.Tensor, iy: torch.Tensor) -> torch.Tensor:
        offsets = (iy * width + ix).unsqueeze(-1).expand(-1, -1, channels)
        return torch.gather(flat, 1, offsets)

    top = gather(x0, y0) * (1.0 - wx) + gather(x1, y0) * wx
    bottom = gather(x0, y1) * (1.0 - wx) + gather(x1, y1) * wx
    return top * (1.0 - wy) + bottom * wy


def _edge_weight_map(intensity: torch.Tensor, beta: float, epsilon_w: float) -> torch.Tensor:
    dx = F.pad(intensity[..., 1:] - intensity[..., :-1], (0, 1, 0, 0))
    dy = F.pad(intensity[..., 1:, :] - intensity[..., :-1, :], (0, 0, 0, 1))
    gradient_squared = dx.square() + dy.square()
    return (gradient_squared.pow(beta) + epsilon_w).reciprocal()


def _fixed_spatial_gradient(field: torch.Tensor) -> torch.Tensor:
    """Finite-difference a fixed coefficient; PINN solution derivatives use autograd."""

    dx = F.pad(field[..., 1:] - field[..., :-1], (0, 1, 0, 0))
    dy = F.pad(field[..., 1:, :] - field[..., :-1, :], (0, 0, 0, 1))
    return torch.cat((dx, dy), dim=1)


def _coordinate_derivatives(
    illumination: torch.Tensor,
    coordinates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gradient = torch.autograd.grad(
        illumination.sum(), coordinates, create_graph=True, retain_graph=True
    )[0]
    def second(component: torch.Tensor, axis: int) -> torch.Tensor:
        if not component.requires_grad:
            return torch.zeros_like(illumination)
        derivative = torch.autograd.grad(
            component.sum(),
            coordinates,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if derivative is None:
            return torch.zeros_like(illumination)
        return derivative[..., axis : axis + 1]

    second_x = second(gradient[..., 0], 0)
    second_y = second(gradient[..., 1], 1)
    return gradient, second_x, second_y


def compute_retinex_pde_residual(
    illumination_network: nn.Module,
    intensity_map: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    lambda_s: float,
    beta: float,
    epsilon_w: float,
    formulation_mode: str,
) -> PDEResidualTerms:
    """Compute the declared weighted Retinex PDE in one selected convention.

    ``legacy`` exactly preserves the Stage 2 implementation: S(p) is detached
    after grid sampling and normalized-coordinate L derivatives are combined
    with pixel-step finite differences of w.

    ``corrected`` uses the Stage 2C pixel-total definition. S(p) is sampled by
    a double-backward-safe bilinear operator without detaching its coordinate
    path. L derivatives are converted to pixel coordinates before combining
    them with one-pixel finite differences of w and grad(w).
    """

    if formulation_mode not in {"legacy", "corrected"}:
        raise ValueError("formulation_mode must be legacy or corrected")
    if intensity_map.ndim != 4 or intensity_map.shape[1] != 1:
        raise ValueError("intensity_map must have shape B x 1 x H x W")
    if not coordinates.requires_grad:
        raise ValueError("coordinates must require gradients")

    fixed_intensity = intensity_map.detach()
    weight_map = _edge_weight_map(fixed_intensity, beta=beta, epsilon_w=epsilon_w).detach()
    weight_gradient_map = _fixed_spatial_gradient(weight_map).detach()

    if formulation_mode == "legacy":
        intensity = _sample_map(fixed_intensity, coordinates).detach()
        illumination = illumination_network(coordinates, intensity)
        gradient, second_x, second_y = _coordinate_derivatives(illumination, coordinates)
        weight = _sample_map(weight_map, coordinates).detach()
        weight_gradient = _sample_map(weight_gradient_map, coordinates).detach()
        spatial_gradient = gradient
        laplacian = second_x + second_y
    else:
        intensity = differentiable_bilinear_sample(fixed_intensity, coordinates)
        illumination = illumination_network(coordinates, intensity)
        gradient, second_x, second_y = _coordinate_derivatives(illumination, coordinates)
        height, width = fixed_intensity.shape[-2:]
        if height < 2 or width < 2:
            raise ValueError("corrected pixel-coordinate PDE requires H,W >= 2")
        scale_x = 2.0 / (width - 1)
        scale_y = 2.0 / (height - 1)
        scale = gradient.new_tensor((scale_x, scale_y))
        spatial_gradient = gradient * scale
        laplacian = second_x * (scale_x**2) + second_y * (scale_y**2)
        weight = differentiable_bilinear_sample(weight_map, coordinates).detach()
        weight_gradient = differentiable_bilinear_sample(
            weight_gradient_map, coordinates
        ).detach()

    divergence = (
        weight * laplacian
        + weight_gradient[..., 0:1] * spatial_gradient[..., 0:1]
        + weight_gradient[..., 1:2] * spatial_gradient[..., 1:2]
    )
    residual = -float(lambda_s) * divergence + illumination - intensity
    return PDEResidualTerms(
        illumination, intensity, spatial_gradient, divergence, residual
    )


def compute_boundary_normal_derivative(
    illumination_network: nn.Module,
    intensity_map: torch.Tensor,
    coordinates: torch.Tensor,
    normals: torch.Tensor,
    *,
    formulation_mode: str,
) -> BoundaryDerivativeTerms:
    """Compute the unchanged Neumann BC in the selected spatial convention."""

    if formulation_mode not in {"legacy", "corrected"}:
        raise ValueError("formulation_mode must be legacy or corrected")
    if coordinates.shape != normals.shape or coordinates.shape[-1] != 2:
        raise ValueError("coordinates and normals must have matching B x N x 2 shapes")
    if not coordinates.requires_grad:
        raise ValueError("coordinates must require gradients")
    fixed_intensity = intensity_map.detach()
    if formulation_mode == "legacy":
        intensity = _sample_map(fixed_intensity, coordinates).detach()
    else:
        intensity = differentiable_bilinear_sample(fixed_intensity, coordinates)
    illumination = illumination_network(coordinates, intensity)
    gradient = torch.autograd.grad(
        illumination.sum(), coordinates, create_graph=True, retain_graph=True
    )[0]
    if formulation_mode == "corrected":
        height, width = fixed_intensity.shape[-2:]
        if height < 2 or width < 2:
            raise ValueError("corrected pixel-coordinate BC requires H,W >= 2")
        gradient = gradient * gradient.new_tensor(
            (2.0 / (width - 1), 2.0 / (height - 1))
        )
    normal_derivative = (gradient * normals).sum(dim=-1)
    return BoundaryDerivativeTerms(
        illumination, intensity, gradient, normal_derivative
    )


class PositionalEncoding(nn.Module):
    def __init__(self, num_frequencies: int):
        super().__init__()
        frequencies = torch.pi * (2.0 ** torch.arange(num_frequencies, dtype=torch.float32))
        self.register_buffer("frequencies", frequencies, persistent=True)
        self.output_channels = 2 + 4 * num_frequencies

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        angles = coordinates.unsqueeze(-1) * self.frequencies
        encoded = torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(start_dim=-2)
        return torch.cat((coordinates, encoded), dim=-1)


class CoordinateIlluminationMLP(nn.Module):
    def __init__(self, config: PINNRetinexConfig):
        super().__init__()
        self.encoding = PositionalEncoding(config.positional_frequencies)
        layers: list[nn.Module] = []
        input_channels = self.encoding.output_channels + 1
        for layer_index in range(config.hidden_layers):
            layers.extend((
                nn.Linear(input_channels if layer_index == 0 else config.hidden_channels, config.hidden_channels),
                nn.SiLU(),
            ))
        layers.append(nn.Linear(config.hidden_channels, 1))
        self.network = nn.Sequential(*layers)
        self.l_max = float(config.l_max)

    def forward(self, coordinates: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
        features = torch.cat((self.encoding(coordinates), intensity), dim=-1)
        return self.network(features).sigmoid() * self.l_max


class PINNRetinex(nn.Module):
    """Small shared coordinate PINN for Retinex prior generation."""

    def __init__(self, config: PINNRetinexConfig | Mapping | None = None):
        super().__init__()
        self.config = config if isinstance(config, PINNRetinexConfig) else PINNRetinexConfig.from_mapping(config)
        self.config.validate()
        self.illumination_network = CoordinateIlluminationMLP(self.config)
        self.last_residual_diagnostics: dict[str, float] = {}

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _validate_visible(visible: torch.Tensor) -> None:
        if visible.ndim != 4 or visible.shape[1] != 3:
            raise ValueError("visible must have shape B x 3 x H x W")
        if not torch.is_floating_point(visible):
            raise TypeError("visible must be floating point")
        if torch.any(visible < 0.0) or torch.any(visible > 1.0):
            raise ValueError("PINNRetinex expects visible values in [0, 1]")

    def forward(self, visible: torch.Tensor) -> RetinexPriors:
        self._validate_visible(visible)
        visible_fp32 = visible.float()
        batch_size, _, height, width = visible_fp32.shape
        intensity = build_intensity_map(visible_fp32)
        coordinates = _dense_coordinates(batch_size, height, width, visible_fp32.device)
        intensity_points = _sample_map(intensity, coordinates)
        illumination = self.illumination_network(coordinates, intensity_points)
        illumination = illumination.transpose(1, 2).reshape(batch_size, 1, height, width)
        reflectance = visible_fp32 / (illumination + self.config.epsilon_r)
        reflectance = reflectance.clamp(0.0, self.config.reflectance_max) / self.config.reflectance_max
        return RetinexPriors(illumination, reflectance, intensity)

    def loss_terms(self, visible: torch.Tensor) -> PINNLosses:
        """Compute sampled PINN terms; spatial derivatives remain float32."""

        self._validate_visible(visible)
        device_type = visible.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            visible_fp32 = visible.float()
            intensity_map = build_intensity_map(visible_fp32).detach()
            batch_size = visible_fp32.shape[0]
            interior = sample_interior_coordinates(
                batch_size,
                self.config.num_interior_points,
                device=visible_fp32.device,
            ).requires_grad_(True)
            pde_terms = compute_retinex_pde_residual(
                self.illumination_network,
                intensity_map,
                interior,
                lambda_s=self.config.lambda_s,
                beta=self.config.beta,
                epsilon_w=self.config.epsilon_w,
                formulation_mode=self.config.formulation_mode,
            )
            illumination = pde_terms.illumination
            gradient = pde_terms.spatial_gradient
            if self.config.use_pde and self.config.lambda_pde > 0.0:
                pde = pde_terms.residual.square().mean()
                residual_values = pde_terms.residual.detach().float()
                self.last_residual_diagnostics = {
                    "pde_residual_mean": float(residual_values.mean()),
                    "pde_residual_std": float(residual_values.std(unbiased=False)),
                    "pde_residual_abs_p95": float(
                        torch.quantile(residual_values.abs(), 0.95)
                    ),
                }
            else:
                pde = illumination.new_zeros(())
                self.last_residual_diagnostics = {
                    "pde_residual_mean": 0.0,
                    "pde_residual_std": 0.0,
                    "pde_residual_abs_p95": 0.0,
                }

            boundary_coordinates, normals = sample_boundary_coordinates(
                batch_size,
                self.config.num_boundary_points,
                device=visible_fp32.device,
            )
            boundary_coordinates.requires_grad_(True)
            boundary_terms = compute_boundary_normal_derivative(
                self.illumination_network,
                intensity_map,
                boundary_coordinates,
                normals,
                formulation_mode=self.config.formulation_mode,
            )
            boundary_illumination = boundary_terms.illumination
            boundary = boundary_terms.normal_derivative.square().mean()
            tv = gradient.abs().sum(dim=-1).mean()
            values = torch.cat((illumination, boundary_illumination), dim=1)
            value = (
                F.relu(-values).square().mean()
                + F.relu(values - self.config.l_max).square().mean()
            )
            total = (
                self.config.lambda_pde * pde
                + self.config.lambda_bc * boundary
                + self.config.lambda_tv * tv
                + self.config.lambda_val * value
            )
        return PINNLosses(total, pde, boundary, tv, value)
