"""Explicit prospective manuscript ablations, separate from historical Pix2Pix runs."""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .generator import UNetGenerator
from .modules.manuscript_retinex import ManuscriptRetinex
from .rghsg import RetinexGuidedHyperbolicGenerator


class ManuscriptUNetAblation(UNetGenerator):
    """Eight-downsample N backbone with optional frozen manuscript PINN guidance.

    Both variants consume RGB in [0, 1]. No legacy input-range conversion is
    applied. Guidance projections remain trainable after PINN pretraining.
    """

    def __init__(self, *, with_pinn=False, base_channels=64, guidance_scale=0.1, pinn=None):
        if type(base_channels) is not int or base_channels < 1:
            raise ValueError("base_channels must be a positive integer")
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and nonnegative")
        super().__init__(in_channels=3, out_channels=1, base_channels=base_channels)
        self.guidance_scale = float(guidance_scale)
        self.retinex = ManuscriptRetinex(pinn) if with_pinn else None
        if self.retinex is not None:
            self.illumination_projection = nn.Conv2d(1, base_channels, 3, padding=1, bias=False)
            self.reflectance_projection = nn.Conv2d(3, base_channels, 3, padding=1, bias=False)
            nn.init.zeros_(self.illumination_projection.weight)
            nn.init.zeros_(self.reflectance_projection.weight)
        self.freeze_retinex()

    def freeze_retinex(self):
        if self.retinex is not None:
            self.retinex.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        if self.retinex is not None:
            self.retinex.eval()
        return self

    def _guided_backbone(self, visible, priors):
        d1 = self.down1(visible)
        size = d1.shape[-2:]
        illumination = F.interpolate(priors.illumination, size=size, mode="bilinear", align_corners=False)
        reflectance = F.interpolate(priors.reflectance, size=size, mode="bilinear", align_corners=False)
        d1 = d1 + self.guidance_scale * (self.illumination_projection(illumination)
                                        + self.reflectance_projection(reflectance))
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        bottleneck = self.bottleneck(d7)
        u1 = self.up1(bottleneck, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        return self.output(self.up7(u6, d1))

    def forward(self, visible, *, return_details=False):
        if visible.ndim != 4 or visible.shape[1] != 3 or any(
            size < 256 or size % 256 for size in visible.shape[-2:]
        ):
            raise ValueError("Manuscript U-Net requires Bx3xHxW with H/W positive multiples of 256")
        if not torch.isfinite(visible).all() or (visible < 0).any() or (visible > 1).any():
            raise ValueError("Manuscript U-Net visible input must be finite in [0, 1]")
        if self.retinex is None:
            generated, details = super().forward(visible), {}
        else:
            with torch.no_grad():
                priors = self.retinex(visible)
            generated = self._guided_backbone(visible, priors)
            details = {"illumination": priors.illumination, "reflectance": priors.reflectance}
        return (generated, details) if return_details else generated


def build_manuscript_generator(model_config: Mapping):
    """Build from config['model']; omitted variant preserves the full RGG path."""
    if model_config.get("name", "rthgan_manuscript") != "rthgan_manuscript":
        raise ValueError("Manuscript generator requires model.name=rthgan_manuscript")
    variant = model_config.get("variant", "rgg")
    generator_config = dict(model_config.get("generator", {}))
    if variant == "rgg":
        return RetinexGuidedHyperbolicGenerator(**generator_config)
    if variant in {"n", "n_pinn"}:
        baseline_config = dict(model_config.get("baseline_generator", {}))
        return ManuscriptUNetAblation(with_pinn=variant == "n_pinn",
            pinn=generator_config.get("pinn"), **baseline_config)
    raise ValueError("Manuscript model.variant must be rgg, n, or n_pinn")
