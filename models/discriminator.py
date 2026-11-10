from __future__ import annotations

import torch
from torch import nn


class PatchDiscriminator(nn.Module):
    """Conditional 70x70 PatchGAN over RGB visible + one-channel IR."""

    def __init__(self, visible_channels=3, infrared_channels=1, base_channels=64):
        super().__init__()
        channels = visible_channels + infrared_channels

        def block(in_channels, out_channels, stride=2, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=not normalize)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        c = base_channels
        self.model = nn.Sequential(
            *block(channels, c, normalize=False),
            *block(c, c * 2),
            *block(c * 2, c * 4),
            *block(c * 4, c * 8, stride=1),
            nn.Conv2d(c * 8, 1, 4, 1, 1),
        )

    def forward(self, visible, infrared):
        return self.model(torch.cat([visible, infrared], dim=1))

