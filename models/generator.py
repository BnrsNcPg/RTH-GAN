from __future__ import annotations

import torch
from torch import nn


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=not normalize)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        return torch.cat([self.block(x), skip], dim=1)


class UNetGenerator(nn.Module):
    """Pix2Pix U-Net mapping normalized RGB images to one-channel IR."""

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        base_channels=64,
        *,
        bottleneck_fusion: nn.Module | None = None,
    ):
        super().__init__()
        c = base_channels
        self.down1 = DownBlock(in_channels, c, normalize=False)
        self.down2 = DownBlock(c, c * 2)
        self.down3 = DownBlock(c * 2, c * 4)
        self.down4 = DownBlock(c * 4, c * 8)
        self.down5 = DownBlock(c * 8, c * 8)
        self.down6 = DownBlock(c * 8, c * 8)
        self.down7 = DownBlock(c * 8, c * 8)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c * 8, c * 8, 4, 2, 1), nn.ReLU(inplace=True)
        )
        self.bottleneck_fusion = bottleneck_fusion or nn.Identity()
        self.up1 = UpBlock(c * 8, c * 8, dropout=0.5)
        self.up2 = UpBlock(c * 16, c * 8, dropout=0.5)
        self.up3 = UpBlock(c * 16, c * 8, dropout=0.5)
        self.up4 = UpBlock(c * 16, c * 8)
        self.up5 = UpBlock(c * 16, c * 4)
        self.up6 = UpBlock(c * 8, c * 2)
        self.up7 = UpBlock(c * 4, c)
        self.output = nn.Sequential(nn.ConvTranspose2d(c * 2, out_channels, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        b = self.bottleneck_fusion(self.bottleneck(d7))
        u1 = self.up1(b, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.output(u7)


def initialize_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)
