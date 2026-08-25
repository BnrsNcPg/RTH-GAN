import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    基础卷积块：Conv2d -> InstanceNorm -> ReLU
    根据论文 Fig.3 中的 Conv/Deconv block 设计
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_norm=True, use_relu=True):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not use_norm)
        ]

        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels))

        if use_relu:
            layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):
    """
    基础反卷积（上采样）块：ConvTranspose2d -> InstanceNorm -> ReLU
    根据论文 Fig.3 中的 Conv/Deconv block 设计
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, use_norm=True,
                 use_relu=True):
        super(DeconvBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, output_padding=output_padding, bias=not use_norm)
        ]

        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels))

        if use_relu:
            layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    残差块：常用于编码器和解码器之间的特征传递，保持特征不丢失。
    包含两个 3x3 卷积层和实例归一化。
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)



#我改进后的模型
import torch
import torch.nn as nn

class GRN(nn.Module):
    """
    全局响应归一化层 (Global Response Normalization)
    ConvNeXt V2 的核心组件，用于增强通道间的特征竞争
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        # x shape: (B, H, W, C)
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNeXtV2Block(nn.Module):
    """
    标准的 ConvNeXt V2 特征提取块
    """
    def __init__(self, dim):
        super().__init__()
        # Depthwise 卷积 (通常使用 7x7 大核)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        # Pointwise 卷积 (通道数放大 4 倍)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        # Pointwise 卷积 (通道数恢复)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        # LayerNorm 和 Linear 层在 PyTorch 中默认对最后一个维度操作
        # 所以需要把通道维度放到最后: (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        # 恢复维度: (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = input + x
        return x

class ConvNeXtV2DownBlock(nn.Module):
    """
    带下采样的 ConvNeXt v2 模块 (替换原 ConvBlock)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=1):
        super().__init__()
        # 1. 下采样层 (改变分辨率和通道数)
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        # 2. 特征提取层
        self.convnext = ConvNeXtV2Block(dim=out_channels)

    def forward(self, x):
        x = self.downsample(x)
        x = self.convnext(x)
        return x

class ConvNeXtV2UpBlock(nn.Module):
    """
    带上采样的 ConvNeXt v2 模块 (替换原 DeconvBlock)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1, use_convnext=True):
        super().__init__()
        # 1. 上采样层
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                                           stride=stride, padding=padding, output_padding=output_padding)
        # 2. 特征提取层 (网络最后一层不使用，以保证输出映射空间正常)
        self.use_convnext = use_convnext
        if self.use_convnext:
            self.convnext = ConvNeXtV2Block(dim=out_channels)

    def forward(self, x):
        x = self.upsample(x)
        if self.use_convnext:
            x = self.convnext(x)
        return x