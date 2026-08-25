import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    # 限制通道数防过拟合
    def __init__(self, input_channels, hidden_channels=32, num_layers=3):
        super(PatchGANDiscriminator, self).__init__()
        layers = [
            nn.Conv2d(input_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        in_c = hidden_channels
        for i in range(1, num_layers):
            out_c = hidden_channels * (2 ** i)
            layers += [
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            in_c = out_c
        out_c = hidden_channels * (2 ** num_layers)
        layers += [
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        layers += [nn.Conv2d(out_c, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class PhysicsInformedDiscriminator(nn.Module):
    r"""联合判别的物理判别器 D(x, \eta)"""
    def __init__(self, image_channels=3, radiance_channels=1, pretrained_path=None):
        super(PhysicsInformedDiscriminator, self).__init__()
        # 输入通道 = 3(原图) + 1(物理图) = 4
        self.discriminator = PatchGANDiscriminator(input_channels=image_channels + radiance_channels)

    def forward(self, x, radiance_map):
        physics_informed_input = torch.cat([x, radiance_map], dim=1)
        return self.discriminator(physics_informed_input)
