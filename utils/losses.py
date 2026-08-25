import torch
import torch.nn as nn
import torch.nn.functional as F  # 【核心修复】：导入 F 模块
from utils.tools import get_high_frequency_pixel, apply_high_frequency_mask_fft
from utils.metrics import batch_ssim, rgb_to_gray


class HingeLoss(nn.Module):
    """
    Hinge Loss (合页损失)
    配合 Spectral Norm 使用的黄金搭档，极其稳定。
    """

    def __init__(self):
        super(HingeLoss, self).__init__()

    # 生成器的损失 (希望骗过判别器，即 D 的输出越大越好)
    def get_G_loss(self, pred_fake):
        return -torch.mean(pred_fake)

    # 判别器的损失 (希望真图得分 >=1，假图得分 <= -1)
    def get_D_loss(self, pred_real, pred_fake):
        # 这里的 F.relu 现在已经可以正常使用了
        loss_real = torch.mean(F.relu(1.0 - pred_real))
        loss_fake = torch.mean(F.relu(1.0 + pred_fake))
        return (loss_real + loss_fake) * 0.5


class PSCLoss(nn.Module):
    """
    像素-频谱多空间约束特征细化损失 (PSC)
    """

    def __init__(self, blur_kernel=5, blur_sigma=1.0, fft_radius=15):
        super(PSCLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.fft_radius = fft_radius

    def forward(self, real_img, fake_img):
        # 1. 像素空间高频损失
        real_h = get_high_frequency_pixel(real_img, self.blur_kernel, self.blur_sigma)
        fake_h = get_high_frequency_pixel(fake_img, self.blur_kernel, self.blur_sigma)
        l_pix = self.l1_loss(real_h, fake_h)

        # 2. 傅里叶频谱空间高频损失
        real_fft_h = apply_high_frequency_mask_fft(real_img, self.fft_radius)
        fake_fft_h = apply_high_frequency_mask_fft(fake_img, self.fft_radius)

        # 【优化】：使用 .abs() 前确保不触发复数强转警告
        # 只有在输入是复数时才取模
        if torch.is_complex(real_fft_h):
            real_mag = torch.abs(real_fft_h)
            fake_mag = torch.abs(fake_fft_h)
        else:
            real_mag = real_fft_h
            fake_mag = fake_fft_h

        l_dfft = self.l1_loss(real_mag, fake_mag)

        return l_pix, l_dfft


class DomainConversionLoss(nn.Module):
    """
    域转换损失 (L_dom)
    """
    def __init__(self):
        super(DomainConversionLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, img_real, img_fake):
        return self.l1(img_real, img_fake)


class MultiScaleSSIMLoss(nn.Module):
    def __init__(self, scales=3):
        super().__init__()
        self.scales = scales

    def forward(self, prediction, target):
        losses = []
        current_prediction = prediction
        current_target = target
        for _ in range(self.scales):
            losses.append(1.0 - batch_ssim(current_prediction, current_target).mean())
            if min(current_prediction.shape[-2:]) < 22:
                break
            current_prediction = F.avg_pool2d(current_prediction, kernel_size=2)
            current_target = F.avg_pool2d(current_target, kernel_size=2)
        return torch.stack(losses).mean()


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3) / 8.0
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, prediction, target):
        prediction_gray = rgb_to_gray(prediction)
        target_gray = rgb_to_gray(target)
        prediction_x = F.conv2d(prediction_gray, self.sobel_x, padding=1)
        prediction_y = F.conv2d(prediction_gray, self.sobel_y, padding=1)
        target_x = F.conv2d(target_gray, self.sobel_x, padding=1)
        target_y = F.conv2d(target_gray, self.sobel_y, padding=1)
        return F.l1_loss(prediction_x, target_x) + F.l1_loss(
            prediction_y, target_y
        )


class TotalVariationLoss(nn.Module):
    def forward(self, image):
        horizontal = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
        vertical = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
        return horizontal + vertical
