import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF


def get_high_frequency_pixel(img, kernel_size=5, sigma=1.0):
    """
    通过高斯滤波提取像素空间的高频分量 (对应论文公式 5 中的 x_h 和 y_h)
    高频分量 = 原始图像 - 低频分量(高斯模糊结果)
    """
    # 获取低频分量 (平滑后的图像)
    img_l = TF.gaussian_blur(img, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    # 原始图像减去低频分量，得到高频细节
    img_h = img - img_l
    return img_h


def apply_high_frequency_mask_fft(img, radius=15):
    """
    通过二维离散傅里叶变换 (DFT) 并在频谱空间应用高频掩码 (对应论文公式 6, 7, 8)
    """
    # 1. 执行 DFT 并将低频移到频谱中心
    # img shape: [B, C, H, W]
    # Orthonormal FFT keeps the loss scale comparable across image resolutions.
    fft_signal = torch.fft.fft2(img, dim=(-2, -1), norm="ortho")
    fft_shifted = torch.fft.fftshift(fft_signal, dim=(-2, -1))

    b, c, h, w = img.size()
    center_y, center_x = h // 2, w // 2

    # 2. 生成高频掩码 M_H (公式 8)
    # 中心区域(距离中心小于 radius)被遮挡为 0，外部保留为 1
    # 【已修改】：移除了 indexing='ij' 以兼容旧版 PyTorch
    Y, X = torch.meshgrid(
        torch.arange(h),
        torch.arange(w),
        indexing="ij",
    )
    Y = Y.to(img.device)
    X = X.to(img.device)

    dist_sq = (X - center_x) ** 2 + (Y - center_y) ** 2
    # 距离平方大于等于半径平方的区域设为 1
    mask = (dist_sq >= radius ** 2).float()

    # 扩展维度以匹配 [B, C, H, W]
    mask = mask.unsqueeze(0).unsqueeze(0).expand(b, c, h, w)

    # 3. 将掩码应用到频谱信号上 (公式 7)
    fft_high_freq = fft_shifted * mask

    return fft_high_freq


# ==========================================
# 以下是新增的 Visualizer 依赖的图像处理与保存函数
# ==========================================

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def tensor2im(input_image, imtype=np.uint8):
    """"将 tensor 转换为 numpy 图像数组"""
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # 取 batch 的第一张
        if image_numpy.shape[0] == 1:  # 如果是单通道灰度图
            image_numpy = np.tile(image_numpy, (3, 1, 1))

        # 将数据从 [-1, 1] 映射回 [0, 255]
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    """保存 numpy 图像到硬盘"""
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
