import torch
import torch.nn as nn
import torch.nn.functional as F


MS_SSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def rgb_to_gray(image):
    if image.shape[1] == 1:
        return image
    return (
        0.299 * image[:, 0:1]
        + 0.587 * image[:, 1:2]
        + 0.114 * image[:, 2:3]
    )


def batch_mae(prediction, target):
    return torch.mean(torch.abs(prediction - target), dim=(1, 2, 3))


def batch_rmse(prediction, target, value_scale=255.0):
    """RMSE in the paper's 8-bit pixel-intensity scale."""
    mse = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))
    return torch.sqrt(torch.clamp(mse, min=0.0)) * value_scale


def batch_psnr(prediction, target, data_range=1.0):
    mse = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))
    max_value = torch.tensor(data_range ** 2, device=mse.device, dtype=mse.dtype)
    return 10.0 * torch.log10(max_value / torch.clamp(mse, min=1e-12))


def _gaussian_window(window_size, sigma, channels, device, dtype):
    coordinates = torch.arange(window_size, device=device, dtype=dtype)
    coordinates = coordinates - (window_size - 1) / 2
    gaussian = torch.exp(-(coordinates ** 2) / (2 * sigma ** 2))
    gaussian = gaussian / gaussian.sum()
    window = torch.outer(gaussian, gaussian)
    return window.expand(channels, 1, window_size, window_size).contiguous()


def _ssim_components(
    prediction,
    target,
    data_range=1.0,
    window_size=11,
    sigma=1.5,
):
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if prediction.ndim != 4:
        raise ValueError("SSIM expects tensors shaped [N, C, H, W]")

    effective_size = min(window_size, prediction.shape[-2], prediction.shape[-1])
    if effective_size % 2 == 0:
        effective_size -= 1
    if effective_size < 1:
        raise ValueError("SSIM input spatial dimensions must be positive")

    channels = prediction.shape[1]
    window = _gaussian_window(
        effective_size,
        sigma,
        channels,
        prediction.device,
        prediction.dtype,
    )
    mu_x = F.conv2d(prediction, window, groups=channels)
    mu_y = F.conv2d(target, window, groups=channels)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = (
        F.conv2d(prediction * prediction, window, groups=channels)
        - mu_x_sq
    )
    sigma_y_sq = (
        F.conv2d(target * target, window, groups=channels)
        - mu_y_sq
    )
    sigma_xy = (
        F.conv2d(prediction * target, window, groups=channels)
        - mu_xy
    )

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    luminance = (2.0 * mu_xy + c1) / torch.clamp(
        mu_x_sq + mu_y_sq + c1,
        min=1e-12,
    )
    contrast_structure = (2.0 * sigma_xy + c2) / torch.clamp(
        sigma_x_sq + sigma_y_sq + c2,
        min=1e-12,
    )
    ssim = torch.mean(luminance * contrast_structure, dim=(1, 2, 3))
    contrast_structure = torch.mean(contrast_structure, dim=(1, 2, 3))
    return ssim, contrast_structure


def batch_ssim(prediction, target, data_range=1.0, window_size=11):
    ssim, _ = _ssim_components(
        prediction,
        target,
        data_range=data_range,
        window_size=window_size,
    )
    return ssim


def batch_ms_ssim(
    prediction,
    target,
    data_range=1.0,
    window_size=11,
    weights=MS_SSIM_WEIGHTS,
):
    weights_tensor = prediction.new_tensor(weights)
    levels = []
    current_prediction = prediction
    current_target = target

    for level in range(len(weights)):
        ssim, contrast_structure = _ssim_components(
            current_prediction,
            current_target,
            data_range=data_range,
            window_size=window_size,
        )
        levels.append(ssim if level == len(weights) - 1 else contrast_structure)
        if level < len(weights) - 1:
            current_prediction = F.avg_pool2d(
                current_prediction,
                kernel_size=2,
                ceil_mode=True,
            )
            current_target = F.avg_pool2d(
                current_target,
                kernel_size=2,
                ceil_mode=True,
            )

    values = torch.stack(levels, dim=0).clamp(min=1e-8)
    return torch.prod(values ** weights_tensor[:, None], dim=0)


class InceptionV3Features(nn.Module):
    """ImageNet Inception-v3 pool features used for dataset-level FID."""

    def __init__(self):
        super().__init__()
        from torchvision.models import Inception_V3_Weights, inception_v3

        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights)
        model.fc = nn.Identity()
        model.eval()
        self.model = model
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, image):
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        if image.shape[1] != 3:
            raise ValueError("FID expects one-channel or three-channel images")
        image = F.interpolate(
            image,
            size=(299, 299),
            mode="bilinear",
            align_corners=False,
        )
        image = (image - self.mean) / self.std
        features = self.model(image)
        if hasattr(features, "logits"):
            features = features.logits
        return features.flatten(1)


def frechet_inception_distance(real_features, generated_features, eps=1e-6):
    """Compute FID from two [N, D] Inception feature matrices."""
    import numpy as np
    from scipy import linalg

    real = real_features.detach().cpu().double().numpy()
    generated = generated_features.detach().cpu().double().numpy()
    if real.ndim != 2 or generated.ndim != 2:
        raise ValueError("FID features must be two-dimensional")
    if real.shape[1] != generated.shape[1]:
        raise ValueError("Real and generated FID features must have equal dimensions")
    if real.shape[0] < 2 or generated.shape[0] < 2:
        return float("nan")

    mean_real = np.mean(real, axis=0)
    mean_generated = np.mean(generated, axis=0)
    covariance_real = np.cov(real, rowvar=False)
    covariance_generated = np.cov(generated, rowvar=False)
    covariance_mean, _ = linalg.sqrtm(
        covariance_real.dot(covariance_generated),
        disp=False,
    )
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(covariance_real.shape[0]) * eps
        covariance_mean = linalg.sqrtm(
            (covariance_real + offset).dot(covariance_generated + offset)
        )
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real

    mean_difference = mean_real - mean_generated
    fid = (
        mean_difference.dot(mean_difference)
        + np.trace(covariance_real)
        + np.trace(covariance_generated)
        - 2.0 * np.trace(covariance_mean)
    )
    return float(max(fid, 0.0))
