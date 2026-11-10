from __future__ import annotations

import math

import numpy as np
import torch
from skimage.metrics import structural_similarity


def evaluation_array(value, *, name="image"):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().float().numpy()
    value = np.asarray(value, dtype=np.float32).squeeze()
    if value.ndim != 2:
        raise ValueError(f"Single-channel HxW data required, got shape {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    minimum, maximum = float(value.min()), float(value.max())
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise ValueError(f"{name} must be float image intensity in [0, 1], got [{minimum}, {maximum}]")
    return np.clip(value, 0.0, 1.0)


def grayscale_to_three_channel(value: torch.Tensor) -> torch.Tensor:
    """Repeat grayscale intensity without pseudo-colour or contrast changes."""
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, dtype=torch.float32)
    value = value.float()
    if value.ndim == 2:
        value = value.unsqueeze(0).unsqueeze(0)
    elif value.ndim == 3:
        value = value.unsqueeze(0) if value.shape[0] == 1 else value.unsqueeze(1)
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError(f"Expected grayscale tensor shaped HxW, 1xHxW, or Nx1xHxW; got {tuple(value.shape)}")
    if not torch.isfinite(value).all() or value.min() < -1e-6 or value.max() > 1.0 + 1e-6:
        raise ValueError("Perceptual/distribution input must be finite and in [0, 1]")
    return value.clamp(0.0, 1.0).repeat(1, 3, 1, 1)


def evaluation_to_lpips_range(value: torch.Tensor) -> torch.Tensor:
    return grayscale_to_three_channel(value) * 2.0 - 1.0


def compute_metrics(prediction, target):
    prediction = evaluation_array(prediction, name="prediction")
    target = evaluation_array(target, name="target")
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} != {target.shape}")
    difference = prediction - target
    mae = float(np.mean(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    psnr = float("inf") if rmse == 0 else float(20.0 * math.log10(1.0 / rmse))
    minimum_side = min(prediction.shape)
    win_size = min(7, minimum_side if minimum_side % 2 else minimum_side - 1)
    if win_size < 3:
        raise ValueError("SSIM requires images with a minimum side of at least 3")
    ssim = float(structural_similarity(target, prediction, data_range=1.0, win_size=win_size))
    return {"mae": mae, "rmse": rmse, "psnr": psnr, "ssim": ssim}


def compute_ms_ssim(prediction, target) -> float:
    try:
        from torchmetrics.functional.image import multiscale_structural_similarity_index_measure
    except ImportError as exc:
        raise RuntimeError("MS-SSIM requires the declared torchmetrics dependency") from exc
    prediction_array = evaluation_array(prediction, name="prediction")
    target_array = evaluation_array(target, name="target")
    if prediction_array.shape != target_array.shape:
        raise ValueError(f"Shape mismatch: {prediction_array.shape} != {target_array.shape}")
    prediction_tensor = torch.from_numpy(prediction_array)[None, None]
    target_tensor = torch.from_numpy(target_array)[None, None]
    return float(multiscale_structural_similarity_index_measure(
        prediction_tensor, target_tensor, data_range=1.0
    ))
