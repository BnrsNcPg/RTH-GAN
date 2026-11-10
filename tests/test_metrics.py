import numpy as np
import pytest

from metrics.image_metrics import compute_metrics


def test_identical_single_channel_images():
    image = np.full((16, 16), 0.5)
    metrics = compute_metrics(image, image)
    assert metrics["mae"] == 0
    assert metrics["rmse"] == 0
    assert metrics["psnr"] == float("inf")
    assert metrics["ssim"] == 1


def test_known_rmse_and_psnr():
    target = np.zeros((16, 16), dtype=np.float32)
    prediction = np.full((16, 16), 0.5, dtype=np.float32)
    metrics = compute_metrics(prediction, target)
    assert metrics["rmse"] == pytest.approx(0.5)
    assert metrics["psnr"] == pytest.approx(6.020599913)
