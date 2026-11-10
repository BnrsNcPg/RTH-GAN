import pytest
import torch

from metrics.image_metrics import evaluation_to_lpips_range, grayscale_to_three_channel


def test_grayscale_is_replicated_to_three_identical_channels():
    image = torch.rand(2, 1, 8, 8)
    adapted = grayscale_to_three_channel(image)
    assert adapted.shape == (2, 3, 8, 8)
    assert torch.equal(adapted[:, 0], adapted[:, 1])
    assert torch.equal(adapted[:, 1], adapted[:, 2])


def test_lpips_adapter_uses_minus_one_to_one_without_colour_mapping():
    image = torch.tensor([[0.0, 1.0], [0.5, 0.25]])
    adapted = evaluation_to_lpips_range(image)
    assert adapted.min() == -1
    assert adapted.max() == 1
    assert torch.equal(adapted[:, 0], adapted[:, 2])


def test_rgb_input_is_rejected_for_pixel_metric_adapter():
    with pytest.raises(ValueError):
        grayscale_to_three_channel(torch.rand(1, 3, 8, 8))
