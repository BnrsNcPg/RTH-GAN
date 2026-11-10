import numpy as np
import pytest
import torch

from data.transforms import denormalize
from metrics.image_metrics import evaluation_array


def test_network_to_evaluation_round_trip():
    evaluation = torch.linspace(0, 1, 64, dtype=torch.float32).reshape(1, 1, 8, 8)
    network = evaluation * 2.0 - 1.0
    restored = denormalize(network)
    assert torch.allclose(restored, evaluation)
    assert np.allclose(evaluation_array(restored), evaluation.numpy().squeeze())


def test_evaluation_range_is_validated_not_silently_clipped():
    image = np.zeros((8, 8), dtype=np.float32)
    image[0, 0] = 1.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        evaluation_array(image)
