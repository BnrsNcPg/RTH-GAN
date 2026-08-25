import unittest

import torch

from utils.metrics import (
    batch_ms_ssim,
    batch_psnr,
    batch_rmse,
    batch_ssim,
    frechet_inception_distance,
)


class ImageMetricsTest(unittest.TestCase):
    def test_identical_images_have_ideal_full_reference_scores(self):
        image = torch.rand(2, 3, 256, 256)

        self.assertTrue((batch_psnr(image, image) > 100.0).all())
        torch.testing.assert_close(batch_rmse(image, image), torch.zeros(2))
        torch.testing.assert_close(batch_ssim(image, image), torch.ones(2))
        torch.testing.assert_close(batch_ms_ssim(image, image), torch.ones(2))

    def test_rmse_uses_paper_8_bit_scale(self):
        prediction = torch.zeros(1, 3, 16, 16)
        target = torch.ones_like(prediction)

        torch.testing.assert_close(
            batch_rmse(prediction, target),
            torch.tensor([255.0]),
        )

    def test_fid_is_zero_for_identical_feature_sets(self):
        features = torch.randn(8, 16)

        self.assertAlmostEqual(
            frechet_inception_distance(features, features),
            0.0,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
