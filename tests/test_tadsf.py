import unittest

import torch

from model.generator import TaskAdaptiveDualSpaceFusion


class TaskAdaptiveDualSpaceFusionTest(unittest.TestCase):
    def test_exp_log_maps_round_trip(self):
        module = TaskAdaptiveDualSpaceFusion(
            background_channels=3,
            foreground_channels=7,
            out_channels=3,
            hidden_channels=8,
        )
        tangent = torch.randn(2, 8, 6, 6) * 0.05

        restored = module._logmap0(module._expmap0(tangent))

        torch.testing.assert_close(restored, tangent, rtol=1e-4, atol=1e-5)

    def test_forward_is_finite_and_differentiable(self):
        module = TaskAdaptiveDualSpaceFusion(
            background_channels=3,
            foreground_channels=7,
            out_channels=3,
            hidden_channels=8,
        )
        background = torch.randn(2, 3, 16, 16, requires_grad=True)
        foreground = torch.randn(2, 7, 16, 16, requires_grad=True)

        output = module(background, foreground)
        output.square().mean().backward()

        self.assertEqual(output.shape, (2, 3, 16, 16))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(background.grad).all())
        self.assertTrue(torch.isfinite(foreground.grad).all())

    def test_fusion_modes_are_finite(self):
        background = torch.randn(1, 3, 8, 8)
        foreground = torch.randn(1, 7, 8, 8)

        for mode in ("full", "euc_only", "hyp_only", "no_gate", "no_hyperagg"):
            with self.subTest(mode=mode):
                module = TaskAdaptiveDualSpaceFusion(
                    background_channels=3,
                    foreground_channels=7,
                    out_channels=3,
                    hidden_channels=8,
                    fusion_mode=mode,
                )
                output = module(background, foreground)

                self.assertEqual(output.shape, (1, 3, 8, 8))
                self.assertTrue(torch.isfinite(output).all())

    def test_spatial_hyperagg_mode_is_supported(self):
        module = TaskAdaptiveDualSpaceFusion(
            background_channels=3,
            foreground_channels=7,
            out_channels=3,
            hidden_channels=8,
            hyperagg_mode="spatial",
        )
        background = torch.randn(1, 3, 8, 8)
        foreground = torch.randn(1, 7, 8, 8)

        output = module(background, foreground)

        self.assertEqual(output.shape, (1, 3, 8, 8))
        self.assertTrue(torch.isfinite(output).all())

    def test_large_tangent_values_stay_inside_ball(self):
        module = TaskAdaptiveDualSpaceFusion(
            background_channels=3,
            foreground_channels=7,
            out_channels=3,
            hidden_channels=8,
        )
        tangent = torch.full((1, 8, 4, 4), 1000.0)

        mapped = module._expmap0(tangent)
        radius = torch.linalg.vector_norm(mapped, dim=1)

        self.assertTrue(torch.isfinite(mapped).all())
        self.assertTrue(torch.all(radius < 1.0))

    def test_curvature_must_be_positive(self):
        with self.assertRaises(ValueError):
            TaskAdaptiveDualSpaceFusion(
                background_channels=3,
                foreground_channels=7,
                out_channels=3,
                curvature=0.0,
            )


if __name__ == "__main__":
    unittest.main()
