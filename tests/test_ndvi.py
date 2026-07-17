from __future__ import annotations

import unittest

import numpy as np

from ndvi import compare_ndvi


class NDVIComparisonTests(unittest.TestCase):
    def test_small_images_do_not_exceed_ssim_window(self) -> None:
        reference = np.linspace(-1, 1, 36).reshape(6, 6)
        candidate = reference + 0.01

        metrics = compare_ndvi(reference, candidate)

        self.assertTrue(np.isfinite(metrics.ssim))
        self.assertAlmostEqual(metrics.rmse, 0.01)

    def test_invalid_pixels_do_not_produce_nan_metrics(self) -> None:
        reference = np.linspace(-1, 1, 64).reshape(8, 8)
        candidate = reference.copy()
        reference[0, 0] = np.nan
        candidate[1, 1] = np.nan

        metrics = compare_ndvi(reference, candidate)

        self.assertTrue(np.isfinite(metrics.ssim))
        self.assertEqual(metrics.valid_pixel_fraction, 62 / 64)

    def test_tiny_identical_images_have_perfect_fallback_ssim(self) -> None:
        reference = np.ones((2, 2))

        metrics = compare_ndvi(reference, reference.copy())

        self.assertEqual(metrics.ssim, 1.0)


if __name__ == "__main__":
    unittest.main()
