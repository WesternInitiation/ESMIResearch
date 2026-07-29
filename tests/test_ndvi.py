from __future__ import annotations

import unittest

import numpy as np

from ndvi import (
    LANDSAT_C2_SR_OFFSET,
    LANDSAT_C2_SR_SCALE,
    compare_index_maps,
    compare_ndvi,
    compare_ndwi,
    compute_ndvi,
    compute_ndwi,
    index_stats,
    looks_like_landsat_c2_dn,
    to_landsat_c2_sr,
)


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


class NDVIComputeTests(unittest.TestCase):
    def test_basic_ndvi_formula(self) -> None:
        red = np.array([[0.1, 0.2], [0.3, 0.0]], dtype=np.float32)
        nir = np.array([[0.4, 0.2], [0.1, 0.5]], dtype=np.float32)

        ndvi = compute_ndvi(red, nir)

        expected = (nir - red) / (nir + red)
        np.testing.assert_allclose(ndvi, expected, rtol=1e-5)
        self.assertTrue(np.all(ndvi >= -1.0) and np.all(ndvi <= 1.0))

    def test_landsat_c2_fill_and_scale(self) -> None:
        red = np.array([[0, 10000], [20000, 15000]], dtype=np.float32)
        nir = np.array([[0, 20000], [10000, 18000]], dtype=np.float32)

        ndvi = compute_ndvi(red, nir, landsat_c2_sr=True)

        self.assertTrue(np.isnan(ndvi[0, 0]))
        red_sr = to_landsat_c2_sr(red)
        nir_sr = to_landsat_c2_sr(nir)
        expected = (nir_sr[0, 1] - red_sr[0, 1]) / (nir_sr[0, 1] + red_sr[0, 1])
        self.assertAlmostEqual(float(ndvi[0, 1]), float(expected), places=5)
        self.assertTrue(looks_like_landsat_c2_dn(red))
        self.assertFalse(looks_like_landsat_c2_dn(red_sr))

    def test_auto_detect_landsat_dn(self) -> None:
        red = np.full((4, 4), 12000.0, dtype=np.float32)
        nir = np.full((4, 4), 18000.0, dtype=np.float32)
        red[0, 0] = 0
        nir[0, 0] = 0

        ndvi = compute_ndvi(red, nir, landsat_c2_sr=None)
        self.assertTrue(np.isnan(ndvi[0, 0]))
        self.assertTrue(np.isfinite(ndvi[1, 1]))

    def test_index_stats_match_rr_style(self) -> None:
        index = np.array([[np.nan, -1.0], [0.25, 1.0]], dtype=np.float32)
        stats = index_stats(index)
        self.assertEqual(stats["valid_pixel_count"], 3)
        self.assertEqual(stats["pixels_at_negative_one"], 1)
        self.assertEqual(stats["pixels_at_positive_one"], 1)
        self.assertAlmostEqual(stats["mean"], (-1.0 + 0.25 + 1.0) / 3)


class NDWITests(unittest.TestCase):
    """NDWI preservation checks mirror the NDVI comparison style."""

    def test_mcfeeters_ndwi_formula(self) -> None:
        green = np.array([[0.3, 0.2], [0.4, 0.1]], dtype=np.float32)
        nir = np.array([[0.1, 0.2], [0.2, 0.4]], dtype=np.float32)

        ndwi = compute_ndwi(green, nir)
        expected = (green - nir) / (green + nir)
        np.testing.assert_allclose(ndwi, expected, rtol=1e-5)

    def test_mndwi_with_swir(self) -> None:
        green = np.array([[0.25, 0.30]], dtype=np.float32)
        swir = np.array([[0.10, 0.35]], dtype=np.float32)
        mndwi = compute_ndwi(green, swir)
        expected = (green - swir) / (green + swir)
        np.testing.assert_allclose(mndwi, expected, rtol=1e-5)

    def test_ndwi_comparison_metrics_match_ndvi_style(self) -> None:
        reference = np.linspace(-0.8, 0.8, 36).reshape(6, 6)
        candidate = reference + 0.02
        metrics = compare_ndwi(reference, candidate)
        self.assertTrue(np.isfinite(metrics.ssim))
        self.assertAlmostEqual(metrics.rmse, 0.02)
        self.assertAlmostEqual(metrics.bias, 0.02)

    def test_landsat_c2_ndwi_masks_fill(self) -> None:
        green = np.array([[0, 11000], [14000, 16000]], dtype=np.float32)
        nir = np.array([[0, 9000], [15000, 12000]], dtype=np.float32)
        ndwi = compute_ndwi(green, nir, landsat_c2_sr=True)
        self.assertTrue(np.isnan(ndwi[0, 0]))
        self.assertTrue(np.isfinite(ndwi[0, 1]))

    def test_compare_index_maps_shared_by_both(self) -> None:
        reference = np.ones((4, 4), dtype=np.float32) * 0.5
        candidate = reference.copy()
        self.assertEqual(compare_index_maps(reference, candidate).rmse, 0.0)
        self.assertEqual(compare_ndvi(reference, candidate).rmse, 0.0)
        self.assertEqual(compare_ndwi(reference, candidate).rmse, 0.0)


class LandsatHelpersTests(unittest.TestCase):
    def test_scale_offset_constants(self) -> None:
        self.assertAlmostEqual(float(LANDSAT_C2_SR_SCALE), 0.0000275)
        self.assertAlmostEqual(float(LANDSAT_C2_SR_OFFSET), -0.2)
        dn = np.array([0.0, 10000.0], dtype=np.float32)
        sr = to_landsat_c2_sr(dn)
        self.assertAlmostEqual(float(sr[1]), 10000 * 0.0000275 - 0.2, places=6)


if __name__ == "__main__":
    unittest.main()
