"""Round-trip checks for every lab compression method."""

from __future__ import annotations

import unittest

import numpy as np

from compression.bandwidth import run_bandwidth_compression
from compression.jpeg2000 import run_jpeg2000_compression
from compression.lzw import run_lzw_compression
from compression.svd import run_svd_compression
from compression.wavelet import run_wavelet_compression
from svd_compression import ChannelCompressionConfig, CompressionConfig


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


def _synthetic(h: int, w: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    band = rng.normal(120.0, 40.0, size=(h, w)).astype(np.float64)
    band = np.clip(band, 0, 255)
    return {"gray": band}


class CompressionRoundtripTests(unittest.TestCase):
    def test_svd_full_rank_near_lossless(self) -> None:
        bands = _synthetic(32, 32)
        h, w = bands["gray"].shape
        rank = min(h, w)
        config = CompressionConfig(
            channels={"gray": ChannelCompressionConfig(rank=rank)},
            mode="rank",
            normalize_before_svd=True,
        )
        out = run_svd_compression(bands, config)
        self.assertLess(_rmse(bands["gray"], out.reconstructed_bands["gray"]), 1e-6)

    def test_wavelet_keep_all_even_and_odd(self) -> None:
        for shape in ((64, 64), (65, 63), (33, 33)):
            with self.subTest(shape=shape):
                bands = _synthetic(*shape)
                out = run_wavelet_compression(
                    bands, wavelet="haar", level=2, keep_fraction=1.0
                )
                self.assertLess(
                    _rmse(bands["gray"], out.reconstructed_bands["gray"]),
                    1e-4,
                )

    def test_bandwidth_keep_all_odd_size(self) -> None:
        bands = _synthetic(33, 33)
        out = run_bandwidth_compression(bands, keep_fraction=1.0)
        self.assertLess(_rmse(bands["gray"], out.reconstructed_bands["gray"]), 1e-8)
        # Mask must cover every bin when keep_fraction=1.
        self.assertEqual(
            out.compressed_bytes_estimate,
            33 * 33 * 16,
        )

    def test_lzw_roundtrip_size_and_quantize(self) -> None:
        bands = _synthetic(48, 40, seed=3)
        out = run_lzw_compression(bands)
        # LZW is lossless on the quantized uint8; reconstruction stays in original range.
        recon = out.reconstructed_bands["gray"]
        self.assertEqual(recon.shape, bands["gray"].shape)
        self.assertGreater(out.compressed_bytes_estimate, 0)
        self.assertLess(_rmse(bands["gray"], recon), 1.0)

    def test_jpeg2000_encodes_or_skips(self) -> None:
        bands = _synthetic(32, 32, seed=4)
        try:
            low = run_jpeg2000_compression(bands, rate=1)
            high = run_jpeg2000_compression(bands, rate=20)
        except RuntimeError as exc:
            self.skipTest(f"JPEG2000 unavailable: {exc}")
        self.assertEqual(low.reconstructed_bands["gray"].shape, (32, 32))
        # Higher OpenJPEG rate → smaller (or equal) payload.
        self.assertLessEqual(
            high.compressed_bytes_estimate,
            low.compressed_bytes_estimate * 1.05,
        )


if __name__ == "__main__":
    unittest.main()
