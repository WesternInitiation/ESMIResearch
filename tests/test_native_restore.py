"""Native-resolution restore behavior used by Cloud Run (all methods, including LZW)."""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from compression.lzw import run_lzw_compression


def downsample_bands(bands: dict[str, np.ndarray], max_dim: int) -> tuple[dict[str, np.ndarray], float]:
    sample = next(iter(bands.values()))
    height, width = sample.shape[:2]
    longest = max(height, width)
    if max_dim <= 0 or longest <= max_dim:
        return bands, 1.0
    scale = max_dim / float(longest)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    out: dict[str, np.ndarray] = {}
    for name, band in bands.items():
        image = Image.fromarray(band.astype(np.float32), mode="F")
        resized = image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        out[name] = np.asarray(resized, dtype=band.dtype)
    return out, scale


def upsample_bands(
    bands: dict[str, np.ndarray], target_width: int, target_height: int
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, band in bands.items():
        image = Image.fromarray(band.astype(np.float32), mode="F")
        resized = image.resize(
            (target_width, target_height),
            resample=Image.Resampling.BILINEAR,
        )
        out[name] = np.asarray(resized, dtype=band.dtype)
    return out


def run_with_restore(bands: dict[str, np.ndarray], max_dim: int):
    """Mirror cloud_run/main.py compress sizing for LZW."""
    native_h, native_w = next(iter(bands.values())).shape[:2]
    process_cap = 0 if int(max_dim) <= 0 else max(64, int(max_dim))
    process_bands, scale = downsample_bands(bands, process_cap)
    result = run_lzw_compression(process_bands)
    process_h, process_w = next(iter(process_bands.values())).shape[:2]
    restored = False
    if scale < 1.0 - 1e-12 or process_w != native_w or process_h != native_h:
        result.reconstructed_bands = upsample_bands(
            result.reconstructed_bands, native_w, native_h
        )
        restored = True
    return result, (native_w, native_h), (process_w, process_h), restored


class NativeRestoreTests(unittest.TestCase):
    def test_lzw_native_max_dim_keeps_shape(self) -> None:
        band = np.linspace(0, 255, 1100 * 1200, dtype=np.float64).reshape(1100, 1200)
        result, native, process, restored = run_with_restore({"gray": band}, 0)
        self.assertEqual(native, (1200, 1100))
        self.assertEqual(process, (1200, 1100))
        self.assertFalse(restored)
        self.assertEqual(result.reconstructed_bands["gray"].shape, (1100, 1200))

    def test_lzw_capped_max_dim_restores_native_shape(self) -> None:
        band = np.linspace(0, 255, 1100 * 1200, dtype=np.float64).reshape(1100, 1200)
        result, native, process, restored = run_with_restore({"gray": band}, 512)
        self.assertEqual(native, (1200, 1100))
        self.assertLess(max(process), 1200)
        self.assertTrue(restored)
        self.assertEqual(result.reconstructed_bands["gray"].shape, (1100, 1200))

    def test_old_max64_trap_is_not_used_for_zero(self) -> None:
        # Regression: legacy code did max(64, max_dim), turning Native into 64px.
        band = np.ones((200, 300), dtype=np.float64)
        _, native, process, restored = run_with_restore({"gray": band}, 0)
        self.assertEqual(process, native)
        self.assertNotEqual(process[0], 64)
        self.assertFalse(restored)


if __name__ == "__main__":
    unittest.main()
