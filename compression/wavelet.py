"""Wavelet-based compression using 2D discrete wavelet transforms."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pywt

from compression.base import CompressionExecutionResult, build_execution_result


def compress_band(
    band: np.ndarray,
    *,
    wavelet: str,
    level: int,
    keep_fraction: float,
) -> tuple[np.ndarray, int]:
    """
    Apply 2D wavelet decomposition, threshold coefficients, and reconstruct.

    Returns the reconstructed band and the count of retained (non-zero) coefficients.
    """
    band_f = band.astype(np.float64)
    wavelet_obj = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(min(band.shape), wavelet_obj.dec_len)
    safe_level = max(1, min(level, max_level if max_level > 0 else 1))

    coeffs = pywt.wavedec2(band_f, wavelet=wavelet_obj, level=safe_level)
    coeff_array, coeff_slices = pywt.coeffs_to_array(coeffs)

    keep_count = max(1, int(np.ceil(coeff_array.size * keep_fraction)))
    threshold = np.partition(np.abs(coeff_array).ravel(), -keep_count)[-keep_count]
    compressed_array = np.where(np.abs(coeff_array) >= threshold, coeff_array, 0.0)
    retained_count = int(np.count_nonzero(compressed_array))

    compressed_coeffs = pywt.array_to_coeffs(
        compressed_array, coeff_slices, output_format="wavedec2"
    )
    reconstructed = pywt.waverec2(compressed_coeffs, wavelet=wavelet_obj)
    reconstructed = reconstructed[: band.shape[0], : band.shape[1]]
    reconstructed = np.clip(reconstructed, band.min(), band.max())
    return reconstructed.astype(band.dtype), retained_count


def run_wavelet_compression(
    bands: dict[str, np.ndarray],
    *,
    wavelet: str,
    level: int,
    keep_fraction: float,
) -> CompressionExecutionResult:
    """Compress all bands independently with wavelet thresholding."""
    start = perf_counter()
    reconstructed_bands: dict[str, np.ndarray] = {}
    retained_total = 0

    for name, band in bands.items():
        reconstructed, retained_count = compress_band(
            band, wavelet=wavelet, level=level, keep_fraction=keep_fraction
        )
        reconstructed_bands[name] = reconstructed
        retained_total += retained_count

    runtime_seconds = perf_counter() - start
    return build_execution_result(
        method="wavelet",
        bands=bands,
        reconstructed_bands=reconstructed_bands,
        compressed_bytes_estimate=retained_total * 8,
        runtime_seconds=runtime_seconds,
        metadata={
            "wavelet": wavelet,
            "level": level,
            "keep_fraction": keep_fraction,
        },
    )
