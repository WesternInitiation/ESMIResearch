"""Bandwidth-domain compression using 2D FFT low-pass filtering."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from compression.base import CompressionExecutionResult, build_execution_result


def compress_band(
    band: np.ndarray,
    *,
    keep_fraction: float,
) -> tuple[np.ndarray, int]:
    """
    Transform to frequency domain, retain a central low-frequency block, and invert.

    keep_fraction controls the fraction of rows/columns (centered on DC) kept in
    the spectrum. Returns reconstructed band and retained frequency-bin count.
    """
    band_f = band.astype(np.float64)
    spectrum = np.fft.fftshift(np.fft.fft2(band_f))
    height, width = band.shape
    half_h = max(1, int(round((height * keep_fraction) / 2)))
    half_w = max(1, int(round((width * keep_fraction) / 2)))
    center_h = height // 2
    center_w = width // 2

    mask = np.zeros_like(spectrum, dtype=bool)
    mask[
        max(0, center_h - half_h) : min(height, center_h + half_h),
        max(0, center_w - half_w) : min(width, center_w + half_w),
    ] = True

    filtered = np.where(mask, spectrum, 0.0)
    reconstructed = np.fft.ifft2(np.fft.ifftshift(filtered)).real
    reconstructed = np.clip(reconstructed, band.min(), band.max())
    return reconstructed.astype(band.dtype), int(np.count_nonzero(mask))


def run_bandwidth_compression(
    bands: dict[str, np.ndarray],
    *,
    keep_fraction: float,
) -> CompressionExecutionResult:
    """Compress all bands with FFT low-pass bandwidth truncation."""
    start = perf_counter()
    reconstructed_bands: dict[str, np.ndarray] = {}
    retained_total = 0

    for name, band in bands.items():
        reconstructed, retained_count = compress_band(band, keep_fraction=keep_fraction)
        reconstructed_bands[name] = reconstructed
        retained_total += retained_count

    runtime_seconds = perf_counter() - start
    return build_execution_result(
        method="bandwidth_transform",
        bands=bands,
        reconstructed_bands=reconstructed_bands,
        compressed_bytes_estimate=retained_total * 16,
        runtime_seconds=runtime_seconds,
        metadata={"keep_fraction": keep_fraction},
    )
