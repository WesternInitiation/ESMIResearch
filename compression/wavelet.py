"""Wavelet-based compression using 2D discrete wavelet transforms (PyWavelets).

Compared with the ESMI Slack `wavelet_compress.py` script (ap-atul/wavelets + db4
full-coefficient NPZ), this module keeps the research knobs we need
(keep-fraction / levels) while adopting the better reconstruction choices:

  - Default basis **db4** (same as the Slack script; far better energy
    compaction on satellite rasters than Haar)
  - True 2D DWT via ``pywt.wavedec2`` (proper for images; the Slack helper
    applies ``FastWaveletTransform`` per band)
  - Always retain the approximation (LL) sub-band, then keep the largest
    detail coefficients up to ``keep_fraction`` of all coefficients — much
    higher PSNR at the same sparsity than hard-thresholding the whole array
  - float32 transform path (matches the Slack script's memory profile)

Cloud Run / Streamlit call this module; the browser engine uses a fixed Haar
implementation with the same LL-preserving policy.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pywt

from compression.base import CompressionExecutionResult, build_execution_result

# Prefer the Slack-script default; Haar remains available for A/B tests.
DEFAULT_WAVELET = "db4"
SUPPORTED_WAVELETS = ("haar", "db2", "db4", "sym2", "sym4")


def _approx_mask(coeff_array: np.ndarray, coeff_slices: list) -> np.ndarray:
    """Boolean mask of the approximation (LL) coefficients in the packed array."""
    mask = np.zeros(coeff_array.shape, dtype=bool)
    approx = coeff_slices[0]
    mask[approx] = True
    return mask


def compress_band(
    band: np.ndarray,
    *,
    wavelet: str,
    level: int,
    keep_fraction: float,
) -> tuple[np.ndarray, int]:
    """
    2D wavelet decompose → keep LL + top detail coeffs → reconstruct.

    Returns the reconstructed band and the count of retained coefficients.
    """
    wavelet_name = wavelet if wavelet in SUPPORTED_WAVELETS else DEFAULT_WAVELET
    band_f = band.astype(np.float32, copy=False)
    wavelet_obj = pywt.Wavelet(wavelet_name)
    max_level = pywt.dwt_max_level(min(band.shape), wavelet_obj.dec_len)
    safe_level = max(1, min(int(level), max_level if max_level > 0 else 1))
    keep_fraction = float(np.clip(keep_fraction, 0.001, 1.0))

    coeffs = pywt.wavedec2(band_f, wavelet=wavelet_obj, level=safe_level)
    coeff_array, coeff_slices = pywt.coeffs_to_array(coeffs)
    approx_mask = _approx_mask(coeff_array, coeff_slices)
    approx_count = int(approx_mask.sum())

    # Budget: at least every LL coeff, then fill up to keep_fraction of all coeffs.
    keep_count = max(approx_count, int(np.ceil(coeff_array.size * keep_fraction)))
    keep_count = min(keep_count, coeff_array.size)

    magnitudes = np.abs(coeff_array)
    # Force LL into the retained set by ranking it above every detail coeff.
    ranking = magnitudes.copy()
    ranking[approx_mask] = np.inf
    flat = ranking.ravel()
    if keep_count >= flat.size:
        compressed_array = coeff_array.copy()
    else:
        threshold = np.partition(flat, -keep_count)[-keep_count]
        compressed_array = np.where(
            (magnitudes >= threshold) | approx_mask,
            coeff_array,
            0.0,
        )

    retained_count = int(np.count_nonzero(compressed_array))
    compressed_coeffs = pywt.array_to_coeffs(
        compressed_array, coeff_slices, output_format="wavedec2"
    )
    reconstructed = pywt.waverec2(compressed_coeffs, wavelet=wavelet_obj)
    reconstructed = reconstructed[: band.shape[0], : band.shape[1]]

    # Match original dynamic range (satellite DN / reflectance).
    lo = float(np.min(band))
    hi = float(np.max(band))
    reconstructed = np.clip(reconstructed, lo, hi)
    return reconstructed.astype(band.dtype, copy=False), retained_count


def run_wavelet_compression(
    bands: dict[str, np.ndarray],
    *,
    wavelet: str = DEFAULT_WAVELET,
    level: int = 3,
    keep_fraction: float = 0.08,
) -> CompressionExecutionResult:
    """Compress all bands independently with LL-preserving wavelet thresholding."""
    start = perf_counter()
    reconstructed_bands: dict[str, np.ndarray] = {}
    retained_total = 0
    wavelet_name = wavelet if wavelet in SUPPORTED_WAVELETS else DEFAULT_WAVELET

    for name, band in bands.items():
        reconstructed, retained_count = compress_band(
            band,
            wavelet=wavelet_name,
            level=level,
            keep_fraction=keep_fraction,
        )
        reconstructed_bands[name] = reconstructed
        retained_total += retained_count

    runtime_seconds = perf_counter() - start
    # float32-ish payload estimate (value + index overhead folded into 8 bytes).
    return build_execution_result(
        method="wavelet",
        bands=bands,
        reconstructed_bands=reconstructed_bands,
        compressed_bytes_estimate=retained_total * 8,
        runtime_seconds=runtime_seconds,
        metadata={
            "wavelet": wavelet_name,
            "level": int(level),
            "keep_fraction": float(keep_fraction),
            "preserve_approximation": True,
            "backend": "pywt.wavedec2",
        },
    )
