"""NDVI computation and benchmark metrics for compression evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.metrics import structural_similarity as ssim


@dataclass
class NDVIMetrics:
    rmse: float
    mae: float
    correlation: float
    ssim: float
    bias: float
    valid_pixel_fraction: float


def compute_ndvi(
    red: np.ndarray,
    nir: np.ndarray,
    *,
    eps: float = 1e-8,
    clip: bool = True,
) -> np.ndarray:
    """Normalized Difference Vegetation Index: (NIR - Red) / (NIR + Red)."""
    denominator = nir.astype(np.float64) + red.astype(np.float64)
    valid = np.abs(denominator) > eps
    ndvi = np.full(red.shape, np.nan, dtype=np.float64)
    ndvi[valid] = (nir[valid] - red[valid]) / denominator[valid]
    if clip:
        ndvi = np.clip(ndvi, -1.0, 1.0)
    return ndvi


def _valid_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & np.isfinite(b)


def compare_ndvi(reference: np.ndarray, candidate: np.ndarray) -> NDVIMetrics:
    """Benchmark how well compressed imagery preserves NDVI."""
    if reference.shape != candidate.shape:
        raise ValueError("NDVI arrays must have the same shape.")
    if reference.ndim != 2:
        raise ValueError("NDVI comparison requires two-dimensional arrays.")

    mask = _valid_mask(reference, candidate)
    if not np.any(mask):
        return NDVIMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    ref = reference[mask]
    cand = candidate[mask]
    diff = cand - ref

    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))

    if ref.std() > 0 and cand.std() > 0:
        correlation = float(np.corrcoef(ref, cand)[0, 1])
    else:
        correlation = 0.0

    data_range = float(ref.max() - ref.min())
    if data_range <= 0:
        data_range = 1.0

    min_dimension = min(reference.shape)
    if min_dimension >= 3:
        window_size = min(7, min_dimension)
        if window_size % 2 == 0:
            window_size -= 1
        reference_filled = np.where(mask, reference, 0.0)
        candidate_filled = np.where(mask, candidate, 0.0)
        _, ssim_map = ssim(
            reference_filled,
            candidate_filled,
            data_range=data_range,
            win_size=window_size,
            full=True,
        )
        ssim_value = float(np.mean(ssim_map[mask]))
    else:
        ssim_value = float(np.allclose(ref, cand))

    return NDVIMetrics(
        rmse=rmse,
        mae=mae,
        correlation=correlation,
        ssim=ssim_value,
        bias=bias,
        valid_pixel_fraction=float(mask.mean()),
    )


def rank_sweep_metrics(
    red: np.ndarray,
    nir: np.ndarray,
    compress_fn,
    ranks: list[int],
) -> list[tuple[int, NDVIMetrics]]:
    """Evaluate NDVI preservation across a sweep of SVD ranks."""
    reference_ndvi = compute_ndvi(red, nir)
    results: list[tuple[int, NDVIMetrics]] = []

    for rank in ranks:
        red_c, nir_c = compress_fn(red, nir, rank)
        candidate_ndvi = compute_ndvi(red_c, nir_c)
        metrics = compare_ndvi(reference_ndvi, candidate_ndvi)
        results.append((rank, metrics))

    return results
