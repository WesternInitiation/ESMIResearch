"""Spectral index computation and benchmark metrics for compression evaluation.

NDVI / NDWI formulas follow the same normalized-difference style used in the
browser lab. Landsat Collection 2 surface-reflectance helpers mirror the
approach in ``NDVI_RR.py`` (scale/offset, DN=0 fill mask, safe divide, clip).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.metrics import structural_similarity as ssim

# Landsat Collection 2 Level-2 surface reflectance (see NDVI_RR.py).
LANDSAT_C2_SR_SCALE = np.float32(0.0000275)
LANDSAT_C2_SR_OFFSET = np.float32(-0.2)
LANDSAT_C2_FILL_VALUE = 0


@dataclass
class IndexMetrics:
    rmse: float
    mae: float
    correlation: float
    ssim: float
    bias: float
    valid_pixel_fraction: float


# Back-compat alias used by Streamlit / persistence call sites.
NDVIMetrics = IndexMetrics


def landsat_c2_fill_mask(*bands: np.ndarray) -> np.ndarray:
    """True where any band is the Landsat C2 fill DN (0)."""
    if not bands:
        raise ValueError("At least one band is required.")
    mask = bands[0] == LANDSAT_C2_FILL_VALUE
    for band in bands[1:]:
        if band.shape != bands[0].shape:
            raise ValueError("All bands must share the same shape.")
        mask |= band == LANDSAT_C2_FILL_VALUE
    return mask


def to_landsat_c2_sr(
    band: np.ndarray,
    *,
    scale: float = float(LANDSAT_C2_SR_SCALE),
    offset: float = float(LANDSAT_C2_SR_OFFSET),
) -> np.ndarray:
    """Convert Landsat Collection 2 DN values to surface reflectance."""
    return band.astype(np.float32) * np.float32(scale) + np.float32(offset)


def looks_like_landsat_c2_dn(band: np.ndarray) -> bool:
    """Heuristic: integer-like values with a typical Landsat C2 DN range."""
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return False
    sample = finite if finite.size <= 100_000 else finite.ravel()[:: max(1, finite.size // 100_000)]
    vmax = float(np.nanmax(sample))
    vmin = float(np.nanmin(sample))
    # Reflectance is typically in ~[-0.2, 1]; DNs span thousands.
    if vmax <= 2.0 and vmin >= -1.0:
        return False
    return vmax >= 100.0


def prepare_landsat_c2_bands(
    *bands: np.ndarray,
    apply_sr: bool | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Optionally convert Landsat C2 DNs to SR and build a shared fill mask.

    When ``apply_sr`` is None, conversion (and DN=0 fill masking) is applied
    if the first band looks like Collection 2 digital numbers.
    """
    if not bands:
        raise ValueError("At least one band is required.")
    should_convert = looks_like_landsat_c2_dn(bands[0]) if apply_sr is None else apply_sr
    if should_convert:
        fill = landsat_c2_fill_mask(*bands)
        prepared = [to_landsat_c2_sr(band) for band in bands]
        return prepared, fill
    prepared = [band.astype(np.float32, copy=False) for band in bands]
    return prepared, np.zeros(bands[0].shape, dtype=bool)


def normalized_difference(
    first: np.ndarray,
    second: np.ndarray,
    *,
    eps: float = 1e-10,
    clip: bool = True,
    nodata_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Generic normalized difference: (first - second) / (first + second).

    Uses an in-place ``np.divide`` with a denominator guard (same pattern as
    ``NDVI_RR.py``). Invalid / fill pixels are set to NaN.
    """
    if first.shape != second.shape:
        raise ValueError("Bands must have the same shape.")

    first_f = first.astype(np.float32, copy=False)
    second_f = second.astype(np.float32, copy=False)
    numerator = first_f - second_f
    denominator = first_f + second_f
    out = np.full(first.shape, np.nan, dtype=np.float32)
    np.divide(
        numerator,
        denominator,
        out=out,
        where=np.abs(denominator) > np.float32(eps),
    )
    if clip:
        np.clip(out, np.float32(-1.0), np.float32(1.0), out=out)
    if nodata_mask is not None:
        if nodata_mask.shape != first.shape:
            raise ValueError("nodata_mask must match band shape.")
        out[nodata_mask] = np.nan
    return out


def _prepare_index_bands(
    *bands: np.ndarray,
    landsat_c2_sr: bool | None,
    nodata_mask: np.ndarray | None,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Apply optional Landsat C2 SR conversion and combine nodata masks."""
    if landsat_c2_sr is False:
        prepared = [band.astype(np.float32, copy=False) for band in bands]
        return prepared, nodata_mask

    prepared, fill = prepare_landsat_c2_bands(*bands, apply_sr=landsat_c2_sr)
    # fill is all-False when SR conversion was not applied.
    combined = fill if nodata_mask is None else (fill | nodata_mask)
    if not np.any(combined):
        return prepared, nodata_mask
    return prepared, combined


def compute_ndvi(
    red: np.ndarray,
    nir: np.ndarray,
    *,
    eps: float = 1e-10,
    clip: bool = True,
    nodata_mask: np.ndarray | None = None,
    landsat_c2_sr: bool | None = False,
) -> np.ndarray:
    """Normalized Difference Vegetation Index: (NIR - Red) / (NIR + Red)."""
    (red_p, nir_p), mask = _prepare_index_bands(
        red, nir, landsat_c2_sr=landsat_c2_sr, nodata_mask=nodata_mask
    )
    return normalized_difference(
        nir_p, red_p, eps=eps, clip=clip, nodata_mask=mask
    )


def compute_ndwi(
    green: np.ndarray,
    second: np.ndarray,
    *,
    eps: float = 1e-10,
    clip: bool = True,
    nodata_mask: np.ndarray | None = None,
    landsat_c2_sr: bool | None = False,
) -> np.ndarray:
    """
    NDWI / MNDWI via normalized difference:

    - McFeeters NDWI: (Green - NIR) / (Green + NIR)
    - MNDWI: (Green - SWIR) / (Green + SWIR)
    """
    (green_p, second_p), mask = _prepare_index_bands(
        green, second, landsat_c2_sr=landsat_c2_sr, nodata_mask=nodata_mask
    )
    return normalized_difference(
        green_p, second_p, eps=eps, clip=clip, nodata_mask=mask
    )


def index_stats(index: np.ndarray) -> dict[str, float | int]:
    """Summary statistics over finite index pixels (as in ``NDVI_RR.py``)."""
    valid_mask = np.isfinite(index)
    valid_pixel_count = int(np.count_nonzero(valid_mask))
    if valid_pixel_count == 0:
        return {
            "valid_pixel_count": 0,
            "valid_pixel_fraction": 0.0,
            "minimum": float("nan"),
            "maximum": float("nan"),
            "mean": float("nan"),
            "pixels_at_negative_one": 0,
            "pixels_at_positive_one": 0,
        }
    return {
        "valid_pixel_count": valid_pixel_count,
        "valid_pixel_fraction": float(valid_pixel_count / index.size),
        "minimum": float(np.nanmin(index)),
        "maximum": float(np.nanmax(index)),
        "mean": float(np.nanmean(index)),
        "pixels_at_negative_one": int(np.count_nonzero(index == -1.0)),
        "pixels_at_positive_one": int(np.count_nonzero(index == 1.0)),
    }


def _valid_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & np.isfinite(b)


def compare_index_maps(reference: np.ndarray, candidate: np.ndarray) -> IndexMetrics:
    """Benchmark how well compressed imagery preserves a spectral index map."""
    if reference.shape != candidate.shape:
        raise ValueError("Index arrays must have the same shape.")
    if reference.ndim != 2:
        raise ValueError("Index comparison requires two-dimensional arrays.")

    mask = _valid_mask(reference, candidate)
    if not np.any(mask):
        return IndexMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

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

    return IndexMetrics(
        rmse=rmse,
        mae=mae,
        correlation=correlation,
        ssim=ssim_value,
        bias=bias,
        valid_pixel_fraction=float(mask.mean()),
    )


def compare_ndvi(reference: np.ndarray, candidate: np.ndarray) -> IndexMetrics:
    """Benchmark how well compressed imagery preserves NDVI."""
    return compare_index_maps(reference, candidate)


def compare_ndwi(reference: np.ndarray, candidate: np.ndarray) -> IndexMetrics:
    """Benchmark how well compressed imagery preserves NDWI / MNDWI."""
    return compare_index_maps(reference, candidate)


def rank_sweep_metrics(
    red: np.ndarray,
    nir: np.ndarray,
    compress_fn,
    ranks: list[int],
) -> list[tuple[int, IndexMetrics]]:
    """Evaluate NDVI preservation across a sweep of SVD ranks."""
    reference_ndvi = compute_ndvi(red, nir)
    results: list[tuple[int, IndexMetrics]] = []

    for rank in ranks:
        red_c, nir_c = compress_fn(red, nir, rank)
        candidate_ndvi = compute_ndvi(red_c, nir_c)
        metrics = compare_ndvi(reference_ndvi, candidate_ndvi)
        results.append((rank, metrics))

    return results
