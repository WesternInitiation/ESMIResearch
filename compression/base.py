"""Shared types and quality metrics for all compression methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


@dataclass
class ChannelReport:
    name: str
    rmse: float
    mae: float
    psnr: float
    ssim: float


@dataclass
class CompressionExecutionResult:
    method: str
    reconstructed_bands: dict[str, np.ndarray]
    reconstructed_image: np.ndarray
    original_bytes: int
    compressed_bytes_estimate: int
    runtime_seconds: float
    metadata: dict[str, Any]
    channel_reports: list[ChannelReport]


def _finite(value: float, *, fallback: float) -> float:
    """JSON-safe float (Cloud Run rejects Inf/NaN in JSONResponse)."""
    number = float(value)
    return number if np.isfinite(number) else fallback


def compute_band_metrics(
    name: str, original: np.ndarray, reconstructed: np.ndarray
) -> ChannelReport:
    original_f = original.astype(np.float64)
    reconstructed_f = reconstructed.astype(np.float64)
    diff = reconstructed_f - original_f
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    data_range = float(original_f.max() - original_f.min())
    if data_range <= 0:
        data_range = 1.0
    # Perfect reconstructions yield +inf from skimage; clamp for JSON clients.
    if rmse <= 0.0:
        psnr = 99.0
    else:
        psnr = _finite(
            float(
                peak_signal_noise_ratio(
                    original_f, reconstructed_f, data_range=data_range
                )
            ),
            fallback=99.0,
        )
    try:
        ssim = _finite(
            float(
                structural_similarity(
                    original_f, reconstructed_f, data_range=data_range
                )
            ),
            fallback=1.0 if rmse <= 0.0 else 0.0,
        )
    except ValueError:
        ssim = 1.0 if rmse <= 0.0 else 0.0
    return ChannelReport(
        name=name,
        rmse=_finite(rmse, fallback=0.0),
        mae=_finite(mae, fallback=0.0),
        psnr=psnr,
        ssim=ssim,
    )


def build_execution_result(
    *,
    method: str,
    bands: dict[str, np.ndarray],
    reconstructed_bands: dict[str, np.ndarray],
    compressed_bytes_estimate: int,
    runtime_seconds: float,
    metadata: dict[str, Any],
) -> CompressionExecutionResult:
    band_order = list(bands.keys())
    reconstructed_image = np.stack(
        [reconstructed_bands[name] for name in band_order], axis=-1
    )
    channel_reports = [
        compute_band_metrics(name, bands[name], reconstructed_bands[name])
        for name in band_order
    ]
    original_bytes = sum(band.nbytes for band in bands.values())
    return CompressionExecutionResult(
        method=method,
        reconstructed_bands=reconstructed_bands,
        reconstructed_image=reconstructed_image,
        original_bytes=original_bytes,
        compressed_bytes_estimate=compressed_bytes_estimate,
        runtime_seconds=runtime_seconds,
        metadata=metadata,
        channel_reports=channel_reports,
    )
