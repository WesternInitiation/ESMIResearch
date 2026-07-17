"""JPEG2000 compression via Pillow encode/decode per band."""

from __future__ import annotations

from io import BytesIO
from time import perf_counter

import numpy as np
from PIL import Image

from compression.base import CompressionExecutionResult, build_execution_result


def compress_band(
    band: np.ndarray,
    *,
    quality_layers: list[int],
) -> tuple[np.ndarray, int]:
    """
    Normalize band to 16-bit, encode as JPEG2000, decode, and rescale.

    Returns reconstructed band and encoded byte size.
    """
    band_f = band.astype(np.float64)
    band_min = float(band_f.min())
    band_max = float(band_f.max())
    scale = band_max - band_min
    if scale <= 0:
        scale = 1.0

    normalized = ((band_f - band_min) / scale * 65535.0).astype(np.uint16)
    image = Image.fromarray(normalized, mode="I;16")

    buffer = BytesIO()
    try:
        image.save(
            buffer, format="JPEG2000", quality_mode="rates", quality_layers=quality_layers
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "JPEG2000 encoding is unavailable in this Python/Pillow build."
        ) from exc

    encoded_bytes = len(buffer.getvalue())
    buffer.seek(0)
    decoded = np.asarray(Image.open(buffer)).astype(np.float64)
    reconstructed = decoded / 65535.0 * scale + band_min
    reconstructed = np.clip(reconstructed, band.min(), band.max())
    return reconstructed.astype(band.dtype), encoded_bytes


def run_jpeg2000_compression(
    bands: dict[str, np.ndarray],
    *,
    rate: int,
) -> CompressionExecutionResult:
    """Compress each band independently with JPEG2000 rate control."""
    start = perf_counter()
    reconstructed_bands: dict[str, np.ndarray] = {}
    encoded_total = 0

    for name, band in bands.items():
        reconstructed, encoded_bytes = compress_band(band, quality_layers=[rate])
        reconstructed_bands[name] = reconstructed
        encoded_total += encoded_bytes

    runtime_seconds = perf_counter() - start
    return build_execution_result(
        method="jpeg2000",
        bands=bands,
        reconstructed_bands=reconstructed_bands,
        compressed_bytes_estimate=encoded_total,
        runtime_seconds=runtime_seconds,
        metadata={"quality_layers": [rate]},
    )
