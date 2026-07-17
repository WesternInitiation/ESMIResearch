"""SVD compression adapter — wraps svd_compression.py with timing and reporting."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from compression.base import CompressionExecutionResult, build_execution_result
from svd_compression import CompressionConfig, compress_multiband


def run_svd_compression(
    bands: dict[str, np.ndarray],
    config: CompressionConfig,
) -> CompressionExecutionResult:
    """
    Compress each band with truncated singular value decomposition.

    Delegates core linear algebra to svd_compression.compress_multiband and
    attaches runtime plus per-band rank/energy metadata for benchmarking.
    """
    start = perf_counter()
    result = compress_multiband(bands, config)
    runtime_seconds = perf_counter() - start

    reconstructed_bands = {
        channel.name: channel.reconstructed for channel in result.channels
    }
    band_metadata = {
        channel.name: {
            "rank_used": channel.rank_used,
            "energy_retained": channel.energy_retained,
            "weight": channel.weight,
        }
        for channel in result.channels
    }

    return build_execution_result(
        method="svd",
        bands=bands,
        reconstructed_bands=reconstructed_bands,
        compressed_bytes_estimate=result.total_bytes_compressed_estimate,
        runtime_seconds=runtime_seconds,
        metadata={"bands": band_metadata},
    )
