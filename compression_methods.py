"""
Backward-compatible re-exports.

Implementations live in the compression/ package — one file per method:
  - compression/svd.py
  - compression/wavelet.py
  - compression/bandwidth.py
  - compression/jpeg2000.py
"""

from compression import (
    ChannelReport,
    CompressionExecutionResult,
    compute_band_metrics,
    run_bandwidth_compression,
    run_jpeg2000_compression,
    run_svd_compression,
    run_wavelet_compression,
)

__all__ = [
    "ChannelReport",
    "CompressionExecutionResult",
    "compute_band_metrics",
    "run_bandwidth_compression",
    "run_jpeg2000_compression",
    "run_svd_compression",
    "run_wavelet_compression",
]
