"""Compression method package — one module per algorithm."""

from compression.base import ChannelReport, CompressionExecutionResult, compute_band_metrics
from compression.bandwidth import run_bandwidth_compression
from compression.jpeg2000 import run_jpeg2000_compression
from compression.lzw import run_lzw_compression
from compression.svd import run_svd_compression
from compression.wavelet import run_wavelet_compression

__all__ = [
    "ChannelReport",
    "CompressionExecutionResult",
    "compute_band_metrics",
    "run_bandwidth_compression",
    "run_jpeg2000_compression",
    "run_lzw_compression",
    "run_svd_compression",
    "run_wavelet_compression",
]
