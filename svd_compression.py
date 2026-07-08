"""SVD-based satellite image compression with per-channel controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class ChannelCompressionConfig:
    """Compression settings for a single spectral band."""

    rank: int | None = None
    energy_fraction: float | None = None
    weight: float = 1.0


@dataclass
class CompressionConfig:
    """Full compression configuration across bands."""

    channels: dict[str, ChannelCompressionConfig]
    mode: Literal["rank", "energy"] = "rank"
    normalize_before_svd: bool = True


@dataclass
class ChannelSVDResult:
    name: str
    original: np.ndarray
    reconstructed: np.ndarray
    singular_values: np.ndarray
    rank_used: int
    energy_retained: float
    weight: float


@dataclass
class CompressionResult:
    channels: list[ChannelSVDResult]
    reconstructed_image: np.ndarray
    total_bytes_original: int
    total_bytes_compressed_estimate: int


def _effective_rank(
    singular_values: np.ndarray,
    rank: int | None,
    energy_fraction: float | None,
    mode: Literal["rank", "energy"],
) -> tuple[int, float]:
    total_energy = float(np.sum(singular_values**2))
    if total_energy == 0:
        return 0, 0.0

    cumulative = np.cumsum(singular_values**2) / total_energy

    if mode == "energy" and energy_fraction is not None:
        target = np.clip(energy_fraction, 0.0, 1.0)
        k = int(np.searchsorted(cumulative, target, side="left") + 1)
    elif rank is not None:
        k = max(0, min(int(rank), len(singular_values)))
    else:
        k = len(singular_values)

    retained = float(cumulative[k - 1]) if k > 0 else 0.0
    return k, retained


def compress_channel(
    channel: np.ndarray,
    config: ChannelCompressionConfig,
    mode: Literal["rank", "energy"],
    normalize: bool,
) -> ChannelSVDResult:
    """Compress one band with truncated SVD."""
    matrix = channel.astype(np.float64)
    offset = 0.0
    scale = 1.0

    if normalize:
        offset = float(matrix.min())
        scale = float(matrix.max() - offset)
        if scale <= 0:
            scale = 1.0
        matrix = (matrix - offset) / scale

    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    rank_used, energy_retained = _effective_rank(
        s, config.rank, config.energy_fraction, mode
    )

    if rank_used == 0:
        reconstructed = np.zeros_like(matrix)
    else:
        reconstructed = (u[:, :rank_used] * s[:rank_used]) @ vt[:rank_used, :]

    if normalize:
        reconstructed = reconstructed * scale + offset

    reconstructed = np.clip(reconstructed, channel.min(), channel.max())

    return ChannelSVDResult(
        name="",
        original=channel,
        reconstructed=reconstructed.astype(channel.dtype),
        singular_values=s,
        rank_used=rank_used,
        energy_retained=energy_retained,
        weight=config.weight,
    )


def compress_multiband(
    bands: dict[str, np.ndarray],
    config: CompressionConfig,
) -> CompressionResult:
    """Compress each named band independently, then stack results."""
    channel_results: list[ChannelSVDResult] = []
    reconstructed_bands: list[np.ndarray] = []

    for name, band in bands.items():
        channel_config = config.channels.get(
            name, ChannelCompressionConfig(rank=min(band.shape))
        )
        result = compress_channel(
            band,
            channel_config,
            config.mode,
            config.normalize_before_svd,
        )
        result.name = name
        channel_results.append(result)
        reconstructed_bands.append(result.reconstructed)

    stacked = np.stack(reconstructed_bands, axis=-1)

    original_bytes = sum(b.nbytes for b in bands.values())
    compressed_bytes = 0
    for result in channel_results:
        h, w = result.original.shape
        k = result.rank_used
        # U: h*k, S: k, Vt: k*w floats
        compressed_bytes += (h * k + k + k * w) * 8

    return CompressionResult(
        channels=channel_results,
        reconstructed_image=stacked,
        total_bytes_original=original_bytes,
        total_bytes_compressed_estimate=compressed_bytes,
    )


def apply_channel_weight_matrix(
    bands: dict[str, np.ndarray],
    weight_matrix: np.ndarray,
    band_order: list[str],
) -> dict[str, np.ndarray]:
    """
    Apply a linear mixing matrix to bands before compression.

    Useful for experimenting with prioritizing spectral combinations
    (e.g., emphasizing NIR-Red contrast relevant to NDVI).
    """
    ordered = [bands[name].astype(np.float64) for name in band_order]
    cube = np.stack(ordered, axis=-1)
    h, w, c = cube.shape
    flat = cube.reshape(-1, c)
    mixed = flat @ weight_matrix.T
    mixed = mixed.reshape(h, w, c)

    return {name: mixed[..., i] for i, name in enumerate(band_order)}


def identity_weight_matrix(n: int) -> np.ndarray:
    return np.eye(n, dtype=np.float64)
