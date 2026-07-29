"""LZW compression for scientific image bands.

Adapted from the ashmeet13/LZW-Image-Compression approach (and the ESMI
``lzw_compress`` TAR-aware CLI), but integrated into the lab pipeline:

  - Works on arbitrary float bands (not only RGB files)
  - Quantizes each band to uint8 (min–max), runs classic byte-oriented LZW,
    then dequantizes — same Lempel–Ziv–Welch idea as the reference, without
    the digit/comma string encoding that balloons size and runtime
  - Reports compressed size from the packed LZW code stream

Reference: https://github.com/ashmeet13/LZW-Image-Compression
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from compression.base import CompressionExecutionResult, build_execution_result

# Cap dictionary growth (12-bit codes) — same spirit as GIF/TIFF LZW.
MAX_DICT_SIZE = 4096


def lzw_compress_bytes(data: bytes) -> list[int]:
    """Classic LZW over a byte stream → list of integer codes."""
    if not data:
        return []
    dictionary: dict[bytes, int] = {bytes([i]): i for i in range(256)}
    dict_size = 256
    w = b""
    out: list[int] = []
    for byte in data:
        c = bytes([byte])
        wc = w + c
        if wc in dictionary:
            w = wc
            continue
        out.append(dictionary[w])
        if dict_size < MAX_DICT_SIZE:
            dictionary[wc] = dict_size
            dict_size += 1
        w = c
    if w:
        out.append(dictionary[w])
    return out


def lzw_decompress_codes(codes: list[int]) -> bytes:
    """Inverse of :func:`lzw_compress_bytes`."""
    if not codes:
        return b""
    dictionary: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    dict_size = 256
    w = dictionary[codes[0]]
    out = bytearray(w)
    for code in codes[1:]:
        if code in dictionary:
            entry = dictionary[code]
        elif code == dict_size:
            entry = w + w[:1]
        else:
            raise ValueError(f"Invalid LZW code: {code}")
        out.extend(entry)
        if dict_size < MAX_DICT_SIZE:
            dictionary[dict_size] = w + entry[:1]
            dict_size += 1
        w = entry
    return bytes(out)


def _quantize_uint8(band: np.ndarray) -> tuple[np.ndarray, float, float]:
    band_f = band.astype(np.float64, copy=False)
    lo = float(np.min(band_f))
    hi = float(np.max(band_f))
    scale = hi - lo
    if scale <= 0:
        return np.zeros(band.shape, dtype=np.uint8), lo, hi
    u8 = np.clip(np.round((band_f - lo) / scale * 255.0), 0, 255).astype(np.uint8)
    return u8, lo, hi


def _dequantize_uint8(u8: np.ndarray, lo: float, hi: float, dtype) -> np.ndarray:
    scale = hi - lo
    if scale <= 0:
        return np.full(u8.shape, lo, dtype=dtype)
    return (u8.astype(np.float64) / 255.0 * scale + lo).astype(dtype, copy=False)


def compress_band_lzw(band: np.ndarray) -> tuple[np.ndarray, int, dict]:
    """Quantize → LZW → dequantize. Returns reconstructed band, compressed bytes, meta."""
    u8, lo, hi = _quantize_uint8(band)
    codes = lzw_compress_bytes(u8.tobytes())
    # Pack codes as 16-bit words (enough for 12-bit LZW codes).
    compressed_bytes = len(codes) * 2
    decoded = lzw_decompress_codes(codes)
    if len(decoded) != u8.size:
        raise ValueError(
            f"LZW round-trip size mismatch: expected {u8.size} bytes, got {len(decoded)}"
        )
    u8_out = np.frombuffer(decoded, dtype=np.uint8).reshape(u8.shape)
    reconstructed = _dequantize_uint8(u8_out, lo, hi, band.dtype)
    return reconstructed, compressed_bytes, {"lo": lo, "hi": hi, "codes": len(codes)}


def run_lzw_compression(bands: dict[str, np.ndarray]) -> CompressionExecutionResult:
    """Compress each band independently with LZW after uint8 quantization."""
    start = perf_counter()
    reconstructed_bands: dict[str, np.ndarray] = {}
    compressed_total = 0
    band_meta: dict[str, dict] = {}

    for name, band in bands.items():
        reconstructed, nbytes, meta = compress_band_lzw(band)
        reconstructed_bands[name] = reconstructed
        compressed_total += nbytes
        band_meta[name] = meta

    return build_execution_result(
        method="lzw",
        bands=bands,
        reconstructed_bands=reconstructed_bands,
        compressed_bytes_estimate=compressed_total,
        runtime_seconds=perf_counter() - start,
        metadata={
            "algorithm": "LZW",
            "quantization": "uint8-minmax",
            "max_dict_size": MAX_DICT_SIZE,
            "reference": "ashmeet13/LZW-Image-Compression (adapted for float bands)",
            "bands": band_meta,
        },
    )
