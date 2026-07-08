"""Load satellite imagery from PNG and GeoTIFF sources."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import BinaryIO

import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.io import MemoryFile

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


@dataclass
class LoadedImage:
    bands: dict[str, np.ndarray]
    band_order: list[str]
    source_type: str
    metadata: dict


def _normalize_band_name(index: int, count: int) -> str:
    defaults = {
        1: ["gray"],
        3: ["red", "green", "blue"],
        4: ["blue", "green", "red", "nir"],
    }
    if count in defaults:
        return defaults[count][index]
    return f"band_{index + 1}"


def load_png(file: BinaryIO) -> LoadedImage:
    image = Image.open(file)
    array = np.asarray(image)

    if array.ndim == 2:
        bands = {"gray": array}
        order = ["gray"]
    elif array.ndim == 3:
        count = array.shape[2]
        order = [_normalize_band_name(i, count) for i in range(count)]
        bands = {name: array[..., i] for i, name in enumerate(order)}
    else:
        raise ValueError("Unsupported PNG shape")

    return LoadedImage(
        bands=bands,
        band_order=order,
        source_type="png",
        metadata={"mode": image.mode, "size": image.size},
    )


def load_geotiff(file: BinaryIO) -> LoadedImage:
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio is required for GeoTIFF uploads")

    data = file.read()
    with MemoryFile(data) as memfile:
        with memfile.open() as dataset:
            array = dataset.read()
            descriptions = dataset.descriptions or tuple(
                f"band_{i + 1}" for i in range(dataset.count)
            )

    bands: dict[str, np.ndarray] = {}
    order: list[str] = []

    for i, desc in enumerate(descriptions):
        name = (desc or f"band_{i + 1}").strip().lower().replace(" ", "_")
        if name in bands:
            name = f"{name}_{i + 1}"
        bands[name] = array[i]
        order.append(name)

    _apply_common_band_aliases(bands, order)

    return LoadedImage(
        bands=bands,
        band_order=order,
        source_type="geotiff",
        metadata={"band_count": len(order)},
    )


def _apply_common_band_aliases(bands: dict[str, np.ndarray], order: list[str]) -> None:
    """Map common satellite band names to red/nir when possible."""
    alias_map = {
        "b4": "red",
        "red": "red",
        "r": "red",
        "b8": "nir",
        "nir": "nir",
        "near_infrared": "nir",
    }
    for key, alias in alias_map.items():
        if key in bands and alias not in bands:
            bands[alias] = bands[key]
            if alias not in order:
                order.append(alias)


def load_image(file: BinaryIO, filename: str) -> LoadedImage:
    lower = filename.lower()
    if lower.endswith((".tif", ".tiff", ".geotiff")):
        return load_geotiff(file)
    if lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return load_png(file)
    raise ValueError(f"Unsupported file type: {filename}")


def to_display_rgb(bands: dict[str, np.ndarray], order: list[str]) -> np.ndarray:
    """Build an RGB preview from available bands."""
    if all(c in bands for c in ("red", "green", "blue")):
        channels = [bands["red"], bands["green"], bands["blue"]]
    elif len(order) >= 3:
        channels = [bands[order[i]] for i in range(3)]
    elif "gray" in bands:
        gray = bands["gray"]
        channels = [gray, gray, gray]
    else:
        first = bands[order[0]]
        channels = [first, first, first]

    stacked = np.stack(channels, axis=-1).astype(np.float64)
    for i in range(3):
        channel = stacked[..., i]
        lo, hi = np.percentile(channel, (2, 98))
        if hi > lo:
            stacked[..., i] = np.clip((channel - lo) / (hi - lo), 0, 1)
        else:
            stacked[..., i] = 0.0
    return (stacked * 255).astype(np.uint8)


def array_to_png_bytes(array: np.ndarray) -> bytes:
    if array.ndim == 2:
        image = Image.fromarray(array)
    else:
        image = Image.fromarray(array.astype(np.uint8))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
