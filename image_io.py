"""Load and export satellite imagery, including images packaged in TAR archives."""

from __future__ import annotations

import tarfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath
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
    raster_profile: dict | None = None


SUPPORTED_IMAGE_SUFFIXES = (".tif", ".tiff", ".geotiff", ".png", ".jpg", ".jpeg", ".webp")
SUPPORTED_ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz")
MAX_ARCHIVE_IMAGE_BYTES = 512 * 1024 * 1024


def is_tar_archive(filename: str) -> bool:
    """Return whether a filename represents a supported TAR archive."""
    return filename.lower().endswith(SUPPORTED_ARCHIVE_SUFFIXES)


def list_archive_images(archive_bytes: bytes) -> list[str]:
    """List supported regular image files in a TAR archive without extracting it."""
    try:
        with tarfile.open(fileobj=BytesIO(archive_bytes), mode="r:*") as archive:
            candidates = [
                member.name
                for member in archive.getmembers()
                if member.isfile()
                and member.name.lower().endswith(SUPPORTED_IMAGE_SUFFIXES)
                and 0 < member.size <= MAX_ARCHIVE_IMAGE_BYTES
            ]
    except (tarfile.TarError, OSError) as exc:
        raise ValueError("The uploaded file is not a readable TAR archive.") from exc

    if not candidates:
        raise ValueError(
            "The TAR archive does not contain a supported image "
            "(GeoTIFF, PNG, JPEG, or WebP)."
        )
    return candidates


def load_archive_image(
    archive_bytes: bytes,
    member_name: str,
) -> LoadedImage:
    """Load one selected image directly from a TAR archive."""
    try:
        with tarfile.open(fileobj=BytesIO(archive_bytes), mode="r:*") as archive:
            matches = [
                member
                for member in archive.getmembers()
                if member.name == member_name and member.isfile()
            ]
            if len(matches) != 1:
                raise ValueError("The selected archive image is missing or ambiguous.")

            member = matches[0]
            if (
                member.size <= 0
                or member.size > MAX_ARCHIVE_IMAGE_BYTES
                or not member.name.lower().endswith(SUPPORTED_IMAGE_SUFFIXES)
            ):
                raise ValueError("The selected archive member is not a supported image.")

            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError("The selected image could not be read from the archive.")
            image_bytes = extracted.read(MAX_ARCHIVE_IMAGE_BYTES + 1)
    except (tarfile.TarError, OSError) as exc:
        raise ValueError("The uploaded file is not a readable TAR archive.") from exc

    if len(image_bytes) > MAX_ARCHIVE_IMAGE_BYTES:
        raise ValueError("The selected image is too large to process.")

    loaded = load_image(BytesIO(image_bytes), PurePosixPath(member_name).name)
    loaded.metadata["archive_member"] = member_name
    return loaded


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
            descriptions = dataset.descriptions
            profile = dataset.profile.copy()
            crs = str(dataset.crs) if dataset.crs else None
            transform = tuple(dataset.transform)

    bands: dict[str, np.ndarray] = {}
    order: list[str] = []

    for i, desc in enumerate(descriptions):
        name = (
            desc.strip().lower().replace(" ", "_")
            if desc
            else _normalize_band_name(i, len(descriptions))
        )
        if name in bands:
            name = f"{name}_{i + 1}"
        bands[name] = array[i]
        order.append(name)

    _apply_common_band_aliases(bands, order)

    return LoadedImage(
        bands=bands,
        band_order=order,
        source_type="geotiff",
        metadata={
            "band_count": len(order),
            "width": int(array.shape[2]),
            "height": int(array.shape[1]),
            "dtype": str(array.dtype),
            "crs": crs,
            "transform": transform,
        },
        raster_profile=profile,
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
    for index, name in enumerate(list(order)):
        alias = alias_map.get(name)
        if alias and alias != name and alias not in bands:
            bands[alias] = bands.pop(name)
            order[index] = alias


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


def encode_reconstructed_image(
    loaded: LoadedImage,
    reconstructed_bands: dict[str, np.ndarray],
) -> tuple[bytes, str, str]:
    """
    Encode reconstructed pixels as a downloadable image.

    GeoTIFF inputs retain their raster profile and are stored with lossless DEFLATE.
    Other inputs are exported as lossless PNG files.
    """
    ordered = [reconstructed_bands[name] for name in loaded.band_order]

    if loaded.source_type == "geotiff":
        if not HAS_RASTERIO:
            raise RuntimeError("rasterio is required to export GeoTIFF results")
        if loaded.raster_profile is None:
            raise ValueError("The source GeoTIFF profile is unavailable.")

        profile = loaded.raster_profile.copy()
        profile.update(
            driver="GTiff",
            count=len(ordered),
            height=ordered[0].shape[0],
            width=ordered[0].shape[1],
            dtype=ordered[0].dtype,
            compress="deflate",
        )
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.pop("tiled", None)

        with MemoryFile() as memfile:
            with memfile.open(**profile) as dataset:
                dataset.write(np.stack(ordered, axis=0))
                for index, name in enumerate(loaded.band_order, start=1):
                    dataset.set_band_description(index, name)
            output = memfile.read()
        return output, "compressed_image.tif", "image/tiff"

    if len(ordered) == 1:
        array = ordered[0]
    else:
        array = np.stack(ordered, axis=-1)
    if array.dtype != np.uint8 and not (array.ndim == 2 and array.dtype == np.uint16):
        raise ValueError(
            "This image's pixel type cannot be exported as PNG; use GeoTIFF input "
            "for high-bit-depth or multiband imagery."
        )
    return array_to_png_bytes(array), "compressed_image.png", "image/png"
