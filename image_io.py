"""Load and export satellite imagery, including images packaged in TAR archives."""

from __future__ import annotations

import tarfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath
from typing import Any, BinaryIO

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
    raster_dataset_tags: dict[str, dict[str, str]] | None = None
    raster_band_tags: list[dict[str, dict[str, str]]] | None = None
    raster_descriptions: tuple[str | None, ...] | None = None
    raster_scales: tuple[float, ...] | None = None
    raster_offsets: tuple[float, ...] | None = None
    raster_units: tuple[str | None, ...] | None = None
    raster_colorinterp: tuple[Any, ...] | None = None
    raster_mask: np.ndarray | None = None
    raster_gcps: tuple[Any, Any] | None = None
    raster_rpcs: Any | None = None


SUPPORTED_IMAGE_SUFFIXES = (
    ".tif",
    ".tiff",
    ".geotiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
    ".jp2",
    ".j2k",
    ".jpx",
)
SUPPORTED_ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz")
# Soft caps for a *selected* member payload (~2 GiB). Listing ignores non-images.
MAX_ARCHIVE_IMAGE_BYTES = 2 * 1024 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 500_000
MAX_DECODED_IMAGE_BYTES = 2 * 1024 * 1024 * 1024


def is_tar_archive(filename: str) -> bool:
    """Return whether a filename represents a supported TAR archive."""
    return filename.lower().endswith(SUPPORTED_ARCHIVE_SUFFIXES)


def _is_junk_archive_path(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    base = parts[-1] if parts else path
    if base == ".DS_Store" or base.startswith("._"):
        return True
    return any(part == "__MACOSX" for part in parts)


def _is_supported_image_path(path: str) -> bool:
    lower = path.lower()
    return (not _is_junk_archive_path(path)) and lower.endswith(SUPPORTED_IMAGE_SUFFIXES)


def _tar_stream_mode(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        return "r|gz"
    return "r|"


def scan_archive_image_entries(
    fileobj: BinaryIO,
    *,
    filename: str = "archive.tar",
) -> list[dict[str, Any]]:
    """
    Stream-scan a TAR for image members.

    Non-image / junk / duplicate / oversized members are skipped — they do not
    fail the archive. Returns dicts with name / offset (data start) / size so
    callers can later fetch a single member with an HTTP Range / GCS ranged
    download. Offsets are only valid for uncompressed .tar (not .tar.gz / .tgz).
    """
    mode = _tar_stream_mode(filename)
    compressed = mode != "r|"
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        with tarfile.open(fileobj=fileobj, mode=mode) as archive:
            for member_count, member in enumerate(archive, start=1):
                if member_count > MAX_ARCHIVE_MEMBERS:
                    raise ValueError("The TAR archive contains too many members to index.")
                if not member.isfile():
                    continue
                if not _is_supported_image_path(member.name):
                    continue
                size = int(member.size or 0)
                if size <= 0 or size > MAX_ARCHIVE_IMAGE_BYTES:
                    continue
                if member.name in seen:
                    continue
                seen.add(member.name)
                entry: dict[str, Any] = {"name": member.name, "size": size}
                # offset_data is reliable for uncompressed streaming TARs.
                if not compressed and getattr(member, "offset_data", None) is not None:
                    entry["offset"] = int(member.offset_data)
                candidates.append(entry)
    except (tarfile.TarError, OSError) as exc:
        raise ValueError("The uploaded file is not a readable TAR archive.") from exc

    if not candidates:
        raise ValueError(
            "The TAR archive does not contain a supported image "
            "(GeoTIFF, PNG, JPEG, WebP, BMP, GIF, or JPEG 2000)."
        )
    return candidates


def list_archive_images(archive_bytes: bytes) -> list[str]:
    """List supported regular image files in a TAR archive without extracting it."""
    entries = scan_archive_image_entries(BytesIO(archive_bytes), filename="archive.tar")
    return sorted(entry["name"] for entry in entries)


def load_archive_image(
    archive_bytes: bytes,
    member_name: str,
) -> LoadedImage:
    """Load one selected image directly from a TAR archive."""
    if not _is_supported_image_path(member_name):
        raise ValueError("The selected archive member is not a supported image.")

    image_bytes: bytes | None = None
    try:
        with tarfile.open(fileobj=BytesIO(archive_bytes), mode="r:*") as archive:
            for member_count, member in enumerate(archive, start=1):
                if member_count > MAX_ARCHIVE_MEMBERS:
                    raise ValueError("The TAR archive contains too many members to index.")
                if not member.isfile():
                    continue
                if member.name != member_name:
                    continue
                if member.size <= 0 or member.size > MAX_ARCHIVE_IMAGE_BYTES:
                    raise ValueError("The selected image is too large to process.")

                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(
                        "The selected image could not be read from the archive."
                    )
                image_bytes = extracted.read(MAX_ARCHIVE_IMAGE_BYTES + 1)
                break  # first match wins
    except (tarfile.TarError, OSError) as exc:
        raise ValueError("The uploaded file is not a readable TAR archive.") from exc

    if image_bytes is None:
        raise ValueError("The selected archive image is missing.")
    if len(image_bytes) > MAX_ARCHIVE_IMAGE_BYTES:
        raise ValueError("The selected image is too large to process.")

    loaded = load_image(BytesIO(image_bytes), PurePosixPath(member_name).name)
    loaded.metadata["archive_member"] = member_name
    return loaded


def _normalize_band_name(index: int, count: int) -> str:
    defaults = {
        1: ["gray"],
        3: ["red", "green", "blue"],
        4: ["red", "green", "blue", "alpha"],
    }
    if count in defaults:
        return defaults[count][index]
    return f"band_{index + 1}"


def load_png(file: BinaryIO) -> LoadedImage:
    image = Image.open(file)
    if image.mode == "P":
        image = image.convert("RGBA" if "transparency" in image.info else "RGB")
    elif image.mode not in ("1", "L", "LA", "I", "I;16", "F", "RGB", "RGBA"):
        image = image.convert("RGB")
    channel_count = max(1, len(image.getbands()))
    estimated_bytes = image.width * image.height * channel_count * 4
    if estimated_bytes > MAX_DECODED_IMAGE_BYTES:
        raise ValueError("The decoded image is too large to process.")
    image.load()
    array = np.asarray(image)

    if array.ndim == 2:
        bands = {"gray": array}
        order = ["gray"]
    elif array.ndim == 3:
        count = array.shape[2]
        pillow_names = {
            "R": "red",
            "G": "green",
            "B": "blue",
            "A": "alpha",
            "L": "gray",
        }
        raw_names = image.getbands()
        order = [
            pillow_names.get(raw_names[i], _normalize_band_name(i, count))
            for i in range(count)
        ]
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
            estimated_bytes = sum(
                dataset.width * dataset.height * np.dtype(dtype).itemsize
                for dtype in dataset.dtypes
            )
            if estimated_bytes > MAX_DECODED_IMAGE_BYTES:
                raise ValueError("The decoded GeoTIFF is too large to process.")
            array = dataset.read()
            descriptions = dataset.descriptions
            profile = dataset.profile.copy()
            crs = str(dataset.crs) if dataset.crs else None
            transform = tuple(dataset.transform)
            dataset_tags = _read_raster_tags(dataset)
            band_tags = [
                _read_raster_tags(dataset, band_index)
                for band_index in range(1, dataset.count + 1)
            ]
            scales = dataset.scales
            offsets = dataset.offsets
            units = dataset.units
            colorinterp = dataset.colorinterp
            mask = dataset.dataset_mask()
            gcps = dataset.gcps
            rpcs = dataset.rpcs

    bands: dict[str, np.ndarray] = {}
    order: list[str] = []

    for i, desc in enumerate(descriptions):
        name = (
            desc.strip().lower().replace(" ", "_")
            if desc
            else f"band_{i + 1}"
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
        raster_dataset_tags=dataset_tags,
        raster_band_tags=band_tags,
        raster_descriptions=descriptions,
        raster_scales=scales,
        raster_offsets=offsets,
        raster_units=units,
        raster_colorinterp=colorinterp,
        raster_mask=mask,
        raster_gcps=gcps,
        raster_rpcs=rpcs,
    )


def _read_raster_tags(dataset, band_index: int = 0) -> dict[str, dict[str, str]]:
    """Collect default and namespaced GeoTIFF tags."""
    result = {"": dataset.tags(band_index)}
    for namespace in dataset.tag_namespaces(band_index):
        result[namespace] = dataset.tags(band_index, ns=namespace)
    return result


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
                    original_description = (
                        loaded.raster_descriptions[index - 1]
                        if loaded.raster_descriptions
                        else None
                    )
                    dataset.set_band_description(
                        index, original_description or name
                    )
                if loaded.raster_scales is not None:
                    dataset.scales = loaded.raster_scales
                if loaded.raster_offsets is not None:
                    dataset.offsets = loaded.raster_offsets
                if loaded.raster_units is not None:
                    dataset.units = loaded.raster_units
                if loaded.raster_colorinterp is not None:
                    dataset.colorinterp = loaded.raster_colorinterp
                if loaded.raster_mask is not None:
                    dataset.write_mask(loaded.raster_mask)
                if loaded.raster_gcps is not None and loaded.raster_gcps[0]:
                    dataset.gcps = loaded.raster_gcps
                if loaded.raster_rpcs is not None:
                    dataset.rpcs = loaded.raster_rpcs
                _write_raster_tags(dataset, loaded)
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


def _write_raster_tags(dataset, loaded: LoadedImage) -> None:
    """Restore dataset-level and per-band GeoTIFF tags."""
    for namespace, tags in (loaded.raster_dataset_tags or {}).items():
        if namespace:
            dataset.update_tags(ns=namespace, **tags)
        else:
            dataset.update_tags(**tags)
    for band_index, tag_groups in enumerate(loaded.raster_band_tags or [], start=1):
        for namespace, tags in tag_groups.items():
            if namespace:
                dataset.update_tags(band_index, ns=namespace, **tags)
            else:
                dataset.update_tags(band_index, **tags)
