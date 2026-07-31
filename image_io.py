"""Load and export satellite imagery, including images packaged in TAR archives."""

from __future__ import annotations

import tarfile
import zipfile
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
    ".gtiff",
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
SUPPORTED_ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".zip")
TAR_ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz")
ZIP_ARCHIVE_SUFFIXES = (".zip",)
GEO_TIFF_SUFFIXES = (".tif", ".tiff", ".geotiff", ".gtiff")
JPEG2000_SUFFIXES = (".jp2", ".j2k", ".jpx")
RASTER_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
SUPPORTED_IMAGE_LABEL = (
    ".tif / .tiff / .geotiff / .png / .jpg / .jpeg / .webp / .bmp / .gif / .jp2"
)
SUPPORTED_ARCHIVE_LABEL = ".tar / .tar.gz / .tgz / .zip"
# Soft caps for a *selected* member payload (~2 GiB). Listing ignores non-images.
MAX_ARCHIVE_IMAGE_BYTES = 2 * 1024 * 1024 * 1024
# Soft ceiling only — real archives stop earlier via tarfile iteration.
MAX_ARCHIVE_MEMBERS = 2_000_000
MAX_DECODED_IMAGE_BYTES = 2 * 1024 * 1024 * 1024


def is_tar_archive(filename: str) -> bool:
    """Return whether a filename represents a TAR / TAR.GZ / ZIP archive."""
    return filename.lower().endswith(SUPPORTED_ARCHIVE_SUFFIXES)


def is_zip_archive(filename: str) -> bool:
    return filename.lower().endswith(ZIP_ARCHIVE_SUFFIXES)


def is_plain_tar_archive(filename: str) -> bool:
    return filename.lower().endswith(TAR_ARCHIVE_SUFFIXES)


def _is_junk_archive_path(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    base = parts[-1] if parts else path
    if base == ".DS_Store" or base.startswith("._"):
        return True
    return any(part == "__MACOSX" for part in parts)


def _is_supported_image_path(path: str) -> bool:
    lower = _normalize_archive_path(path).lower()
    return (not _is_junk_archive_path(path)) and lower.endswith(SUPPORTED_IMAGE_SUFFIXES)


def _looks_like_tiff(head: bytes) -> bool:
    if len(head) < 4:
        return False
    return head[:4] in (
        b"II*\x00",
        b"MM\x00*",
        b"II+\x00",
        b"MM\x00+",
    )


def _looks_like_jpeg2000(head: bytes) -> bool:
    if len(head) >= 8 and head[4:8] == b"jP  ":
        return True
    return len(head) >= 4 and head[:4] == b"\xff\x4f\xff\x51"


def _peek_head(file: BinaryIO, size: int = 16) -> bytes:
    pos = file.tell()
    try:
        return file.read(size) or b""
    finally:
        file.seek(pos)


def _folder_prefixes_for_path(path: str) -> list[str]:
    parts = [p for p in path.replace("\\", "/").split("/") if p]
    if len(parts) <= 1:
        return []
    return ["/".join(parts[:i]) for i in range(1, len(parts))]


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
                    raise ValueError(
                        "The TAR archive contains too many members to index."
                    )
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
            f"({SUPPORTED_IMAGE_LABEL})."
        )
    return candidates


def _normalize_archive_path(path: str) -> str:
    p = path.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    while "//" in p:
        p = p.replace("//", "/")
    return p


def _looks_like_zip(head: bytes) -> bool:
    return len(head) >= 4 and head[0] == 0x50 and head[1] == 0x4B and head[2] in (0x03, 0x05, 0x07)


def _is_zip_bytes(archive_bytes: bytes, filename: str = "") -> bool:
    return is_zip_archive(filename) or _looks_like_zip(archive_bytes[:4])


def scan_zip_image_entries(archive_bytes: bytes) -> list[dict[str, Any]]:
    """Index image members in a ZIP without extracting payloads."""
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
            for member_count, info in enumerate(archive.infolist(), start=1):
                if member_count > MAX_ARCHIVE_MEMBERS:
                    raise ValueError("The ZIP archive contains too many members to index.")
                if info.is_dir():
                    continue
                name = _normalize_archive_path(info.filename)
                if not _is_supported_image_path(name):
                    continue
                size = int(info.file_size or 0)
                if size <= 0 or size > MAX_ARCHIVE_IMAGE_BYTES:
                    continue
                if name in seen:
                    continue
                seen.add(name)
                candidates.append({"name": name, "size": size})
    except (zipfile.BadZipFile, OSError) as exc:
        raise ValueError("The uploaded file is not a readable ZIP archive.") from exc

    if not candidates:
        raise ValueError(
            "The ZIP archive does not contain a supported image "
            f"({SUPPORTED_IMAGE_LABEL})."
        )
    return candidates


def list_archive_listing(
    archive_bytes: bytes,
    *,
    filename: str = "archive.tar",
) -> dict[str, list[str]]:
    """List image paths and folder prefixes inside a TAR / TAR.GZ / TGZ / ZIP."""
    name = filename or "archive.tar"
    if _is_zip_bytes(archive_bytes, name):
        entries = scan_zip_image_entries(archive_bytes)
    else:
        # Detect gzip even when the caller passes a bare .tar name.
        if (
            len(archive_bytes) >= 2
            and archive_bytes[0] == 0x1F
            and archive_bytes[1] == 0x8B
            and not name.lower().endswith((".gz", ".tgz"))
        ):
            name = "archive.tar.gz"
        entries = scan_archive_image_entries(BytesIO(archive_bytes), filename=name)
    images = sorted(_normalize_archive_path(entry["name"]) for entry in entries)
    folders: set[str] = set()
    for image_name in images:
        folders.update(_folder_prefixes_for_path(image_name))
    return {"images": images, "folders": sorted(folders)}


def list_archive_images(archive_bytes: bytes, *, filename: str = "archive.tar") -> list[str]:
    """List supported regular image files in an archive without extracting it."""
    return list_archive_listing(archive_bytes, filename=filename)["images"]


def load_archive_image(
    archive_bytes: bytes,
    member_name: str,
) -> LoadedImage:
    """Load one selected image directly from a TAR or ZIP archive."""
    want = _normalize_archive_path(member_name)
    if not _is_supported_image_path(want):
        raise ValueError("The selected archive member is not a supported image.")

    image_bytes: bytes | None = None
    if _is_zip_bytes(archive_bytes, ""):
        try:
            with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
                match = None
                for info in archive.infolist():
                    if _normalize_archive_path(info.filename) == want and not info.is_dir():
                        match = info
                        break
                if match is None:
                    raise ValueError("The selected archive image is missing.")
                if match.file_size <= 0 or match.file_size > MAX_ARCHIVE_IMAGE_BYTES:
                    raise ValueError("The selected image is too large to process.")
                image_bytes = archive.read(match)[: MAX_ARCHIVE_IMAGE_BYTES + 1]
        except (zipfile.BadZipFile, OSError) as exc:
            raise ValueError("The uploaded file is not a readable ZIP archive.") from exc
    else:
        try:
            with tarfile.open(fileobj=BytesIO(archive_bytes), mode="r:*") as archive:
                match = None
                for member in archive.getmembers():
                    if _normalize_archive_path(member.name) == want and member.isfile():
                        match = member
                        break
                if match is None:
                    raise ValueError("The selected archive image is missing.")
                if match.size <= 0 or match.size > MAX_ARCHIVE_IMAGE_BYTES:
                    raise ValueError("The selected image is too large to process.")
                extracted = archive.extractfile(match)
                if extracted is None:
                    raise ValueError(
                        "The selected image could not be read from the archive."
                    )
                image_bytes = extracted.read(MAX_ARCHIVE_IMAGE_BYTES + 1)
        except (tarfile.TarError, OSError) as exc:
            raise ValueError("The uploaded file is not a readable TAR archive.") from exc

    if image_bytes is None:
        raise ValueError("The selected archive image is missing.")
    if len(image_bytes) > MAX_ARCHIVE_IMAGE_BYTES:
        raise ValueError("The selected image is too large to process.")

    loaded = load_image(BytesIO(image_bytes), PurePosixPath(want).name)
    loaded.metadata["archive_member"] = want
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
    """Load a Pillow-decodable raster (PNG / JPEG / WebP / BMP / GIF / …)."""
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
    """Load an image by extension and/or magic bytes (TIFF / JP2 / raster)."""
    lower = filename.lower()
    head = _peek_head(file)
    named_tiff = lower.endswith(GEO_TIFF_SUFFIXES)
    named_jp2 = lower.endswith(JPEG2000_SUFFIXES)
    named_raster = lower.endswith(RASTER_SUFFIXES)
    looks_tiff = _looks_like_tiff(head)
    looks_jp2 = _looks_like_jpeg2000(head)

    if named_tiff or looks_tiff:
        try:
            return load_geotiff(file)
        except Exception as geotiff_exc:
            if looks_tiff:
                raise
            # Extension says .TIF but payload isn't TIFF (common in test zips /
            # mislabeled exports) — fall back to Pillow.
            file.seek(0)
            try:
                return load_png(file)
            except Exception:
                raise geotiff_exc from None

    if named_jp2 or looks_jp2:
        # Prefer GDAL/rasterio when available (JP2OpenJPEG), else Pillow.
        if HAS_RASTERIO:
            try:
                pos = file.tell()
                loaded = load_geotiff(file)
                loaded.metadata = {**loaded.metadata, "format_hint": "jpeg2000"}
                return loaded
            except Exception:
                file.seek(pos)
        try:
            return load_png(file)
        except Exception as exc:
            raise ValueError(
                f"Could not decode JPEG 2000 '{filename}'. "
                "Install a GDAL build with JP2 support, or convert to GeoTIFF/PNG."
            ) from exc

    if named_raster or lower.endswith(SUPPORTED_IMAGE_SUFFIXES):
        return load_png(file)

    raise ValueError(
        f"Unsupported file type: {filename}. Supported images: {SUPPORTED_IMAGE_LABEL}."
    )


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

    GeoTIFF inputs keep CRS / transform / tags and are written as float32
    GeoTIFFs (lossless DEFLATE) so analysis keeps full reconstructed precision
    without re-quantizing to uint8/uint16. Other inputs export as a float32
    GeoTIFF without georeferencing when rasterio is available; otherwise PNG
    when the pixel type is display-compatible.
    """
    ordered = [reconstructed_bands[name] for name in loaded.band_order]

    if loaded.source_type == "geotiff":
        if not HAS_RASTERIO:
            raise RuntimeError("rasterio is required to export GeoTIFF results")
        if loaded.raster_profile is None:
            raise ValueError("The source GeoTIFF profile is unavailable.")

        # Always float32 for analysis — reconstructed values are already continuous.
        float_bands = [np.asarray(band, dtype=np.float32) for band in ordered]
        height, width = float_bands[0].shape[:2]

        profile = loaded.raster_profile.copy()
        profile.update(
            driver="GTiff",
            count=len(float_bands),
            height=height,
            width=width,
            dtype="float32",
            compress="deflate",
        )
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.pop("tiled", None)
        # nodata from integer sources may not apply cleanly to float reconstructions.
        profile.pop("nodata", None)

        with MemoryFile() as memfile:
            with memfile.open(**profile) as dataset:
                dataset.write(np.stack(float_bands, axis=0))
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
                    mask = loaded.raster_mask
                    if mask.shape[0] != height or mask.shape[1] != width:
                        # Native-size restore can change dims; skip mismatched masks.
                        mask = None
                    if mask is not None:
                        dataset.write_mask(mask)
                if loaded.raster_gcps is not None and loaded.raster_gcps[0]:
                    dataset.gcps = loaded.raster_gcps
                if loaded.raster_rpcs is not None:
                    dataset.rpcs = loaded.raster_rpcs
                _write_raster_tags(dataset, loaded)
            output = memfile.read()
        return output, "compressed_image.tif", "image/tiff"

    # Non-GeoTIFF sources: still prefer float32 GeoTIFF for analysis downloads.
    if HAS_RASTERIO:
        float_bands = [np.asarray(band, dtype=np.float32) for band in ordered]
        height, width = float_bands[0].shape[:2]
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": len(float_bands),
            "dtype": "float32",
            "compress": "deflate",
        }
        with MemoryFile() as memfile:
            with memfile.open(**profile) as dataset:
                dataset.write(np.stack(float_bands, axis=0))
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
