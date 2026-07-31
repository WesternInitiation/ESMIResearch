"""Load and export satellite imagery, including images packaged in TAR archives."""

from __future__ import annotations

import re
import tarfile
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath
from typing import Any, BinaryIO, Literal

import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.enums import Resampling as RioResampling
    from rasterio.io import MemoryFile
    from rasterio.transform import Affine

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    RioResampling = None  # type: ignore[misc, assignment]
    Affine = None  # type: ignore[misc, assignment]


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
            band_count = int(dataset.count)
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
            else ("gray" if band_count == 1 else f"band_{i + 1}")
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


def load_geotiff_preview(
    data: bytes,
    *,
    max_dim: int = 1024,
) -> tuple[dict[str, np.ndarray], list[str], int, int, int, int]:
    """
    Decode a GeoTIFF only at preview resolution via rasterio ``out_shape``.

    Returns ``(bands, band_order, native_w, native_h, preview_w, preview_h)``.
    Much faster/cheaper than ``load_geotiff`` + downsample for UI previews.
    """
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio is required for GeoTIFF previews")
    from rasterio.enums import Resampling

    cap = max(64, int(max_dim)) if int(max_dim) > 0 else 1024
    with MemoryFile(data) as memfile:
        with memfile.open() as dataset:
            native_w = int(dataset.width)
            native_h = int(dataset.height)
            longest = max(native_w, native_h)
            if longest > cap:
                scale = cap / float(longest)
                preview_w = max(1, int(round(native_w * scale)))
                preview_h = max(1, int(round(native_h * scale)))
            else:
                preview_w, preview_h = native_w, native_h
            array = dataset.read(
                out_shape=(dataset.count, preview_h, preview_w),
                resampling=Resampling.bilinear,
            )
            descriptions = dataset.descriptions

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
    return bands, order, native_w, native_h, preview_w, preview_h


def preview_raster_bytes(
    data: bytes,
    filename: str,
    *,
    max_dim: int = 1024,
) -> tuple[dict[str, np.ndarray], list[str], int, int, int, int]:
    """Load preview-sized bands from image bytes (GeoTIFF preferred)."""
    lower = filename.lower()
    if lower.endswith(GEO_TIFF_SUFFIXES) or _looks_like_tiff(data[:16]):
        try:
            return load_geotiff_preview(data, max_dim=max_dim)
        except Exception:
            # Fall through to full load + caller can downsample.
            pass
    loaded = load_image(BytesIO(data), filename)
    sample = next(iter(loaded.bands.values()))
    native_h, native_w = int(sample.shape[0]), int(sample.shape[1])
    return loaded.bands, loaded.band_order, native_w, native_h, native_w, native_h


def _read_raster_tags(dataset, band_index: int = 0) -> dict[str, dict[str, str]]:
    """Collect default and namespaced GeoTIFF tags."""
    result = {"": dataset.tags(band_index)}
    for namespace in dataset.tag_namespaces(band_index):
        result[namespace] = dataset.tags(band_index, ns=namespace)
    return result


def _apply_common_band_aliases(bands: dict[str, np.ndarray], order: list[str]) -> None:
    """Map common satellite band names to red/nir when possible."""
    alias_map = {
        "b2": "blue",
        "b3": "green",
        "b4": "red",
        "red": "red",
        "r": "red",
        "b5": "nir",
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
    """Encode an 8-bit display preview as PNG (UI only — not scientific export)."""
    if array.ndim == 2:
        image = Image.fromarray(array)
    else:
        image = Image.fromarray(array.astype(np.uint8))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


_LANDSAT_BAND_RE = re.compile(
    r"(?:^|[_./\\-])(?:SR_|ST_|TOA_|BND_)?B0*(\d{1,2})(?:\b|[_./\\-]|$)",
    re.IGNORECASE,
)

# OLI / MSI role → Landsat 8-style labels (B2–B5 are the core optical set).
_ROLE_TO_LANDSAT = {
    "blue": "B2",
    "coastal": "B1",
    "coastal_aerosol": "B1",
    "green": "B3",
    "red": "B4",
    "nir": "B5",
    "near_infrared": "B5",
    "swir1": "B6",
    "swir_1": "B6",
    "swir2": "B7",
    "swir_2": "B7",
    "cirrus": "B9",
    "tirs1": "B10",
    "tirs2": "B11",
}


def landsat_band_label(
    band_key: str,
    *,
    description: str | None = None,
    source_filename: str | None = None,
    band_index: int = 1,
) -> str:
    """
    Prefer Landsat-style labels (B2–B5, …) from description, key, or filename.
    Falls back to role aliases, then ``B{index}``.
    """
    for text in (description, band_key, source_filename):
        if not text:
            continue
        match = _LANDSAT_BAND_RE.search(str(text).replace(" ", "_"))
        if match:
            return f"B{int(match.group(1))}"
    role = _ROLE_TO_LANDSAT.get(str(band_key).strip().lower())
    if role:
        return role
    return f"B{max(1, int(band_index))}"


def resize_band_rasterio(
    band: np.ndarray,
    out_height: int,
    out_width: int,
    *,
    resampling: str = "bilinear",
) -> np.ndarray:
    """
    Resize a 2D scientific band with rasterio (``out_shape``), not Pillow.

    Pillow strips geospatial context and is the wrong tool for satellite rasters.
    This keeps dtype semantics suitable for compression pipelines.
    """
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio is required to resize satellite bands")
    out_h = max(1, int(out_height))
    out_w = max(1, int(out_width))
    src_h, src_w = int(band.shape[0]), int(band.shape[1])
    if src_h == out_h and src_w == out_w:
        return band
    method = getattr(RioResampling, resampling, RioResampling.bilinear)
    # Write a tiny in-memory GeoTIFF then read at the target shape. Using a
    # unit affine keeps pixel alignment consistent for process/preview sizes;
    # final scientific exports always restore the source CRS + transform.
    profile = {
        "driver": "GTiff",
        "height": src_h,
        "width": src_w,
        "count": 1,
        "dtype": "float32",
        "crs": None,
        "transform": Affine.identity(),
    }
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(np.asarray(band, dtype=np.float32), 1)
        with memfile.open() as dataset:
            resized = dataset.read(
                1,
                out_shape=(out_h, out_w),
                resampling=method,
            )
    return np.asarray(resized, dtype=np.float32)


def resize_bands_rasterio(
    bands: dict[str, np.ndarray],
    out_height: int,
    out_width: int,
    *,
    resampling: str = "bilinear",
) -> dict[str, np.ndarray]:
    """Resize every band with rasterio (scientific path — not Pillow)."""
    return {
        name: resize_band_rasterio(
            band, out_height, out_width, resampling=resampling
        )
        for name, band in bands.items()
    }


def downsample_bands_rasterio(
    bands: dict[str, np.ndarray],
    max_dim: int,
) -> tuple[dict[str, np.ndarray], float]:
    """Downsample scientific bands when the longest side exceeds ``max_dim``."""
    sample = next(iter(bands.values()))
    height, width = int(sample.shape[0]), int(sample.shape[1])
    longest = max(height, width)
    if max_dim <= 0 or longest <= max_dim:
        return bands, 1.0
    scale = max_dim / float(longest)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return resize_bands_rasterio(bands, new_h, new_w), scale


def _cast_scientific_band(
    original: np.ndarray,
    reconstructed: np.ndarray,
    *,
    dtype_mode: Literal["float32", "source"] = "float32",
) -> np.ndarray:
    """
    Cast reconstructed pixels for GeoTIFF export.

    Default is float32 (scientific). When ``dtype_mode='source'`` and the
    original band was integer (e.g. Landsat uint16 DN), round/clip to that dtype.
    """
    arr = np.asarray(reconstructed, dtype=np.float64)
    if dtype_mode == "source" and np.issubdtype(original.dtype, np.integer):
        info = np.iinfo(original.dtype)
        return np.clip(np.rint(arr), info.min, info.max).astype(original.dtype)
    if dtype_mode == "source" and original.dtype == np.float32:
        return arr.astype(np.float32)
    return arr.astype(np.float32)


def _single_band_export_profile(
    loaded: LoadedImage,
    *,
    height: int,
    width: int,
    dtype: np.dtype,
) -> dict[str, Any]:
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio is required to export GeoTIFF results")
    if loaded.raster_profile is not None:
        profile = loaded.raster_profile.copy()
    else:
        profile = {
            "driver": "GTiff",
            "crs": None,
            "transform": Affine.identity() if Affine is not None else None,
            "nodata": None,
        }
    profile.update(
        driver="GTiff",
        count=1,
        height=int(height),
        width=int(width),
        dtype=dtype,
        compress="deflate",
        photometric="minisblack",
    )
    # Strip tiling so GDAL writes a simple strip layout at the native grid.
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.pop("tiled", None)
    profile.pop("interleave", None)
    return profile


def encode_reconstructed_band_geotiffs(
    loaded: LoadedImage,
    reconstructed_bands: dict[str, np.ndarray],
    *,
    source_filename: str | None = None,
    dtype_mode: Literal["float32", "source"] = "float32",
) -> list[dict[str, Any]]:
    """
    Write **one GeoTIFF per reconstructed band** for Landsat-comparable output.

    Preserves from the source (when available):
      - original dimensions (must match reconstructed arrays)
      - CRS
      - affine transform / pixel size / pixel alignment
      - NoData
      - scales / offsets / units / tags / mask / GCPs / RPCs (per band when present)

    Pixel values are float32 by default, or the original integer dtype when
    ``dtype_mode='source'`` (typical for Landsat uint16 DN / scaled reflectance).

    Returns a list of dicts with keys:
      ``bytes``, ``filename``, ``mime``, ``band``, ``label``, ``dtype``,
      ``width``, ``height``, ``crs``, ``transform``, ``nodata``.
    """
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio is required to export GeoTIFF results")
    if loaded.source_type != "geotiff" and loaded.raster_profile is None:
        raise ValueError(
            "GeoTIFF geospatial export requires a GeoTIFF source (rasterio profile)."
        )

    artifacts: list[dict[str, Any]] = []
    for index, name in enumerate(loaded.band_order, start=1):
        if name not in reconstructed_bands:
            continue
        original = loaded.bands[name]
        reconstructed = reconstructed_bands[name]
        if reconstructed.shape != original.shape:
            raise ValueError(
                f"Band {name} shape {reconstructed.shape} != original {original.shape}; "
                "restore native dimensions before GeoTIFF export."
            )
        data = _cast_scientific_band(
            original, reconstructed, dtype_mode=dtype_mode
        )
        # Landsat uint16 sources stay uint16 when requested; else float32.
        if (
            dtype_mode == "float32"
            and original.dtype == np.uint16
            and np.issubdtype(data.dtype, np.floating)
        ):
            # Still prefer float32 for lossy reconstructions (explicit default).
            pass
        elif dtype_mode == "source" and original.dtype == np.uint16:
            data = _cast_scientific_band(
                original, reconstructed, dtype_mode="source"
            )

        description = (
            loaded.raster_descriptions[index - 1]
            if loaded.raster_descriptions and index - 1 < len(loaded.raster_descriptions)
            else None
        )
        label = landsat_band_label(
            name,
            description=description,
            source_filename=source_filename,
            band_index=index,
        )
        profile = _single_band_export_profile(
            loaded,
            height=int(data.shape[0]),
            width=int(data.shape[1]),
            dtype=data.dtype,
        )
        with MemoryFile() as memfile:
            with memfile.open(**profile) as dataset:
                dataset.write(data, 1)
                dataset.set_band_description(1, description or label)
                if loaded.raster_scales is not None and index - 1 < len(
                    loaded.raster_scales
                ):
                    dataset.scales = (loaded.raster_scales[index - 1],)
                if loaded.raster_offsets is not None and index - 1 < len(
                    loaded.raster_offsets
                ):
                    dataset.offsets = (loaded.raster_offsets[index - 1],)
                if loaded.raster_units is not None and index - 1 < len(
                    loaded.raster_units
                ):
                    dataset.units = (loaded.raster_units[index - 1],)
                if loaded.raster_mask is not None:
                    dataset.write_mask(loaded.raster_mask)
                if loaded.raster_gcps is not None and loaded.raster_gcps[0]:
                    dataset.gcps = loaded.raster_gcps
                if loaded.raster_rpcs is not None:
                    dataset.rpcs = loaded.raster_rpcs
                _write_raster_tags_for_band(dataset, loaded, source_band_index=index)
                dataset.update_tags(
                    ESMI_BAND=name,
                    ESMI_BAND_LABEL=label,
                    ESMI_RECONSTRUCTED="1",
                )
            payload = memfile.read()

        crs = profile.get("crs")
        transform = profile.get("transform")
        artifacts.append(
            {
                "bytes": payload,
                "filename": f"reconstructed_{label}.tif",
                "mime": "image/tiff",
                "band": name,
                "label": label,
                "dtype": str(np.dtype(data.dtype)),
                "width": int(data.shape[1]),
                "height": int(data.shape[0]),
                "crs": str(crs) if crs else None,
                "transform": (
                    tuple(transform) if transform is not None else None
                ),
                "nodata": profile.get("nodata"),
                "size": len(payload),
            }
        )
    if not artifacts:
        raise ValueError("No reconstructed bands available to encode as GeoTIFF")
    return artifacts


def encode_reconstructed_bands_zip(
    loaded: LoadedImage,
    reconstructed_bands: dict[str, np.ndarray],
    *,
    source_filename: str | None = None,
    dtype_mode: Literal["float32", "source"] = "float32",
) -> tuple[bytes, str, str]:
    """Zip one GeoTIFF per band (B2–B5, …) for a single download."""
    artifacts = encode_reconstructed_band_geotiffs(
        loaded,
        reconstructed_bands,
        source_filename=source_filename,
        dtype_mode=dtype_mode,
    )
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in artifacts:
            zf.writestr(item["filename"], item["bytes"])
    return (
        buffer.getvalue(),
        "reconstructed_bands.zip",
        "application/zip",
    )


def encode_reconstructed_image(
    loaded: LoadedImage,
    reconstructed_bands: dict[str, np.ndarray],
) -> tuple[bytes, str, str]:
    """
    Encode reconstructed pixels as a downloadable image.

    GeoTIFF inputs are exported as a ZIP of one single-band GeoTIFF per band
    (CRS / transform / NoData / native dims preserved via rasterio).
    Other inputs are exported as lossless PNG files (display only).
    """
    ordered = [reconstructed_bands[name] for name in loaded.band_order]

    if loaded.source_type == "geotiff" or loaded.raster_profile is not None:
        # Prefer uint16 when the source was Landsat-style integer DN.
        sample = next(iter(loaded.bands.values()))
        dtype_mode: Literal["float32", "source"] = (
            "source" if sample.dtype == np.uint16 else "float32"
        )
        return encode_reconstructed_bands_zip(
            loaded,
            reconstructed_bands,
            dtype_mode=dtype_mode,
        )

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
    """Restore dataset-level and per-band GeoTIFF tags (multi-band write)."""
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


def _write_raster_tags_for_band(
    dataset,
    loaded: LoadedImage,
    *,
    source_band_index: int,
) -> None:
    """Restore dataset tags + tags for one source band onto export band 1."""
    for namespace, tags in (loaded.raster_dataset_tags or {}).items():
        if namespace:
            dataset.update_tags(ns=namespace, **tags)
        else:
            dataset.update_tags(**tags)
    if not loaded.raster_band_tags:
        return
    if source_band_index < 1 or source_band_index > len(loaded.raster_band_tags):
        return
    tag_groups = loaded.raster_band_tags[source_band_index - 1]
    for namespace, tags in tag_groups.items():
        if namespace:
            dataset.update_tags(1, ns=namespace, **tags)
        else:
            dataset.update_tags(1, **tags)
