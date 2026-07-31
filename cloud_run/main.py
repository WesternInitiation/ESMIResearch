"""
ESMI compression API for Google Cloud Run (free tier friendly).

Endpoints:
  GET  /health
  POST /v1/archive/list   — list image members in a TAR / TAR.GZ
  POST /v1/compress       — run SVD / wavelet / bandwidth / JPEG2000 / LZW

Build from the repository root:
  docker build -f cloud_run/Dockerfile .
"""

from __future__ import annotations

import base64
import json
import os
import tarfile
from io import BytesIO
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from compression.bandwidth import run_bandwidth_compression
from compression.jpeg2000 import run_jpeg2000_compression
from compression.lzw import run_lzw_compression
from compression.svd import run_svd_compression
from compression.wavelet import run_wavelet_compression
from image_io import (
    is_tar_archive,
    list_archive_images,
    list_archive_listing,
    load_archive_image,
    load_image,
    scan_archive_image_entries,
    to_display_rgb,
)
from ndvi import compare_ndvi, compute_ndvi
from svd_compression import ChannelCompressionConfig, CompressionConfig

MethodName = Literal[
    "SVD",
    "Wavelet transformation",
    "Bandwidth transformation",
    "JPEG2000",
    "LZW",
]

app = FastAPI(title="ESMI Compression API", version="1.0.0")

_allowed = os.environ.get("CORS_ORIGINS", "*")
_origins = [o.strip() for o in _allowed.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
# Cloud Run HTTP request bodies are capped by the platform at ~32 MiB.
# Larger jobs should use gcs_uri (browser → GCS signed PUT → Cloud Run download).
CLOUD_RUN_HTTP_MAX_BYTES = int(
    os.environ.get("CLOUD_RUN_HTTP_MAX_BYTES", str(30 * 1024 * 1024))
)
DEFAULT_MAX_DIM = int(os.environ.get("DEFAULT_MAX_DIM", "1024"))
DELETE_GCS_AFTER_JOB = os.environ.get("DELETE_GCS_AFTER_JOB", "1").strip() not in (
    "0",
    "false",
    "False",
    "no",
)


SUPPORTED_METHODS: tuple[str, ...] = (
    "SVD",
    "Wavelet transformation",
    "Bandwidth transformation",
    "JPEG2000",
    "LZW",
)


def _json_float(value: float, *, fallback: float = 0.0) -> float:
    number = float(value)
    return number if np.isfinite(number) else fallback


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "esmi-compress",
        "gcs": True,
        "methods": list(SUPPORTED_METHODS),
        "commit": os.environ.get("COMMIT_SHA", "").strip() or None,
        # Feature flags so deploys are easy to verify from /api/compress.
        "features": {
            "lzw": True,
            "nativeRestore": True,  # max_dim<=0 keeps native; else upsample back
            "methods": list(SUPPORTED_METHODS),
        },
    }


def _png_b64(rgb: np.ndarray) -> str:
    image = Image.fromarray(rgb.astype(np.uint8))
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _downsample_bands(
    bands: dict[str, np.ndarray],
    max_dim: int,
) -> tuple[dict[str, np.ndarray], float]:
    sample = next(iter(bands.values()))
    height, width = sample.shape[:2]
    longest = max(height, width)
    # max_dim <= 0 means native resolution (no downsampling).
    if max_dim <= 0 or longest <= max_dim:
        return bands, 1.0
    scale = max_dim / float(longest)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    out: dict[str, np.ndarray] = {}
    for name, band in bands.items():
        image = Image.fromarray(band.astype(np.float32), mode="F")
        resized = image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        out[name] = np.asarray(resized, dtype=band.dtype)
    return out, scale


def _upsample_bands(
    bands: dict[str, np.ndarray],
    target_width: int,
    target_height: int,
) -> dict[str, np.ndarray]:
    """Bilinear upsample so reconstructed bands match the native raster size."""
    sample = next(iter(bands.values()))
    height, width = int(sample.shape[0]), int(sample.shape[1])
    if width == target_width and height == target_height:
        return bands
    out: dict[str, np.ndarray] = {}
    for name, band in bands.items():
        image = Image.fromarray(band.astype(np.float32), mode="F")
        resized = image.resize(
            (target_width, target_height),
            resample=Image.Resampling.BILINEAR,
        )
        out[name] = np.asarray(resized, dtype=band.dtype)
    return out


def _preview_rgb_capped(bands: dict[str, np.ndarray], band_order: list[str], max_side: int = 1024) -> np.ndarray:
    """Display RGB only — codecs keep native size; previews stay small for the UI."""
    rgb = to_display_rgb(bands, band_order)
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if max_side <= 0 or longest <= max_side:
        return rgb
    scale = max_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    return np.asarray(
        image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def _load_from_upload(
    raw: bytes,
    filename: str,
    archive_member: str | None,
) -> tuple[dict[str, np.ndarray], list[str], str]:
    if is_tar_archive(filename):
        if not archive_member:
            raise HTTPException(
                status_code=400,
                detail="archive_member is required for TAR uploads",
            )
        loaded = load_archive_image(raw, archive_member)
        return loaded.bands, loaded.band_order, archive_member
    loaded = load_image(BytesIO(raw), filename)
    return loaded.bands, loaded.band_order, filename


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise HTTPException(status_code=400, detail="gcs_uri must start with gs://")
    rest = uri[5:]
    bucket_name, sep, object_name = rest.partition("/")
    if not sep or not bucket_name or not object_name:
        raise HTTPException(status_code=400, detail="Invalid gcs_uri")
    return bucket_name, object_name


def _download_gcs(uri: str) -> tuple[bytes, str, Any]:
    """Download gs:// object; returns (bytes, filename, blob_for_optional_cleanup)."""
    from google.cloud import storage  # lazy import so local tests without GCS still import

    bucket_name, object_name = _parse_gcs_uri(uri)
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(object_name)
    if not blob.exists():
        raise HTTPException(status_code=404, detail=f"GCS object not found: {uri}")
    raw = blob.download_as_bytes()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload exceeds size limit (~2 GiB)")
    filename = object_name.rsplit("/", 1)[-1] or "upload.bin"
    return raw, filename, blob


def _maybe_delete_gcs_blob(blob: Any) -> None:
    if not DELETE_GCS_AFTER_JOB or blob is None:
        return
    try:
        blob.delete()
    except Exception:
        pass


def _run_method(
    method: MethodName,
    bands: dict[str, np.ndarray],
    *,
    svd_rank: int,
    wavelet_keep_fraction: float,
    wavelet_levels: int,
    wavelet_name: str,
    bandwidth_keep_fraction: float,
    jpeg_rate: float,
) -> Any:
    if method == "SVD":
        config = CompressionConfig(
            channels={
                name: ChannelCompressionConfig(rank=max(1, int(svd_rank)))
                for name in bands
            },
            mode="rank",
            normalize_before_svd=True,
        )
        return run_svd_compression(bands, config)
    if method == "Wavelet transformation":
        return run_wavelet_compression(
            bands,
            wavelet=(wavelet_name or "db4").strip() or "db4",
            level=max(1, int(wavelet_levels)),
            keep_fraction=float(np.clip(wavelet_keep_fraction, 0.001, 1.0)),
        )
    if method == "Bandwidth transformation":
        return run_bandwidth_compression(
            bands,
            keep_fraction=float(np.clip(bandwidth_keep_fraction, 0.001, 1.0)),
        )
    if method == "LZW":
        # Same sizing path as every other method: caller may have downsampled
        # `bands`; compress/decompress here, then caller upsamples back to native.
        return run_lzw_compression(bands)
    if method != "JPEG2000":
        raise ValueError(f"Unknown method: {method}")

    # UI jpeg_rate is quality-like in (0, 1] (same as the browser JPEG stand-in).
    # OpenJPEG quality_mode="rates": larger layer values → stronger compression.
    # Map high quality → low rate so the slider direction matches the browser.
    quality = float(np.clip(jpeg_rate, 0.05, 0.95))
    openjpeg_rate = max(1, int(round((1.0 - quality) * 40 + 1)))
    try:
        return run_jpeg2000_compression(bands, rate=openjpeg_rate)
    except Exception:
        from compression.base import build_execution_result
        from time import perf_counter

        start = perf_counter()
        reconstructed: dict[str, np.ndarray] = {}
        encoded_total = 0
        for name, band in bands.items():
            band_f = band.astype(np.float64)
            lo, hi = float(band_f.min()), float(band_f.max())
            scale = hi - lo or 1.0
            norm = ((band_f - lo) / scale * 255.0).astype(np.uint8)
            image = Image.fromarray(norm, mode="L")
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=int(quality * 100))
            encoded_total += len(buffer.getvalue())
            buffer.seek(0)
            decoded = np.asarray(Image.open(buffer), dtype=np.float64)
            reconstructed[name] = (decoded / 255.0 * scale + lo).astype(band.dtype)
        return build_execution_result(
            method="jpeg2000_jpeg_fallback",
            bands=bands,
            reconstructed_bands=reconstructed,
            compressed_bytes_estimate=encoded_total,
            runtime_seconds=perf_counter() - start,
            metadata={
                "codec": "jpeg-fallback",
                "quality": quality,
                "requested_openjpeg_rate": openjpeg_rate,
            },
        )


def _demo_bucket_default() -> str:
    return (
        os.environ.get("GCS_DEMO_BUCKET", "").strip()
        or "esmi-research-demo-data"
    )


def _staging_bucket_default() -> str | None:
    name = os.environ.get("GCS_UPLOAD_BUCKET", "").strip()
    return name or None


def _build_demo_manifest(archive: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    members = sorted(e["name"] for e in entries)
    payload: dict[str, Any] = {"archive": archive, "members": members}
    ranged = [
        {"name": e["name"], "offset": e["offset"], "size": e["size"]}
        for e in entries
        if "offset" in e
    ]
    if ranged:
        payload["entries"] = sorted(ranged, key=lambda e: e["name"])
    return payload


@app.post("/v1/demo/build-manifest")
async def demo_build_manifest(
    bucket: str | None = Form(None),
    archive: str | None = Form(None),
    write_manifest: str = Form("1"),
) -> JSONResponse:
    """
    Stream-scan a large demo TAR in GCS, return members (+ offsets), and
    optionally write manifest.json so Vercel never downloads the full TAR.
    """
    from google.cloud import storage

    bucket_name = (bucket or "").strip() or _demo_bucket_default()
    archive_name = (archive or "").strip()
    if not archive_name:
        raise HTTPException(status_code=400, detail="archive is required")
    if not is_tar_archive(archive_name):
        raise HTTPException(status_code=400, detail="archive must be a TAR")

    client = storage.Client()
    blob = client.bucket(bucket_name).blob(archive_name)
    if not blob.exists():
        raise HTTPException(
            status_code=404,
            detail=f"GCS object not found: gs://{bucket_name}/{archive_name}",
        )

    try:
        with blob.open("rb") as handle:
            entries = scan_archive_image_entries(handle, filename=archive_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to scan archive: {exc}",
        ) from exc

    manifest = _build_demo_manifest(archive_name, entries)
    wrote = False
    if write_manifest.strip() not in ("0", "false", "False", "no"):
        try:
            out = client.bucket(bucket_name).blob("manifest.json")
            out.upload_from_string(
                json.dumps(manifest, indent=2) + "\n",
                content_type="application/json",
            )
            wrote = True
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Scanned OK but failed to write manifest.json: {exc}",
            ) from exc

    return JSONResponse(
        {
            **manifest,
            "bucket": bucket_name,
            "manifestWritten": wrote,
        }
    )


@app.post("/v1/demo/extract")
async def demo_extract(
    bucket: str | None = Form(None),
    archive: str | None = Form(None),
    member: str | None = Form(None),
    staging_bucket: str | None = Form(None),
    offset: int | None = Form(None),
    size: int | None = Form(None),
) -> JSONResponse:
    """
    Extract one image member from a demo TAR in GCS and stage it to the upload
    bucket. Prefer offset+size (from manifest entries) for a ranged download.
    """
    import time
    import uuid

    from google.cloud import storage

    bucket_name = (bucket or "").strip() or _demo_bucket_default()
    archive_name = (archive or "").strip()
    member_name = (member or "").strip()
    staging = (staging_bucket or "").strip() or _staging_bucket_default()
    if not archive_name or not is_tar_archive(archive_name):
        raise HTTPException(status_code=400, detail="archive must be a TAR")
    if not member_name:
        raise HTTPException(status_code=400, detail="member is required")
    if not staging:
        raise HTTPException(
            status_code=400,
            detail="staging_bucket / GCS_UPLOAD_BUCKET is required",
        )

    client = storage.Client()
    src = client.bucket(bucket_name).blob(archive_name)
    if not src.exists():
        raise HTTPException(
            status_code=404,
            detail=f"GCS object not found: gs://{bucket_name}/{archive_name}",
        )

    filename = member_name.rsplit("/", 1)[-1] or "member.bin"
    raw: bytes | None = None

    if offset is not None and size is not None and int(size) > 0:
        start = int(offset)
        end = start + int(size) - 1
        if end < start:
            raise HTTPException(status_code=400, detail="Invalid offset/size")
        try:
            raw = src.download_as_bytes(start=start, end=end)
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Ranged download failed: {exc}",
            ) from exc
    else:
        try:
            with src.open("rb") as handle:
                with tarfile.open(
                    fileobj=handle,
                    mode="r|gz"
                    if archive_name.lower().endswith((".tar.gz", ".tgz"))
                    else "r|",
                ) as tf:
                    for info in tf:
                        if not info.isfile() or info.name != member_name:
                            continue
                        extracted = tf.extractfile(info)
                        if extracted is None:
                            raise HTTPException(
                                status_code=400,
                                detail="Could not read archive member",
                            )
                        raw = extracted.read()
                        break
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Stream extract failed: {exc}",
            ) from exc

    if raw is None:
        raise HTTPException(status_code=404, detail="Archive member not found")

    object_name = f"demo-extracts/{int(time.time())}-{uuid.uuid4().hex}/{filename}"
    dest = client.bucket(staging).blob(object_name)
    try:
        dest.upload_from_string(raw, content_type="application/octet-stream")
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to stage extract: {exc}",
        ) from exc

    return JSONResponse(
        {
            "gcsUri": f"gs://{staging}/{object_name}",
            "bucket": staging,
            "objectName": object_name,
            "filename": filename,
            "size": len(raw),
        }
    )


@app.post("/v1/archive/list")
async def archive_list(file: UploadFile = File(...)) -> JSONResponse:
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload exceeds size limit (~2 GiB)")
    if len(raw) > CLOUD_RUN_HTTP_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                "Cloud Run direct uploads are limited to ~30 MB by the platform. "
                "Use Engine → Browser for files up to ~2 GiB, or split the archive."
            ),
        )
    filename = file.filename or "upload.tar"
    if not is_tar_archive(filename):
        raise HTTPException(status_code=400, detail="File is not a TAR/ZIP archive")
    try:
        listing = list_archive_listing(raw, filename=filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(
        {
            "members": listing["images"],
            "folders": listing["folders"],
            "filename": filename,
        }
    )


@app.post("/v1/compress")
async def compress(
    file: UploadFile | None = File(None),
    gcs_uri: str | None = Form(None),
    filename: str | None = Form(None),
    method: str = Form(...),
    archive_member: str | None = Form(None),
    max_dim: int = Form(DEFAULT_MAX_DIM),
    svd_rank: int = Form(32),
    wavelet_keep_fraction: float = Form(0.08),
    wavelet_levels: int = Form(3),
    wavelet_name: str = Form("db4"),
    bandwidth_keep_fraction: float = Form(0.12),
    jpeg_rate: float = Form(0.45),
    red_band: str | None = Form(None),
    nir_band: str | None = Form(None),
) -> JSONResponse:
    if method not in SUPPORTED_METHODS:
        raise HTTPException(status_code=400, detail=f"Unknown method: {method}")

    gcs_blob: Any = None
    uri = (gcs_uri or "").strip() or None
    if uri:
        try:
            raw, gcs_filename, gcs_blob = _download_gcs(uri)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to download gcs_uri: {exc}",
            ) from exc
        source_filename = (filename or "").strip() or gcs_filename
    else:
        if file is None:
            raise HTTPException(
                status_code=400,
                detail="Provide either multipart file or gcs_uri",
            )
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty upload")
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Upload exceeds size limit (~2 GiB)")
        if len(raw) > CLOUD_RUN_HTTP_MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Cloud Run direct uploads are limited to ~30 MB by the platform. "
                    "Upload via GCS (gcs_uri) for 80–100+ MB images, or use Engine → Browser."
                ),
            )
        source_filename = file.filename or filename or "upload.bin"

    member = (archive_member or "").strip() or None
    try:
        bands, band_order, source_name = _load_from_upload(raw, source_filename, member)
    except ValueError as exc:
        _maybe_delete_gcs_blob(gcs_blob)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        _maybe_delete_gcs_blob(gcs_blob)
        raise HTTPException(status_code=400, detail=f"Failed to load image: {exc}") from exc

    native = next(iter(bands.values())).shape
    native_h, native_w = int(native[0]), int(native[1])
    # max_dim <= 0 → native resolution; otherwise enforce a small lower bound.
    process_cap = 0 if int(max_dim) <= 0 else max(64, int(max_dim))
    process_bands, scale = _downsample_bands(bands, process_cap)
    sample = next(iter(process_bands.values()))
    process_h, process_w = int(sample.shape[0]), int(sample.shape[1])

    try:
        result = _run_method(
            method,  # type: ignore[arg-type]
            process_bands,
            svd_rank=svd_rank,
            wavelet_keep_fraction=wavelet_keep_fraction,
            wavelet_levels=wavelet_levels,
            wavelet_name=wavelet_name,
            bandwidth_keep_fraction=bandwidth_keep_fraction,
            jpeg_rate=jpeg_rate,
        )
    except Exception as exc:
        _maybe_delete_gcs_blob(gcs_blob)
        raise HTTPException(status_code=500, detail=f"Compression failed: {exc}") from exc

    # Restore native raster size so clients can compare / download at original dims.
    if scale < 1.0 - 1e-12 or process_w != native_w or process_h != native_h:
        result.reconstructed_bands = _upsample_bands(
            result.reconstructed_bands, native_w, native_h
        )
        result.metadata = {
            **result.metadata,
            "processWidth": process_w,
            "processHeight": process_h,
            "restoredToNative": True,
        }
    width, height = native_w, native_h

    # Previews are capped for JSON transport; width/height above stay native.
    original_preview = _preview_rgb_capped(bands, band_order)
    reconstructed_preview = _preview_rgb_capped(
        result.reconstructed_bands, band_order
    )

    ndvi_payload: dict[str, float] | None = None
    red_name = red_band if red_band in bands else ("red" if "red" in bands else None)
    nir_name = nir_band if nir_band in bands else ("nir" if "nir" in bands else None)
    if red_name and nir_name:
        ref = compute_ndvi(bands[red_name], bands[nir_name])
        cand = compute_ndvi(
            result.reconstructed_bands[red_name],
            result.reconstructed_bands[nir_name],
        )
        metrics = compare_ndvi(ref, cand)
        ndvi_payload = {
            "rmse": metrics.rmse,
            "mae": metrics.mae,
            "correlation": metrics.correlation,
            "ssim": metrics.ssim,
            "bias": metrics.bias,
            "valid_pixel_fraction": metrics.valid_pixel_fraction,
        }

    ratio = (
        result.compressed_bytes_estimate / result.original_bytes
        if result.original_bytes > 0
        else 0.0
    )

    payload = {
        "engine": "cloud-run",
        "method": method,
        "source": source_name,
        "runtimeSeconds": result.runtime_seconds,
        "originalBytes": result.original_bytes,
        "compressedBytesEstimate": result.compressed_bytes_estimate,
        "compressionRatio": ratio,
        "width": width,
        "height": height,
        "nativeWidth": native_w,
        "nativeHeight": native_h,
        "processScale": scale,
        "bandOrder": band_order,
        "channelReports": [
            {
                "band": report.name,
                "rmse": _json_float(report.rmse),
                "mae": _json_float(report.mae),
                "psnrDb": _json_float(report.psnr, fallback=99.0),
                "ssim": _json_float(report.ssim, fallback=1.0),
            }
            for report in result.channel_reports
        ],
        "metadata": {
            **result.metadata,
            **({"gcsUri": uri} if uri else {}),
        },
        "ndvi": ndvi_payload,
        "originalPreviewPngBase64": _png_b64(original_preview),
        "previewPngBase64": _png_b64(reconstructed_preview),
    }
    _maybe_delete_gcs_blob(gcs_blob)
    try:
        return JSONResponse(payload)
    except (TypeError, ValueError) as exc:
        # Last resort: never 500 on metric serialization (e.g. leftover Inf).
        raise HTTPException(
            status_code=500,
            detail=f"Failed to serialize compression response: {exc}",
        ) from exc
