"""Persist compression runs, metrics, and artifacts to Supabase."""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from supabase_client import get_storage_bucket, get_supabase_client

try:
    from supabase import Client
except ImportError:  # pragma: no cover
    Client = Any  # type: ignore[misc,assignment]


@dataclass
class SavedRun:
    run_id: str
    share_token: str
    original_storage_path: str | None
    compressed_storage_path: str | None


@dataclass
class LoadedRun:
    run: dict[str, Any]
    band_metrics: list[dict[str, Any]]
    ndvi: dict[str, Any] | None
    original_bytes: bytes | None
    compressed_bytes: bytes | None
    original_filename: str | None
    compressed_filename: str | None


def _content_type_for(path: str, fallback: str = "application/octet-stream") -> str:
    guessed, _ = mimetypes.guess_type(path)
    return guessed or fallback


def upload_bytes(
    client: Client,
    *,
    bucket: str,
    storage_path: str,
    data: bytes,
    content_type: str,
) -> str:
    """Upload bytes to Storage, overwriting if the path already exists."""
    client.storage.from_(bucket).upload(
        path=storage_path,
        file=data,
        file_options={
            "content-type": content_type,
            "upsert": "true",
        },
    )
    return storage_path


def download_bytes(client: Client, *, bucket: str, storage_path: str) -> bytes:
    return client.storage.from_(bucket).download(storage_path)


def save_compression_run(
    *,
    method: str,
    source_filename: str | None,
    archive_member: str | None,
    params: dict[str, Any],
    runtime_seconds: float,
    original_bytes_count: int,
    compressed_bytes_estimate: int,
    compression_ratio: float,
    band_metrics: list[dict[str, Any]],
    original_file_bytes: bytes | None,
    original_filename: str | None,
    compressed_file_bytes: bytes | None,
    compressed_filename: str | None,
    notes: str | None = None,
    client: Client | None = None,
) -> SavedRun:
    """
    Upload artifacts and insert run + band_metrics rows.

    Returns identifiers used for share links.
    """
    client = client or get_supabase_client()
    bucket = get_storage_bucket()
    run_id = str(uuid4())
    share_token = str(uuid4())

    original_path: str | None = None
    compressed_path: str | None = None

    if original_file_bytes is not None and original_filename:
        suffix = original_filename.rsplit(".", 1)[-1] if "." in original_filename else "bin"
        original_path = f"runs/{run_id}/original.{suffix}"
        upload_bytes(
            client,
            bucket=bucket,
            storage_path=original_path,
            data=original_file_bytes,
            content_type=_content_type_for(original_filename),
        )

    if compressed_file_bytes is not None and compressed_filename:
        suffix = (
            compressed_filename.rsplit(".", 1)[-1]
            if "." in compressed_filename
            else "bin"
        )
        compressed_path = f"runs/{run_id}/compressed.{suffix}"
        upload_bytes(
            client,
            bucket=bucket,
            storage_path=compressed_path,
            data=compressed_file_bytes,
            content_type=_content_type_for(compressed_filename),
        )

    run_row = {
        "id": run_id,
        "method": method,
        "source_filename": source_filename,
        "archive_member": archive_member,
        "params": params,
        "original_storage_path": original_path,
        "compressed_storage_path": compressed_path,
        "runtime_seconds": runtime_seconds,
        "original_bytes": original_bytes_count,
        "compressed_bytes_estimate": compressed_bytes_estimate,
        "compression_ratio": compression_ratio,
        "share_token": share_token,
        "notes": notes,
    }
    client.table("runs").insert(run_row).execute()

    metric_rows = []
    for item in band_metrics:
        metric_rows.append(
            {
                "run_id": run_id,
                "band": item["band"],
                "rmse": item.get("rmse"),
                "mae": item.get("mae"),
                "psnr_db": item.get("psnr_db"),
                "ssim": item.get("ssim"),
            }
        )
    if metric_rows:
        client.table("band_metrics").insert(metric_rows).execute()

    return SavedRun(
        run_id=run_id,
        share_token=share_token,
        original_storage_path=original_path,
        compressed_storage_path=compressed_path,
    )


def save_ndvi_for_run(
    *,
    run_id: str,
    red_band: str,
    nir_band: str,
    rmse: float,
    mae: float,
    correlation: float,
    ssim: float,
    bias: float,
    client: Client | None = None,
) -> None:
    """Upsert NDVI metrics for an existing run."""
    client = client or get_supabase_client()
    payload = {
        "run_id": run_id,
        "red_band": red_band,
        "nir_band": nir_band,
        "rmse": rmse,
        "mae": mae,
        "correlation": correlation,
        "ssim": ssim,
        "bias": bias,
    }
    client.table("ndvi_results").upsert(payload).execute()


def save_method_comparison(
    *,
    results: list[dict[str, Any]],
    params: dict[str, Any],
    run_id: str | None = None,
    client: Client | None = None,
) -> dict[str, str]:
    """Persist an all-method comparison table; optionally link to a primary run."""
    client = client or get_supabase_client()
    comparison_id = str(uuid4())
    share_token = str(uuid4())
    row = {
        "id": comparison_id,
        "run_id": run_id,
        "share_token": share_token,
        "params": params,
        "results": results,
    }
    client.table("method_comparisons").insert(row).execute()
    return {"comparison_id": comparison_id, "share_token": share_token}


def list_recent_runs(limit: int = 20, client: Client | None = None) -> list[dict[str, Any]]:
    client = client or get_supabase_client()
    response = (
        client.table("runs")
        .select(
            "id, created_at, method, source_filename, archive_member, "
            "runtime_seconds, compression_ratio, share_token"
        )
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return list(response.data or [])


def load_run_by_share_token(
    share_token: str,
    *,
    download_artifacts: bool = True,
    client: Client | None = None,
) -> LoadedRun | None:
    """Fetch a run and related rows by share token."""
    client = client or get_supabase_client()
    bucket = get_storage_bucket()
    token = share_token.strip()
    if not token:
        return None

    run_response = (
        client.table("runs").select("*").eq("share_token", token).limit(1).execute()
    )
    rows = run_response.data or []
    if not rows:
        return None
    run = rows[0]
    run_id = run["id"]

    metrics_response = (
        client.table("band_metrics").select("*").eq("run_id", run_id).execute()
    )
    band_metrics = list(metrics_response.data or [])

    ndvi_response = (
        client.table("ndvi_results").select("*").eq("run_id", run_id).limit(1).execute()
    )
    ndvi_rows = ndvi_response.data or []
    ndvi = ndvi_rows[0] if ndvi_rows else None

    original_bytes = None
    compressed_bytes = None
    original_filename = None
    compressed_filename = None

    if download_artifacts:
        original_path = run.get("original_storage_path")
        compressed_path = run.get("compressed_storage_path")
        if original_path:
            original_bytes = download_bytes(
                client, bucket=bucket, storage_path=original_path
            )
            original_filename = original_path.rsplit("/", 1)[-1]
        if compressed_path:
            compressed_bytes = download_bytes(
                client, bucket=bucket, storage_path=compressed_path
            )
            compressed_filename = compressed_path.rsplit("/", 1)[-1]

    return LoadedRun(
        run=run,
        band_metrics=band_metrics,
        ndvi=ndvi,
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        original_filename=original_filename,
        compressed_filename=compressed_filename,
    )


def build_share_url(base_url: str, share_token: str) -> str:
    base = base_url.rstrip("/")
    return f"{base}/?run={share_token}"
