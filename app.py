"""
ESMI Research — satellite image compression benchmarking workbench.

Run: streamlit run app.py
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import PurePosixPath
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from compression.base import CompressionExecutionResult, compute_band_metrics
from compression.bandwidth import run_bandwidth_compression
from compression.jpeg2000 import run_jpeg2000_compression
from compression.svd import run_svd_compression
from compression.wavelet import run_wavelet_compression
from image_io import (
    encode_reconstructed_image,
    is_tar_archive,
    list_archive_images,
    load_archive_image,
    load_image,
    to_display_rgb,
)
from ndvi import compare_ndvi, compute_ndvi
from persistence import (
    build_share_url,
    list_recent_runs,
    load_run_by_share_token,
    save_compression_run,
    save_method_comparison,
    save_ndvi_for_run,
)
from supabase_client import SupabaseNotConfiguredError, supabase_configured
from svd_compression import (
    ChannelCompressionConfig,
    CompressionConfig,
    apply_channel_weight_matrix,
    identity_weight_matrix,
)

st.set_page_config(page_title="ESMI Compression Lab", page_icon="🛰️", layout="wide")

st.title("ESMI Research — Compression Analysis Lab")
st.caption(
    "Benchmark satellite image compression with SVD, wavelets, bandwidth-domain "
    "filtering, and JPEG2000 while tracking runtime, fidelity, and NDVI preservation."
)

COMPRESSION_METHODS = [
    "SVD",
    "Wavelet transformation",
    "Bandwidth transformation",
    "JPEG2000",
]

UPLOAD_FILE_TYPES = [
    "tif",
    "tiff",
    "png",
    "jpg",
    "jpeg",
    "webp",
    "tar",
    "tar.gz",
    "tgz",
    "gz",
]


@st.cache_data(show_spinner=False)
def _load_uploaded(
    file_bytes: bytes,
    filename: str,
    archive_member: str | None = None,
):
    if archive_member is not None:
        return load_archive_image(file_bytes, archive_member)
    return load_image(io.BytesIO(file_bytes), filename)


@st.cache_data(show_spinner=False)
def _list_archive_images(file_bytes: bytes) -> list[str]:
    return list_archive_images(file_bytes)


def _reset_ndvi_confirmation() -> None:
    st.session_state["confirm_ndvi"] = False
    st.session_state.pop("ndvi_run", None)


def _max_rank(shape: tuple[int, ...]) -> int:
    return min(shape[0], shape[1])


def _show_image(target, image, caption: str | None = None, **kwargs):
    """Show an image with Streamlit API compatibility across versions."""
    # Prefer older Anaconda-friendly kwargs first, then newer APIs, then bare call.
    attempts = (
        {"use_column_width": True},
        {},
        {"use_container_width": True},
        {"width": "stretch"},
    )
    last_error: TypeError | None = None
    for options in attempts:
        try:
            return target.image(image, caption=caption, **options, **kwargs)
        except TypeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return None


def _show_dataframe(target, data, **kwargs):
    """Show a dataframe with Streamlit API compatibility across versions."""
    attempts = (
        {},
        {"use_container_width": True},
        {"width": "stretch"},
    )
    last_error: TypeError | None = None
    for options in attempts:
        try:
            return target.dataframe(data, **options, **kwargs)
        except TypeError as exc:
            last_error = exc
            # Older Streamlit may not support hide_index either.
            if "hide_index" in kwargs:
                try:
                    trimmed = dict(kwargs)
                    trimmed.pop("hide_index", None)
                    return target.dataframe(data, **options, **trimmed)
                except TypeError as inner_exc:
                    last_error = inner_exc
    if last_error is not None:
        raise last_error
    return None


def _query_run_token() -> str | None:
    params = st.query_params
    value = params.get("run")
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    token = str(value).strip()
    return token or None


def _app_base_url() -> str:
    try:
        return str(st.secrets.get("app_base_url", "")).strip().rstrip("/")
    except Exception:
        return ""


def _render_shared_run_viewer() -> None:
    """Load and display a shared Supabase run from ?run= or manual token entry."""
    st.subheader("Shared runs")
    if not supabase_configured():
        st.caption(
            "Configure Supabase secrets to save and share compression runs with "
            "your research group."
        )
        return

    query_token = _query_run_token()
    token_input = st.text_input(
        "Share token",
        value=query_token or "",
        help="Paste a share token or open a link like /?run=<token>.",
        key="shared_run_token_input",
    )
    load_clicked = st.button("Load shared run", key="load_shared_run_btn")

    if load_clicked and token_input.strip():
        st.query_params["run"] = token_input.strip()

    active_token = (st.query_params.get("run") or token_input or "").strip()
    if isinstance(active_token, list):
        active_token = active_token[0] if active_token else ""

    try:
        recent = list_recent_runs(limit=15)
    except Exception as exc:
        st.warning(f"Could not list recent runs: {exc}")
        recent = []

    if recent:
        st.caption("Recent saved runs")
        recent_df = pd.DataFrame(recent)
        display_cols = [
            col
            for col in (
                "created_at",
                "method",
                "source_filename",
                "runtime_seconds",
                "compression_ratio",
                "share_token",
            )
            if col in recent_df.columns
        ]
        _show_dataframe(st, recent_df[display_cols], hide_index=True)

    if not active_token:
        return

    try:
        with st.spinner("Loading shared run from Supabase…"):
            loaded_shared = load_run_by_share_token(active_token)
    except Exception as exc:
        st.error(f"Failed to load shared run: {exc}")
        return

    if loaded_shared is None:
        st.warning("No run found for that share token.")
        return

    run = loaded_shared.run
    st.success(
        f"Loaded shared run · method **{run.get('method')}** · "
        f"source `{run.get('source_filename') or 'unknown'}`"
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Runtime", f"{float(run.get('runtime_seconds') or 0):.4f} s")
    m2.metric(
        "Compression ratio",
        f"{float(run.get('compression_ratio') or 0):.2%}",
    )
    m3.metric("Original bytes", f"{int(run.get('original_bytes') or 0):,}")
    m4.metric(
        "Compressed estimate",
        f"{int(run.get('compressed_bytes_estimate') or 0):,}",
    )

    if loaded_shared.band_metrics:
        _show_dataframe(
            st,
            pd.DataFrame(loaded_shared.band_metrics),
            hide_index=True,
        )

    if loaded_shared.ndvi is not None:
        ndvi = loaded_shared.ndvi
        st.markdown(
            f"NDVI (`{ndvi.get('red_band')}` / `{ndvi.get('nir_band')}`): "
            f"RMSE={ndvi.get('rmse')}, SSIM={ndvi.get('ssim')}, "
            f"corr={ndvi.get('correlation')}"
        )

    dl1, dl2 = st.columns(2)
    if loaded_shared.original_bytes is not None:
        dl1.download_button(
            "Download original artifact",
            data=loaded_shared.original_bytes,
            file_name=loaded_shared.original_filename or "original.bin",
            mime="application/octet-stream",
            key="shared_original_dl",
        )
    if loaded_shared.compressed_bytes is not None:
        dl2.download_button(
            "Download compressed artifact",
            data=loaded_shared.compressed_bytes,
            file_name=loaded_shared.compressed_filename or "compressed.bin",
            mime="application/octet-stream",
            key="shared_compressed_dl",
        )

    base = _app_base_url()
    if base:
        st.code(build_share_url(base, str(run["share_token"])), language=None)
    else:
        st.code(f"?run={run['share_token']}", language=None)
        st.caption(
            "Set `app_base_url` in secrets to show a full shareable Streamlit Cloud URL."
        )


def _render_ndvi(ax, ndvi: np.ndarray, title: str) -> None:
    im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _render_error_map(ax, original_rgb: np.ndarray, compressed_rgb: np.ndarray) -> None:
    diff = np.abs(
        compressed_rgb.astype(np.float64) - original_rgb.astype(np.float64)
    ).mean(axis=-1)
    im = ax.imshow(diff, cmap="magma")
    ax.set_title("Absolute RGB error")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _build_weight_matrix(
    band_count: int,
    band_order: list[str],
    use_custom: bool,
    preset: str,
) -> np.ndarray:
    if not use_custom:
        if preset == "ndvi_emphasis" and band_count >= 2:
            matrix = identity_weight_matrix(band_count)
            if "nir" in band_order and "red" in band_order:
                nir_i = band_order.index("nir")
                red_i = band_order.index("red")
                matrix[nir_i, nir_i] = 1.5
                matrix[red_i, red_i] = 1.5
            return matrix
        return identity_weight_matrix(band_count)

    matrix = np.zeros((band_count, band_count), dtype=np.float64)
    for row in range(band_count):
        for col in range(band_count):
            key = f"w_{row}_{col}"
            matrix[row, col] = st.session_state.get(key, 1.0 if row == col else 0.0)
    return matrix


def _recover_original_band_space(
    mixed_bands: dict[str, np.ndarray],
    weight_matrix: np.ndarray,
    band_order: list[str],
) -> dict[str, np.ndarray]:
    if np.allclose(weight_matrix, np.eye(weight_matrix.shape[0])):
        return mixed_bands

    inverse = np.linalg.pinv(weight_matrix)
    ordered = [mixed_bands[name].astype(np.float64) for name in band_order]
    cube = np.stack(ordered, axis=-1)
    flat = cube.reshape(-1, len(band_order))
    restored = flat @ inverse.T
    restored = restored.reshape(cube.shape)
    return {name: restored[..., i] for i, name in enumerate(band_order)}


def _align_dtypes(
    original_bands: dict[str, np.ndarray],
    restored_bands: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    aligned: dict[str, np.ndarray] = {}
    for name, original in original_bands.items():
        restored = np.clip(restored_bands[name], original.min(), original.max())
        aligned[name] = restored.astype(original.dtype)
    return aligned


def _report_table(result: CompressionExecutionResult) -> pd.DataFrame:
    rows = []
    for item in result.channel_reports:
        rows.append(
            {
                "band": item.name,
                "rmse": item.rmse,
                "mae": item.mae,
                "psnr_db": item.psnr,
                "ssim": item.ssim,
            }
        )
    return pd.DataFrame(rows)


def _run_selected_method(
    method: str,
    original_bands: dict[str, np.ndarray],
    working_bands: dict[str, np.ndarray],
    band_order: list[str],
    weight_matrix: np.ndarray,
    svd_config: CompressionConfig,
    wavelet_name: str,
    wavelet_level: int,
    wavelet_keep_fraction: float,
    bandwidth_keep_fraction: float,
    jpeg2000_rate: int,
) -> CompressionExecutionResult:
    if method == "SVD":
        raw_result = run_svd_compression(working_bands, svd_config)
    elif method == "Wavelet transformation":
        raw_result = run_wavelet_compression(
            working_bands,
            wavelet=wavelet_name,
            level=wavelet_level,
            keep_fraction=wavelet_keep_fraction,
        )
    elif method == "Bandwidth transformation":
        raw_result = run_bandwidth_compression(
            working_bands,
            keep_fraction=bandwidth_keep_fraction,
        )
    else:
        raw_result = run_jpeg2000_compression(
            working_bands,
            rate=jpeg2000_rate,
        )

    restored_bands = _recover_original_band_space(
        raw_result.reconstructed_bands, weight_matrix, band_order
    )
    reconstructed_bands = _align_dtypes(original_bands, restored_bands)
    return CompressionExecutionResult(
        method=raw_result.method,
        reconstructed_bands=reconstructed_bands,
        reconstructed_image=np.stack(
            [reconstructed_bands[name] for name in band_order], axis=-1
        ),
        original_bytes=raw_result.original_bytes,
        compressed_bytes_estimate=raw_result.compressed_bytes_estimate,
        runtime_seconds=raw_result.runtime_seconds,
        metadata=raw_result.metadata,
        channel_reports=[
            type(report)(
                name=report.name,
                rmse=float(
                    np.sqrt(
                        np.mean(
                            (
                                reconstructed_bands[report.name].astype(np.float64)
                                - original_bands[report.name].astype(np.float64)
                            )
                            ** 2
                        )
                    )
                ),
                mae=float(
                    np.mean(
                        np.abs(
                            reconstructed_bands[report.name].astype(np.float64)
                            - original_bands[report.name].astype(np.float64)
                        )
                    )
                ),
                psnr=report.psnr,
                ssim=report.ssim,
            )
            for report in raw_result.channel_reports
        ],
    )


def _recompute_channel_report(
    result: CompressionExecutionResult,
    original_bands: dict[str, np.ndarray],
) -> CompressionExecutionResult:
    channel_reports = [
        compute_band_metrics(name, original_bands[name], result.reconstructed_bands[name])
        for name in original_bands
    ]
    result.channel_reports = channel_reports
    return result


def _comparison_runner(
    runners: dict[str, Callable[[], CompressionExecutionResult]],
    original_bands: dict[str, np.ndarray],
    progress_callback: Callable[[str, float], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, CompressionExecutionResult]]:
    """Run each method once and collect benchmarking rows plus full results."""
    rows = []
    results: dict[str, CompressionExecutionResult] = {}
    labels = list(runners.keys())
    for index, label in enumerate(labels):
        if progress_callback is not None:
            progress_callback(label, index / max(len(labels), 1))
        try:
            result = _recompute_channel_report(runners[label](), original_bands)
            results[label] = result
            ratio = (
                result.compressed_bytes_estimate / result.original_bytes
                if result.original_bytes
                else 0.0
            )
            report_df = _report_table(result)
            mean_psnr = float(report_df["psnr_db"].replace([np.inf, -np.inf], np.nan).mean())
            rows.append(
                {
                    "method": label,
                    "runtime_seconds": result.runtime_seconds,
                    "compression_ratio": ratio,
                    "compressed_bytes_estimate": result.compressed_bytes_estimate,
                    "mean_rmse": float(report_df["rmse"].mean()),
                    "mean_mae": float(report_df["mae"].mean()),
                    "mean_psnr_db": mean_psnr,
                    "mean_ssim": float(report_df["ssim"].mean()),
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "method": label,
                    "runtime_seconds": np.nan,
                    "compression_ratio": np.nan,
                    "compressed_bytes_estimate": np.nan,
                    "mean_rmse": np.nan,
                    "mean_mae": np.nan,
                    "mean_psnr_db": np.nan,
                    "mean_ssim": np.nan,
                    "status": str(exc),
                }
            )
    if progress_callback is not None:
        progress_callback("done", 1.0)
    return pd.DataFrame(rows), results


def _method_runners(
    bands: dict[str, np.ndarray],
    working_bands: dict[str, np.ndarray],
    band_order: list[str],
    weight_matrix: np.ndarray,
    svd_config: CompressionConfig,
    wavelet_name: str,
    wavelet_level: int,
    wavelet_keep_fraction: float,
    bandwidth_keep_fraction: float,
    jpeg2000_rate: int,
) -> dict[str, Callable[[], CompressionExecutionResult]]:
    common = (
        bands,
        working_bands,
        band_order,
        weight_matrix,
        svd_config,
        wavelet_name,
        wavelet_level,
        wavelet_keep_fraction,
        bandwidth_keep_fraction,
        jpeg2000_rate,
    )
    return {
        label: (lambda method=label: _run_selected_method(method, *common))
        for label in COMPRESSION_METHODS
    }


with st.expander("Shared runs (Supabase)", expanded=bool(_query_run_token())):
    _render_shared_run_viewer()


method = st.selectbox(
    "Compression method",
    options=COMPRESSION_METHODS,
    help="Choose which algorithm compresses the uploaded image.",
)

uploaded = st.file_uploader(
    "Upload an image or TAR archive for compression",
    type=UPLOAD_FILE_TYPES,
)

if uploaded is None:
    st.info(
        "Select a compression method and upload an image or TAR archive. GeoTIFF is "
        "best for optional NDVI analysis because it can preserve Red and NIR bands."
    )
    st.stop()

uploaded_bytes = uploaded.getvalue()
archive_member: str | None = None
try:
    if is_tar_archive(uploaded.name):
        archive_images = _list_archive_images(uploaded_bytes)
        archive_member = st.selectbox(
            "Image inside archive",
            options=archive_images,
            format_func=lambda name: f"{PurePosixPath(name).name} — {name}",
            help="Only supported image files are listed; archive contents are read in memory.",
        )
    loaded = _load_uploaded(uploaded_bytes, uploaded.name, archive_member)
except (OSError, RuntimeError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

bands = {k: v.copy() for k, v in loaded.bands.items()}
band_order = loaded.band_order
band_count = len(band_order)

with st.sidebar:
    st.header("Compression settings")
    compare_all_methods = st.checkbox(
        "Enable all-method runtime comparison",
        value=False,
        help=(
            "Unlocks the Method Comparison tab so you can benchmark SVD, wavelet, "
            "bandwidth, and JPEG2000 on the same image without re-running on every "
            "page interaction."
        ),
    )

    st.subheader("Matrix preprocessing")
    use_custom_matrix = st.checkbox("Edit preprocessing matrix", value=False)
    matrix_preset = st.selectbox(
        "Matrix preset",
        options=["identity", "ndvi_emphasis"],
        format_func=lambda x: {
            "identity": "Identity (no mixing)",
            "ndvi_emphasis": "NDVI emphasis (boost Red & NIR)",
        }[x],
        disabled=use_custom_matrix,
    )
    if use_custom_matrix:
        st.caption("Rows are transformed output bands. Columns are input bands.")
        for row, out_name in enumerate(band_order):
            cols = st.columns(min(band_count, 4))
            for col, in_name in enumerate(band_order):
                with cols[col % len(cols)]:
                    st.number_input(
                        f"{out_name}<-{in_name}",
                        key=f"w_{row}_{col}",
                        value=1.0 if row == col else 0.0,
                        step=0.1,
                        format="%.2f",
                    )

    mode = "rank"
    normalize = True
    channel_configs: dict[str, ChannelCompressionConfig] = {}
    wavelet_name = "db2"
    wavelet_level = 2
    wavelet_keep_fraction = 0.2
    bandwidth_keep_fraction = 0.25
    jpeg2000_rate = 20

    if method == "SVD":
        mode = st.radio(
            "SVD truncation mode",
            options=["rank", "energy"],
            format_func=lambda x: "Fixed rank (k)" if x == "rank" else "Energy fraction",
        )
        normalize = st.checkbox("Normalize bands before SVD", value=True)
        st.subheader("Per-band SVD parameters")
        for name in band_order:
            shape = bands[name].shape
            max_k = _max_rank(shape)
            st.markdown(f"**{name}** `{shape[1]}x{shape[0]}`")
            weight = st.slider(
                f"{name} priority weight",
                min_value=0.1,
                max_value=3.0,
                value=1.5 if name in ("red", "nir") else 1.0,
                step=0.1,
                key=f"weight_{name}",
            )
            if mode == "rank":
                base_rank = st.slider(
                    f"{name} rank (k)",
                    min_value=1,
                    max_value=max_k,
                    value=min(32, max_k),
                    key=f"rank_{name}",
                )
                effective_rank = max(1, min(max_k, int(round(base_rank * weight))))
                channel_configs[name] = ChannelCompressionConfig(
                    rank=effective_rank, weight=weight
                )
                st.caption(f"Effective rank: {effective_rank}")
            else:
                energy = st.slider(
                    f"{name} energy retained",
                    min_value=0.5,
                    max_value=0.999,
                    value=0.95,
                    step=0.001,
                    format="%.3f",
                    key=f"energy_{name}",
                )
                adjusted = min(0.999, energy * (0.8 + 0.2 * weight))
                channel_configs[name] = ChannelCompressionConfig(
                    energy_fraction=adjusted, weight=weight
                )
                st.caption(f"Effective energy target: {adjusted:.3f}")
    elif method == "Wavelet transformation":
        wavelet_name = st.selectbox("Wavelet family", options=["db1", "db2", "haar", "sym2"])
        wavelet_level = st.slider("Wavelet decomposition level", min_value=1, max_value=5, value=2)
        wavelet_keep_fraction = st.slider(
            "Wavelet coefficient keep fraction",
            min_value=0.01,
            max_value=1.0,
            value=0.2,
            step=0.01,
        )
    elif method == "Bandwidth transformation":
        bandwidth_keep_fraction = st.slider(
            "Low-frequency bandwidth keep fraction",
            min_value=0.01,
            max_value=1.0,
            value=0.25,
            step=0.01,
        )
    else:
        jpeg2000_rate = st.slider(
            "JPEG2000 rate",
            min_value=5,
            max_value=100,
            value=20,
            help="Larger values generally preserve more detail but use more bytes.",
        )

if not channel_configs:
    for name in band_order:
        default_rank = min(32, _max_rank(bands[name].shape))
        channel_configs[name] = ChannelCompressionConfig(rank=default_rank, weight=1.0)

weight_matrix = _build_weight_matrix(
    band_count, band_order, use_custom_matrix, matrix_preset
)
working_bands = apply_channel_weight_matrix(bands, weight_matrix, band_order)
svd_config = CompressionConfig(
    channels=channel_configs,
    mode=mode,
    normalize_before_svd=normalize,
)

signature_payload = {
    "input_sha256": hashlib.sha256(uploaded_bytes).hexdigest(),
    "archive_member": archive_member,
    "method": method,
    "weight_matrix": weight_matrix.tolist(),
    "svd_mode": mode,
    "svd_normalize": normalize,
    "svd_channels": {
        name: {
            "rank": config.rank,
            "energy_fraction": config.energy_fraction,
            "weight": config.weight,
        }
        for name, config in channel_configs.items()
    },
    "wavelet": [wavelet_name, wavelet_level, wavelet_keep_fraction],
    "bandwidth_keep_fraction": bandwidth_keep_fraction,
    "jpeg2000_rate": jpeg2000_rate,
}
compression_signature = hashlib.sha256(
    json.dumps(signature_payload, sort_keys=True).encode("utf-8")
).hexdigest()

comparison_signature = hashlib.sha256(
    json.dumps(
        {
            "input_sha256": signature_payload["input_sha256"],
            "archive_member": archive_member,
            "weight_matrix": weight_matrix.tolist(),
            "svd_mode": mode,
            "svd_normalize": normalize,
            "svd_channels": signature_payload["svd_channels"],
            "wavelet": [wavelet_name, wavelet_level, wavelet_keep_fraction],
            "bandwidth_keep_fraction": bandwidth_keep_fraction,
            "jpeg2000_rate": jpeg2000_rate,
        },
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()

run_compression = st.button(
    "Run compression",
    type="primary",
    help="Compression only. NDVI testing is a separate confirmed step.",
)

if run_compression:
    try:
        with st.spinner(f"Running {method} compression…"):
            new_result = _recompute_channel_report(
                _run_selected_method(
                    method,
                    bands,
                    working_bands,
                    band_order,
                    weight_matrix,
                    svd_config,
                    wavelet_name,
                    wavelet_level,
                    wavelet_keep_fraction,
                    bandwidth_keep_fraction,
                    jpeg2000_rate,
                ),
                bands,
            )
            artifact_bytes, artifact_filename, artifact_mime = (
                encode_reconstructed_image(loaded, new_result.reconstructed_bands)
            )

        source_name = archive_member or uploaded.name
        source_stem = PurePosixPath(source_name).stem
        artifact_suffix = PurePosixPath(artifact_filename).suffix
        method_slug = method.lower().replace(" ", "_")
        download_name = f"{source_stem}_{method_slug}_compressed{artifact_suffix}"
        st.session_state["compression_run"] = {
            "signature": compression_signature,
            "result": new_result,
            "artifact_bytes": artifact_bytes,
            "artifact_filename": download_name,
            "artifact_mime": artifact_mime,
        }
        st.session_state.pop("ndvi_run", None)
        st.session_state["confirm_ndvi"] = False
        st.session_state.pop("supabase_saved_run", None)
        # Keep method comparison only if its settings signature still matches.
        stored_comparison = st.session_state.get("method_comparison")
        if (
            stored_comparison is not None
            and stored_comparison.get("signature") != comparison_signature
        ):
            st.session_state.pop("method_comparison", None)
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        st.error(f"Compression failed: {exc}")
        st.stop()

stored_run = st.session_state.get("compression_run")
if stored_run is None or stored_run["signature"] != compression_signature:
    st.info("Review the settings, then select **Run compression** to create the output.")
    st.stop()

result = stored_run["result"]
original_rgb = to_display_rgb(bands, band_order)
compressed_rgb = to_display_rgb(result.reconstructed_bands, band_order)
ratio = (
    result.compressed_bytes_estimate / result.original_bytes
    if result.original_bytes
    else 0.0
)
report_df = _report_table(result)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Method", method)
col2.metric("Runtime", f"{result.runtime_seconds:.4f} s")
col3.metric("Compression ratio", f"{ratio:.2%}")
col4.metric("Bands", band_count)
col5.metric("Estimated compressed size", f"{result.compressed_bytes_estimate:,} B")

st.download_button(
    "Download compressed image",
    data=stored_run["artifact_bytes"],
    file_name=stored_run["artifact_filename"],
    mime=stored_run["artifact_mime"],
    type="primary",
)
st.caption(
    f"Download size: {len(stored_run['artifact_bytes']):,} B. "
    "The algorithm size above estimates its compact representation; the download "
    "contains the reconstructed pixels in a portable image format."
)

st.subheader("Save & share with Supabase")
if not supabase_configured():
    st.info(
        "Add Supabase secrets to enable cloud storage and shareable run links. "
        "See `.streamlit/secrets.toml.example`."
    )
else:
    notes = st.text_input(
        "Optional notes for this run",
        key="supabase_run_notes",
        placeholder="e.g. Sentinel-2 scene, k=32 NDVI emphasis",
    )
    if st.button("Save run to Supabase", type="secondary", key="save_run_supabase"):
        try:
            with st.spinner("Uploading artifacts and saving metrics…"):
                params_payload = {
                    "method": method,
                    "matrix_preset": matrix_preset,
                    "use_custom_matrix": use_custom_matrix,
                    "weight_matrix": weight_matrix.tolist(),
                    "svd_mode": mode,
                    "svd_normalize": normalize,
                    "svd_channels": signature_payload["svd_channels"],
                    "wavelet": {
                        "name": wavelet_name,
                        "level": wavelet_level,
                        "keep_fraction": wavelet_keep_fraction,
                    },
                    "bandwidth_keep_fraction": bandwidth_keep_fraction,
                    "jpeg2000_rate": jpeg2000_rate,
                    "band_order": band_order,
                }
                band_metric_rows = report_df.to_dict(orient="records")
                saved = save_compression_run(
                    method=method,
                    source_filename=uploaded.name,
                    archive_member=archive_member,
                    params=params_payload,
                    runtime_seconds=float(result.runtime_seconds),
                    original_bytes_count=int(result.original_bytes),
                    compressed_bytes_estimate=int(result.compressed_bytes_estimate),
                    compression_ratio=float(ratio),
                    band_metrics=band_metric_rows,
                    original_file_bytes=uploaded_bytes,
                    original_filename=uploaded.name,
                    compressed_file_bytes=stored_run["artifact_bytes"],
                    compressed_filename=stored_run["artifact_filename"],
                    notes=notes or None,
                )
            st.session_state["supabase_saved_run"] = {
                "run_id": saved.run_id,
                "share_token": saved.share_token,
                "compression_signature": compression_signature,
            }
            st.success("Run saved to Supabase.")
        except (SupabaseNotConfiguredError, Exception) as exc:
            st.error(f"Save failed: {exc}")

    saved_meta = st.session_state.get("supabase_saved_run")
    if (
        saved_meta is not None
        and saved_meta.get("compression_signature") == compression_signature
    ):
        share_token = saved_meta["share_token"]
        base = _app_base_url()
        share_url = (
            build_share_url(base, share_token)
            if base
            else f"?run={share_token}"
        )
        st.markdown(f"**Share link / token:** `{share_token}`")
        st.code(share_url, language=None)
        if not base:
            st.caption(
                "Set `app_base_url` in secrets (e.g. your "
                "`https://….streamlit.app` URL) for a full share link."
            )

tab_overview, tab_ndvi, tab_methods, tab_matrix, tab_report = st.tabs(
    ["Overview", "NDVI", "Method Comparison", "Matrices", "Analysis Report"]
)

with tab_overview:
    st.subheader("Side-by-side image comparison")
    st.caption(
        "Primary result for the selected compression method: original vs reconstructed "
        "preview, absolute error map, and per-band quality metrics."
    )
    c1, c2 = st.columns(2)
    _show_image(c1, original_rgb, caption="Original preview")
    _show_image(c2, compressed_rgb, caption="Compressed preview")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(compressed_rgb)
    axes[1].set_title("Compressed")
    axes[1].axis("off")
    _render_error_map(axes[2], original_rgb, compressed_rgb)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    st.subheader("Per-band quality metrics")
    if report_df.empty:
        st.warning("No per-band metrics were produced for this run.")
    else:
        _show_dataframe(st, report_df, hide_index=True)

ndvi_run = st.session_state.get("ndvi_run")
with tab_ndvi:
    st.subheader("Optional NDVI preservation test")
    st.caption(
        "Use this when your image has Red and NIR bands (typical for GeoTIFF "
        "satellite scenes). NDVI is not run during compression — confirm and run "
        "it here to measure vegetation-index fidelity after compression."
    )
    red_name = st.selectbox(
        "Red band for NDVI",
        options=band_order,
        index=band_order.index("red") if "red" in band_order else 0,
        key="ndvi_red_band",
        on_change=_reset_ndvi_confirmation,
    )
    nir_name = st.selectbox(
        "NIR band for NDVI",
        options=band_order,
        index=band_order.index("nir") if "nir" in band_order else min(1, band_count - 1),
        key="ndvi_nir_band",
        on_change=_reset_ndvi_confirmation,
    )

    distinct_ndvi_bands = red_name != nir_name
    if red_name == nir_name:
        st.warning("Select distinct Red and NIR bands to compute NDVI.")

    st.checkbox(
        "I confirm that I want to run the NDVI test",
        key="confirm_ndvi",
    )
    run_ndvi = st.button(
        "Confirm and run NDVI test",
        disabled=not st.session_state.get("confirm_ndvi", False)
        or not distinct_ndvi_bands,
    )
    ndvi_signature = f"{compression_signature}:{red_name}:{nir_name}"

    if run_ndvi:
        with st.spinner("Running NDVI preservation test…"):
            ndvi_orig = compute_ndvi(bands[red_name], bands[nir_name])
            ndvi_comp = compute_ndvi(
                result.reconstructed_bands[red_name],
                result.reconstructed_bands[nir_name],
            )
            ndvi_metrics = compare_ndvi(ndvi_orig, ndvi_comp)
        ndvi_run = {
            "signature": ndvi_signature,
            "original": ndvi_orig,
            "compressed": ndvi_comp,
            "metrics": ndvi_metrics,
        }
        st.session_state["ndvi_run"] = ndvi_run

    if ndvi_run is not None and ndvi_run["signature"] == ndvi_signature:
        ndvi_orig = ndvi_run["original"]
        ndvi_comp = ndvi_run["compressed"]
        ndvi_metrics = ndvi_run["metrics"]

        n1, n2, n3, n4, n5 = st.columns(5)
        n1.metric("NDVI RMSE", f"{ndvi_metrics.rmse:.5f}")
        n2.metric("NDVI MAE", f"{ndvi_metrics.mae:.5f}")
        n3.metric("NDVI correlation", f"{ndvi_metrics.correlation:.4f}")
        n4.metric("NDVI SSIM", f"{ndvi_metrics.ssim:.4f}")
        n5.metric("NDVI bias", f"{ndvi_metrics.bias:.5f}")

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        _render_ndvi(axes[0], ndvi_orig, "Original NDVI")
        _render_ndvi(axes[1], ndvi_comp, "Compressed NDVI")
        diff = ndvi_comp - ndvi_orig
        im = axes[2].imshow(diff, cmap="coolwarm", vmin=-0.3, vmax=0.3)
        axes[2].set_title("NDVI difference")
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        saved_meta = st.session_state.get("supabase_saved_run")
        can_attach_ndvi = (
            supabase_configured()
            and saved_meta is not None
            and saved_meta.get("compression_signature") == compression_signature
        )
        if can_attach_ndvi:
            if st.button("Save NDVI results to Supabase", key="save_ndvi_supabase"):
                try:
                    save_ndvi_for_run(
                        run_id=saved_meta["run_id"],
                        red_band=red_name,
                        nir_band=nir_name,
                        rmse=float(ndvi_metrics.rmse),
                        mae=float(ndvi_metrics.mae),
                        correlation=float(ndvi_metrics.correlation),
                        ssim=float(ndvi_metrics.ssim),
                        bias=float(ndvi_metrics.bias),
                    )
                    st.success("NDVI metrics attached to the saved Supabase run.")
                except Exception as exc:
                    st.error(f"Failed to save NDVI: {exc}")
        elif supabase_configured():
            st.caption(
                "Save the compression run to Supabase first, then attach NDVI metrics."
            )
    else:
        st.info("NDVI has not been run for this compression result and band selection.")

with tab_methods:
    st.subheader("Benchmark every compression method")
    st.caption(
        "Compare SVD, wavelet, bandwidth, and JPEG2000 on the same image using the "
        "current sidebar parameters. Results are stored until you change settings or "
        "upload a new file — they do not re-run on every page refresh."
    )

    if not compare_all_methods:
        st.info(
            "Turn on **Enable all-method runtime comparison** in the sidebar, then "
            "click **Run all-method comparison** below."
        )
    else:
        run_comparison = st.button(
            "Run all-method comparison",
            type="primary",
            help="Runs all four algorithms once and caches runtime/fidelity metrics.",
        )
        if run_comparison:
            progress = st.progress(0.0, text="Starting method comparison…")
            status = st.empty()

            def _on_progress(label: str, fraction: float) -> None:
                if label == "done":
                    progress.progress(1.0, text="Comparison complete")
                    status.empty()
                else:
                    progress.progress(fraction, text=f"Running {label}…")
                    status.caption(f"Currently benchmarking: {label}")

            try:
                comparison_df, comparison_results = _comparison_runner(
                    _method_runners(
                        bands,
                        working_bands,
                        band_order,
                        weight_matrix,
                        svd_config,
                        wavelet_name,
                        wavelet_level,
                        wavelet_keep_fraction,
                        bandwidth_keep_fraction,
                        jpeg2000_rate,
                    ),
                    bands,
                    progress_callback=_on_progress,
                )
                st.session_state["method_comparison"] = {
                    "signature": comparison_signature,
                    "table": comparison_df,
                    "previews": {
                        label: to_display_rgb(item.reconstructed_bands, band_order)
                        for label, item in comparison_results.items()
                    },
                }
            except Exception as exc:
                st.error(f"All-method comparison failed: {exc}")

        stored_comparison = st.session_state.get("method_comparison")
        if (
            stored_comparison is not None
            and stored_comparison["signature"] == comparison_signature
        ):
            comparison_df = stored_comparison["table"]
            st.success("Comparison results are ready for this image and settings.")
            _show_dataframe(st, comparison_df, hide_index=True)

            numeric_df = comparison_df[
                comparison_df["status"] == "ok"
            ].dropna(subset=["runtime_seconds"])
            if not numeric_df.empty:
                chart_df = numeric_df.set_index("method")[
                    ["runtime_seconds", "compression_ratio", "mean_rmse", "mean_ssim"]
                ]
                st.subheader("Runtime and compression ratio")
                st.bar_chart(chart_df[["runtime_seconds", "compression_ratio"]])
                st.subheader("Fidelity (lower RMSE / higher SSIM is better)")
                st.bar_chart(chart_df[["mean_rmse", "mean_ssim"]])

            previews = stored_comparison.get("previews", {})
            if previews:
                st.subheader("Reconstructed previews by method")
                preview_cols = st.columns(min(len(previews), 4))
                for column, (label, preview) in zip(preview_cols, previews.items()):
                    _show_image(column, preview, caption=label)

            csv_comparison = comparison_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download method comparison CSV",
                data=csv_comparison,
                file_name="method_comparison.csv",
                mime="text/csv",
            )

            if supabase_configured():
                if st.button(
                    "Save method comparison to Supabase",
                    key="save_comparison_supabase",
                ):
                    try:
                        saved_meta = st.session_state.get("supabase_saved_run")
                        linked_run_id = None
                        if (
                            saved_meta is not None
                            and saved_meta.get("compression_signature")
                            == compression_signature
                        ):
                            linked_run_id = saved_meta["run_id"]
                        comparison_params = {
                            "comparison_signature": comparison_signature,
                            "wavelet": [
                                wavelet_name,
                                wavelet_level,
                                wavelet_keep_fraction,
                            ],
                            "bandwidth_keep_fraction": bandwidth_keep_fraction,
                            "jpeg2000_rate": jpeg2000_rate,
                            "svd_channels": signature_payload["svd_channels"],
                        }
                        saved_comparison = save_method_comparison(
                            results=comparison_df.to_dict(orient="records"),
                            params=comparison_params,
                            run_id=linked_run_id,
                        )
                        st.success(
                            "Comparison saved. Token: "
                            f"`{saved_comparison['share_token']}`"
                        )
                    except Exception as exc:
                        st.error(f"Failed to save comparison: {exc}")
        elif compare_all_methods:
            st.warning(
                "No comparison has been run yet for the current image/settings. "
                "Click **Run all-method comparison**."
            )

    st.subheader("Selected-method diagnostics")
    st.caption("Metadata from the currently selected compression run.")
    st.json(result.metadata)

with tab_matrix:
    st.subheader("Preprocessing matrix")
    st.caption(
        "Optional spectral mixing applied before compression and inverted afterward. "
        "Use identity for raw bands, or NDVI emphasis / a custom matrix when you want "
        "to prioritize Red/NIR combinations."
    )
    mix_df = pd.DataFrame(
        weight_matrix,
        index=[f"out:{name}" for name in band_order],
        columns=[f"in:{name}" for name in band_order],
    )
    _show_dataframe(st, mix_df)

    st.subheader("Source metadata")
    if loaded.metadata:
        metadata_df = pd.DataFrame(
            [{"key": key, "value": str(value)} for key, value in loaded.metadata.items()]
        )
        _show_dataframe(st, metadata_df, hide_index=True)
    else:
        st.info("No source metadata was attached to this upload.")

    st.subheader("Loaded bands")
    band_rows = [
        {
            "band": name,
            "height": int(bands[name].shape[0]),
            "width": int(bands[name].shape[1]),
            "dtype": str(bands[name].dtype),
            "min": float(np.min(bands[name])),
            "max": float(np.max(bands[name])),
        }
        for name in band_order
    ]
    _show_dataframe(st, pd.DataFrame(band_rows), hide_index=True)

with tab_report:
    st.subheader("Analysis-ready summary")
    st.caption(
        "Exportable summary for research notes: compression settings, runtime, "
        "size estimates, band quality, and optional NDVI metrics."
    )
    report_rows = [
        {"metric": "compression_method", "value": method},
        {"metric": "runtime_seconds", "value": result.runtime_seconds},
        {"metric": "original_bytes", "value": result.original_bytes},
        {"metric": "compressed_bytes_estimate", "value": result.compressed_bytes_estimate},
        {"metric": "compression_ratio", "value": ratio},
        {"metric": "band_count", "value": band_count},
        {"metric": "source_type", "value": loaded.source_type},
        {
            "metric": "archive_member",
            "value": archive_member if archive_member is not None else "",
        },
    ]
    if ndvi_run is not None and ndvi_run["signature"] == ndvi_signature:
        ndvi_metrics = ndvi_run["metrics"]
        report_rows.extend(
            [
                {"metric": "ndvi_status", "value": "completed"},
                {"metric": "ndvi_rmse", "value": ndvi_metrics.rmse},
                {"metric": "ndvi_mae", "value": ndvi_metrics.mae},
                {"metric": "ndvi_correlation", "value": ndvi_metrics.correlation},
                {"metric": "ndvi_ssim", "value": ndvi_metrics.ssim},
                {"metric": "ndvi_bias", "value": ndvi_metrics.bias},
            ]
        )
    else:
        report_rows.append({"metric": "ndvi_status", "value": "not_run"})

    stored_comparison = st.session_state.get("method_comparison")
    if (
        stored_comparison is not None
        and stored_comparison["signature"] == comparison_signature
    ):
        report_rows.append({"metric": "all_method_comparison", "value": "completed"})
    else:
        report_rows.append({"metric": "all_method_comparison", "value": "not_run"})

    summary_df = pd.DataFrame(report_rows)
    summary_df["value"] = summary_df["value"].map(str)
    _show_dataframe(st, summary_df, hide_index=True)

    st.subheader("Per-band metrics")
    _show_dataframe(st, report_df, hide_index=True)

    export_df = report_df.copy()
    export_df.insert(0, "method", method)
    export_df.insert(1, "runtime_seconds", result.runtime_seconds)
    csv_data = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download analysis CSV",
        data=csv_data,
        file_name="compression_analysis_report.csv",
        mime="text/csv",
    )
