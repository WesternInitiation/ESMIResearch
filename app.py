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
) -> pd.DataFrame:
    rows = []
    for label, runner in runners.items():
        try:
            result = _recompute_channel_report(runner(), original_bands)
            ratio = (
                result.compressed_bytes_estimate / result.original_bytes
                if result.original_bytes
                else 0.0
            )
            report_df = _report_table(result)
            rows.append(
                {
                    "method": label,
                    "runtime_seconds": result.runtime_seconds,
                    "compression_ratio": ratio,
                    "compressed_bytes_estimate": result.compressed_bytes_estimate,
                    "mean_rmse": float(report_df["rmse"].mean()),
                    "mean_psnr_db": float(report_df["psnr_db"].replace(np.inf, np.nan).mean()),
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
                    "mean_psnr_db": np.nan,
                    "mean_ssim": np.nan,
                    "status": str(exc),
                }
            )
    return pd.DataFrame(rows)


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
    compare_all_methods = st.checkbox("Enable all-method runtime comparison", value=False)

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

tab_overview, tab_ndvi, tab_methods, tab_matrix, tab_report = st.tabs(
    ["Overview", "NDVI", "Method Comparison", "Matrices", "Analysis Report"]
)

with tab_overview:
    st.subheader("Side-by-side image comparison")
    c1, c2 = st.columns(2)
    c1.image(original_rgb, caption="Original preview", use_container_width=True)
    c2.image(compressed_rgb, caption="Compressed preview", use_container_width=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(compressed_rgb)
    axes[1].set_title("Compressed")
    axes[1].axis("off")
    _render_error_map(axes[2], original_rgb, compressed_rgb)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Per-band quality metrics")
    st.dataframe(report_df, use_container_width=True, hide_index=True)

ndvi_run = st.session_state.get("ndvi_run")
with tab_ndvi:
    st.subheader("Optional NDVI preservation test")
    st.caption(
        "NDVI is not run during compression. Select the bands and explicitly confirm "
        "this separate test when you are ready."
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
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("NDVI has not been run for this compression result and band selection.")

with tab_methods:
    st.subheader("Runtime and fidelity across compression methods")
    if compare_all_methods:
        comparison_df = _comparison_runner(
            {
                "SVD": lambda: _run_selected_method(
                    "SVD",
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
                "Wavelet transformation": lambda: _run_selected_method(
                    "Wavelet transformation",
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
                "Bandwidth transformation": lambda: _run_selected_method(
                    "Bandwidth transformation",
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
                "JPEG2000": lambda: _run_selected_method(
                    "JPEG2000",
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
            },
            bands,
        )
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        numeric_df = comparison_df.dropna(subset=["runtime_seconds", "compression_ratio"])
        if not numeric_df.empty:
            st.line_chart(
                numeric_df.set_index("method")[["runtime_seconds", "compression_ratio"]]
            )
    else:
        st.info("Enable all-method comparison in the sidebar to benchmark every method.")

    st.subheader("Method diagnostics")
    st.json(result.metadata)

with tab_matrix:
    st.subheader("Preprocessing matrix")
    st.caption("The matrix is applied before compression and inverted after reconstruction.")
    mix_df = pd.DataFrame(
        weight_matrix,
        index=[f"out:{name}" for name in band_order],
        columns=[f"in:{name}" for name in band_order],
    )
    st.dataframe(mix_df, use_container_width=True)

    st.subheader("Source metadata")
    metadata_df = pd.DataFrame(
        [{"key": key, "value": str(value)} for key, value in loaded.metadata.items()]
    )
    st.dataframe(metadata_df, use_container_width=True, hide_index=True)

with tab_report:
    st.subheader("Analysis-ready summary")
    report_rows = [
        {"metric": "compression_method", "value": method},
        {"metric": "runtime_seconds", "value": result.runtime_seconds},
        {"metric": "original_bytes", "value": result.original_bytes},
        {"metric": "compressed_bytes_estimate", "value": result.compressed_bytes_estimate},
        {"metric": "compression_ratio", "value": ratio},
        {"metric": "band_count", "value": band_count},
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

    summary_df = pd.DataFrame(report_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.dataframe(report_df, use_container_width=True, hide_index=True)

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
