"""
ESMI Research — SVD satellite image compression with NDVI benchmarking.

Run: streamlit run app.py
"""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from image_io import load_image, to_display_rgb
from ndvi import compare_ndvi, compute_ndvi
from svd_compression import (
    ChannelCompressionConfig,
    CompressionConfig,
    apply_channel_weight_matrix,
    compress_multiband,
    identity_weight_matrix,
)

st.set_page_config(
    page_title="ESMI SVD Compression",
    page_icon="🛰️",
    layout="wide",
)

st.title("ESMI Research — SVD Satellite Compression")
st.caption(
    "Compress multispectral imagery with truncated SVD, tune per-band priorities, "
    "and benchmark NDVI preservation."
)


@st.cache_data(show_spinner=False)
def _load_uploaded(file_bytes: bytes, filename: str):
    return load_image(io.BytesIO(file_bytes), filename)


def _max_rank(shape: tuple[int, ...]) -> int:
    return min(shape[0], shape[1])


def _render_ndvi(ax, ndvi: np.ndarray, title: str) -> None:
    im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_title(title)
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


uploaded = st.file_uploader(
    "Upload satellite image (GeoTIFF or PNG)",
    type=["tif", "tiff", "png", "jpg", "jpeg"],
)

if uploaded is None:
    st.info(
        "Upload a multispectral GeoTIFF (recommended for NDVI) or a PNG/JPEG. "
        "For NDVI benchmarking, provide **Red** and **NIR** bands."
    )
    st.stop()

loaded = _load_uploaded(uploaded.getvalue(), uploaded.name)
bands = {k: v.copy() for k, v in loaded.bands.items()}
band_order = loaded.band_order
band_count = len(band_order)

with st.sidebar:
    st.header("Compression settings")

    mode = st.radio(
        "Truncation mode",
        options=["rank", "energy"],
        format_func=lambda x: "Fixed rank (k)" if x == "rank" else "Energy fraction",
    )

    normalize = st.checkbox("Normalize bands before SVD", value=True)

    st.subheader("Band mixing matrix")
    use_custom_matrix = st.checkbox("Edit mixing matrix", value=False)
    matrix_preset = st.selectbox(
        "Preset",
        options=["identity", "ndvi_emphasis"],
        format_func=lambda x: {
            "identity": "Identity (no mixing)",
            "ndvi_emphasis": "NDVI emphasis (boost Red & NIR)",
        }[x],
        disabled=use_custom_matrix,
    )

    if use_custom_matrix:
        st.caption("Rows: output bands. Columns: input bands.")
        for row, out_name in enumerate(band_order):
            cols = st.columns(min(band_count, 4))
            for col, in_name in enumerate(band_order):
                with cols[col % len(cols)]:
                    key = f"w_{row}_{col}"
                    default = 1.0 if row == col else 0.0
                    st.number_input(
                        f"{out_name}←{in_name}",
                        key=key,
                        value=default,
                        step=0.1,
                        format="%.2f",
                    )

    st.subheader("Per-band SVD parameters")
    channel_configs: dict[str, ChannelCompressionConfig] = {}

    for name in band_order:
        shape = bands[name].shape
        max_k = _max_rank(shape)
        st.markdown(f"**{name}** `{shape[1]}×{shape[0]}`")

        weight = st.slider(
            f"{name} priority weight",
            min_value=0.1,
            max_value=3.0,
            value=1.5 if name in ("red", "nir") else 1.0,
            step=0.1,
            help="Scales effective rank for this band (higher = more singular values).",
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
            st.caption(f"Effective k after weight: **{effective_rank}**")
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
            st.caption(f"Effective energy target: **{adjusted:.3f}**")

    run_benchmark = st.checkbox("Run rank sweep benchmark", value=False)
    sweep_max = st.slider(
        "Sweep max rank",
        min_value=5,
        max_value=min(128, _max_rank(next(iter(bands.values())).shape)),
        value=64,
        disabled=not run_benchmark,
    )

weight_matrix = _build_weight_matrix(
    band_count, band_order, use_custom_matrix, matrix_preset
)
working_bands = apply_channel_weight_matrix(bands, weight_matrix, band_order)

config = CompressionConfig(
    channels=channel_configs,
    mode=mode,
    normalize_before_svd=normalize,
)

result = compress_multiband(working_bands, config)

col_meta1, col_meta2, col_meta3 = st.columns(3)
ratio = (
    result.total_bytes_compressed_estimate / result.total_bytes_original
    if result.total_bytes_original
    else 0
)
col_meta1.metric("Bands", band_count)
col_meta2.metric("Est. compression ratio", f"{ratio:.2%}")
col_meta3.metric("Source", loaded.source_type.upper())

red_name = st.selectbox(
    "Red band (for NDVI)",
    options=band_order,
    index=band_order.index("red") if "red" in band_order else 0,
)
nir_name = st.selectbox(
    "NIR band (for NDVI)",
    options=band_order,
    index=band_order.index("nir") if "nir" in band_order else min(1, band_count - 1),
)

tab_preview, tab_ndvi, tab_svd, tab_benchmark, tab_matrix = st.tabs(
    ["Preview", "NDVI", "Singular values", "Benchmark", "Matrices"]
)

original_rgb = to_display_rgb(bands, band_order)
compressed_bands = {
    name: result.channels[i].reconstructed for i, name in enumerate(band_order)
}
compressed_rgb = to_display_rgb(compressed_bands, band_order)

with tab_preview:
    c1, c2 = st.columns(2)
    c1.image(original_rgb, caption="Original (RGB preview)", use_container_width=True)
    c2.image(compressed_rgb, caption="Compressed (RGB preview)", use_container_width=True)

    st.subheader("Per-band error")
    rows = []
    for ch in result.channels:
        diff = ch.reconstructed.astype(np.float64) - ch.original.astype(np.float64)
        rows.append(
            {
                "band": ch.name,
                "rank_used": ch.rank_used,
                "energy_retained": f"{ch.energy_retained:.4f}",
                "weight": ch.weight,
                "rmse": float(np.sqrt(np.mean(diff**2))),
                "mae": float(np.mean(np.abs(diff))),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab_ndvi:
    if red_name == nir_name:
        st.warning("Select distinct Red and NIR bands for NDVI.")
    else:
        red_orig = bands[red_name]
        nir_orig = bands[nir_name]
        red_comp = compressed_bands[red_name]
        nir_comp = compressed_bands[nir_name]

        ndvi_orig = compute_ndvi(red_orig, nir_orig)
        ndvi_comp = compute_ndvi(red_comp, nir_comp)
        metrics = compare_ndvi(ndvi_orig, ndvi_comp)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("NDVI RMSE", f"{metrics.rmse:.5f}")
        m2.metric("NDVI MAE", f"{metrics.mae:.5f}")
        m3.metric("Correlation", f"{metrics.correlation:.4f}")
        m4.metric("NDVI SSIM", f"{metrics.ssim:.4f}")

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

with tab_svd:
    selected = st.selectbox("Band to inspect", options=band_order)
    ch = next(c for c in result.channels if c.name == selected)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ch.singular_values, marker="o", markersize=3)
    ax.axvline(ch.rank_used - 1, color="red", linestyle="--", label=f"k = {ch.rank_used}")
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular value")
    ax.set_title(f"Singular value spectrum — {selected}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    energy = np.cumsum(ch.singular_values**2) / np.sum(ch.singular_values**2)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(energy, marker="o", markersize=3)
    ax2.axhline(ch.energy_retained, color="red", linestyle="--", label="Retained")
    ax2.set_xlabel("Rank k")
    ax2.set_ylabel("Cumulative energy fraction")
    ax2.set_title(f"Energy capture — {selected}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    plt.close(fig2)

with tab_benchmark:
    if not run_benchmark:
        st.info("Enable **Run rank sweep benchmark** in the sidebar.")
    elif red_name == nir_name:
        st.warning("Configure distinct Red and NIR bands on the NDVI tab first.")
    else:
        st.subheader("NDVI preservation vs. SVD rank")
        ranks = list(range(1, sweep_max + 1, max(1, sweep_max // 20)))

        def _compress_pair(red, nir, rank):
            red_cfg = ChannelCompressionConfig(rank=rank)
            nir_cfg = ChannelCompressionConfig(rank=rank)
            cfg = CompressionConfig(
                channels={
                    red_name: red_cfg,
                    nir_name: nir_cfg,
                },
                mode="rank",
                normalize_before_svd=normalize,
            )
            subset = {red_name: red, nir_name: nir}
            out = compress_multiband(subset, cfg)
            return (
                out.channels[0].reconstructed,
                out.channels[1].reconstructed,
            )

        ref_ndvi = compute_ndvi(bands[red_name], bands[nir_name])
        bench_rows = []
        for rank in ranks:
            r_c, n_c = _compress_pair(bands[red_name], bands[nir_name], rank)
            m = compare_ndvi(ref_ndvi, compute_ndvi(r_c, n_c))
            bench_rows.append(
                {
                    "rank": rank,
                    "ndvi_rmse": m.rmse,
                    "ndvi_mae": m.mae,
                    "correlation": m.correlation,
                    "ssim": m.ssim,
                }
            )

        df = pd.DataFrame(bench_rows)
        st.line_chart(df.set_index("rank")[["ndvi_rmse", "ndvi_mae"]])
        st.line_chart(df.set_index("rank")[["correlation", "ssim"]])
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download benchmark CSV",
            data=csv,
            file_name="ndvi_svd_benchmark.csv",
            mime="text/csv",
        )

with tab_matrix:
    st.subheader("Band mixing matrix")
    st.caption("Applied before SVD. Identity = compress raw bands.")
    mix_df = pd.DataFrame(
        weight_matrix,
        index=[f"out:{n}" for n in band_order],
        columns=[f"in:{n}" for n in band_order],
    )
    st.dataframe(mix_df, use_container_width=True)

    st.subheader("Effective compression parameters")
    param_rows = []
    for name in band_order:
        cfg = channel_configs[name]
        ch = next(c for c in result.channels if c.name == name)
        param_rows.append(
            {
                "band": name,
                "weight": cfg.weight,
                "rank_setting": cfg.rank,
                "energy_setting": cfg.energy_fraction,
                "rank_used": ch.rank_used,
                "energy_retained": ch.energy_retained,
            }
        )
    st.dataframe(pd.DataFrame(param_rows), use_container_width=True, hide_index=True)
