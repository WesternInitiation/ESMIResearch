"""Reference Landsat NDVI script (from Renan's-script branch).

Uses the shared ``ndvi`` helpers so Collection 2 scale/offset, fill masking,
and normalized-difference math stay consistent with the lab's NDVI/NDWI tests.
Update ``folder`` / band paths for your local Landsat scene before running.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import rasterio

from ndvi import compute_ndvi, index_stats

# Folder containing the Landsat files
folder = Path(
    r"C:\Users\renan\CS1112 - Fall 2025\ESMI"
    r"\LC09_L2SP_016030_20260526_20260527_02_T1"
)

# File paths
red_file = (
    folder
    / "LC09_L2SP_016030_20260526_20260527_02_T1_SR_B4.TIF"
)
nir_file = (
    folder
    / "LC09_L2SP_016030_20260526_20260527_02_T1_SR_B5.TIF"
)

# Read Band 4 — Red
print("Reading Red band...")
with rasterio.open(red_file) as src:
    red_raw = src.read(1)

# Read Band 5 — Near Infrared
print("Reading NIR band...")
with rasterio.open(nir_file) as src:
    nir_raw = src.read(1)

# Check that both bands have the same dimensions
if red_raw.shape != nir_raw.shape:
    raise ValueError("Red and NIR bands must have the same shape.")

# Landsat C2 DN → SR + DN=0 fill mask + NDVI (same path as the lab index test)
print("Computing NDVI...")
ndvi = compute_ndvi(red_raw, nir_raw, landsat_c2_sr=True)

del red_raw
del nir_raw

# Calculate statistics without creating a large copied array
print("Calculating statistics...")
stats = index_stats(ndvi)

# Print information
print()
print("NDVI shape:", ndvi.shape)
print("Valid pixels:", stats["valid_pixel_count"])
print("Valid pixel fraction:", stats["valid_pixel_fraction"])
print("Minimum:", stats["minimum"])
print("Maximum:", stats["maximum"])
print("Mean:", stats["mean"])
print("Pixels at -1:", stats["pixels_at_negative_one"])
print("Pixels at  1:", stats["pixels_at_positive_one"])

# Create a colormap where invalid pixels appear white
cmap = plt.get_cmap("RdYlGn").copy()
cmap.set_bad(color="white")

# Downsample only for display purposes (stats above already used full resolution)
# Ajuste display_step se ainda der MemoryError: aumente para 8 ou 10
display_step = 5
ndvi_display = ndvi[::display_step, ::display_step]

# Display NDVI
print("Creating NDVI map...")
plt.figure(figsize=(10, 10))
image = plt.imshow(
    ndvi_display,
    cmap=cmap,
    vmin=-1,
    vmax=1,
)
plt.colorbar(image, label="NDVI")
plt.title("Normalized Difference Vegetation Index (NDVI)")
plt.axis("off")
plt.tight_layout()
plt.show()
