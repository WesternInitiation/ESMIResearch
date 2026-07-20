from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio

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
    red_raw = src.read(1).astype(np.float32)

# Read Band 5 — Near Infrared
print("Reading NIR band...")
with rasterio.open(nir_file) as src:
    nir_raw = src.read(1).astype(np.float32)

# Check that both bands have the same dimensions
if red_raw.shape != nir_raw.shape:
    raise ValueError("Red and NIR bands must have the same shape.")

# Create nodata/fill mask before applying scale and offset
# Landsat Collection 2 Surface Reflectance uses DN = 0 as fill value
nodata_mask = (red_raw == 0) | (nir_raw == 0)

# Landsat Collection 2 Level-2 scale and offset
scale_factor = np.float32(0.0000275)
offset = np.float32(-0.2)

# Convert digital numbers to surface reflectance
print("Converting bands to surface reflectance...")
red = red_raw * scale_factor + offset
nir = nir_raw * scale_factor + offset

# Raw arrays are no longer needed
del red_raw
del nir_raw

# Compute NDVI
print("Computing NDVI...")
denominator = nir + red
numerator = nir - red
ndvi = np.full(
    red.shape,
    np.nan,
    dtype=np.float32,
)
np.divide(
    numerator,
    denominator,
    out=ndvi,
    where=np.abs(denominator) > np.float32(1e-10),
)

# Temporary arrays are no longer needed
del numerator
del denominator
del red
del nir

# Keep NDVI inside its expected range
np.clip(
    ndvi,
    np.float32(-1.0),
    np.float32(1.0),
    out=ndvi,
)

# Remove nodata/fill pixels from the NDVI image
ndvi[nodata_mask] = np.nan
del nodata_mask

# Calculate statistics without creating a large copied array
print("Calculating statistics...")
valid_mask = np.isfinite(ndvi)
valid_pixel_count = np.count_nonzero(valid_mask)
valid_pixel_fraction = valid_pixel_count / ndvi.size
minimum = float(np.nanmin(ndvi))
maximum = float(np.nanmax(ndvi))
mean = float(np.nanmean(ndvi))
pixels_at_negative_one = np.count_nonzero(ndvi == -1.0)
pixels_at_positive_one = np.count_nonzero(ndvi == 1.0)

# valid_mask is no longer needed
del valid_mask

# Print information
print()
print("NDVI shape:", ndvi.shape)
print("Valid pixels:", valid_pixel_count)
print("Valid pixel fraction:", valid_pixel_fraction)
print("Minimum:", minimum)
print("Maximum:", maximum)
print("Mean:", mean)
print("Pixels at -1:", pixels_at_negative_one)
print("Pixels at  1:", pixels_at_positive_one)

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