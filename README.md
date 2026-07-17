# ESMIResearch

Streamlit workbench for applying SVD, wavelet, bandwidth-domain, and JPEG2000
compression to satellite imagery.

## Run

```bash
streamlit run app.py
```

Upload a GeoTIFF, PNG, JPEG, WebP, or TAR/TAR.GZ archive. When an archive contains
multiple supported images, select the image to process, configure the algorithm, and
choose **Run compression**. The reconstructed compressed image can then be downloaded
as GeoTIFF (with source georeferencing) or PNG.

NDVI preservation testing is deliberately separate from compression. Open the NDVI
tab, choose distinct Red and NIR bands, confirm the test, and run it explicitly.