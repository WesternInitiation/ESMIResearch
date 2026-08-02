# ESMIResearch

- Next.js satellite image compression research workbench
- Upload formats: GeoTIFF, PNG, JPEG, and TAR archives
- Compression methods: SVD, wavelet, bandwidth-domain, LZW, and JPEG2000
- Engines: browser (Web Worker) or Google Cloud Run Python backend for larger jobs
- Evaluation metrics: compression ratio, RMSE, MAE, PSNR, and SSIM
- Optional NDVI/NDWI preservation checks against the original bands
- Defaults: processing size Native; engine Cloud Run when configured

**Primary UI: Vercel** (Next.js at repo root). The Streamlit prototype
(`streamlit_app.py`) remains for local research notebooks.

## Default compression settings

These are the lab UI defaults (`DEFAULT_PARAMS` in
`src/components/CompressionLab.tsx`) sent to both the browser worker and Cloud
Run unless you change the sliders. Processing size defaults to **Native**
(no downsampling); engine defaults to **Cloud Run** when configured.

| Method | Default settings |
|--------|------------------|
| **SVD** | Fixed rank `k = 24` (slider range 1–64). Per-band truncated SVD. |
| **Wavelet** | Family `db4`, keep fraction `0.08`, levels `3`. LL coefficients are always retained; keep fraction budgets the remaining detail coeffs. Browser engine uses multilevel Haar; Cloud Run uses the selected family (default `db4`). |
| **Bandwidth** | Low-frequency keep fraction `0.12` (centered FFT passband). |
| **JPEG2000** | Quality-like rate `0.45` (0.1–0.95). Browser uses a JPEG encode/decode stand-in at that quality; Cloud Run maps it to an OpenJPEG rate ≈ `round((1 − 0.45) × 40 + 1) = 23`. |
| **LZW** | No tunable knobs. Each band is min–max quantized to uint8, then classic LZW with max dictionary size `4096` (12-bit codes). |

Python module fallbacks (if a call omits params) match these for wavelet /
bandwidth / JPEG quality; Cloud Run’s Form default for SVD rank is `32` when
the field is absent, but the Next.js UI always sends `24`.

## Deploy on Vercel

1. Push this repository to GitHub.
2. In [vercel.com](https://vercel.com): **Add New Project** → import the repo.
3. Leave **Root Directory** as `.` (repo root). Framework should detect **Next.js**.
4. Add environment variables:

| Name | Value |
|------|-------|
| `NEXT_PUBLIC_SUPABASE_URL` | `https://YOUR_PROJECT.supabase.co` |
| `SUPABASE_SERVICE_ROLE_KEY` | your Supabase **service_role** key (server-only) |
| `NEXT_PUBLIC_SUPABASE_BUCKET` | `esmi-images` (optional; default) |
| `COMPRESS_API_URL` | your Cloud Run URL (optional; enables **Cloud Run** engine) |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | full JSON key for a SA with Cloud Run Invoker (required if the service is private) |
| `GCS_UPLOAD_BUCKET` | GCS bucket for large Cloud Run uploads (80–100+ MB) |

5. Deploy. Open the assigned `*.vercel.app` URL.

Local preview:

```bash
cp .env.example .env.local   # fill in values
npm install
npm run dev
```

## Google Cloud Run (free tier) — optional backend

Heavy compression (larger GeoTIFF/TAR, real Python JPEG2000 when available) can run
on Cloud Run. Idle services scale to zero, so light research use is typically **$0**
within Google’s free allowance.

1. Follow [`cloud_run/README.md`](cloud_run/README.md) (or run `./cloud_run/deploy.sh YOUR_PROJECT_ID`).
2. If your org blocks public (`allUsers`) access, use the **private + service account**
   path in that README: set `COMPRESS_API_URL`, `GOOGLE_SERVICE_ACCOUNT_JSON`, and
   `GCS_UPLOAD_BUCKET` on Vercel (run `./cloud_run/setup_gcs.sh` once).
3. In the lab UI, set **Engine → Cloud Run** and click **Run on Cloud Run**.

Small jobs go through `/api/compress`. Larger files (80–100+&nbsp;MB) upload to GCS with a
signed URL, then Cloud Run downloads `gcs_uri`. Method / NDVI compare stay in the browser.

## Supabase setup

1. Create a project at [supabase.com](https://supabase.com).
2. Run [`supabase_schema/migrations/001_init.sql`](supabase_schema/migrations/001_init.sql) in the SQL editor.
3. Create a **private** Storage bucket named `esmi-images`.
4. Put the project URL + service role key in Vercel env vars (or `.env.local`).

## Features

- Upload GeoTIFF (`.tif` / `.tiff`), PNG, JPEG, WebP, BMP, GIF, JPEG 2000, or **TAR / TAR.GZ / ZIP** archives of band files (**up to ~2 GiB** in the browser / Streamlit)
- Engines: **Browser** (Web Worker, large files) or **Cloud Run** (Python; direct HTTP uploads ~30 MB max due to platform limits)
- Methods: SVD, wavelet, bandwidth, JPEG2000, LZW (see [defaults](#default-compression-settings))
- Per-band RMSE / MAE / PSNR / SSIM + optional NDVI/NDWI
- Optional Supabase save + shared run links

## Streamlit prototype (local / research)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
streamlit run streamlit_app.py
```

## Project layout

| Path | Role |
|------|------|
| `src/` | Next.js app — Vercel entrypoint |
| `src/lib/compression/` | TypeScript browser compression |
| `cloud_run/` | FastAPI worker for Google Cloud Run |
| `compression/` | Python compression implementations |
| `streamlit_app.py` | Streamlit UI (research / local) |
| `supabase_schema/` | SQL migrations |
