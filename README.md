# ESMIResearch

Satellite image compression research workbench: SVD, wavelet, bandwidth-domain,
and JPEG2000 methods with NDVI preservation checks and optional Supabase sharing.

**Primary UI: Vercel** (Next.js at repo root). Compression can run in the
**browser** (Web Worker) or on **Google Cloud Run** (Python free tier) for heavier jobs.

The Streamlit prototype (`streamlit_app.py`) remains for local research notebooks.

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
| `NEXT_PUBLIC_COMPRESS_API_URL` | your Cloud Run URL (optional; enables **Cloud Run** engine) |

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
2. Copy the service URL into Vercel as `NEXT_PUBLIC_COMPRESS_API_URL`.
3. In the lab UI, set **Engine → Cloud Run** and click **Run on Cloud Run**.

The browser talks to Cloud Run directly (CORS enabled) so large uploads are not
blocked by Vercel’s serverless body size limit.

## Supabase setup

1. Create a project at [supabase.com](https://supabase.com).
2. Run [`supabase_schema/migrations/001_init.sql`](supabase_schema/migrations/001_init.sql) in the SQL editor.
3. Create a **private** Storage bucket named `esmi-images`.
4. Put the project URL + service role key in Vercel env vars (or `.env.local`).

## Features

- Upload GeoTIFF, PNG, JPEG, or **TAR / TAR.GZ** archives (**up to ~2 GiB** in the browser / Streamlit)
- Engines: **Browser** (Web Worker, large files) or **Cloud Run** (Python; direct HTTP uploads ~30 MB max due to platform limits)
- Methods: SVD, wavelet, bandwidth, JPEG2000
- Per-band RMSE / MAE / PSNR / SSIM + optional NDVI
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
