# ESMIResearch

Satellite image compression research workbench: SVD, wavelet, bandwidth-domain,
and JPEG-rate methods with NDVI preservation checks and optional Supabase sharing.

**Primary deploy target: Vercel** (Next.js app at the repo root). Compression runs
in the browser — no Python/GDAL runtime on the server.

The Streamlit prototype (`streamlit_app.py`) remains for local research notebooks
and offline GeoTIFF workflows.

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

5. Deploy. Open the assigned `*.vercel.app` URL.

If an older Vercel project still has **Root Directory** set to `web`, clear it
(set to `.`) and redeploy — that folder no longer exists.

Local preview:

```bash
cp .env.example .env.local   # fill in Supabase values
npm install
npm run dev
```

## Supabase setup

1. Create a project at [supabase.com](https://supabase.com).
2. Run [`supabase_schema/migrations/001_init.sql`](supabase_schema/migrations/001_init.sql) in the SQL editor.
3. Create a **private** Storage bucket named `esmi-images`.
4. Put the project URL + service role key in Vercel env vars (or `.env.local`).

The Next.js API routes (`/api/runs`, etc.) use the service role on the server so
the key is never shipped to the browser. After a compression run you can **Save
run to Supabase** and share with `?run=<share_token>`.

## Features (Vercel)

- Upload GeoTIFF (via `geotiff.js`), PNG, or JPEG
- Methods: SVD, Haar wavelet, FFT bandwidth keep, JPEG quality stand-in for JPEG2000
- Per-band RMSE / MAE / PSNR / SSIM
- NDVI preservation compare (choose Red / NIR bands)
- Compare all methods table
- Optional Supabase save + shared run links

Notes:

- True JPEG2000 isn't widely available in browsers; the JPEG2000 option uses a
  quality-controlled JPEG encode/decode while keeping the same lab workflow.
- Large GeoTIFFs are heavy in-browser (especially SVD). Prefer modest scenes for demos.

## Streamlit prototype (local / research)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
streamlit run streamlit_app.py
```

Streamlit Community Cloud is no longer the intended host. Prefer Vercel for demos;
keep Streamlit for local deep research with native GeoTIFF write-back.

## Project layout

| Path | Role |
|------|------|
| `src/` | Next.js app — Vercel entrypoint |
| `src/lib/compression/` | TypeScript SVD / wavelet / bandwidth / JPEG |
| `src/app/api/` | Supabase save/list/load API routes |
| `streamlit_app.py` | Streamlit UI (research / local) |
| `compression/` | Python compression implementations |
| `supabase_schema/` | SQL migrations |
