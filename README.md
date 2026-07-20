# ESMIResearch

Streamlit workbench for applying SVD, wavelet, bandwidth-domain, and JPEG2000
compression to satellite imagery, with optional Supabase storage for shared runs.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your Supabase project URL + service role key
streamlit run app.py
```

Upload a GeoTIFF, PNG, JPEG, WebP, or TAR/TAR.GZ archive. When an archive contains
multiple supported images, select the image to process, configure the algorithm, and
choose **Run compression**. The reconstructed compressed image can then be downloaded
as GeoTIFF (with source georeferencing) or PNG.

NDVI preservation testing is deliberately separate from compression. Open the NDVI
tab, choose distinct Red and NIR bands, confirm the test, and run it explicitly.

## Supabase setup (storage + shared results)

1. Create a free project at [supabase.com](https://supabase.com).
2. In the SQL editor, run [`supabase_schema/migrations/001_init.sql`](supabase_schema/migrations/001_init.sql).
3. In Storage, create a **private** bucket named `esmi-images`.
4. Copy Project URL and `service_role` key into Streamlit secrets
   (see [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example)).

Once configured, the app can:

- **Save run to Supabase** after compression (uploads original + compressed artifacts)
- Attach **NDVI** metrics to a saved run
- Save **all-method comparison** tables
- Share via `?run=<share_token>` and browse recent runs in **Shared runs (Supabase)**

## Deploy on Streamlit Community Cloud (free)

Streamlit Community Cloud hosts this app for free at a `*.streamlit.app` URL.
Vercel cannot host Streamlit/Python apps; Community Cloud is the intended free path.

1. Push this repository to GitHub.
2. Open [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. **Create app** → select `ESMIResearch`, branch `main`, main file `app.py`.
4. Under Advanced / Secrets, paste:

```toml
[supabase]
url = "https://YOUR_PROJECT.supabase.co"
service_role_key = "YOUR_SERVICE_ROLE_KEY"
bucket = "esmi-images"

app_base_url = "https://YOUR_APP.streamlit.app"
```

5. Deploy, then set `app_base_url` to the assigned Streamlit URL if needed and reboot.

Free-tier notes:

- Apps may sleep after inactivity
- Memory is limited (~1 GB); prefer smaller scenes for cloud demos
- Keep the service role key in Cloud secrets only — never commit it

## Project layout

| Path | Role |
|------|------|
| `app.py` | Streamlit UI |
| `compression/` | SVD / wavelet / bandwidth / JPEG2000 implementations |
| `persistence.py` | Supabase save/load helpers |
| `supabase_client.py` | Client from Streamlit secrets |
| `supabase_schema/migrations/` | Database schema SQL |
| `packages.txt` | System deps for Community Cloud (GDAL) |
