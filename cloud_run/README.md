# ESMI compression worker — Google Cloud Run (free tier)

Python FastAPI service that runs the same SVD / wavelet / bandwidth / JPEG2000
stack as the research code, so heavy jobs leave the browser.

**Idle scale-to-zero** means light research demos usually stay within Google’s
[Cloud Run free monthly allowance](https://cloud.google.com/run/pricing) (~$0).
Always check your GCP billing page for current free-tier limits.

---

## Step 1 — Create a Google Cloud project

1. Go to [console.cloud.google.com](https://console.cloud.google.com/) and sign in
   with a Google account.
2. Open the project picker (top bar) → **New Project**.
3. Name it something like `esmi-research`, create it, then **select** that project.
4. Copy the **Project ID** (not just the display name). You’ll need it later
   (example: `esmi-research-123456`).

### Enable billing (required, still free-tier eligible)

Cloud Run requires a billing account even when you stay in the free allowance.

1. Go to **Billing** → link a billing account (credit/debit card).
2. Google’s free tier still applies; you won’t be charged for normal light usage
   if you stay under the free quotas. Set a budget alert if you want:
   **Billing → Budgets & alerts → Create budget**.

### Enable the APIs

In the same project:

1. Open **APIs & Services → Library**.
2. Enable:
   - **Cloud Run API**
   - **Cloud Build API**
   - **Artifact Registry API** (or Container Registry, depending on your account)

Or the deploy script will try to enable them for you.

---

## Step 2 — Install and log in with the Google Cloud SDK (`gcloud`)

1. Install the SDK: [cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install)
   - macOS: `brew install --cask google-cloud-sdk` (or the official installer)
   - Windows: use the interactive installer from that page
   - Linux: follow the apt/yum or package instructions on that page
2. Open a terminal and run:

```bash
gcloud auth login
```

Sign in in the browser window that opens.

3. Point gcloud at your project:

```bash
gcloud config set project YOUR_PROJECT_ID
```

Replace `YOUR_PROJECT_ID` with the ID from Step 1.

4. (First time only) allow Cloud Build to deploy to Cloud Run — the deploy
   script handles enabling APIs; if a later deploy fails with permissions,
   grant the Cloud Build service account the **Cloud Run Admin** and
   **Service Account User** roles in **IAM**.

---

## Step 3 — Deploy from this repository

1. Clone or pull the latest `main` of ESMIResearch on your machine.
2. From the **repository root** (the folder that contains `cloud_run/` and `package.json`):

```bash
./cloud_run/deploy.sh YOUR_PROJECT_ID
```

Optional region (default `us-central1`):

```bash
./cloud_run/deploy.sh YOUR_PROJECT_ID us-central1
```

### What the script does

1. Sets the active GCP project  
2. Enables Cloud Run / Cloud Build APIs  
3. Builds the Docker image with Cloud Build (`cloudbuild.yaml` + `cloud_run/Dockerfile`)  
4. Deploys service name **`esmi-compress`** with:
   - 1 GiB memory, 1 CPU  
   - 300s timeout  
   - max 3 instances  
   - unauthenticated invoke (so the Vercel site can call it)  

### When it finishes

The script prints a URL like:

```text
https://esmi-compress-xxxxxxxxx-uc.a.run.app
```

**Copy that URL.** That is your compression backend.

### If `./cloud_run/deploy.sh` won’t run

```bash
chmod +x cloud_run/deploy.sh
./cloud_run/deploy.sh YOUR_PROJECT_ID
```

On Windows, use **Git Bash** or WSL, or run the `gcloud` commands from
[`deploy.sh`](deploy.sh) manually in PowerShell.

### Manual deploy (alternative)

```bash
gcloud builds submit --config cloudbuild.yaml --substitutions=COMMIT_SHA=manual
gcloud run deploy esmi-compress \
  --image gcr.io/YOUR_PROJECT_ID/esmi-compress:latest \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300 \
  --max-instances 3 \
  --set-env-vars "CORS_ORIGINS=*,DEFAULT_MAX_DIM=1024"
```

---

## Step 4 — Connect Vercel to Cloud Run

1. Open your project on [vercel.com](https://vercel.com).
2. Go to **Settings → Environment Variables**.
3. Add:

| Name | Value | Environments |
|------|--------|----------------|
| `NEXT_PUBLIC_COMPRESS_API_URL` | `https://esmi-compress-xxxxx-uc.a.run.app` | Production, Preview, Development |

Use **your** URL from Step 3 (no trailing slash).

Optional (same URL): `COMPRESS_API_URL` — only needed for the Next.js
`/api/compress` proxy; the UI prefers the public URL for large uploads.

4. **Redeploy** so the new env var is baked into the client bundle:
   - **Deployments** → latest deployment → **⋯ → Redeploy**  
   - or push a new commit to `main`

`NEXT_PUBLIC_…` variables are embedded at **build** time. Changing them without
redeploying will not update the live site.

---

## Step 5 — Use it in the app

1. Open your Vercel site.
2. Confirm the header pill says **Cloud Run online** (green).  
   If it says **unset**, the env var wasn’t set or you didn’t redeploy.  
   If it says **offline**, the URL is wrong or the service failed health checks.
3. Upload an image or TAR archive.
4. Set **Engine → Cloud Run**.
5. Optionally raise **Max processing size** (512–2048px).
6. Click **Run on Cloud Run**.

The first request after the service has been idle can take **~15–30 seconds**
(cold start). Later requests are much faster.

---

## Local test (optional, before Vercel)

```bash
# From repo root
docker build -f cloud_run/Dockerfile -t esmi-compress .
docker run --rm -p 8080:8080 esmi-compress
```

In another terminal:

```bash
curl -s http://127.0.0.1:8080/health
```

For local Next.js:

```bash
# .env.local
NEXT_PUBLIC_COMPRESS_API_URL=http://127.0.0.1:8080
npm run dev
```

---

## Troubleshooting

| Symptom | What to try |
|---------|-------------|
| `Permission denied` on deploy | Re-run `gcloud auth login`; confirm project ID; check IAM for Cloud Build |
| Build fails on Dockerfile | Run deploy from **repo root**, not from inside `cloud_run/` |
| Vercel pill: Cloud Run unset | Set `NEXT_PUBLIC_COMPRESS_API_URL` and **redeploy** |
| Vercel pill: offline | Open `YOUR_URL/health` in a browser; fix Cloud Run service / region |
| CORS errors in browser console | Redeploy Cloud Run with `CORS_ORIGINS=*` (deploy script already sets this) |
| Timeout on huge files | Lower max processing size, or raise Cloud Run `--timeout` / memory |
| Unexpected charges | Set a budget alert; keep `--max-instances 3`; delete unused projects |

### Useful commands

```bash
# Service URL
gcloud run services describe esmi-compress --region us-central1 --format='value(status.url)'

# Recent logs
gcloud run services logs read esmi-compress --region us-central1 --limit 50

# Delete the service later (stops compute)
gcloud run services delete esmi-compress --region us-central1
```

---

## Notes

- Default long-edge processing cap inside the API is **1024px** (override with the UI slider / `max_dim`).
- TAR / TAR.GZ uploads are supported via `archive_member`.
- Keep `--max-instances` low on free tier.
- Browser engine still works with **no** Cloud Run setup.
