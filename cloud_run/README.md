# ESMI compression worker — Google Cloud Run (free tier)

Python FastAPI service that runs the same SVD / wavelet / bandwidth / JPEG2000
stack as the research code, so heavy jobs leave the browser.

## Cost

Cloud Run has a [free monthly allowance](https://cloud.google.com/run/pricing)
(requests, CPU, memory, and egress caps). Idle scale-to-zero means you typically
pay **$0** for light research demos. Always check current free-tier limits in
your GCP billing account.

## One-time GCP setup (free tier)

1. Create a Google account project at [console.cloud.google.com](https://console.cloud.google.com/).
2. Enable **Cloud Run** and **Cloud Build** APIs.
3. Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) and run:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

## Deploy

From the **repository root**:

```bash
gcloud run deploy esmi-compress \
  --source . \
  --dockerfile cloud_run/Dockerfile \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300 \
  --max-instances 3 \
  --set-env-vars "CORS_ORIGINS=*,DEFAULT_MAX_DIM=1024"
```

If your `gcloud` version does not support `--dockerfile`, use:

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/esmi-compress -f cloud_run/Dockerfile .
gcloud run deploy esmi-compress \
  --image gcr.io/YOUR_PROJECT_ID/esmi-compress \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300 \
  --max-instances 3
```

When deploy finishes, copy the service URL, for example:

`https://esmi-compress-xxxxx-uc.a.run.app`

## Connect the Vercel web app

In the Vercel project → Settings → Environment Variables:

| Name | Value |
|------|-------|
| `COMPRESS_API_URL` | `https://esmi-compress-xxxxx-uc.a.run.app` |

Redeploy Vercel. The lab UI gets a **Cloud Run** engine option that proxies
through `/api/compress` (so the browser never needs CORS gymnastics).

Local Next.js:

```bash
# .env.local
COMPRESS_API_URL=http://127.0.0.1:8080
```

## Local test of the API

```bash
docker build -f cloud_run/Dockerfile -t esmi-compress .
docker run --rm -p 8080:8080 esmi-compress

curl -s http://127.0.0.1:8080/health
curl -s -F file=@scene.png -F method=SVD -F svd_rank=24 \
  http://127.0.0.1:8080/v1/compress | head
```

## Notes

- First request after idle can be a **cold start** (10–30s). Later calls are faster.
- Default processing cap is 1024px on the long edge (override with `max_dim`).
- TAR / TAR.GZ uploads are supported via `archive_member`.
- Keep `--max-instances` low on free tier to avoid surprise usage.
