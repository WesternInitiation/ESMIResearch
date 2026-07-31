#!/usr/bin/env bash
# Deploy the ESMI compression API to Google Cloud Run (free-tier friendly).
# Usage: ./cloud_run/deploy.sh YOUR_GCP_PROJECT_ID [region] [gcs-bucket]
set -euo pipefail

PROJECT_ID="${1:-}"
REGION="${2:-us-central1}"
GCS_BUCKET="${3:-${PROJECT_ID}-esmi-uploads}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "Usage: $0 YOUR_GCP_PROJECT_ID [region] [gcs-bucket]"
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

gcloud config set project "${PROJECT_ID}"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com storage.googleapis.com

IMAGE="gcr.io/${PROJECT_ID}/esmi-compress"

COMMIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo manual)"
echo "Building ${IMAGE} from commit ${COMMIT_SHA} …"
# Ensure local tree includes LZW / latest methods before shipping the image.
if ! grep -q '"LZW"' cloud_run/main.py; then
  echo "error: cloud_run/main.py has no LZW — run 'git pull origin main' and redeploy." >&2
  exit 1
fi
gcloud builds submit --config cloudbuild.yaml --substitutions=COMMIT_SHA="${COMMIT_SHA}"

echo "Deploying Cloud Run service esmi-compress …"
gcloud run deploy esmi-compress \
  --image "${IMAGE}:latest" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 1 \
  --timeout 600 \
  --max-instances 3 \
  --set-env-vars "CORS_ORIGINS=*,DEFAULT_MAX_DIM=1024,MAX_UPLOAD_BYTES=2147483648,DELETE_GCS_AFTER_JOB=1,GCS_UPLOAD_BUCKET=${GCS_BUCKET},GCS_DEMO_BUCKET=esmi-research-demo-data,COMMIT_SHA=${COMMIT_SHA}"

URL="$(gcloud run services describe esmi-compress --region "${REGION}" --format='value(status.url)')"
echo ""
echo "Deployed: ${URL}"
echo "Commit:   ${COMMIT_SHA}"
echo "Set this in Vercel / .env.local:"
echo "  COMPRESS_API_URL=${URL}"
echo "  GOOGLE_SERVICE_ACCOUNT_JSON={...esmi-vercel key...}"
echo "  GCS_UPLOAD_BUCKET=${GCS_BUCKET}"
echo ""
echo "Verify (should show commit=${COMMIT_SHA} and features.nativeRestore=true):"
echo "  curl.exe -s https://esmi-research.vercel.app/api/compress"
echo ""
echo "If the GCS bucket is new, run: ./cloud_run/setup_gcs.sh ${PROJECT_ID} ${GCS_BUCKET} ${REGION}"
