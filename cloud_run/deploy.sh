#!/usr/bin/env bash
# Deploy the ESMI compression API to Google Cloud Run (free-tier friendly).
# Usage: ./cloud_run/deploy.sh YOUR_GCP_PROJECT_ID [region]
set -euo pipefail

PROJECT_ID="${1:-}"
REGION="${2:-us-central1}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "Usage: $0 YOUR_GCP_PROJECT_ID [region]"
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

gcloud config set project "${PROJECT_ID}"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

IMAGE="gcr.io/${PROJECT_ID}/esmi-compress"

echo "Building ${IMAGE} …"
gcloud builds submit --config cloudbuild.yaml --substitutions=COMMIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo manual)"

echo "Deploying Cloud Run service esmi-compress …"
gcloud run deploy esmi-compress \
  --image "${IMAGE}:latest" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300 \
  --max-instances 3 \
  --set-env-vars "CORS_ORIGINS=*,DEFAULT_MAX_DIM=1024"

URL="$(gcloud run services describe esmi-compress --region "${REGION}" --format='value(status.url)')"
echo ""
echo "Deployed: ${URL}"
echo "Set this in Vercel / .env.local:"
echo "  NEXT_PUBLIC_COMPRESS_API_URL=${URL}"
echo "  COMPRESS_API_URL=${URL}   # optional proxy"
