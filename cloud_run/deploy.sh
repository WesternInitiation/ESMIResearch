#!/usr/bin/env bash
# Deploy the ESMI compression API to Google Cloud Run (free-tier friendly).
# Usage: ./cloud_run/deploy.sh YOUR_GCP_PROJECT_ID [region] [gcs-bucket]
#
# Windows tip: prefer PowerShell  →  .\cloud_run\deploy.ps1 esmi-research
# or:  bash ./cloud_run/deploy.sh esmi-research   (forward slashes; LF line endings)

# Normalize CRLF if the script was checked out on Windows.
if [ -n "${BASH_VERSION:-}" ]; then
  set -eu
  # pipefail is bash-only; ignore if unsupported
  set -o pipefail 2>/dev/null || true
fi

PROJECT_ID="${1:-}"
REGION="${2:-us-central1}"
GCS_BUCKET="${3:-${PROJECT_ID}-esmi-uploads}"

if [ -z "${PROJECT_ID}" ]; then
  echo "Usage: $0 YOUR_GCP_PROJECT_ID [region] [gcs-bucket]"
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

gcloud config set project "${PROJECT_ID}"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com storage.googleapis.com

IMAGE="gcr.io/${PROJECT_ID}/esmi-compress"

COMMIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo manual)"
echo "Building ${IMAGE} from commit ${COMMIT_SHA} ..."

if ! grep -q '"LZW"' cloud_run/main.py; then
  echo "error: cloud_run/main.py has no LZW — run 'git pull origin main' and redeploy." >&2
  exit 1
fi
if ! grep -q 'LZW_SAFE_MAX_DIM' cloud_run/main.py; then
  echo "error: cloud_run/main.py missing LZW_SAFE_MAX_DIM — run 'git pull origin main' and redeploy." >&2
  exit 1
fi
if ! grep -q '/v1/compress/jobs' cloud_run/main.py; then
  echo "error: cloud_run/main.py missing /v1/compress/jobs — run 'git pull origin main' and redeploy." >&2
  exit 1
fi

gcloud builds submit --config cloudbuild.yaml --substitutions=COMMIT_SHA="${COMMIT_SHA}"

echo "Deploying Cloud Run service esmi-compress ..."
gcloud run deploy esmi-compress \
  --image "${IMAGE}:latest" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --memory 8Gi \
  --cpu 4 \
  --no-cpu-throttling \
  --timeout 600 \
  --max-instances 3 \
  --concurrency 1 \
  --set-env-vars "CORS_ORIGINS=*,DEFAULT_MAX_DIM=2048,MAX_UPLOAD_BYTES=2147483648,DELETE_GCS_AFTER_JOB=1,LZW_SAFE_MAX_DIM=4096,GCS_UPLOAD_BUCKET=${GCS_BUCKET},GCS_DEMO_BUCKET=esmi-research-demo-data,COMMIT_SHA=${COMMIT_SHA}"

URL="$(gcloud run services describe esmi-compress --region "${REGION}" --format='value(status.url)')"
echo ""
echo "Deployed: ${URL}"
echo "Commit:   ${COMMIT_SHA}"
echo "Verify:"
echo "  curl.exe -s https://esmi-research.vercel.app/api/compress"
echo "Expect features.asyncJobs=true and a new commit sha."
