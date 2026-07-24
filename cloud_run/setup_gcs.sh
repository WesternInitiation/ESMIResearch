#!/usr/bin/env bash
# Create a private GCS bucket for large Cloud Run uploads (80–100+ MB).
# Usage: ./cloud_run/setup_gcs.sh YOUR_GCP_PROJECT_ID [bucket-name] [region]
set -euo pipefail

PROJECT_ID="${1:-}"
BUCKET="${2:-${PROJECT_ID}-esmi-uploads}"
REGION="${3:-us-central1}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "Usage: $0 YOUR_GCP_PROJECT_ID [bucket-name] [region]"
  exit 1
fi

gcloud config set project "${PROJECT_ID}"
gcloud services enable storage.googleapis.com iam.googleapis.com

if gsutil ls -b "gs://${BUCKET}" >/dev/null 2>&1; then
  echo "Bucket gs://${BUCKET} already exists"
else
  echo "Creating gs://${BUCKET} …"
  gcloud storage buckets create "gs://${BUCKET}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --uniform-bucket-level-access
fi

# Lifecycle: delete staged uploads after 7 days
TMP_LC="$(mktemp)"
cat >"${TMP_LC}" <<'JSON'
{
  "rule": [
    {
      "action": { "type": "Delete" },
      "condition": { "age": 7 }
    }
  ]
}
JSON
gcloud storage buckets update "gs://${BUCKET}" --lifecycle-file="${TMP_LC}"
rm -f "${TMP_LC}"

# CORS so the browser can PUT with signed URLs
TMP_CORS="$(mktemp)"
cat >"${TMP_CORS}" <<'JSON'
[
  {
    "origin": ["*"],
    "method": ["GET", "HEAD", "PUT", "OPTIONS"],
    "responseHeader": ["Content-Type", "x-goog-resumable"],
    "maxAgeSeconds": 3600
  }
]
JSON
gcloud storage buckets update "gs://${BUCKET}" --cors-file="${TMP_CORS}"
rm -f "${TMP_CORS}"

SA_EMAIL="esmi-vercel@${PROJECT_ID}.iam.gserviceaccount.com"
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

echo "Granting objectAdmin to ${SA_EMAIL} (sign + write)…"
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.objectAdmin"

echo "Allowing ${SA_EMAIL} to sign blobs…"
gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/iam.serviceAccountTokenCreator"

echo "Granting objectViewer to Cloud Run runtime SA ${RUNTIME_SA}…"
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/storage.objectViewer"

echo ""
echo "Done. Set on Vercel:"
echo "  GCS_UPLOAD_BUCKET=${BUCKET}"
echo "Redeploy Vercel after saving the env var."
echo ""
echo "Also redeploy Cloud Run so it can download objects:"
echo "  ./cloud_run/deploy.sh ${PROJECT_ID} ${REGION}"
