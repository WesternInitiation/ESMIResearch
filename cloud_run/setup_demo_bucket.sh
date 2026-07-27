#!/usr/bin/env bash
# Allow the Vercel service account to read demo assets + browser CORS for signed GETs.
# Usage: ./cloud_run/setup_demo_bucket.sh [project-id] [bucket]
set -euo pipefail

PROJECT_ID="${1:-esmi-research}"
BUCKET="${2:-esmi-research-demo-data}"

gcloud config set project "${PROJECT_ID}"

SA_EMAIL="esmi-vercel@${PROJECT_ID}.iam.gserviceaccount.com"

echo "Granting objectViewer to ${SA_EMAIL} on gs://${BUCKET}…"
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.objectViewer"

echo "Allowing ${SA_EMAIL} to sign blobs (needed for V4 signed URLs)…"
gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/iam.serviceAccountTokenCreator" \
  || true

TMP_CORS="$(mktemp)"
cat >"${TMP_CORS}" <<'JSON'
[
  {
    "origin": ["*"],
    "method": ["GET", "HEAD", "OPTIONS"],
    "responseHeader": ["Content-Type", "Content-Length", "Content-Range", "Accept-Ranges"],
    "maxAgeSeconds": 3600
  }
]
JSON
echo "Setting CORS on gs://${BUCKET}…"
gcloud storage buckets update "gs://${BUCKET}" --cors-file="${TMP_CORS}"
rm -f "${TMP_CORS}"

echo "Granting objectCreator to ${SA_EMAIL} on gs://${BUCKET} (so Load demo can write manifest.json)…"
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.objectCreator" || true

# Cloud Run needs objectViewer (+ create) to scan/write the demo TAR manifest.
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)' 2>/dev/null || true)"
if [[ -n "${PROJECT_NUMBER}" ]]; then
  RUN_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
  echo "Granting objectAdmin to Cloud Run runtime SA ${RUN_SA} on gs://${BUCKET}…"
  gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
    --member="serviceAccount:${RUN_SA}" \
    --role="roles/storage.objectAdmin" || true
fi

echo ""
echo "Done. For large TARs, build the listing index once:"
echo "  ./cloud_run/write_demo_manifest.sh ${PROJECT_ID} ${BUCKET}"
echo "Then redeploy Vercel if needed and click Load demo data."
echo "Quick checks:"
echo "  gcloud storage ls gs://${BUCKET}"
echo "  Open https://YOUR-SITE.vercel.app/api/demo"
