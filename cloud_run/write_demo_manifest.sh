#!/usr/bin/env bash
# One-shot: scan the demo Landsat TAR and upload manifest.json.
# Usage: ./cloud_run/write_demo_manifest.sh [project-id] [bucket] [archive]
set -euo pipefail

PROJECT_ID="${1:-esmi-research}"
BUCKET="${2:-esmi-research-demo-data}"
ARCHIVE="${3:-LC09_L2SP_016030_20260526_20260527_02_T1.tar}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

gcloud config set project "${PROJECT_ID}" >/dev/null

if ! python3 -c "import google.cloud.storage" 2>/dev/null; then
  pip3 install --user -q google-cloud-storage
fi

python3 cloud_run/write_demo_manifest.py \
  --bucket "${BUCKET}" \
  --archive "${ARCHIVE}"

echo ""
echo "Next: open your Vercel site and click Load demo data."
