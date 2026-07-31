# Deploy esmi-compress to Cloud Run from Windows PowerShell (no bash required).
# Usage:  .\cloud_run\deploy.ps1 esmi-research
# Optional: .\cloud_run\deploy.ps1 esmi-research us-central1 esmi-research-esmi-uploads

param(
  [Parameter(Mandatory = $true)][string]$ProjectId,
  [string]$Region = "us-central1",
  [string]$GcsBucket = ""
)

$ErrorActionPreference = "Stop"

if (-not $GcsBucket) {
  $GcsBucket = "$ProjectId-esmi-uploads"
}

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

Write-Host "Repo: $Root"
git pull origin main

$CommitSha = (git rev-parse --short HEAD).Trim()
if (-not $CommitSha) { $CommitSha = "manual" }

$MainPy = Get-Content -Raw (Join-Path $Root "cloud_run\main.py")
if ($MainPy -notmatch '"LZW"') {
  throw "cloud_run/main.py has no LZW — git pull origin main and retry."
}
if ($MainPy -notmatch 'LZW_SAFE_MAX_DIM') {
  throw "cloud_run/main.py missing LZW_SAFE_MAX_DIM — git pull origin main and retry."
}

Write-Host "Building gcr.io/$ProjectId/esmi-compress from commit $CommitSha ..."
gcloud config set project $ProjectId
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com storage.googleapis.com
gcloud builds submit --config cloudbuild.yaml --substitutions="COMMIT_SHA=$CommitSha"

$EnvVars = "CORS_ORIGINS=*,DEFAULT_MAX_DIM=1024,MAX_UPLOAD_BYTES=2147483648,DELETE_GCS_AFTER_JOB=1,LZW_SAFE_MAX_DIM=4096,GCS_UPLOAD_BUCKET=$GcsBucket,GCS_DEMO_BUCKET=esmi-research-demo-data,COMMIT_SHA=$CommitSha"

Write-Host "Deploying Cloud Run service esmi-compress ..."
gcloud run deploy esmi-compress `
  --image "gcr.io/${ProjectId}/esmi-compress:latest" `
  --region $Region `
  --platform managed `
  --allow-unauthenticated `
  --memory 4Gi `
  --cpu 1 `
  --timeout 600 `
  --max-instances 3 `
  --set-env-vars $EnvVars

$Url = gcloud run services describe esmi-compress --region $Region --format="value(status.url)"
Write-Host ""
Write-Host "Deployed: $Url"
Write-Host "Commit:   $CommitSha"
Write-Host "Verify:   curl.exe -s https://esmi-research.vercel.app/api/compress"
Write-Host "Expect commit=$CommitSha (LZW auto-cap is in this image)."
