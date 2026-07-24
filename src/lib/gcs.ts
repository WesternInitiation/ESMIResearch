import { randomUUID } from 'crypto'
import { Storage } from '@google-cloud/storage'

export const VERCEL_PROXY_UPLOAD_BYTES = Math.floor(4 * 1024 * 1024)

function serviceAccountCredentials(): Record<string, unknown> | null {
  const raw = process.env.GOOGLE_SERVICE_ACCOUNT_JSON?.trim()
  if (!raw) return null
  try {
    return JSON.parse(raw) as Record<string, unknown>
  } catch {
    throw new Error(
      'GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON. Paste the full service-account key file contents.',
    )
  }
}

export function gcsUploadBucket(): string | null {
  const name = process.env.GCS_UPLOAD_BUCKET?.trim()
  return name || null
}

export function gcsUploadsConfigured(): boolean {
  return Boolean(gcsUploadBucket() && serviceAccountCredentials())
}

function storageClient(): Storage {
  const credentials = serviceAccountCredentials()
  if (!credentials) {
    throw new Error('GOOGLE_SERVICE_ACCOUNT_JSON is required for GCS uploads')
  }
  const projectId =
    (typeof credentials.project_id === 'string' && credentials.project_id) ||
    process.env.GCP_PROJECT_ID ||
    undefined
  return new Storage({ credentials, projectId })
}

function safeObjectName(filename: string): string {
  const base = filename.split(/[/\\]/).pop() || 'upload.bin'
  return base.replace(/[^\w.\-()+ ]+/g, '_').slice(0, 180) || 'upload.bin'
}

export async function createGcsSignedUpload(input: {
  filename: string
  contentType: string
  size: number
}): Promise<{
  uploadUrl: string
  gcsUri: string
  objectName: string
  bucket: string
  expiresAt: string
}> {
  const bucketName = gcsUploadBucket()
  if (!bucketName) {
    throw new Error('GCS_UPLOAD_BUCKET is not set on Vercel')
  }
  if (input.size <= 0) {
    throw new Error('Upload size must be positive')
  }
  if (input.size > 2 * 1024 * 1024 * 1024) {
    throw new Error('Upload exceeds the ~2 GiB ingest limit')
  }

  const objectName = `uploads/${Date.now()}-${randomUUID()}/${safeObjectName(input.filename)}`
  const contentType = input.contentType || 'application/octet-stream'
  const expires = Date.now() + 60 * 60 * 1000
  const storage = storageClient()
  const file = storage.bucket(bucketName).file(objectName)
  const [uploadUrl] = await file.getSignedUrl({
    version: 'v4',
    action: 'write',
    expires,
    contentType,
  })

  return {
    uploadUrl,
    gcsUri: `gs://${bucketName}/${objectName}`,
    objectName,
    bucket: bucketName,
    expiresAt: new Date(expires).toISOString(),
  }
}
