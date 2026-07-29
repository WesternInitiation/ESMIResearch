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

/** Exported for demo catalog / member extract routes. */
export function serviceAccountCredentialsForDemo(): Record<string, unknown> | null {
  return serviceAccountCredentials()
}

export function storageClientForDemo(): Storage {
  return storageClient()
}

export function gcsUploadBucket(): string | null {
  const name = process.env.GCS_UPLOAD_BUCKET?.trim()
  return name || null
}

/** Demo assets bucket (defaults to esmi-research-demo-data). */
export function gcsDemoBucket(): string {
  return (
    process.env.GCS_DEMO_BUCKET?.trim() ||
    'esmi-research-demo-data'
  )
}

import { ARCHIVE_EXT_RE, DEMO_OBJECT_EXT_RE } from '@/lib/imageFormats'

const DEMO_IMAGE_EXT = DEMO_OBJECT_EXT_RE
const DEMO_ARCHIVE_EXT = ARCHIVE_EXT_RE

export type DemoObjectInfo = {
  name: string
  size: number
  contentType: string
  downloadUrl: string
  expiresAt: string
}

export type GcsConfigStatus = {
  configured: boolean
  bucketConfigured: boolean
  authConfigured: boolean
  authValid: boolean
  bucket: string | null
}

/** Safe status probe — never throws (invalid JSON → authValid false). */
export function getGcsConfigStatus(): GcsConfigStatus {
  const bucket = gcsUploadBucket()
  const raw = process.env.GOOGLE_SERVICE_ACCOUNT_JSON?.trim() || ''
  const authConfigured = Boolean(raw)
  let authValid = false
  if (authConfigured) {
    try {
      JSON.parse(raw)
      authValid = true
    } catch {
      authValid = false
    }
  }
  return {
    configured: Boolean(bucket && authConfigured && authValid),
    bucketConfigured: Boolean(bucket),
    authConfigured,
    authValid,
    bucket,
  }
}

export function gcsUploadsConfigured(): boolean {
  return getGcsConfigStatus().configured
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

export async function listDemoObjectsWithSignedUrls(): Promise<{
  bucket: string
  objects: DemoObjectInfo[]
  primary: DemoObjectInfo | null
}> {
  const credentials = serviceAccountCredentials()
  if (!credentials) {
    throw new Error(
      'GOOGLE_SERVICE_ACCOUNT_JSON is required to read demo data from GCS',
    )
  }
  const bucketName = gcsDemoBucket()
  const storage = storageClient()
  const bucket = storage.bucket(bucketName)
  const [files] = await bucket.getFiles({ autoPaginate: true, maxResults: 200 })

  const candidates = files
    .filter((f) => !f.name.endsWith('/'))
    .filter((f) => DEMO_IMAGE_EXT.test(f.name))
    .map((f) => ({
      file: f,
      name: f.name,
      size: Number(f.metadata.size || 0),
      contentType: String(f.metadata.contentType || 'application/octet-stream'),
    }))

  if (!candidates.length) {
    return { bucket: bucketName, objects: [], primary: null }
  }

  // Prefer a TAR archive when present; otherwise the largest image.
  const archives = candidates.filter((c) => DEMO_ARCHIVE_EXT.test(c.name))
  const preferred =
    archives.sort((a, b) => b.size - a.size)[0] ||
    [...candidates].sort((a, b) => b.size - a.size)[0]

  const expires = Date.now() + 60 * 60 * 1000
  const objects: DemoObjectInfo[] = []
  for (const c of candidates) {
    const [downloadUrl] = await c.file.getSignedUrl({
      version: 'v4',
      action: 'read',
      expires,
    })
    objects.push({
      name: c.name,
      size: c.size,
      contentType: c.contentType,
      downloadUrl,
      expiresAt: new Date(expires).toISOString(),
    })
  }

  const primary =
    objects.find((o) => o.name === preferred.name) || objects[0] || null

  return { bucket: bucketName, objects, primary }
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
