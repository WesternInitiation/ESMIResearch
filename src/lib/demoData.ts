import { readJsonResponse } from '@/lib/httpJson'

/** Client helpers for lazy demo loading from GCS (list first, download one member). */

export type DemoCatalogResponse =
  | {
      kind: 'archive'
      bucket: string
      objectName: string
      archiveName: string
      members: string[]
    }
  | {
      kind: 'objects'
      bucket: string
      members: string[]
      objects: Array<{ name: string; size: number }>
    }

export type GcsBucketListResponse = {
  buckets: Array<{ name: string; location?: string }>
  allowed: string[]
  defaultBucket: string
}

function networkError(stage: string, err: unknown): Error {
  const raw = err instanceof Error ? err.message : String(err)
  if (/failed to fetch|networkerror|load failed/i.test(raw)) {
    if (stage === 'download') {
      return new Error(
        'Browser blocked the staged demo download (CORS). Ensure GCS_UPLOAD_BUCKET has CORS GET for origin * (see cloud_run/setup_gcs.sh).',
      )
    }
    return new Error(
      `Could not reach the demo API (${stage}). Confirm the latest Vercel deploy is live.`,
    )
  }
  return err instanceof Error ? err : new Error(raw || `Demo ${stage} failed`)
}

export async function fetchGcsBuckets(
  onProgress?: (message: string) => void,
): Promise<GcsBucketListResponse> {
  onProgress?.('Listing Cloud Storage buckets…')
  let res: Response
  try {
    res = await fetch('/api/gcs/buckets', { cache: 'no-store' })
  } catch (err) {
    throw networkError('buckets', err)
  }
  let data: GcsBucketListResponse & { error?: string }
  try {
    data = await readJsonResponse(res, '/api/gcs/buckets')
  } catch (err) {
    throw networkError('buckets', err)
  }
  if (!res.ok) {
    throw new Error(data.error || `Bucket list failed (HTTP ${res.status})`)
  }
  return data
}

export async function fetchDemoCatalog(
  onProgress?: (message: string) => void,
  bucket?: string,
): Promise<DemoCatalogResponse> {
  onProgress?.(
    bucket
      ? `Listing files in gs://${bucket}…`
      : 'Listing demo files in Cloud Storage…',
  )
  const qs = bucket ? `?bucket=${encodeURIComponent(bucket)}` : ''
  let res: Response
  try {
    res = await fetch(`/api/demo${qs}`, { cache: 'no-store' })
  } catch (err) {
    throw networkError('list', err)
  }
  let data: DemoCatalogResponse & { error?: string }
  try {
    data = await readJsonResponse(res, '/api/demo')
  } catch (err) {
    throw networkError('list', err)
  }
  if (!res.ok) {
    throw new Error(data.error || `Demo list failed (HTTP ${res.status})`)
  }
  if (!data.kind || !Array.isArray(data.members) || !data.members.length) {
    throw new Error('Demo catalog is empty')
  }
  return data
}

export type DemoMemberFetchResult = {
  file: File
  /** Staged gs:// URI in GCS_UPLOAD_BUCKET for Cloud Run (skips browser re-upload). */
  gcsUri: string
  size: number
}

export async function fetchDemoMemberFile(input: {
  kind: 'archive' | 'objects'
  objectName?: string
  member: string
  bucket?: string
  onProgress?: (message: string) => void
}): Promise<DemoMemberFetchResult> {
  const label = input.member.split('/').pop() || input.member
  input.onProgress?.(`Preparing ${label} from demo storage…`)

  let res: Response
  try {
    res = await fetch('/api/demo/member', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        kind: input.kind,
        objectName: input.objectName,
        member: input.member,
        bucket: input.bucket,
      }),
    })
  } catch (err) {
    throw networkError('prepare', err)
  }

  let data: {
    error?: string
    downloadUrl?: string
    filename?: string
    size?: number
    gcsUri?: string
  }
  try {
    data = await readJsonResponse(res, '/api/demo/member')
  } catch (err) {
    throw networkError('prepare', err)
  }
  if (!res.ok) {
    throw new Error(data.error || `Demo prepare failed (HTTP ${res.status})`)
  }
  if (!data.downloadUrl) {
    throw new Error('Demo prepare returned no download URL')
  }
  if (!data.gcsUri || !data.gcsUri.startsWith('gs://')) {
    throw new Error('Demo prepare returned no staged gcsUri for Cloud Run')
  }

  input.onProgress?.(
    `Downloading preview of ${data.filename || label}${
      data.size ? ` (${(Number(data.size) / (1024 * 1024)).toFixed(1)} MB)` : ''
    }…`,
  )

  let fileRes: Response
  try {
    fileRes = await fetch(data.downloadUrl)
  } catch (err) {
    throw networkError('download', err)
  }
  if (!fileRes.ok) {
    throw new Error(`Demo member download failed (HTTP ${fileRes.status})`)
  }
  const blob = await fileRes.blob()
  const filename = data.filename || label
  return {
    file: new File([blob], filename, {
      type: blob.type || 'application/octet-stream',
    }),
    gcsUri: data.gcsUri,
    size: Number(data.size || blob.size),
  }
}
