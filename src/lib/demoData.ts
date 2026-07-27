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

export async function fetchDemoCatalog(
  onProgress?: (message: string) => void,
): Promise<DemoCatalogResponse> {
  onProgress?.('Listing demo files in Cloud Storage…')
  let res: Response
  try {
    res = await fetch('/api/demo', { cache: 'no-store' })
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

export async function fetchDemoMemberFile(input: {
  kind: 'archive' | 'objects'
  objectName?: string
  member: string
  onProgress?: (message: string) => void
}): Promise<File> {
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

  input.onProgress?.(
    `Downloading ${data.filename || label}${
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
  return new File([blob], filename, {
    type: blob.type || 'application/octet-stream',
  })
}
