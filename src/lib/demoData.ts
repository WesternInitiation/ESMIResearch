/** Client helper: fetch signed demo object from /api/demo and return a File. */

export type DemoListing = {
  bucket: string
  primary: {
    name: string
    size: number
    contentType: string
    downloadUrl: string
  }
  objects: Array<{ name: string; size: number }>
}

function failedFetchMessage(stage: 'list' | 'download', err: unknown): string {
  const raw = err instanceof Error ? err.message : String(err)
  if (/failed to fetch|networkerror|load failed/i.test(raw)) {
    if (stage === 'list') {
      return (
        'Could not reach /api/demo (network). Confirm the latest Vercel deploy is live, then open /api/demo in a new tab.'
      )
    }
    return (
      'Browser blocked the GCS download (usually CORS). On the demo bucket run CORS allow GET/HEAD/OPTIONS for origin *, ' +
      'and grant objectViewer to esmi-vercel@esmi-research.iam.gserviceaccount.com.'
    )
  }
  return raw || 'Request failed'
}

export async function fetchDemoFileFromGcs(
  onProgress?: (message: string) => void,
): Promise<File> {
  onProgress?.('Requesting demo data from Cloud Storage…')

  let res: Response
  try {
    res = await fetch('/api/demo', { cache: 'no-store' })
  } catch (err) {
    throw new Error(failedFetchMessage('list', err))
  }

  let data: {
    error?: string
    primary?: DemoListing['primary'] | null
    bucket?: string
  }
  try {
    data = await res.json()
  } catch {
    throw new Error(
      `Demo API returned non-JSON (HTTP ${res.status}). Redeploy Vercel so /api/demo exists.`,
    )
  }

  if (!res.ok) {
    throw new Error(data.error || `Could not list demo bucket (HTTP ${res.status})`)
  }
  const primary = data.primary
  if (!primary?.downloadUrl) {
    throw new Error('Demo bucket returned no downloadable object')
  }

  const baseName = primary.name.split('/').pop() || primary.name
  onProgress?.(
    `Downloading ${baseName}${
      primary.size > 0 ? ` (${(primary.size / (1024 * 1024)).toFixed(1)} MB)` : ''
    }…`,
  )

  let fileRes: Response
  try {
    fileRes = await fetch(primary.downloadUrl)
  } catch (err) {
    throw new Error(failedFetchMessage('download', err))
  }

  if (!fileRes.ok) {
    throw new Error(
      `Demo download failed (HTTP ${fileRes.status}). Check objectViewer IAM on gs://esmi-research-demo-data for esmi-vercel.`,
    )
  }
  const blob = await fileRes.blob()
  const type =
    primary.contentType && primary.contentType !== 'application/octet-stream'
      ? primary.contentType
      : blob.type || 'application/octet-stream'
  return new File([blob], baseName, { type })
}
