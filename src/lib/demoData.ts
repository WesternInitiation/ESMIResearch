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

export async function fetchDemoFileFromGcs(
  onProgress?: (message: string) => void,
): Promise<File> {
  onProgress?.('Requesting demo data from Cloud Storage…')
  const res = await fetch('/api/demo', { cache: 'no-store' })
  const data = await res.json()
  if (!res.ok) {
    throw new Error(data.error || 'Could not list demo bucket')
  }
  const primary = data.primary as DemoListing['primary'] | null
  if (!primary?.downloadUrl) {
    throw new Error('Demo bucket returned no downloadable object')
  }

  const baseName = primary.name.split('/').pop() || primary.name
  onProgress?.(
    `Downloading ${baseName}${
      primary.size > 0 ? ` (${(primary.size / (1024 * 1024)).toFixed(1)} MB)` : ''
    }…`,
  )

  const fileRes = await fetch(primary.downloadUrl)
  if (!fileRes.ok) {
    throw new Error(
      `Demo download failed (${fileRes.status}). Check bucket IAM / signed URL access.`,
    )
  }
  const blob = await fileRes.blob()
  const type =
    primary.contentType && primary.contentType !== 'application/octet-stream'
      ? primary.contentType
      : blob.type || 'application/octet-stream'
  return new File([blob], baseName, { type })
}
