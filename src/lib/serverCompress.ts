import { readJsonResponse } from '@/lib/httpJson'

export type ServerCompressResponse = {
  engine: string
  method: string
  source: string
  runtimeSeconds: number
  originalBytes: number
  compressedBytesEstimate: number
  compressionRatio: number
  width: number
  height: number
  nativeWidth: number
  nativeHeight: number
  processScale: number
  bandOrder: string[]
  channelReports: Array<{
    band: string
    rmse: number
    mae: number
    psnrDb: number
    ssim: number
  }>
  metadata: Record<string, unknown>
  ndvi: {
    rmse: number
    mae: number
    correlation: number
    ssim: number
    bias: number
    valid_pixel_fraction?: number
  } | null
  originalPreviewPngBase64: string
  previewPngBase64: string
}

/** Soft limit for multipart bodies through the Vercel /api/compress proxy. */
export const VERCEL_PROXY_UPLOAD_BYTES = Math.floor(4 * 1024 * 1024)

/**
 * Cloud Run is reached via the Next.js /api/compress proxy (private SA auth).
 * Large files (>~4MB) upload directly to GCS with a signed URL, then Cloud Run
 * reads `gcs_uri` — bypassing Vercel and Cloud Run HTTP body caps.
 */
export async function fetchCloudRunStatus(): Promise<{
  configured: boolean
  urlConfigured: boolean
  gcsUploads: boolean
  gcsBucketConfigured: boolean
  gcsAuthConfigured: boolean
  gcsAuthValid: boolean
  gcsBucket: string | null
}> {
  const res = await fetch('/api/compress', { cache: 'no-store' })
  if (!res.ok) {
    return {
      configured: false,
      urlConfigured: false,
      gcsUploads: false,
      gcsBucketConfigured: false,
      gcsAuthConfigured: false,
      gcsAuthValid: false,
      gcsBucket: null,
    }
  }
  try {
    const data = await readJsonResponse<{
      configured?: boolean
      urlConfigured?: boolean
      gcsUploads?: boolean
      gcsBucketConfigured?: boolean
      gcsAuthConfigured?: boolean
      gcsAuthValid?: boolean
      gcsBucket?: string | null
    }>(res, '/api/compress')
    return {
      configured: Boolean(data.configured),
      urlConfigured: Boolean(data.urlConfigured),
      gcsUploads: Boolean(data.gcsUploads),
      gcsBucketConfigured: Boolean(data.gcsBucketConfigured),
      gcsAuthConfigured: Boolean(data.gcsAuthConfigured),
      gcsAuthValid: data.gcsAuthValid !== false,
      gcsBucket: data.gcsBucket ?? null,
    }
  } catch {
    return {
      configured: false,
      urlConfigured: false,
      gcsUploads: false,
      gcsBucketConfigured: false,
      gcsAuthConfigured: false,
      gcsAuthValid: false,
      gcsBucket: null,
    }
  }
}

async function uploadToGcs(
  file: Blob,
  filename: string,
  onProgress?: (message: string) => void,
): Promise<string> {
  onProgress?.(`Requesting GCS upload URL for ${filename}…`)
  const signRes = await fetch('/api/uploads/sign', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      filename,
      contentType: file.type || 'application/octet-stream',
      size: file.size,
    }),
  })
  const signed = await readJsonResponse<{
    error?: string
    uploadUrl?: string
    gcsUri?: string
  }>(signRes, '/api/uploads/sign')
  if (!signRes.ok) {
    throw new Error(signed.error || 'Failed to sign GCS upload URL')
  }
  if (!signed.uploadUrl || !signed.gcsUri) {
    throw new Error('GCS sign response missing uploadUrl/gcsUri')
  }

  onProgress?.(
    `Uploading ${(file.size / (1024 * 1024)).toFixed(1)} MB to Cloud Storage…`,
  )
  const put = await fetch(signed.uploadUrl, {
    method: 'PUT',
    headers: {
      'Content-Type': (file.type || 'application/octet-stream') as string,
    },
    body: file,
  })
  if (!put.ok) {
    const text = await put.text().catch(() => '')
    throw new Error(
      `GCS upload failed (${put.status})${text ? `: ${text.slice(0, 200)}` : ''}`,
    )
  }
  return signed.gcsUri
}

export async function runServerCompression(input: {
  file: Blob
  filename: string
  method: string
  archiveMember?: string | null
  maxDim: number
  svdRank: number
  waveletKeepFraction: number
  waveletLevels: number
  waveletName?: string
  bandwidthKeepFraction: number
  jpegRate: number
  redBand?: string
  nirBand?: string
  gcsUploads?: boolean
  onProgress?: (message: string) => void
}): Promise<ServerCompressResponse> {
  const form = new FormData()
  form.set('method', input.method)
  form.set('max_dim', String(input.maxDim))
  form.set('svd_rank', String(input.svdRank))
  form.set('wavelet_keep_fraction', String(input.waveletKeepFraction))
  form.set('wavelet_levels', String(input.waveletLevels))
  form.set('wavelet_name', input.waveletName || 'db4')
  form.set('bandwidth_keep_fraction', String(input.bandwidthKeepFraction))
  form.set('jpeg_rate', String(input.jpegRate))
  if (input.archiveMember) form.set('archive_member', input.archiveMember)
  if (input.redBand) form.set('red_band', input.redBand)
  if (input.nirBand) form.set('nir_band', input.nirBand)

  const useGcs = Boolean(input.gcsUploads) && input.file.size > VERCEL_PROXY_UPLOAD_BYTES
  if (useGcs) {
    const gcsUri = await uploadToGcs(input.file, input.filename, input.onProgress)
    form.set('gcs_uri', gcsUri)
    form.set('filename', input.filename)
    input.onProgress?.('Starting Cloud Run job from Cloud Storage…')
  } else if (input.file.size > VERCEL_PROXY_UPLOAD_BYTES) {
    throw new Error(
      `File is ${(input.file.size / (1024 * 1024)).toFixed(1)} MB. Set GCS_UPLOAD_BUCKET on Vercel (see cloud_run/README.md) to send large jobs to Cloud Run, or use Engine → Browser.`,
    )
  } else {
    form.set('file', input.file, input.filename)
  }

  const res = await fetch('/api/compress', { method: 'POST', body: form })
  const data = await readJsonResponse<{
    detail?: unknown
    error?: string
  } & Partial<ServerCompressResponse>>(res, '/api/compress')
  if (!res.ok) {
    const detail = data.detail
    const message =
      typeof detail === 'string'
        ? detail
        : Array.isArray(detail)
          ? detail.map((d: { msg?: string }) => d.msg).filter(Boolean).join('; ')
          : data.error || 'Cloud Run compression failed'
    throw new Error(message)
  }
  return data as ServerCompressResponse
}
