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

function publicApiBase(): string | null {
  const url = process.env.NEXT_PUBLIC_COMPRESS_API_URL?.trim()
  if (!url) return null
  return url.replace(/\/$/, '')
}

export async function fetchCloudRunStatus(): Promise<{
  configured: boolean
  urlConfigured: boolean
}> {
  const direct = publicApiBase()
  if (direct) {
    try {
      const res = await fetch(`${direct}/health`, { cache: 'no-store' })
      return { configured: res.ok, urlConfigured: true }
    } catch {
      return { configured: false, urlConfigured: true }
    }
  }

  const res = await fetch('/api/compress', { cache: 'no-store' })
  if (!res.ok) return { configured: false, urlConfigured: false }
  const data = (await res.json()) as {
    configured?: boolean
    urlConfigured?: boolean
  }
  return {
    configured: Boolean(data.configured),
    urlConfigured: Boolean(data.urlConfigured),
  }
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
  bandwidthKeepFraction: number
  jpegRate: number
  redBand?: string
  nirBand?: string
}): Promise<ServerCompressResponse> {
  const form = new FormData()
  form.set('file', input.file, input.filename)
  form.set('method', input.method)
  form.set('max_dim', String(input.maxDim))
  form.set('svd_rank', String(input.svdRank))
  form.set('wavelet_keep_fraction', String(input.waveletKeepFraction))
  form.set('wavelet_levels', String(input.waveletLevels))
  form.set('bandwidth_keep_fraction', String(input.bandwidthKeepFraction))
  form.set('jpeg_rate', String(input.jpegRate))
  if (input.archiveMember) form.set('archive_member', input.archiveMember)
  if (input.redBand) form.set('red_band', input.redBand)
  if (input.nirBand) form.set('nir_band', input.nirBand)

  // Prefer calling Cloud Run directly from the browser so large TAR/GeoTIFF
  // uploads are not capped by Vercel serverless body limits (~4.5MB).
  const direct = publicApiBase()
  const endpoint = direct ? `${direct}/v1/compress` : '/api/compress'

  const res = await fetch(endpoint, { method: 'POST', body: form })
  const data = await res.json()
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
