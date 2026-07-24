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

/**
 * Cloud Run is reached only via the Next.js /api/compress proxy so private
 * services (org policy blocking allUsers) can authenticate with a service account.
 * Note: Vercel serverless request bodies are capped (~4.5MB on hobby).
 */
export async function fetchCloudRunStatus(): Promise<{
  configured: boolean
  urlConfigured: boolean
}> {
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

  const res = await fetch('/api/compress', { method: 'POST', body: form })
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
