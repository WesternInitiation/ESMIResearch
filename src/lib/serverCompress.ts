import { readJsonResponse } from '@/lib/httpJson'

export type ReconstructedBandGeotiff = {
  band: string
  label: string
  filename: string
  gcsUri: string
  bucket?: string
  objectName?: string
  size: number
  dtype: string
  width: number
  height: number
  crs?: string | null
  transform?: number[] | null
  nodata?: number | null
  method?: string
}

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
  residualPreviewPngBase64?: string | null
  /** One rasterio GeoTIFF per band (CRS/transform/NoData), staged in GCS. */
  reconstructedBandGeotiffs?: ReconstructedBandGeotiff[]
  geotiffError?: string | null
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

export type CompressProgress = {
  progress: number
  phase: string
  message: string
}

async function uploadToGcs(
  file: Blob,
  filename: string,
  onProgress?: (update: CompressProgress) => void,
): Promise<string> {
  onProgress?.({
    progress: 2,
    phase: 'upload',
    message: `Requesting GCS upload URL for ${filename}…`,
  })
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

  onProgress?.({
    progress: 5,
    phase: 'upload',
    message: `Uploading ${(file.size / (1024 * 1024)).toFixed(1)} MB to Cloud Storage…`,
  })
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

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms))
}

async function pollCompressJob(
  jobId: string,
  onProgress?: (update: CompressProgress) => void,
): Promise<ServerCompressResponse> {
  const started = Date.now()
  const maxWaitMs = 15 * 60 * 1000
  let delay = 900

  while (Date.now() - started < maxWaitMs) {
    const res = await fetch(`/api/compress/jobs/${encodeURIComponent(jobId)}`, {
      cache: 'no-store',
    })
    const data = await readJsonResponse<{
      status?: string
      progress?: number
      phase?: string
      message?: string
      error?: string
      result?: ServerCompressResponse
      detail?: unknown
    }>(res, `/api/compress/jobs/${jobId}`)

    if (!res.ok) {
      throw new Error(
        data.error ||
          (typeof data.detail === 'string' ? data.detail : null) ||
          `Job poll failed (HTTP ${res.status})`,
      )
    }

    const progress = Math.max(0, Math.min(100, Number(data.progress) || 0))
    const phase = data.phase || data.status || 'running'
    const message = data.message || `Cloud Run job ${data.status || 'running'}…`
    onProgress?.({ progress, phase, message })

    if (data.status === 'done') {
      if (!data.result) {
        throw new Error('Cloud Run job finished without a result payload')
      }
      onProgress?.({
        progress: 100,
        phase: 'done',
        message: 'Compression complete',
      })
      return data.result
    }
    if (data.status === 'error') {
      throw new Error(data.error || data.message || 'Cloud Run compression failed')
    }

    await sleep(delay)
    delay = Math.min(2500, Math.round(delay * 1.15))
  }
  throw new Error(
    'Cloud Run job timed out after 15 minutes. Try a smaller Max processing size (2048 or 1024).',
  )
}

export async function runServerCompression(input: {
  /** Required unless `gcsUri` is already staged (demo GCS path). */
  file?: Blob | null
  filename: string
  /** When set, Cloud Run reads this URI directly — no browser → GCS re-upload. */
  gcsUri?: string | null
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
  onProgress?: (update: CompressProgress | string) => void
}): Promise<ServerCompressResponse> {
  const emit = (update: CompressProgress | string) => {
    if (!input.onProgress) return
    if (typeof update === 'string') {
      input.onProgress({ progress: 0, phase: 'status', message: update })
    } else {
      input.onProgress(update)
    }
  }

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

  let durableGcsUri = (input.gcsUri || '').trim()
  let uploadFile: Blob | null = input.file ?? null

  if (durableGcsUri.startsWith('gs://')) {
    form.set('gcs_uri', durableGcsUri)
    form.set('filename', input.filename)
    emit({
      progress: 8,
      phase: 'start',
      message: 'Starting Cloud Run job from staged Cloud Storage…',
    })
  } else {
    const file = uploadFile
    if (!file) {
      throw new Error('Missing file for Cloud Run (and no staged gcsUri)')
    }
    if (input.gcsUploads) {
      durableGcsUri = await uploadToGcs(file, input.filename, (u) => emit(u))
      form.set('gcs_uri', durableGcsUri)
      form.set('filename', input.filename)
      uploadFile = null
      emit({
        progress: 10,
        phase: 'start',
        message: 'Starting Cloud Run async job…',
      })
    } else if (file.size > VERCEL_PROXY_UPLOAD_BYTES) {
      throw new Error(
        `File is ${(file.size / (1024 * 1024)).toFixed(1)} MB. Set GCS_UPLOAD_BUCKET on Vercel (see cloud_run/README.md) to send large jobs to Cloud Run, or use Engine → Browser.`,
      )
    } else {
      form.set('file', file, input.filename)
      emit({
        progress: 10,
        phase: 'start',
        message: 'Starting Cloud Run async job…',
      })
    }
  }

  const buildForm = () => {
    const next = new FormData()
    next.set('method', input.method)
    next.set('max_dim', String(input.maxDim))
    next.set('svd_rank', String(input.svdRank))
    next.set('wavelet_keep_fraction', String(input.waveletKeepFraction))
    next.set('wavelet_levels', String(input.waveletLevels))
    next.set('wavelet_name', input.waveletName || 'db4')
    next.set('bandwidth_keep_fraction', String(input.bandwidthKeepFraction))
    next.set('jpeg_rate', String(input.jpegRate))
    if (input.archiveMember) next.set('archive_member', input.archiveMember)
    if (input.redBand) next.set('red_band', input.redBand)
    if (input.nirBand) next.set('nir_band', input.nirBand)
    if (durableGcsUri.startsWith('gs://')) {
      next.set('gcs_uri', durableGcsUri)
      next.set('filename', input.filename)
    } else if (uploadFile) {
      next.set('file', uploadFile, input.filename)
    }
    return next
  }

  // Prefer async jobs so Vercel does not hold the request for the full codec run.
  const startRes = await fetch('/api/compress/jobs', {
    method: 'POST',
    body: buildForm(),
  })
  const startData = await readJsonResponse<{
    jobId?: string
    error?: string
    detail?: unknown
    status?: string
    progress?: number
    phase?: string
    message?: string
  }>(startRes, '/api/compress/jobs')

  if (startRes.ok && startData.jobId) {
    emit({
      progress: Number(startData.progress) || 12,
      phase: startData.phase || 'queued',
      message: startData.message || 'Cloud Run job queued…',
    })
    return pollCompressJob(startData.jobId, (u) => emit(u))
  }

  // Fallback: older Cloud Run without async jobs.
  const startErr =
    startData.error ||
    (typeof startData.detail === 'string' ? startData.detail : '') ||
    ''
  if (/async jobs require|not found|404|503/i.test(startErr) || startRes.status === 404) {
    emit({
      progress: 15,
      phase: 'sync',
      message:
        'Async jobs unavailable on this Cloud Run image — using sync compress (may hit Vercel 5‑minute limit). Redeploy Cloud Run for progress + long jobs.',
    })
    const res = await fetch('/api/compress', { method: 'POST', body: buildForm() })
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
      if (/FUNCTION_INVOCATION_TIMEOUT|504/i.test(message) || res.status === 504) {
        throw new Error(
          'Vercel timed out after ~5 minutes while waiting on Cloud Run (FUNCTION_INVOCATION_TIMEOUT). ' +
            'Redeploy Cloud Run for async jobs, or lower Max processing size to 2048/1024.',
        )
      }
      throw new Error(message)
    }
    emit({ progress: 100, phase: 'done', message: 'Compression complete' })
    return data as ServerCompressResponse
  }

  throw new Error(startErr || `Failed to start Cloud Run job (HTTP ${startRes.status})`)
}
