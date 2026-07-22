export type SharedRunSummary = {
  id: string
  created_at: string
  method: string
  source_filename: string | null
  runtime_seconds: number | null
  compression_ratio: number | null
  share_token: string
}

export async function fetchSupabaseStatus(): Promise<boolean> {
  const res = await fetch('/api/supabase/status')
  if (!res.ok) return false
  const data = (await res.json()) as { configured: boolean }
  return data.configured
}

export async function listRecentRuns(limit = 15): Promise<SharedRunSummary[]> {
  const res = await fetch(`/api/runs?limit=${limit}`)
  if (!res.ok) throw new Error('Failed to list runs')
  const data = (await res.json()) as { runs: SharedRunSummary[] }
  return data.runs ?? []
}

export type SaveRunPayload = {
  method: string
  sourceFilename: string
  params: Record<string, unknown>
  runtimeSeconds: number
  originalBytes: number
  compressedBytesEstimate: number
  compressionRatio: number
  bandMetrics: Array<{
    band: string
    rmse: number
    mae: number
    psnr_db: number
    ssim: number
  }>
  originalFile: Blob
  originalFilename: string
  compressedFile: Blob
  compressedFilename: string
  notes?: string
}

export async function saveCompressionRun(input: SaveRunPayload) {
  const form = new FormData()
  form.set('method', input.method)
  form.set('sourceFilename', input.sourceFilename)
  form.set('params', JSON.stringify(input.params))
  form.set('runtimeSeconds', String(input.runtimeSeconds))
  form.set('originalBytes', String(input.originalBytes))
  form.set('compressedBytesEstimate', String(input.compressedBytesEstimate))
  form.set('compressionRatio', String(input.compressionRatio))
  form.set('bandMetrics', JSON.stringify(input.bandMetrics))
  form.set('originalFile', input.originalFile, input.originalFilename)
  form.set('compressedFile', input.compressedFile, input.compressedFilename)
  form.set('originalFilename', input.originalFilename)
  form.set('compressedFilename', input.compressedFilename)
  if (input.notes) form.set('notes', input.notes)

  const res = await fetch('/api/runs', { method: 'POST', body: form })
  const data = await res.json()
  if (!res.ok) throw new Error(data.error || 'Save failed')
  return data as { runId: string; shareToken: string }
}

export async function loadRunByShareToken(shareToken: string) {
  const res = await fetch(`/api/runs/${encodeURIComponent(shareToken)}`)
  const data = await res.json()
  if (!res.ok) throw new Error(data.error || 'Load failed')
  return data as {
    run: Record<string, unknown>
    metrics: Array<Record<string, unknown>>
    ndvi: Record<string, unknown> | null
    originalUrl: string | null
    compressedUrl: string | null
  }
}

export async function attachNdviToRun(
  shareToken: string,
  payload: {
    redBand: string
    nirBand: string
    rmse: number
    mae: number
    correlation: number
    ssim: number
    bias: number
  },
) {
  const res = await fetch(`/api/runs/${encodeURIComponent(shareToken)}/ndvi`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  const data = await res.json()
  if (!res.ok) throw new Error(data.error || 'NDVI save failed')
  return data
}
