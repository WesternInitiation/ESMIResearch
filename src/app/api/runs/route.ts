import { NextRequest, NextResponse } from 'next/server'
import {
  createServerSupabase,
  storageBucket,
  supabaseConfigured,
} from '@/lib/supabase/server'

export async function GET(request: NextRequest) {
  if (!supabaseConfigured()) {
    return NextResponse.json({ runs: [], configured: false })
  }
  const limit = Number(request.nextUrl.searchParams.get('limit') || '15')
  const client = createServerSupabase()
  const { data, error } = await client
    .from('runs')
    .select(
      'id, created_at, method, source_filename, runtime_seconds, compression_ratio, share_token',
    )
    .order('created_at', { ascending: false })
    .limit(Math.min(Math.max(limit, 1), 50))
  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 })
  }
  return NextResponse.json({ runs: data ?? [], configured: true })
}

export async function POST(request: NextRequest) {
  if (!supabaseConfigured()) {
    return NextResponse.json(
      { error: 'Supabase is not configured on the server' },
      { status: 503 },
    )
  }

  try {
    const form = await request.formData()
    const method = String(form.get('method') || '')
    const sourceFilename = String(form.get('sourceFilename') || '')
    const params = JSON.parse(String(form.get('params') || '{}'))
    const runtimeSeconds = Number(form.get('runtimeSeconds') || 0)
    const originalBytes = Number(form.get('originalBytes') || 0)
    const compressedBytesEstimate = Number(form.get('compressedBytesEstimate') || 0)
    const compressionRatio = Number(form.get('compressionRatio') || 0)
    const bandMetrics = JSON.parse(String(form.get('bandMetrics') || '[]')) as Array<{
      band: string
      rmse: number
      mae: number
      psnr_db: number
      ssim: number
    }>
    const notes = form.get('notes') ? String(form.get('notes')) : null
    const originalFile = form.get('originalFile')
    const compressedFile = form.get('compressedFile')
    const originalFilename = String(form.get('originalFilename') || 'original.bin')
    const compressedFilename = String(form.get('compressedFilename') || 'compressed.png')

    if (!(originalFile instanceof Blob) || !(compressedFile instanceof Blob)) {
      return NextResponse.json({ error: 'Missing file uploads' }, { status: 400 })
    }
    if (!method) {
      return NextResponse.json({ error: 'method is required' }, { status: 400 })
    }

    const client = createServerSupabase()
    const bucket = storageBucket()
    const runId = crypto.randomUUID()
    const shareToken = crypto.randomUUID()
    const originalPath = `runs/${runId}/original-${originalFilename}`
    const compressedPath = `runs/${runId}/compressed-${compressedFilename}`

    const up1 = await client.storage.from(bucket).upload(originalPath, originalFile, {
      upsert: true,
      contentType: originalFile.type || 'application/octet-stream',
    })
    if (up1.error) {
      return NextResponse.json({ error: up1.error.message }, { status: 500 })
    }

    const up2 = await client.storage.from(bucket).upload(compressedPath, compressedFile, {
      upsert: true,
      contentType: compressedFile.type || 'image/png',
    })
    if (up2.error) {
      return NextResponse.json({ error: up2.error.message }, { status: 500 })
    }

    const { error: runError } = await client.from('runs').insert({
      id: runId,
      method,
      source_filename: sourceFilename,
      params,
      original_storage_path: originalPath,
      compressed_storage_path: compressedPath,
      runtime_seconds: runtimeSeconds,
      original_bytes: originalBytes,
      compressed_bytes_estimate: compressedBytesEstimate,
      compression_ratio: compressionRatio,
      share_token: shareToken,
      notes,
    })
    if (runError) {
      return NextResponse.json({ error: runError.message }, { status: 500 })
    }

    if (bandMetrics.length) {
      const { error: metricError } = await client.from('band_metrics').insert(
        bandMetrics.map((m) => ({
          run_id: runId,
          band: m.band,
          rmse: m.rmse,
          mae: m.mae,
          psnr_db: m.psnr_db,
          ssim: m.ssim,
        })),
      )
      if (metricError) {
        return NextResponse.json({ error: metricError.message }, { status: 500 })
      }
    }

    return NextResponse.json({ runId, shareToken })
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Save failed'
    return NextResponse.json({ error: message }, { status: 500 })
  }
}
