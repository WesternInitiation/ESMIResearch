import { NextRequest, NextResponse } from 'next/server'
import {
  createServerSupabase,
  storageBucket,
  supabaseConfigured,
} from '@/lib/supabase/server'

type Params = { params: Promise<{ token: string }> }

export async function GET(_request: NextRequest, { params }: Params) {
  if (!supabaseConfigured()) {
    return NextResponse.json(
      { error: 'Supabase is not configured on the server' },
      { status: 503 },
    )
  }

  const { token } = await params
  const client = createServerSupabase()
  const bucket = storageBucket()

  const { data: runs, error } = await client
    .from('runs')
    .select('*')
    .eq('share_token', token)
    .limit(1)
  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 })
  }
  if (!runs?.length) {
    return NextResponse.json({ error: 'Run not found' }, { status: 404 })
  }
  const run = runs[0]

  const { data: metrics } = await client
    .from('band_metrics')
    .select('*')
    .eq('run_id', run.id)

  const { data: ndviRows } = await client
    .from('ndvi_results')
    .select('*')
    .eq('run_id', run.id)
    .limit(1)

  let originalUrl: string | null = null
  let compressedUrl: string | null = null

  if (run.original_storage_path) {
    const signed = await client.storage
      .from(bucket)
      .createSignedUrl(run.original_storage_path, 3600)
    originalUrl = signed.data?.signedUrl ?? null
  }
  if (run.compressed_storage_path) {
    const signed = await client.storage
      .from(bucket)
      .createSignedUrl(run.compressed_storage_path, 3600)
    compressedUrl = signed.data?.signedUrl ?? null
  }

  return NextResponse.json({
    run,
    metrics: metrics ?? [],
    ndvi: ndviRows?.[0] ?? null,
    originalUrl,
    compressedUrl,
  })
}
