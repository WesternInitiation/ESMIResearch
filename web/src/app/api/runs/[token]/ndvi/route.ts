import { NextRequest, NextResponse } from 'next/server'
import { createServerSupabase, supabaseConfigured } from '@/lib/supabase/server'

type Params = { params: Promise<{ token: string }> }

export async function POST(request: NextRequest, { params }: Params) {
  if (!supabaseConfigured()) {
    return NextResponse.json(
      { error: 'Supabase is not configured on the server' },
      { status: 503 },
    )
  }

  const { token } = await params
  const body = await request.json()
  const client = createServerSupabase()

  const { data: runs, error } = await client
    .from('runs')
    .select('id')
    .eq('share_token', token)
    .limit(1)
  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 })
  }
  if (!runs?.length) {
    return NextResponse.json({ error: 'Run not found' }, { status: 404 })
  }

  const { error: upsertError } = await client.from('ndvi_results').upsert({
    run_id: runs[0].id,
    red_band: body.redBand,
    nir_band: body.nirBand,
    rmse: body.rmse,
    mae: body.mae,
    correlation: body.correlation,
    ssim: body.ssim,
    bias: body.bias,
  })
  if (upsertError) {
    return NextResponse.json({ error: upsertError.message }, { status: 500 })
  }
  return NextResponse.json({ ok: true })
}
