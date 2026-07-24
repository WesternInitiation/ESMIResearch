import { NextRequest, NextResponse } from 'next/server'
import { cloudRunAuthHeaders } from '@/lib/cloudRunAuth'

export const runtime = 'nodejs'
export const maxDuration = 300

function apiBase(): string | null {
  const url = process.env.COMPRESS_API_URL?.trim()
  if (!url) return null
  return url.replace(/\/$/, '')
}

export async function GET() {
  const base = apiBase()
  if (!base) {
    return NextResponse.json({ configured: false, urlConfigured: false })
  }
  try {
    const headers = await cloudRunAuthHeaders(base)
    const res = await fetch(`${base}/health`, { cache: 'no-store', headers })
    const ok = res.ok
    const body = ok ? await res.json().catch(() => ({})) : null
    return NextResponse.json({
      configured: ok,
      health: body,
      urlConfigured: true,
      authConfigured: Boolean(process.env.GOOGLE_SERVICE_ACCOUNT_JSON?.trim()),
      upstreamStatus: res.status,
    })
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Health check failed'
    return NextResponse.json({
      configured: false,
      urlConfigured: true,
      authConfigured: Boolean(process.env.GOOGLE_SERVICE_ACCOUNT_JSON?.trim()),
      error: message,
    })
  }
}

export async function POST(request: NextRequest) {
  const base = apiBase()
  if (!base) {
    return NextResponse.json(
      {
        error:
          'Cloud Run is not configured. Set COMPRESS_API_URL (and GOOGLE_SERVICE_ACCOUNT_JSON for private services) on Vercel.',
      },
      { status: 503 },
    )
  }

  try {
    const form = await request.formData()
    const headers = await cloudRunAuthHeaders(base)
    const upstream = await fetch(`${base}/v1/compress`, {
      method: 'POST',
      body: form,
      headers,
    })
    const text = await upstream.text()
    let payload: unknown = null
    try {
      payload = JSON.parse(text)
    } catch {
      payload = { error: text || 'Upstream returned non-JSON' }
    }
    return NextResponse.json(payload, { status: upstream.status })
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Proxy failed'
    return NextResponse.json({ error: message }, { status: 502 })
  }
}
