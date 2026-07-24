import { NextRequest, NextResponse } from 'next/server'

function apiBase(): string | null {
  const url = process.env.COMPRESS_API_URL?.trim()
  if (!url) return null
  return url.replace(/\/$/, '')
}

export async function GET() {
  const base = apiBase()
  if (!base) {
    return NextResponse.json({ configured: false })
  }
  try {
    const res = await fetch(`${base}/health`, { cache: 'no-store' })
    const ok = res.ok
    const body = ok ? await res.json().catch(() => ({})) : null
    return NextResponse.json({ configured: ok, health: body, urlConfigured: true })
  } catch {
    return NextResponse.json({ configured: false, urlConfigured: true })
  }
}

export async function POST(request: NextRequest) {
  const base = apiBase()
  if (!base) {
    return NextResponse.json(
      {
        error:
          'Cloud Run is not configured. Set COMPRESS_API_URL to your Cloud Run service URL.',
      },
      { status: 503 },
    )
  }

  try {
    const form = await request.formData()
    const upstream = await fetch(`${base}/v1/compress`, {
      method: 'POST',
      body: form,
      // Large satellite uploads; do not set a tiny timeout here.
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
