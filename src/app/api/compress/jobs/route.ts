import { NextRequest, NextResponse } from 'next/server'
import { cloudRunAuthHeaders } from '@/lib/cloudRunAuth'

export const runtime = 'nodejs'
/** Job start returns quickly; keep modest in case of large form staging. */
export const maxDuration = 60

function apiBase(): string | null {
  const url = process.env.COMPRESS_API_URL?.trim()
  if (!url) return null
  return url.replace(/\/$/, '')
}

/**
 * Start an async Cloud Run compress job (avoids long Vercel proxy waits).
 */
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
    const upstream = await fetch(`${base}/v1/compress/jobs`, {
      method: 'POST',
      body: form,
      headers,
    })
    const text = await upstream.text()
    let payload: unknown = null
    try {
      payload = text ? JSON.parse(text) : null
    } catch {
      return NextResponse.json(
        {
          error: `Cloud Run /v1/compress/jobs returned non-JSON (HTTP ${upstream.status}): ${text.slice(0, 200)}`,
        },
        { status: upstream.status || 502 },
      )
    }
    return NextResponse.json(payload, { status: upstream.status })
  } catch (err) {
    const message =
      err instanceof Error ? err.message : 'Failed to start Cloud Run compress job'
    return NextResponse.json({ error: message }, { status: 500 })
  }
}
