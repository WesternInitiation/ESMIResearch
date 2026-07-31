import { NextRequest, NextResponse } from 'next/server'
import { cloudRunAuthHeaders } from '@/lib/cloudRunAuth'

export const runtime = 'nodejs'
export const maxDuration = 30

function apiBase(): string | null {
  const url = process.env.COMPRESS_API_URL?.trim()
  if (!url) return null
  return url.replace(/\/$/, '')
}

/**
 * Poll async Cloud Run compress job status / result.
 */
export async function GET(
  _request: NextRequest,
  context: { params: Promise<{ id: string }> | { id: string } },
) {
  const base = apiBase()
  if (!base) {
    return NextResponse.json(
      { error: 'Cloud Run is not configured.' },
      { status: 503 },
    )
  }

  const rawParams = context.params
  const params = typeof (rawParams as Promise<{ id: string }>).then === 'function'
    ? await (rawParams as Promise<{ id: string }>)
    : (rawParams as { id: string })
  const jobId = (params.id || '').trim()
  if (!jobId) {
    return NextResponse.json({ error: 'job id is required' }, { status: 400 })
  }

  try {
    const headers = await cloudRunAuthHeaders(base)
    const upstream = await fetch(`${base}/v1/compress/jobs/${encodeURIComponent(jobId)}`, {
      cache: 'no-store',
      headers,
    })
    const text = await upstream.text()
    let payload: unknown = null
    try {
      payload = text ? JSON.parse(text) : null
    } catch {
      return NextResponse.json(
        {
          error: `Cloud Run job status returned non-JSON (HTTP ${upstream.status}): ${text.slice(0, 200)}`,
        },
        { status: upstream.status || 502 },
      )
    }
    return NextResponse.json(payload, { status: upstream.status })
  } catch (err) {
    const message =
      err instanceof Error ? err.message : 'Failed to poll Cloud Run job'
    return NextResponse.json({ error: message }, { status: 500 })
  }
}
