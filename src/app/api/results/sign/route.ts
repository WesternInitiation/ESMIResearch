import { NextRequest, NextResponse } from 'next/server'
import { createGcsSignedDownload, gcsUploadsConfigured } from '@/lib/gcs'

export const runtime = 'nodejs'

/**
 * Sign short-lived GET URLs for reconstructed GeoTIFF artifacts staged by Cloud Run.
 * Body: { gcsUri } or { gcsUris: string[] }
 */
export async function POST(request: NextRequest) {
  if (!gcsUploadsConfigured()) {
    return NextResponse.json(
      {
        error:
          'GCS downloads are not configured. Set GCS_UPLOAD_BUCKET and GOOGLE_SERVICE_ACCOUNT_JSON on Vercel.',
      },
      { status: 503 },
    )
  }

  try {
    const body = (await request.json()) as {
      gcsUri?: string
      gcsUris?: string[]
    }
    const uris = [
      ...(Array.isArray(body.gcsUris) ? body.gcsUris : []),
      ...(body.gcsUri ? [body.gcsUri] : []),
    ]
      .map((u) => String(u || '').trim())
      .filter((u) => u.startsWith('gs://'))

    if (!uris.length) {
      return NextResponse.json(
        { error: 'gcsUri or gcsUris is required' },
        { status: 400 },
      )
    }

    const downloads = []
    for (const gcsUri of uris) {
      downloads.push(await createGcsSignedDownload({ gcsUri }))
    }
    return NextResponse.json(
      downloads.length === 1
        ? { ...downloads[0], downloads }
        : { downloads },
    )
  } catch (err) {
    const message =
      err instanceof Error ? err.message : 'Failed to sign download URL'
    return NextResponse.json({ error: message }, { status: 500 })
  }
}
