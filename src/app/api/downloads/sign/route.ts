import { NextRequest, NextResponse } from 'next/server'
import { createGcsSignedDownload, gcsUploadsConfigured } from '@/lib/gcs'

export const runtime = 'nodejs'

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
    const body = (await request.json()) as { gcsUri?: string }
    const gcsUri = (body.gcsUri || '').trim()
    if (!gcsUri) {
      return NextResponse.json({ error: 'gcsUri is required' }, { status: 400 })
    }
    const signed = await createGcsSignedDownload(gcsUri)
    return NextResponse.json(signed)
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Failed to sign download URL'
    return NextResponse.json({ error: message }, { status: 500 })
  }
}
