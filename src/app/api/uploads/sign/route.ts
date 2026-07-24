import { NextRequest, NextResponse } from 'next/server'
import { createGcsSignedUpload, gcsUploadsConfigured } from '@/lib/gcs'

export const runtime = 'nodejs'

export async function GET() {
  return NextResponse.json({
    configured: gcsUploadsConfigured(),
    bucket: process.env.GCS_UPLOAD_BUCKET?.trim() || null,
  })
}

export async function POST(request: NextRequest) {
  if (!gcsUploadsConfigured()) {
    return NextResponse.json(
      {
        error:
          'GCS uploads are not configured. Set GCS_UPLOAD_BUCKET and GOOGLE_SERVICE_ACCOUNT_JSON on Vercel.',
      },
      { status: 503 },
    )
  }

  try {
    const body = (await request.json()) as {
      filename?: string
      contentType?: string
      size?: number
    }
    const filename = (body.filename || 'upload.bin').trim()
    const contentType = (body.contentType || 'application/octet-stream').trim()
    const size = Number(body.size)
    if (!Number.isFinite(size) || size <= 0) {
      return NextResponse.json({ error: 'size is required' }, { status: 400 })
    }

    const signed = await createGcsSignedUpload({ filename, contentType, size })
    return NextResponse.json(signed)
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Failed to sign upload URL'
    return NextResponse.json({ error: message }, { status: 500 })
  }
}
