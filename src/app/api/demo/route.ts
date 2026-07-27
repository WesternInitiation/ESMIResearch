import { NextResponse } from 'next/server'
import { listDemoObjectsWithSignedUrls } from '@/lib/gcs'

export const runtime = 'nodejs'

export async function GET() {
  try {
    const listed = await listDemoObjectsWithSignedUrls()
    if (!listed.primary) {
      return NextResponse.json(
        {
          error: `No demo images found in gs://${listed.bucket}. Upload a .tif / .png / .tar(.gz) object.`,
          bucket: listed.bucket,
          objects: [],
        },
        { status: 404 },
      )
    }
    return NextResponse.json(listed)
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Failed to list demo bucket'
    const permissionHint =
      /permission|denied|403|401/i.test(message)
        ? ' Grant roles/storage.objectViewer on gs://esmi-research-demo-data to esmi-vercel@esmi-research.iam.gserviceaccount.com'
        : ''
    return NextResponse.json(
      { error: `${message}.${permissionHint}`.replace(/\.\./g, '.') },
      { status: 500 },
    )
  }
}
