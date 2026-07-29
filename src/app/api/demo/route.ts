import { NextRequest, NextResponse } from 'next/server'
import { buildDemoCatalog } from '@/lib/demoCatalog'

export const runtime = 'nodejs'
export const maxDuration = 300

export async function GET(request: NextRequest) {
  try {
    const bucket = request.nextUrl.searchParams.get('bucket')
    const catalog = await buildDemoCatalog(bucket)
    return NextResponse.json(catalog)
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Failed to list demo bucket'
    const permissionHint = /permission|denied|403|401/i.test(message)
      ? ' Grant roles/storage.objectViewer on the selected bucket to esmi-vercel@esmi-research.iam.gserviceaccount.com'
      : ''
    return NextResponse.json(
      { error: `${message}.${permissionHint}`.replace(/\.\./g, '.') },
      { status: 500 },
    )
  }
}
