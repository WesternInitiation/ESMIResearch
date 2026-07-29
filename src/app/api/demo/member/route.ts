import { NextRequest, NextResponse } from 'next/server'
import { prepareDemoMember } from '@/lib/demoCatalog'

export const runtime = 'nodejs'
export const maxDuration = 300

export async function POST(request: NextRequest) {
  try {
    const body = (await request.json()) as {
      kind?: 'archive' | 'objects'
      objectName?: string
      member?: string
      bucket?: string
    }
    const kind = body.kind === 'objects' ? 'objects' : 'archive'
    const member = (body.member || '').trim()
    if (!member) {
      return NextResponse.json({ error: 'member is required' }, { status: 400 })
    }
    const prepared = await prepareDemoMember({
      kind,
      objectName: body.objectName,
      member,
      bucket: body.bucket,
    })
    return NextResponse.json(prepared)
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Failed to prepare demo member'
    return NextResponse.json({ error: message }, { status: 500 })
  }
}
