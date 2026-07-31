import { NextRequest, NextResponse } from 'next/server'
import { prepareDemoMember, prepareDemoMemberLight } from '@/lib/demoCatalog'

export const runtime = 'nodejs'
export const maxDuration = 300

export async function POST(request: NextRequest) {
  try {
    const body = (await request.json()) as {
      kind?: 'archive' | 'objects'
      objectName?: string
      member?: string
      bucket?: string
      /** When true, stage + Cloud Run ≤1024px preview (skip browser full download). */
      lightPreview?: boolean
      maxDim?: number
    }
    const kind = (body.kind === 'objects' ? 'objects' : 'archive') as
      | 'archive'
      | 'objects'
    const member = (body.member || '').trim()
    if (!member) {
      return NextResponse.json({ error: 'member is required' }, { status: 400 })
    }
    const input: {
      kind: 'archive' | 'objects'
      objectName?: string
      member: string
      bucket?: string
      maxDim?: number
    } = {
      kind,
      objectName: body.objectName,
      member,
      bucket: body.bucket,
      maxDim: body.maxDim,
    }
    const prepared = body.lightPreview
      ? await prepareDemoMemberLight(input)
      : await prepareDemoMember(input)
    return NextResponse.json(prepared)
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Failed to prepare demo member'
    return NextResponse.json({ error: message }, { status: 500 })
  }
}
