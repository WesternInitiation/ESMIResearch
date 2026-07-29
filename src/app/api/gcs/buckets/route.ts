import { NextResponse } from 'next/server'
import { listProjectBuckets } from '@/lib/gcsBuckets'

export const runtime = 'nodejs'
export const maxDuration = 60

export async function GET() {
  try {
    const payload = await listProjectBuckets()
    return NextResponse.json(payload)
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Failed to list GCS buckets'
    return NextResponse.json({ error: message }, { status: 500 })
  }
}
