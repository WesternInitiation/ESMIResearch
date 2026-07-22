import { NextResponse } from 'next/server'
import { supabaseConfigured } from '@/lib/supabase/server'

export async function GET() {
  return NextResponse.json({ configured: supabaseConfigured() })
}
