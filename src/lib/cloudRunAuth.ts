import { GoogleAuth } from 'google-auth-library'

/**
 * Build Authorization headers for a private Cloud Run service using a
 * service-account JSON key stored in GOOGLE_SERVICE_ACCOUNT_JSON.
 * Returns {} when no key is configured (public / local unauthenticated).
 */
export async function cloudRunAuthHeaders(audience: string): Promise<HeadersInit> {
  const raw = process.env.GOOGLE_SERVICE_ACCOUNT_JSON?.trim()
  if (!raw) return {}

  let credentials: Record<string, unknown>
  try {
    credentials = JSON.parse(raw) as Record<string, unknown>
  } catch {
    throw new Error(
      'GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON. Paste the full service-account key file contents.',
    )
  }

  const auth = new GoogleAuth({ credentials })
  const client = await auth.getIdTokenClient(audience.replace(/\/$/, ''))
  return client.getRequestHeaders()
}
