/** Safe JSON parse for fetch responses that may be HTML error pages. */
export async function readJsonResponse<T = Record<string, unknown>>(
  res: Response,
  label: string,
): Promise<T> {
  const contentType = res.headers.get('content-type') || ''
  const text = await res.text()
  const trimmed = text.trim()

  if (
    contentType.includes('text/html') ||
    trimmed.startsWith('<!DOCTYPE') ||
    trimmed.startsWith('<html')
  ) {
    throw new Error(
      `${label} returned HTML instead of JSON (HTTP ${res.status}). ` +
        `Usually the API timed out, crashed, or isn’t deployed yet. ` +
        `Open ${res.url || 'the API URL'} in a new tab, or check the latest Vercel deployment logs.`,
    )
  }

  if (!trimmed) {
    throw new Error(`${label} returned an empty body (HTTP ${res.status}).`)
  }

  try {
    return JSON.parse(trimmed) as T
  } catch {
    if (
      res.status === 504 ||
      /FUNCTION_INVOCATION_TIMEOUT/i.test(trimmed)
    ) {
      throw new Error(
        `${label} timed out (HTTP ${res.status}, FUNCTION_INVOCATION_TIMEOUT). ` +
          `Vercel’s proxy limit is ~5 minutes. Redeploy Cloud Run for async jobs, ` +
          `or lower Max processing size to 2048/1024.`,
      )
    }
    throw new Error(
      `${label} returned non-JSON (HTTP ${res.status}): ${trimmed.slice(0, 160)}`,
    )
  }
}
