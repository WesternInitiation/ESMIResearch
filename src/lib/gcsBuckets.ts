import {
  gcsDemoBucket,
  gcsUploadBucket,
  serviceAccountCredentialsForDemo,
  storageClientForDemo,
} from '@/lib/gcs'

/**
 * GCP / tooling buckets that are not imagery catalogs. The lab lists every
 * project bucket when GCS_ALLOWED_BUCKETS is unset; without this filter,
 * picking e.g. `{project}_cloudbuild` tries to open Cloud Build source
 * .tar.gz archives (app code) and fails with "archive does not contain a
 * supported image".
 */
export function isNonImagerySystemBucket(name: string): boolean {
  const n = name.trim().toLowerCase()
  if (!n) return true
  if (n.endsWith('_cloudbuild')) return true
  if (n.endsWith('.appspot.com')) return true
  if (n.startsWith('artifacts.') && n.endsWith('.appspot.com')) return true
  if (n.includes('cloudbuild-logs')) return true
  if (n.startsWith('gcf-sources-')) return true
  if (n.startsWith('gcf-v2-')) return true
  if (n.startsWith('run-sources-')) return true
  return false
}

/** Buckets the Vercel SA may browse for demo/data loading. */
export function allowedGcsBuckets(): string[] {
  const raw = process.env.GCS_ALLOWED_BUCKETS?.trim()
  if (raw) {
    return [...new Set(raw.split(/[,;\s]+/).map((s) => s.trim()).filter(Boolean))]
  }
  const defaults = [gcsDemoBucket(), gcsUploadBucket()].filter(
    (b): b is string => Boolean(b),
  )
  return [...new Set(defaults)]
}

export function assertBucketAllowed(bucket: string): string {
  const name = bucket.trim()
  if (!name) throw new Error('bucket is required')
  if (!/^[a-z0-9][a-z0-9._-]{1,220}$/i.test(name)) {
    throw new Error(`Invalid GCS bucket name: ${name}`)
  }
  if (isNonImagerySystemBucket(name)) {
    throw new Error(
      `Bucket "${name}" is a GCP system/build bucket (not imagery). ` +
        `Use a demo/data bucket such as ${gcsDemoBucket()}.`,
    )
  }
  if (process.env.GCS_ALLOWED_BUCKETS?.trim()) {
    const allow = allowedGcsBuckets()
    if (!allow.includes(name)) {
      throw new Error(
        `Bucket "${name}" is not in GCS_ALLOWED_BUCKETS (${allow.join(', ')})`,
      )
    }
  }
  return name
}

export async function listProjectBuckets(): Promise<{
  buckets: Array<{ name: string; location?: string }>
  allowed: string[]
  defaultBucket: string
}> {
  if (!serviceAccountCredentialsForDemo()) {
    throw new Error('GOOGLE_SERVICE_ACCOUNT_JSON is required to list GCS buckets')
  }
  const storage = storageClientForDemo()
  const allow = allowedGcsBuckets()
  const defaultBucket = gcsDemoBucket()

  if (process.env.GCS_ALLOWED_BUCKETS?.trim()) {
    const buckets: Array<{ name: string; location?: string }> = []
    for (const name of allow) {
      if (isNonImagerySystemBucket(name)) continue
      try {
        const [meta] = await storage.bucket(name).getMetadata()
        buckets.push({
          name,
          location: typeof meta.location === 'string' ? meta.location : undefined,
        })
      } catch {
        buckets.push({ name })
      }
    }
    return { buckets, allowed: allow, defaultBucket }
  }

  try {
    const [all] = await storage.getBuckets()
    const buckets = all
      .map((b) => ({
        name: b.name,
        location:
          typeof b.metadata?.location === 'string' ? b.metadata.location : undefined,
      }))
      .filter((b) => !isNonImagerySystemBucket(b.name))
      .sort((a, b) => a.name.localeCompare(b.name))
    return {
      buckets: buckets.length ? buckets : allow.map((name) => ({ name })),
      allowed: allow,
      defaultBucket,
    }
  } catch {
    return {
      buckets: allow.map((name) => ({ name })),
      allowed: allow,
      defaultBucket,
    }
  }
}

/** Resolve demo bucket from query/body, falling back to env default. */
export function resolveDemoBucket(requested?: string | null): string {
  const trimmed = (requested || '').trim()
  if (!trimmed) return gcsDemoBucket()
  return assertBucketAllowed(trimmed)
}
