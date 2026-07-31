import { randomUUID } from 'crypto'
import {
  extractArchiveMember,
  isTarArchive,
  listArchiveImages,
  scanUncompressedTarImageEntries,
  type TarImageEntry,
} from '@/lib/archive'
import { cloudRunAuthHeaders } from '@/lib/cloudRunAuth'
import {
  gcsUploadBucket,
  serviceAccountCredentialsForDemo,
  storageClientForDemo,
} from '@/lib/gcs'
import { DEMO_OBJECT_EXT_RE, ARCHIVE_EXT_RE, IMAGE_EXT_RE, SUPPORTED_IMAGE_LABEL } from '@/lib/imageFormats'

const DEMO_IMAGE_EXT = DEMO_OBJECT_EXT_RE
const DEMO_ARCHIVE_EXT = ARCHIVE_EXT_RE
/** Standalone rasters in a bucket (not archives). */
const DEMO_LOOSE_IMAGE_EXT = IMAGE_EXT_RE
/** Archives larger than this must use manifest.json or Cloud Run listing. */
const LARGE_ARCHIVE_BYTES = 40 * 1024 * 1024

export type PreparedDemoMember = {
  downloadUrl: string
  filename: string
  size: number
  /** Staged object in GCS_UPLOAD_BUCKET — Cloud Run can read this without a browser re-upload. */
  gcsUri: string
}

export type DemoLightPreview = {
  previewPngBase64: string
  nativeWidth: number
  nativeHeight: number
  previewWidth: number
  previewHeight: number
  bandOrder: string[]
}

export type PreparedDemoMemberWithPreview = PreparedDemoMember &
  Partial<DemoLightPreview> & {
    lightPreview?: boolean
    previewError?: string
  }

type ManifestEntry = { name: string; offset: number; size: number }

type ArchiveCache = {
  buffer: ArrayBuffer
  archiveName: string
  members: string[]
  entries?: ManifestEntry[]
  loadedAt: number
}

/** Warm-instance cache so list + extract don't re-download the TAR every click. */
const archiveCache = new Map<string, ArchiveCache>()
const CACHE_TTL_MS = 30 * 60 * 1000

/** Manifest parsed from demo bucket (kept for ranged extracts). */
let manifestCache:
  | {
      bucket: string
      objectName: string
      archiveName: string
      members: string[]
      entries: ManifestEntry[]
      loadedAt: number
    }
  | null = null

function cacheKey(bucket: string, objectName: string): string {
  return `${bucket}/${objectName}`
}

function compressApiBase(): string | null {
  const url = process.env.COMPRESS_API_URL?.trim()
  return url ? url.replace(/\/$/, '') : null
}

function parseManifestEntries(raw: unknown): ManifestEntry[] {
  if (!Array.isArray(raw)) return []
  const out: ManifestEntry[] = []
  for (const item of raw) {
    if (!item || typeof item !== 'object') continue
    const rec = item as Record<string, unknown>
    const name = typeof rec.name === 'string' ? rec.name : ''
    const offset = Number(rec.offset)
    const size = Number(rec.size)
    if (!name || !Number.isFinite(offset) || !Number.isFinite(size) || size <= 0) {
      continue
    }
    out.push({ name, offset: Math.floor(offset), size: Math.floor(size) })
  }
  return out
}

async function downloadGcsObject(bucketName: string, objectName: string): Promise<Buffer> {
  const storage = storageClientForDemo()
  const [buf] = await storage.bucket(bucketName).file(objectName).download()
  return buf
}

async function downloadGcsRange(
  bucketName: string,
  objectName: string,
  start: number,
  endInclusive: number,
): Promise<Buffer> {
  const storage = storageClientForDemo()
  const [buf] = await storage.bucket(bucketName).file(objectName).download({
    start,
    end: endInclusive,
  })
  return buf
}

async function getCachedArchive(
  bucketName: string,
  objectName: string,
): Promise<ArchiveCache> {
  const key = cacheKey(bucketName, objectName)
  const hit = archiveCache.get(key)
  if (hit && Date.now() - hit.loadedAt < CACHE_TTL_MS) return hit

  const buf = await downloadGcsObject(bucketName, objectName)
  const copy = Uint8Array.from(buf).buffer
  const archiveName = objectName.split('/').pop() || objectName
  const members = listArchiveImages(copy, archiveName)
  const entry: ArchiveCache = {
    buffer: copy,
    archiveName,
    members,
    loadedAt: Date.now(),
  }
  archiveCache.set(key, entry)
  return entry
}

export type DemoCatalog =
  | {
      kind: 'archive'
      bucket: string
      objectName: string
      archiveName: string
      members: string[]
      entries?: ManifestEntry[]
    }
  | {
      kind: 'objects'
      bucket: string
      members: string[]
      objects: Array<{ name: string; size: number }>
    }

async function buildManifestViaCloudRun(
  bucketName: string,
  archiveName: string,
): Promise<{ members: string[]; entries: ManifestEntry[] } | null> {
  const base = compressApiBase()
  if (!base) return null

  const form = new FormData()
  form.set('bucket', bucketName)
  form.set('archive', archiveName)
  form.set('write_manifest', '1')

  const headers = await cloudRunAuthHeaders(base)
  const res = await fetch(`${base}/v1/demo/build-manifest`, {
    method: 'POST',
    headers,
    body: form,
  })
  const text = await res.text()
  if (!res.ok) {
    // Old Cloud Run revisions don't have this route yet (404) — caller falls back.
    if (res.status === 404) return null
    throw new Error(
      `Cloud Run build-manifest failed (${res.status}): ${text.slice(0, 400)}`,
    )
  }
  let parsed: {
    members?: string[]
    entries?: unknown
  }
  try {
    parsed = JSON.parse(text) as { members?: string[]; entries?: unknown }
  } catch {
    throw new Error('Cloud Run build-manifest returned non-JSON')
  }
  const members = Array.isArray(parsed.members)
    ? parsed.members.filter((m) => typeof m === 'string')
    : []
  if (!members.length) return null
  return { members, entries: parseManifestEntries(parsed.entries) }
}

/**
 * Index an uncompressed .tar with GCS Range GETs (headers only), then try to
 * persist manifest.json so the next cold start skips the walk.
 */
async function buildManifestViaGcsRangeScan(
  bucketName: string,
  archiveName: string,
  archiveByteLength: number,
): Promise<{ members: string[]; entries: ManifestEntry[] }> {
  const lower = archiveName.toLowerCase()
  if (lower.endsWith('.tar.gz') || lower.endsWith('.tgz')) {
    throw new Error(
      'Compressed .tar.gz demo archives need manifest.json or a Cloud Run rebuild; ranged header scan only works for uncompressed .tar.',
    )
  }

  const scanned: TarImageEntry[] = await scanUncompressedTarImageEntries(
    archiveByteLength,
    async (start, endInclusive) => {
      const buf = await downloadGcsRange(bucketName, archiveName, start, endInclusive)
      return new Uint8Array(buf)
    },
  )

  const entries: ManifestEntry[] = scanned.map((e) => ({
    name: e.name,
    offset: e.offset,
    size: e.size,
  }))
  const members = entries.map((e) => e.name)

  // Best-effort write so subsequent requests don't re-walk headers.
  try {
    const storage = storageClientForDemo()
    const payload = JSON.stringify(
      {
        archive: archiveName,
        members,
        entries,
      },
      null,
      2,
    )
    await storage.bucket(bucketName).file('manifest.json').save(payload, {
      contentType: 'application/json',
      resumable: false,
      metadata: { cacheControl: 'public, max-age=300' },
    })
  } catch {
    // objectViewer-only SA still works for this request via in-memory entries.
  }

  return { members, entries }
}

async function extractViaCloudRun(input: {
  bucket: string
  archive: string
  member: string
  offset?: number
  size?: number
}): Promise<PreparedDemoMember> {
  const base = compressApiBase()
  const staging = gcsUploadBucket()
  if (!base) {
    throw new Error(
      'COMPRESS_API_URL is required to extract members from large demo archives without a ranged manifest.',
    )
  }
  if (!staging) {
    throw new Error('GCS_UPLOAD_BUCKET is required to stage demo extracts.')
  }

  const form = new FormData()
  form.set('bucket', input.bucket)
  form.set('archive', input.archive)
  form.set('member', input.member)
  form.set('staging_bucket', staging)
  if (input.offset != null && input.size != null) {
    form.set('offset', String(input.offset))
    form.set('size', String(input.size))
  }

  const headers = await cloudRunAuthHeaders(base)
  const res = await fetch(`${base}/v1/demo/extract`, {
    method: 'POST',
    headers,
    body: form,
  })
  const text = await res.text()
  if (!res.ok) {
    throw new Error(`Cloud Run demo extract failed (${res.status}): ${text.slice(0, 400)}`)
  }
  const parsed = JSON.parse(text) as {
    objectName?: string
    bucket?: string
    filename?: string
    size?: number
  }
  if (!parsed.objectName || !parsed.bucket) {
    throw new Error('Cloud Run demo extract response missing objectName/bucket')
  }
  const storage = storageClientForDemo()
  const file = storage.bucket(parsed.bucket).file(parsed.objectName)
  const expires = Date.now() + 60 * 60 * 1000
  const [downloadUrl] = await file.getSignedUrl({
    version: 'v4',
    action: 'read',
    expires,
  })
  return {
    downloadUrl,
    filename: parsed.filename || input.member.split('/').pop() || 'member.bin',
    size: Number(parsed.size || 0),
    gcsUri: `gs://${parsed.bucket}/${parsed.objectName}`,
  }
}

/** ≤1024px PNG preview from a staged gs:// object (Cloud Run; no browser full download). */
export async function previewStagedGcsObject(input: {
  gcsUri: string
  filename?: string
  maxDim?: number
}): Promise<DemoLightPreview> {
  const base = compressApiBase()
  if (!base) {
    throw new Error('COMPRESS_API_URL is required for Cloud Run light previews')
  }
  const form = new FormData()
  form.set('gcs_uri', input.gcsUri)
  form.set('max_dim', String(input.maxDim ?? 1024))
  if (input.filename) form.set('filename', input.filename)

  const headers = await cloudRunAuthHeaders(base)
  const res = await fetch(`${base}/v1/demo/preview`, {
    method: 'POST',
    headers,
    body: form,
  })
  const text = await res.text()
  if (!res.ok) {
    throw new Error(`Cloud Run preview failed (${res.status}): ${text.slice(0, 400)}`)
  }
  const parsed = JSON.parse(text) as Partial<DemoLightPreview> & { detail?: unknown }
  if (!parsed.previewPngBase64 || !parsed.nativeWidth || !parsed.nativeHeight) {
    throw new Error('Cloud Run preview response missing PNG / dimensions')
  }
  return {
    previewPngBase64: parsed.previewPngBase64,
    nativeWidth: Number(parsed.nativeWidth),
    nativeHeight: Number(parsed.nativeHeight),
    previewWidth: Number(parsed.previewWidth || parsed.nativeWidth),
    previewHeight: Number(parsed.previewHeight || parsed.nativeHeight),
    bandOrder: Array.isArray(parsed.bandOrder)
      ? parsed.bandOrder.filter((b): b is string => typeof b === 'string')
      : ['gray'],
  }
}

/**
 * Stage a demo member, then (when possible) build a light Cloud Run preview so
 * the browser never downloads the full GeoTIFF for Engine → Cloud Run.
 */
export async function prepareDemoMemberLight(input: {
  kind: 'archive' | 'objects'
  objectName?: string
  member: string
  bucket?: string
  maxDim?: number
}): Promise<PreparedDemoMemberWithPreview> {
  const prepared = await prepareDemoMember(input)
  try {
    const preview = await previewStagedGcsObject({
      gcsUri: prepared.gcsUri,
      filename: prepared.filename,
      maxDim: input.maxDim ?? 1024,
    })
    return { ...prepared, ...preview, lightPreview: true }
  } catch (err) {
    return {
      ...prepared,
      lightPreview: false,
      previewError: err instanceof Error ? err.message : String(err),
    }
  }
}

export async function buildDemoCatalog(
  requestedBucket?: string | null,
): Promise<DemoCatalog> {
  if (!serviceAccountCredentialsForDemo()) {
    throw new Error(
      'GOOGLE_SERVICE_ACCOUNT_JSON is required to read demo data from GCS',
    )
  }
  const { resolveDemoBucket } = await import('@/lib/gcsBuckets')
  const bucketName = resolveDemoBucket(requestedBucket)
  const storage = storageClientForDemo()
  const [files] = await storage.bucket(bucketName).getFiles({
    autoPaginate: true,
    maxResults: 200,
  })

  const candidates = files
    .filter((f) => !f.name.endsWith('/'))
    .filter((f) => DEMO_IMAGE_EXT.test(f.name))
    .map((f) => ({
      name: f.name,
      size: Number(f.metadata.size || 0),
    }))

  // Optional lightweight index so listing does not download a huge TAR on Vercel.
  const manifestFile = files.find(
    (f) =>
      f.name === 'manifest.json' ||
      f.name.endsWith('/manifest.json') ||
      f.name === 'demo-manifest.json',
  )
  if (manifestFile) {
    try {
      const [buf] = await manifestFile.download()
      const parsed = JSON.parse(buf.toString('utf8')) as {
        archive?: string
        objectName?: string
        archiveName?: string
        members?: string[]
        kind?: string
        entries?: unknown
      }
      const members = Array.isArray(parsed.members)
        ? parsed.members.filter((m) => typeof m === 'string')
        : []
      const objectName = parsed.archive || parsed.objectName
      const entries = parseManifestEntries(parsed.entries)
      const imageMembers = members.filter((m) => DEMO_LOOSE_IMAGE_EXT.test(m))
      if (imageMembers.length && objectName) {
        const archiveName =
          parsed.archiveName || objectName.split('/').pop() || objectName
        manifestCache = {
          bucket: bucketName,
          objectName,
          archiveName,
          members: [...imageMembers].sort(),
          entries: entries.filter((e) => DEMO_LOOSE_IMAGE_EXT.test(e.name)),
          loadedAt: Date.now(),
        }
        return {
          kind: 'archive',
          bucket: bucketName,
          objectName,
          archiveName,
          members: manifestCache.members,
          entries: manifestCache.entries.length ? manifestCache.entries : undefined,
        }
      }
      if (imageMembers.length && parsed.kind === 'objects') {
        return {
          kind: 'objects',
          bucket: bucketName,
          members: [...imageMembers].sort(),
          objects: imageMembers.map((name) => ({ name, size: 0 })),
        }
      }
    } catch {
      // Fall through to live listing / TAR scan.
    }
  }

  if (!candidates.length) {
    throw new Error(
      `No demo images found in gs://${bucketName}. Upload ${SUPPORTED_IMAGE_LABEL} or a .tar / .tar.gz / .zip.`,
    )
  }

  // Prefer loose rasters when present. Otherwise a leftover empty/docs ZIP/TAR in
  // the same bucket steals the catalog and fails with "archive does not contain
  // a supported image" even though the TIFs are still there.
  const looseImages = candidates.filter(
    (c) => DEMO_LOOSE_IMAGE_EXT.test(c.name) && !DEMO_ARCHIVE_EXT.test(c.name),
  )
  if (looseImages.length) {
    const objects = [...looseImages].sort((a, b) => a.name.localeCompare(b.name))
    return {
      kind: 'objects',
      bucket: bucketName,
      members: objects.map((o) => o.name),
      objects,
    }
  }

  const archives = candidates.filter((c) => DEMO_ARCHIVE_EXT.test(c.name))
  if (archives.length) {
    // Try largest first, but fall through to smaller archives if one is empty/unreadable.
    const ordered = [...archives].sort((a, b) => b.size - a.size)
    const errors: string[] = []

    for (const preferred of ordered) {
      const archiveName = preferred.name.split('/').pop() || preferred.name

      if (preferred.size > LARGE_ARCHIVE_BYTES) {
        let built: { members: string[]; entries: ManifestEntry[] } | null = null
        try {
          built = await buildManifestViaCloudRun(bucketName, preferred.name)
        } catch (err) {
          errors.push(
            `${preferred.name}: ${err instanceof Error ? err.message : String(err)}`,
          )
        }

        if (!built?.members.length) {
          try {
            built = await buildManifestViaGcsRangeScan(
              bucketName,
              preferred.name,
              preferred.size,
            )
          } catch (err) {
            errors.push(
              `${preferred.name}: ${err instanceof Error ? err.message : String(err)}`,
            )
          }
        }

        if (built?.members.length) {
          manifestCache = {
            bucket: bucketName,
            objectName: preferred.name,
            archiveName,
            members: [...built.members].sort(),
            entries: built.entries,
            loadedAt: Date.now(),
          }
          return {
            kind: 'archive',
            bucket: bucketName,
            objectName: preferred.name,
            archiveName,
            members: manifestCache.members,
            entries: built.entries.length ? built.entries : undefined,
          }
        }
        continue
      }

      try {
        const cached = await getCachedArchive(bucketName, preferred.name)
        if (cached.members.length) {
          return {
            kind: 'archive',
            bucket: bucketName,
            objectName: preferred.name,
            archiveName: cached.archiveName,
            members: cached.members,
          }
        }
        errors.push(`${preferred.name}: archive listed zero supported images`)
      } catch (err) {
        errors.push(
          `${preferred.name}: ${err instanceof Error ? err.message : String(err)}`,
        )
      }
    }

    throw new Error(
      `No usable archive images found in gs://${bucketName}. ` +
        `Checked ${ordered.map((a) => a.name).join(', ')}. ` +
        `(${errors.join(' | ') || 'no details'})`,
    )
  }

  throw new Error(
    `No demo images found in gs://${bucketName}. Upload ${SUPPORTED_IMAGE_LABEL} or a .tar / .tar.gz / .zip.`,
  )
}

function safeName(filename: string): string {
  return filename.replace(/[^\w.\-()+ ]+/g, '_').slice(0, 180) || 'member.bin'
}

async function stageBytesAndSign(input: {
  bytes: Buffer | Uint8Array
  filename: string
  contentType?: string
}): Promise<PreparedDemoMember> {
  const staging = gcsUploadBucket()
  const storage = storageClientForDemo()
  const filename = input.filename.split('/').pop() || input.filename
  const contentType = input.contentType || 'application/octet-stream'
  const expires = Date.now() + 60 * 60 * 1000

  if (staging) {
    const objectName = `demo-extracts/${Date.now()}-${randomUUID()}/${safeName(filename)}`
    const file = storage.bucket(staging).file(objectName)
    await file.save(Buffer.from(input.bytes), {
      contentType,
      resumable: false,
      metadata: { cacheControl: 'private, max-age=3600' },
    })
    const [downloadUrl] = await file.getSignedUrl({
      version: 'v4',
      action: 'read',
      expires,
    })
    return {
      downloadUrl,
      filename,
      size: input.bytes.byteLength,
      gcsUri: `gs://${staging}/${objectName}`,
    }
  }

  // Fallback: signed read from demo bucket only works if that bucket has CORS.
  throw new Error(
    'GCS_UPLOAD_BUCKET is not set — needed to stage demo extracts with CORS for the browser.',
  )
}

/**
 * Server-side GCS copy into the staging bucket. Works for any source bucket the
 * Vercel SA can read (and that passes GCS_ALLOWED_BUCKETS). Avoids pulling the
 * full object through Vercel just to re-upload it.
 */
async function copyObjectToStaging(input: {
  sourceBucket: string
  sourceObject: string
  filename: string
}): Promise<PreparedDemoMember> {
  const staging = gcsUploadBucket()
  if (!staging) {
    throw new Error(
      'GCS_UPLOAD_BUCKET is not set — needed to stage demo objects for Cloud Run.',
    )
  }
  const storage = storageClientForDemo()
  const filename = input.filename.split('/').pop() || input.filename
  const destObject = `demo-extracts/${Date.now()}-${randomUUID()}/${safeName(filename)}`
  const src = storage.bucket(input.sourceBucket).file(input.sourceObject)
  const dest = storage.bucket(staging).file(destObject)

  const [meta] = await src.getMetadata()
  const size = Number(meta.size || 0)
  await src.copy(dest)

  const expires = Date.now() + 60 * 60 * 1000
  const [downloadUrl] = await dest.getSignedUrl({
    version: 'v4',
    action: 'read',
    expires,
  })
  return {
    downloadUrl,
    filename,
    size,
    gcsUri: `gs://${staging}/${destObject}`,
  }
}

function lookupManifestEntry(
  objectName: string,
  member: string,
): ManifestEntry | null {
  if (
    manifestCache &&
    manifestCache.objectName === objectName &&
    Date.now() - manifestCache.loadedAt < CACHE_TTL_MS
  ) {
    return manifestCache.entries.find((e) => e.name === member) || null
  }
  return null
}

export async function prepareDemoMember(input: {
  kind: 'archive' | 'objects'
  objectName?: string
  member: string
  bucket?: string
}): Promise<PreparedDemoMember> {
  const { resolveDemoBucket } = await import('@/lib/gcsBuckets')
  const bucketName = resolveDemoBucket(input.bucket)
  const member = input.member.trim()
  if (!member) throw new Error('member is required')

  if (input.kind === 'objects') {
    const filename = member.split('/').pop() || member
    // Prefer GCS server-side copy so other allowed buckets work without
    // streaming the full object through Vercel.
    try {
      return await copyObjectToStaging({
        sourceBucket: bucketName,
        sourceObject: member,
        filename,
      })
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err)
      // Fall back to download+reupload (same SA; useful if copy IAM is limited).
      try {
        const buf = await downloadGcsObject(bucketName, member)
        return stageBytesAndSign({ bytes: buf, filename })
      } catch (fallbackErr) {
        const fallback =
          fallbackErr instanceof Error ? fallbackErr.message : String(fallbackErr)
        throw new Error(
          `Failed to stage gs://${bucketName}/${member} into GCS_UPLOAD_BUCKET. ` +
            `Copy error: ${detail}. Download fallback: ${fallback}`,
        )
      }
    }
  }

  const objectName = (input.objectName || '').trim()
  if (!objectName || !isTarArchive(objectName)) {
    throw new Error('objectName must be a TAR archive for archive demos')
  }

  // Prefer ranged GET using manifest offsets (no full TAR download).
  let entry = lookupManifestEntry(objectName, member)
  if (!entry && manifestCache?.objectName !== objectName) {
    // Rebuild catalog once so manifestCache is warm (cheap when manifest.json exists).
    try {
      await buildDemoCatalog()
      entry = lookupManifestEntry(objectName, member)
    } catch {
      // continue
    }
  }

  if (entry) {
    const buf = await downloadGcsRange(
      bucketName,
      objectName,
      entry.offset,
      entry.offset + entry.size - 1,
    )
    const filename = member.split('/').pop() || member
    return stageBytesAndSign({ bytes: buf, filename })
  }

  // Large archives without offsets: extract on Cloud Run (streams TAR once).
  const storage = storageClientForDemo()
  const [meta] = await storage.bucket(bucketName).file(objectName).getMetadata()
  const archiveSize = Number(meta.size || 0)
  if (archiveSize > LARGE_ARCHIVE_BYTES) {
    // Last resort: walk headers to find this member's byte range, then Range-GET it.
    try {
      const scanned = await buildManifestViaGcsRangeScan(
        bucketName,
        objectName,
        archiveSize,
      )
      const found = scanned.entries.find((e) => e.name === member)
      if (found) {
        const buf = await downloadGcsRange(
          bucketName,
          objectName,
          found.offset,
          found.offset + found.size - 1,
        )
        const filename = member.split('/').pop() || member
        return stageBytesAndSign({ bytes: buf, filename })
      }
    } catch {
      // fall through to Cloud Run
    }
    return extractViaCloudRun({
      bucket: bucketName,
      archive: objectName,
      member,
    })
  }

  const cached = await getCachedArchive(bucketName, objectName)
  const { bytes, memberFilename } = extractArchiveMember(
    cached.buffer,
    cached.archiveName,
    member,
  )
  return stageBytesAndSign({
    bytes: new Uint8Array(bytes),
    filename: memberFilename,
  })
}
