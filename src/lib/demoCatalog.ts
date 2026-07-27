import { randomUUID } from 'crypto'
import {
  extractArchiveMember,
  isTarArchive,
  listArchiveImages,
} from '@/lib/archive'
import {
  gcsDemoBucket,
  gcsUploadBucket,
  serviceAccountCredentialsForDemo,
  storageClientForDemo,
} from '@/lib/gcs'

const DEMO_IMAGE_EXT = /\.(tif|tiff|geotiff|png|jpe?g|webp|tar\.gz|tgz|tar)$/i
const DEMO_ARCHIVE_EXT = /\.(tar\.gz|tgz|tar)$/i

type ArchiveCache = {
  buffer: ArrayBuffer
  archiveName: string
  members: string[]
  loadedAt: number
}

/** Warm-instance cache so list + extract don't re-download the TAR every click. */
const archiveCache = new Map<string, ArchiveCache>()
const CACHE_TTL_MS = 30 * 60 * 1000

function cacheKey(bucket: string, objectName: string): string {
  return `${bucket}/${objectName}`
}

async function downloadGcsObject(bucketName: string, objectName: string): Promise<Buffer> {
  const storage = storageClientForDemo()
  const [buf] = await storage.bucket(bucketName).file(objectName).download()
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
    }
  | {
      kind: 'objects'
      bucket: string
      members: string[]
      objects: Array<{ name: string; size: number }>
    }

export async function buildDemoCatalog(): Promise<DemoCatalog> {
  if (!serviceAccountCredentialsForDemo()) {
    throw new Error(
      'GOOGLE_SERVICE_ACCOUNT_JSON is required to read demo data from GCS',
    )
  }
  const bucketName = gcsDemoBucket()
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

  if (!candidates.length) {
    throw new Error(
      `No demo images found in gs://${bucketName}. Upload a .tif / .png / .tar(.gz).`,
    )
  }

  const archives = candidates.filter((c) => DEMO_ARCHIVE_EXT.test(c.name))
  if (archives.length) {
    const preferred = [...archives].sort((a, b) => b.size - a.size)[0]
    const cached = await getCachedArchive(bucketName, preferred.name)
    return {
      kind: 'archive',
      bucket: bucketName,
      objectName: preferred.name,
      archiveName: cached.archiveName,
      members: cached.members,
    }
  }

  const objects = [...candidates].sort((a, b) => a.name.localeCompare(b.name))
  return {
    kind: 'objects',
    bucket: bucketName,
    members: objects.map((o) => o.name),
    objects,
  }
}

async function stageBytesAndSign(input: {
  bytes: Buffer | Uint8Array
  filename: string
  contentType?: string
}): Promise<{ downloadUrl: string; filename: string; size: number }> {
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
    return { downloadUrl, filename, size: input.bytes.byteLength }
  }

  // Fallback: signed read from demo bucket only works if that bucket has CORS.
  throw new Error(
    'GCS_UPLOAD_BUCKET is not set — needed to stage demo extracts with CORS for the browser.',
  )
}

function safeName(filename: string): string {
  return filename.replace(/[^\w.\-()+ ]+/g, '_').slice(0, 180) || 'member.bin'
}

export async function prepareDemoMember(input: {
  kind: 'archive' | 'objects'
  objectName?: string
  member: string
}): Promise<{ downloadUrl: string; filename: string; size: number }> {
  const bucketName = gcsDemoBucket()
  const member = input.member.trim()
  if (!member) throw new Error('member is required')

  if (input.kind === 'objects') {
    const buf = await downloadGcsObject(bucketName, member)
    const filename = member.split('/').pop() || member
    return stageBytesAndSign({ bytes: buf, filename })
  }

  const objectName = (input.objectName || '').trim()
  if (!objectName || !isTarArchive(objectName)) {
    throw new Error('objectName must be a TAR archive for archive demos')
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
