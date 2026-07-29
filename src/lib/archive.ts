import { gunzipSync } from 'fflate'

/** Raster-like members the lab can try to load. */
const IMAGE_EXT =
  /\.(tif|tiff|geotiff|png|jpe?g|webp|bmp|gif|jp2|j2k|jpx)$/i
/** Soft ceiling — real archives stop earlier via checksum / ustar magic. */
const MAX_MEMBERS = 2_000_000
/** Soft cap for a single extracted member payload (~2 GiB). */
export const MAX_INGEST_BYTES = 2 * 1024 * 1024 * 1024

export function isTarArchive(filename: string): boolean {
  const name = filename.toLowerCase()
  return name.endsWith('.tar') || name.endsWith('.tar.gz') || name.endsWith('.tgz')
}

export function isSupportedArchiveImage(path: string): boolean {
  return IMAGE_EXT.test(path) && !isJunkArchivePath(path)
}

function isJunkArchivePath(path: string): boolean {
  const parts = path.split('/').filter(Boolean)
  const base = parts[parts.length - 1] || path
  if (base === '.DS_Store' || base.startsWith('._')) return true
  if (parts.some((p) => p === '__MACOSX')) return true
  return false
}

function isRegularFileType(typeFlag: string): boolean {
  // '0' / NUL = normal, '7' = contiguous, 'S' = GNU sparse (payload still usable).
  return typeFlag === '0' || typeFlag === '\0' || typeFlag === '7' || typeFlag === 'S'
}

function isDirectoryType(typeFlag: string): boolean {
  return typeFlag === '5'
}

function readCString(bytes: Uint8Array, start: number, length: number): string {
  let end = start
  const limit = start + length
  while (end < limit && bytes[end] !== 0) end++
  return new TextDecoder().decode(bytes.subarray(start, end)).trim()
}

function parseOctal(bytes: Uint8Array, start: number, length: number): number {
  const raw = readCString(bytes, start, length).replace(/\0/g, '').trim()
  if (!raw) return 0
  const n = parseInt(raw, 8)
  return Number.isFinite(n) && n >= 0 ? n : 0
}

/** Validate ustar/old-tar header checksum so image payloads aren't scanned as members. */
function isValidTarHeader(header: Uint8Array): boolean {
  if (header.every((b) => b === 0)) return false
  let sum = 0
  for (let i = 0; i < 512; i++) {
    sum += i >= 148 && i < 156 ? 0x20 : header[i]
  }
  const stored = parseOctal(header, 148, 8)
  // Exact checksum match covers ustar and old pre-ustar headers. Reject
  // anything else so TIFF/JPEG payloads are never treated as headers.
  return stored === sum
}

function expandTarBytes(buffer: ArrayBuffer, filename: string): Uint8Array {
  const input = new Uint8Array(buffer)
  const lower = filename.toLowerCase()
  const looksGzip =
    lower.endsWith('.gz') ||
    lower.endsWith('.tgz') ||
    (input.length >= 2 && input[0] === 0x1f && input[1] === 0x8b)
  if (!looksGzip) return input
  try {
    return gunzipSync(input)
  } catch {
    throw new Error(
      'Could not decompress the .tar.gz / .tgz archive. Re-upload as .tar or check the file.',
    )
  }
}

type TarEntry = {
  name: string
  size: number
  typeFlag: string
  offset: number
}

function* iterateTar(data: Uint8Array): Generator<TarEntry> {
  let offset = 0
  let members = 0
  let pendingLongName: string | null = null

  while (offset + 512 <= data.length) {
    const header = data.subarray(offset, offset + 512)
    const isEmpty = header.every((b) => b === 0)
    if (isEmpty) break

    // Stop at first invalid block — do not scan binary payloads as headers
    // (that used to inflate member counts into the millions).
    if (!isValidTarHeader(header)) break

    members += 1
    if (members > MAX_MEMBERS) {
      throw new Error(
        'The TAR archive contains too many members to index. Try a scene folder TAR, or upload individual images.',
      )
    }

    const name = readCString(header, 0, 100)
    const size = parseOctal(header, 124, 12)
    const typeFlag = String.fromCharCode(header[156] || 48)
    const prefix = readCString(header, 345, 155)
    let fullName = (prefix ? `${prefix}/${name}` : name).replace(/^\.\//, '')
    const dataOffset = offset + 512
    const padded = Math.ceil(size / 512) * 512

    // GNU long-name / pax path: next header uses this name.
    if (typeFlag === 'L' || typeFlag === 'x' || typeFlag === 'g') {
      if (size > 0 && size < 64 * 1024 && dataOffset + size <= data.length) {
        const raw = data.subarray(dataOffset, dataOffset + size)
        const text = new TextDecoder().decode(raw).replace(/\0/g, '').trim()
        if (typeFlag === 'L') {
          pendingLongName = text
        } else {
          const m = text.match(/(?:^|\n)\d+ path=(.+?)(?:\n|$)/)
          if (m?.[1]) pendingLongName = m[1].replace(/\0/g, '').trim()
        }
      }
      offset = dataOffset + padded
      continue
    }

    if (pendingLongName) {
      fullName = pendingLongName.replace(/^\.\//, '')
      pendingLongName = null
    }

    yield { name: fullName, size, typeFlag, offset: dataOffset }
    offset = dataOffset + padded
  }
}

export type ArchiveListing = {
  /** All image member paths (sorted). */
  images: string[]
  /** Folder prefixes that contain images (sorted, no trailing slash). */
  folders: string[]
}

function folderPrefixesForPath(path: string): string[] {
  const parts = path.split('/').filter(Boolean)
  if (parts.length <= 1) return []
  const out: string[] = []
  for (let i = 1; i < parts.length; i++) {
    out.push(parts.slice(0, i).join('/'))
  }
  return out
}

/**
 * Index images + folders inside a TAR / TAR.GZ / TGZ.
 * Non-image / junk / duplicate / oversized members are skipped.
 */
export function listArchiveListing(
  buffer: ArrayBuffer,
  filename: string,
): ArchiveListing {
  const data = expandTarBytes(buffer, filename)
  const images: string[] = []
  const folderSet = new Set<string>()
  const seen = new Set<string>()

  for (const entry of iterateTar(data)) {
    const name = entry.name.replace(/\/$/, '')
    if (!name || isJunkArchivePath(name)) continue

    if (isDirectoryType(entry.typeFlag)) {
      folderSet.add(name)
      for (const p of folderPrefixesForPath(name)) folderSet.add(p)
      continue
    }
    if (!isRegularFileType(entry.typeFlag)) continue
    if (!IMAGE_EXT.test(name)) {
      // Still record parent folders so users can navigate into dirs that only
      // contain metadata + images deeper down… parents of any file help UX.
      for (const p of folderPrefixesForPath(name)) folderSet.add(p)
      continue
    }
    if (entry.size <= 0 || entry.size > MAX_INGEST_BYTES) continue
    if (seen.has(name)) continue
    seen.add(name)
    images.push(name)
    for (const p of folderPrefixesForPath(name)) folderSet.add(p)
  }

  if (!images.length) {
    throw new Error(
      'The TAR archive does not contain a supported image (.tif, .tiff, .png, .jpg, .jpeg, .webp, .bmp, .gif, .jp2).',
    )
  }
  return {
    images: images.sort((a, b) => a.localeCompare(b)),
    folders: [...folderSet].sort((a, b) => a.localeCompare(b)),
  }
}

/** @deprecated Prefer listArchiveListing — kept for callers that only need image paths. */
export function listArchiveImages(buffer: ArrayBuffer, filename: string): string[] {
  return listArchiveListing(buffer, filename).images
}

/** Build a listing from a flat member path list (e.g. demo catalog / manifest). */
export function listingFromMembers(members: string[]): ArchiveListing {
  const folderSet = new Set<string>()
  const images: string[] = []
  const seen = new Set<string>()
  for (const raw of members) {
    const name = raw.replace(/\/$/, '')
    if (!name || seen.has(name)) continue
    seen.add(name)
    images.push(name)
    for (const p of folderPrefixesForPath(name)) folderSet.add(p)
  }
  return {
    images: images.sort((a, b) => a.localeCompare(b)),
    folders: [...folderSet].sort((a, b) => a.localeCompare(b)),
  }
}

/**
 * Descend into single-child folder chains so Landsat-style TARs open
 * on the first folder that actually contains choices.
 */
export function initialArchiveFolder(listing: ArchiveListing): string {
  let path = ''
  for (let depth = 0; depth < 32; depth++) {
    const { folders, images } = archiveChildren(listing, path)
    if (images.length === 0 && folders.length === 1) {
      path = folders[0]
      continue
    }
    return path
  }
  return path
}

/**
 * Immediate child folders + images under ``prefix`` ('' = archive root).
 */
export function archiveChildren(
  listing: ArchiveListing,
  prefix = '',
): { folders: string[]; images: string[] } {
  const norm = prefix.replace(/^\/+|\/+$/g, '')
  const folderSet = new Set<string>()
  const images: string[] = []

  for (const image of listing.images) {
    if (norm) {
      if (image === norm || !image.startsWith(`${norm}/`)) continue
      const rest = image.slice(norm.length + 1)
      const slash = rest.indexOf('/')
      if (slash === -1) images.push(image)
      else folderSet.add(`${norm}/${rest.slice(0, slash)}`)
    } else {
      const slash = image.indexOf('/')
      if (slash === -1) images.push(image)
      else folderSet.add(image.slice(0, slash))
    }
  }

  // Include empty dirs recorded during index (rare but useful).
  for (const folder of listing.folders) {
    if (norm) {
      if (folder === norm || !folder.startsWith(`${norm}/`)) continue
      const rest = folder.slice(norm.length + 1)
      if (!rest.includes('/')) folderSet.add(folder)
    } else if (!folder.includes('/')) {
      folderSet.add(folder)
    }
  }

  return {
    folders: [...folderSet].sort((a, b) => a.localeCompare(b)),
    images: images.sort((a, b) => a.localeCompare(b)),
  }
}

export type TarImageEntry = { name: string; offset: number; size: number }

/**
 * Walk an uncompressed .tar via ranged reads (512-byte headers only).
 * Used for huge GCS demo archives so Vercel never downloads the whole object.
 */
export async function scanUncompressedTarImageEntries(
  archiveByteLength: number,
  readRange: (start: number, endInclusive: number) => Promise<Uint8Array>,
): Promise<TarImageEntry[]> {
  const entries: TarImageEntry[] = []
  const seen = new Set<string>()
  let offset = 0
  let members = 0
  let pendingLongName: string | null = null

  while (offset + 512 <= archiveByteLength) {
    const header = await readRange(offset, offset + 511)
    if (header.byteLength < 512) break
    if (header.every((b) => b === 0)) break
    if (!isValidTarHeader(header)) break

    members += 1
    if (members > MAX_MEMBERS) {
      throw new Error(
        'The TAR archive contains too many members to index. Try browsing a scene subfolder archive.',
      )
    }

    const name = readCString(header, 0, 100)
    const size = parseOctal(header, 124, 12)
    const typeFlag = String.fromCharCode(header[156] || 48)
    const prefix = readCString(header, 345, 155)
    let fullName = (prefix ? `${prefix}/${name}` : name).replace(/^\.\//, '')
    const dataOffset = offset + 512
    const padded = Math.ceil(size / 512) * 512

    if (typeFlag === 'L' || typeFlag === 'x' || typeFlag === 'g') {
      if (size > 0 && size < 64 * 1024 && dataOffset + size - 1 < archiveByteLength) {
        const raw = await readRange(dataOffset, dataOffset + size - 1)
        const text = new TextDecoder().decode(raw).replace(/\0/g, '').trim()
        if (typeFlag === 'L') {
          pendingLongName = text
        } else {
          const m = text.match(/(?:^|\n)\d+ path=(.+?)(?:\n|$)/)
          if (m?.[1]) pendingLongName = m[1].replace(/\0/g, '').trim()
        }
      }
      offset = dataOffset + padded
      continue
    }

    if (pendingLongName) {
      fullName = pendingLongName.replace(/^\.\//, '')
      pendingLongName = null
    }

    if (
      isRegularFileType(typeFlag) &&
      !isJunkArchivePath(fullName) &&
      IMAGE_EXT.test(fullName) &&
      size > 0 &&
      size <= MAX_INGEST_BYTES &&
      !seen.has(fullName)
    ) {
      seen.add(fullName)
      entries.push({ name: fullName, offset: dataOffset, size })
    }

    offset = dataOffset + padded
  }

  if (!entries.length) {
    throw new Error(
      'The TAR archive does not contain a supported image (.tif, .tiff, .png, .jpg, .jpeg, .webp, .bmp, .gif, .jp2).',
    )
  }
  return entries.sort((a, b) => a.name.localeCompare(b.name))
}

export function extractArchiveMember(
  buffer: ArrayBuffer,
  filename: string,
  memberName: string,
): { bytes: ArrayBuffer; memberFilename: string } {
  if (!isSupportedArchiveImage(memberName)) {
    throw new Error('The selected archive member is not a supported image.')
  }
  const data = expandTarBytes(buffer, filename)
  let match: TarEntry | null = null
  for (const entry of iterateTar(data)) {
    if (entry.name !== memberName) continue
    if (!isRegularFileType(entry.typeFlag)) continue
    match = entry
    break
  }
  if (!match) throw new Error('The selected archive image is missing.')
  if (match.size <= 0 || match.size > MAX_INGEST_BYTES) {
    throw new Error('The selected archive image is larger than the ~2 GiB ingest limit.')
  }
  const slice = data.subarray(match.offset, match.offset + match.size)
  const copy = new Uint8Array(slice.byteLength)
  copy.set(slice)
  const base = memberName.split('/').pop() || memberName
  return { bytes: copy.buffer, memberFilename: base }
}
