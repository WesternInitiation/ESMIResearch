import { gunzipSync } from 'fflate'

/** Raster-like members the lab can try to load. */
const IMAGE_EXT =
  /\.(tif|tiff|geotiff|png|jpe?g|webp|bmp|gif|jp2|j2k|jpx)$/i
/** Soft ceiling so pathological archives don't hang the tab forever. */
const MAX_MEMBERS = 500_000
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
  const parts = path.split('/')
  const base = parts[parts.length - 1] || path
  if (base === '.DS_Store' || base.startsWith('._')) return true
  if (parts.some((p) => p === '__MACOSX')) return true
  return false
}

function isRegularFileType(typeFlag: string): boolean {
  // '0' / NUL = normal, '7' = contiguous, 'S' = GNU sparse (payload still usable).
  return typeFlag === '0' || typeFlag === '\0' || typeFlag === '7' || typeFlag === 'S'
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
  return parseInt(raw, 8) || 0
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
    throw new Error('Could not decompress the .tar.gz / .tgz archive.')
  }
}

type TarEntry = { name: string; size: number; typeFlag: string; offset: number }

function* iterateTar(data: Uint8Array): Generator<TarEntry> {
  let offset = 0
  let members = 0
  let pendingLongName: string | null = null

  while (offset + 512 <= data.length) {
    const header = data.subarray(offset, offset + 512)
    const isEmpty = header.every((b) => b === 0)
    if (isEmpty) break

    members += 1
    if (members > MAX_MEMBERS) {
      throw new Error('The TAR archive contains too many members to index.')
    }

    const name = readCString(header, 0, 100)
    const size = parseOctal(header, 124, 12)
    const typeFlag = String.fromCharCode(header[156] || 48)
    const prefix = readCString(header, 345, 155)
    let fullName = (prefix ? `${prefix}/${name}` : name).replace(/^\.\//, '')
    const dataOffset = offset + 512
    const padded = Math.ceil(Math.max(size, 0) / 512) * 512

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

/**
 * List loadable image members in a TAR.
 * Non-image / junk / duplicate / oversized members are skipped — they do not
 * fail the archive. Any member count / mix of files is fine as long as ≥1 image.
 */
export function listArchiveImages(buffer: ArrayBuffer, filename: string): string[] {
  const data = expandTarBytes(buffer, filename)
  const names: string[] = []
  const seen = new Set<string>()
  for (const entry of iterateTar(data)) {
    if (!isRegularFileType(entry.typeFlag)) continue
    if (isJunkArchivePath(entry.name)) continue
    if (!IMAGE_EXT.test(entry.name)) continue
    if (entry.size <= 0 || entry.size > MAX_INGEST_BYTES) continue
    if (seen.has(entry.name)) continue
    seen.add(entry.name)
    names.push(entry.name)
  }
  if (!names.length) {
    throw new Error(
      'The TAR archive does not contain a supported image (.tif, .tiff, .png, .jpg, .jpeg, .webp, .bmp, .gif, .jp2).',
    )
  }
  return names.sort()
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
    const isEmpty = header.every((b) => b === 0)
    if (isEmpty) break

    members += 1
    if (members > MAX_MEMBERS) {
      throw new Error('The TAR archive contains too many members to index.')
    }

    const name = readCString(header, 0, 100)
    const size = parseOctal(header, 124, 12)
    const typeFlag = String.fromCharCode(header[156] || 48)
    const prefix = readCString(header, 345, 155)
    let fullName = (prefix ? `${prefix}/${name}` : name).replace(/^\.\//, '')
    const dataOffset = offset + 512
    const padded = Math.ceil(Math.max(size, 0) / 512) * 512

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
    break // first match wins (duplicates ignored)
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
