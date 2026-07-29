/**
 * Canonical image / archive format lists for the whole lab (upload UI, TAR/ZIP
 * indexing, demo GCS catalog, and loaders). Keep Python `image_io.py` in sync.
 */

export const GEO_TIFF_EXTENSIONS = ['.tif', '.tiff', '.geotiff', '.gtiff'] as const

export const RASTER_EXTENSIONS = [
  '.png',
  '.jpg',
  '.jpeg',
  '.webp',
  '.bmp',
  '.gif',
] as const

export const JPEG2000_EXTENSIONS = ['.jp2', '.j2k', '.jpx'] as const

export const IMAGE_EXTENSIONS = [
  ...GEO_TIFF_EXTENSIONS,
  ...RASTER_EXTENSIONS,
  ...JPEG2000_EXTENSIONS,
] as const

export const ARCHIVE_EXTENSIONS = ['.tar', '.tar.gz', '.tgz', '.zip'] as const

/** Match a supported image path (case-insensitive), including Landsat `.TIF`. */
export const IMAGE_EXT_RE =
  /\.(tif|tiff|geotiff|gtiff|png|jpe?g|webp|bmp|gif|jp2|j2k|jpx)$/i

/** Match a supported archive path (TAR / TAR.GZ / ZIP). */
export const ARCHIVE_EXT_RE = /\.(tar\.gz|tgz|tar|zip)$/i

export const TAR_EXT_RE = /\.(tar\.gz|tgz|tar)$/i
export const ZIP_EXT_RE = /\.zip$/i

/** Demo / GCS objects: images + archives. */
export const DEMO_OBJECT_EXT_RE =
  /\.(tif|tiff|geotiff|gtiff|png|jpe?g|webp|bmp|gif|jp2|j2k|jpx|tar\.gz|tgz|tar|zip)$/i

/**
 * `<input type="file" accept=…>` for the main upload control.
 * Include uppercase extensions — some OS file pickers are case-sensitive.
 */
export const FILE_INPUT_ACCEPT = [
  ...IMAGE_EXTENSIONS,
  ...IMAGE_EXTENSIONS.map((e) => e.toUpperCase()),
  ...ARCHIVE_EXTENSIONS,
  ...ARCHIVE_EXTENSIONS.map((e) => e.toUpperCase()),
  '.TIF',
  '.TIFF',
  '.ZIP',
  'image/tiff',
  'image/png',
  'image/jpeg',
  'image/webp',
  'image/bmp',
  'image/gif',
  'image/jp2',
  'application/x-tar',
  'application/gzip',
  'application/zip',
  'application/x-zip-compressed',
].join(',')

/** Single-band / NIR companion upload (rasters only, no archives). */
export const RASTER_FILE_INPUT_ACCEPT = [
  ...GEO_TIFF_EXTENSIONS,
  ...GEO_TIFF_EXTENSIONS.map((e) => e.toUpperCase()),
  ...RASTER_EXTENSIONS,
  ...JPEG2000_EXTENSIONS,
  '.TIF',
  '.TIFF',
  'image/tiff',
  'image/png',
  'image/jpeg',
  'image/webp',
  'image/bmp',
  'image/gif',
  'image/jp2',
].join(',')

export const SUPPORTED_IMAGE_LABEL =
  '.tif / .tiff / .geotiff / .png / .jpg / .jpeg / .webp / .bmp / .gif / .jp2'

export const SUPPORTED_ARCHIVE_LABEL = '.tar / .tar.gz / .tgz / .zip'

export function extensionOf(filename: string): string {
  const lower = filename.toLowerCase().replace(/\\/g, '/')
  const base = lower.split('/').pop() || lower
  if (base.endsWith('.tar.gz')) return '.tar.gz'
  const i = base.lastIndexOf('.')
  return i >= 0 ? base.slice(i) : ''
}

export function isSupportedImageFilename(filename: string): boolean {
  return IMAGE_EXT_RE.test(filename.replace(/\\/g, '/'))
}

export function isGeoTiffFilename(filename: string): boolean {
  return (GEO_TIFF_EXTENSIONS as readonly string[]).includes(extensionOf(filename))
}

export function isJpeg2000Filename(filename: string): boolean {
  return (JPEG2000_EXTENSIONS as readonly string[]).includes(extensionOf(filename))
}

export function isArchiveFilename(filename: string): boolean {
  return ARCHIVE_EXT_RE.test(filename)
}

export function isTarFilename(filename: string): boolean {
  return TAR_EXT_RE.test(filename)
}

export function isZipFilename(filename: string): boolean {
  return ZIP_EXT_RE.test(filename)
}

export function mimeForImageFilename(filename: string): string {
  switch (extensionOf(filename)) {
    case '.tif':
    case '.tiff':
    case '.geotiff':
    case '.gtiff':
      return 'image/tiff'
    case '.png':
      return 'image/png'
    case '.jpg':
    case '.jpeg':
      return 'image/jpeg'
    case '.webp':
      return 'image/webp'
    case '.bmp':
      return 'image/bmp'
    case '.gif':
      return 'image/gif'
    case '.jp2':
    case '.j2k':
    case '.jpx':
      return 'image/jp2'
    default:
      return 'application/octet-stream'
  }
}

/** Classic TIFF / BigTIFF magic (II*\0, MM\0*, II+\0, MM\0+). */
export function bufferLooksLikeTiff(buffer: ArrayBuffer | Uint8Array): boolean {
  const u = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer)
  if (u.length < 4) return false
  const leTiff = u[0] === 0x49 && u[1] === 0x49 && u[2] === 0x2a && u[3] === 0x00
  const beTiff = u[0] === 0x4d && u[1] === 0x4d && u[2] === 0x00 && u[3] === 0x2a
  const leBig = u[0] === 0x49 && u[1] === 0x49 && u[2] === 0x2b && u[3] === 0x00
  const beBig = u[0] === 0x4d && u[1] === 0x4d && u[2] === 0x00 && u[3] === 0x2b
  return leTiff || beTiff || leBig || beBig
}

/** JP2 box signature or JPEG 2000 codestream. */
export function bufferLooksLikeJpeg2000(buffer: ArrayBuffer | Uint8Array): boolean {
  const u = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer)
  if (u.length >= 8 && u[4] === 0x6a && u[5] === 0x50 && u[6] === 0x20 && u[7] === 0x20) {
    return true
  }
  return u.length >= 4 && u[0] === 0xff && u[1] === 0x4f && u[2] === 0xff && u[3] === 0x51
}

/** ZIP local/central/end signatures start with PK. */
export function bufferLooksLikeZip(buffer: ArrayBuffer | Uint8Array): boolean {
  const u = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer)
  if (u.length < 4) return false
  return u[0] === 0x50 && u[1] === 0x4b && (u[2] === 0x03 || u[2] === 0x05 || u[2] === 0x07)
}

export function bufferLooksLikeGzip(buffer: ArrayBuffer | Uint8Array): boolean {
  const u = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer)
  return u.length >= 2 && u[0] === 0x1f && u[1] === 0x8b
}

export function bufferLooksLikeTar(buffer: ArrayBuffer | Uint8Array): boolean {
  const u = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer)
  if (u.length < 262) return false
  // ustar magic at offset 257
  return (
    u[257] === 0x75 &&
    u[258] === 0x73 &&
    u[259] === 0x74 &&
    u[260] === 0x61 &&
    u[261] === 0x72
  )
}

export type ArchiveKind = 'zip' | 'tar' | 'tar.gz'

/** Detect archive kind from filename and/or magic bytes. */
export function detectArchiveKind(
  buffer: ArrayBuffer | Uint8Array,
  filename = '',
): ArchiveKind | null {
  const lower = filename.toLowerCase()
  if (lower.endsWith('.zip') || bufferLooksLikeZip(buffer)) return 'zip'
  if (lower.endsWith('.tar.gz') || lower.endsWith('.tgz') || bufferLooksLikeGzip(buffer)) {
    return 'tar.gz'
  }
  if (lower.endsWith('.tar') || bufferLooksLikeTar(buffer)) return 'tar'
  return null
}
