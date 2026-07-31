import { fromArrayBuffer } from 'geotiff'
import {
  extractArchiveMember,
  initialArchiveFolder,
  isTarArchive,
  listArchiveListing,
  listingFromMembers,
  type ArchiveListing,
} from './archive'
import {
  SUPPORTED_IMAGE_LABEL,
  bufferLooksLikeJpeg2000,
  bufferLooksLikeTiff,
  detectArchiveKind,
  isGeoTiffFilename,
  isJpeg2000Filename,
  isSupportedImageFilename,
  mimeForImageFilename,
} from './imageFormats'
import type { BandMap, ImageSize } from './types'

export type LoadedImage = {
  bands: BandMap
  bandOrder: string[]
  size: ImageSize
  sourceType: 'geotiff' | 'raster'
  originalBytes: number
  previewRgba: Uint8ClampedArray
  filename: string
  archiveMember?: string
  /**
   * Full file dimensions when the raster was decoded below native size for the UI
   * (e.g. Cloud Run preview path). Codecs on the server still see the original file.
   */
  fileNativeWidth?: number
  fileNativeHeight?: number
  /**
   * GeoTIFF ground sample distance when available (CRS units / pixel, usually meters).
   * x = pixel width, y = pixel height.
   */
  groundResolution?: { x: number; y: number }
}

export type LoadImageOptions = {
  /**
   * Decode GeoTIFF rasters no larger than this on the longest side.
   * Use for UI previews so huge Landsat scenes do not OOM the tab;
   * Cloud Run still receives the original File bytes.
   */
  maxDecodeDim?: number
}

export type ArchiveSelection = {
  /** Present for locally uploaded archives; empty/omitted for GCS demo catalogs. */
  buffer?: ArrayBuffer
  archiveName: string
  /** All image paths in the archive (for NDVI/NDWI pairing across folders). */
  members: string[]
  /** Full folder + image index for browsing. */
  listing: ArchiveListing
  /** Current folder prefix inside the archive ('' = root). */
  folderPath: string
  /** Lazy demo archive/objects — members are fetched on demand via /api/demo/member. */
  demoRemote?: {
    kind: 'archive' | 'objects'
    objectName?: string
    bucket?: string
  }
}

export function archiveSelectionFromMembers(
  archiveName: string,
  members: string[],
  extras: Partial<Pick<ArchiveSelection, 'buffer' | 'demoRemote'>> = {},
): ArchiveSelection {
  const listing = listingFromMembers(members)
  return {
    archiveName,
    members: listing.images,
    listing,
    folderPath: initialArchiveFolder(listing),
    ...extras,
  }
}

function normalizeBandNames(count: number): string[] {
  if (count === 1) return ['gray']
  if (count === 3) return ['red', 'green', 'blue']
  if (count === 4) return ['blue', 'green', 'red', 'nir']
  return Array.from({ length: count }, (_, i) => `band_${i + 1}`)
}

function percentileStretch(channel: Float64Array): Uint8ClampedArray {
  // Sample for speed on large rasters instead of sorting every pixel.
  const n = channel.length
  const sampleCount = Math.min(n, 20000)
  const step = Math.max(1, Math.floor(n / sampleCount))
  const sample: number[] = []
  for (let i = 0; i < n; i += step) sample.push(channel[i])
  sample.sort((a, b) => a - b)
  const lo = sample[Math.floor(sample.length * 0.02)] ?? 0
  const hi = sample[Math.floor(sample.length * 0.98)] ?? 1
  const out = new Uint8ClampedArray(channel.length)
  const scale = hi > lo ? 255 / (hi - lo) : 1
  for (let i = 0; i < channel.length; i++) {
    out[i] = Math.max(0, Math.min(255, Math.round((channel[i] - lo) * scale)))
  }
  return out
}

export function toPreviewRgba(
  bands: BandMap,
  bandOrder: string[],
  width: number,
  height: number,
): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(width * height * 4)
  let r: Float64Array
  let g: Float64Array
  let b: Float64Array
  if (bands.red && bands.green && bands.blue) {
    r = bands.red
    g = bands.green
    b = bands.blue
  } else if (bandOrder.length >= 3) {
    r = bands[bandOrder[0]]
    g = bands[bandOrder[1]]
    b = bands[bandOrder[2]]
  } else {
    const gray = bands[bandOrder[0]]
    r = g = b = gray
  }
  const rs = percentileStretch(r)
  const gs = percentileStretch(g)
  const bs = percentileStretch(b)
  for (let i = 0; i < width * height; i++) {
    const o = i * 4
    rgba[o] = rs[i]
    rgba[o + 1] = gs[i]
    rgba[o + 2] = bs[i]
    rgba[o + 3] = 255
  }
  return rgba
}

export async function inspectUpload(
  file: File,
): Promise<{ kind: 'image'; file: File } | { kind: 'archive'; selection: ArchiveSelection }> {
  const buffer = await file.arrayBuffer()
  // Detect ZIP/TAR by extension or magic bytes (folder-of-.TIF zips, misnamed downloads).
  if (detectArchiveKind(buffer, file.name) || isTarArchive(file.name)) {
    const listing = listArchiveListing(buffer, file.name)
    return {
      kind: 'archive',
      selection: {
        buffer,
        archiveName: file.name,
        members: listing.images,
        listing,
        folderPath: initialArchiveFolder(listing),
      },
    }
  }
  return { kind: 'image', file }
}

export async function loadArchiveMemberImage(
  selection: ArchiveSelection,
  memberName: string,
  options: LoadImageOptions = {},
): Promise<LoadedImage> {
  if (selection.demoRemote) {
    const { fetchDemoMemberFile } = await import('@/lib/demoData')
    const { file } = await fetchDemoMemberFile({
      kind: selection.demoRemote.kind,
      objectName: selection.demoRemote.objectName,
      member: memberName,
      bucket: selection.demoRemote.bucket,
    })
    const loaded = await loadImageFile(file, options)
    const memberFilename = memberName.split('/').pop() || memberName
    return {
      ...loaded,
      archiveMember: memberName,
      filename: `${selection.archiveName} → ${memberFilename}`,
    }
  }
  if (!selection.buffer) {
    throw new Error('Archive bytes are missing')
  }
  const { bytes, memberFilename } = extractArchiveMember(
    selection.buffer,
    selection.archiveName,
    memberName,
  )
  const loaded = await loadImageBuffer(bytes, memberFilename, bytes.byteLength)
  return { ...loaded, archiveMember: memberName, filename: `${selection.archiveName} → ${memberFilename}` }
}

/** First band array from a loaded image (works for single-band gray TIFs). */
export function firstBand(loaded: LoadedImage): Float64Array {
  const name = loaded.bandOrder[0]
  if (!name || !loaded.bands[name]) {
    throw new Error('Image has no readable bands')
  }
  return loaded.bands[name]
}

export function isLikelySingleBand(loaded: LoadedImage): boolean {
  return loaded.bandOrder.length === 1
}

/** Heuristic: Landsat/Sentinel-style names → spectral role. */
export function guessBandRole(
  filename: string,
): 'red' | 'nir' | 'green' | 'swir' | null {
  const n = filename.toLowerCase()
  if (
    /(?:^|[_\-.])(?:sr[_-]?)?b3(?:[_\-.]|$)/.test(n) ||
    /(?:^|[_\-.])green(?:[_\-.]|$)/.test(n) ||
    /band[_-]?3/.test(n)
  ) {
    return 'green'
  }
  if (
    /(?:^|[_\-.])(?:sr[_-]?)?b4(?:[_\-.]|$)/.test(n) ||
    /(?:^|[_\-.])red(?:[_\-.]|$)/.test(n) ||
    /band[_-]?4/.test(n)
  ) {
    return 'red'
  }
  if (
    /(?:^|[_\-.])(?:sr[_-]?)?b5(?:[_\-.]|$)/.test(n) ||
    /(?:^|[_\-.])(?:sr[_-]?)?b8(?:[_\-.]|$)/.test(n) ||
    /(?:^|[_\-.])nir(?:[_\-.]|$)/.test(n) ||
    /band[_-]?5/.test(n) ||
    /band[_-]?8/.test(n)
  ) {
    return 'nir'
  }
  if (
    /(?:^|[_\-.])(?:sr[_-]?)?b6(?:[_\-.]|$)/.test(n) ||
    /(?:^|[_\-.])(?:sr[_-]?)?b7(?:[_\-.]|$)/.test(n) ||
    /(?:^|[_\-.])swir/.test(n) ||
    /band[_-]?6/.test(n) ||
    /band[_-]?7/.test(n)
  ) {
    return 'swir'
  }
  return null
}

/** @deprecated Use guessBandRole */
export function guessNdviRole(filename: string): 'red' | 'nir' | null {
  const role = guessBandRole(filename)
  if (role === 'red' || role === 'nir') return role
  return null
}

function resizeBandNearest(
  src: Float64Array,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number,
): Float64Array {
  if (srcW === dstW && srcH === dstH) return new Float64Array(src)
  const out = new Float64Array(dstW * dstH)
  for (let y = 0; y < dstH; y++) {
    const sy = Math.min(srcH - 1, Math.floor((y + 0.5) * (srcH / dstH)))
    for (let x = 0; x < dstW; x++) {
      const sx = Math.min(srcW - 1, Math.floor((x + 0.5) * (srcW / dstW)))
      out[y * dstW + x] = src[sy * srcW + sx]
    }
  }
  return out
}

export type NamedBandSource = {
  name: string
  image: LoadedImage
}

/**
 * Build a multi-band working image from named single-band sources.
 * Later bands are resampled to the first source's grid if needed.
 */
export function pairNamedBandImages(
  sources: NamedBandSource[],
  label: string,
): LoadedImage {
  if (sources.length < 2) {
    throw new Error('Need at least two band files to pair')
  }
  const anchor = sources[0]
  const width = anchor.image.size.width
  const height = anchor.image.size.height
  const bands: BandMap = {}
  let originalBytes = 0
  const members: string[] = []

  for (const src of sources) {
    let band = firstBand(src.image)
    if (src.image.size.width !== width || src.image.size.height !== height) {
      band = resizeBandNearest(
        band,
        src.image.size.width,
        src.image.size.height,
        width,
        height,
      )
    } else {
      band = new Float64Array(band)
    }
    bands[src.name] = band
    originalBytes += src.image.originalBytes
    if (src.image.archiveMember) members.push(src.image.archiveMember)
  }

  const bandOrder = sources.map((s) => s.name)
  const filename = `${label} · ${sources
    .map((s) => `${s.name}: ${s.image.filename}`)
    .join(' · ')}`
  return {
    bands,
    bandOrder,
    size: { width, height },
    sourceType: 'geotiff',
    originalBytes,
    previewRgba: toPreviewRgba(bands, bandOrder, width, height),
    filename,
    archiveMember: members.length ? members.join(' + ') : undefined,
    ...(anchor.image.groundResolution
      ? { groundResolution: anchor.image.groundResolution }
      : {}),
  }
}

/** Merge additional named bands into an existing loaded image (same grid). */
export function mergeNamedBands(
  base: LoadedImage,
  additions: NamedBandSource[],
  label: string,
): LoadedImage {
  const width = base.size.width
  const height = base.size.height
  const bands: BandMap = { ...base.bands }
  const bandOrder = [...base.bandOrder]
  let originalBytes = base.originalBytes
  const members: string[] = base.archiveMember ? [base.archiveMember] : []

  for (const src of additions) {
    let band = firstBand(src.image)
    if (src.image.size.width !== width || src.image.size.height !== height) {
      band = resizeBandNearest(
        band,
        src.image.size.width,
        src.image.size.height,
        width,
        height,
      )
    } else {
      band = new Float64Array(band)
    }
    bands[src.name] = band
    if (!bandOrder.includes(src.name)) bandOrder.push(src.name)
    originalBytes += src.image.originalBytes
    if (src.image.archiveMember) members.push(src.image.archiveMember)
  }

  return {
    bands,
    bandOrder,
    size: { width, height },
    sourceType: base.sourceType,
    originalBytes,
    previewRgba: toPreviewRgba(bands, bandOrder, width, height),
    filename: `${label} · ${base.filename}`,
    archiveMember: members.length ? members.join(' + ') : base.archiveMember,
    ...(base.groundResolution ? { groundResolution: base.groundResolution } : {}),
  }
}

/**
 * Build a Red+NIR working image from two sources (typically single-band GeoTIFFs).
 */
export function pairRedNirImages(
  redSource: LoadedImage,
  nirSource: LoadedImage,
): LoadedImage {
  return pairNamedBandImages(
    [
      { name: 'red', image: redSource },
      { name: 'nir', image: nirSource },
    ],
    'NDVI pair',
  )
}

/** Green + NIR (McFeeters NDWI) or Green + SWIR (MNDWI). */
export function pairNdwiImages(
  greenSource: LoadedImage,
  secondSource: LoadedImage,
  secondName: 'nir' | 'swir',
): LoadedImage {
  return pairNamedBandImages(
    [
      { name: 'green', image: greenSource },
      { name: secondName, image: secondSource },
    ],
    secondName === 'swir' ? 'MNDWI pair' : 'NDWI pair',
  )
}

/** Suggest Red/NIR member names from a TAR member list (Landsat B4/B5 etc.). */
export function suggestNdviMembers(members: string[]): { red?: string; nir?: string } {
  let red: string | undefined
  let nir: string | undefined
  for (const m of members) {
    const role = guessBandRole(m)
    if (role === 'red' && !red) red = m
    if (role === 'nir' && !nir) nir = m
  }
  return { red, nir }
}

/** Suggest Green + NIR/SWIR members for NDWI / MNDWI. */
export function suggestNdwiMembers(
  members: string[],
  prefer: 'nir' | 'swir' = 'nir',
): { green?: string; second?: string; secondRole: 'nir' | 'swir' } {
  let green: string | undefined
  let nir: string | undefined
  let swir: string | undefined
  for (const m of members) {
    const role = guessBandRole(m)
    if (role === 'green' && !green) green = m
    if (role === 'nir' && !nir) nir = m
    if (role === 'swir' && !swir) swir = m
  }
  if (prefer === 'swir' && swir) {
    return { green, second: swir, secondRole: 'swir' }
  }
  if (nir) return { green, second: nir, secondRole: 'nir' }
  if (swir) return { green, second: swir, secondRole: 'swir' }
  return { green, second: undefined, secondRole: prefer }
}

export async function loadImageFile(
  file: File,
  options: LoadImageOptions = {},
): Promise<LoadedImage> {
  const buffer = await file.arrayBuffer()
  return loadImageBuffer(buffer, file.name, file.size, options)
}

export async function loadImageBuffer(
  buffer: ArrayBuffer,
  filename: string,
  originalBytes: number,
  options: LoadImageOptions = {},
): Promise<LoadedImage> {
  const looksTiff = bufferLooksLikeTiff(buffer)
  const looksJp2 = bufferLooksLikeJpeg2000(buffer)
  const namedTiff = isGeoTiffFilename(filename)
  const namedJp2 = isJpeg2000Filename(filename)

  // Prefer GeoTIFF / classic TIFF whenever the name or magic says so — including
  // Landsat-style .TIF and mislabeled members that are still TIFF bytes.
  if (namedTiff || looksTiff) {
    try {
      return await loadGeoTiff(buffer, filename, originalBytes, options)
    } catch (err) {
      if (namedTiff || !isSupportedImageFilename(filename)) {
        const detail = err instanceof Error ? err.message : String(err)
        throw new Error(
          `Could not read TIFF/GeoTIFF "${filename.split('/').pop() || filename}": ${detail}`,
        )
      }
      // Rare false-positive magic on a normal raster — fall through.
    }
  }

  if (namedJp2 || looksJp2) {
    // Most browsers cannot decode JPEG 2000; try createImageBitmap, then guide the user.
    try {
      return await loadRasterViaCanvas(buffer, filename, originalBytes)
    } catch {
      throw new Error(
        `JPEG 2000 (${filename.split('/').pop() || filename}) is not decodable in this browser. ` +
          'Use Engine → Cloud Run, or convert the band to GeoTIFF / PNG first.',
      )
    }
  }

  if (!isSupportedImageFilename(filename)) {
    throw new Error(
      `Unsupported file type: ${filename}. Supported images: ${SUPPORTED_IMAGE_LABEL}.`,
    )
  }

  try {
    return await loadRasterViaCanvas(buffer, filename, originalBytes)
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err)
    throw new Error(
      `Could not decode image "${filename.split('/').pop() || filename}": ${detail}`,
    )
  }
}

async function loadGeoTiff(
  buffer: ArrayBuffer,
  filename: string,
  originalBytes: number,
  options: LoadImageOptions = {},
): Promise<LoadedImage> {
  const tiff = await fromArrayBuffer(buffer)
  const image = await tiff.getImage()
  const nativeWidth = image.getWidth()
  const nativeHeight = image.getHeight()
  const longest = Math.max(nativeWidth, nativeHeight)
  const maxDecode = options.maxDecodeDim ?? 0
  const scale =
    maxDecode > 0 && longest > maxDecode ? maxDecode / longest : 1
  const width = Math.max(1, Math.round(nativeWidth * scale))
  const height = Math.max(1, Math.round(nativeHeight * scale))

  const raster =
    width === nativeWidth && height === nativeHeight
      ? await image.readRasters({ interleave: false })
      : await image.readRasters({
          interleave: false,
          width,
          height,
          resampleMethod: 'bilinear',
        })

  const count = Array.isArray(raster) ? raster.length : 1
  const bandOrder = normalizeBandNames(count)
  const bands: BandMap = {}
  const pixels = width * height
  for (let i = 0; i < count; i++) {
    const src = (Array.isArray(raster) ? raster[i] : raster) as ArrayLike<number>
    const arr = new Float64Array(pixels)
    for (let p = 0; p < arr.length; p++) arr[p] = Number(src[p])
    bands[bandOrder[i]] = arr
  }
  if (bands.b4 && !bands.red) bands.red = bands.b4
  if (bands.b8 && !bands.nir) bands.nir = bands.b8
  const order = Object.keys(bands)
  const groundResolution = readGroundResolution(image as {
    getResolution?: () => number[] | undefined
    getFileDirectory?: () => Record<string, unknown> | object
  })
  return {
    bands,
    bandOrder: order,
    size: { width, height },
    sourceType: 'geotiff',
    originalBytes,
    previewRgba: toPreviewRgba(bands, order, width, height),
    filename,
    ...(scale < 1
      ? { fileNativeWidth: nativeWidth, fileNativeHeight: nativeHeight }
      : {}),
    ...(groundResolution ? { groundResolution } : {}),
  }
}

function readGroundResolution(image: {
  getResolution?: () => number[] | undefined
  getFileDirectory?: () => Record<string, unknown> | object
}): { x: number; y: number } | undefined {
  try {
    const res = image.getResolution?.()
    if (Array.isArray(res) && res.length >= 2) {
      const x = Math.abs(Number(res[0]))
      const y = Math.abs(Number(res[1]))
      if (Number.isFinite(x) && Number.isFinite(y) && x > 0 && y > 0) {
        return { x, y }
      }
    }
  } catch {
    // ignore
  }
  try {
    const dir = (image.getFileDirectory?.() || {}) as Record<string, unknown>
    const scale = dir.ModelPixelScale
    if (Array.isArray(scale) && scale.length >= 2) {
      const x = Math.abs(Number(scale[0]))
      const y = Math.abs(Number(scale[1]))
      if (Number.isFinite(x) && Number.isFinite(y) && x > 0 && y > 0) {
        return { x, y }
      }
    }
  } catch {
    // ignore
  }
  return undefined
}

async function loadRasterViaCanvas(
  buffer: ArrayBuffer,
  filename: string,
  originalBytes: number,
): Promise<LoadedImage> {
  const mime = mimeForImageFilename(filename)
  const blob = new Blob([buffer], { type: mime !== 'application/octet-stream' ? mime : undefined })
  let bitmap: ImageBitmap
  try {
    bitmap = await createImageBitmap(blob)
  } catch {
    // Retry with a typed blob when the extension was missing / wrong.
    const typed = new Blob([buffer], { type: mime || 'image/png' })
    bitmap = await createImageBitmap(typed)
  }
  const width = bitmap.width
  const height = bitmap.height
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Could not create canvas context')
  ctx.drawImage(bitmap, 0, 0)
  const imageData = ctx.getImageData(0, 0, width, height)
  const { data } = imageData
  const red = new Float64Array(width * height)
  const green = new Float64Array(width * height)
  const blue = new Float64Array(width * height)
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    red[p] = data[i]
    green[p] = data[i + 1]
    blue[p] = data[i + 2]
  }
  bitmap.close?.()
  const bands: BandMap = { red, green, blue }
  const bandOrder = ['red', 'green', 'blue']
  return {
    bands,
    bandOrder,
    size: { width, height },
    sourceType: 'raster',
    originalBytes,
    previewRgba: toPreviewRgba(bands, bandOrder, width, height),
    filename,
  }
}

export function rgbaToDataUrl(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
): string {
  const w = Math.max(1, Math.floor(width))
  const h = Math.max(1, Math.floor(height))
  const expected = w * h * 4
  if (rgba.length < expected) {
    throw new Error(
      `Preview buffer too small for ${w}×${h} (have ${rgba.length} B, need ${expected} B)`,
    )
  }
  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Could not create canvas context')
  const pixels = new Uint8ClampedArray(expected)
  pixels.set(rgba.subarray(0, expected))
  ctx.putImageData(new ImageData(pixels, w, h), 0, 0)
  return canvas.toDataURL('image/png')
}

export function bandsToPngBlob(
  bands: BandMap,
  bandOrder: string[],
  width: number,
  height: number,
): Promise<Blob> {
  const rgba = toPreviewRgba(bands, bandOrder, width, height)
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (!ctx) return Promise.reject(new Error('Could not create canvas context'))
  const pixels = new Uint8ClampedArray(rgba)
  ctx.putImageData(new ImageData(pixels, width, height), 0, 0)
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) reject(new Error('PNG encode failed'))
      else resolve(blob)
    }, 'image/png')
  })
}
