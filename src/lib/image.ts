import { fromArrayBuffer } from 'geotiff'
import {
  extractArchiveMember,
  isTarArchive,
  listArchiveImages,
} from './archive'
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
}

export type ArchiveSelection = {
  buffer: ArrayBuffer
  archiveName: string
  members: string[]
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
  if (isTarArchive(file.name)) {
    const members = listArchiveImages(buffer, file.name)
    return {
      kind: 'archive',
      selection: { buffer, archiveName: file.name, members },
    }
  }
  return { kind: 'image', file }
}

export async function loadArchiveMemberImage(
  selection: ArchiveSelection,
  memberName: string,
): Promise<LoadedImage> {
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

/** Heuristic: Landsat/Sentinel-style names → red vs nir. */
export function guessNdviRole(filename: string): 'red' | 'nir' | null {
  const n = filename.toLowerCase()
  if (
    /(?:^|[_\-.])(?:sr[_-]?)?b4(?:[_\-.]|$)/.test(n) ||
    /(?:^|[_\-.])red(?:[_\-.]|$)/.test(n) ||
    /band[_-]?4/.test(n)
  ) {
    return 'red'
  }
  if (
    /(?:^|[_\-.])(?:sr[_-]?)?b5(?:[_\-.]|$)/.test(n) || // Landsat 8/9 NIR
    /(?:^|[_\-.])(?:sr[_-]?)?b8(?:[_\-.]|$)/.test(n) || // Sentinel-2 NIR approx naming
    /(?:^|[_\-.])nir(?:[_\-.]|$)/.test(n) ||
    /band[_-]?5/.test(n) ||
    /band[_-]?8/.test(n)
  ) {
    return 'nir'
  }
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

/**
 * Build a Red+NIR working image from two sources (typically single-band GeoTIFFs).
 * NIR is resampled to the Red grid if dimensions differ.
 */
export function pairRedNirImages(
  redSource: LoadedImage,
  nirSource: LoadedImage,
): LoadedImage {
  const redBand = firstBand(redSource)
  let nirBand = firstBand(nirSource)
  const width = redSource.size.width
  const height = redSource.size.height

  if (
    nirSource.size.width !== width ||
    nirSource.size.height !== height
  ) {
    nirBand = resizeBandNearest(
      nirBand,
      nirSource.size.width,
      nirSource.size.height,
      width,
      height,
    )
  }

  const bands: BandMap = { red: new Float64Array(redBand), nir: nirBand }
  const bandOrder = ['red', 'nir']
  const filename = `NDVI pair · red: ${redSource.filename} · nir: ${nirSource.filename}`
  return {
    bands,
    bandOrder,
    size: { width, height },
    sourceType: 'geotiff',
    originalBytes: redSource.originalBytes + nirSource.originalBytes,
    previewRgba: toPreviewRgba(bands, bandOrder, width, height),
    filename,
    archiveMember: redSource.archiveMember
      ? `${redSource.archiveMember} + ${nirSource.archiveMember ?? nirSource.filename}`
      : undefined,
  }
}

/** Suggest Red/NIR member names from a TAR member list (Landsat B4/B5 etc.). */
export function suggestNdviMembers(members: string[]): { red?: string; nir?: string } {
  let red: string | undefined
  let nir: string | undefined
  for (const m of members) {
    const role = guessNdviRole(m)
    if (role === 'red' && !red) red = m
    if (role === 'nir' && !nir) nir = m
  }
  return { red, nir }
}

export async function loadImageFile(file: File): Promise<LoadedImage> {
  const buffer = await file.arrayBuffer()
  return loadImageBuffer(buffer, file.name, file.size)
}

export async function loadImageBuffer(
  buffer: ArrayBuffer,
  filename: string,
  originalBytes: number,
): Promise<LoadedImage> {
  const name = filename.toLowerCase()
  if (name.endsWith('.tif') || name.endsWith('.tiff') || name.endsWith('.geotiff')) {
    return loadGeoTiff(buffer, filename, originalBytes)
  }
  return loadRasterViaCanvas(buffer, filename, originalBytes)
}

async function loadGeoTiff(
  buffer: ArrayBuffer,
  filename: string,
  originalBytes: number,
): Promise<LoadedImage> {
  const tiff = await fromArrayBuffer(buffer)
  const image = await tiff.getImage()
  const width = image.getWidth()
  const height = image.getHeight()
  const raster = await image.readRasters({ interleave: false })
  const count = Array.isArray(raster) ? raster.length : 1
  const bandOrder = normalizeBandNames(count)
  const bands: BandMap = {}
  for (let i = 0; i < count; i++) {
    const src = (Array.isArray(raster) ? raster[i] : raster) as ArrayLike<number>
    const arr = new Float64Array(width * height)
    for (let p = 0; p < arr.length; p++) arr[p] = Number(src[p])
    bands[bandOrder[i]] = arr
  }
  if (bands.b4 && !bands.red) bands.red = bands.b4
  if (bands.b8 && !bands.nir) bands.nir = bands.b8
  const order = Object.keys(bands)
  return {
    bands,
    bandOrder: order,
    size: { width, height },
    sourceType: 'geotiff',
    originalBytes,
    previewRgba: toPreviewRgba(bands, order, width, height),
    filename,
  }
}

async function loadRasterViaCanvas(
  buffer: ArrayBuffer,
  filename: string,
  originalBytes: number,
): Promise<LoadedImage> {
  const blob = new Blob([buffer])
  const bitmap = await createImageBitmap(blob)
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
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Could not create canvas context')
  const pixels = new Uint8ClampedArray(rgba)
  ctx.putImageData(new ImageData(pixels, width, height), 0, 0)
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
