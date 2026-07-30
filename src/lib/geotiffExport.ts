import type { BandMap } from './types'

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.rel = 'noopener'
  a.style.display = 'none'
  document.body.appendChild(a)
  a.click()
  // Keep the object URL alive briefly so the browser can start the download.
  window.setTimeout(() => {
    a.remove()
    URL.revokeObjectURL(url)
  }, 2500)
}

function downloadArrayBuffer(buffer: ArrayBuffer, filename: string, mime: string) {
  downloadBlob(new Blob([buffer], { type: mime }), filename)
}

function assertPositiveInt(name: string, value: number): number {
  const n = Math.floor(Number(value))
  if (!Number.isFinite(n) || n <= 0) {
    throw new Error(`Invalid ${name} for GeoTIFF export: ${value}`)
  }
  return n
}

/**
 * Pixel-interleave band planes into one contiguous buffer.
 */
function interleaveBands(
  bandArrays: Array<Float32Array | Uint8Array | Float64Array>,
  width: number,
  height: number,
  asFloat: boolean,
): Float32Array | Uint8Array {
  const w = assertPositiveInt('width', width)
  const h = assertPositiveInt('height', height)
  const n = w * h
  const bands = bandArrays.length
  if (!bands) throw new Error('No bands available to write as GeoTIFF')
  for (const band of bandArrays) {
    if (band.length < n) {
      throw new Error(`Band length ${band.length} is shorter than ${w}×${h}`)
    }
  }
  if (asFloat) {
    const out = new Float32Array(n * bands)
    for (let i = 0; i < n; i++) {
      const o = i * bands
      for (let b = 0; b < bands; b++) out[o + b] = Number(bandArrays[b][i])
    }
    return out
  }
  const out = new Uint8Array(n * bands)
  for (let i = 0; i < n; i++) {
    const o = i * bands
    for (let b = 0; b < bands; b++) {
      out[o + b] = Math.max(0, Math.min(255, Math.round(Number(bandArrays[b][i]))))
    }
  }
  return out
}

type TiffWriteOptions = {
  width: number
  height: number
  samplesPerPixel: number
  bitsPerSample: 8 | 32
  /** 1 = unsigned int, 3 = IEEE float */
  sampleFormat: 1 | 3
  /** 1 = min-is-black, 2 = RGB */
  photometric: 1 | 2
  /** Contiguous pixel-interleaved sample bytes / floats */
  data: Uint8Array | Float32Array
}

/**
 * Minimal uncompressed little-endian TIFF writer.
 *
 * We deliberately do NOT use geotiff.js `writeArrayBuffer` — it treats arrays of
 * TypedArrays as `[band][row][col]` and throws "width of type undefined".
 */
export function writeClassicTiff(options: TiffWriteOptions): ArrayBuffer {
  const width = assertPositiveInt('width', options.width)
  const height = assertPositiveInt('height', options.height)
  const spp = assertPositiveInt('samplesPerPixel', options.samplesPerPixel)
  const bps = options.bitsPerSample
  const bytesPerSample = bps / 8
  const expected = width * height * spp * bytesPerSample
  const raw =
    options.data instanceof Uint8Array
      ? options.data
      : new Uint8Array(
          options.data.buffer,
          options.data.byteOffset,
          options.data.byteLength,
        )
  if (raw.byteLength < expected) {
    throw new Error(
      `TIFF payload too small: got ${raw.byteLength} B, need ${expected} B for ${width}×${height}×${spp}`,
    )
  }
  const strip = raw.subarray(0, expected)

  // Layout (word-aligned sections):
  // [0..8)     header
  // [8..)      optional BitsPerSample array (if spp > 1)
  //            optional SampleFormat array (if spp > 1)
  //            strip bytes
  //            IFD
  const needsBitsArray = spp > 1
  const needsFormatArray = spp > 1
  const bitsPerSampleBytes = needsBitsArray ? spp * 2 : 0
  const sampleFormatBytes = needsFormatArray ? spp * 2 : 0
  const bitsOffset = 8
  const sampleFormatOffset = bitsOffset + bitsPerSampleBytes
  const stripOffset = sampleFormatOffset + sampleFormatBytes
  const ifdOffset = stripOffset + strip.byteLength

  const entryCount = 11
  const ifdSize = 2 + entryCount * 12 + 4
  const total = ifdOffset + ifdSize
  const out = new ArrayBuffer(total)
  const view = new DataView(out)
  const u8 = new Uint8Array(out)

  // Header: little-endian classic TIFF
  view.setUint16(0, 0x4949, true) // II
  view.setUint16(2, 42, true)
  view.setUint32(4, ifdOffset, true)

  if (needsBitsArray) {
    for (let i = 0; i < spp; i++) {
      view.setUint16(bitsOffset + i * 2, bps, true)
    }
  }
  if (needsFormatArray) {
    for (let i = 0; i < spp; i++) {
      view.setUint16(sampleFormatOffset + i * 2, options.sampleFormat, true)
    }
  }

  // Strip data — little-endian sample bytes (matches II header)
  u8.set(strip, stripOffset)

  // IFD
  let p = ifdOffset
  view.setUint16(p, entryCount, true)
  p += 2

  const writeEntry = (
    tag: number,
    type: number,
    count: number,
    valueOrOffset: number,
  ) => {
    view.setUint16(p, tag, true)
    view.setUint16(p + 2, type, true)
    view.setUint32(p + 4, count, true)
    view.setUint32(p + 8, valueOrOffset, true)
    p += 12
  }

  // TIFF types: 3=SHORT 4=LONG
  // Values ≤4 bytes must be stored inline in the value field (not as an offset).
  writeEntry(256, 4, 1, width) // ImageWidth
  writeEntry(257, 4, 1, height) // ImageLength
  writeEntry(258, 3, spp, needsBitsArray ? bitsOffset : bps) // BitsPerSample
  writeEntry(259, 3, 1, 1) // Compression = none
  writeEntry(262, 3, 1, options.photometric) // PhotometricInterpretation
  writeEntry(273, 4, 1, stripOffset) // StripOffsets
  writeEntry(277, 3, 1, spp) // SamplesPerPixel
  writeEntry(278, 4, 1, height) // RowsPerStrip
  writeEntry(279, 4, 1, strip.byteLength) // StripByteCounts
  writeEntry(284, 3, 1, 1) // PlanarConfiguration = chunky
  writeEntry(
    339,
    3,
    spp,
    needsFormatArray ? sampleFormatOffset : options.sampleFormat,
  ) // SampleFormat

  view.setUint32(p, 0, true) // next IFD = none
  return out
}

/**
 * Write multi-band float GeoTIFF (uncompressed) from laboratory band maps.
 */
export async function bandsToGeoTiffArrayBuffer(
  bands: BandMap,
  bandOrder: string[],
  width: number,
  height: number,
): Promise<ArrayBuffer> {
  const names = bandOrder.filter((b) => bands[b])
  if (!names.length) throw new Error('No bands available to write as GeoTIFF')

  const samples = names.map((name) => {
    const src = bands[name]
    const f32 = new Float32Array(src.length)
    for (let i = 0; i < src.length; i++) f32[i] = src[i]
    return f32
  })

  const interleaved = interleaveBands(samples, width, height, true)
  return writeClassicTiff({
    width,
    height,
    samplesPerPixel: samples.length,
    bitsPerSample: 32,
    sampleFormat: 3,
    photometric: samples.length >= 3 ? 2 : 1,
    data: interleaved,
  })
}

export async function downloadBandsAsGeoTiff(
  bands: BandMap,
  bandOrder: string[],
  width: number,
  height: number,
  filename: string,
): Promise<void> {
  const buffer = await bandsToGeoTiffArrayBuffer(bands, bandOrder, width, height)
  downloadArrayBuffer(buffer, filename, 'image/tiff')
}

/**
 * Decode a PNG/JPEG data URL and write an 8-bit RGB GeoTIFF.
 * Used when Cloud Run results have no in-browser band arrays.
 */
export async function downloadRgbPreviewAsGeoTiff(
  dataUrl: string,
  filename: string,
): Promise<void> {
  const img = await new Promise<HTMLImageElement>((resolve, reject) => {
    const el = new Image()
    el.onload = () => resolve(el)
    el.onerror = () => reject(new Error('Failed to decode preview for GeoTIFF export'))
    el.src = dataUrl
  })
  const width = assertPositiveInt('preview width', img.naturalWidth || img.width)
  const height = assertPositiveInt('preview height', img.naturalHeight || img.height)

  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  if (!ctx) throw new Error('Could not create canvas context')
  ctx.drawImage(img, 0, 0)
  const rgba = ctx.getImageData(0, 0, width, height).data
  const n = width * height
  const r = new Uint8Array(n)
  const g = new Uint8Array(n)
  const b = new Uint8Array(n)
  for (let i = 0; i < n; i++) {
    const o = i * 4
    r[i] = rgba[o]
    g[i] = rgba[o + 1]
    b[i] = rgba[o + 2]
  }

  const interleaved = interleaveBands([r, g, b], width, height, false)
  const buffer = writeClassicTiff({
    width,
    height,
    samplesPerPixel: 3,
    bitsPerSample: 8,
    sampleFormat: 1,
    photometric: 2,
    data: interleaved,
  })
  downloadArrayBuffer(buffer, filename, 'image/tiff')
}

/**
 * Fallback PNG download when the caller already has a preview data URL.
 */
export function downloadPngDataUrl(dataUrl: string, filename: string) {
  const a = document.createElement('a')
  a.href = dataUrl
  a.download = filename.endsWith('.png')
    ? filename
    : `${filename.replace(/\.tiff?$/i, '')}.png`
  a.rel = 'noopener'
  a.style.display = 'none'
  document.body.appendChild(a)
  a.click()
  window.setTimeout(() => a.remove(), 500)
}
