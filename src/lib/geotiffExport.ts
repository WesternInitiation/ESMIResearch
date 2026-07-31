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

export type GeoTiffMetadata = {
  /** Native raster size the geo tags were measured against. */
  nativeWidth: number
  nativeHeight: number
  modelPixelScale?: number[]
  modelTiepoint?: number[]
  modelTransformation?: number[]
  geoKeyDirectory?: number[]
  geoDoubleParams?: number[]
  geoAsciiParams?: string
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
  geo?: GeoTiffMetadata | null
}

function scaleGeoForExport(
  geo: GeoTiffMetadata | null | undefined,
  width: number,
  height: number,
): GeoTiffMetadata | null {
  if (!geo) return null
  const nativeW = Math.max(1, Math.floor(geo.nativeWidth) || width)
  const nativeH = Math.max(1, Math.floor(geo.nativeHeight) || height)
  const sx = nativeW / width
  const sy = nativeH / height
  const out: GeoTiffMetadata = {
    nativeWidth: width,
    nativeHeight: height,
    geoKeyDirectory: geo.geoKeyDirectory ? [...geo.geoKeyDirectory] : undefined,
    geoDoubleParams: geo.geoDoubleParams ? [...geo.geoDoubleParams] : undefined,
    geoAsciiParams: geo.geoAsciiParams,
    modelTiepoint: geo.modelTiepoint ? [...geo.modelTiepoint] : undefined,
    modelTransformation: geo.modelTransformation
      ? [...geo.modelTransformation]
      : undefined,
  }
  if (geo.modelPixelScale && geo.modelPixelScale.length >= 2) {
    out.modelPixelScale = [
      Number(geo.modelPixelScale[0]) * sx,
      Number(geo.modelPixelScale[1]) * sy,
      Number(geo.modelPixelScale[2] ?? 0),
    ]
  }
  // Scale affine transform pixel→map coefficients when present (GDAL ModelTransformation).
  if (out.modelTransformation && out.modelTransformation.length >= 16) {
    const t = out.modelTransformation
    // | a  b  0  x |   a,b scale with pixel size
    // | c  d  0  y |
    t[0] *= sx
    t[1] *= sy
    t[4] *= sx
    t[5] *= sy
  }
  return out
}

function writeDoubles(view: DataView, offset: number, values: number[]) {
  for (let i = 0; i < values.length; i++) {
    view.setFloat64(offset + i * 8, Number(values[i]), true)
  }
}

function writeShorts(view: DataView, offset: number, values: number[]) {
  for (let i = 0; i < values.length; i++) {
    view.setUint16(offset + i * 2, Number(values[i]) & 0xffff, true)
  }
}

/**
 * Minimal uncompressed little-endian TIFF / GeoTIFF writer.
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
  const geo = scaleGeoForExport(options.geo, width, height)

  const needsBitsArray = spp > 1
  const needsFormatArray = spp > 1
  const bitsPerSampleBytes = needsBitsArray ? spp * 2 : 0
  const sampleFormatBytes = needsFormatArray ? spp * 2 : 0

  const geoTags: Array<{
    tag: number
    type: number
    count: number
    bytes: number
    write: (view: DataView, offset: number) => void
  }> = []
  if (geo?.modelPixelScale && geo.modelPixelScale.length >= 3) {
    const values = geo.modelPixelScale.slice(0, 3)
    geoTags.push({
      tag: 33550,
      type: 12,
      count: 3,
      bytes: 24,
      write: (v, o) => writeDoubles(v, o, values),
    })
  }
  if (geo?.modelTiepoint && geo.modelTiepoint.length >= 6) {
    const values = geo.modelTiepoint.slice(
      0,
      Math.floor(geo.modelTiepoint.length / 6) * 6,
    )
    geoTags.push({
      tag: 33922,
      type: 12,
      count: values.length,
      bytes: values.length * 8,
      write: (v, o) => writeDoubles(v, o, values),
    })
  }
  if (geo?.modelTransformation && geo.modelTransformation.length >= 16) {
    const values = geo.modelTransformation.slice(0, 16)
    geoTags.push({
      tag: 34264,
      type: 12,
      count: 16,
      bytes: 128,
      write: (v, o) => writeDoubles(v, o, values),
    })
  }
  if (geo?.geoKeyDirectory && geo.geoKeyDirectory.length >= 4) {
    const values = geo.geoKeyDirectory.map((n) => Number(n) & 0xffff)
    geoTags.push({
      tag: 34735,
      type: 3,
      count: values.length,
      bytes: values.length * 2,
      write: (v, o) => writeShorts(v, o, values),
    })
  }
  if (geo?.geoDoubleParams && geo.geoDoubleParams.length > 0) {
    const values = [...geo.geoDoubleParams]
    geoTags.push({
      tag: 34736,
      type: 12,
      count: values.length,
      bytes: values.length * 8,
      write: (v, o) => writeDoubles(v, o, values),
    })
  }
  if (geo?.geoAsciiParams) {
    const ascii = `${geo.geoAsciiParams}\0`
    const encoded = new TextEncoder().encode(ascii)
    geoTags.push({
      tag: 34737,
      type: 2,
      count: encoded.length,
      bytes: encoded.length,
      write: (v, o) => {
        new Uint8Array(v.buffer).set(encoded, o)
      },
    })
  }

  // Layout (word-aligned sections):
  // [0..8)     header
  // [8..)      optional BitsPerSample / SampleFormat arrays
  //            geo tag payloads
  //            strip bytes
  //            IFD
  const bitsOffset = 8
  const sampleFormatOffset = bitsOffset + bitsPerSampleBytes
  let cursor = sampleFormatOffset + sampleFormatBytes
  const geoOffsets: number[] = []
  for (const g of geoTags) {
    if (cursor & 1) cursor += 1
    geoOffsets.push(cursor)
    cursor += g.bytes
  }
  if (cursor & 1) cursor += 1
  const stripOffset = cursor
  let ifdOffset = stripOffset + strip.byteLength
  if (ifdOffset & 1) ifdOffset += 1

  const entryCount = 11 + geoTags.length
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
  for (let i = 0; i < geoTags.length; i++) {
    geoTags[i].write(view, geoOffsets[i])
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

  // TIFF types: 3=SHORT 4=LONG 12=DOUBLE
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
  for (let i = 0; i < geoTags.length; i++) {
    const g = geoTags[i]
    writeEntry(g.tag, g.type, g.count, geoOffsets[i])
  }

  view.setUint32(p, 0, true) // next IFD = none
  return out
}

/**
 * Write multi-band float32 GeoTIFF (uncompressed) from laboratory band maps.
 * Optionally embeds GeoTIFF tags from the source raster.
 */
export async function bandsToGeoTiffArrayBuffer(
  bands: BandMap,
  bandOrder: string[],
  width: number,
  height: number,
  geo?: GeoTiffMetadata | null,
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
  // Scientific float stacks are MinIsBlack; only true 3-band RGB uses photometric RGB.
  const looksLikeRgb =
    samples.length === 3 &&
    names.length === 3 &&
    names.every((n) => /^(r|red|g|green|b|blue)$/i.test(n))
  return writeClassicTiff({
    width,
    height,
    samplesPerPixel: samples.length,
    bitsPerSample: 32,
    sampleFormat: 3,
    photometric: looksLikeRgb ? 2 : 1,
    data: interleaved,
    geo: geo ?? null,
  })
}

export async function downloadBandsAsGeoTiff(
  bands: BandMap,
  bandOrder: string[],
  width: number,
  height: number,
  filename: string,
  geo?: GeoTiffMetadata | null,
): Promise<void> {
  const buffer = await bandsToGeoTiffArrayBuffer(bands, bandOrder, width, height, geo)
  downloadArrayBuffer(buffer, filename, 'image/tiff')
}

/**
 * Decode a PNG/JPEG data URL and write an 8-bit RGB GeoTIFF.
 * Used when Cloud Run results have no in-browser band arrays.
 * Optional targetWidth/Height upscales the display preview to native size
 * so downloads match the original raster dimensions.
 */
export async function downloadRgbPreviewAsGeoTiff(
  dataUrl: string,
  filename: string,
  targetWidth?: number,
  targetHeight?: number,
): Promise<void> {
  const img = await new Promise<HTMLImageElement>((resolve, reject) => {
    const el = new Image()
    el.onload = () => resolve(el)
    el.onerror = () => reject(new Error('Failed to decode preview for GeoTIFF export'))
    el.src = dataUrl
  })
  const srcW = assertPositiveInt('preview width', img.naturalWidth || img.width)
  const srcH = assertPositiveInt('preview height', img.naturalHeight || img.height)
  const width =
    targetWidth && targetWidth > 0
      ? assertPositiveInt('target width', targetWidth)
      : srcW
  const height =
    targetHeight && targetHeight > 0
      ? assertPositiveInt('target height', targetHeight)
      : srcH

  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  if (!ctx) throw new Error('Could not create canvas context')
  ctx.imageSmoothingEnabled = true
  ctx.drawImage(img, 0, 0, width, height)
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
 * Fallback download for a PNG/JPEG (or other) preview data URL.
 * Uses a Blob so large previews are not limited by data-URL href length, and
 * picks the file extension from the MIME type (compressed previews are JPEG).
 */
export async function downloadPreviewDataUrl(
  dataUrl: string,
  filename: string,
): Promise<void> {
  const res = await fetch(dataUrl)
  const blob = await res.blob()
  const mime = (blob.type || '').toLowerCase()
  const ext = mime.includes('jpeg') || mime.includes('jpg')
    ? 'jpg'
    : mime.includes('png')
      ? 'png'
      : mime.includes('tif')
        ? 'tif'
        : 'bin'
  const base = filename.replace(/\.(png|jpe?g|tiff?|bin)$/i, '')
  downloadBlob(blob, `${base}.${ext}`)
}

/** @deprecated Prefer downloadPreviewDataUrl — kept for callers that expect sync PNG naming. */
export function downloadPngDataUrl(dataUrl: string, filename: string) {
  void downloadPreviewDataUrl(dataUrl, filename)
}
