import { writeArrayBuffer } from 'geotiff'
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
  }, 2000)
}

function downloadArrayBuffer(buffer: ArrayBuffer, filename: string, mime: string) {
  downloadBlob(new Blob([buffer], { type: mime }), filename)
}

/**
 * geotiff.writeArrayBuffer expects either:
 * - a flat TypedArray / number[] of pixel-interleaved samples, with width+height in metadata, or
 * - a nested [band][row][column] number[][][].
 * Passing [Float32Array, …] planar arrays fails (width becomes undefined).
 */
function interleaveBands(
  bandArrays: Array<Float32Array | Uint8Array | Float64Array>,
  width: number,
  height: number,
  asFloat: boolean,
): Float32Array | Uint8Array {
  const n = width * height
  const bands = bandArrays.length
  if (!bands) throw new Error('No bands available to write as GeoTIFF')
  for (const band of bandArrays) {
    if (band.length < n) {
      throw new Error(`Band length ${band.length} is shorter than ${width}×${height}`)
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
  const metadata = {
    width,
    height,
    BitsPerSample: samples.map(() => 32),
    SampleFormat: samples.map(() => 3), // IEEE floating point
    SamplesPerPixel: samples.length,
    PhotometricInterpretation: samples.length >= 3 ? 2 : 1,
    PlanarConfiguration: 1, // chunky / pixel-interleaved
  }

  const written = writeArrayBuffer(interleaved, metadata)
  return written instanceof Promise ? await written : written
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
  const width = img.naturalWidth || img.width
  const height = img.naturalHeight || img.height
  if (!width || !height) throw new Error('Preview image has no dimensions')

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
  const written = writeArrayBuffer(interleaved, {
    width,
    height,
    BitsPerSample: [8, 8, 8],
    SamplesPerPixel: 3,
    PhotometricInterpretation: 2,
    PlanarConfiguration: 1,
  })
  const buffer = written instanceof Promise ? await written : written
  downloadArrayBuffer(buffer, filename, 'image/tiff')
}

/**
 * Fallback PNG download when GeoTIFF encoding is unavailable.
 * Prefer GeoTIFF helpers for scientific rasters.
 */
export function downloadPngDataUrl(dataUrl: string, filename: string) {
  const a = document.createElement('a')
  a.href = dataUrl
  a.download = filename.endsWith('.png') ? filename : `${filename.replace(/\.tif+$/i, '')}.png`
  a.rel = 'noopener'
  a.style.display = 'none'
  document.body.appendChild(a)
  a.click()
  window.setTimeout(() => a.remove(), 500)
}
