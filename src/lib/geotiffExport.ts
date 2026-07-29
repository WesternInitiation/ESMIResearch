import { writeArrayBuffer } from 'geotiff'
import type { BandMap } from './types'
import { downloadDataUrl } from './preview'

function downloadArrayBuffer(buffer: ArrayBuffer, filename: string, mime: string) {
  const blob = new Blob([buffer], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.rel = 'noopener'
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
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

  const planar = samples as Float32Array[] & { width: number; height: number }
  planar.width = width
  planar.height = height

  const metadata = {
    width,
    height,
    BitsPerSample: samples.map(() => 32),
    SampleFormat: samples.map(() => 3), // IEEE floating point
    SamplesPerPixel: samples.length,
    PhotometricInterpretation: samples.length === 1 ? 1 : 2,
    PlanarConfiguration: 1,
  }

  const written = writeArrayBuffer(
    planar as unknown as Parameters<typeof writeArrayBuffer>[0],
    metadata,
  )
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

/** Fallback: decode a PNG/JPEG data URL to an 8-bit RGB GeoTIFF. */
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
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
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
  const planar = [r, g, b] as Uint8Array[] & { width: number; height: number }
  planar.width = width
  planar.height = height
  const written = writeArrayBuffer(
    planar as unknown as Parameters<typeof writeArrayBuffer>[0],
    {
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

/** @deprecated Prefer GeoTIFF helpers — kept for non-image text downloads. */
export { downloadDataUrl }
