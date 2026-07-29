import type { BandMap } from './types'

/**
 * Lossy JPEG round-trip for selected bands — approximates the lab's
 * "Compressed" preview path so NDVI/NDWI can run against after-compression pixels.
 */
export async function jpegRoundtripBands(
  bands: BandMap,
  bandNames: string[],
  width: number,
  height: number,
  quality: number,
): Promise<BandMap> {
  const q = Math.min(0.95, Math.max(0.05, quality))
  const out: BandMap = {}
  for (const name of bandNames) {
    const src = bands[name]
    if (!src) throw new Error(`Missing band "${name}" for compressed-index compare`)
    out[name] = await jpegRoundtripBand(src, width, height, q)
  }
  return out
}

async function jpegRoundtripBand(
  src: Float64Array,
  width: number,
  height: number,
  quality: number,
): Promise<Float64Array> {
  let min = Number.POSITIVE_INFINITY
  let max = Number.NEGATIVE_INFINITY
  for (let i = 0; i < src.length; i++) {
    const v = src[i]
    if (!Number.isFinite(v)) continue
    if (v < min) min = v
    if (v > max) max = v
  }
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) {
    min = 0
    max = 1
  }
  const range = max - min || 1

  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Could not create canvas context')
  const rgba = new Uint8ClampedArray(width * height * 4)
  for (let i = 0; i < src.length; i++) {
    const t = Number.isFinite(src[i]) ? Math.round(((src[i] - min) / range) * 255) : 0
    const o = i * 4
    rgba[o] = t
    rgba[o + 1] = t
    rgba[o + 2] = t
    rgba[o + 3] = 255
  }
  ctx.putImageData(new ImageData(rgba, width, height), 0, 0)
  const dataUrl = canvas.toDataURL('image/jpeg', quality)

  const img = await loadHtmlImage(dataUrl)
  const decoded = document.createElement('canvas')
  decoded.width = width
  decoded.height = height
  const dctx = decoded.getContext('2d')
  if (!dctx) throw new Error('Could not create canvas context')
  dctx.drawImage(img, 0, 0, width, height)
  const pixels = dctx.getImageData(0, 0, width, height).data
  const out = new Float64Array(src.length)
  for (let i = 0; i < src.length; i++) {
    out[i] = min + (pixels[i * 4] / 255) * range
  }
  return out
}

function loadHtmlImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('Failed to decode JPEG round-trip band'))
    img.src = src
  })
}
