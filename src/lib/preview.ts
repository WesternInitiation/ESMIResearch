import type { BandMap } from './types'
import { toPreviewRgba } from './image'

/** Encode an RGBA buffer as a PNG data URL. */
export function rgbaToPngDataUrl(
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

/** Encode an RGBA buffer as a JPEG data URL (compressed artifact preview). */
export function rgbaToJpegDataUrl(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  quality: number,
): string {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Could not create canvas context')
  const pixels = new Uint8ClampedArray(rgba)
  ctx.putImageData(new ImageData(pixels, width, height), 0, 0)
  const q = Math.min(0.95, Math.max(0.05, quality))
  return canvas.toDataURL('image/jpeg', q)
}

/**
 * Absolute residual map |original − decompressed|, stretched for visibility.
 * Used as an optional compression-impact preview.
 */
export function residualPreviewRgba(
  original: BandMap,
  decompressed: BandMap,
  bandOrder: string[],
  width: number,
  height: number,
): Uint8ClampedArray {
  const n = width * height
  const err = new Float64Array(n)
  let maxErr = 1e-12
  const names =
    original.red && decompressed.red && original.green && decompressed.green
      ? ['red', 'green', 'blue'].filter((b) => original[b] && decompressed[b])
      : bandOrder.filter((b) => original[b] && decompressed[b])

  for (let i = 0; i < n; i++) {
    let acc = 0
    let count = 0
    for (const name of names) {
      acc += Math.abs(decompressed[name][i] - original[name][i])
      count++
    }
    const v = count ? acc / count : 0
    err[i] = v
    if (v > maxErr) maxErr = v
  }

  const rgba = new Uint8ClampedArray(n * 4)
  for (let i = 0; i < n; i++) {
    const t = Math.min(1, err[i] / maxErr)
    // Warm residual ramp (dark → amber → white)
    const o = i * 4
    rgba[o] = Math.round(40 + t * 215)
    rgba[o + 1] = Math.round(20 + t * 140)
    rgba[o + 2] = Math.round(10 + t * 40)
    rgba[o + 3] = 255
  }
  return rgba
}

export function bandsToDecompressedPreview(
  bands: BandMap,
  bandOrder: string[],
  width: number,
  height: number,
): string {
  return rgbaToPngDataUrl(toPreviewRgba(bands, bandOrder, width, height), width, height)
}

export function bandsToCompressedArtifactPreview(
  bands: BandMap,
  bandOrder: string[],
  width: number,
  height: number,
  jpegQuality: number,
): string {
  return rgbaToJpegDataUrl(
    toPreviewRgba(bands, bandOrder, width, height),
    width,
    height,
    jpegQuality,
  )
}

/** Re-encode an existing image data URL as JPEG for a compressed-artifact look. */
export async function dataUrlToJpegDataUrl(
  dataUrl: string,
  quality: number,
): Promise<string> {
  const image = await loadHtmlImage(dataUrl)
  const canvas = document.createElement('canvas')
  canvas.width = image.naturalWidth || image.width
  canvas.height = image.naturalHeight || image.height
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Could not create canvas context')
  ctx.drawImage(image, 0, 0)
  return canvas.toDataURL('image/jpeg', Math.min(0.95, Math.max(0.05, quality)))
}

function loadHtmlImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('Failed to decode preview image'))
    img.src = src
  })
}

export function downloadDataUrl(dataUrl: string, filename: string) {
  const a = document.createElement('a')
  a.href = dataUrl
  a.download = filename
  a.rel = 'noopener'
  document.body.appendChild(a)
  a.click()
  a.remove()
}
