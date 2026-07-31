import type { BandMap } from './types'

/** Nearest-neighbor resize used before compression to keep browser runs interactive. */
export function downsampleBands(
  bands: BandMap,
  bandOrder: string[],
  width: number,
  height: number,
  maxDim: number,
): { bands: BandMap; width: number; height: number; scale: number } {
  const longest = Math.max(width, height)
  // maxDim <= 0 means "native / no limit".
  if (maxDim <= 0 || longest <= maxDim) {
    return { bands, width, height, scale: 1 }
  }
  const scale = maxDim / longest
  const newW = Math.max(1, Math.round(width * scale))
  const newH = Math.max(1, Math.round(height * scale))
  const out: BandMap = {}
  for (const name of bandOrder) {
    const src = bands[name]
    const dst = new Float64Array(newW * newH)
    for (let y = 0; y < newH; y++) {
      const sy = Math.min(height - 1, Math.floor((y + 0.5) / scale))
      for (let x = 0; x < newW; x++) {
        const sx = Math.min(width - 1, Math.floor((x + 0.5) / scale))
        dst[y * newW + x] = src[sy * width + sx]
      }
    }
    out[name] = dst
  }
  return { bands: out, width: newW, height: newH, scale }
}

/**
 * Bilinear upsample so reconstructed bands match the native raster size.
 * Used when compression ran on a downsampled working copy.
 */
export function upsampleBands(
  bands: BandMap,
  bandOrder: string[],
  width: number,
  height: number,
  targetWidth: number,
  targetHeight: number,
): BandMap {
  if (width === targetWidth && height === targetHeight) {
    const out: BandMap = {}
    for (const name of bandOrder) {
      if (bands[name]) out[name] = new Float64Array(bands[name])
    }
    return out
  }
  const out: BandMap = {}
  const xScale = width / targetWidth
  const yScale = height / targetHeight
  for (const name of bandOrder) {
    const src = bands[name]
    if (!src) continue
    const dst = new Float64Array(targetWidth * targetHeight)
    for (let y = 0; y < targetHeight; y++) {
      const fy = (y + 0.5) * yScale - 0.5
      const y0 = Math.max(0, Math.min(height - 1, Math.floor(fy)))
      const y1 = Math.max(0, Math.min(height - 1, y0 + 1))
      const ty = Math.max(0, Math.min(1, fy - y0))
      for (let x = 0; x < targetWidth; x++) {
        const fx = (x + 0.5) * xScale - 0.5
        const x0 = Math.max(0, Math.min(width - 1, Math.floor(fx)))
        const x1 = Math.max(0, Math.min(width - 1, x0 + 1))
        const tx = Math.max(0, Math.min(1, fx - x0))
        const v00 = src[y0 * width + x0]
        const v10 = src[y0 * width + x1]
        const v01 = src[y1 * width + x0]
        const v11 = src[y1 * width + x1]
        dst[y * targetWidth + x] =
          v00 * (1 - tx) * (1 - ty) +
          v10 * tx * (1 - ty) +
          v01 * (1 - tx) * ty +
          v11 * tx * ty
      }
    }
    out[name] = dst
  }
  return out
}

export function cloneBandMap(bands: BandMap, bandOrder: string[]): BandMap {
  const out: BandMap = {}
  for (const name of bandOrder) {
    if (bands[name]) out[name] = new Float64Array(bands[name])
  }
  return out
}
