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

export function cloneBandMap(bands: BandMap, bandOrder: string[]): BandMap {
  const out: BandMap = {}
  for (const name of bandOrder) out[name] = new Float64Array(bands[name])
  return out
}
