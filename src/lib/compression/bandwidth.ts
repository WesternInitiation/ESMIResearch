import FFT from 'fft.js'
import type { BandMap } from '../types'

function nextPow2(n: number): number {
  let p = 1
  while (p < n) p <<= 1
  return p
}

function compressBandBandwidth(
  band: Float64Array,
  width: number,
  height: number,
  keepFraction: number,
): { reconstructed: Float64Array; retained: number } {
  const size = nextPow2(Math.max(width, height))
  const fft = new FFT(size * size)
  const input = new Array(size * size * 2).fill(0)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * size + x) * 2
      input[idx] = band[y * width + x]
    }
  }
  const spectrum = fft.createComplexArray()
  fft.transform(spectrum, input)

  // Keep a centered low-frequency block after shifting conceptually via index wrap.
  const keepH = Math.max(1, Math.round((height * keepFraction) / 2))
  const keepW = Math.max(1, Math.round((width * keepFraction) / 2))
  const filtered = new Array(spectrum.length).fill(0)
  let retained = 0
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const cy = y < size / 2 ? y : y - size
      const cx = x < size / 2 ? x : x - size
      if (Math.abs(cy) <= keepH && Math.abs(cx) <= keepW) {
        const idx = (y * size + x) * 2
        filtered[idx] = spectrum[idx]
        filtered[idx + 1] = spectrum[idx + 1]
        retained++
      }
    }
  }

  const inverse = fft.createComplexArray()
  fft.inverseTransform(inverse, filtered)
  const out = new Float64Array(width * height)
  let bmin = Infinity
  let bmax = -Infinity
  for (const v of band) {
    if (v < bmin) bmin = v
    if (v > bmax) bmax = v
  }
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const val = inverse[(y * size + x) * 2]
      out[y * width + x] = Math.min(bmax, Math.max(bmin, val))
    }
  }
  return { reconstructed: out, retained }
}

export function runBandwidthCompression(
  bands: BandMap,
  bandOrder: string[],
  width: number,
  height: number,
  keepFraction: number,
): { bands: BandMap; metadata: Record<string, unknown>; compressedBytesEstimate: number } {
  const out: BandMap = {}
  let retained = 0
  for (const name of bandOrder) {
    const result = compressBandBandwidth(bands[name], width, height, keepFraction)
    out[name] = result.reconstructed
    retained += result.retained
  }
  return {
    bands: out,
    metadata: { keepFraction },
    compressedBytesEstimate: retained * 16,
  }
}
