import FFT from 'fft.js'
import type { BandMap } from '../types'

/** Cap FFT grid so bandwidth runs stay interactive in-browser. */
const MAX_FFT_SIDE = 512

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
  const size = Math.min(MAX_FFT_SIDE, nextPow2(Math.max(width, height)))
  const fft = new FFT(size * size)
  const input = new Array(size * size * 2).fill(0)
  const scaleX = width / size
  const scaleY = height / size
  for (let y = 0; y < size; y++) {
    const sy = Math.min(height - 1, Math.floor(y * scaleY))
    for (let x = 0; x < size; x++) {
      const sx = Math.min(width - 1, Math.floor(x * scaleX))
      input[(y * size + x) * 2] = band[sy * width + sx]
    }
  }
  const spectrum = fft.createComplexArray()
  fft.transform(spectrum, input)

  const keepH = Math.max(1, Math.round((size * keepFraction) / 2))
  const keepW = Math.max(1, Math.round((size * keepFraction) / 2))
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
  for (let i = 0; i < band.length; i++) {
    const v = band[i]
    if (v < bmin) bmin = v
    if (v > bmax) bmax = v
  }
  for (let y = 0; y < height; y++) {
    const fy = Math.min(size - 1, Math.floor((y / height) * size))
    for (let x = 0; x < width; x++) {
      const fx = Math.min(size - 1, Math.floor((x / width) * size))
      const val = inverse[(fy * size + fx) * 2]
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
    metadata: { keepFraction, maxFftSide: MAX_FFT_SIDE },
    compressedBytesEstimate: retained * 16,
  }
}
