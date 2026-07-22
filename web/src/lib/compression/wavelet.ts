import type { BandMap } from '../types'

function haarForward(row: Float64Array): Float64Array {
  const n = row.length
  const out = new Float64Array(n)
  const half = Math.floor(n / 2)
  for (let i = 0; i < half; i++) {
    const a = row[i * 2] ?? row[i * 2 - 1] ?? 0
    const b = row[i * 2 + 1] ?? a
    out[i] = (a + b) / 2
    out[half + i] = (a - b) / 2
  }
  if (n % 2 === 1) out[half] = row[n - 1]
  return out
}

function haarInverse(row: Float64Array): Float64Array {
  const n = row.length
  const out = new Float64Array(n)
  const half = Math.floor(n / 2)
  for (let i = 0; i < half; i++) {
    const approx = row[i]
    const detail = row[half + i] ?? 0
    out[i * 2] = approx + detail
    if (i * 2 + 1 < n) out[i * 2 + 1] = approx - detail
  }
  if (n % 2 === 1) out[n - 1] = row[half] ?? out[n - 2]
  return out
}

function transform2D(band: Float64Array, width: number, height: number, inverse = false): Float64Array {
  const temp = new Float64Array(band.length)
  const out = new Float64Array(band.length)
  const op = inverse ? haarInverse : haarForward

  for (let y = 0; y < height; y++) {
    const row = band.slice(y * width, y * width + width)
    const transformed = op(row)
    temp.set(transformed, y * width)
  }

  for (let x = 0; x < width; x++) {
    const col = new Float64Array(height)
    for (let y = 0; y < height; y++) col[y] = temp[y * width + x]
    const transformed = op(col)
    for (let y = 0; y < height; y++) out[y * width + x] = transformed[y]
  }
  return out
}

function compressBandWavelet(
  band: Float64Array,
  width: number,
  height: number,
  keepFraction: number,
  levels: number,
): { reconstructed: Float64Array; retained: number } {
  let coeffs = new Float64Array(band)
  for (let level = 0; level < levels; level++) {
    coeffs = new Float64Array(transform2D(coeffs, width, height, false))
  }

  const magnitudes = new Float64Array(coeffs.length)
  for (let i = 0; i < coeffs.length; i++) magnitudes[i] = Math.abs(coeffs[i])
  const sorted = new Float64Array(magnitudes).sort()
  const keepCount = Math.max(1, Math.ceil(sorted.length * keepFraction))
  const threshold = sorted[sorted.length - keepCount] ?? 0
  let retained = 0
  const sparse = new Float64Array(coeffs.length)
  for (let i = 0; i < coeffs.length; i++) {
    if (Math.abs(coeffs[i]) >= threshold) {
      sparse[i] = coeffs[i]
      retained++
    }
  }

  let reconstructed = sparse
  for (let level = 0; level < levels; level++) {
    reconstructed = new Float64Array(transform2D(reconstructed, width, height, true))
  }

  let bmin = Infinity
  let bmax = -Infinity
  for (const v of band) {
    if (v < bmin) bmin = v
    if (v > bmax) bmax = v
  }
  for (let i = 0; i < reconstructed.length; i++) {
    reconstructed[i] = Math.min(bmax, Math.max(bmin, reconstructed[i]))
  }
  return { reconstructed, retained }
}

export function runWaveletCompression(
  bands: BandMap,
  bandOrder: string[],
  width: number,
  height: number,
  options: { keepFraction: number; levels: number },
): { bands: BandMap; metadata: Record<string, unknown>; compressedBytesEstimate: number } {
  const out: BandMap = {}
  let retainedTotal = 0
  for (const name of bandOrder) {
    const result = compressBandWavelet(
      bands[name],
      width,
      height,
      options.keepFraction,
      options.levels,
    )
    out[name] = result.reconstructed
    retainedTotal += result.retained
  }
  return {
    bands: out,
    metadata: {
      wavelet: 'haar',
      keepFraction: options.keepFraction,
      levels: options.levels,
    },
    compressedBytesEstimate: retainedTotal * 8,
  }
}
