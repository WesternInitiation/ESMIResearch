import type { BandMap } from '../types'

/**
 * Browser wavelet compression — fixed multilevel Haar with LL preserved.
 *
 * The previous implementation re-transformed the *entire* image at every level
 * (incorrect). True multilevel DWT only recurses into the LL quadrant.
 * Approximation coefficients are always kept; ``keepFraction`` then budgets the
 * largest detail coefficients (same policy as the Python/db4 path).
 *
 * Daubechies-4 lives on Cloud Run / PyWavelets; Haar is the portable browser basis.
 */

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
  // Keep the unpaired sample at the end — never overwrite the first detail at `half`.
  if (n % 2 === 1) out[n - 1] = row[n - 1]
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
  if (n % 2 === 1) out[n - 1] = row[n - 1]
  return out
}

/** In-place 2D Haar on the top-left region [0, regionH) × [0, regionW). */
function haar2DRegion(
  data: Float64Array,
  stride: number,
  regionW: number,
  regionH: number,
  inverse: boolean,
): void {
  const op = inverse ? haarInverse : haarForward

  // Rows
  for (let y = 0; y < regionH; y++) {
    const row = new Float64Array(regionW)
    const base = y * stride
    for (let x = 0; x < regionW; x++) row[x] = data[base + x]
    const transformed = op(row)
    for (let x = 0; x < regionW; x++) data[base + x] = transformed[x]
  }

  // Columns
  for (let x = 0; x < regionW; x++) {
    const col = new Float64Array(regionH)
    for (let y = 0; y < regionH; y++) col[y] = data[y * stride + x]
    const transformed = op(col)
    for (let y = 0; y < regionH; y++) data[y * stride + x] = transformed[y]
  }
}

function maxSafeLevels(width: number, height: number): number {
  let levels = 0
  let w = width
  let h = height
  while (w >= 2 && h >= 2) {
    levels += 1
    w = Math.floor(w / 2)
    h = Math.floor(h / 2)
  }
  return Math.max(1, levels)
}

function compressBandWavelet(
  band: Float64Array,
  width: number,
  height: number,
  keepFraction: number,
  levels: number,
): { reconstructed: Float64Array; retained: number } {
  const safeLevels = Math.max(1, Math.min(levels, maxSafeLevels(width, height)))
  const coeffs = new Float64Array(band)

  // Forward multilevel DWT (LL-only recursion).
  let regionW = width
  let regionH = height
  for (let level = 0; level < safeLevels; level++) {
    haar2DRegion(coeffs, width, regionW, regionH, false)
    regionW = Math.floor(regionW / 2)
    regionH = Math.floor(regionH / 2)
    if (regionW < 1 || regionH < 1) break
  }

  const llW = Math.max(1, Math.floor(width / 2 ** safeLevels))
  const llH = Math.max(1, Math.floor(height / 2 ** safeLevels))
  const approxCount = llW * llH
  const keepCount = Math.min(
    coeffs.length,
    Math.max(approxCount, Math.ceil(coeffs.length * Math.min(1, Math.max(0.001, keepFraction)))),
  )

  const magnitudes = new Float64Array(coeffs.length)
  for (let i = 0; i < coeffs.length; i++) magnitudes[i] = Math.abs(coeffs[i])

  // Rank details; force LL above every detail so it is always retained.
  const ranking = new Float64Array(magnitudes)
  for (let y = 0; y < llH; y++) {
    for (let x = 0; x < llW; x++) {
      ranking[y * width + x] = Number.POSITIVE_INFINITY
    }
  }
  const sorted = Float64Array.from(ranking).sort()
  const threshold = sorted[sorted.length - keepCount] ?? 0

  const sparse = new Float64Array(coeffs.length)
  let retained = 0
  for (let i = 0; i < coeffs.length; i++) {
    const y = Math.floor(i / width)
    const x = i % width
    const inLL = y < llH && x < llW
    if (inLL || Math.abs(coeffs[i]) >= threshold) {
      sparse[i] = coeffs[i]
      retained++
    }
  }

  // Inverse multilevel DWT.
  for (let level = safeLevels - 1; level >= 0; level--) {
    const w = Math.max(1, Math.floor(width / 2 ** level))
    const h = Math.max(1, Math.floor(height / 2 ** level))
    haar2DRegion(sparse, width, w, h, true)
  }

  let bmin = Infinity
  let bmax = -Infinity
  for (let i = 0; i < band.length; i++) {
    const v = band[i]
    if (v < bmin) bmin = v
    if (v > bmax) bmax = v
  }
  for (let i = 0; i < sparse.length; i++) {
    sparse[i] = Math.min(bmax, Math.max(bmin, sparse[i]))
  }
  return { reconstructed: sparse, retained }
}

export const BROWSER_WAVELET = 'haar'
export const WAVELET_FAMILIES = ['haar', 'db2', 'db4', 'sym2', 'sym4'] as const
export type WaveletFamily = (typeof WAVELET_FAMILIES)[number]

export function runWaveletCompression(
  bands: BandMap,
  bandOrder: string[],
  width: number,
  height: number,
  options: { keepFraction: number; levels: number; wavelet?: string },
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
      wavelet: BROWSER_WAVELET,
      requestedWavelet: options.wavelet || BROWSER_WAVELET,
      keepFraction: options.keepFraction,
      levels: options.levels,
      preserveApproximation: true,
      backend: 'haar-multilevel-ll',
      note:
        options.wavelet && options.wavelet !== 'haar'
          ? `Browser engine uses Haar; select Cloud Run for ${options.wavelet}`
          : undefined,
    },
    compressedBytesEstimate: retainedTotal * 8,
  }
}
