import type { BandMap, ChannelReport } from './types'

export function bandStats(original: Float64Array, reconstructed: Float64Array): ChannelReport {
  const n = original.length
  let sse = 0
  let sae = 0
  let min = Number.POSITIVE_INFINITY
  let max = Number.NEGATIVE_INFINITY
  for (let i = 0; i < n; i++) {
    const o = original[i]
    const d = reconstructed[i] - o
    sse += d * d
    sae += Math.abs(d)
    if (o < min) min = o
    if (o > max) max = o
  }
  const mse = sse / n
  const rmse = Math.sqrt(mse)
  const mae = sae / n
  const range = Math.max(max - min, 1e-12)
  const psnrDb = mse <= 1e-20 ? 99 : 10 * Math.log10((range * range) / mse)
  const ssim = approximateSsim(original, reconstructed, range)
  return { band: '', rmse, mae, psnrDb, ssim }
}

function approximateSsim(a: Float64Array, b: Float64Array, dataRange: number): number {
  const n = a.length
  let meanA = 0
  let meanB = 0
  for (let i = 0; i < n; i++) {
    meanA += a[i]
    meanB += b[i]
  }
  meanA /= n
  meanB /= n
  let varA = 0
  let varB = 0
  let cov = 0
  for (let i = 0; i < n; i++) {
    const da = a[i] - meanA
    const db = b[i] - meanB
    varA += da * da
    varB += db * db
    cov += da * db
  }
  varA /= n - 1
  varB /= n - 1
  cov /= n - 1
  const c1 = (0.01 * dataRange) ** 2
  const c2 = (0.03 * dataRange) ** 2
  return ((2 * meanA * meanB + c1) * (2 * cov + c2)) /
    ((meanA ** 2 + meanB ** 2 + c1) * (varA + varB + c2))
}

export function reportAllBands(
  original: BandMap,
  reconstructed: BandMap,
  bandOrder: string[],
): ChannelReport[] {
  return bandOrder.map((band) => {
    const stats = bandStats(original[band], reconstructed[band])
    return { ...stats, band }
  })
}

export function estimateByteSize(bands: BandMap): number {
  return Object.values(bands).reduce((sum, arr) => sum + arr.byteLength, 0)
}
