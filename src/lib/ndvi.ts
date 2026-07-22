export type NdviMetrics = {
  rmse: number
  mae: number
  correlation: number
  ssim: number
  bias: number
}

export function computeNdvi(red: Float64Array, nir: Float64Array): Float64Array {
  const out = new Float64Array(red.length)
  for (let i = 0; i < red.length; i++) {
    const denom = nir[i] + red[i]
    out[i] = Math.abs(denom) > 1e-8 ? (nir[i] - red[i]) / denom : Number.NaN
    if (!Number.isNaN(out[i])) out[i] = Math.max(-1, Math.min(1, out[i]))
  }
  return out
}

export function compareNdvi(reference: Float64Array, candidate: Float64Array): NdviMetrics {
  const ref: number[] = []
  const cand: number[] = []
  for (let i = 0; i < reference.length; i++) {
    if (Number.isFinite(reference[i]) && Number.isFinite(candidate[i])) {
      ref.push(reference[i])
      cand.push(candidate[i])
    }
  }
  if (!ref.length) {
    return { rmse: 0, mae: 0, correlation: 0, ssim: 0, bias: 0 }
  }
  let sse = 0
  let sae = 0
  let bias = 0
  let meanR = 0
  let meanC = 0
  for (let i = 0; i < ref.length; i++) {
    const d = cand[i] - ref[i]
    sse += d * d
    sae += Math.abs(d)
    bias += d
    meanR += ref[i]
    meanC += cand[i]
  }
  meanR /= ref.length
  meanC /= cand.length
  let varR = 0
  let varC = 0
  let cov = 0
  for (let i = 0; i < ref.length; i++) {
    const dr = ref[i] - meanR
    const dc = cand[i] - meanC
    varR += dr * dr
    varC += dc * dc
    cov += dr * dc
  }
  const correlation =
    varR > 0 && varC > 0 ? cov / Math.sqrt(varR * varC) : 0
  const dataRange = 2
  const c1 = (0.01 * dataRange) ** 2
  const c2 = (0.03 * dataRange) ** 2
  const ssim =
    ((2 * meanR * meanC + c1) * (2 * (cov / (ref.length - 1)) + c2)) /
    ((meanR ** 2 + meanC ** 2 + c1) * (varR / (ref.length - 1) + varC / (ref.length - 1) + c2))
  return {
    rmse: Math.sqrt(sse / ref.length),
    mae: sae / ref.length,
    correlation,
    ssim,
    bias: bias / ref.length,
  }
}
