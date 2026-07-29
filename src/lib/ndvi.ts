/** Landsat Collection 2 Level-2 surface reflectance (mirrors NDVI_RR.py / ndvi.py). */
export const LANDSAT_C2_SR_SCALE = 0.0000275
export const LANDSAT_C2_SR_OFFSET = -0.2
export const LANDSAT_C2_FILL_VALUE = 0

export type IndexMetrics = {
  rmse: number
  mae: number
  correlation: number
  ssim: number
  bias: number
}

/** @deprecated Use IndexMetrics — kept as alias for existing imports. */
export type NdviMetrics = IndexMetrics

export type IndexOptions = {
  eps?: number
  clip?: boolean
  /** Explicit fill/nodata mask (true = invalid). */
  nodataMask?: Uint8Array | boolean[]
  /**
   * Apply Landsat C2 DN→SR scale/offset and DN=0 fill masking.
   * - false (default): raw bands, matching prior browser behaviour
   * - true: always convert
   * - undefined / 'auto': convert when values look like Landsat C2 DNs
   */
  landsatC2Sr?: boolean | 'auto'
}

function landsatC2FillMask(...bands: Float64Array[]): Uint8Array {
  const mask = new Uint8Array(bands[0].length)
  for (let i = 0; i < mask.length; i++) {
    let fill = false
    for (const band of bands) {
      if (band[i] === LANDSAT_C2_FILL_VALUE) {
        fill = true
        break
      }
    }
    mask[i] = fill ? 1 : 0
  }
  return mask
}

export function toLandsatC2Sr(
  band: Float64Array,
  scale = LANDSAT_C2_SR_SCALE,
  offset = LANDSAT_C2_SR_OFFSET,
): Float64Array {
  const out = new Float64Array(band.length)
  for (let i = 0; i < band.length; i++) out[i] = band[i] * scale + offset
  return out
}

export function looksLikeLandsatC2Dn(band: Float64Array): boolean {
  let min = Number.POSITIVE_INFINITY
  let max = Number.NEGATIVE_INFINITY
  let count = 0
  const step = Math.max(1, Math.floor(band.length / 100_000))
  for (let i = 0; i < band.length; i += step) {
    const v = band[i]
    if (!Number.isFinite(v)) continue
    count++
    if (v < min) min = v
    if (v > max) max = v
  }
  if (!count) return false
  if (max <= 2 && min >= -1) return false
  return max >= 100
}

function mergeNodataMasks(
  fill: Uint8Array | null,
  nodataMask?: Uint8Array | boolean[],
): Uint8Array | null {
  if (!fill && !nodataMask) return null
  const length = fill?.length ?? nodataMask!.length
  const out = new Uint8Array(length)
  for (let i = 0; i < length; i++) {
    const fromFill = fill ? fill[i] === 1 : false
    const fromUser = nodataMask ? Boolean(nodataMask[i]) : false
    out[i] = fromFill || fromUser ? 1 : 0
  }
  return out
}

function prepareBands(
  bands: Float64Array[],
  landsatC2Sr: boolean | 'auto' | undefined,
  nodataMask?: Uint8Array | boolean[],
): { bands: Float64Array[]; nodata: Uint8Array | null } {
  // Default / false: keep historical browser behaviour (no Landsat fill/SR).
  if (landsatC2Sr === false || landsatC2Sr === undefined) {
    return {
      bands,
      nodata: nodataMask ? mergeNodataMasks(null, nodataMask) : null,
    }
  }

  const shouldConvert =
    landsatC2Sr === true ||
    (landsatC2Sr === 'auto' && looksLikeLandsatC2Dn(bands[0]))

  if (!shouldConvert) {
    return {
      bands,
      nodata: nodataMask ? mergeNodataMasks(null, nodataMask) : null,
    }
  }

  const fill = landsatC2FillMask(...bands)
  const converted = bands.map((b) => toLandsatC2Sr(b))
  return { bands: converted, nodata: mergeNodataMasks(fill, nodataMask) }
}

function normalizedDifference(
  a: Float64Array,
  b: Float64Array,
  options: IndexOptions = {},
): Float64Array {
  const eps = options.eps ?? 1e-10
  const clip = options.clip !== false
  const prepared = prepareBands(
    [a, b],
    options.landsatC2Sr === undefined ? false : options.landsatC2Sr,
    options.nodataMask,
  )
  const [first, second] = prepared.bands
  const out = new Float64Array(first.length)
  for (let i = 0; i < first.length; i++) {
    if (prepared.nodata && prepared.nodata[i]) {
      out[i] = Number.NaN
      continue
    }
    const denom = first[i] + second[i]
    let value =
      Math.abs(denom) > eps ? (first[i] - second[i]) / denom : Number.NaN
    if (!Number.isNaN(value) && clip) value = Math.max(-1, Math.min(1, value))
    out[i] = value
  }
  return out
}

/** NDVI = (NIR - Red) / (NIR + Red) */
export function computeNdvi(
  red: Float64Array,
  nir: Float64Array,
  options: IndexOptions = {},
): Float64Array {
  return normalizedDifference(nir, red, options)
}

/**
 * NDWI variants:
 * - mcfeeters: (Green - NIR) / (Green + NIR)
 * - mndwi: (Green - SWIR) / (Green + SWIR)
 */
export function computeNdwi(
  green: Float64Array,
  second: Float64Array,
  options: IndexOptions = {},
): Float64Array {
  return normalizedDifference(green, second, options)
}

export function indexStats(index: Float64Array): {
  validPixelCount: number
  validPixelFraction: number
  minimum: number
  maximum: number
  mean: number
  pixelsAtNegativeOne: number
  pixelsAtPositiveOne: number
} {
  let valid = 0
  let min = Number.POSITIVE_INFINITY
  let max = Number.NEGATIVE_INFINITY
  let sum = 0
  let atNeg = 0
  let atPos = 0
  for (let i = 0; i < index.length; i++) {
    const v = index[i]
    if (!Number.isFinite(v)) continue
    valid++
    sum += v
    if (v < min) min = v
    if (v > max) max = v
    if (v === -1) atNeg++
    if (v === 1) atPos++
  }
  if (!valid) {
    return {
      validPixelCount: 0,
      validPixelFraction: 0,
      minimum: Number.NaN,
      maximum: Number.NaN,
      mean: Number.NaN,
      pixelsAtNegativeOne: 0,
      pixelsAtPositiveOne: 0,
    }
  }
  return {
    validPixelCount: valid,
    validPixelFraction: valid / index.length,
    minimum: min,
    maximum: max,
    mean: sum / valid,
    pixelsAtNegativeOne: atNeg,
    pixelsAtPositiveOne: atPos,
  }
}

export function compareIndexMaps(
  reference: Float64Array,
  candidate: Float64Array,
): IndexMetrics {
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

/** @deprecated Prefer compareIndexMaps */
export function compareNdvi(
  reference: Float64Array,
  candidate: Float64Array,
): IndexMetrics {
  return compareIndexMaps(reference, candidate)
}
