import { Matrix, SVD } from 'ml-matrix'
import type { BandMap } from '../types'

export type SvdOptions = {
  rank: number
  normalize?: boolean
}

function compressBandSvd(
  band: Float64Array,
  width: number,
  height: number,
  options: SvdOptions,
): { reconstructed: Float64Array; rankUsed: number; energyRetained: number } {
  const matrixData: number[][] = []
  for (let y = 0; y < height; y++) {
    const row: number[] = []
    for (let x = 0; x < width; x++) row.push(band[y * width + x])
    matrixData.push(row)
  }

  let offset = 0
  let scale = 1
  if (options.normalize !== false) {
    let min = Infinity
    let max = -Infinity
    for (const v of band) {
      if (v < min) min = v
      if (v > max) max = v
    }
    offset = min
    scale = max - min || 1
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        matrixData[y][x] = (matrixData[y][x] - offset) / scale
      }
    }
  }

  const matrix = new Matrix(matrixData)
  const svd = new SVD(matrix, { autoTranspose: true })
  const singularValues = svd.diagonal
  const totalEnergy = singularValues.reduce((s, v) => s + v * v, 0)
  const maxRank = Math.min(options.rank, singularValues.length)
  let energy = 0
  for (let i = 0; i < maxRank; i++) energy += singularValues[i] * singularValues[i]
  const energyRetained = totalEnergy > 0 ? energy / totalEnergy : 0

  const U = svd.leftSingularVectors
  const V = svd.rightSingularVectors
  const s = singularValues
  const reconstructedMatrix = Matrix.zeros(height, width)
  for (let k = 0; k < maxRank; k++) {
    const uk = U.getColumn(k)
    const vk = V.getColumn(k)
    const sk = s[k]
    for (let i = 0; i < height; i++) {
      for (let j = 0; j < width; j++) {
        reconstructedMatrix.set(
          i,
          j,
          reconstructedMatrix.get(i, j) + sk * uk[i] * vk[j],
        )
      }
    }
  }

  const out = new Float64Array(width * height)
  let bmin = Infinity
  let bmax = -Infinity
  for (const v of band) {
    if (v < bmin) bmin = v
    if (v > bmax) bmax = v
  }
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let val = reconstructedMatrix.get(y, x)
      if (options.normalize !== false) val = val * scale + offset
      out[y * width + x] = Math.min(bmax, Math.max(bmin, val))
    }
  }
  return { reconstructed: out, rankUsed: maxRank, energyRetained }
}

export function runSvdCompression(
  bands: BandMap,
  bandOrder: string[],
  width: number,
  height: number,
  options: SvdOptions,
): { bands: BandMap; metadata: Record<string, unknown>; compressedBytesEstimate: number } {
  const out: BandMap = {}
  const meta: Record<string, unknown> = {}
  let compressedBytes = 0
  for (const name of bandOrder) {
    const result = compressBandSvd(bands[name], width, height, options)
    out[name] = result.reconstructed
    meta[name] = {
      rankUsed: result.rankUsed,
      energyRetained: result.energyRetained,
    }
    compressedBytes += (height * result.rankUsed + result.rankUsed + result.rankUsed * width) * 8
  }
  return { bands: out, metadata: { bands: meta }, compressedBytesEstimate: compressedBytes }
}
