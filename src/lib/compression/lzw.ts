import type { BandMap } from '../types'

/** Cap dictionary growth (12-bit codes), matching the Python LZW module. */
const MAX_DICT_SIZE = 4096

function pairKey(prefix: number, byte: number): number {
  // prefix < MAX_DICT_SIZE, byte < 256
  return prefix * 256 + byte
}

function lzwCompressBytes(data: Uint8Array): number[] {
  if (!data.length) return []
  const dictionary = new Map<number, number>()
  let dictSize = 256
  let w = data[0]
  const out: number[] = []
  for (let i = 1; i < data.length; i++) {
    const c = data[i]
    const key = pairKey(w, c)
    const existing = dictionary.get(key)
    if (existing !== undefined) {
      w = existing
      continue
    }
    out.push(w)
    if (dictSize < MAX_DICT_SIZE) {
      dictionary.set(key, dictSize)
      dictSize += 1
    }
    w = c
  }
  out.push(w)
  return out
}

function lzwDecompressCodes(codes: number[]): Uint8Array {
  if (!codes.length) return new Uint8Array()
  const dictionary: number[][] = []
  for (let i = 0; i < 256; i++) dictionary[i] = [i]
  let dictSize = 256
  let w = dictionary[codes[0]].slice()
  const out: number[] = w.slice()
  for (let i = 1; i < codes.length; i++) {
    const code = codes[i]
    let entry: number[]
    if (code < dictSize && dictionary[code]) {
      entry = dictionary[code]
    } else if (code === dictSize) {
      entry = w.concat(w[0])
    } else {
      throw new Error(`Invalid LZW code: ${code}`)
    }
    for (let j = 0; j < entry.length; j++) out.push(entry[j])
    if (dictSize < MAX_DICT_SIZE) {
      dictionary[dictSize] = w.concat(entry[0])
      dictSize += 1
    }
    w = entry
  }
  return Uint8Array.from(out)
}

function quantizeUint8(band: Float64Array): {
  u8: Uint8Array
  lo: number
  hi: number
} {
  let lo = Infinity
  let hi = -Infinity
  for (let i = 0; i < band.length; i++) {
    const v = band[i]
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  const scale = hi - lo
  const u8 = new Uint8Array(band.length)
  if (scale <= 0) return { u8, lo, hi }
  for (let i = 0; i < band.length; i++) {
    u8[i] = Math.max(0, Math.min(255, Math.round(((band[i] - lo) / scale) * 255)))
  }
  return { u8, lo, hi }
}

function dequantizeUint8(u8: Uint8Array, lo: number, hi: number): Float64Array {
  const out = new Float64Array(u8.length)
  const scale = hi - lo
  if (scale <= 0) {
    out.fill(lo)
    return out
  }
  for (let i = 0; i < u8.length; i++) out[i] = (u8[i] / 255) * scale + lo
  return out
}

function compressBandLzw(band: Float64Array): {
  reconstructed: Float64Array
  compressedBytes: number
} {
  const { u8, lo, hi } = quantizeUint8(band)
  const codes = lzwCompressBytes(u8)
  const compressedBytes = codes.length * 2
  const decoded = lzwDecompressCodes(codes)
  if (decoded.length !== u8.length) {
    throw new Error(
      `LZW round-trip size mismatch: expected ${u8.length}, got ${decoded.length}`,
    )
  }
  return {
    reconstructed: dequantizeUint8(decoded, lo, hi),
    compressedBytes,
  }
}

export function runLzwCompression(
  bands: BandMap,
  bandOrder: string[],
): {
  bands: BandMap
  metadata: Record<string, unknown>
  compressedBytesEstimate: number
} {
  const out: BandMap = {}
  let compressedTotal = 0
  for (const name of bandOrder) {
    const result = compressBandLzw(bands[name])
    out[name] = result.reconstructed
    compressedTotal += result.compressedBytes
  }
  return {
    bands: out,
    metadata: {
      algorithm: 'LZW',
      quantization: 'uint8-minmax',
      maxDictSize: MAX_DICT_SIZE,
      reference: 'ashmeet13/LZW-Image-Compression (adapted for float bands)',
    },
    compressedBytesEstimate: compressedTotal,
  }
}
