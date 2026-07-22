import type { BandMap } from '../types'

/**
 * Browser-friendly JPEG2000 stand-in: per-band canvas JPEG encode/decode.
 * True JPEG2000 isn't widely available client-side; this preserves the
 * research workflow (rate/quality control + encode size metrics) on Vercel.
 */
async function compressBandJpeg(
  band: Float64Array,
  width: number,
  height: number,
  quality: number,
): Promise<{ reconstructed: Float64Array; encodedBytes: number }> {
  let min = Infinity
  let max = -Infinity
  for (const v of band) {
    if (v < min) min = v
    if (v > max) max = v
  }
  const scale = max - min || 1
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Canvas unavailable')
  const image = ctx.createImageData(width, height)
  for (let i = 0; i < band.length; i++) {
    const gray = Math.round(((band[i] - min) / scale) * 255)
    const o = i * 4
    image.data[o] = gray
    image.data[o + 1] = gray
    image.data[o + 2] = gray
    image.data[o + 3] = 255
  }
  ctx.putImageData(image, 0, 0)

  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error('JPEG encode failed'))),
      'image/jpeg',
      Math.min(1, Math.max(0.05, quality / 100)),
    )
  })

  const bitmap = await createImageBitmap(blob)
  const decodeCanvas = document.createElement('canvas')
  decodeCanvas.width = width
  decodeCanvas.height = height
  const decodeCtx = decodeCanvas.getContext('2d')
  if (!decodeCtx) throw new Error('Canvas unavailable')
  decodeCtx.drawImage(bitmap, 0, 0)
  const decoded = decodeCtx.getImageData(0, 0, width, height).data
  const out = new Float64Array(width * height)
  for (let i = 0, p = 0; i < decoded.length; i += 4, p++) {
    out[p] = (decoded[i] / 255) * scale + min
  }
  return { reconstructed: out, encodedBytes: blob.size }
}

export async function runJpeg2000Compression(
  bands: BandMap,
  bandOrder: string[],
  width: number,
  height: number,
  rate: number,
): Promise<{ bands: BandMap; metadata: Record<string, unknown>; compressedBytesEstimate: number }> {
  const out: BandMap = {}
  let encodedTotal = 0
  for (const name of bandOrder) {
    const result = await compressBandJpeg(bands[name], width, height, rate)
    out[name] = result.reconstructed
    encodedTotal += result.encodedBytes
  }
  return {
    bands: out,
    metadata: {
      codec: 'jpeg-canvas-proxy',
      note: 'JPEG encode/decode proxy for JPEG2000 workflow on Vercel/browser',
      quality: rate,
    },
    compressedBytesEstimate: encodedTotal,
  }
}
