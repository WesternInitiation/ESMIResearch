import type { BandMap } from '../types'

function getCanvas(width: number, height: number): {
  canvas: OffscreenCanvas | HTMLCanvasElement
  ctx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D
} {
  if (typeof OffscreenCanvas !== 'undefined') {
    const canvas = new OffscreenCanvas(width, height)
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('Canvas unavailable')
    return { canvas, ctx }
  }
  if (typeof document === 'undefined') {
    throw new Error('Canvas unavailable in this environment')
  }
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Canvas unavailable')
  return { canvas, ctx }
}

async function canvasToJpegBlob(
  canvas: OffscreenCanvas | HTMLCanvasElement,
  quality: number,
): Promise<Blob> {
  const q = Math.min(1, Math.max(0.05, quality <= 1 ? quality : quality / 100))
  if ('convertToBlob' in canvas) {
    return canvas.convertToBlob({ type: 'image/jpeg', quality: q })
  }
  return new Promise((resolve, reject) => {
    ;(canvas as HTMLCanvasElement).toBlob(
      (b) => (b ? resolve(b) : reject(new Error('JPEG encode failed'))),
      'image/jpeg',
      q,
    )
  })
}

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
  for (let i = 0; i < band.length; i++) {
    const v = band[i]
    if (v < min) min = v
    if (v > max) max = v
  }
  const scale = max - min || 1
  const { canvas, ctx } = getCanvas(width, height)
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

  const blob = await canvasToJpegBlob(canvas, quality)
  const bitmap = await createImageBitmap(blob)
  const decoded = getCanvas(width, height)
  decoded.ctx.drawImage(bitmap, 0, 0)
  const pixels = decoded.ctx.getImageData(0, 0, width, height).data
  const out = new Float64Array(width * height)
  for (let i = 0, p = 0; i < pixels.length; i += 4, p++) {
    out[p] = (pixels[i] / 255) * scale + min
  }
  bitmap.close?.()
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
