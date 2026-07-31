/** Quick check: downsample then upsample restores approximate size/content. */

function downsample(src, w, h, maxDim) {
  const longest = Math.max(w, h)
  if (maxDim <= 0 || longest <= maxDim) return { data: src, w, h }
  const scale = maxDim / longest
  const nw = Math.max(1, Math.round(w * scale))
  const nh = Math.max(1, Math.round(h * scale))
  const dst = new Float64Array(nw * nh)
  for (let y = 0; y < nh; y++) {
    const sy = Math.min(h - 1, Math.floor((y + 0.5) / scale))
    for (let x = 0; x < nw; x++) {
      const sx = Math.min(w - 1, Math.floor((x + 0.5) / scale))
      dst[y * nw + x] = src[sy * w + sx]
    }
  }
  return { data: dst, w: nw, h: nh }
}

function upsample(src, w, h, tw, th) {
  const dst = new Float64Array(tw * th)
  const xScale = w / tw
  const yScale = h / th
  for (let y = 0; y < th; y++) {
    const fy = (y + 0.5) * yScale - 0.5
    const y0 = Math.max(0, Math.min(h - 1, Math.floor(fy)))
    const y1 = Math.max(0, Math.min(h - 1, y0 + 1))
    const ty = Math.max(0, Math.min(1, fy - y0))
    for (let x = 0; x < tw; x++) {
      const fx = (x + 0.5) * xScale - 0.5
      const x0 = Math.max(0, Math.min(w - 1, Math.floor(fx)))
      const x1 = Math.max(0, Math.min(w - 1, x0 + 1))
      const tx = Math.max(0, Math.min(1, fx - x0))
      const v00 = src[y0 * w + x0]
      const v10 = src[y0 * w + x1]
      const v01 = src[y1 * w + x0]
      const v11 = src[y1 * w + x1]
      dst[y * tw + x] =
        v00 * (1 - tx) * (1 - ty) +
        v10 * tx * (1 - ty) +
        v01 * (1 - tx) * ty +
        v11 * tx * ty
    }
  }
  return dst
}

const W = 8011
const H = 7901
const src = Float64Array.from({ length: W * H }, (_, i) => (i % 255) + 0.25)
const mid = downsample(src, W, H, 1024)
if (mid.w >= W || mid.h >= H) throw new Error('expected downsample')
const back = upsample(mid.data, mid.w, mid.h, W, H)
if (back.length !== W * H) throw new Error(`size ${back.length} != ${W * H}`)
console.log(`resize OK: ${W}x${H} → ${mid.w}x${mid.h} → ${W}x${H}`)
