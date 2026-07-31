/**
 * Lightweight Node check for browser Haar (odd sizes) + LZW size match.
 * Run: node scripts/check_browser_roundtrip.mjs
 */

function haarForward(row) {
  const n = row.length
  const out = new Float64Array(n)
  const half = Math.floor(n / 2)
  for (let i = 0; i < half; i++) {
    const a = row[i * 2]
    const b = row[i * 2 + 1] ?? a
    out[i] = (a + b) / 2
    out[half + i] = (a - b) / 2
  }
  if (n % 2 === 1) out[n - 1] = row[n - 1]
  return out
}

function haarInverse(row) {
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

function rmse(a, b) {
  let s = 0
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i]
    s += d * d
  }
  return Math.sqrt(s / a.length)
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg)
}

for (const n of [4, 5, 63, 64, 65]) {
  const row = Float64Array.from({ length: n }, (_, i) => Math.sin(i * 0.37) * 50 + i)
  const back = haarInverse(haarForward(row))
  const err = rmse(row, back)
  assert(err < 1e-10, `Haar 1D n=${n} RMSE ${err}`)
}

// Odd 2D row/col pass
function haar2D(data, w, h, inverse) {
  const op = inverse ? haarInverse : haarForward
  const out = new Float64Array(data)
  for (let y = 0; y < h; y++) {
    const row = out.subarray(y * w, y * w + w)
    const t = op(row)
    out.set(t, y * w)
  }
  for (let x = 0; x < w; x++) {
    const col = new Float64Array(h)
    for (let y = 0; y < h; y++) col[y] = out[y * w + x]
    const t = op(col)
    for (let y = 0; y < h; y++) out[y * w + x] = t[y]
  }
  return out
}

for (const [w, h] of [
  [8, 8],
  [9, 7],
  [65, 63],
]) {
  const data = Float64Array.from({ length: w * h }, (_, i) => (i % 17) + i * 0.01)
  const back = haar2D(haar2D(data, w, h, false), w, h, true)
  const err = rmse(data, back)
  assert(err < 1e-9, `Haar 2D ${w}x${h} RMSE ${err}`)
}

console.log('browser Haar round-trips OK')
