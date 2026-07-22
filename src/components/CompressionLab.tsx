'use client'

import { useEffect, useMemo, useState, useTransition } from 'react'
import {
  attachNdviToRun,
  fetchSupabaseStatus,
  listRecentRuns,
  loadRunByShareToken,
  saveCompressionRun,
  type SharedRunSummary,
} from '@/lib/api'
import { runCompression, type MethodParams } from '@/lib/compression'
import {
  bandsToPngBlob,
  loadImageFile,
  rgbaToDataUrl,
  toPreviewRgba,
  type LoadedImage,
} from '@/lib/image'
import { compareNdvi, computeNdvi, type NdviMetrics } from '@/lib/ndvi'
import {
  COMPRESSION_METHODS,
  type CompressionMethod,
  type CompressionResult,
} from '@/lib/types'

type CompareRow = {
  method: CompressionMethod
  runtimeSeconds: number
  compressionRatio: number
  meanRmse: number
  meanPsnr: number
  meanSsim: number
}

function fmt(n: number, digits = 4): string {
  if (!Number.isFinite(n)) return '—'
  return n.toPrecision(digits)
}

function bytesLabel(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(2)} MB`
}

const DEFAULT_PARAMS: MethodParams = {
  svdRank: 32,
  waveletKeepFraction: 0.08,
  waveletLevels: 3,
  bandwidthKeepFraction: 0.12,
  jpegRate: 0.45,
}

export default function CompressionLab() {
  const [image, setImage] = useState<LoadedImage | null>(null)
  const [rawFile, setRawFile] = useState<File | null>(null)
  const [method, setMethod] = useState<CompressionMethod>('SVD')
  const [params, setParams] = useState<MethodParams>(DEFAULT_PARAMS)
  const [result, setResult] = useState<CompressionResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [status, setStatus] = useState<string | null>(null)
  const [pending, startTransition] = useTransition()
  const [busy, setBusy] = useState(false)

  const [redBand, setRedBand] = useState('red')
  const [nirBand, setNirBand] = useState('nir')
  const [ndvi, setNdvi] = useState<NdviMetrics | null>(null)

  const [compareRows, setCompareRows] = useState<CompareRow[] | null>(null)
  const [supabaseOk, setSupabaseOk] = useState(false)
  const [recentRuns, setRecentRuns] = useState<SharedRunSummary[]>([])
  const [shareToken, setShareToken] = useState<string | null>(null)
  const [sharedView, setSharedView] = useState<{
    method: string
    source: string
    ratio: number
    runtime: number
    metrics: Array<Record<string, unknown>>
    originalUrl: string | null
    compressedUrl: string | null
    ndvi: Record<string, unknown> | null
  } | null>(null)

  const originalPreview = useMemo(() => {
    if (!image) return null
    return rgbaToDataUrl(
      image.previewRgba,
      image.size.width,
      image.size.height,
    )
  }, [image])

  const compressedPreview = useMemo(() => {
    if (!result) return null
    const rgba = toPreviewRgba(
      result.bands,
      result.bandOrder,
      result.width,
      result.height,
    )
    return rgbaToDataUrl(rgba, result.width, result.height)
  }, [result])

  useEffect(() => {
    void (async () => {
      try {
        const ok = await fetchSupabaseStatus()
        setSupabaseOk(ok)
        if (ok) {
          const runs = await listRecentRuns()
          setRecentRuns(runs)
        }
      } catch {
        setSupabaseOk(false)
      }
    })()
  }, [])

  useEffect(() => {
    const token = new URLSearchParams(window.location.search).get('run')
    if (!token) return
    void (async () => {
      try {
        setStatus('Loading shared run…')
        const data = await loadRunByShareToken(token)
        setShareToken(token)
        setSharedView({
          method: String(data.run.method ?? ''),
          source: String(data.run.source_filename ?? ''),
          ratio: Number(data.run.compression_ratio ?? 0),
          runtime: Number(data.run.runtime_seconds ?? 0),
          metrics: data.metrics,
          originalUrl: data.originalUrl,
          compressedUrl: data.compressedUrl,
          ndvi: data.ndvi,
        })
        setStatus(null)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Could not load shared run')
        setStatus(null)
      }
    })()
  }, [])

  useEffect(() => {
    if (!image) return
    const order = image.bandOrder
    if (order.includes('red')) setRedBand('red')
    else if (order.length) setRedBand(order[0])
    if (order.includes('nir')) setNirBand('nir')
    else if (order.length > 1) setNirBand(order[Math.min(3, order.length - 1)])
  }, [image])

  async function onFile(file: File | null) {
    if (!file) return
    setError(null)
    setResult(null)
    setNdvi(null)
    setCompareRows(null)
    setSharedView(null)
    setShareToken(null)
    setBusy(true)
    setStatus('Loading image…')
    try {
      const loaded = await loadImageFile(file)
      setImage(loaded)
      setRawFile(file)
      setStatus(
        `Loaded ${loaded.filename} · ${loaded.size.width}×${loaded.size.height} · ${loaded.bandOrder.length} bands`,
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load image')
      setStatus(null)
    } finally {
      setBusy(false)
    }
  }

  function onRun() {
    if (!image) return
    setError(null)
    setNdvi(null)
    setBusy(true)
    setStatus(`Running ${method}…`)
    startTransition(() => {
      void (async () => {
        try {
          const out = await runCompression(
            method,
            image.bands,
            image.bandOrder,
            image.size.width,
            image.size.height,
            image.originalBytes,
            params,
          )
          setResult(out)
          setStatus(
            `Done in ${out.runtimeSeconds.toFixed(2)}s · ratio ${fmt(out.compressionRatio, 3)}`,
          )
        } catch (err) {
          setError(err instanceof Error ? err.message : 'Compression failed')
          setStatus(null)
        } finally {
          setBusy(false)
        }
      })()
    })
  }

  function onCompareAll() {
    if (!image) return
    setError(null)
    setBusy(true)
    setStatus('Comparing all methods…')
    startTransition(() => {
      void (async () => {
        try {
          const rows: CompareRow[] = []
          for (const m of COMPRESSION_METHODS) {
            const out = await runCompression(
              m,
              image.bands,
              image.bandOrder,
              image.size.width,
              image.size.height,
              image.originalBytes,
              params,
            )
            const meanRmse =
              out.channelReports.reduce((s, r) => s + r.rmse, 0) /
              Math.max(out.channelReports.length, 1)
            const meanPsnr =
              out.channelReports.reduce((s, r) => s + r.psnrDb, 0) /
              Math.max(out.channelReports.length, 1)
            const meanSsim =
              out.channelReports.reduce((s, r) => s + r.ssim, 0) /
              Math.max(out.channelReports.length, 1)
            rows.push({
              method: m,
              runtimeSeconds: out.runtimeSeconds,
              compressionRatio: out.compressionRatio,
              meanRmse,
              meanPsnr,
              meanSsim,
            })
          }
          setCompareRows(rows)
          setStatus('Comparison complete')
        } catch (err) {
          setError(err instanceof Error ? err.message : 'Comparison failed')
        } finally {
          setBusy(false)
        }
      })()
    })
  }

  function onNdvi() {
    if (!image || !result) return
    const redO = image.bands[redBand]
    const nirO = image.bands[nirBand]
    const redC = result.bands[redBand]
    const nirC = result.bands[nirBand]
    if (!redO || !nirO || !redC || !nirC) {
      setError(`Need both ${redBand} and ${nirBand} in original and reconstructed bands`)
      return
    }
    const ref = computeNdvi(redO, nirO)
    const cand = computeNdvi(redC, nirC)
    setNdvi(compareNdvi(ref, cand))
    setStatus('NDVI comparison ready')
  }

  async function onSave() {
    if (!image || !result || !rawFile) return
    setBusy(true)
    setError(null)
    setStatus('Saving to Supabase…')
    try {
      const compressedBlob = await bandsToPngBlob(
        result.bands,
        result.bandOrder,
        result.width,
        result.height,
      )
      const saved = await saveCompressionRun({
        method: result.method,
        sourceFilename: image.filename,
        params: { ...params, method: result.method, metadata: result.metadata },
        runtimeSeconds: result.runtimeSeconds,
        originalBytes: result.originalBytes,
        compressedBytesEstimate: result.compressedBytesEstimate,
        compressionRatio: result.compressionRatio,
        bandMetrics: result.channelReports.map((r) => ({
          band: r.band,
          rmse: r.rmse,
          mae: r.mae,
          psnr_db: r.psnrDb,
          ssim: r.ssim,
        })),
        originalFile: rawFile,
        originalFilename: image.filename,
        compressedFile: compressedBlob,
        compressedFilename: `${image.filename.replace(/\.[^.]+$/, '')}-compressed.png`,
      })
      setShareToken(saved.shareToken)
      const url = new URL(window.location.href)
      url.searchParams.set('run', saved.shareToken)
      window.history.replaceState({}, '', url.toString())
      setStatus(`Saved. Share link uses ?run=${saved.shareToken}`)
      const runs = await listRecentRuns()
      setRecentRuns(runs)
      if (ndvi) {
        await attachNdviToRun(saved.shareToken, {
          redBand,
          nirBand,
          rmse: ndvi.rmse,
          mae: ndvi.mae,
          correlation: ndvi.correlation,
          ssim: ndvi.ssim,
          bias: ndvi.bias,
        })
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Save failed')
      setStatus(null)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="lab">
      <header className="lab-header">
        <div>
          <p className="eyebrow">Earth Sensing · Matrix Imaging</p>
          <h1>ESMI Research</h1>
          <p className="lede">
            Browser-side satellite compression lab — SVD, wavelet, bandwidth, and
            JPEG-rate methods with NDVI checks. Deployed on Vercel.
          </p>
        </div>
        <div className="header-meta">
          <span className={`pill ${supabaseOk ? 'ok' : 'warn'}`}>
            {supabaseOk ? 'Supabase connected' : 'Supabase offline'}
          </span>
        </div>
      </header>

      {sharedView && (
        <section className="panel shared">
          <h2>Shared run</h2>
          <p>
            {sharedView.method} · {sharedView.source} · ratio{' '}
            {fmt(sharedView.ratio, 3)} · {sharedView.runtime.toFixed(2)}s
          </p>
          <div className="preview-grid">
            {sharedView.originalUrl && (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={sharedView.originalUrl} alt="Original artifact" />
            )}
            {sharedView.compressedUrl && (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={sharedView.compressedUrl} alt="Compressed artifact" />
            )}
          </div>
          {sharedView.metrics.length > 0 && (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Band</th>
                    <th>RMSE</th>
                    <th>MAE</th>
                    <th>PSNR</th>
                    <th>SSIM</th>
                  </tr>
                </thead>
                <tbody>
                  {sharedView.metrics.map((m) => (
                    <tr key={String(m.band)}>
                      <td>{String(m.band)}</td>
                      <td>{fmt(Number(m.rmse))}</td>
                      <td>{fmt(Number(m.mae))}</td>
                      <td>{fmt(Number(m.psnr_db), 3)}</td>
                      <td>{fmt(Number(m.ssim), 3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}

      <div className="layout">
        <aside className="panel controls">
          <h2>Source</h2>
          <label className="file">
            <span>Upload GeoTIFF / PNG / JPEG</span>
            <input
              type="file"
              accept=".tif,.tiff,.geotiff,.png,.jpg,.jpeg,.webp"
              onChange={(e) => void onFile(e.target.files?.[0] ?? null)}
            />
          </label>
          {image && (
            <p className="hint">
              {image.filename}
              <br />
              {image.size.width}×{image.size.height} · {image.sourceType} ·{' '}
              {bytesLabel(image.originalBytes)}
              <br />
              Bands: {image.bandOrder.join(', ')}
            </p>
          )}

          <h2>Method</h2>
          <select
            value={method}
            onChange={(e) => setMethod(e.target.value as CompressionMethod)}
          >
            {COMPRESSION_METHODS.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>

          {method === 'SVD' && (
            <label>
              SVD rank
              <input
                type="range"
                min={1}
                max={128}
                value={params.svdRank}
                onChange={(e) =>
                  setParams((p) => ({ ...p, svdRank: Number(e.target.value) }))
                }
              />
              <span>{params.svdRank}</span>
            </label>
          )}
          {method === 'Wavelet transformation' && (
            <>
              <label>
                Keep fraction
                <input
                  type="range"
                  min={0.01}
                  max={0.5}
                  step={0.01}
                  value={params.waveletKeepFraction}
                  onChange={(e) =>
                    setParams((p) => ({
                      ...p,
                      waveletKeepFraction: Number(e.target.value),
                    }))
                  }
                />
                <span>{params.waveletKeepFraction.toFixed(2)}</span>
              </label>
              <label>
                Levels
                <input
                  type="range"
                  min={1}
                  max={5}
                  value={params.waveletLevels}
                  onChange={(e) =>
                    setParams((p) => ({
                      ...p,
                      waveletLevels: Number(e.target.value),
                    }))
                  }
                />
                <span>{params.waveletLevels}</span>
              </label>
            </>
          )}
          {method === 'Bandwidth transformation' && (
            <label>
              Keep fraction
              <input
                type="range"
                min={0.02}
                max={0.5}
                step={0.01}
                value={params.bandwidthKeepFraction}
                onChange={(e) =>
                  setParams((p) => ({
                    ...p,
                    bandwidthKeepFraction: Number(e.target.value),
                  }))
                }
              />
              <span>{params.bandwidthKeepFraction.toFixed(2)}</span>
            </label>
          )}
          {method === 'JPEG2000' && (
            <label>
              JPEG quality (browser stand-in)
              <input
                type="range"
                min={0.1}
                max={0.95}
                step={0.05}
                value={params.jpegRate}
                onChange={(e) =>
                  setParams((p) => ({ ...p, jpegRate: Number(e.target.value) }))
                }
              />
              <span>{params.jpegRate.toFixed(2)}</span>
            </label>
          )}

          <div className="actions">
            <button type="button" disabled={!image || busy || pending} onClick={onRun}>
              Run compression
            </button>
            <button
              type="button"
              className="secondary"
              disabled={!image || busy || pending}
              onClick={onCompareAll}
            >
              Compare all methods
            </button>
            <button
              type="button"
              className="secondary"
              disabled={!result || !supabaseOk || busy}
              onClick={() => void onSave()}
            >
              Save run to Supabase
            </button>
          </div>

          {(status || error) && (
            <p className={error ? 'error' : 'status'}>{error || status}</p>
          )}
          {shareToken && (
            <p className="hint">
              Share: <code>?run={shareToken}</code>
            </p>
          )}
        </aside>

        <main className="main-col">
          <section className="panel">
            <h2>Preview</h2>
            <div className="preview-grid">
              <figure>
                <figcaption>Original</figcaption>
                {originalPreview ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={originalPreview} alt="Original preview" />
                ) : (
                  <div className="empty">Upload an image to begin</div>
                )}
              </figure>
              <figure>
                <figcaption>Reconstructed</figcaption>
                {compressedPreview ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={compressedPreview} alt="Compressed preview" />
                ) : (
                  <div className="empty">Run a method to see reconstruction</div>
                )}
              </figure>
            </div>
          </section>

          {result && (
            <section className="panel">
              <h2>Band metrics</h2>
              <p className="hint">
                Runtime {result.runtimeSeconds.toFixed(2)}s · Estimate{' '}
                {bytesLabel(result.compressedBytesEstimate)} · Ratio{' '}
                {fmt(result.compressionRatio, 3)}
              </p>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Band</th>
                      <th>RMSE</th>
                      <th>MAE</th>
                      <th>PSNR (dB)</th>
                      <th>SSIM</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.channelReports.map((r) => (
                      <tr key={r.band}>
                        <td>{r.band}</td>
                        <td>{fmt(r.rmse)}</td>
                        <td>{fmt(r.mae)}</td>
                        <td>{fmt(r.psnrDb, 3)}</td>
                        <td>{fmt(r.ssim, 3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {image && result && (
            <section className="panel">
              <h2>NDVI preservation</h2>
              <div className="ndvi-row">
                <label>
                  Red band
                  <select value={redBand} onChange={(e) => setRedBand(e.target.value)}>
                    {image.bandOrder.map((b) => (
                      <option key={b} value={b}>
                        {b}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  NIR band
                  <select value={nirBand} onChange={(e) => setNirBand(e.target.value)}>
                    {image.bandOrder.map((b) => (
                      <option key={b} value={b}>
                        {b}
                      </option>
                    ))}
                  </select>
                </label>
                <button type="button" className="secondary" onClick={onNdvi}>
                  Compare NDVI
                </button>
              </div>
              {ndvi && (
                <div className="metric-grid">
                  <div>
                    <span>RMSE</span>
                    <strong>{fmt(ndvi.rmse)}</strong>
                  </div>
                  <div>
                    <span>MAE</span>
                    <strong>{fmt(ndvi.mae)}</strong>
                  </div>
                  <div>
                    <span>Corr</span>
                    <strong>{fmt(ndvi.correlation, 3)}</strong>
                  </div>
                  <div>
                    <span>SSIM</span>
                    <strong>{fmt(ndvi.ssim, 3)}</strong>
                  </div>
                  <div>
                    <span>Bias</span>
                    <strong>{fmt(ndvi.bias)}</strong>
                  </div>
                </div>
              )}
            </section>
          )}

          {compareRows && (
            <section className="panel">
              <h2>Method comparison</h2>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Method</th>
                      <th>Runtime (s)</th>
                      <th>Ratio</th>
                      <th>Mean RMSE</th>
                      <th>Mean PSNR</th>
                      <th>Mean SSIM</th>
                    </tr>
                  </thead>
                  <tbody>
                    {compareRows.map((r) => (
                      <tr key={r.method}>
                        <td>{r.method}</td>
                        <td>{r.runtimeSeconds.toFixed(2)}</td>
                        <td>{fmt(r.compressionRatio, 3)}</td>
                        <td>{fmt(r.meanRmse)}</td>
                        <td>{fmt(r.meanPsnr, 3)}</td>
                        <td>{fmt(r.meanSsim, 3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {supabaseOk && recentRuns.length > 0 && (
            <section className="panel">
              <h2>Recent shared runs</h2>
              <ul className="run-list">
                {recentRuns.map((r) => (
                  <li key={r.id}>
                    <a href={`?run=${r.share_token}`}>
                      {r.method} · {r.source_filename || 'untitled'} ·{' '}
                      {new Date(r.created_at).toLocaleString()}
                    </a>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </main>
      </div>
    </div>
  )
}
