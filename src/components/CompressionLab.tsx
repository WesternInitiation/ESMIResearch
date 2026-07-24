'use client'

import { useEffect, useMemo, useState } from 'react'
import {
  attachNdviToRun,
  fetchSupabaseStatus,
  listRecentRuns,
  loadRunByShareToken,
  saveCompressionRun,
  type SharedRunSummary,
} from '@/lib/api'
import { runCompressionAsync } from '@/lib/compressClient'
import type { MethodParams } from '@/lib/compression'
import {
  bandsToPngBlob,
  inspectUpload,
  isLikelySingleBand,
  loadArchiveMemberImage,
  loadImageFile,
  pairRedNirImages,
  rgbaToDataUrl,
  suggestNdviMembers,
  toPreviewRgba,
  type ArchiveSelection,
  type LoadedImage,
} from '@/lib/image'
import { compareNdvi, computeNdvi, type NdviMetrics } from '@/lib/ndvi'
import {
  bandsToCompressedArtifactPreview,
  bandsToDecompressedPreview,
  dataUrlToJpegDataUrl,
  downloadDataUrl,
  residualPreviewRgba,
  rgbaToPngDataUrl,
} from '@/lib/preview'
import { downsampleBands } from '@/lib/resize'
import { fetchCloudRunStatus, runServerCompression } from '@/lib/serverCompress'
import { MAX_INGEST_BYTES } from '@/lib/archive'
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

type WorkingImage = LoadedImage & {
  nativeWidth: number
  nativeHeight: number
  processScale: number
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
  svdRank: 24,
  waveletKeepFraction: 0.08,
  waveletLevels: 3,
  bandwidthKeepFraction: 0.12,
  jpegRate: 0.45,
}

const PROCESS_DIM_OPTIONS_BROWSER = [512, 768, 1024, 1536, 2048, 3072, 4096] as const
const PROCESS_DIM_OPTIONS_SERVER = [1024, 1536, 2048, 3072, 4096, 6144, 8192] as const

type Engine = 'browser' | 'cloud-run'

function toWorkingImage(loaded: LoadedImage, maxDim: number): WorkingImage {
  const resized = downsampleBands(
    loaded.bands,
    loaded.bandOrder,
    loaded.size.width,
    loaded.size.height,
    maxDim,
  )
  return {
    ...loaded,
    bands: resized.bands,
    size: { width: resized.width, height: resized.height },
    previewRgba: toPreviewRgba(
      resized.bands,
      loaded.bandOrder,
      resized.width,
      resized.height,
    ),
    nativeWidth: loaded.size.width,
    nativeHeight: loaded.size.height,
    processScale: resized.scale,
  }
}

export default function CompressionLab() {
  const [image, setImage] = useState<WorkingImage | null>(null)
  const [rawFile, setRawFile] = useState<File | null>(null)
  const [archive, setArchive] = useState<ArchiveSelection | null>(null)
  const [archiveMember, setArchiveMember] = useState<string>('')
  const [ndviRedMember, setNdviRedMember] = useState<string>('')
  const [ndviNirMember, setNdviNirMember] = useState<string>('')
  const [pairMode, setPairMode] = useState(false)
  const [pendingRedSingle, setPendingRedSingle] = useState<LoadedImage | null>(null)
  const [maxProcessDim, setMaxProcessDim] = useState<number>(1024)
  const [engine, setEngine] = useState<Engine>('browser')
  const [cloudRunOk, setCloudRunOk] = useState(false)
  const [cloudRunConfigured, setCloudRunConfigured] = useState(false)
  const [serverOriginalPreview, setServerOriginalPreview] = useState<string | null>(null)
  const [compressedArtifactPreview, setCompressedArtifactPreview] = useState<string | null>(
    null,
  )
  const [decompressedPreview, setDecompressedPreview] = useState<string | null>(null)
  const [residualPreview, setResidualPreview] = useState<string | null>(null)
  const [method, setMethod] = useState<CompressionMethod>('SVD')
  const [params, setParams] = useState<MethodParams>(DEFAULT_PARAMS)
  const [result, setResult] = useState<CompressionResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [status, setStatus] = useState<string | null>(null)
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
    if (serverOriginalPreview) return serverOriginalPreview
    if (!image) return null
    return rgbaToDataUrl(
      image.previewRgba,
      image.size.width,
      image.size.height,
    )
  }, [image, serverOriginalPreview])

  function jpegQualityForMethod(): number {
    if (method === 'JPEG2000') return params.jpegRate
    if (method === 'Wavelet transformation') {
      return Math.min(0.92, Math.max(0.08, params.waveletKeepFraction * 4))
    }
    if (method === 'Bandwidth transformation') {
      return Math.min(0.92, Math.max(0.08, params.bandwidthKeepFraction * 4))
    }
    // SVD: lower rank → lower JPEG quality stand-in for the compressed artifact
    return Math.min(0.92, Math.max(0.08, params.svdRank / 64))
  }

  function clearResultPreviews() {
    setCompressedArtifactPreview(null)
    setDecompressedPreview(null)
    setResidualPreview(null)
    setServerOriginalPreview(null)
  }

  function applyBrowserResultPreviews(
    out: CompressionResult,
    source: WorkingImage,
  ) {
    const decompressed = bandsToDecompressedPreview(
      out.bands,
      out.bandOrder,
      out.width,
      out.height,
    )
    const compressed = bandsToCompressedArtifactPreview(
      out.bands,
      out.bandOrder,
      out.width,
      out.height,
      jpegQualityForMethod(),
    )
    const residual = rgbaToPngDataUrl(
      residualPreviewRgba(
        source.bands,
        out.bands,
        out.bandOrder,
        out.width,
        out.height,
      ),
      out.width,
      out.height,
    )
    setDecompressedPreview(decompressed)
    setCompressedArtifactPreview(compressed)
    setResidualPreview(residual)
  }

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
    void (async () => {
      try {
        const status = await fetchCloudRunStatus()
        setCloudRunConfigured(status.urlConfigured)
        setCloudRunOk(status.configured)
      } catch {
        setCloudRunConfigured(false)
        setCloudRunOk(false)
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

  async function applyLoaded(loaded: LoadedImage, file: File | null) {
    const working = toWorkingImage(loaded, maxProcessDim)
    setImage(working)
    setRawFile(file)
    setResult(null)
    setNdvi(null)
    setCompareRows(null)
    clearResultPreviews()
    setPendingRedSingle(null)
    setStatus('Image ready')
  }

  async function loadArchiveNdviPair(redMember: string, nirMember: string) {
    if (!archive) return
    if (!redMember || !nirMember) {
      setError('Pick both a Red band file and a NIR band file')
      return
    }
    if (redMember === nirMember) {
      setError('Red and NIR must be different single-band files')
      return
    }
    setBusy(true)
    setError(null)
    setStatus(`Pairing NDVI bands…`)
    try {
      const redImg = await loadArchiveMemberImage(archive, redMember)
      const nirImg = await loadArchiveMemberImage(archive, nirMember)
      const paired = pairRedNirImages(redImg, nirImg)
      setArchiveMember(redMember)
      setNdviRedMember(redMember)
      setNdviNirMember(nirMember)
      setPairMode(true)
      await applyLoaded(paired, rawFile)
      setStatus('NDVI pair loaded (Red + NIR). Run compression, then Compare NDVI.')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load NDVI band pair')
      setStatus(null)
    } finally {
      setBusy(false)
    }
  }

  async function onFile(file: File | null) {
    if (!file) return
    setError(null)
    setResult(null)
    setNdvi(null)
    setCompareRows(null)
    setSharedView(null)
    setShareToken(null)
    setArchive(null)
    setArchiveMember('')
    setNdviRedMember('')
    setNdviNirMember('')
    setPairMode(false)
    setPendingRedSingle(null)
    if (file.size > MAX_INGEST_BYTES) {
      setError(
        `File is ${(file.size / (1024 * 1024 * 1024)).toFixed(2)} GiB — ingest limit is ~2 GiB.`,
      )
      setStatus(null)
      setImage(null)
      return
    }
    setBusy(true)
    setStatus('Loading…')
    try {
      const inspected = await inspectUpload(file)
      if (inspected.kind === 'archive') {
        setArchive(inspected.selection)
        setRawFile(file)
        const suggested = suggestNdviMembers(inspected.selection.members)
        const first = inspected.selection.members[0]
        setArchiveMember(first)
        setNdviRedMember(suggested.red || first)
        setNdviNirMember(
          suggested.nir ||
            inspected.selection.members.find((m) => m !== (suggested.red || first)) ||
            first,
        )
        if (suggested.red && suggested.nir) {
          setStatus(
            `Archive ${file.name}: found likely Red (${suggested.red.split('/').pop()}) and NIR (${suggested.nir.split('/').pop()}). Load NDVI pair or pick one image.`,
          )
          // Auto-load the Landsat-style pair when both B4/B5-like members exist.
          const redImg = await loadArchiveMemberImage(inspected.selection, suggested.red)
          const nirImg = await loadArchiveMemberImage(inspected.selection, suggested.nir)
          const paired = pairRedNirImages(redImg, nirImg)
          setArchiveMember(suggested.red)
          setPairMode(true)
          await applyLoaded(paired, file)
          setStatus(
            'Auto-paired Red+NIR from archive for NDVI. You can change the members below and reload.',
          )
        } else {
          setPairMode(false)
          const loaded = await loadArchiveMemberImage(inspected.selection, first)
          await applyLoaded(loaded, file)
          setStatus(
            `Archive loaded. For NDVI with single-band TIFs, pick Red + NIR members and click Load NDVI pair.`,
          )
        }
      } else {
        const loaded = await loadImageFile(file)
        if (isLikelySingleBand(loaded) && !loaded.bands.red && !loaded.bands.nir) {
          // Hold as Red candidate; ask for a second single-band NIR upload.
          setPendingRedSingle(loaded)
          setRawFile(file)
          setImage(null)
          setStatus(
            `Loaded single-band ${file.name} as Red candidate. Upload a second single-band TIF for NIR to enable NDVI.`,
          )
        } else {
          await applyLoaded(loaded, file)
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load file')
      setStatus(null)
      setImage(null)
    } finally {
      setBusy(false)
    }
  }

  async function onNirPairFile(file: File | null) {
    if (!file || !pendingRedSingle) return
    setBusy(true)
    setError(null)
    setStatus('Pairing Red + NIR single-band files…')
    try {
      const nirLoaded = await loadImageFile(file)
      const paired = pairRedNirImages(pendingRedSingle, nirLoaded)
      setPairMode(true)
      await applyLoaded(paired, rawFile)
      setStatus('NDVI pair ready from two single-band uploads. Run compression, then Compare NDVI.')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to pair Red/NIR files')
      setStatus(null)
    } finally {
      setBusy(false)
    }
  }

  async function onArchiveMemberChange(member: string) {
    if (!archive) return
    setArchiveMember(member)
    setPairMode(false)
    setBusy(true)
    setError(null)
    setStatus(`Loading ${member}…`)
    try {
      const loaded = await loadArchiveMemberImage(archive, member)
      await applyLoaded(loaded, rawFile)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load archive member')
      setStatus(null)
    } finally {
      setBusy(false)
    }
  }

  async function onMaxDimChange(next: number) {
    setMaxProcessDim(next)
    if (!image) return
    // Re-derive from current working bands is lossy if already downsampled.
    // Prefer reloading from file/archive when possible.
    setBusy(true)
    setError(null)
    setStatus('Re-sampling for processing size…')
    try {
      if (archive && pairMode && ndviRedMember && ndviNirMember) {
        const redImg = await loadArchiveMemberImage(archive, ndviRedMember)
        const nirImg = await loadArchiveMemberImage(archive, ndviNirMember)
        const paired = pairRedNirImages(redImg, nirImg)
        const working = toWorkingImage(paired, next)
        setImage(working)
        setResult(null)
        setNdvi(null)
        setCompareRows(null)
        clearResultPreviews()
        setStatus(
          `Processing size ${working.size.width}×${working.size.height} (native ${working.nativeWidth}×${working.nativeHeight})`,
        )
      } else if (archive && archiveMember) {
        const loaded = await loadArchiveMemberImage(archive, archiveMember)
        const working = toWorkingImage(loaded, next)
        setImage(working)
        setResult(null)
        setNdvi(null)
        setCompareRows(null)
        clearResultPreviews()
        setStatus(
          `Processing size ${working.size.width}×${working.size.height} (native ${working.nativeWidth}×${working.nativeHeight})`,
        )
      } else if (rawFile && !archive) {
        const loaded = await loadImageFile(rawFile)
        const working = toWorkingImage(loaded, next)
        setImage(working)
        setResult(null)
        setNdvi(null)
        setCompareRows(null)
        clearResultPreviews()
        setStatus(
          `Processing size ${working.size.width}×${working.size.height} (native ${working.nativeWidth}×${working.nativeHeight})`,
        )
      } else {
        const working = toWorkingImage(image, next)
        setImage(working)
        setResult(null)
        clearResultPreviews()
        setStatus(`Processing size ${working.size.width}×${working.size.height}`)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to resample')
    } finally {
      setBusy(false)
    }
  }

  async function onRun() {
    if (!image || !rawFile) return
    setError(null)
    setNdvi(null)
    setBusy(true)

    if (engine === 'cloud-run') {
      if (rawFile.size > 30 * 1024 * 1024) {
        setError(
          'Cloud Run direct uploads are limited to ~30 MB by Google. Switch Engine → Browser for files up to ~2 GiB.',
        )
        setBusy(false)
        return
      }
      setStatus('Sending job to Cloud Run (cold start may take ~15s)…')
      try {
        const out = await runServerCompression({
          file: rawFile,
          filename: rawFile.name,
          method,
          archiveMember: archive ? archiveMember : null,
          maxDim: maxProcessDim,
          svdRank: params.svdRank,
          waveletKeepFraction: params.waveletKeepFraction,
          waveletLevels: params.waveletLevels,
          bandwidthKeepFraction: params.bandwidthKeepFraction,
          jpegRate: params.jpegRate,
          redBand,
          nirBand,
        })
        setServerOriginalPreview(
          `data:image/png;base64,${out.originalPreviewPngBase64}`,
        )
        const decompressed = `data:image/png;base64,${out.previewPngBase64}`
        setDecompressedPreview(decompressed)
        try {
          const compressed = await dataUrlToJpegDataUrl(
            decompressed,
            jpegQualityForMethod(),
          )
          setCompressedArtifactPreview(compressed)
        } catch {
          setCompressedArtifactPreview(decompressed)
        }
        setResidualPreview(null)
        setResult({
          method: out.method,
          bands: {},
          bandOrder: out.bandOrder,
          width: out.width,
          height: out.height,
          runtimeSeconds: out.runtimeSeconds,
          originalBytes: out.originalBytes,
          compressedBytesEstimate: out.compressedBytesEstimate,
          compressionRatio: out.compressionRatio,
          channelReports: out.channelReports,
          metadata: { ...out.metadata, engine: out.engine },
        })
        if (out.ndvi) {
          setNdvi({
            rmse: out.ndvi.rmse,
            mae: out.ndvi.mae,
            correlation: out.ndvi.correlation,
            ssim: out.ndvi.ssim,
            bias: out.ndvi.bias,
          })
        }
        setStatus(
          `Cloud Run done in ${out.runtimeSeconds.toFixed(2)}s · ${out.width}×${out.height} (native ${out.nativeWidth}×${out.nativeHeight})`,
        )
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Cloud Run compression failed')
        setStatus(null)
      } finally {
        setBusy(false)
      }
      return
    }

    clearResultPreviews()
    setStatus(`Running ${method} in browser worker…`)
    try {
      const out = await runCompressionAsync({
        method,
        bands: image.bands,
        bandOrder: image.bandOrder,
        width: image.size.width,
        height: image.size.height,
        originalBytes: image.originalBytes,
        params,
        onProgress: (message) => setStatus(message),
      })
      setResult(out)
      applyBrowserResultPreviews(out, image)
      setStatus(
        `Done in ${out.runtimeSeconds.toFixed(2)}s · ratio ${fmt(out.compressionRatio, 3)} · ${out.width}×${out.height}`,
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Compression failed')
      setStatus(null)
    } finally {
      setBusy(false)
    }
  }

  async function onCompareAll() {
    if (!image) return
    if (engine === 'cloud-run') {
      setError('Compare-all on Cloud Run is not enabled yet — switch to Browser engine.')
      return
    }
    setError(null)
    setBusy(true)
    clearResultPreviews()
    setStatus('Comparing all methods…')
    try {
      const rows: CompareRow[] = []
      for (const m of COMPRESSION_METHODS) {
        setStatus(`Comparing: ${m}…`)
        const out = await runCompressionAsync({
          method: m,
          bands: image.bands,
          bandOrder: image.bandOrder,
          width: image.size.width,
          height: image.size.height,
          originalBytes: image.originalBytes,
          params,
        })
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
  }

  function onNdvi() {
    if (!image || !result) return
    if (Object.keys(result.bands).length === 0) {
      setStatus('NDVI for Cloud Run results is computed on the server when Red/NIR exist.')
      return
    }
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
      let compressedBlob: Blob
      if (decompressedPreview) {
        const res = await fetch(decompressedPreview)
        compressedBlob = await res.blob()
      } else {
        compressedBlob = await bandsToPngBlob(
          result.bands,
          result.bandOrder,
          result.width,
          result.height,
        )
      }
      const saved = await saveCompressionRun({
        method: result.method,
        sourceFilename: image.filename,
        params: {
          ...params,
          method: result.method,
          metadata: result.metadata,
          archiveMember: image.archiveMember ?? null,
          maxProcessDim,
          engine,
          processSize: image.size,
          nativeSize: { width: image.nativeWidth, height: image.nativeHeight },
        },
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
        originalFilename: image.archiveMember
          ? `${rawFile.name}__${image.archiveMember.replace(/\//g, '_')}`
          : image.filename,
        compressedFile: compressedBlob,
        compressedFilename: `compressed-${Date.now()}.png`,
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
            Browser-side satellite compression lab with optional Google Cloud Run
            backend for heavier jobs — SVD, wavelet, bandwidth, and JPEG2000, plus
            TAR archives and NDVI checks.
          </p>
        </div>
        <div className="header-meta">
          <span className={`pill ${supabaseOk ? 'ok' : 'warn'}`}>
            {supabaseOk ? 'Supabase connected' : 'Supabase offline'}
          </span>
          <span className={`pill ${cloudRunOk ? 'ok' : 'warn'}`}>
            {cloudRunOk
              ? 'Cloud Run online'
              : cloudRunConfigured
                ? 'Cloud Run offline'
                : 'Cloud Run unset'}
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
            <span>Upload GeoTIFF / PNG / JPEG / TAR / TAR.GZ (up to ~2 GiB)</span>
            <input
              type="file"
              accept=".tif,.tiff,.geotiff,.png,.jpg,.jpeg,.webp,.tar,.tar.gz,.tgz"
              onChange={(e) => void onFile(e.target.files?.[0] ?? null)}
            />
          </label>

          {pendingRedSingle && (
            <label className="file">
              <span>Upload NIR single-band TIF (pairs with Red above)</span>
              <input
                type="file"
                accept=".tif,.tiff,.geotiff"
                disabled={busy}
                onChange={(e) => void onNirPairFile(e.target.files?.[0] ?? null)}
              />
            </label>
          )}

          {archive && (
            <>
              <label>
                Single image inside archive
                <select
                  value={archiveMember}
                  disabled={busy}
                  onChange={(e) => void onArchiveMemberChange(e.target.value)}
                >
                  {archive.members.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </label>

              <h2>NDVI band pair</h2>
              <p className="hint">
                For single-band Landsat/Sentinel TIFs, pick Red (e.g. B4) and NIR (e.g. B5),
                then load the pair.
              </p>
              <label>
                Red band file
                <select
                  value={ndviRedMember}
                  disabled={busy}
                  onChange={(e) => setNdviRedMember(e.target.value)}
                >
                  {archive.members.map((m) => (
                    <option key={`red-${m}`} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                NIR band file
                <select
                  value={ndviNirMember}
                  disabled={busy}
                  onChange={(e) => setNdviNirMember(e.target.value)}
                >
                  {archive.members.map((m) => (
                    <option key={`nir-${m}`} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </label>
              <button
                type="button"
                className="secondary"
                disabled={busy || !ndviRedMember || !ndviNirMember}
                onClick={() => void loadArchiveNdviPair(ndviRedMember, ndviNirMember)}
              >
                {pairMode ? 'Reload NDVI pair' : 'Load NDVI pair (Red + NIR)'}
              </button>
            </>
          )}

          <h2>Engine</h2>
          <select
            value={engine}
            onChange={(e) => {
              const next = e.target.value as Engine
              setEngine(next)
              if (next === 'cloud-run' && maxProcessDim < 1024) {
                void onMaxDimChange(2048)
              } else if (next === 'browser' && maxProcessDim > 4096) {
                void onMaxDimChange(1024)
              }
            }}
          >
            <option value="browser">Browser (Web Worker)</option>
            <option value="cloud-run" disabled={!cloudRunConfigured}>
              Cloud Run {cloudRunOk ? '(online)' : cloudRunConfigured ? '(unreachable)' : '(not configured)'}
            </option>
          </select>
          {!cloudRunConfigured && (
            <p className="hint">
              Set <code>COMPRESS_API_URL</code> on Vercel to enable Cloud Run. See{' '}
              <code>cloud_run/README.md</code>.
            </p>
          )}

          <label>
            Max processing size (faster ← → sharper)
            <select
              value={maxProcessDim}
              disabled={busy}
              onChange={(e) => void onMaxDimChange(Number(e.target.value))}
            >
              {(engine === 'cloud-run'
                ? PROCESS_DIM_OPTIONS_SERVER
                : PROCESS_DIM_OPTIONS_BROWSER
              ).map((d) => (
                <option key={d} value={d}>
                  {d}px
                </option>
              ))}
            </select>
          </label>

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
                max={64}
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
            <button type="button" disabled={!image || !rawFile || busy} onClick={() => void onRun()}>
              {busy
                ? 'Working…'
                : engine === 'cloud-run'
                  ? 'Run on Cloud Run'
                  : 'Run compression'}
            </button>
            <button
              type="button"
              className="secondary"
              disabled={!image || busy || engine === 'cloud-run'}
              onClick={() => void onCompareAll()}
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
            <div className="preview-grid preview-grid-3">
              <figure>
                <figcaption>Original</figcaption>
                {originalPreview ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={originalPreview} alt="Original preview" />
                ) : (
                  <div className="empty">Upload an image or TAR archive to begin</div>
                )}
              </figure>
              <figure>
                <figcaption>Compressed</figcaption>
                {compressedArtifactPreview ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={compressedArtifactPreview} alt="Compressed artifact preview" />
                ) : (
                  <div className="empty">Run compression to see the encoded artifact</div>
                )}
              </figure>
              <figure>
                <figcaption>Decompressed</figcaption>
                {decompressedPreview ? (
                  <>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={decompressedPreview} alt="Decompressed preview" />
                    <button
                      type="button"
                      className="secondary preview-download"
                      onClick={() =>
                        downloadDataUrl(
                          decompressedPreview,
                          `esmi-decompressed-${Date.now()}.png`,
                        )
                      }
                    >
                      Download decompressed PNG
                    </button>
                  </>
                ) : (
                  <div className="empty">Run compression to decompress and preview</div>
                )}
              </figure>
            </div>
            {residualPreview && (
              <div className="residual-block">
                <figcaption>Compression residual (|original − decompressed|)</figcaption>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={residualPreview} alt="Compression residual map" />
              </div>
            )}
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
