'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import {
  fetchSupabaseStatus,
  listRecentRuns,
  loadRunByShareToken,
  type SharedRunSummary,
} from '@/lib/api'
import { runCompressionAsync } from '@/lib/compressClient'
import type { MethodParams } from '@/lib/compression'
import { compareIndexMaps, computeNdvi, computeNdwi, type IndexMetrics } from '@/lib/ndvi'
import {
  bandsToCompressedArtifactPreview,
  bandsToDecompressedPreview,
  dataUrlToJpegDataUrl,
  residualPreviewRgba,
  rgbaToPngDataUrl,
} from '@/lib/preview'
import { downloadCsv } from '@/lib/csv'
import {
  downloadBandsAsGeoTiff,
  downloadRgbPreviewAsGeoTiff,
} from '@/lib/geotiffExport'
import { jpegRoundtripBands } from '@/lib/lossyBands'
import { estimateByteSize } from '@/lib/metrics'
import { fetchDemoCatalog, fetchDemoMemberFile, fetchGcsBuckets } from '@/lib/demoData'
import { downsampleBands } from '@/lib/resize'
import { fetchCloudRunStatus, runServerCompression, VERCEL_PROXY_UPLOAD_BYTES } from '@/lib/serverCompress'
import {
  MAX_INGEST_BYTES,
  archiveChildren,
  extractArchiveMember,
} from '@/lib/archive'
import {
  archiveSelectionFromMembers,
  inspectUpload,
  isLikelySingleBand,
  loadArchiveMemberImage,
  loadImageFile,
  mergeNamedBands,
  pairNdwiImages,
  pairRedNirImages,
  rgbaToDataUrl,
  suggestNdviMembers,
  suggestNdwiMembers,
  toPreviewRgba,
  type ArchiveSelection,
  type LoadedImage,
} from '@/lib/image'
import {
  COMPRESSION_METHODS,
  type CompressionMethod,
  type CompressionResult,
} from '@/lib/types'

type CompareRow = {
  method: CompressionMethod
  runtimeSeconds: number
  compressionRatio: number
  originalBytes: number
  compressedBytesEstimate: number
  decompressedBytes: number
  meanRmse: number
  meanPsnr: number
  meanSsim: number
}

type IndexCompareTarget = 'decompressed' | 'compressed'

type WorkingImage = LoadedImage & {
  nativeWidth: number
  nativeHeight: number
  processScale: number
}

function memberLabel(path: string): string {
  return path.split('/').pop() || path
}

function folderBreadcrumb(path: string): string {
  return path ? path : '(archive root)'
}

function parentFolder(path: string): string {
  if (!path) return ''
  const i = path.lastIndexOf('/')
  return i === -1 ? '' : path.slice(0, i)
}

function fmt(n: number, digits = 4): string {
  if (!Number.isFinite(n)) return '—'
  return n.toPrecision(digits)
}

function bytesLabel(n: number): string {
  if (!Number.isFinite(n) || n < 0) return '—'
  if (n < 1024) return `${Math.round(n)} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(2)} MB`
}

function estimateDecompressedBytes(result: {
  decompressedBytes?: number
  bands?: CompressionResult['bands']
  width: number
  height: number
  channelReports: CompressionResult['channelReports']
  bandOrder: string[]
}): number {
  if (result.decompressedBytes && result.decompressedBytes > 0) {
    return result.decompressedBytes
  }
  if (result.bands && Object.keys(result.bands).length > 0) {
    return estimateByteSize(result.bands)
  }
  const channels = Math.max(
    result.channelReports.length,
    result.bandOrder.length,
    1,
  )
  return result.width * result.height * channels * 8
}

const DEFAULT_PARAMS: MethodParams = {
  svdRank: 24,
  waveletKeepFraction: 0.08,
  waveletLevels: 3,
  waveletName: 'db4',
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

/** Multi-member TAR pairs (Red+NIR etc.) must stay in the browser. */
function isMultiMemberStack(img: LoadedImage | null | undefined): boolean {
  return Boolean(img?.archiveMember?.includes(' + '))
}

export default function CompressionLab() {
  const [image, setImage] = useState<WorkingImage | null>(null)
  const [rawFile, setRawFile] = useState<File | null>(null)
  const [archive, setArchive] = useState<ArchiveSelection | null>(null)
  const [archiveMember, setArchiveMember] = useState<string>('')
  const [ndviRedMember, setNdviRedMember] = useState<string>('')
  const [ndviNirMember, setNdviNirMember] = useState<string>('')
  const [ndwiGreenMember, setNdwiGreenMember] = useState<string>('')
  const [ndwiSecondMember, setNdwiSecondMember] = useState<string>('')
  const [ndwiSecondRole, setNdwiSecondRole] = useState<'nir' | 'swir'>('nir')
  const [ndviPairLoaded, setNdviPairLoaded] = useState(false)
  const [ndwiPairLoaded, setNdwiPairLoaded] = useState(false)
  const [pairMode, setPairMode] = useState(false)
  const [pendingRedSingle, setPendingRedSingle] = useState<LoadedImage | null>(null)
  const [maxProcessDim, setMaxProcessDim] = useState<number>(1024)
  const [engine, setEngine] = useState<Engine>('cloud-run')
  const [cloudRunOk, setCloudRunOk] = useState(false)
  const [cloudRunConfigured, setCloudRunConfigured] = useState(false)
  const [gcsUploads, setGcsUploads] = useState(false)
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

  const [indexKind, setIndexKind] = useState<'ndvi' | 'ndwi'>('ndvi')
  const [indexCompareTarget, setIndexCompareTarget] =
    useState<IndexCompareTarget>('decompressed')
  const [redBand, setRedBand] = useState('red')
  const [nirBand, setNirBand] = useState('nir')
  const [greenBand, setGreenBand] = useState('green')
  const [ndwiSecondBand, setNdwiSecondBand] = useState('nir')
  const [indexMetrics, setIndexMetrics] = useState<IndexMetrics | null>(null)

  const [compareRows, setCompareRows] = useState<CompareRow[] | null>(null)
  const [gcsBucketOptions, setGcsBucketOptions] = useState<string[]>([])
  const [selectedDemoBucket, setSelectedDemoBucket] = useState('')
  const [customDemoBucket, setCustomDemoBucket] = useState('')
  const demoLoadSeq = useRef(0)
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

  const archiveFolderView = useMemo(() => {
    if (!archive) return { folders: [] as string[], images: [] as string[] }
    return archiveChildren(archive.listing, archive.folderPath)
  }, [archive])

  function navigateArchiveFolder(nextPath: string) {
    if (!archive) return
    setArchive({ ...archive, folderPath: nextPath })
    const child = archiveChildren(archive.listing, nextPath)
    if (archiveMember && child.images.includes(archiveMember)) return
    // Update the picker highlight without forcing a download/reload.
    setArchiveMember(child.images[0] ?? '')
  }

  function jpegQualityForMethod(): number {
    if (method === 'JPEG2000') return params.jpegRate
    if (method === 'LZW') return 0.85
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
        setGcsUploads(status.gcsUploads)
        // Keep Cloud Run as default when configured; otherwise fall back to Browser.
        if (!status.urlConfigured) {
          setEngine('browser')
        }
      } catch {
        setCloudRunConfigured(false)
        setCloudRunOk(false)
        setGcsUploads(false)
        setEngine('browser')
      }
    })()
  }, [])

  useEffect(() => {
    void (async () => {
      try {
        const data = await fetchGcsBuckets()
        const names = data.buckets.map((b) => b.name)
        setGcsBucketOptions(names)
        const initial = data.defaultBucket || names[0] || ''
        if (initial) {
          setSelectedDemoBucket(initial)
          void onLoadDemo(initial)
        }
      } catch {
        setGcsBucketOptions([])
      }
    })()
    // Intentionally mount-only: load the default bucket once options arrive.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Custom bucket name: debounce so typing doesn't spam list requests.
  useEffect(() => {
    const bucket = customDemoBucket.trim()
    if (!bucket) return
    const timer = window.setTimeout(() => {
      void onLoadDemo(bucket)
    }, 700)
    return () => window.clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [customDemoBucket])

  function activeDemoBucket(): string {
    return (customDemoBucket.trim() || selectedDemoBucket).trim()
  }

  async function onLoadDemo(bucketOverride?: string) {
    const bucket = (bucketOverride ?? activeDemoBucket()).trim()
    if (!bucket) {
      setError('Select a GCS bucket to load demo data')
      return
    }

    const seq = ++demoLoadSeq.current
    setBusy(true)
    setError(null)
    setStatus(`Listing files in gs://${bucket}…`)
    try {
      setImage(null)
      setRawFile(null)
      setResult(null)
      setIndexMetrics(null)
      setCompareRows(null)
      setSharedView(null)
      setShareToken(null)
      setPendingRedSingle(null)
      setPairMode(false)
      setNdviPairLoaded(false)
      setNdwiPairLoaded(false)
      clearResultPreviews()

      const catalog = await fetchDemoCatalog((message) => {
        if (seq === demoLoadSeq.current) setStatus(message)
      }, bucket)
      if (seq !== demoLoadSeq.current) return

      const members = catalog.members
      const suggested = suggestNdviMembers(members)
      const suggestedNdwi = suggestNdwiMembers(members, 'nir')

      if (catalog.kind === 'archive') {
        setArchive(
          archiveSelectionFromMembers(catalog.archiveName, members, {
            demoRemote: {
              kind: 'archive',
              objectName: catalog.objectName,
              bucket: catalog.bucket,
            },
          }),
        )
      } else {
        setArchive(
          archiveSelectionFromMembers(`gs://${catalog.bucket}`, members, {
            demoRemote: { kind: 'objects', bucket: catalog.bucket },
          }),
        )
      }

      // Same dropdown UX as a dropped TAR — do not download until the user picks.
      setArchiveMember('')
      setNdviRedMember(suggested.red || '')
      setNdviNirMember(
        suggested.nir || members.find((m) => m !== suggested.red) || '',
      )
      setNdwiGreenMember(suggestedNdwi.green || '')
      setNdwiSecondMember(
        suggestedNdwi.second ||
          members.find((m) => m !== suggestedNdwi.green) ||
          '',
      )
      setNdwiSecondRole(suggestedNdwi.secondRole)
      setStatus(
        `Demo ready from gs://${catalog.bucket} (${members.length} images). Pick one from the dropdown — only that file is downloaded.`,
      )
    } catch (err) {
      if (seq !== demoLoadSeq.current) return
      setError(err instanceof Error ? err.message : 'Failed to load demo catalog')
      setStatus(null)
      setArchive(null)
    } finally {
      if (seq === demoLoadSeq.current) setBusy(false)
    }
  }

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
    if (order.includes('green')) setGreenBand('green')
    else if (order.includes('green') === false && order.length) {
      // leave greenBand as-is unless green exists
    }
    if (order.includes('swir')) setNdwiSecondBand('swir')
    else if (order.includes('nir')) setNdwiSecondBand('nir')
  }, [image])

  async function applyLoaded(loaded: LoadedImage, file: File | null) {
    const working = toWorkingImage(loaded, maxProcessDim)
    setImage(working)
    setRawFile(file)
    setResult(null)
    setIndexMetrics(null)
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
      let paired = pairRedNirImages(redImg, nirImg)
      // Keep any already-loaded NDWI bands (green/swir) if present.
      if (image && (image.bands.green || image.bands.swir)) {
        const keep: { name: string; image: LoadedImage }[] = []
        // Re-load green/swir from members if we know them; otherwise keep raster arrays via synthetic LoadedImage
        if (image.bands.green) {
          keep.push({
            name: 'green',
            image: {
              ...image,
              bands: { green: image.bands.green },
              bandOrder: ['green'],
              previewRgba: image.previewRgba,
              filename: 'green',
            },
          })
        }
        if (image.bands.swir) {
          keep.push({
            name: 'swir',
            image: {
              ...image,
              bands: { swir: image.bands.swir },
              bandOrder: ['swir'],
              previewRgba: image.previewRgba,
              filename: 'swir',
            },
          })
        }
        if (keep.length) paired = mergeNamedBands(paired, keep, 'NDVI+NDWI stack')
      }
      setArchiveMember(redMember)
      setNdviRedMember(redMember)
      setNdviNirMember(nirMember)
      setNdviPairLoaded(true)
      setPairMode(true)
      await applyLoaded(paired, rawFile)
      setStatus('NDVI pair loaded (Red + NIR). Optionally load an NDWI pair too, then run compression.')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load NDVI band pair')
      setStatus(null)
    } finally {
      setBusy(false)
    }
  }

  async function loadArchiveNdwiPair(greenMember: string, secondMember: string) {
    if (!archive) return
    if (!greenMember || !secondMember) {
      setError('Pick both a Green band file and a NIR/SWIR band file')
      return
    }
    if (greenMember === secondMember) {
      setError('Green and NIR/SWIR must be different single-band files')
      return
    }
    setBusy(true)
    setError(null)
    setStatus(`Pairing NDWI bands…`)
    try {
      const greenImg = await loadArchiveMemberImage(archive, greenMember)
      const secondImg = await loadArchiveMemberImage(archive, secondMember)
      const role = ndwiSecondRole
      let paired = pairNdwiImages(greenImg, secondImg, role)

      // Merge onto existing NDVI red/nir stack when available.
      if (image && (image.bands.red || image.bands.nir)) {
        const keep: { name: string; image: LoadedImage }[] = []
        if (image.bands.red) {
          keep.push({
            name: 'red',
            image: {
              ...image,
              bands: { red: image.bands.red },
              bandOrder: ['red'],
              previewRgba: image.previewRgba,
              filename: 'red',
            },
          })
        }
        if (image.bands.nir && role !== 'nir') {
          keep.push({
            name: 'nir',
            image: {
              ...image,
              bands: { nir: image.bands.nir },
              bandOrder: ['nir'],
              previewRgba: image.previewRgba,
              filename: 'nir',
            },
          })
        }
        // If NDWI uses NIR and red exists, start from NDVI-like base then add green.
        if (image.bands.red && image.bands.nir && role === 'nir') {
          paired = mergeNamedBands(
            {
              ...image,
              bands: { red: image.bands.red, nir: image.bands.nir },
              bandOrder: ['red', 'nir'],
            },
            [
              { name: 'green', image: greenImg },
            ],
            'NDVI+NDWI stack',
          )
        } else if (keep.length) {
          paired = mergeNamedBands(paired, keep, 'NDVI+NDWI stack')
        }
      }

      setNdwiGreenMember(greenMember)
      setNdwiSecondMember(secondMember)
      setNdwiPairLoaded(true)
      setPairMode(true)
      setGreenBand('green')
      setNdwiSecondBand(role)
      setIndexKind('ndwi')
      await applyLoaded(paired, rawFile)
      setStatus(
        role === 'swir'
          ? 'MNDWI pair loaded (Green + SWIR). Run compression, then Compare NDWI.'
          : 'NDWI pair loaded (Green + NIR). Run compression, then Compare NDWI.',
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load NDWI band pair')
      setStatus(null)
    } finally {
      setBusy(false)
    }
  }

  async function onFile(file: File | null) {
    if (!file) return
    setError(null)
    setResult(null)
    setIndexMetrics(null)
    setCompareRows(null)
    setSharedView(null)
    setShareToken(null)
    setArchive(null)
    setArchiveMember('')
    setNdviRedMember('')
    setNdviNirMember('')
    setNdwiGreenMember('')
    setNdwiSecondMember('')
    setNdviPairLoaded(false)
    setNdwiPairLoaded(false)
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
        const suggestedNdwi = suggestNdwiMembers(inspected.selection.members, 'nir')
        const first = inspected.selection.members[0]
        setArchiveMember(first)
        setNdviRedMember(suggested.red || first)
        setNdviNirMember(
          suggested.nir ||
            inspected.selection.members.find((m) => m !== (suggested.red || first)) ||
            first,
        )
        setNdwiGreenMember(suggestedNdwi.green || first)
        setNdwiSecondMember(
          suggestedNdwi.second ||
            inspected.selection.members.find((m) => m !== (suggestedNdwi.green || first)) ||
            first,
        )
        setNdwiSecondRole(suggestedNdwi.secondRole)
        if (suggested.red && suggested.nir) {
          setStatus(
            `Archive ${file.name}: ${inspected.selection.members.length} images` +
              (inspected.selection.listing.folders.length
                ? ` in ${inspected.selection.listing.folders.length} folders`
                : '') +
              `. Found likely Red (${suggested.red.split('/').pop()}) and NIR (${suggested.nir.split('/').pop()}). Load NDVI pair or browse folders.`,
          )
          // Auto-load the Landsat-style pair when both B4/B5-like members exist.
          const redImg = await loadArchiveMemberImage(inspected.selection, suggested.red)
          const nirImg = await loadArchiveMemberImage(inspected.selection, suggested.nir)
          let paired = pairRedNirImages(redImg, nirImg)
          // Also merge green if present for NDWI readiness.
          if (suggestedNdwi.green) {
            const greenImg = await loadArchiveMemberImage(
              inspected.selection,
              suggestedNdwi.green,
            )
            paired = mergeNamedBands(
              paired,
              [{ name: 'green', image: greenImg }],
              'NDVI+NDWI stack',
            )
            setNdwiGreenMember(suggestedNdwi.green)
            setNdwiSecondMember(suggested.nir)
            setNdwiSecondRole('nir')
            setNdwiPairLoaded(true)
          }
          setArchiveMember(suggested.red)
          setNdviPairLoaded(true)
          setPairMode(true)
          await applyLoaded(paired, file)
          setStatus(
            suggestedNdwi.green
              ? 'Auto-paired Red+NIR (+ Green) from archive. Load NDWI pair if you want SWIR/MNDWI instead.'
              : 'Auto-paired Red+NIR from archive for NDVI. Optionally load an NDWI Green+NIR/SWIR pair below.',
          )
        } else {
          setPairMode(false)
          const loaded = await loadArchiveMemberImage(inspected.selection, first)
          await applyLoaded(loaded, file)
          setStatus(
            `Archive loaded (${inspected.selection.members.length} images` +
              (inspected.selection.listing.folders.length
                ? `, ${inspected.selection.listing.folders.length} folders`
                : '') +
              `). Browse folders or pick a band pair for NDVI/NDWI.`,
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
    if (!archive || !member) return
    setArchiveMember(member)
    const parent = parentFolder(member)
    if (parent !== archive.folderPath) {
      setArchive({ ...archive, folderPath: parent })
    }
    setPairMode(false)
    setNdviPairLoaded(false)
    setNdwiPairLoaded(false)
    setBusy(true)
    setError(null)
    setStatus(
      archive.demoRemote
        ? `Downloading ${member.split('/').pop() || member} from demo storage…`
        : `Loading ${member}…`,
    )
    try {
      if (archive.demoRemote) {
        const file = await fetchDemoMemberFile({
          kind: archive.demoRemote.kind,
          objectName: archive.demoRemote.objectName,
          member,
          onProgress: (message) => setStatus(message),
        })
        const loaded = await loadImageFile(file)
        const memberFilename = member.split('/').pop() || member
        await applyLoaded(
          {
            ...loaded,
            archiveMember: member,
            filename: `${archive.archiveName} → ${memberFilename}`,
          },
          file,
        )
      } else {
        const loaded = await loadArchiveMemberImage(archive, member)
        await applyLoaded(loaded, rawFile)
      }
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
      if (archive && pairMode && ndviPairLoaded && ndviRedMember && ndviNirMember) {
        const redImg = await loadArchiveMemberImage(archive, ndviRedMember)
        const nirImg = await loadArchiveMemberImage(archive, ndviNirMember)
        let paired = pairRedNirImages(redImg, nirImg)
        if (ndwiPairLoaded && ndwiGreenMember) {
          const greenImg = await loadArchiveMemberImage(archive, ndwiGreenMember)
          const additions: { name: string; image: LoadedImage }[] = [
            { name: 'green', image: greenImg },
          ]
          if (ndwiSecondRole === 'swir' && ndwiSecondMember) {
            const swirImg = await loadArchiveMemberImage(archive, ndwiSecondMember)
            additions.push({ name: 'swir', image: swirImg })
          }
          paired = mergeNamedBands(paired, additions, 'NDVI+NDWI stack')
        }
        const working = toWorkingImage(paired, next)
        setImage(working)
        setResult(null)
        setIndexMetrics(null)
        setCompareRows(null)
        clearResultPreviews()
        setStatus(
          `Processing size ${working.size.width}×${working.size.height} (native ${working.nativeWidth}×${working.nativeHeight})`,
        )
      } else if (
        archive &&
        pairMode &&
        ndwiPairLoaded &&
        ndwiGreenMember &&
        ndwiSecondMember
      ) {
        const greenImg = await loadArchiveMemberImage(archive, ndwiGreenMember)
        const secondImg = await loadArchiveMemberImage(archive, ndwiSecondMember)
        const paired = pairNdwiImages(greenImg, secondImg, ndwiSecondRole)
        const working = toWorkingImage(paired, next)
        setImage(working)
        setResult(null)
        setIndexMetrics(null)
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
        setIndexMetrics(null)
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
        setIndexMetrics(null)
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
    setIndexMetrics(null)
    setBusy(true)

    if (engine === 'cloud-run') {
      // Full NDVI/NDWI stacks from several TAR members — compress locally only.
      if (isMultiMemberStack(image)) {
        setStatus(
          'Multi-band TAR stack detected — running locally so NDVI/NDWI comparison uses the full pair…',
        )
      } else {
      try {
        // Prefer the selected TAR member only — never upload the whole archive.
        let uploadFile: Blob = rawFile
        let uploadName = rawFile.name
        let usedMember: string | null = null

        if (archive) {
          const memberPath =
            archiveMember && !archiveMember.includes(' + ')
              ? archiveMember
              : image.archiveMember && !image.archiveMember.includes(' + ')
                ? image.archiveMember
                : ''
          if (!memberPath) {
            setError(
              'Pick a single archive image to send to Cloud Run (or use Browser for multi-band TAR pairs).',
            )
            setBusy(false)
            return
          }
          if (archive.demoRemote) {
            // rawFile is already the selected demo member download.
            uploadFile = rawFile
            uploadName = rawFile.name
            usedMember = memberPath
          } else {
            if (!archive.buffer) {
              setError('Archive bytes are missing')
              setBusy(false)
              return
            }
            setStatus(`Extracting ${memberPath.split('/').pop()} from archive…`)
            const { bytes, memberFilename } = extractArchiveMember(
              archive.buffer,
              archive.archiveName,
              memberPath,
            )
            uploadFile = new File([new Uint8Array(bytes)], memberFilename, {
              type: 'application/octet-stream',
            })
            uploadName = memberFilename
            usedMember = memberPath
          }
        }

        // Small files go through Vercel multipart; larger ones use GCS signed PUT.
        if (uploadFile.size > VERCEL_PROXY_UPLOAD_BYTES && !gcsUploads) {
          setError(
            `Selected upload is ${(uploadFile.size / (1024 * 1024)).toFixed(1)} MB. ` +
              `Set GCS_UPLOAD_BUCKET on Vercel (see cloud_run/README.md) for 80–100+ MB Cloud Run jobs, ` +
              `or switch Engine → Browser.`,
          )
          setBusy(false)
          return
        }

        setStatus(
          usedMember
            ? `Preparing ${uploadName} for Cloud Run…`
            : 'Preparing Cloud Run job…',
        )
        const out = await runServerCompression({
          file: uploadFile,
          filename: uploadName,
          method,
          // Member already extracted client-side — do not re-send the TAR.
          archiveMember: null,
          maxDim: maxProcessDim,
          svdRank: params.svdRank,
          waveletKeepFraction: params.waveletKeepFraction,
          waveletLevels: params.waveletLevels,
          waveletName: params.waveletName,
          bandwidthKeepFraction: params.bandwidthKeepFraction,
          jpegRate: params.jpegRate,
          redBand,
          nirBand,
          gcsUploads,
          onProgress: (message) => setStatus(message),
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
        setIndexMetrics(null)
        setResult({
          method: out.method,
          bands: {},
          bandOrder: out.bandOrder,
          width: out.width,
          height: out.height,
          runtimeSeconds: out.runtimeSeconds,
          originalBytes: out.originalBytes,
          compressedBytesEstimate: out.compressedBytesEstimate,
          decompressedBytes:
            out.width *
            out.height *
            Math.max(out.channelReports.length, out.bandOrder.length, 1) *
            8,
          compressionRatio: out.compressionRatio,
          channelReports: out.channelReports,
          metadata: {
            ...out.metadata,
            engine: out.engine,
            ...(usedMember ? { archiveMemberUploaded: usedMember } : {}),
          },
        })
        setStatus(
          `Cloud Run done in ${out.runtimeSeconds.toFixed(2)}s · ${out.width}×${out.height} (native ${out.nativeWidth}×${out.nativeHeight}). NDVI/NDWI compare stays local — use Browser (or a multi-band stack) for index comparison.`,
        )
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Cloud Run compression failed')
        setStatus(null)
      } finally {
        setBusy(false)
      }
      return
      }
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
    setError(null)
    setBusy(true)
    clearResultPreviews()

    const useCloudRun =
      engine === 'cloud-run' && !isMultiMemberStack(image) && Boolean(rawFile)

    if (engine === 'cloud-run' && isMultiMemberStack(image)) {
      setStatus(
        'Multi-band stack — comparing all methods locally so the full pair is used…',
      )
    } else if (useCloudRun) {
      setStatus('Comparing all methods on Cloud Run…')
    } else {
      setStatus('Comparing all methods in browser…')
    }

    try {
      let cloudUpload: { file: Blob; filename: string } | null = null
      if (useCloudRun && rawFile) {
        let uploadFile: Blob = rawFile
        let uploadName = rawFile.name
        if (archive) {
          const memberPath =
            archiveMember && !archiveMember.includes(' + ')
              ? archiveMember
              : image.archiveMember && !image.archiveMember.includes(' + ')
                ? image.archiveMember
                : ''
          if (!memberPath) {
            throw new Error('Pick a single archive image before comparing on Cloud Run')
          }
          if (archive.demoRemote) {
            uploadFile = rawFile
            uploadName = rawFile.name
          } else {
            if (!archive.buffer) {
              throw new Error('Archive bytes are missing')
            }
            setStatus(`Extracting ${memberPath.split('/').pop()} for Cloud Run compare…`)
            const { bytes, memberFilename } = extractArchiveMember(
              archive.buffer,
              archive.archiveName,
              memberPath,
            )
            uploadFile = new File([new Uint8Array(bytes)], memberFilename, {
              type: 'application/octet-stream',
            })
            uploadName = memberFilename
          }
        }
        if (uploadFile.size > VERCEL_PROXY_UPLOAD_BYTES && !gcsUploads) {
          throw new Error(
            `Selected upload is ${(uploadFile.size / (1024 * 1024)).toFixed(1)} MB. ` +
              `Set GCS_UPLOAD_BUCKET for large Cloud Run jobs, or switch Engine → Browser.`,
          )
        }
        cloudUpload = { file: uploadFile, filename: uploadName }
      }

      const rows: CompareRow[] = []
      for (const m of COMPRESSION_METHODS) {
        setStatus(
          useCloudRun
            ? `Cloud Run comparing: ${m}…`
            : `Comparing: ${m}…`,
        )
        const out = useCloudRun && cloudUpload
          ? await runServerCompression({
              file: cloudUpload.file,
              filename: cloudUpload.filename,
              method: m,
              archiveMember: null,
              maxDim: maxProcessDim,
              svdRank: params.svdRank,
              waveletKeepFraction: params.waveletKeepFraction,
              waveletLevels: params.waveletLevels,
              waveletName: params.waveletName,
              bandwidthKeepFraction: params.bandwidthKeepFraction,
              jpegRate: params.jpegRate,
              redBand,
              nirBand,
              gcsUploads,
              onProgress: (message) => setStatus(`${m}: ${message}`),
            })
          : await runCompressionAsync({
              method: m,
              bands: image.bands,
              bandOrder: image.bandOrder,
              width: image.size.width,
              height: image.size.height,
              originalBytes: image.originalBytes,
              params,
            })

        const reports = out.channelReports
        const meanRmse =
          reports.reduce((s, r) => s + r.rmse, 0) / Math.max(reports.length, 1)
        const meanPsnr =
          reports.reduce((s, r) => s + r.psnrDb, 0) / Math.max(reports.length, 1)
        const meanSsim =
          reports.reduce((s, r) => s + r.ssim, 0) / Math.max(reports.length, 1)
        rows.push({
          method: m,
          runtimeSeconds: out.runtimeSeconds,
          compressionRatio: out.compressionRatio,
          originalBytes: out.originalBytes,
          compressedBytesEstimate: out.compressedBytesEstimate,
          decompressedBytes: estimateDecompressedBytes(out),
          meanRmse,
          meanPsnr,
          meanSsim,
        })
      }
      setCompareRows(rows)
      setStatus(
        useCloudRun
          ? 'Cloud Run comparison complete'
          : 'Comparison complete',
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Comparison failed')
    } finally {
      setBusy(false)
    }
  }

  async function onIndexCompare() {
    if (!image || !result) return
    if (Object.keys(result.bands).length === 0) {
      setError(
        'NDVI/NDWI comparison runs only locally. Run compression with Engine → Browser (multi-band TAR stacks do this automatically).',
      )
      setStatus(null)
      return
    }

    setBusy(true)
    setError(null)
    try {
      const needed =
        indexKind === 'ndvi' ? [redBand, nirBand] : [greenBand, ndwiSecondBand]
      let candidateBands = result.bands
      if (indexCompareTarget === 'compressed') {
        candidateBands = await jpegRoundtripBands(
          result.bands,
          needed,
          result.width,
          result.height,
          jpegQualityForMethod(),
        )
      }

      if (indexKind === 'ndvi') {
        const redO = image.bands[redBand]
        const nirO = image.bands[nirBand]
        const redC = candidateBands[redBand]
        const nirC = candidateBands[nirBand]
        if (!redO || !nirO || !redC || !nirC) {
          setError(`Need both ${redBand} and ${nirBand} in original and candidate bands`)
          return
        }
        const ref = computeNdvi(redO, nirO)
        const cand = computeNdvi(redC, nirC)
        setIndexMetrics(compareIndexMaps(ref, cand))
        setStatus(
          `NDVI: original vs after-${indexCompareTarget} (local)`,
        )
        return
      }

      const greenO = image.bands[greenBand]
      const secondO = image.bands[ndwiSecondBand]
      const greenC = candidateBands[greenBand]
      const secondC = candidateBands[ndwiSecondBand]
      if (!greenO || !secondO || !greenC || !secondC) {
        setError(
          `Need both ${greenBand} and ${ndwiSecondBand} in original and candidate bands for NDWI`,
        )
        return
      }
      const ref = computeNdwi(greenO, secondO)
      const cand = computeNdwi(greenC, secondC)
      setIndexMetrics(compareIndexMaps(ref, cand))
      const label = ndwiSecondBand === 'swir' ? 'MNDWI' : 'NDWI'
      setStatus(`${label}: original vs after-${indexCompareTarget} (local)`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Index comparison failed')
    } finally {
      setBusy(false)
    }
  }

  function onExportCsv() {
    if (!result && !compareRows?.length && !indexMetrics) {
      setError('Nothing to export yet — run compression or an index compare first.')
      return
    }
    const headers = [
      'section',
      'method',
      'index',
      'compare_target',
      'band',
      'runtime_seconds',
      'original_bytes',
      'compressed_bytes',
      'decompressed_bytes',
      'compression_ratio',
      'rmse',
      'mae',
      'psnr_db',
      'ssim',
      'correlation',
      'bias',
      'source',
      'engine',
    ]
    const rows: Array<Array<unknown>> = []
    if (result) {
      const decomp = estimateDecompressedBytes(result)
      rows.push([
        'run',
        result.method,
        '',
        '',
        '',
        result.runtimeSeconds,
        result.originalBytes,
        result.compressedBytesEstimate,
        decomp,
        result.compressionRatio,
        '',
        '',
        '',
        '',
        '',
        '',
        image?.filename || '',
        String(result.metadata.engine || engine),
      ])
      for (const ch of result.channelReports) {
        rows.push([
          'band_metric',
          result.method,
          '',
          '',
          ch.band,
          result.runtimeSeconds,
          result.originalBytes,
          result.compressedBytesEstimate,
          decomp,
          result.compressionRatio,
          ch.rmse,
          ch.mae,
          ch.psnrDb,
          ch.ssim,
          '',
          '',
          image?.filename || '',
          String(result.metadata.engine || engine),
        ])
      }
    }
    if (indexMetrics) {
      rows.push([
        'index_metric',
        result?.method || '',
        indexKind === 'ndwi' && ndwiSecondBand === 'swir' ? 'mndwi' : indexKind,
        indexCompareTarget,
        '',
        result?.runtimeSeconds ?? '',
        result?.originalBytes ?? '',
        result?.compressedBytesEstimate ?? '',
        result ? estimateDecompressedBytes(result) : '',
        result?.compressionRatio ?? '',
        indexMetrics.rmse,
        indexMetrics.mae,
        '',
        indexMetrics.ssim,
        indexMetrics.correlation,
        indexMetrics.bias,
        image?.filename || '',
        engine,
      ])
    }
    if (compareRows?.length) {
      for (const row of compareRows) {
        rows.push([
          'method_compare',
          row.method,
          '',
          '',
          '',
          row.runtimeSeconds,
          row.originalBytes,
          row.compressedBytesEstimate,
          row.decompressedBytes,
          row.compressionRatio,
          row.meanRmse,
          '',
          row.meanPsnr,
          row.meanSsim,
          '',
          '',
          image?.filename || '',
          engine,
        ])
      }
    }
    downloadCsv(headers, rows, `esmi-results-${Date.now()}.csv`)
    setStatus('CSV exported')
  }

  async function onDownloadDecompressedTif() {
    if (!result && !decompressedPreview) return
    setBusy(true)
    setError(null)
    try {
      const filename = `esmi-decompressed-${Date.now()}.tif`
      if (result && Object.keys(result.bands).length > 0) {
        await downloadBandsAsGeoTiff(
          result.bands,
          result.bandOrder,
          result.width,
          result.height,
          filename,
        )
      } else if (decompressedPreview) {
        await downloadRgbPreviewAsGeoTiff(decompressedPreview, filename)
      } else {
        throw new Error('No decompressed image available to download')
      }
      setStatus('Downloaded decompressed GeoTIFF')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'GeoTIFF download failed')
    } finally {
      setBusy(false)
    }
  }

  async function onDownloadCompressedTif() {
    if (!result && !compressedArtifactPreview) return
    setBusy(true)
    setError(null)
    try {
      const filename = `esmi-compressed-${Date.now()}.tif`
      if (result && Object.keys(result.bands).length > 0) {
        // Same lossy stand-in used for the Compressed preview / index compare.
        const compressedBands = await jpegRoundtripBands(
          result.bands,
          result.bandOrder,
          result.width,
          result.height,
          jpegQualityForMethod(),
        )
        await downloadBandsAsGeoTiff(
          compressedBands,
          result.bandOrder,
          result.width,
          result.height,
          filename,
        )
      } else if (compressedArtifactPreview) {
        await downloadRgbPreviewAsGeoTiff(compressedArtifactPreview, filename)
      } else {
        throw new Error('No compressed image available to download')
      }
      setStatus('Downloaded compressed GeoTIFF')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Compressed GeoTIFF download failed')
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
            backend for heavier jobs — SVD, wavelet, bandwidth, JPEG2000, and LZW, plus
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
                    <th>
                      Band
                      <span className="th-blurb">Spectral channel</span>
                    </th>
                    <th title="Root Mean Square Error — average size of pixel differences (lower is better)">
                      RMSE
                      <span className="th-blurb">Root mean square error</span>
                    </th>
                    <th title="Mean Absolute Error — average absolute pixel difference (lower is better)">
                      MAE
                      <span className="th-blurb">Mean absolute error</span>
                    </th>
                    <th title="Peak Signal-to-Noise Ratio in decibels — higher means closer to the original">
                      PSNR (dB)
                      <span className="th-blurb">Peak signal-to-noise ratio</span>
                    </th>
                    <th title="Structural Similarity Index — how alike structure/texture look (1 is identical)">
                      SSIM
                      <span className="th-blurb">Structural similarity</span>
                    </th>
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
            <div className="archive-browser">
              {(archiveFolderView.folders.length > 0 || archive.folderPath) && (
                <label>
                  <span>Folder in archive</span>
                  <div className="archive-folder-bar">
                    <button
                      type="button"
                      className="secondary"
                      disabled={busy || !archive.folderPath}
                      onClick={() => navigateArchiveFolder(parentFolder(archive.folderPath))}
                    >
                      Up
                    </button>
                    <select
                      value={archive.folderPath}
                      disabled={busy}
                      onChange={(e) => navigateArchiveFolder(e.target.value)}
                    >
                      <option value="">
                        {folderBreadcrumb('')}
                      </option>
                      {archive.listing.folders.map((folder) => (
                        <option key={folder} value={folder}>
                          {folder}
                        </option>
                      ))}
                    </select>
                  </div>
                  {archiveFolderView.folders.length > 0 && (
                    <div className="archive-folder-chips">
                      {archiveFolderView.folders.map((folder) => (
                        <button
                          key={folder}
                          type="button"
                          className="secondary"
                          disabled={busy}
                          onClick={() => navigateArchiveFolder(folder)}
                        >
                          {memberLabel(folder)}/
                        </button>
                      ))}
                    </div>
                  )}
                </label>
              )}
              <label>
                {archive.demoRemote
                  ? 'Demo image (downloaded on select)'
                  : 'Image in this folder'}
                <select
                  value={
                    archiveFolderView.images.includes(archiveMember)
                      ? archiveMember
                      : ''
                  }
                  disabled={busy}
                  onChange={(e) => void onArchiveMemberChange(e.target.value)}
                >
                  <option value="" disabled>
                    {archiveFolderView.images.length
                      ? 'Select an image…'
                      : archiveFolderView.folders.length
                        ? 'Open a subfolder…'
                        : 'No images here'}
                  </option>
                  {archiveFolderView.images.map((m) => (
                    <option key={m} value={m}>
                      {memberLabel(m)}
                    </option>
                  ))}
                </select>
              </label>
              <p className="hint">
                {archive.members.length} images
                {archive.listing.folders.length
                  ? ` · ${archive.listing.folders.length} folders`
                  : ''}
                {archive.folderPath ? ` · in ${archive.folderPath}` : ''}
              </p>
            </div>
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
              Set <code>COMPRESS_API_URL</code> + <code>GOOGLE_SERVICE_ACCOUNT_JSON</code> on
              Vercel (private Cloud Run). See <code>cloud_run/README.md</code>.
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
                Wavelet
                <select
                  value={params.waveletName}
                  onChange={(e) =>
                    setParams((p) => ({ ...p, waveletName: e.target.value }))
                  }
                >
                  <option value="db4">db4 (best quality)</option>
                  <option value="db2">db2</option>
                  <option value="sym4">sym4</option>
                  <option value="sym2">sym2</option>
                  <option value="haar">haar</option>
                </select>
              </label>
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
              {engine === 'browser' && params.waveletName !== 'haar' && (
                <p className="hint">
                  Browser uses Haar multilevel; switch Engine → Cloud Run for{' '}
                  {params.waveletName}.
                </p>
              )}
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
          {method === 'LZW' && (
            <p className="hint">
              LZW quantizes each band to 8-bit then runs classic dictionary coding
              (ashmeet13-style, adapted for float satellite bands). No extra knobs.
            </p>
          )}

          <div className="actions">
            <button type="button" disabled={!image || !rawFile || busy} onClick={() => void onRun()}>
              {busy
                ? 'Working…'
                : engine === 'cloud-run' && !isMultiMemberStack(image)
                  ? 'Run on Cloud Run'
                  : 'Run compression'}
            </button>
            <button
              type="button"
              className="secondary"
              disabled={!image || busy || (engine === 'cloud-run' && !rawFile)}
              onClick={() => void onCompareAll()}
              title={
                engine === 'cloud-run'
                  ? 'Runs all methods on Cloud Run'
                  : 'Runs all methods in the browser worker'
              }
            >
              Compare all methods
            </button>
            <button
              type="button"
              className="secondary"
              disabled={busy || !activeDemoBucket()}
              onClick={() => void onLoadDemo()}
              title="Reload the selected GCS bucket catalog"
            >
              Reload bucket
            </button>
            <button
              type="button"
              className="secondary"
              disabled={busy || (!result && !compareRows?.length && !indexMetrics)}
              onClick={onExportCsv}
              title="Export band metrics, sizes, and index results as CSV"
            >
              Export CSV
            </button>
          </div>

          <div className="gcs-bucket-row">
            <label>
              GCS bucket
              <select
                value={selectedDemoBucket}
                disabled={busy || (!gcsBucketOptions.length && !selectedDemoBucket)}
                onChange={(e) => {
                  const name = e.target.value
                  setSelectedDemoBucket(name)
                  setCustomDemoBucket('')
                  if (name) void onLoadDemo(name)
                }}
              >
                {(gcsBucketOptions.length
                  ? gcsBucketOptions
                  : selectedDemoBucket
                    ? [selectedDemoBucket]
                    : ['esmi-research-demo-data']
                ).map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Or custom bucket
              <input
                type="text"
                value={customDemoBucket}
                disabled={busy}
                placeholder="my-bucket-name"
                onChange={(e) => setCustomDemoBucket(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault()
                    const name = customDemoBucket.trim()
                    if (name) void onLoadDemo(name)
                  }
                }}
              />
            </label>
          </div>
          <p className="hint">
            Choosing a bucket loads its catalog automatically (custom names debounce ~0.7s).
          </p>

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
                  <div className="empty">Waiting for upload</div>
                )}
              </figure>
              <figure>
                <figcaption>Compressed</figcaption>
                {compressedArtifactPreview ? (
                  <>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={compressedArtifactPreview} alt="Compressed artifact preview" />
                    <button
                      type="button"
                      className="secondary preview-download"
                      disabled={busy}
                      onClick={() => void onDownloadCompressedTif()}
                    >
                      Download compressed TIF
                    </button>
                  </>
                ) : (
                  <div className="empty">Waiting for compression</div>
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
                      disabled={busy}
                      onClick={() => void onDownloadDecompressedTif()}
                    >
                      Download decompressed TIF
                    </button>
                  </>
                ) : (
                  <div className="empty">Waiting for decompression</div>
                )}
              </figure>
            </div>
            <div className="residual-block">
              <figcaption>Compression residual (|original − decompressed|)</figcaption>
              {residualPreview ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={residualPreview} alt="Compression residual map" />
              ) : (
                <div className="empty">Waiting for residual map</div>
              )}
            </div>
          </section>

          <section className="panel">
            <h2>Band metrics</h2>
            <p className={`hint ${result ? '' : 'placeholder'}`}>
              {result
                ? `Runtime ${result.runtimeSeconds.toFixed(2)}s · Original ${bytesLabel(result.originalBytes)} · Compressed ${bytesLabel(result.compressedBytesEstimate)} · Decompressed ${bytesLabel(estimateDecompressedBytes(result))} · Ratio ${fmt(result.compressionRatio, 3)}`
                : 'Runtime — · Original — · Compressed — · Decompressed — · Ratio —'}
            </p>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>
                      Band
                      <span className="th-blurb">Spectral channel</span>
                    </th>
                    <th title="Root Mean Square Error — average size of pixel differences (lower is better)">
                      RMSE
                      <span className="th-blurb">Root mean square error</span>
                    </th>
                    <th title="Mean Absolute Error — average absolute pixel difference (lower is better)">
                      MAE
                      <span className="th-blurb">Mean absolute error</span>
                    </th>
                    <th title="Peak Signal-to-Noise Ratio in decibels — higher means closer to the original">
                      PSNR (dB)
                      <span className="th-blurb">Peak signal-to-noise ratio</span>
                    </th>
                    <th title="Structural Similarity Index — how alike structure/texture look (1 is identical)">
                      SSIM
                      <span className="th-blurb">Structural similarity</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {result && result.channelReports.length > 0 ? (
                    result.channelReports.map((r) => (
                      <tr key={r.band}>
                        <td>{r.band}</td>
                        <td>{fmt(r.rmse)}</td>
                        <td>{fmt(r.mae)}</td>
                        <td>{fmt(r.psnrDb, 3)}</td>
                        <td>{fmt(r.ssim, 3)}</td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={5} className="hint placeholder">
                        Waiting for compression results
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>

          <section className="panel pair-panel">
            <h2>Index band pairs</h2>
            <p className="hint pair-hint">
              From a TAR, pick single-band files for NDVI (B4/B5) and NDWI (B3/B5 or B3/B6).
            </p>
            <div className="pair-grid">
              <div className="pair-card">
                <h3>NDVI</h3>
                <div className="pair-fields">
                  <label>
                    Red
                    <select
                      value={ndviRedMember}
                      disabled={busy || !archive}
                      onChange={(e) => setNdviRedMember(e.target.value)}
                    >
                      {(archive?.members?.length ? archive.members : ['—']).map((m) => (
                        <option key={`red-${m}`} value={m}>
                          {memberLabel(m)}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    NIR
                    <select
                      value={ndviNirMember}
                      disabled={busy || !archive}
                      onChange={(e) => setNdviNirMember(e.target.value)}
                    >
                      {(archive?.members?.length ? archive.members : ['—']).map((m) => (
                        <option key={`nir-${m}`} value={m}>
                          {memberLabel(m)}
                        </option>
                      ))}
                    </select>
                  </label>
                  <button
                    type="button"
                    className="secondary"
                    disabled={busy || !archive || !ndviRedMember || !ndviNirMember}
                    onClick={() => void loadArchiveNdviPair(ndviRedMember, ndviNirMember)}
                  >
                    {ndviPairLoaded ? 'Reload' : 'Load'}
                  </button>
                </div>
              </div>

              <div className="pair-card">
                <h3>NDWI</h3>
                <div className="pair-fields ndwi">
                  <label>
                    Formula
                    <select
                      value={ndwiSecondRole}
                      disabled={busy || !archive}
                      onChange={(e) => {
                        const role = e.target.value as 'nir' | 'swir'
                        setNdwiSecondRole(role)
                        if (archive) {
                          const suggested = suggestNdwiMembers(archive.members, role)
                          if (suggested.green) setNdwiGreenMember(suggested.green)
                          if (suggested.second) setNdwiSecondMember(suggested.second)
                        }
                      }}
                    >
                      <option value="nir">Green-NIR</option>
                      <option value="swir">Green-SWIR (MNDWI)</option>
                    </select>
                  </label>
                  <label>
                    Green
                    <select
                      value={ndwiGreenMember}
                      disabled={busy || !archive}
                      onChange={(e) => setNdwiGreenMember(e.target.value)}
                    >
                      {(archive?.members?.length ? archive.members : ['—']).map((m) => (
                        <option key={`green-${m}`} value={m}>
                          {memberLabel(m)}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    {ndwiSecondRole === 'swir' ? 'SWIR' : 'NIR'}
                    <select
                      value={ndwiSecondMember}
                      disabled={busy || !archive}
                      onChange={(e) => setNdwiSecondMember(e.target.value)}
                    >
                      {(archive?.members?.length ? archive.members : ['—']).map((m) => (
                        <option key={`ndwi2-${m}`} value={m}>
                          {memberLabel(m)}
                        </option>
                      ))}
                    </select>
                  </label>
                  <button
                    type="button"
                    className="secondary"
                    disabled={busy || !archive || !ndwiGreenMember || !ndwiSecondMember}
                    onClick={() => void loadArchiveNdwiPair(ndwiGreenMember, ndwiSecondMember)}
                  >
                    {ndwiPairLoaded ? 'Reload' : 'Load'}
                  </button>
                </div>
              </div>
            </div>
            {!archive && (
              <p className="hint placeholder">Upload a TAR archive to enable band pairing</p>
            )}
          </section>

          <section className="panel">
            <h2>NDVI/NDWI preservation</h2>
            <p className="hint">
              Compare the index on the original image against after-compression
              (JPEG stand-in of reconstructed bands) or after-decompression
              (reconstructed float bands). Runs in the browser.
            </p>
            <div className="ndvi-row">
              <label>
                Index
                <select
                  value={indexKind}
                  onChange={(e) => setIndexKind(e.target.value as 'ndvi' | 'ndwi')}
                >
                  <option value="ndvi">NDVI</option>
                  <option value="ndwi">
                    {ndwiSecondBand === 'swir' ? 'MNDWI' : 'NDWI'}
                  </option>
                </select>
              </label>
              <label>
                Compare against
                <select
                  value={indexCompareTarget}
                  onChange={(e) =>
                    setIndexCompareTarget(e.target.value as IndexCompareTarget)
                  }
                >
                  <option value="decompressed">After decompression</option>
                  <option value="compressed">After compression</option>
                </select>
              </label>
              {indexKind === 'ndvi' ? (
                <>
                  <label>
                    Red band
                    <select
                      value={redBand}
                      disabled={!image}
                      onChange={(e) => setRedBand(e.target.value)}
                    >
                      {(image?.bandOrder?.length ? image.bandOrder : ['—']).map((b) => (
                        <option key={b} value={b}>
                          {b}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    NIR band
                    <select
                      value={nirBand}
                      disabled={!image}
                      onChange={(e) => setNirBand(e.target.value)}
                    >
                      {(image?.bandOrder?.length ? image.bandOrder : ['—']).map((b) => (
                        <option key={b} value={b}>
                          {b}
                        </option>
                      ))}
                    </select>
                  </label>
                </>
              ) : (
                <>
                  <label>
                    Green band
                    <select
                      value={greenBand}
                      disabled={!image}
                      onChange={(e) => setGreenBand(e.target.value)}
                    >
                      {(image?.bandOrder?.length ? image.bandOrder : ['—']).map((b) => (
                        <option key={b} value={b}>
                          {b}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    {ndwiSecondBand === 'swir' ? 'SWIR band' : 'NIR band'}
                    <select
                      value={ndwiSecondBand}
                      disabled={!image}
                      onChange={(e) => setNdwiSecondBand(e.target.value)}
                    >
                      {(image?.bandOrder?.length ? image.bandOrder : ['—']).map((b) => (
                        <option key={b} value={b}>
                          {b}
                        </option>
                      ))}
                    </select>
                  </label>
                </>
              )}
              <button
                type="button"
                className="secondary"
                disabled={!image || !result || busy}
                onClick={() => void onIndexCompare()}
              >
                Compare{' '}
                {indexKind === 'ndvi'
                  ? 'NDVI'
                  : ndwiSecondBand === 'swir'
                    ? 'MNDWI'
                    : 'NDWI'}
              </button>
            </div>
            <div className="metric-grid">
              <div title="Root Mean Square Error — average size of index differences (lower is better)">
                <span>
                  RMSE
                  <em className="th-blurb">Root mean square error</em>
                </span>
                <strong className={indexMetrics ? undefined : 'placeholder'}>
                  {indexMetrics ? fmt(indexMetrics.rmse) : '—'}
                </strong>
              </div>
              <div title="Mean Absolute Error — average absolute index difference (lower is better)">
                <span>
                  MAE
                  <em className="th-blurb">Mean absolute error</em>
                </span>
                <strong className={indexMetrics ? undefined : 'placeholder'}>
                  {indexMetrics ? fmt(indexMetrics.mae) : '—'}
                </strong>
              </div>
              <div title="Pearson correlation between original and reconstructed index maps">
                <span>
                  Corr
                  <em className="th-blurb">Pearson correlation</em>
                </span>
                <strong className={indexMetrics ? undefined : 'placeholder'}>
                  {indexMetrics ? fmt(indexMetrics.correlation, 3) : '—'}
                </strong>
              </div>
              <div title="Structural Similarity Index — how alike structure/texture look (1 is identical)">
                <span>
                  SSIM
                  <em className="th-blurb">Structural similarity</em>
                </span>
                <strong className={indexMetrics ? undefined : 'placeholder'}>
                  {indexMetrics ? fmt(indexMetrics.ssim, 3) : '—'}
                </strong>
              </div>
              <div title="Mean signed difference (reconstructed − original); near zero is ideal">
                <span>
                  Bias
                  <em className="th-blurb">Mean signed difference</em>
                </span>
                <strong className={indexMetrics ? undefined : 'placeholder'}>
                  {indexMetrics ? fmt(indexMetrics.bias) : '—'}
                </strong>
              </div>
            </div>
          </section>

          <section className="panel">
            <h2>Method comparison</h2>
            <p className="hint">
              Runs all methods on the selected engine (Browser or Cloud Run).
            </p>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Method</th>
                    <th>Runtime (s)</th>
                    <th>Original</th>
                    <th>Compressed</th>
                    <th>Decompressed</th>
                    <th>Ratio</th>
                    <th>Mean RMSE</th>
                    <th>Mean PSNR</th>
                    <th>Mean SSIM</th>
                  </tr>
                </thead>
                <tbody>
                  {compareRows && compareRows.length > 0 ? (
                    compareRows.map((r) => (
                      <tr key={r.method}>
                        <td>{r.method}</td>
                        <td>{r.runtimeSeconds.toFixed(2)}</td>
                        <td>{bytesLabel(r.originalBytes)}</td>
                        <td>{bytesLabel(r.compressedBytesEstimate)}</td>
                        <td>{bytesLabel(r.decompressedBytes)}</td>
                        <td>{fmt(r.compressionRatio, 3)}</td>
                        <td>{fmt(r.meanRmse)}</td>
                        <td>{fmt(r.meanPsnr, 3)}</td>
                        <td>{fmt(r.meanSsim, 3)}</td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={9} className="hint placeholder">
                        Waiting for compare-all results
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>

          <section className="panel">
            <h2>Recent shared runs</h2>
            {supabaseOk && recentRuns.length > 0 ? (
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
            ) : (
              <p className="hint placeholder">
                {supabaseOk
                  ? 'No shared runs yet'
                  : 'Waiting for Supabase connection / shared runs'}
              </p>
            )}
          </section>
        </main>
      </div>
    </div>
  )
}
