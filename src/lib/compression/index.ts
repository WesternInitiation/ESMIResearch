import { runBandwidthCompression } from './bandwidth'
import { runJpeg2000Compression } from './jpeg2000'
import { runSvdCompression } from './svd'
import { runWaveletCompression } from './wavelet'
import { estimateByteSize, reportAllBands } from '../metrics'
import type { BandMap, CompressionMethod, CompressionResult } from '../types'

export type MethodParams = {
  svdRank: number
  waveletKeepFraction: number
  waveletLevels: number
  /** Wavelet family for Cloud Run / Python (browser always uses Haar). */
  waveletName: string
  bandwidthKeepFraction: number
  jpegRate: number
}

export async function runCompression(
  method: CompressionMethod,
  bands: BandMap,
  bandOrder: string[],
  width: number,
  height: number,
  originalBytes: number,
  params: MethodParams,
): Promise<CompressionResult> {
  const started = performance.now()
  let result: {
    bands: BandMap
    metadata: Record<string, unknown>
    compressedBytesEstimate: number
  }

  if (method === 'SVD') {
    result = runSvdCompression(bands, bandOrder, width, height, {
      rank: params.svdRank,
      normalize: true,
    })
  } else if (method === 'Wavelet transformation') {
    result = runWaveletCompression(bands, bandOrder, width, height, {
      keepFraction: params.waveletKeepFraction,
      levels: params.waveletLevels,
      wavelet: params.waveletName,
    })
  } else if (method === 'Bandwidth transformation') {
    result = runBandwidthCompression(
      bands,
      bandOrder,
      width,
      height,
      params.bandwidthKeepFraction,
    )
  } else {
    result = await runJpeg2000Compression(
      bands,
      bandOrder,
      width,
      height,
      params.jpegRate,
    )
  }

  const runtimeSeconds = (performance.now() - started) / 1000
  const channelReports = reportAllBands(bands, result.bands, bandOrder)
  const original = originalBytes || estimateByteSize(bands)
  const decompressedBytes = estimateByteSize(result.bands)
  const compressionRatio =
    original > 0 ? result.compressedBytesEstimate / original : 0

  return {
    method,
    bands: result.bands,
    bandOrder,
    width,
    height,
    runtimeSeconds,
    originalBytes: original,
    compressedBytesEstimate: result.compressedBytesEstimate,
    decompressedBytes,
    compressionRatio,
    channelReports,
    metadata: result.metadata,
  }
}
