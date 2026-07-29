export type BandMap = Record<string, Float64Array>
export type ImageSize = { width: number; height: number }

export type ChannelReport = {
  band: string
  rmse: number
  mae: number
  psnrDb: number
  ssim: number
}

export type CompressionResult = {
  method: string
  bands: BandMap
  bandOrder: string[]
  width: number
  height: number
  runtimeSeconds: number
  originalBytes: number
  compressedBytesEstimate: number
  /** Reconstructed float raster size (or estimate when bands are unavailable). */
  decompressedBytes?: number
  compressionRatio: number
  channelReports: ChannelReport[]
  metadata: Record<string, unknown>
}

export type CompressionMethod =
  | 'SVD'
  | 'Wavelet transformation'
  | 'Bandwidth transformation'
  | 'JPEG2000'
  | 'LZW'

export const COMPRESSION_METHODS: CompressionMethod[] = [
  'SVD',
  'Wavelet transformation',
  'Bandwidth transformation',
  'JPEG2000',
  'LZW',
]
