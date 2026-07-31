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
  /**
   * Cloud Run rasterio exports: one GeoTIFF per band with CRS/transform/NoData.
   * Prefer these over browser classic-TIFF / RGB-preview downloads.
   */
  reconstructedBandGeotiffs?: Array<{
    band: string
    label: string
    filename: string
    gcsUri: string
    size: number
    dtype: string
    width: number
    height: number
    crs?: string | null
    transform?: number[] | null
    nodata?: number | null
  }>
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
