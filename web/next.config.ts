import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  // geotiff / compression run entirely in the browser; keep server bundle lean.
  serverExternalPackages: [],
}

export default nextConfig
