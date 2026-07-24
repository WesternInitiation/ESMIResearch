import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  // geotiff / compression run in the browser (and a Web Worker).
  serverExternalPackages: [],
}

export default nextConfig
