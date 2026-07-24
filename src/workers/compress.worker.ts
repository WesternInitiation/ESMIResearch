/// <reference lib="webworker" />

import { runCompression, type MethodParams } from '../lib/compression'
import type { BandMap, CompressionMethod, CompressionResult } from '../lib/types'

export type WorkerRequest = {
  id: string
  method: CompressionMethod
  bandOrder: string[]
  width: number
  height: number
  originalBytes: number
  params: MethodParams
  /** Transferable band payloads keyed by band name. */
  bandBuffers: Record<string, ArrayBuffer>
}

export type WorkerResponse =
  | { id: string; type: 'progress'; message: string }
  | {
      id: string
      type: 'result'
      result: Omit<CompressionResult, 'bands'> & {
        bandBuffers: Record<string, ArrayBuffer>
      }
    }
  | { id: string; type: 'error'; message: string }

declare const self: DedicatedWorkerGlobalScope

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data
  void (async () => {
    try {
      self.postMessage({
        id: msg.id,
        type: 'progress',
        message: `Running ${msg.method}…`,
      } satisfies WorkerResponse)

      const bands: BandMap = {}
      for (const name of msg.bandOrder) {
        bands[name] = new Float64Array(msg.bandBuffers[name])
      }

      const result = await runCompression(
        msg.method,
        bands,
        msg.bandOrder,
        msg.width,
        msg.height,
        msg.originalBytes,
        msg.params,
      )

      const bandBuffers: Record<string, ArrayBuffer> = {}
      const transfer: Transferable[] = []
      for (const name of result.bandOrder) {
        const src = result.bands[name]
        const copy = new Float64Array(src.length)
        copy.set(src)
        const buf = copy.buffer as ArrayBuffer
        bandBuffers[name] = buf
        transfer.push(buf)
      }

      const { bands: _bands, ...rest } = result
      self.postMessage(
        {
          id: msg.id,
          type: 'result',
          result: { ...rest, bandBuffers },
        } satisfies WorkerResponse,
        transfer,
      )
    } catch (err) {
      self.postMessage({
        id: msg.id,
        type: 'error',
        message: err instanceof Error ? err.message : 'Compression failed',
      } satisfies WorkerResponse)
    }
  })()
}

export {}
