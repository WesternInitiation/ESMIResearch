import type { MethodParams } from './compression'
import type { BandMap, CompressionMethod, CompressionResult } from './types'
import type { WorkerRequest, WorkerResponse } from '../workers/compress.worker'

type RunArgs = {
  method: CompressionMethod
  bands: BandMap
  bandOrder: string[]
  width: number
  height: number
  originalBytes: number
  params: MethodParams
  onProgress?: (message: string) => void
}

let worker: Worker | null = null
let workerFailed = false

function getWorker(): Worker | null {
  if (workerFailed) return null
  if (typeof window === 'undefined') return null
  if (worker) return worker
  try {
    worker = new Worker(new URL('../workers/compress.worker.ts', import.meta.url), {
      type: 'module',
    })
    return worker
  } catch {
    workerFailed = true
    return null
  }
}

function runOnMainThread(args: RunArgs): Promise<CompressionResult> {
  return import('./compression').then(({ runCompression }) =>
    runCompression(
      args.method,
      args.bands,
      args.bandOrder,
      args.width,
      args.height,
      args.originalBytes,
      args.params,
    ),
  )
}

export async function runCompressionAsync(args: RunArgs): Promise<CompressionResult> {
  const w = getWorker()
  if (!w) return runOnMainThread(args)

  const id = crypto.randomUUID()
  const bandBuffers: Record<string, ArrayBuffer> = {}
  const transfer: Transferable[] = []
  // Copy so the UI keeps its working bands if the worker consumes transfers.
  for (const name of args.bandOrder) {
    const copy = new Float64Array(args.bands[name])
    const buf = copy.buffer
    bandBuffers[name] = buf
    transfer.push(buf)
  }

  const request: WorkerRequest = {
    id,
    method: args.method,
    bandOrder: args.bandOrder,
    width: args.width,
    height: args.height,
    originalBytes: args.originalBytes,
    params: args.params,
    bandBuffers,
  }

  return new Promise((resolve, reject) => {
    const onMessage = (event: MessageEvent<WorkerResponse>) => {
      const data = event.data
      if (!data || data.id !== id) return
      if (data.type === 'progress') {
        args.onProgress?.(data.message)
        return
      }
      w.removeEventListener('message', onMessage)
      w.removeEventListener('error', onError)
      if (data.type === 'error') {
        reject(new Error(data.message))
        return
      }
      const bands: BandMap = {}
      for (const name of data.result.bandOrder) {
        bands[name] = new Float64Array(data.result.bandBuffers[name])
      }
      const { bandBuffers: _bb, ...rest } = data.result
      resolve({ ...rest, bands })
    }
    const onError = (err: ErrorEvent) => {
      w.removeEventListener('message', onMessage)
      w.removeEventListener('error', onError)
      workerFailed = true
      worker?.terminate()
      worker = null
      void runOnMainThread(args).then(resolve, reject)
      console.warn('Compression worker failed; falling back to main thread.', err.message)
    }
    w.addEventListener('message', onMessage)
    w.addEventListener('error', onError)
    w.postMessage(request, transfer)
  })
}
